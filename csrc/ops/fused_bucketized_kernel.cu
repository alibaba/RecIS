#include <ATen/cuda/CUDAContext.h>

#include "cuda/cuda_param.cuh"
#include "cuda/element_wise_kernel.cuh"
#include "cuda/utils.cuh"
#include "cuda_runtime.h"
#include "ops/fused_bucketized.h"

namespace recis {
namespace functional {
struct BucketizeData {
  float* boundaries;
  int len;
  BucketizeData() : boundaries(nullptr), len(0) {}
  BucketizeData(float* boundaries, int len)
      : boundaries(boundaries), len(len) {}
};

#ifndef __HIP_PLATFORM_AMD__
struct BucketizeFactory {
  __device__ int operator()(const float value, const BucketizeData& data) {
    int bucket = 0;
    int count = data.len;
    auto boundaries = data.boundaries;
    while (count > 0) {
      int left = bucket;
      int step = count / 2;
      left += step;
      if (!(value < boundaries[left])) {
        bucket = ++left;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    return bucket;
  }
};
#else
#define KBLOCK_SIZE 256
struct BucketizeFactory {
  __device__ __forceinline__ int operator()(const float value, 
                                            const float* boundaries,
                                            int count) {
    int c = count;
    if (c <= 0) return 0;
    int bucket = 0;
    // ceil(log2(c)) for positive c
    int iters = 32 - __builtin_clz(c);
    #pragma unroll
    for (int i = 0; i < iters; ++i) {
      int step = c >> 1;
      int left = bucket + step;
      int cond = (value >= boundaries[left]);
      bucket = cond ? (left + 1) : bucket;
      c = cond ? (c - step - 1) : step;
    }
    return bucket;
  }
};

template <typename A, typename B, typename C>
__global__ __launch_bounds__(KBLOCK_SIZE, 2)
void fused_bucketize_kernel(const A** a, const B** b, C** c,
                            int64_t N, int64_t* sizes, int64_t* b_sizes,
                            BucketizeFactory factory) {
  int vec_id = blockIdx.y;
  if (vec_id >= N) return;
  // use 32-bit indices locally to reduce register pressure
  int size_local = static_cast<int>(sizes[vec_id]);
  int b_size_local = static_cast<int>(b_sizes[vec_id]);
  int threads_num = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ unsigned char shmem[];
  B* sm_boundaries = reinterpret_cast<B*>(shmem);

  // move data to shared buffer
  const B* __restrict__ b_vec = b[vec_id];
  if (threadIdx.x < b_size_local) {
    sm_boundaries[threadIdx.x] = b_vec[threadIdx.x];
  }
  __syncthreads();

  const A* __restrict__ a_vec = a[vec_id];
  C* __restrict__ c_vec = c[vec_id];

  // Process multiple elements per thread to improve ILP and reduce loop overhead
  const int UNROLL = 8;
  for (int index = tid; index < size_local; index += threads_num * UNROLL) {
    int i0 = index;
    // software prefetch of a few elements in the unroll group
    A v0 = (i0 < size_local) ? a_vec[i0] : A(0);
    A v1 = ((i0 + threads_num) < size_local) ? a_vec[i0 + threads_num] : A(0);
    A v2 = ((i0 + 2 * threads_num) < size_local) ? a_vec[i0 + 2 * threads_num] : A(0);
    A v3 = ((i0 + 3 * threads_num) < size_local) ? a_vec[i0 + 3 * threads_num] : A(0);
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int ii = i0 + u * threads_num;
      if (ii < size_local) {
        A val = (u == 0) ? v0 : (u == 1) ? v1 : (u == 2) ? v2 : (u == 3) ? v3 : a_vec[ii];
        c_vec[ii] = factory(val, sm_boundaries, b_size_local);
      }
    }
  }
}

template <typename A, typename B, typename C>
void fused_bucketize_launcher(const A** a, const B** b, C** c, int64_t* sizes,
                                 int64_t* b_sizes, int64_t N,
                                 BucketizeFactory factor,
                                 bool with_pack,
                                 hipStream_t stream) {
  using namespace recis::cuda;
  int64_t sm_count = get_sm_count();
  int64_t max_size = 0;
  for (int64_t i = 0; i < N; ++i) {
    max_size = std::max(max_size, sizes[i]);
  }
  if (max_size == 0) return;
  
 // copy sizes to device
  int64_t* d_sizes = cuda_malloc_and_copy<int64_t>(sizes, N, stream);
  int64_t* d_b_sizes = cuda_malloc_and_copy<int64_t>(b_sizes, N, stream);

  // shared memory size based on maximum boundary size across tensors
  int64_t max_b_size = 0;
  for (int64_t i = 0; i < N; ++i) {
    max_b_size = std::max(max_b_size, b_sizes[i]);
  }
  size_t shared_memory_size = static_cast<size_t>(max_b_size + 1) * sizeof(B);

  int maxActiveBlocksPerSM = 0;
  C10_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocksPerSM,
      fused_bucketize_kernel<A, B, C>,
      KBLOCK_SIZE,
      shared_memory_size));
  if (maxActiveBlocksPerSM <= 0) {
    maxActiveBlocksPerSM = 8; // fallback consistent with original heuristic
  }

  int64_t block_num = std::min<int64_t>(sm_count * maxActiveBlocksPerSM,
                                        (max_size + KBLOCK_SIZE - 1) / KBLOCK_SIZE);
  if (block_num <= 0) block_num = sm_count * maxActiveBlocksPerSM;
  dim3 grid(block_num, N);
  dim3 block(KBLOCK_SIZE);

  fused_bucketize_kernel<A, B, C>
        <<<grid, block, shared_memory_size, stream>>>(a, b, c, N, d_sizes, d_b_sizes,
                                                    factor);
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  delete_cuda_ptr(d_sizes);
  delete_cuda_ptr(d_b_sizes);
}
#endif

void fused_bucketized_cuda(std::vector<torch::Tensor>& inputs,
                           std::vector<torch::Tensor>& outputs,
                           std::vector<torch::Tensor>& boundaries) {
  using namespace recis::cuda;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t N = inputs.size();
  std::vector<int64_t> sizes(N);
  CudaVecParam<float*> inputs_ptrs(N, stream);
  CudaVecParam<int64_t*> outputs_ptrs(N, stream);
  
#ifdef __HIP_PLATFORM_AMD__
  CudaVecParam<float*> boundaries_ptrs(N, stream);
  std::vector<int64_t> b_sizes(N);
  for (int64_t i = 0; i < N; ++i) {
    sizes[i] = inputs[i].numel();
    inputs_ptrs[i] = inputs[i].data_ptr<float>();
    outputs_ptrs[i] = outputs[i].data_ptr<int64_t>();
    boundaries_ptrs[i] = boundaries[i].data_ptr<float>();
    b_sizes[i] = static_cast<int>(boundaries[i].numel());
  }

  fused_bucketize_launcher<float, float, int64_t>(
      const_cast<const float**>(inputs_ptrs.data()),
      const_cast<const float**>(boundaries_ptrs.data()),
      outputs_ptrs.data(),
      sizes.data(),
      b_sizes.data(),
      N,
      BucketizeFactory(),
      false,
      stream);
#else
  CudaVecParam<BucketizeData> bucketize_datas(N, stream);
  for (int64_t i = 0; i < N; ++i) {
    sizes[i] = inputs[i].numel();
    inputs_ptrs[i] = inputs[i].data_ptr<float>();
    outputs_ptrs[i] = outputs[i].data_ptr<int64_t>();
    bucketize_datas[i] =
        BucketizeData(boundaries[i].data_ptr<float>(), boundaries[i].numel());
  }

  fused_element_wise_launcher<float, BucketizeData, int64_t, BucketizeFactory>(
      const_cast<const float**>(inputs_ptrs.data()), bucketize_datas.data(),
      outputs_ptrs.data(), sizes.data(), N, BucketizeFactory(), false, stream);
#endif
}
}  // namespace functional
}  // namespace recis
