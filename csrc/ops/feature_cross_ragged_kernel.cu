#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <torch/extension.h>

#ifdef USE_ROCM
#include <cuda/std/limits>
#include <hipcub/device/device_segmented_radix_sort.hpp>
#include <hipcub/device/device_segmented_sort.hpp>
#include <hipcub/hipcub.hpp>
#else
#include <cub/cub.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_segmented_sort.cuh>
#endif

#include "cuda/cuda_param.cuh"
#include "cuda/utils.cuh"
#include "feature_cross_ragged.h"
#include "ragged_common.cuh"

namespace recis {
namespace functional {

#ifdef USE_ROCM
namespace cub = hipcub;
#endif

template <typename key_t, typename value_t, typename index_t>
void device_segment_sort(const key_t* key_input,    // [total_elems]
                         index_t* offsets,          // [num_rows+1]=. constant
                         const value_t* val_input,  // [total_elems]
                         key_t* key_output,         // output
                         value_t* val_output,       // output
                         int num_items, int num_rows, cudaStream_t stream) {
  void* d_temp_storage = nullptr;
  size_t temp_bytes = 0;
  int begin_bit = 0;
  int end_bit = sizeof(key_t) * 8;
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_bytes, key_input, key_output, val_input, val_output,
      num_items, num_rows, offsets, offsets + 1, begin_bit, end_bit, stream);

  d_temp_storage = cuda::cuda_malloc<void>(temp_bytes);

  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_bytes, key_input, key_output, val_input, val_output,
      num_items, num_rows, offsets, offsets + 1, begin_bit, end_bit, stream);

  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  cuda::delete_cuda_ptr(d_temp_storage);
}

template <typename key_t, typename value_t, typename index_t>
__global__ void block_segment_unique_by_key(
    key_t* keys,             // [total_elems], in-place
    const index_t* offsets,  // [num_rows+1]. constant.
    value_t* values,         // [total_elems], in-place
    thrust::pair<key_t*, key_t*>* key_segs,
    thrust::pair<value_t*, value_t*>* value_segs, int num_seg) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockIdx.x;
  int tid = threadIdx.x;

  if (row >= num_seg) {
    return;
  }

  extern __shared__ char shmem[];
  index_t* shm_offsets = reinterpret_cast<index_t*>(shmem);
  for (int i = tid; i < 2; i += blockDim.x) {
    shm_offsets[i] = offsets[row + i];
  }
  __syncthreads();
  index_t beg_idx = shm_offsets[0];
  index_t end_idx = shm_offsets[1];
  index_t row_len = end_idx - beg_idx;

  key_t* shm_keys = reinterpret_cast<key_t*>(shmem + 2 * sizeof(index_t));
  value_t* shm_vals = reinterpret_cast<value_t*>(shmem + 2 * sizeof(index_t) +
                                                 blockDim.x * sizeof(key_t));
  if (tid < row_len) {
    shm_keys[tid] = keys[beg_idx + tid];
    shm_vals[tid] = values[beg_idx + tid];
  } else {
    shm_keys[tid] = ::cuda::std::numeric_limits<key_t>::max();
    shm_vals[tid] = ::cuda::std::numeric_limits<value_t>::max();
  }
  __syncthreads();

  key_t tid_key = ::cuda::std::numeric_limits<key_t>::max();
  key_t prev_tid_key = ::cuda::std::numeric_limits<key_t>::max();
  value_t tid_val = ::cuda::std::numeric_limits<value_t>::max();

  if (tid < row_len) {
    tid_key = shm_keys[tid];
    prev_tid_key =
        tid == 0 ? shm_keys[tid] : shm_keys[tid - 1];  // boundary for idx 0
    tid_val = shm_vals[tid];
  }

  __shared__ int valid_index;
  if (tid == 0) {
    if (row_len > 0) {
      valid_index = 1;
    } else {
      valid_index = 0;
    }
  }

  __syncthreads();

  if (tid < row_len) {
    if (tid_key != prev_tid_key) {
      int idx = atomicAdd(&valid_index, 1);
      shm_keys[idx] = tid_key;
      shm_vals[idx] = tid_val;
    }
  }

  __syncthreads();

  if (tid < valid_index) {
    keys[beg_idx + tid] = shm_keys[tid];
    values[beg_idx + tid] = shm_vals[tid];
  }
  if (tid == 0) {
    key_segs[row] =
        thrust::make_pair(keys + beg_idx, keys + beg_idx + valid_index);
    value_segs[row] =
        thrust::make_pair(values + beg_idx, values + beg_idx + valid_index);
  }
}

template <typename T>
struct FeatureComputeSegLength {
  __host__ __device__ auto operator()(const T& seg) const
      -> decltype(seg.second - seg.first) {
    return seg.second - seg.first;
  }
};

// values, offsets, weights: constant
// return std::make_tuple(val_work, wt_work, val_segs, wt_segs, valid_len);
std::tuple<torch::Tensor, torch::Tensor,
           thrust::device_vector<thrust::pair<void*, void*>>,
           thrust::device_vector<thrust::pair<void*, void*>>,
           thrust::device_vector<int>>
segment_sort_and_unique_by_key(torch::Tensor& value, torch::Tensor& offsets,
                               torch::Tensor& weight, int num_seg,
                               cudaStream_t stream) {
  int max_row_len =
      (offsets.narrow(0, 1, num_seg) - offsets.narrow(0, 0, num_seg))
          .max()
          .item<int>();

  torch::Tensor val_work = torch::empty_like(value);
  torch::Tensor wt_work = torch::empty_like(weight);
  thrust::device_vector<int> valid_len(num_seg);
  thrust::device_vector<thrust::pair<int64_t*, int64_t*>> val_segs(num_seg);
  thrust::device_vector<thrust::pair<void*, void*>> wt_segs(num_seg);

  int num_items = value.numel();

  AT_DISPATCH_INDEX_TYPES(
      offsets.scalar_type(), "segment_sort_unique_cuda_0", ([&] {
        AT_DISPATCH_ALL_TYPES(
            weight.scalar_type(), "segment_sort_unique_cuda_1", ([&] {
              device_segment_sort<int64_t, scalar_t, index_t>(
                  value.data_ptr<int64_t>(), offsets.data_ptr<index_t>(),
                  weight.data_ptr<scalar_t>(), val_work.data_ptr<int64_t>(),
                  wt_work.data_ptr<scalar_t>(), num_items, num_seg, stream);

              const int shm_size = 2 * sizeof(index_t) +
                                   max_row_len * sizeof(scalar_t) +
                                   max_row_len * sizeof(int64_t);

              block_segment_unique_by_key<int64_t, scalar_t, index_t>
                  <<<num_seg, max_row_len, shm_size, stream>>>(
                      val_work.data_ptr<int64_t>(), offsets.data_ptr<index_t>(),
                      wt_work.data_ptr<scalar_t>(),
                      thrust::raw_pointer_cast(val_segs.data()),
                      reinterpret_cast<thrust::pair<scalar_t*, scalar_t*>*>(
                          thrust::raw_pointer_cast(wt_segs.data())),
                      num_seg);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
      }));

  thrust::transform(
      thrust::cuda::par_nosync.on(stream), val_segs.begin(), val_segs.end(),
      valid_len.begin(),
      FeatureComputeSegLength<thrust::pair<int64_t*, int64_t*>>());

  return std::make_tuple(val_work, wt_work, val_segs, wt_segs, valid_len);
}

template <typename scalar_t, typename index_t, bool use_shared_mem>
__global__ void feature_cross_ragged_kernel(
    const int64_t* __restrict__ x_value, const index_t* __restrict__ x_offsets,
    const scalar_t* __restrict__ x_weight, const int64_t* __restrict__ y_value,
    const index_t* __restrict__ y_offsets, const scalar_t* __restrict__ _weight,
    int64_t* __restrict__ val_output, index_t* __restrict__ offsets,
    scalar_t* __restrict__ wt_output,
    const thrust::pair<int64_t*, int64_t*>* x_val_segs,
    const thrust::pair<int64_t*, int64_t*>* y_val_segs,
    const thrust::pair<scalar_t*, scalar_t*>* x_wt_segs,
    const thrust::pair<scalar_t*, scalar_t*>* y_wt_segs, const int num_seg) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tn = blockDim.x * gridDim.x;
  int tid = threadIdx.x;

  extern __shared__ unsigned char shm[];
  if constexpr (use_shared_mem) {
    thrust::pair<int64_t*, int64_t*>* sm_val_segs =
        reinterpret_cast<thrust::pair<int64_t*, int64_t*>*>(shm);
    thrust::pair<scalar_t*, scalar_t*>* sm_wt_segs =
        reinterpret_cast<thrust::pair<scalar_t*, scalar_t*>*>(
            shm + 2 * num_seg * sizeof(thrust::pair<void*, void*>));
    index_t* sm_offsets = reinterpret_cast<index_t*>(
        shm + num_seg * 4 * sizeof(thrust::pair<void*, void*>));
    for (int i = tid; i < num_seg; i += blockDim.x) {
      sm_val_segs[i] = x_val_segs[i];
    }
    for (int i = tid; i < num_seg; i += blockDim.x) {
      sm_val_segs[i + num_seg] = y_val_segs[i];
    }
    for (int i = tid; i < num_seg; i += blockDim.x) {
      sm_wt_segs[i] = x_wt_segs[i];
    }
    for (int i = tid; i < num_seg; i += blockDim.x) {
      sm_wt_segs[i + num_seg] = y_wt_segs[i];
    }
    for (int i = tid; i < num_seg + 1; i += blockDim.x) {
      sm_offsets[i] = offsets[i];
    }
    __syncthreads();
    x_val_segs = sm_val_segs;
    y_val_segs = sm_val_segs + num_seg;
    x_wt_segs = sm_wt_segs;
    y_wt_segs = sm_wt_segs + num_seg;
    offsets = sm_offsets;
  }

  index_t M = offsets[num_seg];
  int64_t vals[2];
  for (index_t i = gid; i < M; i += tn) {
    index_t seg_id =
        binary_search(i, offsets, static_cast<index_t>(num_seg + 1));
    index_t rel_ele_id = i - offsets[seg_id];
    int64_t* x_val_seg_end = x_val_segs[seg_id].second;
    int64_t* x_val_seg_beg = x_val_segs[seg_id].first;
    int64_t* y_val_seg_end = y_val_segs[seg_id].second;
    int64_t* y_val_seg_beg = y_val_segs[seg_id].first;
    int y_seg_len = y_val_seg_end - y_val_seg_beg;
    int x_id = rel_ele_id / y_seg_len;
    int y_id = rel_ele_id % y_seg_len;
    vals[0] = x_val_seg_beg[x_id];
    vals[1] = y_val_seg_beg[y_id];
    val_output[i] = murmur_hash_64(reinterpret_cast<uint64_t*>(vals), 0, 2);
    scalar_t* x_wt_seg_beg = x_wt_segs[seg_id].first;
    scalar_t* y_wt_seg_beg = y_wt_segs[seg_id].first;
    wt_output[i] = x_wt_seg_beg[x_id] * y_wt_seg_beg[y_id];
  }
}

template <typename scalar_t, typename index_t>
__global__ void feature_cross_ragged_kernel_single_pass(
    const int64_t* __restrict__ x_val, const scalar_t* __restrict__ x_wt,
    const int64_t* __restrict__ y_val, const scalar_t* __restrict__ y_wt,
    int64_t* __restrict__ val_output, const index_t* offsets,
    scalar_t* __restrict__ wt_output, const int num_seg) {
  index_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  index_t tn = blockDim.x * gridDim.x;
  int tid = threadIdx.x;

  index_t M = offsets[num_seg];
  int64_t vals[2];

  for (index_t i = gid; i < M; i += tn) {
    index_t seg_id = i;
    vals[0] = x_val[seg_id];
    vals[1] = y_val[seg_id];
    val_output[i] = murmur_hash_64(reinterpret_cast<uint64_t*>(vals), 0, 2);
    wt_output[i] = x_wt[seg_id] * y_wt[seg_id];
  }
}

static __host__ bool is_diff_one(const torch::Tensor& offsets, int num_seg) {
  auto diff = offsets.narrow(0, 1, num_seg) - offsets.narrow(0, 0, num_seg);
  return diff.eq(1).all().item<bool>();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
feature_cross_ragged_cuda(torch::Tensor x_value, torch::Tensor x_offsets,
                          torch::Tensor x_weight, torch::Tensor y_value,
                          torch::Tensor y_offsets, torch::Tensor y_weight) {
  TORCH_CHECK(all_same_type({x_value, y_value}, torch::kInt64),
              "ids for x and y must be torch::kInt64")
  TORCH_CHECK(all_same_type({x_offsets, y_offsets}, torch::kInt32) ||
                  all_same_type({x_offsets, y_offsets}, torch::kInt64),
              "offsets for x and y must be torch::kInt32")
  TORCH_CHECK(
      all_cuda({x_value, y_value, x_offsets, y_offsets, x_weight, y_weight}),
      "ids, offsets, weights for x and y must be on gpu")

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto policy = thrust::cuda::par_nosync.on(stream);

  int num_seg = min(x_offsets.numel() - 1, y_offsets.numel() - 1);
  const auto threads = dim3(MAX_THREADS_PER_BLOCK);

  if (is_diff_one(x_offsets, num_seg) && is_diff_one(y_offsets, num_seg)) {
    at::Tensor offsets_output = at::arange(num_seg + 1, x_offsets.options());
    at::Tensor value_output = at::empty(num_seg, x_value.options());
    at::Tensor weight_output = at::empty(num_seg, x_weight.options());

    const auto blocks =
        dim3((num_seg + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);

    AT_DISPATCH_INDEX_TYPES(
        x_offsets.scalar_type(), "feature_cross_ragged_single_pass_cuda_op_0",
        [&] {
          AT_DISPATCH_ALL_TYPES(
              x_weight.scalar_type(),
              "feature_cross_ragged_single_pass_cuda_op_1", [&] {
                feature_cross_ragged_kernel_single_pass<<<blocks, threads, 0,
                                                          stream>>>(
                    x_value.data_ptr<int64_t>(), x_weight.data_ptr<scalar_t>(),
                    y_value.data_ptr<int64_t>(), y_weight.data_ptr<scalar_t>(),
                    value_output.data_ptr<int64_t>(),
                    offsets_output.data_ptr<index_t>(),
                    weight_output.data_ptr<scalar_t>(), num_seg);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });

    return std::make_tuple(value_output, offsets_output, weight_output);

  } else {
    at::Tensor offsets_output = at::zeros(num_seg + 1, x_offsets.options());

    torch::Tensor x_val_work, x_wt_work;
    thrust::device_vector<thrust::pair<void*, void*>> x_val_segs, x_wt_segs;
    thrust::device_vector<int> x_valid_len;
    std::tie(x_val_work, x_wt_work, x_val_segs, x_wt_segs, x_valid_len) =
        segment_sort_and_unique_by_key(x_value, x_offsets, x_weight, num_seg,
                                       stream);

    torch::Tensor y_val_work, y_wt_work;
    thrust::device_vector<thrust::pair<void*, void*>> y_val_segs, y_wt_segs;
    thrust::device_vector<int> y_valid_len;
    std::tie(y_val_work, y_wt_work, y_val_segs, y_wt_segs, y_valid_len) =
        segment_sort_and_unique_by_key(y_value, y_offsets, y_weight, num_seg,
                                       stream);

    thrust::device_vector<int> output_seg_len(num_seg);
    thrust::transform(policy, x_valid_len.data(), x_valid_len.data() + num_seg,
                      y_valid_len.data(), output_seg_len.data(),
                      thrust::multiplies<int>());

    int64_t val_shape = 0;
    AT_DISPATCH_INDEX_TYPES(
        x_offsets.scalar_type(), "feature_cross_ragged_cuda_pre_op", [&] {
          thrust::inclusive_scan(policy, output_seg_len.data(),
                                 output_seg_len.data() + num_seg,
                                 offsets_output.data_ptr<index_t>() + 1);
          cudaMemcpy(&val_shape, offsets_output.data_ptr<index_t>() + num_seg,
                     sizeof(index_t), cudaMemcpyDeviceToHost);
        });

    at::Tensor value_output = at::empty(
        val_shape,
        torch::TensorOptions().dtype(x_value.dtype()).device(torch::kCUDA));
    at::Tensor weight_output = at::empty(
        val_shape,
        torch::TensorOptions().dtype(x_weight.dtype()).device(torch::kCUDA));

    AT_DISPATCH_INDEX_TYPES(
        x_offsets.scalar_type(), "feature_cross_ragged_cuda_op_0", [&] {
          AT_DISPATCH_ALL_TYPES(
              x_weight.scalar_type(), "feature_cross_ragged_cuda_op_1", [&] {
                index_t shared_mem_size =
                    sizeof(index_t) * (num_seg + 1) +
                    sizeof(std::pair<void*, void*>) * num_seg * 4;
                const auto blocks =
                    dim3((val_shape + MAX_THREADS_PER_BLOCK - 1) /
                         MAX_THREADS_PER_BLOCK);

                LAUNCH_KERNEL_SHMEM_DISPATCH(
                    feature_cross_ragged_kernel, (scalar_t, index_t), blocks,
                    threads, shared_mem_size, stream,
                    x_val_work.data_ptr<int64_t>(),
                    x_offsets.data_ptr<index_t>(),
                    x_wt_work.data_ptr<scalar_t>(),
                    y_val_work.data_ptr<int64_t>(),
                    y_offsets.data_ptr<index_t>(),
                    y_wt_work.data_ptr<scalar_t>(),
                    value_output.data_ptr<int64_t>(),
                    offsets_output.data_ptr<index_t>(),
                    weight_output.data_ptr<scalar_t>(),
                    reinterpret_cast<const thrust::pair<int64_t*, int64_t*>*>(
                        thrust::raw_pointer_cast(x_val_segs.data())),
                    reinterpret_cast<const thrust::pair<int64_t*, int64_t*>*>(
                        thrust::raw_pointer_cast(y_val_segs.data())),
                    reinterpret_cast<const thrust::pair<scalar_t*, scalar_t*>*>(
                        thrust::raw_pointer_cast(x_wt_segs.data())),
                    reinterpret_cast<const thrust::pair<scalar_t*, scalar_t*>*>(
                        thrust::raw_pointer_cast(y_wt_segs.data())),
                    num_seg);
              });
        });

    return std::make_tuple(value_output, offsets_output, weight_output);
  }
}
}  // namespace functional
}  // namespace recis
