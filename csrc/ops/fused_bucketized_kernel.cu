#include <ATen/cuda/CUDAContext.h>

#include "cuda/cuda_param.cuh"
#include "cuda/element_wise_kernel.cuh"
#include "cuda/utils.cuh"
#include "cuda_runtime.h"
#include "ops/fused_bucketized.h"

namespace recis {
namespace functional {

template <typename T>
struct BucketizeData {
  T* boundaries;
  int len;
  BucketizeData() : boundaries(nullptr), len(0) {}
  BucketizeData(T* boundaries, int len) : boundaries(boundaries), len(len) {}
};

template <typename T>
struct BucketizeFactory {
  __device__ int operator()(const T value, const BucketizeData<T>& data) {
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

template <typename T>
void fused_bucketized_cuda_impl(std::vector<torch::Tensor>& inputs,
                                std::vector<torch::Tensor>& outputs,
                                std::vector<torch::Tensor>& boundaries) {
  using namespace recis::cuda;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t N = inputs.size();
  std::vector<int64_t> sizes(N);
  CudaVecParam<T*> inputs_ptrs(N, stream);
  CudaVecParam<int64_t*> outputs_ptrs(N, stream);
  CudaVecParam<BucketizeData<T>> bucketize_datas(N, stream);
  for (int64_t i = 0; i < N; ++i) {
    sizes[i] = inputs[i].numel();
    inputs_ptrs[i] = inputs[i].data_ptr<T>();
    outputs_ptrs[i] = outputs[i].data_ptr<int64_t>();
    bucketize_datas[i] =
        BucketizeData<T>(boundaries[i].data_ptr<T>(), boundaries[i].numel());
  }

  fused_element_wise_launcher<T, BucketizeData<T>, int64_t,
                              BucketizeFactory<T>>(
      const_cast<const T**>(inputs_ptrs.data()), bucketize_datas.data(),
      outputs_ptrs.data(), sizes.data(), N, BucketizeFactory<T>(), false,
      stream);
}

void fused_bucketized_cuda(std::vector<torch::Tensor>& inputs,
                           std::vector<torch::Tensor>& outputs,
                           std::vector<torch::Tensor>& boundaries) {
  // Check input dtype and dispatch to appropriate implementation
  auto dtype = inputs[0].scalar_type();
  if (dtype == torch::kFloat64) {
    fused_bucketized_cuda_impl<double>(inputs, outputs, boundaries);
  } else {
    // Default to float32 for backward compatibility
    fused_bucketized_cuda_impl<float>(inputs, outputs, boundaries);
  }
}
}  // namespace functional
}  // namespace recis
