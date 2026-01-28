#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "cuda/cuda_param.cuh"
#include "cuda/utils.cuh"
#include "ops/int64_to_string_int8.h"

namespace recis {
namespace functional {

__device__ int int64_to_string_device(int64_t value, char* buffer) {
  if (value == 0) {
    buffer[0] = '0';
    return 1;
  }

  bool is_negative = value < 0;
  uint64_t abs_value = is_negative ? -static_cast<uint64_t>(value) : value;

  int length = 0;
  uint64_t temp = abs_value;
  while (temp > 0) {
    length++;
    temp /= 10;
  }

  if (is_negative) {
    length++;
  }

  int pos = length - 1;
  while (abs_value > 0) {
    buffer[pos--] = '0' + (abs_value % 10);
    abs_value /= 10;
  }

  if (is_negative) {
    buffer[0] = '-';
  }

  return length;
}

__global__ void fused_calculate_offsets_kernel(const int64_t** inputs,
                                               int64_t** offsets,
                                               int64_t* sizes,
                                               int64_t num_tensors) {
  int64_t tensor_id = blockIdx.y;
  if (tensor_id >= num_tensors) return;

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t num_elements = sizes[tensor_id];

  if (idx < num_elements) {
    char buffer[32];
    offsets[tensor_id][idx + 1] =
        int64_to_string_device(inputs[tensor_id][idx], buffer);
  }
}

__global__ void fused_int64_to_string_kernel(const int64_t** inputs,
                                             int8_t** outputs,
                                             const int64_t** offsets,
                                             int64_t* sizes,
                                             int64_t num_tensors) {
  int64_t tensor_id = blockIdx.y;
  if (tensor_id >= num_tensors) return;

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t num_elements = sizes[tensor_id];

  if (idx < num_elements) {
    char buffer[32];
    int length = int64_to_string_device(inputs[tensor_id][idx], buffer);

    int64_t output_offset = offsets[tensor_id][idx];
    for (int i = 0; i < length; i++) {
      outputs[tensor_id][output_offset + i] = static_cast<int8_t>(buffer[i]);
    }
  }
}

void fused_int64_to_string_int8_cuda(std::vector<torch::Tensor>& inputs,
                                     std::vector<torch::Tensor>& outputs,
                                     std::vector<torch::Tensor>& offsets) {
  using namespace recis::cuda;
  int64_t num_tensors = inputs.size();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  std::vector<int64_t> sizes(num_tensors);
  CudaVecParam<int64_t*> inputs_ptrs(num_tensors, stream);
  CudaVecParam<int8_t*> outputs_ptrs(num_tensors, stream);
  CudaVecParam<int64_t*> offsets_ptrs(num_tensors, stream);

  int64_t max_size = 0;
  for (int64_t i = 0; i < num_tensors; ++i) {
    sizes[i] = inputs[i].numel();
    max_size = std::max(max_size, sizes[i]);
    inputs_ptrs[i] = inputs[i].data_ptr<int64_t>();
    offsets_ptrs[i] = offsets[i].data_ptr<int64_t>();
  }

  int64_t* d_sizes =
      cuda_malloc_and_copy<int64_t>(sizes.data(), num_tensors, stream);

  int threads = 256;
  int blocks = (max_size + threads - 1) / threads;
  dim3 grid(blocks, num_tensors);
  dim3 block(threads);

  fused_calculate_offsets_kernel<<<grid, block, 0, stream>>>(
      const_cast<const int64_t**>(inputs_ptrs.data()), offsets_ptrs.data(),
      d_sizes, num_tensors);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  CudaVecParam<int64_t*> cumsum_offsets_ptrs(num_tensors, stream);

  for (int64_t i = 0; i < num_tensors; ++i) {
    offsets[i] = torch::cumsum(offsets[i], 0);
    int64_t total_length = offsets[i].cpu()[-1].item<int64_t>();

    outputs[i].resize_({total_length});
    outputs_ptrs[i] = outputs[i].data_ptr<int8_t>();
    cumsum_offsets_ptrs[i] = offsets[i].data_ptr<int64_t>();
  }

  fused_int64_to_string_kernel<<<grid, block, 0, stream>>>(
      const_cast<const int64_t**>(inputs_ptrs.data()), outputs_ptrs.data(),
      const_cast<const int64_t**>(cumsum_offsets_ptrs.data()), d_sizes,
      num_tensors);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  delete_cuda_ptr(d_sizes);
}

}  // namespace functional
}  // namespace recis
