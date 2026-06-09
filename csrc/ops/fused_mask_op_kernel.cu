#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "cuda/cuda_param.cuh"
#include "cuda/utils.cuh"
#include "ops/fused_mask_op.h"

namespace recis {
namespace functional {

using namespace recis::cuda;

constexpr int KBLOCK_SIZE = 256;

// Device function: compare two byte sequences for equality
__device__ bool device_str_equal(const int8_t* str_a, int32_t len_a,
                                 const int8_t* str_b, int32_t len_b) {
  if (len_a != len_b) return false;
  for (int32_t i = 0; i < len_a; ++i) {
    if (str_a[i] != str_b[i]) return false;
  }
  return true;
}

// CUDA kernel: for each string in inputs, check if it matches any mask string.
// If matched -> 0.0, otherwise -> 1.0
template <typename scalar_t>
__global__ void string_mask_kernel(
    int8_t** inputs_ptrs, scalar_t** input_offsets_ptrs, float** outputs_ptrs,
    int8_t** mask_data_ptrs, int32_t** mask_offsets_ptrs, int32_t* mask_counts,
    int64_t* sizes, int64_t num_tensors) {
  int64_t vec_id = blockIdx.y;
  if (vec_id >= num_tensors) return;

  int64_t size_local = sizes[vec_id];
  int64_t threads_num = blockDim.x * gridDim.x;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  int8_t* input_data = inputs_ptrs[vec_id];
  scalar_t* offsets = input_offsets_ptrs[vec_id];
  float* output = outputs_ptrs[vec_id];
  int8_t* mask_data = mask_data_ptrs[vec_id];
  int32_t* mask_offsets = mask_offsets_ptrs[vec_id];
  int32_t num_masks = mask_counts[vec_id];

  for (int64_t index = tid; index < size_local; index += threads_num) {
    scalar_t str_start = offsets[index];
    scalar_t str_len = offsets[index + 1] - str_start;
    const int8_t* str_ptr = input_data + str_start;

    bool matched = false;
    for (int32_t m = 0; m < num_masks; ++m) {
      int32_t mask_start = mask_offsets[m];
      int32_t mask_len = mask_offsets[m + 1] - mask_start;
      const int8_t* mask_ptr = mask_data + mask_start;

      if (device_str_equal(str_ptr, static_cast<int32_t>(str_len), mask_ptr,
                           mask_len)) {
        matched = true;
        break;
      }
    }
    output[index] = matched ? 0.0f : 1.0f;
  }
}

void fused_string_mask_cuda(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& input_offsets,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<std::vector<std::string>>& masks) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t num_tensors = inputs.size();
  if (num_tensors == 0) return;

  AT_DISPATCH_INTEGRAL_TYPES(
      input_offsets[0].scalar_type(), "fused_string_mask_cuda", ([&] {
        // Prepare device pointers for inputs, offsets, outputs
        CudaVecParam<int8_t*> inputs_ptrs(num_tensors, stream);
        CudaVecParam<scalar_t*> input_offsets_ptrs(num_tensors, stream);
        CudaVecParam<float*> outputs_ptrs(num_tensors, stream);
        CudaVecParam<int8_t*> mask_data_ptrs(num_tensors, stream);
        CudaVecParam<int32_t*> mask_offsets_ptrs(num_tensors, stream);

        std::vector<int64_t> sizes(num_tensors);
        std::vector<int32_t> mask_counts_host(num_tensors);

        // Temporary storage for device-side mask data to ensure cleanup
        std::vector<int8_t*> device_mask_data(num_tensors, nullptr);
        std::vector<int32_t*> device_mask_offsets(num_tensors, nullptr);

        for (int64_t i = 0; i < num_tensors; ++i) {
          sizes[i] = outputs[i].numel();
          inputs_ptrs[i] = inputs[i].data_ptr<int8_t>();
          input_offsets_ptrs[i] = input_offsets[i].data_ptr<scalar_t>();
          outputs_ptrs[i] = outputs[i].data_ptr<float>();

          // Build mask data: concatenate all mask strings into a flat int8
          // array
          const auto& mask_list = masks[i];
          int32_t num_masks = static_cast<int32_t>(mask_list.size());
          mask_counts_host[i] = num_masks;

          // Build host-side mask bytes and offsets
          std::vector<int8_t> mask_bytes;
          std::vector<int32_t> mask_offs(num_masks + 1);
          mask_offs[0] = 0;
          for (int32_t m = 0; m < num_masks; ++m) {
            const std::string& s = mask_list[m];
            for (char c : s) {
              mask_bytes.push_back(static_cast<int8_t>(c));
            }
            mask_offs[m + 1] = static_cast<int32_t>(mask_bytes.size());
          }

          // Copy mask data to device
          if (!mask_bytes.empty()) {
            device_mask_data[i] = cuda_malloc_and_copy<int8_t>(
                mask_bytes.data(), static_cast<int>(mask_bytes.size()), stream);
          } else {
            device_mask_data[i] = nullptr;
          }
          device_mask_offsets[i] = cuda_malloc_and_copy<int32_t>(
              mask_offs.data(), static_cast<int>(mask_offs.size()), stream);

          mask_data_ptrs[i] = device_mask_data[i];
          mask_offsets_ptrs[i] = device_mask_offsets[i];
        }

        // Compute grid dimensions
        int64_t sm_count = get_sm_count();
        int64_t max_size = 0;
        for (int64_t i = 0; i < num_tensors; ++i) {
          max_size = std::max(max_size, sizes[i]);
        }
        int64_t block_num =
            std::min(sm_count * 8, (max_size + KBLOCK_SIZE - 1) / KBLOCK_SIZE);
        dim3 grid(block_num, num_tensors);
        dim3 block(KBLOCK_SIZE);

        int64_t* d_sizes =
            cuda_malloc_and_copy<int64_t>(sizes.data(), num_tensors, stream);
        int32_t* d_mask_counts = cuda_malloc_and_copy<int32_t>(
            mask_counts_host.data(), num_tensors, stream);

        string_mask_kernel<scalar_t><<<grid, block, 0, stream>>>(
            inputs_ptrs.data(), input_offsets_ptrs.data(), outputs_ptrs.data(),
            mask_data_ptrs.data(), mask_offsets_ptrs.data(), d_mask_counts,
            d_sizes, num_tensors);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Cleanup temporary device allocations
        delete_cuda_ptr(d_sizes);
        delete_cuda_ptr(d_mask_counts);
        for (int64_t i = 0; i < num_tensors; ++i) {
          if (device_mask_data[i]) delete_cuda_ptr(device_mask_data[i]);
          delete_cuda_ptr(device_mask_offsets[i]);
        }
      }));
}

// CUDA kernel: for each number in inputs, check if it matches any mask value.
// If matched -> 0.0, otherwise -> 1.0
template <typename scalar_t>
__global__ void number_mask_kernel(scalar_t** inputs_ptrs, float** outputs_ptrs,
                                   double** mask_values_ptrs,
                                   int32_t* mask_counts, int64_t* sizes,
                                   int64_t num_tensors) {
  int64_t vec_id = blockIdx.y;
  if (vec_id >= num_tensors) return;

  int64_t size_local = sizes[vec_id];
  int64_t threads_num = blockDim.x * gridDim.x;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  scalar_t* input_data = inputs_ptrs[vec_id];
  float* output = outputs_ptrs[vec_id];
  double* mask_vals = mask_values_ptrs[vec_id];
  int32_t num_masks = mask_counts[vec_id];

  for (int64_t index = tid; index < size_local; index += threads_num) {
    scalar_t val = input_data[index];
    bool matched = false;
    for (int32_t m = 0; m < num_masks; ++m) {
      if (val == static_cast<scalar_t>(mask_vals[m])) {
        matched = true;
        break;
      }
    }
    output[index] = matched ? 0.0f : 1.0f;
  }
}

void fused_number_mask_cuda(const std::vector<torch::Tensor>& inputs,
                            const std::vector<torch::Tensor>& outputs,
                            const std::vector<std::vector<double>>& masks) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t num_tensors = inputs.size();
  if (num_tensors == 0) return;

  AT_DISPATCH_ALL_TYPES(
      inputs[0].scalar_type(), "fused_number_mask_cuda", ([&]() {
        CudaVecParam<scalar_t*> inputs_ptrs(num_tensors, stream);
        CudaVecParam<float*> outputs_ptrs(num_tensors, stream);
        CudaVecParam<double*> mask_values_ptrs(num_tensors, stream);

        std::vector<int64_t> sizes(num_tensors);
        std::vector<int32_t> mask_counts_host(num_tensors);
        std::vector<double*> device_mask_values(num_tensors, nullptr);

        for (int64_t i = 0; i < num_tensors; ++i) {
          sizes[i] = inputs[i].numel();
          inputs_ptrs[i] = inputs[i].data_ptr<scalar_t>();
          outputs_ptrs[i] = outputs[i].data_ptr<float>();

          auto mask_list = masks[i];
          int32_t num_masks = static_cast<int32_t>(mask_list.size());
          mask_counts_host[i] = num_masks;

          if (num_masks > 0) {
            device_mask_values[i] = cuda_malloc_and_copy<double>(
                mask_list.data(), num_masks, stream);
          }
          mask_values_ptrs[i] = device_mask_values[i];
        }

        int64_t sm_count = get_sm_count();
        int64_t max_size = 0;
        for (int64_t i = 0; i < num_tensors; ++i) {
          max_size = std::max(max_size, sizes[i]);
        }
        int64_t block_num =
            std::min(sm_count * 8, (max_size + KBLOCK_SIZE - 1) / KBLOCK_SIZE);
        dim3 grid(block_num, num_tensors);
        dim3 block(KBLOCK_SIZE);

        int64_t* d_sizes =
            cuda_malloc_and_copy<int64_t>(sizes.data(), num_tensors, stream);
        int32_t* d_mask_counts = cuda_malloc_and_copy<int32_t>(
            mask_counts_host.data(), num_tensors, stream);

        number_mask_kernel<scalar_t><<<grid, block, 0, stream>>>(
            inputs_ptrs.data(), outputs_ptrs.data(), mask_values_ptrs.data(),
            d_mask_counts, d_sizes, num_tensors);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));

        delete_cuda_ptr(d_sizes);
        delete_cuda_ptr(d_mask_counts);
        for (int64_t i = 0; i < num_tensors; ++i) {
          if (device_mask_values[i]) delete_cuda_ptr(device_mask_values[i]);
        }
      }));
}

}  // namespace functional
}  // namespace recis
