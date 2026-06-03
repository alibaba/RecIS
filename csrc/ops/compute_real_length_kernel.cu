#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>

#include "ragged_common.cuh"

namespace recis {
namespace functional {

// Kernel for 3D RaggedTensor (numeric types: int32, int64, float, etc.)
// Each block processes one batch sample
template <typename index_t, typename value_t>
__global__ void compute_real_length_3d_kernel(
    const value_t* __restrict__ item_values, StackArray<index_t*> item_offsets,
    const value_t* __restrict__ seller_values,
    StackArray<index_t*> seller_offsets, float* __restrict__ output,
    StackArray<index_t> offsets_size) {
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  // Get the range of sequence positions for this batch
  // offsets[0] is the batch-level offset, offsets[1] is the sequence-level
  // offset
  const index_t seq_start = item_offsets.vals[0][batch_idx];
  const index_t seq_end = item_offsets.vals[0][batch_idx + 1];
  const index_t num_seq_positions = seq_end - seq_start;

  // Count valid sequence positions
  int local_count = 0;

  // Each thread processes multiple sequence positions
  for (index_t seq_local_idx = tid; seq_local_idx < num_seq_positions;
       seq_local_idx += block_size) {
    const index_t seq_global_idx = seq_start + seq_local_idx;

    // Get the range of values for this sequence position in item
    const index_t item_start = item_offsets.vals[1][seq_global_idx];
    const index_t item_end = item_offsets.vals[1][seq_global_idx + 1];
    const index_t item_len = item_end - item_start;

    // Get the range of values for this sequence position in seller
    const index_t seller_start = seller_offsets.vals[1][seq_global_idx];
    const index_t seller_end = seller_offsets.vals[1][seq_global_idx + 1];
    const index_t seller_len = seller_end - seller_start;

    // Check if this position is valid (both item and seller have non-zero
    // values)
    bool item_has_nonzero = false;
    bool seller_has_nonzero = false;

    if (item_len > 0) {
      for (index_t i = 0; i < item_len; ++i) {
        if (item_values[item_start + i] != static_cast<value_t>(0)) {
          item_has_nonzero = true;
          break;
        }
      }
    }

    if (seller_len > 0) {
      for (index_t i = 0; i < seller_len; ++i) {
        if (seller_values[seller_start + i] != static_cast<value_t>(0)) {
          seller_has_nonzero = true;
          break;
        }
      }
    }

    // If both are valid, increment count
    if (item_has_nonzero && seller_has_nonzero) {
      local_count++;
    }
  }

  // Block-level reduction using shared memory
  extern __shared__ int shared_counts[];
  shared_counts[tid] = local_count;
  __syncthreads();

  // Parallel reduction
  for (int stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared_counts[tid] += shared_counts[tid + stride];
    }
    __syncthreads();
  }

  // Write result
  if (tid == 0) {
    output[batch_idx] = static_cast<float>(shared_counts[0]);
  }
}

// Kernel for 4D RaggedTensor (string type stored as int8)
// Each block processes one batch sample
template <typename index_t>
__global__ void compute_real_length_4d_kernel(
    const int8_t* __restrict__ item_values, StackArray<index_t*> item_offsets,
    const int8_t* __restrict__ seller_values,
    StackArray<index_t*> seller_offsets, float* __restrict__ output,
    StackArray<index_t> offsets_size) {
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  // Get the range of sequence positions for this batch
  const index_t seq_start = item_offsets.vals[0][batch_idx];
  const index_t seq_end = item_offsets.vals[0][batch_idx + 1];
  const index_t num_seq_positions = seq_end - seq_start;

  // Count valid sequence positions
  int local_count = 0;

  // Each thread processes multiple sequence positions
  for (index_t seq_local_idx = tid; seq_local_idx < num_seq_positions;
       seq_local_idx += block_size) {
    const index_t seq_global_idx = seq_start + seq_local_idx;

    // Get the range of string indices for this sequence position in item
    const index_t item_str_start = item_offsets.vals[1][seq_global_idx];
    const index_t item_str_end = item_offsets.vals[1][seq_global_idx + 1];
    const index_t item_num_strings = item_str_end - item_str_start;

    // Get the range of string indices for this sequence position in seller
    const index_t seller_str_start = seller_offsets.vals[1][seq_global_idx];
    const index_t seller_str_end = seller_offsets.vals[1][seq_global_idx + 1];
    const index_t seller_num_strings = seller_str_end - seller_str_start;

    // Check if this position is valid (both item and seller have non-zero
    // values) For strings: check if the string is not empty and not "0"
    bool item_has_nonzero = false;
    bool seller_has_nonzero = false;

    // Check item strings
    if (item_num_strings > 0) {
      for (index_t s = 0; s < item_num_strings && !item_has_nonzero; ++s) {
        const index_t str_idx = item_str_start + s;
        const index_t char_start = item_offsets.vals[2][str_idx];
        const index_t char_end = item_offsets.vals[2][str_idx + 1];
        const index_t str_len = char_end - char_start;

        // Check if string is not empty and not "0"
        if (str_len > 0) {
          // Check if it's not just "0"
          bool is_zero_string =
              (str_len == 1 && item_values[char_start] == '0');
          if (!is_zero_string) {
            item_has_nonzero = true;
          }
        }
      }
    }

    // Check seller strings
    if (seller_num_strings > 0) {
      for (index_t s = 0; s < seller_num_strings && !seller_has_nonzero; ++s) {
        const index_t str_idx = seller_str_start + s;
        const index_t char_start = seller_offsets.vals[2][str_idx];
        const index_t char_end = seller_offsets.vals[2][str_idx + 1];
        const index_t str_len = char_end - char_start;

        // Check if string is not empty and not "0"
        if (str_len > 0) {
          // Check if it's not just "0"
          bool is_zero_string =
              (str_len == 1 && seller_values[char_start] == '0');
          if (!is_zero_string) {
            seller_has_nonzero = true;
          }
        }
      }
    }

    // If both are valid, increment count
    if (item_has_nonzero && seller_has_nonzero) {
      local_count++;
    }
  }

  // Block-level reduction using shared memory
  extern __shared__ int shared_counts[];
  shared_counts[tid] = local_count;
  __syncthreads();

  // Parallel reduction
  for (int stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared_counts[tid] += shared_counts[tid + stride];
    }
    __syncthreads();
  }

  // Write result
  if (tid == 0) {
    output[batch_idx] = static_cast<float>(shared_counts[0]);
  }
}

// Helper function to copy offsets to device-accessible structure
template <typename index_t>
void prepare_offsets(const std::vector<torch::Tensor>& offsets,
                     StackArray<index_t*>& dev_offsets,
                     StackArray<index_t>& offsets_size,
                     std::vector<index_t*>& host_ptrs) {
  const size_t num_dims = offsets.size();
  TORCH_CHECK(num_dims <= kStackArrayMaxDims, "Too many ragged dimensions");

  dev_offsets.ndim = num_dims;
  offsets_size.ndim = num_dims;

  host_ptrs.resize(num_dims);
  for (size_t d = 0; d < num_dims; ++d) {
    host_ptrs[d] = offsets[d].data_ptr<index_t>();
    dev_offsets.vals[d] = host_ptrs[d];
    offsets_size.vals[d] = offsets[d].size(0);
  }
}

void compute_real_length_cuda(torch::Tensor item_values,
                              const std::vector<torch::Tensor>& item_offsets,
                              torch::Tensor seller_values,
                              const std::vector<torch::Tensor>& seller_offsets,
                              int64_t num_ragged_dim, torch::Tensor output) {
  auto stream = c10::cuda::getCurrentCUDAStream();

  const int64_t batch_size = output.size(0);

  // Determine index type from offsets
  TORCH_CHECK(all_same_type(item_offsets, torch::kInt32) ||
                  all_same_type(item_offsets, torch::kInt64),
              "Offsets must be int32 or int64");

  bool use_int64 = item_offsets[0].scalar_type() == torch::kInt64;

  // Launch kernel based on ragged dimension
  const int threads_per_block = 256;
  const int blocks = batch_size;
  const size_t shared_mem_size = threads_per_block * sizeof(int);

  // Helper lambda to launch 3D kernel
  auto launch_3d_kernel = [&]([[maybe_unused]] auto value_type_tag) {
    using value_t = decltype(value_type_tag);
    AT_DISPATCH_INDEX_TYPES(
        item_offsets[0].scalar_type(), "compute_real_length_3d_kernel_index",
        ([&] {
          StackArray<index_t*> dev_item_offsets, dev_seller_offsets;
          StackArray<index_t> item_offsets_size, seller_offsets_size;
          std::vector<index_t*> item_host_ptrs, seller_host_ptrs;

          prepare_offsets(item_offsets, dev_item_offsets, item_offsets_size,
                          item_host_ptrs);
          prepare_offsets(seller_offsets, dev_seller_offsets,
                          seller_offsets_size, seller_host_ptrs);

          compute_real_length_3d_kernel<index_t, value_t>
              <<<blocks, threads_per_block, shared_mem_size, stream>>>(
                  item_values.data_ptr<value_t>(), dev_item_offsets,
                  seller_values.data_ptr<value_t>(), dev_seller_offsets,
                  output.data_ptr<float>(), item_offsets_size);
        }));
  };

  if (num_ragged_dim == 2) {
    // 3D RaggedTensor (numeric types) - manual type dispatch
    auto scalar_type = item_values.scalar_type();
    switch (scalar_type) {
      case torch::kInt8:
        launch_3d_kernel(int8_t{});
        break;
      case torch::kInt16:
        launch_3d_kernel(int16_t{});
        break;
      case torch::kInt32:
        launch_3d_kernel(int32_t{});
        break;
      case torch::kInt64:
        launch_3d_kernel(int64_t{});
        break;
      case torch::kFloat32:
        launch_3d_kernel(float{});
        break;
      case torch::kFloat64:
        launch_3d_kernel(double{});
        break;
      default:
        TORCH_CHECK(false, "Unsupported value type for 3D RaggedTensor: ",
                    item_values.scalar_type());
    }
  } else if (num_ragged_dim == 3) {
    // 4D RaggedTensor (string as int8)
    TORCH_CHECK(item_values.scalar_type() == torch::kInt8,
                "4D RaggedTensor values must be int8 type for string storage");
    TORCH_CHECK(seller_values.scalar_type() == torch::kInt8,
                "4D RaggedTensor values must be int8 type for string storage");

    AT_DISPATCH_INDEX_TYPES(
        item_offsets[0].scalar_type(), "compute_real_length_4d_kernel_index",
        ([&] {
          StackArray<index_t*> dev_item_offsets, dev_seller_offsets;
          StackArray<index_t> item_offsets_size, seller_offsets_size;
          std::vector<index_t*> item_host_ptrs, seller_host_ptrs;

          prepare_offsets(item_offsets, dev_item_offsets, item_offsets_size,
                          item_host_ptrs);
          prepare_offsets(seller_offsets, dev_seller_offsets,
                          seller_offsets_size, seller_host_ptrs);

          compute_real_length_4d_kernel<index_t>
              <<<blocks, threads_per_block, shared_mem_size, stream>>>(
                  item_values.data_ptr<int8_t>(), dev_item_offsets,
                  seller_values.data_ptr<int8_t>(), dev_seller_offsets,
                  output.data_ptr<float>(), item_offsets_size);
        }));
  } else {
    TORCH_CHECK(false, "Unsupported num_ragged_dim: ", num_ragged_dim,
                ". Only 2 (3D tensor) and 3 (4D tensor) are supported.");
  }

  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace functional
}  // namespace recis
