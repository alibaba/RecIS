#include "compute_real_length.h"

#include "ragged_common.cuh"

namespace recis {
namespace functional {

torch::Tensor compute_real_length(
    torch::Tensor item_values, const std::vector<torch::Tensor>& item_offsets,
    torch::Tensor seller_values,
    const std::vector<torch::Tensor>& seller_offsets, int64_t num_ragged_dim) {
  // Input validation
  TORCH_CHECK(item_values.dim() == 1 || item_values.numel() == 0,
              "item_values must be 1D tensor or empty");
  TORCH_CHECK(seller_values.dim() == 1 || seller_values.numel() == 0,
              "seller_values must be 1D tensor or empty");
  TORCH_CHECK(item_offsets.size() == static_cast<size_t>(num_ragged_dim),
              "item_offsets size must equal num_ragged_dim");
  TORCH_CHECK(seller_offsets.size() == static_cast<size_t>(num_ragged_dim),
              "seller_offsets size must equal num_ragged_dim");

  // Check all tensors are on the same device
  // Combine all tensors for device check
  std::vector<at::Tensor> all_item_tensors = item_offsets;
  all_item_tensors.push_back(item_values);
  std::vector<at::Tensor> all_seller_tensors = seller_offsets;
  all_seller_tensors.push_back(seller_values);

  bool use_cuda = all_cuda(all_item_tensors) && all_cuda(all_seller_tensors);
  bool use_cpu = all_cpu(all_item_tensors) && all_cpu(all_seller_tensors);
  TORCH_CHECK(use_cuda || use_cpu,
              "All tensors must be on the same device (CPU or CUDA)");

  // Get batch size from the first offset tensor
  // For 3D ragged tensor: offsets[0] has shape [batch_size + 1]
  // For 4D ragged tensor: offsets[0] has shape [batch_size + 1]
  int64_t batch_size = item_offsets[0].size(0) - 1;

  // Create output tensor
  torch::Tensor output =
      torch::zeros({batch_size}, torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(item_values.device()));

  // Handle edge cases: empty batch or empty values
  // If batch_size is 0, or either item_values or seller_values is empty,
  // all real lengths should be 0 (already initialized)
  if (batch_size == 0 || item_values.numel() == 0 ||
      seller_values.numel() == 0) {
    return output;
  }

  if (use_cuda) {
    compute_real_length_cuda(item_values, item_offsets, seller_values,
                             seller_offsets, num_ragged_dim, output);
  } else {
    // CPU fallback implementation
    TORCH_CHECK(false, "CPU implementation not yet supported, please use CUDA");
  }

  return output;
}

}  // namespace functional
}  // namespace recis
