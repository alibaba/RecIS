#pragma once

#include <torch/extension.h>

namespace recis {
namespace functional {

/**
 * @brief Compute real (valid) sequence length for ragged tensors.
 *
 * For each batch sample, counts the number of sequence positions where
 * both item and seller features are non-zero (valid).
 *
 * @param item_values Flattened item feature values
 * @param item_offsets Item feature offsets (list of offset tensors)
 * @param seller_values Flattened seller feature values
 * @param seller_offsets Seller feature offsets (list of offset tensors)
 * @param num_ragged_dim Number of ragged dimensions (2 for 3D tensor, 3 for 4D
 * tensor)
 * @return torch::Tensor Real lengths array of shape [batch_size]
 */
torch::Tensor compute_real_length(
    torch::Tensor item_values, const std::vector<torch::Tensor>& item_offsets,
    torch::Tensor seller_values,
    const std::vector<torch::Tensor>& seller_offsets, int64_t num_ragged_dim);

/**
 * @brief CUDA kernel implementation for computing real sequence length.
 *
 * @param item_values Flattened item feature values
 * @param item_offsets Item feature offsets
 * @param seller_values Flattened seller feature values
 * @param seller_offsets Seller feature offsets
 * @param num_ragged_dim Number of ragged dimensions
 * @param output Output tensor for real lengths
 */
void compute_real_length_cuda(torch::Tensor item_values,
                              const std::vector<torch::Tensor>& item_offsets,
                              torch::Tensor seller_values,
                              const std::vector<torch::Tensor>& seller_offsets,
                              int64_t num_ragged_dim, torch::Tensor output);

}  // namespace functional
}  // namespace recis
