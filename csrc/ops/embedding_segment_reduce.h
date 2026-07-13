#pragma once
#include <torch/extension.h>

#include <cstdint>
#include <vector>
namespace recis {
namespace functional {
enum class ReduceMode { SUM, MEAN, TILE };
at::Tensor segment_reduce_forward(at::Tensor unique_emb,
                                  c10::optional<at::Tensor> weight,
                                  at::Tensor reverse_indices,
                                  at::Tensor offsets, std::string mode);
// Backward function dispatcher
at::Tensor segment_reduce_backward(at::Tensor grad_output,
                                   c10::optional<at::Tensor> weight,
                                   at::Tensor reverse_indices,
                                   at::Tensor offsets, int64_t unique_size,
                                   std::string mode);
at::Tensor merge_offsets(const std::vector<at::Tensor>& offsets,
                         torch::Tensor& max_value);

at::Tensor gen_segment_indices_by_offset(torch::Tensor offset,
                                         int64_t total_size);

// void embedding_segment_reduce_backward();
}  // namespace functional

}  // namespace recis
