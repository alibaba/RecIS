#pragma once
#include <torch/extension.h>

#include <tuple>

#include "ATen/core/TensorBody.h"
namespace recis {
namespace functional {
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
gather_ragged_by_padded_index(at::Tensor drop_num, at::Tensor drop_side,
                              at::Tensor pad_num, at::Tensor pad_side,
                              at::Tensor padded_indices,
                              at::Tensor src_row_indices,
                              std::vector<at::Tensor> offsets,
                              std::vector<at::Tensor> values);

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
gather_ragged_by_padded_index_cuda(at::Tensor drop_num, at::Tensor drop_side,
                                   at::Tensor pad_num, at::Tensor pad_side,
                                   at::Tensor padded_indices,
                                   at::Tensor src_row_indices,
                                   std::vector<at::Tensor> offsets,
                                   std::vector<at::Tensor> values);
}  // namespace functional
}  // namespace recis