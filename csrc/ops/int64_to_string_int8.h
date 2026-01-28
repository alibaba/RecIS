#pragma once
#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace recis {
namespace functional {

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
fused_int64_to_string_int8(std::vector<torch::Tensor> inputs);

void fused_int64_to_string_int8_cuda(std::vector<torch::Tensor>& inputs,
                                     std::vector<torch::Tensor>& outputs,
                                     std::vector<torch::Tensor>& offsets);

}  // namespace functional
}  // namespace recis
