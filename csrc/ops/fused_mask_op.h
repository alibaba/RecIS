#pragma once
#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace recis {
namespace functional {

std::vector<torch::Tensor> fused_string_mask(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& input_offsets,
    const std::vector<std::vector<std::string>>& masks);

void fused_string_mask_cuda(const std::vector<torch::Tensor>& inputs,
                            const std::vector<torch::Tensor>& input_offsets,
                            const std::vector<torch::Tensor>& outputs,
                            const std::vector<std::vector<std::string>>& masks);

std::vector<torch::Tensor> fused_number_mask(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<std::vector<double>>& masks);

void fused_number_mask_cuda(const std::vector<torch::Tensor>& inputs,
                            const std::vector<torch::Tensor>& outputs,
                            const std::vector<std::vector<double>>& masks);

}  // namespace functional
}  // namespace recis
