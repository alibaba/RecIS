#include <ATen/ATen.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/Parallel.h"

namespace recis {
namespace functional {
std::tuple<torch::Tensor, std::vector<torch::Tensor>> dense_to_ragged(
    const torch::Tensor& data, bool check_invalid,
    const torch::Tensor& invalid_value);

std::tuple<torch::Tensor, torch::Tensor> dense_to_ragged_cpu(
    const torch::Tensor& data, const torch::Tensor& invalid_value);

std::tuple<torch::Tensor, torch::Tensor> dense_to_ragged_cuda(
    const torch::Tensor& data, const torch::Tensor& invalid_value);

}  // namespace functional
}  // namespace recis
