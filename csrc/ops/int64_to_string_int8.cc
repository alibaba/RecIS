#include "ops/int64_to_string_int8.h"

#include <sstream>
#include <string>

namespace recis {
namespace functional {

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
fused_int64_to_string_int8(std::vector<torch::Tensor> inputs) {
  int num_tensors = inputs.size();
  TORCH_CHECK(num_tensors > 0, "Input vector must not be empty");

  TORCH_CHECK(inputs[0].device().is_cuda(), "This op only supports CUDA");

  for (int i = 0; i < num_tensors; i++) {
    TORCH_CHECK(inputs[i].dtype() == torch::kInt64,
                "All input tensors must be of type int64");
    TORCH_CHECK(inputs[i].dim() == 1,
                "All input tensors must be 1-dimensional");
  }

  std::vector<torch::Tensor> outputs(num_tensors);
  std::vector<torch::Tensor> offsets(num_tensors);

  for (int i = 0; i < num_tensors; i++) {
    int64_t num_elements = inputs[i].numel();
    offsets[i] =
        torch::zeros({num_elements + 1},
                     torch::dtype(torch::kInt64).device(inputs[i].device()));
    outputs[i] = torch::empty(
        {0}, torch::dtype(torch::kInt8).device(inputs[i].device()));
  }

  fused_int64_to_string_int8_cuda(inputs, outputs, offsets);

  return std::make_tuple(outputs, offsets);
}

}  // namespace functional
}  // namespace recis
