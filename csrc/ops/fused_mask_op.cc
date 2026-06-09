#include "ops/fused_mask_op.h"

#include <sstream>
#include <string>

namespace recis {
namespace functional {

std::vector<torch::Tensor> fused_string_mask(
    const std::vector<torch::Tensor> &inputs,
    const std::vector<torch::Tensor> &input_offsets,
    const std::vector<std::vector<std::string>> &masks) {
  int num_tensors = inputs.size();
  TORCH_CHECK(num_tensors > 0, "Input vector must not be empty");

  TORCH_CHECK(inputs[0].device().is_cuda(), "This op only supports CUDA");

  for (int i = 0; i < num_tensors; i++) {
    TORCH_CHECK(inputs[i].dtype() == torch::kChar,
                "All input tensors must be of type int8");
    TORCH_CHECK(inputs[i].dim() == 1,
                "All input tensors must be 1-dimensional");
  }
  std::vector<torch::Tensor> outputs(num_tensors);
  int total_num = 0;
  for (int i = 0; i < num_tensors; i++) {
    total_num = total_num + input_offsets[i].numel() - 1;
    outputs[i] = torch::empty({input_offsets[i].numel() - 1},
                              torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(inputs[i].device()));
  }
  if (total_num > 0) {
    if (inputs[0].device().is_cuda()) {
      fused_string_mask_cuda(inputs, input_offsets, outputs, masks);
    } else {
      throw std::runtime_error(
          "Fused string mask op only supports cuda tensors.");
    }
  }
  return outputs;
}

std::vector<torch::Tensor> fused_number_mask(
    const std::vector<torch::Tensor> &inputs,
    const std::vector<std::vector<double>> &masks) {
  int num_tensors = inputs.size();
  TORCH_CHECK(num_tensors > 0, "Input vector must not be empty");
  TORCH_CHECK(inputs[0].device().is_cuda(), "This op only supports CUDA");

  for (int i = 0; i < num_tensors; i++) {
    auto dtype = inputs[i].dtype();
    TORCH_CHECK(
        dtype == torch::kFloat32 || dtype == torch::kFloat64 ||
            dtype == torch::kInt32 || dtype == torch::kInt64,
        "All input tensors must be of type float32, float64, int32, or int64");
    TORCH_CHECK(inputs[i].dim() == 1,
                "All input tensors must be 1-dimensional");
  }

  std::vector<torch::Tensor> outputs(num_tensors);
  int total_num = 0;
  for (int i = 0; i < num_tensors; i++) {
    total_num += inputs[i].numel();
    outputs[i] =
        torch::empty({inputs[i].numel()}, torch::TensorOptions()
                                              .dtype(torch::kFloat32)
                                              .device(inputs[i].device()));
  }
  if (total_num > 0) {
    fused_number_mask_cuda(inputs, outputs, masks);
  }
  return outputs;
}

}  // namespace functional
}  // namespace recis
