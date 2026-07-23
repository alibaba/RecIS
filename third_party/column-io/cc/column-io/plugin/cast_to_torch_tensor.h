#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include <torch/torch.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>

namespace column {
namespace plugin {
  extern void deleter(DLManagedTensor* tensor);
  extern torch::Tensor CastTensorToTorchTensor(Tensor tensor, const DLDataType& dtype);
  extern DLDataType ToDLDataType(column::DataType type);
}
}
