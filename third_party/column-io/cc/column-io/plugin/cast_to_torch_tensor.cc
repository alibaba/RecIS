#include "column-io/plugin/cast_to_torch_tensor.h"

namespace column {
namespace plugin {

DLDataType ToDLDataType(column::DataType type) {
  switch (type) {
    case column::kBool:   return DLDataType{kDLBool,  8, 1};
    case column::kInt8:   return DLDataType{kDLInt,   8, 1};
    case column::kInt16:  return DLDataType{kDLInt,  16, 1};
    case column::kInt32:  return DLDataType{kDLInt,  32, 1};
    case column::kInt64:  return DLDataType{kDLInt,  64, 1};
    case column::kUInt8:  return DLDataType{kDLUInt,  8, 1};
    case column::kUInt16: return DLDataType{kDLUInt, 16, 1};
    case column::kUInt32: return DLDataType{kDLUInt, 32, 1};
    case column::kUInt64: return DLDataType{kDLUInt, 64, 1};
    case column::kFloat:  return DLDataType{kDLFloat, 32, 1};
    case column::kDouble: return DLDataType{kDLFloat, 64, 1};
    default:
      return DLDataType{0, 0, 0};
  }
}

inline void deleter(DLManagedTensor* tensor) {
  delete[] tensor->dl_tensor.shape;
  delete static_cast<Tensor*>(tensor->manager_ctx);
  delete tensor;
}

torch::Tensor CastTensorToTorchTensor(Tensor tensor, const DLDataType& dtype) {
  DLManagedTensor* managed = new DLManagedTensor();
  managed->dl_tensor.data = static_cast<void*>(tensor.mutable_data());
  managed->dl_tensor.device = tensor.Dev();
  managed->dl_tensor.dtype = dtype;
  managed->dl_tensor.ndim = tensor.dims();
  managed->dl_tensor.shape = new int64_t[tensor.dims()];
  memcpy(managed->dl_tensor.shape, tensor.Shape().Dims().data(),
    sizeof(int64_t) * tensor.dims());
  managed->dl_tensor.strides = nullptr;
  managed->manager_ctx = new Tensor(tensor);
  managed->deleter = deleter;
  return at::fromDLPack(managed);
}

}
}
