/* Copyright (C) 2016-2018 Alibaba Group Holding Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#include <cstddef>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "absl/log/check.h"
#include "absl/log/globals.h"
// #include "absl/log/log.h"
#include "arrow/record_batch.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/tensor_shape.h"
#include "column-io/framework/types.h"
namespace column {
namespace {
template <DataType T> std::string Tensor2Str(const Tensor &tensor) {
  if (!tensor.Initialized()) {
    return "";
  }
  std::stringstream ss;
  ss << "[";
  auto vec = tensor.Raw<typename EnumToDataType<T>::Type>();
  for (size_t idx = 0; idx < tensor.NumElements(); idx++) {
    ss << "," << vec[idx];
  }
  ss << "]";

  return ss.str();
}
} // namespace
Buffer::Buffer(Allocator *allocator, size_t size)
    : allocator_(allocator), size_(size),
      own_(true), parent_(nullptr) {
  auto pair = allocator->Allocate(size);
  begin_ = pair.first;
  ctx_ = pair.second;
}

Buffer::Buffer(Allocator *allocator, void *begin, void *ctx, size_t size, bool own)
    : allocator_(allocator), begin_(begin), ctx_(ctx), size_(size), parent_(nullptr),
      own_(own) {}

Buffer::Buffer(Allocator *allocator, void *begin, void *ctx, size_t size, Buffer *parent)
    : allocator_(allocator), begin_(begin), ctx_(ctx), size_(size), parent_(parent) {}

Buffer::~Buffer() {
  if (this->own_) {
    switch (dtype_) {
    case kString: {
      auto ptr = static_cast<EnumToDataType<kString>::Type *>(begin_);
      size_t size = SizeOfType(kString);
      for (size_t index = 0; index < size_ / size; index++) {
        ptr[index].~basic_string();
      }
      break;
    }
    default:
      break;
    }
    allocator_->Deallocate(ctx_);
  }
}

Tensor::Tensor() : state_(nullptr) {}

Tensor::Tensor(Allocator *allocator, const TensorShape &shape, DataType type, const DLDevice& dev)
    : state_(RefCountedPtr<State>::Create(
          RefCountedPtr<Buffer>::Create(allocator,
                                        shape.NumElements() * SizeOfType(type)),
          shape, type, dev)) {
  InitWithType(type);
}

Tensor::Tensor(Allocator *allocator, const TensorShape &shape, DataType type)
    : state_(RefCountedPtr<State>::Create(
          RefCountedPtr<Buffer>::Create(allocator,
                                        shape.NumElements() * SizeOfType(type)),
          shape, type)) {
  InitWithType(type);
}

Tensor::Tensor(DataType type) : Tensor(TensorShape(), type) {}

Tensor::Tensor(const TensorShape &shape, DataType type, Buffer *buffer)
    : state_(RefCountedPtr<State>::Create(RefCountedPtr<Buffer>(buffer), shape,
                                          type)) {
  InitWithType(type);
}

// NOTE: deal with null item
int64_t Tensor::NullCount() const {
    CHECK(Initialized());
    return null_count_;
}
bool Tensor::IsNull(int64_t i) const {
    return null_bitmap_buffer_ != nullptr &&
        !ArrowBitUtilGetBit(null_bitmap_buffer_->data(), i + null_bitmap_offset_);
}
void Tensor::SetNullBitmap(std::shared_ptr<ArrowBuffer> null_bitmap_buffer, int64_t null_bitmap_offset, int64_t null_count){
    CHECK(Initialized());
    null_bitmap_buffer_ = null_bitmap_buffer;
    null_bitmap_offset_ = null_bitmap_offset;
    null_count_ = null_count;
}
void Tensor::SetNullBitmapFromArray(const arrow::Array &array){
    SetNullBitmap(array.null_bitmap(), array.offset(), array.null_count());
}
// NOTE: deal with null item done


namespace {
static column::Allocator *EulerTensorAllocator() {
  return GetAllocator(false);
}
} // namespace

Tensor::Tensor(const TensorShape &shape, DataType type)
    : Tensor(EulerTensorAllocator(), shape, type) {}

Tensor::Tensor(Allocator *allocator, const TensorShape &shape, DataType type,
               void *data)
    : state_(RefCountedPtr<State>::Create(
          RefCountedPtr<Buffer>::Create(allocator, data, data,
                                        shape.NumElements() * SizeOfType(type)),
          shape, type)) {}

Tensor::~Tensor() {}

void Tensor::InitWithType(DataType type) {
  state_->buffer->SetType(type);
  switch (type) {
  case kString: {
    auto ptr = Raw<EnumToDataType<kString>::Type>();
    for (size_t index = 0; index < NumElements(); index++) {
      new (ptr + index) std::string;
    }
  }
  default: {
  }
  }
}
std::string Tensor::DebugString() const {
  switch (Type()) {
  case DataType::kString:
    return Tensor2Str<DataType::kString>(*this);
  case DataType::kBool:
    return Tensor2Str<DataType::kBool>(*this);
  case DataType::kInt8:
    return Tensor2Str<DataType::kInt8>(*this);
  case DataType::kInt16:
    return Tensor2Str<DataType::kInt16>(*this);
  case DataType::kInt32:
    return Tensor2Str<DataType::kInt32>(*this);
  case DataType::kInt64:
    return Tensor2Str<DataType::kInt64>(*this);
  case DataType::kUInt8:
    return Tensor2Str<DataType::kUInt8>(*this);
  case DataType::kUInt16:
    return Tensor2Str<DataType::kUInt16>(*this);
  case DataType::kUInt32:
    return Tensor2Str<DataType::kUInt32>(*this);
  case DataType::kUInt64:
    return Tensor2Str<DataType::kUInt64>(*this);
  case DataType::kFloat:
    return Tensor2Str<DataType::kFloat>(*this);
  case DataType::kDouble:
    return Tensor2Str<DataType::kDouble>(*this);
  default:
    return "Unknown type";
  }
}
} // namespace column
