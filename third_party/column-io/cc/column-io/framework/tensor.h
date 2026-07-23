/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef EULER_CORE_FRAMEWORK_TENSOR_H_
#define EULER_CORE_FRAMEWORK_TENSOR_H_

#include <memory>
#include <string>
#include <vector>
#include <dlpack.h>

// #include "absl/log/log.h"
#include "arrow/buffer.h"
#include "arrow/array/array_base.h"
#include "column-io/framework/allocator.h"
#include "column-io/framework/refcount.h"
#include "column-io/framework/tensor_shape.h"
#include "column-io/framework/types.h"

namespace column {

typedef arrow::Buffer ArrowBuffer; // to diff with column::Buffer
class Buffer : public RefCounted {
public:
  Buffer(Allocator *allocator, size_t size);
  Buffer(Allocator *allocator, void *begin, void *ctx, size_t size, bool own = true);
  Buffer(Allocator *allocator, void *begin, void *ctx, size_t size, Buffer *parent);
  ~Buffer();

  virtual void *begin() const { return begin_; }

  virtual size_t size() const { return size_; }

  virtual void SetType(DataType dtype) { dtype_ = dtype; }

protected:
  RefCountedPtr<Allocator> allocator_;
  void *begin_;
  void *ctx_;
  size_t size_;
  bool own_{true};
  DataType dtype_;
  RefCountedPtr<Buffer> parent_;
};

class Tensor {
public:
  // for empty
  Tensor();
  Tensor(Tensor&& t) : state_(std::move(t.state_)) {}
  Tensor(const Tensor& t) : state_(t.state_) {}
  Tensor& operator=(Tensor&& t) {
    state_ = std::move(t.state_);
    return *this;
  }
  Tensor& operator=(const Tensor& t) {
    state_ = t.state_;
    return *this;
  }
  // for scalar
  Tensor(DataType type);
  // for tensor
  Tensor(const TensorShape &shape, DataType type);
  Tensor(DataType type, const TensorShape &shape) : Tensor(shape, type) {}
  Tensor(Allocator *allocator, const TensorShape &shape, DataType type);
  Tensor(Allocator *allocator, const TensorShape &shape, DataType type, const DLDevice& dev);
  Tensor(const TensorShape &shape, DataType type, Buffer *buffer);
  Tensor(Allocator *allocator, const TensorShape &shape, DataType type,
         void *data);

  ~Tensor();

  bool Initialized() const { return state_.get() != nullptr; }

  template <typename T> T *Raw() const {
    CHECK(Initialized()) << "Tensor Not Initialized";
    /* CHECK(DataTypeToEnum<T>::v() == Type())
            << "type error tensor type is [" << Type() << "]"
            << " cast type is [" << DataTypeToEnum<T>::value << "]";
    */
    return reinterpret_cast<T *>(state_->buffer->begin());
  }

  const char *data() const {
    return reinterpret_cast<char *>(state_->buffer->begin());
  }
  char *mutable_data() {
    return reinterpret_cast<char *>(state_->buffer->begin());
  }

  const TensorShape &Shape() const {
    CHECK(Initialized());
    return state_->shape;
  }

  int NumElements() const {
    CHECK(Initialized());
    return state_->shape.NumElements();
  }
  int32_t dims() const { return Shape().Size(); }

  DataType Type() const {
    CHECK(Initialized());
    return state_->type;
  }

  size_t TotalBytes() const {
    return state_->shape.NumElements() * SizeOfType(state_->type);
  }

  template <typename T> T Scalar() const {
    CHECK(Shape().IsScalar()) << "tensor is not scalar";
    return *(Raw<T>());
  }

  std::vector<std::string> Flat() const {
    CHECK(Type() == kString) << "Tensor type is not kString";

    const std::string* src = Raw<std::string>();
    std::vector<std::string> ret;
    ret.reserve(NumElements());
    for (int i = 0; i < NumElements(); ++i) {
      ret.emplace_back(src[i]);   // deep copy
    }
    return ret;
  }

  Buffer *GetBuffer() const {
    CHECK(Initialized());
    return state_->buffer.get();
  }

  const DLDevice& Dev() const {
    return state_->dev;
  } 

  std::string DebugString() const;

  int64_t NullCount() const; // NOTE: deal with null item
  bool IsNull(int64_t i) const; // NOTE: deal with null item
  void SetNullBitmap(std::shared_ptr<ArrowBuffer> null_bitmap_buffer, int64_t null_bitmap_offset, int64_t null_count);
  void SetNullBitmapFromArray(const arrow::Array &array);

private:
  static inline bool ArrowBitUtilGetBit(const uint8_t* bits, uint64_t i) { // clone from arrow/util/bit_util.h
    return (bits[i >> 3] >> (i & 0x07)) & 1;
  }
  int64_t null_count_ = 0;
  int64_t null_bitmap_offset_ = 0;
  std::shared_ptr<ArrowBuffer> null_bitmap_buffer_ = nullptr; 

  void InitWithType(DataType type);
  class State : public RefCounted {
  public:
    State(const RefCountedPtr<Buffer> &buffer_, const TensorShape &shape_,
          DataType type_, const DLDevice& dev_)
        : buffer(buffer_), shape(shape_), type(type_), dev(dev_) {}
    State(const RefCountedPtr<Buffer> &buffer_, const TensorShape &shape_,
          DataType type_)
        : buffer(buffer_), shape(shape_), type(type_) {}
    RefCountedPtr<Buffer> buffer;
    TensorShape shape;
    DataType type;
    DLDevice dev{kDLCPU, 0};
  };
  RefCountedPtr<State> state_;
};
} // namespace column

#endif // EULER_CORE_FRAMEWORK_TENSOR_H_
