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

#ifndef _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_ALLOCATOR_H_
#define _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_ALLOCATOR_H_

#include "column-io/framework/gpu_runtime.h"
#include "column-io/framework/refcount.h"

namespace column {

class Allocator : public RefCounted {
public:
  // Return the data pointer and the context pointer.
  // The data pointer refers to the buffer allocated.
  // The context pointer is:
  //   1. the data pointer if normal allocator.
  //   2. the cache block pointer if pinned memory allocator.
  virtual std::pair<void*, void*> Allocate(size_t size) = 0;
  // Deallocate the context.
  virtual void Deallocate(void *ctx) = 0;
};

Allocator* GetAllocator(bool pin);

Allocator* CreateCudaAllocator(cudaStream_t stream, int device_id);

} // namespace column

#endif  // _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_ALLOCATOR_H_

