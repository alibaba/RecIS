#pragma once
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <atomic>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/string_view.h"
#include "embedding/id_allocator.h"
#include "torch/types.h"

namespace recis {
namespace embedding {

class IdMap : public torch::CustomClassHolder {
 public:
  const int64_t kNullIndex{-1};

  int64_t GetIdNum() { return id_allocator_->GetSize(); }
  int64_t GetActiveIdNum() { return id_allocator_->GetActiveSize(); }
  virtual void Lookup(const torch::Tensor &ids, torch::Tensor &output) = 0;
  virtual void LookupReadOnly(const torch::Tensor &ids,
                              torch::Tensor &output) = 0;
  virtual void InsertIds(const torch::Tensor &ids, torch::Tensor &output) = 0;
  virtual torch::Tensor Ids() = 0;
  virtual torch::Tensor Index() = 0;
  virtual void DeleteIds(const torch::Tensor &ids,
                         const torch::Tensor &index) = 0;
  virtual std::pair<torch::Tensor, torch::Tensor> SnapShot() = 0;
  virtual void Clear() = 0;
  virtual void Reserve(size_t id_size) = 0;
  virtual int64_t Size() const = 0;
  virtual int64_t Capacity() const = 0;
  ~IdMap() = default;

 protected:
  at::intrusive_ptr<recis::embedding::IdAllocator> id_allocator_;
};

torch::intrusive_ptr<IdMap> MakeCpuIdMap(torch::Device id_device);
torch::intrusive_ptr<IdMap> MakeGpuIdMap(torch::Device id_device);
torch::intrusive_ptr<IdMap> MakeIdMap(torch::Device id_device);

}  // namespace embedding
}  // namespace recis
