#pragma once
#include <c10/util/flat_hash_map.h>
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
#include "embedding/children_info.h"
#include "embedding/cpu_id_map.h"
#include "embedding/id_map.h"
#include "embedding/initializer.h"
#include "embedding/slice_info.h"
#include "embedding/slot_group.h"
#include "torch/types.h"

class Hashtable;
using HashTablePtr = c10::intrusive_ptr<Hashtable>;

class Hashtable : public torch::CustomClassHolder {
 public:
  using IndexType = int64_t;
  const IndexType kNullIndex{-1};

  struct HashtableConfig {
    int64_t block_size;
    std::vector<int64_t> embedding_shape;
    torch::Dtype dtype;
    torch::Device device;
    bool coalesce;
    std::vector<std::string> children;
    at::intrusive_ptr<recis::embedding::Generator> generator;
    int64_t slice_begin;
    int64_t slice_end;
    int64_t slice_size;

    HashtableConfig(int64_t bs, const std::vector<int64_t> &shape,
                    torch::Dtype dt, torch::Device dev, bool coal,
                    const std::vector<std::string> &childs,
                    at::intrusive_ptr<recis::embedding::Generator> gen,
                    int64_t sb, int64_t se, int64_t ss)
        : block_size(bs),
          embedding_shape(shape),
          dtype(dt),
          device(dev),
          coalesce(coal),
          children(childs),
          generator(gen),
          slice_begin(sb),
          slice_end(se),
          slice_size(ss) {}
  };

  static c10::intrusive_ptr<Hashtable> Make(
      int64_t block_size, const std::vector<int64_t> &embedding_shape,
      torch::Dtype dtype, torch::Device device, bool coalesce,
      const std::vector<std::string> &children,
      at::intrusive_ptr<recis::embedding::Generator> generator,
      int64_t slice_begin, int64_t slice_end, int64_t slice_size);
  Hashtable(int64_t block_size, const std::vector<int64_t> &embedding_shape,
            torch::Dtype dtype, torch::Device device, bool coalesce,
            const std::vector<std::string> &children,
            at::intrusive_ptr<recis::embedding::Generator> generator,
            int64_t slice_begin, int64_t slice_end, int64_t slice_size);

  std::tuple<torch::Tensor, torch::Tensor> EmbeddingLookup(
      const torch::Tensor &ids, bool readonly);
  torch::Tensor InsertLookupIndex(const torch::Tensor &ids);
  torch::Tensor InsertLookupIndexWithIndicator(const torch::Tensor &ids,
                                               const torch::Tensor &indicator);

  void Insert(const torch::Tensor &ids, const torch::Tensor &embedding);
  void UpdateSlot(const std::string &slot_name, const torch::Tensor &ids,
                  torch::Tensor embedding);
  void AppendStepFilterSlot(const std::string &filter_name, int64_t init_step);
  void AppendFilterSlot(
      const std::string &filter_name, torch::Dtype dtype, int64_t slot_dim,
      at::intrusive_ptr<recis::embedding::Generator> generator);

  void ReserveBlocksForIds(int64_t ids_num);
  void IncrementBlocksForIds(int64_t additional_ids_num);
  void Reserve(size_t id_size);
  void Delete(const torch::Tensor &ids, const torch::Tensor &index,
              const std::string &preserve_slot = {});
  // Resets the hashtable to its initial factory state: clears all
  // IDs/embeddings and physically releases underlying memory resources.
  void Reset() { ResetInternalState(); }
  // Performs a logical clear of target IDs, typically related to a specific
  // child table: preserves underlying memory capacity for fast reuse.
  void Clear(const std::string &child = {}) {
    auto [child_delete_ids, child_delete_index] = GatherChildIds(child);
    Delete(child_delete_ids, child_delete_index);
  }
  int64_t IdsNum() { return id_map_->GetIdNum(); }
  // ids - index
  std::tuple<torch::Tensor, torch::Tensor> IdsMap(
      const std::string &child = {}) {
    return GatherChildIds(child);
  }
  // ids
  torch::Tensor Ids(const std::string &child = {}) {
    return std::get<0>(GatherChildIds(child));
  }
  // index - embs
  std::tuple<torch::Tensor, torch::Tensor> EmbsMap(
      const std::string &child = {}) {
    auto [_, child_index, child_emb] = GatherChildEmbs(child);
    return std::make_tuple(child_index, child_emb);
  }
  // embs
  torch::Tensor Embs(const std::string &child = {}) {
    auto [_1, _2, child_emb] = GatherChildEmbs(child);
    return child_emb;
  }
  // ids - embs
  std::tuple<torch::Tensor, torch::Tensor> IdsEmbs(
      const std::string &child = {}) {
    auto [child_id, _, child_emb] = GatherChildEmbs(child);
    return std::make_tuple(child_id, child_emb);
  }
  // ids - index - embs
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SnapShot(
      const std::string &child = {}) {
    return GatherChildEmbs(child);
  }

  std::tuple<int64_t, int64_t> IdInfo() {
    int64_t size = id_map_->Size();
    int64_t capacity = id_map_->Capacity();
    return std::make_tuple(size, capacity);
  }

  std::tuple<int64_t, int64_t> AllocatorIdInfo() {
    int64_t active_size = id_map_->GetActiveIdNum();
    int64_t total_size = id_map_->GetIdNum();
    return std::make_tuple(active_size, total_size);
  }

  int64_t IdMemoryInfo() { return id_map_->Capacity() * sizeof(int64_t) * 2; }

  std::tuple<int64_t, int64_t> EmbMemoryInfo() {
    int64_t emb_memory = slot_group_->EmbSlot()->Bytes();
    int64_t total_memory = 0;
    for (auto &slot : slot_group_->Slots()) {
      total_memory += slot->Bytes();
    }
    return std::make_tuple(emb_memory, total_memory);
  }

  static int64_t EncodeIdWithMask(int64_t id, int64_t mask_id);
  static std::pair<int64_t, int64_t> DecodeIdToIndexAndMask(int64_t id);

  // for opt
  bool HasGrad() const { return grad_.size() > 0; };
  void AcceptGrad(const torch::Tensor &grad_index, const torch::Tensor &grad);
  torch::Tensor Grad();
  void ClearGrad();

  const at::intrusive_ptr<recis::embedding::SliceInfo> SliceInfo();
  at::intrusive_ptr<recis::embedding::SlotGroup> SlotGroup();
  at::intrusive_ptr<recis::embedding::ChildrenInfo> ChildrenInfo();
  at::intrusive_ptr<recis::embedding::Slot> GetSlot(const std::string &name);

 private:
  std::mutex blocknum_mutex_;
  void ResetInternalState();
  std::tuple<torch::Tensor, torch::Tensor> EmbeddingLookupReadOnly(
      const torch::Tensor &ids);
  torch::Tensor LookupIndexReadOnly(const torch::Tensor &ids);
  std::tuple<torch::Tensor, torch::Tensor> GatherChildIds(
      const std::string &child);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GatherChildEmbs(
      const std::string &child);
  std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
  GatherChild(const std::string &child,
              const std::vector<std::string> &slot_names);

  // for id generate
  at::intrusive_ptr<recis::embedding::IdMap> id_map_;
  at::intrusive_ptr<recis::embedding::SliceInfo> slice_info_;
  // for coalesce
  at::intrusive_ptr<recis::embedding::ChildrenInfo> children_info_;
  // for param slot
  at::intrusive_ptr<recis::embedding::SlotGroup> slot_group_;

  // for opt
  std::vector<torch::Tensor> grad_index_;
  std::vector<torch::Tensor> grad_;
  HashtableConfig config_;
};