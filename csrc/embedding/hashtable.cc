#include "hashtable.h"

#include <ATen/ParallelOpenMP.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <tuple>
#include <utility>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/SparseTensorImpl.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/arange.h"
#include "ATen/ops/concat.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/eq.h"
#include "ATen/ops/index_select.h"
#include "ATen/ops/ones_like.h"
#include "ATen/record_function.h"
#include "c10/core/Device.h"
#include "c10/core/Layout.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Exception.h"
#include "c10/util/Logging.h"
#include "c10/util/flat_hash_map.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/irange.h"
#include "c10/util/string_utils.h"
#include "c10/util/string_view.h"
#include "embedding/children_info.h"
#include "embedding/initializer.h"
#include "embedding/parallel_util.h"
#include "embedding/slice_info.h"
#include "embedding/slot_group.h"
#include "ops/block_ops.h"
#include "torch/types.h"

// part for coalesce
namespace {
constexpr uint64_t kEncodeLength = 12;
constexpr uint64_t kMaskLength = 64 - kEncodeLength;
constexpr uint64_t kMaskBits = (1LL << kMaskLength) - 1;
}  // namespace

int64_t Hashtable::EncodeIdWithMask(int64_t id, int64_t mask_id) {
  return (id & kMaskBits) | (mask_id << kMaskLength);
}

std::pair<int64_t, int64_t> Hashtable::DecodeIdToIndexAndMask(int64_t id) {
  int64_t decode_id = (id & kMaskBits);
  int64_t index = (static_cast<size_t>(id) >> kMaskLength);
  return std::make_pair(decode_id, index);
}

Hashtable::Hashtable(int64_t block_size,
                     const std::vector<int64_t> &embedding_shape,
                     torch::Dtype dtype, torch::Device device, bool coalesce,
                     const std::vector<std::string> &children,
                     at::intrusive_ptr<recis::embedding::Generator> generator,
                     int64_t slice_begin, int64_t slice_end, int64_t slice_size)
    : config_(block_size, embedding_shape, dtype, device, coalesce, children,
              generator, slice_begin, slice_end, slice_size) {
  ResetInternalState();
}

c10::intrusive_ptr<Hashtable> Hashtable::Make(
    int64_t block_size, const std::vector<int64_t> &embedding_shape,
    torch::Dtype dtype, torch::Device device, bool coalesce,
    const std::vector<std::string> &children,
    at::intrusive_ptr<recis::embedding::Generator> generator,
    int64_t slice_begin, int64_t slice_end, int64_t slice_size) {
  auto ret = c10::make_intrusive<Hashtable>(
      block_size, embedding_shape, dtype, device, coalesce, children, generator,
      slice_begin, slice_end, slice_size);
  return ret;
}

void Hashtable::ResetInternalState() {
  TORCH_CHECK(config_.block_size > 0,
              "Block size must be positive during reset.");
  TORCH_CHECK(config_.embedding_shape.size() > 0,
              "Embedding shape cannot be empty.");

  slot_group_ = torch::make_intrusive<recis::embedding::SlotGroup>(
      config_.block_size, config_.device);

  std::vector<int64_t> partial_shape = config_.embedding_shape;
  partial_shape.insert(partial_shape.begin(), config_.block_size);
  slot_group_->AppendEmbSlot(config_.dtype, partial_shape, config_.generator);

  children_info_ =
      torch::make_intrusive<recis::embedding::ChildrenInfo>(config_.coalesce);
  for (const auto &child : config_.children) {
    children_info_->AddChild(child);
  }
  children_info_->Validate();

  slice_info_ = at::make_intrusive<recis::embedding::SliceInfo>(
      config_.slice_begin, config_.slice_end, config_.slice_size);

  id_map_ = recis::embedding::MakeIdMap(config_.device);

  grad_ = {};
  grad_index_ = {};
}

void Hashtable::AcceptGrad(const torch::Tensor &grad_index,
                           const torch::Tensor &grad) {
  TORCH_CHECK(grad_index.scalar_type() == torch::kLong);
  TORCH_CHECK(grad.scalar_type() == slot_group_->EmbSlot()->Dtype());
  grad_index_.push_back(grad_index);
  grad_.push_back(grad);
}

torch::Tensor Hashtable::Grad() {
  auto index = torch::cat(grad_index_, 0);
  auto grad_outputs = torch::cat(grad_, 0);
  auto output = at::_unique(index, false, true);
  torch::Tensor unique_values = std::get<0>(output);
  torch::Tensor unique_index = std::get<1>(output);
  std::vector<int64_t> final_shape = {};
  final_shape.push_back(0);
  auto emb_shape = grad_outputs.sizes().slice(1);
  final_shape.insert(final_shape.end(), emb_shape.data(),
                     emb_shape.data() + emb_shape.size());
  torch::Tensor final_indices;
  torch::Tensor final_grad;
  if (index.numel() == unique_values.numel()) {
    final_indices = index.view({1, -1});
    final_grad = grad_outputs;
    if (index.numel() > 0) {
      final_shape[0] = index.max().item<int64_t>() + 1;
    } else {
      final_shape[0] = 0;
    }
  } else {
    final_indices = unique_values.view({1, -1});
    final_shape[0] = unique_values.numel();
    final_grad = torch::zeros(
        final_shape,
        torch::dtype(grad_outputs.dtype()).device(grad_outputs.device()));
    final_grad.index_add_(0, unique_index.view({-1}), grad_outputs);
    if (unique_values.numel() > 0) {
      final_shape[0] = unique_values.max().item<int64_t>() + 1;
    } else {
      final_shape[0] = 0;
    }
  }
  auto sparse_grad =
      torch::sparse_coo_tensor(final_indices, final_grad, final_shape);
  sparse_grad = sparse_grad.detach_();
  return sparse_grad;
}

void Hashtable::ClearGrad() {
  grad_index_.clear();
  grad_.clear();
}

const at::intrusive_ptr<recis::embedding::SliceInfo> Hashtable::SliceInfo() {
  return slice_info_;
}

at::intrusive_ptr<recis::embedding::SlotGroup> Hashtable::SlotGroup() {
  return slot_group_;
}

at::intrusive_ptr<recis::embedding::ChildrenInfo> Hashtable::ChildrenInfo() {
  return children_info_;
}

std::tuple<torch::Tensor, torch::Tensor> Hashtable::EmbeddingLookup(
    const torch::Tensor &ids, bool readonly) {
  if (readonly) return EmbeddingLookupReadOnly(ids);
  torch::Tensor ids_t =
      ids.to(slot_group_->EmbSlot()->TensorOptions().device());
  torch::Tensor index = id_map_->Lookup(ids_t);
  IncrementBlocknum(id_map_->GetIdNum());
  auto emb_slot = slot_group_->EmbSlot();
  torch::Tensor out_embedding = recis::functional::block_gather(
      index, (*emb_slot->Values()), slot_group_->BlockSize(), kNullIndex,
      false);
  return std::make_tuple(index, out_embedding);
}

void Hashtable::Insert(const torch::Tensor &ids,
                       const torch::Tensor &embedding) {
  auto emb_slot = slot_group_->EmbSlot();
  TORCH_CHECK(embedding.scalar_type() == emb_slot->Dtype(),
              "embeding type must be [", emb_slot->Dtype(), "] but get ",
              embedding.dtype());
  TORCH_CHECK(ids.size(0) == embedding.size(0),
              "size of ids and embedding dismatch, ids: ", ids.size(0),
              " embedding: ", embedding.size(0));
  torch::Tensor ids_t = ids.to(emb_slot->TensorOptions().device());
  torch::Tensor embedding_t = embedding.to(emb_slot->TensorOptions().device());
  auto index = InsertLookupIndex(ids_t);
  auto &embedding_blocks = (*slot_group_->EmbSlot()->Values());
  auto block_size = slot_group_->BlockSize();
  recis::functional::block_insert(index, embedding_t, embedding_blocks,
                                  block_size);
}

void Hashtable::UpdateSlot(const std::string &slot_name,
                           const torch::Tensor &index,
                           torch::Tensor embedding) {
  auto slot = slot_group_->GetSlotByName(slot_name);
  TORCH_CHECK(embedding.scalar_type() == slot->Dtype(),
              "embedding type must be [", slot->Dtype(), "] but get ",
              embedding.scalar_type());
  torch::Tensor index_t = index.to(slot->TensorOptions().device());
  torch::Tensor embedding_t = embedding.to(slot->TensorOptions().device());

  slot->IndexInsert(index_t, embedding_t);
}

void Hashtable::Reserve(size_t id_size) {
  id_size = id_size / slice_info_->partition_num();
  id_map_->Reserve(id_size);
}

std::tuple<torch::Tensor, torch::Tensor> Hashtable::EmbeddingLookupReadOnly(
    const torch::Tensor &ids) {
  torch::Tensor ids_t =
      ids.to(slot_group_->EmbSlot()->TensorOptions().device());
  torch::Tensor index = id_map_->LookupReadOnly(ids_t);
  IncrementBlocknum(id_map_->GetIdNum());
  auto emb_slot = slot_group_->EmbSlot();
  torch::Tensor out_embedding = recis::functional::block_gather(
      index, (*emb_slot->Values()), slot_group_->BlockSize(), kNullIndex, true);
  return std::make_tuple(index, out_embedding);
}

torch::Tensor Hashtable::InsertLookupIndex(const torch::Tensor &ids) {
  torch::Tensor ids_t =
      ids.to(slot_group_->EmbSlot()->TensorOptions().device());
  auto index = id_map_->InsertIds(ids_t);
  IncrementBlocknum(id_map_->GetIdNum());
  return index;
}

torch::Tensor Hashtable::InsertLookupIndexWithIndicator(
    const torch::Tensor &ids, const torch::Tensor &indicator) {
  {
    // remove duplicate ids & keep first
    auto [unique_ids, reverse_index] = torch::_unique(ids, true, true);
    if (unique_ids.numel() < ids.numel()) {
      auto copy_indicator = indicator.clone();
      indicator.fill_(false);
      auto perm = torch::arange(reverse_index.size(0), ids.options());
      auto first_indices = torch::empty_like(unique_ids);
      first_indices.scatter_reduce_(0, reverse_index, perm, "amin", false);
      indicator.index_fill_(0, first_indices, true);
      indicator.bitwise_and_(copy_indicator);
    }
  }
  torch::Tensor ids_t =
      ids.to(slot_group_->EmbSlot()->TensorOptions().device());
  torch::Tensor indicator_t =
      indicator.to(slot_group_->EmbSlot()->TensorOptions().device());
  auto insert_ids = torch::masked_select(ids_t, indicator_t);
  auto insert_index = InsertLookupIndex(insert_ids);
  auto index = torch::zeros_like(ids_t, insert_index.options());
  index.masked_scatter_(indicator_t, insert_index);
  return index;
}

torch::Tensor Hashtable::LookupIndexReadOnly(const torch::Tensor &ids) {
  torch::Tensor index = id_map_->LookupReadOnly(ids);
  return index;
}

void Hashtable::IncrementBlocknum(int64_t ids_num) {
  std::lock_guard<std::mutex> lock(blocknum_mutex_);
  size_t block_num =
      (ids_num + slot_group_->BlockSize()) / slot_group_->BlockSize();
  while (slot_group_->BlockNum() < block_num) {
    slot_group_->IncrementBlock();
  }
}

void Hashtable::Delete(const torch::Tensor &ids, const torch::Tensor &index,
                       const std::string &preserve_slot) {
  if (ids.numel() == 0) {
    return;
  }
  torch::Tensor ids_t =
      ids.to(slot_group_->EmbSlot()->TensorOptions().device());
  torch::Tensor index_t =
      index.to(slot_group_->EmbSlot()->TensorOptions().device());
  id_map_->DeleteIds(ids_t, index_t);
  for (int64_t i = 0; i < slot_group_->GroupSize(); ++i) {
    if (!preserve_slot.empty() && slot_group_->GetSlotIndex(preserve_slot) == i)
      continue;
    slot_group_->ResetSlotByIndex(i, ids_t.numel(), index_t);
  }
}

std::tuple<torch::Tensor, torch::Tensor> Hashtable::GatherChildIds(
    const std::string &child) {
  if (child.empty()) {
    auto [ids, index] = id_map_->SnapShot();
    return std::make_tuple(ids, index);
  }
  TORCH_CHECK(children_info_->HasChild(child), "hashtable ", child, "Child '",
              child, "' not found in hashtable")
  int64_t child_index = children_info_->ChildIndex(child);
  auto [coalesced_ids, coalseced_index] = id_map_->SnapShot();
  auto decode_index = at::bitwise_right_shift(coalesced_ids, kMaskLength);
  auto mask = at::eq(decode_index, child_index);
  auto gather_ids = at::masked_select(coalesced_ids, mask);
  auto gather_index = at::masked_select(coalseced_index, mask);
  return std::make_tuple(gather_ids, gather_index);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Hashtable::GatherChildEmbs(const std::string &child) {
  auto [child_ids, child_index, child_slots] = GatherChild(
      child, {std::string(recis::embedding::SlotGroup::EmbSlotName())});
  return std::make_tuple(child_ids, child_index, child_slots[0]);
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
Hashtable::GatherChild(const std::string &child,
                       const std::vector<std::string> &slot_names) {
  auto [child_ids, child_index] = GatherChildIds(child);
  IncrementBlocknum(id_map_->GetIdNum());
  std::vector<torch::Tensor> child_slots;
  child_slots.reserve(slot_names.size());
  for (const auto &slot_name : slot_names) {
    auto slot = slot_group_->GetSlotByName(slot_name);
    torch::Tensor child_slot = recis::functional::block_gather(
        child_index, *(slot->Values()), slot_group_->BlockSize(), kNullIndex,
        true);
    child_slots.push_back(std::move(child_slot));
  }
  return {child_ids, child_index, child_slots};
}

void Hashtable::AppendFilterSlot(
    const std::string &filter_name, torch::Dtype dtype, int64_t slot_dim,
    at::intrusive_ptr<recis::embedding::Generator> generator) {
  std::vector<int64_t> slot_shape{0, slot_dim};
  slot_group_->AppendSlot(filter_name, dtype, slot_shape, generator);
  return;
}

void Hashtable::AppendStepFilterSlot(const std::string &filter_name,
                                     int64_t init_step) {
  slot_group_->AppendStepFilterSlot(filter_name, init_step);
}

at::intrusive_ptr<recis::embedding::Slot> Hashtable::GetSlot(
    const std::string &name) {
  return slot_group_->GetSlotByName(name);
}
