#include "serialize/write_block.h"

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/util/StringUtil.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/irange.h"
#include "c10/util/string_view.h"
#include "embedding/children_info.h"
#include "embedding/slice_info.h"
#include "embedding/slot_group.h"
#include "nlohmann/json_fwd.hpp"
#include "platform/status.h"
#include "serialize/block_info.h"
#include "serialize/dtype_serialize.h"
#include "serialize/name.h"
#include "serialize/string_sep.h"
#include "torch/types.h"
namespace recis {
namespace serialize {
// implementation interface of WriteBlock

at::intrusive_ptr<WriteBlock> WriteBlock::MakeTensorWriteBlock(
    const std::string &block_name, const at::Tensor &tensor) {
  return TensorWriteBlock::Make(block_name, tensor);
}

std::vector<at::intrusive_ptr<WriteBlock>> WriteBlock::MakeHTWriteBlock(
    HashTablePtr ht) {
  std::vector<at::intrusive_ptr<WriteBlock>> ret;
  std::vector<std::shared_ptr<std::vector<int64_t>>> children_ids(
      ht->ChildrenInfo()->Children().size());
  std::vector<std::shared_ptr<std::vector<int64_t>>> children_indexs(
      ht->ChildrenInfo()->Children().size());
  for (auto i : c10::irange(children_ids.size())) {
    children_ids[i].reset(new std::vector<int64_t>);
    children_indexs[i].reset(new std::vector<int64_t>);
  }
  torch::Tensor ids, index;
  std::tie(ids, index) = [&] {
    auto [a, b] = ht->IdsMap();
    return std::make_tuple(a.cpu(), b.cpu());
  }();
  auto ids_vec = ids.data_ptr<int64_t>();
  auto index_vec = index.data_ptr<int64_t>();
  int64_t child_index = 0;
  for (auto i = 0; i < ids.numel(); ++i) {
    int64_t id = ids_vec[i];
    if (ht->ChildrenInfo()->IsCoalesce()) {
      child_index = (id & embedding::ChildrenInfo::IndexMask()) >>
                    embedding::ChildrenInfo::IdBitsNum();
      id = id & embedding::ChildrenInfo::IdMask();
    }
    children_ids[child_index]->push_back(id);
    children_indexs[child_index]->push_back(index_vec[i]);
  }
  auto slice_info = embedding::SliceInfo::ToString(ht->SliceInfo());
  for (auto i : c10::irange(children_ids.size())) {
    ret.push_back(HTIdWriteBlock::Make(ht->ChildrenInfo()->ChildAt(i),
                                       slice_info, children_ids[i]));
    for (auto j : c10::irange(ht->SlotGroup()->Slots().size())) {
      ret.push_back(HTSlotWriteBlock::Make(
          ht->ChildrenInfo()->ChildAt(i),
          ht->SlotGroup()->GetSlotByIndex(j)->Name(), slice_info,
          children_indexs[i], ht->SlotGroup()->GetSlotByIndex(j)));
    }
  }
  return ret;
}

const std::string &WriteBlock::TensorName() { return tensor_name_; }
const std::string &WriteBlock::SliceInfo() { return slice_info_; }
const bool WriteBlock::IsDense() { return is_dense_; };

void WriteBlock::SetTensorName(const std::string &new_tensor_name) {
  if (new_tensor_name.size() > 0) {
    tensor_name_ = new_tensor_name;
  }
}

torch::intrusive_ptr<WriteBlock> TensorWriteBlock::Make(
    const std::string &tensor_name, const at::Tensor &tensor) {
  auto ret = torch::make_intrusive<TensorWriteBlock>();
  ret->tensor_name_ = tensor_name;
  ret->slice_info_ = " ";
  ret->tensor_ = tensor.detach().cpu();
  ret->is_dense_ = true;
  return ret;
}

void TensorWriteBlock::WriteData(FileOutputBuffer *file) {
  RECIS_THROW_IF_ERROR(file->Append(
      torch::string_view((const char *)tensor_.data_ptr(), tensor_.nbytes())));
}

int64_t TensorWriteBlock::WriteMeta(nlohmann::ordered_json &meta,
                                    int64_t offset) {
  nlohmann::ordered_json ::object_t block_meta;
  block_meta[BlockKeyDtype()] = SerializeDtype(tensor_.scalar_type());
  auto shape = nlohmann::ordered_json ::array();
  for (auto s : tensor_.sizes().vec()) {
    shape.push_back(s);
  }
  block_meta[BlockKeyShape()] = shape;
  block_meta[BlockKeyOffsets()] = {offset, offset + tensor_.nbytes()};
  block_meta["dense"] = is_dense_;
  if (tensor_name_.find(TensorKeyPart()) != std::string::npos) {
    meta[BlockNameEncode(tensor_name_, "", "")] = block_meta;
  } else {
    meta[BlockNameEncode(tensor_name_, slice_info_)] = block_meta;
  }
  return offset + tensor_.nbytes();
}

torch::intrusive_ptr<WriteBlock> HTIdWriteBlock::Make(
    const std::string &shared_name, const std::string &slice_info,
    const std::shared_ptr<std::vector<int64_t>> ids) {
  auto ret = torch::make_intrusive<HTIdWriteBlock>();
  ret->tensor_name_ = HTSlotNameEncode(shared_name, HTIdSlotName());
  ret->slice_info_ = slice_info;
  ret->ids_ = ids;
  ret->is_dense_ = false;
  return ret;
}

void HTIdWriteBlock::WriteData(FileOutputBuffer *file) {
  RECIS_THROW_IF_ERROR(file->Append(torch::string_view(
      (const char *)(ids_->data()), ids_->size() * sizeof(int64_t))));
}

int64_t HTIdWriteBlock::WriteMeta(nlohmann::ordered_json &meta,
                                  int64_t offset) {
  nlohmann::ordered_json ::object_t block_meta;
  block_meta[BlockKeyDtype()] = SerializeDtype(torch::kInt64);
  block_meta[BlockKeyShape()] = {ids_->size()};
  block_meta[BlockKeyOffsets()] = {offset,
                                   offset + ids_->size() * sizeof(int64_t)};
  block_meta["dense"] = is_dense_;
  if (tensor_name_.find(TensorKeyPart()) != std::string::npos) {
    meta[BlockNameEncode(tensor_name_, "", "")] = block_meta;
  } else {
    meta[BlockNameEncode(tensor_name_, slice_info_)] = block_meta;
  }
  return offset + ids_->size() * sizeof(int64_t);
}

torch::intrusive_ptr<WriteBlock> HTSlotWriteBlock::Make(
    const std::string &shared_name, const std::string &slot_name,
    const std::string &slice_info,
    const std::shared_ptr<std::vector<int64_t>> index,
    at::intrusive_ptr<embedding::Slot> slot) {
  auto ret = torch::make_intrusive<HTSlotWriteBlock>();
  ret->tensor_name_ = HTSlotNameEncode(shared_name, slot_name);
  ret->index_ = index;
  ret->slot_ = slot;
  ret->slice_info_ = slice_info;
  ret->write_size_ = 0;
  ret->is_dense_ = false;
  return ret;
}

void HTSlotWriteBlock::WriteData(FileOutputBuffer *file) {
  auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  torch::Tensor index = torch::from_blob(
      index_->data(), {static_cast<int64_t>(index_->size())}, options);
  index = index.to(slot_->TensorOptions().device());
  for (auto beg = 0; beg < index.numel(); beg += BlockWriteSize()) {
    auto end = std::min(beg + BlockWriteSize(), index.numel());
    auto data = slot_->IndexSelect(index, beg, end);
    data = data.cpu();
    RECIS_THROW_IF_ERROR(file->Append(
        torch::string_view((const char *)data.data_ptr(), data.nbytes())));
  }
}

int64_t HTSlotWriteBlock::WriteMeta(nlohmann::ordered_json &meta,
                                    int64_t offset) {
  nlohmann::ordered_json ::object_t block_meta;
  block_meta[BlockKeyDtype()] = SerializeDtype(slot_->Dtype());
  auto shape = slot_->FullShape(index_->size());
  block_meta[BlockKeyShape()] = shape;
  write_size_ =
      std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1),
                      std::multiplies<int64_t>());
  write_size_ *= at::scalarTypeToTypeMeta(slot_->Dtype()).itemsize();
  block_meta[BlockKeyOffsets()] = {offset, offset + write_size_};
  block_meta["dense"] = is_dense_;
  if (tensor_name_.find(TensorKeyPart()) != std::string::npos) {
    meta[BlockNameEncode(tensor_name_, "", "")] = block_meta;
  } else {
    meta[BlockNameEncode(tensor_name_, slice_info_)] = block_meta;
  }
  return offset + write_size_;
}
}  // namespace serialize
}  // namespace recis
