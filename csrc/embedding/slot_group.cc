#include "embedding/slot_group.h"

#include <cstring>
#include <mutex>
#include <numeric>
#include <string>
#include <type_traits>

#include "ATen/Dispatch.h"
#include "ATen/SparseTensorImpl.h"
#include "ATen/core/jit_type.h"
#include "ATen/ops/cat.h"
#include "ATen/ops/empty.h"
#include "c10/core/Allocator.h"
#include "c10/core/ScalarType.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/irange.h"
#include "embedding/hashtable.h"
#include "embedding/initializer.h"
#include "ops/block_ops.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
namespace recis {
namespace embedding {

Slot::Slot(const std::string &name, int64_t block_size, torch::Dtype dtype,
           const std::vector<int64_t> &partial_shape,
           at::intrusive_ptr<recis ::embedding::Generator> generator)
    : name_(name),
      dtype_(dtype),
      partial_shape_(partial_shape),
      generator_(generator),
      block_size_(block_size) {
  values_ = std::make_shared<std::vector<torch::Tensor>>();
  flat_size_ = std::accumulate(partial_shape.begin() + 1, partial_shape.end(),
                               1, std::multiplies<int64_t>());
}

void Slot::IncrementBlock() {
  std::lock_guard<std::mutex> l(mutex_);
  values_->push_back(generator_->Generate());
}

void Slot::Clear() { values_->clear(); }

int64_t Slot::BlockSize() const { return block_size_; }

int64_t Slot::Bytes() const {
  if (!values_ || values_->empty()) {
    return 0;
  }
  int64_t total_memory = 0;
  for (const auto &tensor : *values_) {
    total_memory += tensor.numel() * tensor.element_size();
  }
  return total_memory;
}

const std::string &Slot::Name() { return name_; }

std::shared_ptr<std::vector<torch::Tensor>> Slot::Values() { return values_; }

std::vector<torch::Tensor> Slot::ValuesDeref() {
  if (values_) {
    return *values_;
  }
  return {};
}

torch::Dtype Slot::Dtype() { return dtype_; }

torch::TensorOptions Slot::TensorOptions() {
  auto device = generator_->TensorOption().device();
  return torch::TensorOptions().dtype(dtype_).device(device);
}

int64_t Slot::FlatSize() const { return flat_size_; }

std::vector<int64_t> Slot::FullShape(int64_t block_size) const {
  std::vector<int64_t> full_shape(partial_shape_);
  full_shape[0] = block_size;
  return full_shape;
}

at::intrusive_ptr<recis::embedding::Generator> Slot::Generator() {
  return generator_;
}

torch::Tensor Slot::Value() {
  if (values_->size() != 0) {
    return at::cat(*values_, 0);
  }
  return at::empty(FullShape(0), generator_->TensorOption());
}

void Slot::Generator(at::intrusive_ptr<recis::embedding::Generator> generator) {
  generator_ = generator;
}

torch::Tensor Slot::IndexSelect(torch::Tensor index, const size_t beg,
                                const size_t end) {
  torch::Tensor ret = recis::functional::block_gather_by_range(
      index, *values_, block_size_, beg, end);
  return ret;
}

void Slot::IndexInsert(torch::Tensor index, torch::Tensor value) {
  TORCH_CHECK(
      value.dim() == 0 || index.numel() == value.size(0) || value.size(0) == 1,
      "index must have the same size with value or")
  recis::functional::block_insert(index, value, *values_, block_size_);
}

void Slot::IndexInsert(torch::Tensor index, torch::Tensor value,
                       torch::Tensor accept_indicator) {
  TORCH_CHECK(accept_indicator.scalar_type() == torch::kBool,
              "accept_indicator must be bool");
  TORCH_CHECK(accept_indicator.numel() == index.numel(),
              "accept_indicator must have the same size with index")
  TORCH_CHECK(accept_indicator.numel() == value.size(0),
              "accept_indicator must have the same size with value")
  std::remove_reference_t<decltype(*values_)> values;
  {
    std::lock_guard<std::mutex> l(mutex_);
    values = *values_;
  }
  recis::functional::block_insert_with_mask(index, value, accept_indicator,
                                            values, block_size_);
}

torch::Tensor Slot::GenVal(int64_t block_size) {
  TORCH_CHECK(block_size >= 0, "block_size must >= 0");
  torch::Tensor ret = generator_->Generate(FullShape(block_size));
  return ret;
}

SlotGroup::SlotGroup(int64_t block_size, torch::Device device)
    : block_size_(block_size),
      block_num_(0),
      emb_slot_index_(-1),
      device_(device) {}

at::intrusive_ptr<Slot> SlotGroup::AppendSlot(
    const std::string &name, torch::Dtype dtype,
    const std::vector<int64_t> &partial_shape,
    at::intrusive_ptr<recis ::embedding::Generator> generator) {
  // lock?
  generator->set_device(device_);
  auto slot = torch::make_intrusive<Slot>(name, block_size_, dtype,
                                          partial_shape, generator);
  slots_.push_back(slot);
  name_to_index_[name] = slots_.size() - 1;
  return slots_.back();
}

at::intrusive_ptr<Slot> SlotGroup::AppendSlot(
    const std::string &name, torch::Dtype dtype,
    const std::vector<int64_t> &partial_shape, double init_val) {
  std::vector<int64_t> full_shape(partial_shape);
  full_shape[0] = block_size_;
  auto gen = MakeConstantGenerator(full_shape, dtype, init_val);
  return AppendSlot(name, dtype, partial_shape, gen);
}

void SlotGroup::AppendEmbSlot(
    torch::Dtype dtype, const std::vector<int64_t> &partial_shape,
    at::intrusive_ptr<recis ::embedding::Generator> generator) {
  AppendSlot(EmbSlotName(), dtype, partial_shape, generator);
  emb_slot_index_ = slots_.size() - 1;
}

void SlotGroup::ResetSlotByIndex(int64_t slot_index, int64_t block_size,
                                 const torch::Tensor &index) {
  TORCH_CHECK(index.device().type() == device_.type(),
              "Index input must be on the same device of slot group");
  auto slot = GetSlotByIndex(slot_index);
  slot->IndexInsert(index, slot->GenVal(block_size));
  return;
}

void SlotGroup::AppendStepFilterSlot(const std::string &name,
                                     int64_t init_step) {
  std::vector<int64_t> full_shape{block_size_, 1};
  AppendSlot(name, torch::kInt64, full_shape,
             MakeConstantGenerator(full_shape, torch::kInt64, init_step));
}

void SlotGroup::IncrementBlock() {
  for (auto &slot : slots_) {
    slot->IncrementBlock();
  }
  block_num_++;
}

void SlotGroup::Clear() {
  for (auto &slot : slots_) {
    slot->Clear();
  }
}

torch::intrusive_ptr<Slot> SlotGroup::GetSlotByName(const std::string &name) {
  return slots_.at(name_to_index_.at(name));
}

torch::intrusive_ptr<Slot> SlotGroup::GetSlotByIndex(int64_t index) {
  return slots_.at(index);
}

int64_t SlotGroup::GetSlotIndex(const std::string &name) {
  return name_to_index_.at(name);
}

torch::intrusive_ptr<Slot> SlotGroup::EmbSlot() {
  return slots_.at(emb_slot_index_);
}

std::vector<torch::intrusive_ptr<Slot>> SlotGroup::Slots() { return slots_; }
int64_t SlotGroup::BlockSize() const { return block_size_; }
int64_t SlotGroup::BlockNum() const { return block_num_; }
int64_t SlotGroup::GroupSize() const {
  return static_cast<int64_t>(slots_.size());
}

}  // namespace embedding
}  // namespace recis
