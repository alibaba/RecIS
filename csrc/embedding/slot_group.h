#pragma once
#include <memory>
#include <mutex>

#include "ATen/Utils.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/flat_hash_map.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/initializer.h"
#include "torch/arg.h"
#include "torch/extension.h"
#include "torch/types.h"
namespace recis {
namespace embedding {

template <typename T>
T &make_lvalue(T &&t) {
  return t;
}

struct Slot : public torch::CustomClassHolder {
  Slot(const std::string &name, int64_t block_size, torch::Dtype dtype,
       const std::vector<int64_t> &partial_shape,
       at::intrusive_ptr<recis ::embedding::Generator> generator);
  void IncrementBlock();
  void Clear();
  int64_t BlockSize() const;
  int64_t Bytes() const;
  const std::string &Name();
  std::shared_ptr<std::vector<torch::Tensor>> Values();
  std::vector<torch::Tensor> ValuesDeref();
  torch::Dtype Dtype();
  torch::TensorOptions TensorOptions();
  int64_t FlatSize() const;
  std::vector<int64_t> FullShape(int64_t block_size) const;
  at::intrusive_ptr<recis::embedding::Generator> Generator();
  void Generator(at::intrusive_ptr<recis::embedding::Generator> generator);
  torch::Tensor Value();
  torch::Tensor IndexSelect(torch::Tensor index, const size_t beg,
                            const size_t end);
  torch::Tensor GenVal(int64_t block_size);
  void IndexInsert(torch::Tensor index, torch::Tensor value);
  void IndexInsert(torch::Tensor index, torch::Tensor value,
                   torch::Tensor accept_indicator);

 private:
  std::mutex mutex_;  // a lock used to protect the values_ while loading.
  std::string name_;
  std::shared_ptr<std::vector<torch::Tensor>> values_;
  torch::Dtype dtype_;
  std::vector<int64_t> partial_shape_;
  at::intrusive_ptr<recis::embedding::Generator> generator_;
  int64_t flat_size_;
  int64_t block_size_;
};

struct SlotGroup : public torch::CustomClassHolder {
  constexpr static const char *EmbSlotName() { return "embedding"; }

  SlotGroup(int64_t block_size, torch::Device device);

  at::intrusive_ptr<Slot> AppendSlot(
      const std::string &name, torch::Dtype dtype,
      const std::vector<int64_t> &partial_shape,
      at::intrusive_ptr<recis ::embedding::Generator> generator);
  at::intrusive_ptr<Slot> AppendSlot(const std::string &name,
                                     torch::Dtype dtype,
                                     const std::vector<int64_t> &partial_shape,
                                     double init_val = 0.0);
  void AppendEmbSlot(torch::Dtype dtype,
                     const std::vector<int64_t> &partial_shape,
                     at::intrusive_ptr<recis ::embedding::Generator> generator);
  void AppendStepFilterSlot(const std::string &name, int64_t init_step);
  void IncrementBlock();
  void Clear();
  void ResetSlotByIndex(int64_t slot_index, int64_t block_size,
                        const torch::Tensor &index);

  torch::intrusive_ptr<Slot> GetSlotByName(const std::string &name);
  torch::intrusive_ptr<Slot> GetSlotByIndex(int64_t index);
  int64_t GetSlotIndex(const std::string &name);
  torch::intrusive_ptr<Slot> EmbSlot();
  std::vector<torch::intrusive_ptr<Slot>> Slots();

  int64_t BlockSize() const;
  int64_t BlockNum() const;
  int64_t GroupSize() const;

 private:
  int64_t block_size_;
  int64_t block_num_;
  int64_t emb_slot_index_;
  torch::Device device_;
  ska::flat_hash_map<std::string, int64_t> name_to_index_;
  std::vector<torch::intrusive_ptr<Slot>> slots_;
};

}  // namespace embedding
}  // namespace recis
