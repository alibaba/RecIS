#pragma once
#include <memory>

#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue_inl.h"
#include "c10/core/thread_pool.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/hashtable.h"
#include "embedding/slot_group.h"
#include "platform/filesystem.h"
#include "serialize/block_info.h"
#include "serialize/count_metric.h"
#include "serialize/table_reader.h"
#include "torch/extension.h"
namespace recis {
namespace serialize {
class ReadBlock : public torch::CustomClassHolder {
 public:
  virtual void Read() = 0;
  virtual ~ReadBlock() = default;
  static SizeCounter size_counter_;
};

class TensorReadBlock : public ReadBlock {
 public:
  TensorReadBlock() = default;
  static at::intrusive_ptr<TensorReadBlock> Make(
      at::Tensor dst_tensor, at::intrusive_ptr<BlockInfo> block_info,
      at::intrusive_ptr<TableReader> reader);
  void Read();

 private:
  at::Tensor tensor_;
  at::intrusive_ptr<BlockInfo> block_info_;
  at::intrusive_ptr<TableReader> table_reader_;
};

class HTSlotReadBlock;
class HTIdReadBlock : public ReadBlock {
 public:
  static at::intrusive_ptr<HTIdReadBlock> Make(
      at::intrusive_ptr<TableReader> reader,
      at::intrusive_ptr<BlockInfo> block_info, HashTablePtr ht);
  static torch::Tensor MarkIdAcceptable(at::Tensor ids, int64_t slice_beg,
                                        int64_t slice_end, int64_t slice_size);
  void Read() override;

  friend HTSlotReadBlock;

  int64_t GetIdShape() const { return block_info_->Shape()[0]; }

  HashTablePtr GetHT() const { return ht_; }

  at::intrusive_ptr<BlockInfo> GetBlockInfo() const { return block_info_; }

  at::intrusive_ptr<TableReader> GetTableReader() const {
    return table_reader_;
  }

  virtual void PrepareIdsForInsert(at::Tensor ids,
                                   at::Tensor accept_indicator) {}

 protected:
  HashTablePtr ht_;
  at::intrusive_ptr<BlockInfo> block_info_;
  at::intrusive_ptr<TableReader> table_reader_;
  at::Tensor index_;
  at::Tensor accept_indicator_;
};

class CoalesceHTSlotReadBlock;
class CoalesceHTIDReadBlock : public HTIdReadBlock {
 public:
  static at::intrusive_ptr<CoalesceHTIDReadBlock> Make(
      HashTablePtr ht, at::intrusive_ptr<BlockInfo> block_info,
      at::intrusive_ptr<TableReader> reader, const std::string &shared_name);
  static void EncodeIds(at::Tensor ids, torch::Tensor accept_indicator,
                        int64_t index);
  void Read() override;
  friend CoalesceHTSlotReadBlock;

  void PrepareIdsForInsert(at::Tensor ids,
                           at::Tensor accept_indicator) override;

 private:
  int64_t child_index_;
};

class HTSlotReadBlock : public ReadBlock {
 public:
  static at::intrusive_ptr<HTSlotReadBlock> Make(
      at::intrusive_ptr<embedding::Slot> slot,
      at::intrusive_ptr<BlockInfo> block_info,
      at::intrusive_ptr<TableReader> reader);
  void Read() override;
  c10::List<at::intrusive_ptr<at::ivalue::Future>> ReadAsync(
      at::ThreadPool *pool);

  int64_t GetFlatBytes() const {
    return slot_->FlatSize() * at::elementSize(slot_->Dtype());
  }

  at::intrusive_ptr<embedding::Slot> Slot();
  virtual void ExtractReadInfo(at::intrusive_ptr<HTIdReadBlock> id_block);

  at::intrusive_ptr<BlockInfo> GetBlockInfo() const { return block_info_; }
  at::intrusive_ptr<TableReader> GetTableReader() const {
    return table_reader_;
  }
  at::intrusive_ptr<embedding::Slot> GetSlot() const { return slot_; }

 protected:
  at::Tensor index_;
  at::intrusive_ptr<embedding::Slot> slot_;
  at::intrusive_ptr<BlockInfo> block_info_;
  at::intrusive_ptr<TableReader> table_reader_;
  at::Tensor accept_indicator_;
};
}  // namespace serialize
}  // namespace recis
