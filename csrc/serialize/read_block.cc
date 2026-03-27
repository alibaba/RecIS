#include "serialize/read_block.h"

#include <algorithm>
#include <exception>
#include <functional>

#include "ATen/TensorUtils.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue_inl.h"
#include "ATen/core/jit_type.h"
#include "ATen/cuda/CUDAContext.h"
#include "c10/core/DeviceGuard.h"
#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/irange.h"
#include "c10/util/string_view.h"
#include "embedding/children_info.h"
#include "embedding/slot_group.h"
#include "platform/status.h"
#include "serialize/block_info.h"
#include "serialize/name.h"
#include "serialize/table_reader.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/types.h"

namespace recis {
namespace serialize {

SizeCounter ReadBlock::size_counter_;

at::intrusive_ptr<TensorReadBlock> TensorReadBlock::Make(
    at::Tensor dst_tensor, at::intrusive_ptr<BlockInfo> block_info,
    at::intrusive_ptr<TableReader> table_reader) {
  auto ret = at::make_intrusive<TensorReadBlock>();
  ret->tensor_ = dst_tensor;
  ret->block_info_ = block_info;
  ret->table_reader_ = table_reader;
  return ret;
}

void TensorReadBlock::Read() {
  size_counter_.AddSize(block_info_->Size());
  // to do: tensor name and filename
  TORCH_CHECK(block_info_->Dtype() == tensor_.dtype(), "dtype not match");
  TORCH_CHECK(block_info_->Shape() == tensor_.sizes().vec(), "shape not match");
  torch::string_view ret;
  auto file = table_reader_->File();
  if (tensor_.device().type() == torch::kCPU) {
    RECIS_STATUS_COND(file->Read(block_info_->OffsetBeg(), block_info_->Size(),
                                 &ret, (char *)tensor_.data_ptr()));
  } else {
    torch::Tensor cpu_t = torch::empty_like(tensor_, torch::kCPU);
    RECIS_STATUS_COND(file->Read(block_info_->OffsetBeg(), block_info_->Size(),
                                 &ret, (char *)cpu_t.data_ptr()));
    tensor_.copy_(cpu_t);
  }
}

at::intrusive_ptr<HTIdReadBlock> HTIdReadBlock::Make(
    at::intrusive_ptr<TableReader> reader,
    at::intrusive_ptr<BlockInfo> block_info, HashTablePtr ht) {
  auto ret = at::make_intrusive<HTIdReadBlock>();
  ret->table_reader_ = reader;
  ret->block_info_ = block_info;
  ret->ht_ = ht;
  return ret;
}

torch::Tensor HTIdReadBlock::MarkIdAcceptable(at::Tensor ids, int64_t slice_beg,
                                              int64_t slice_end,
                                              int64_t slice_size) {
  at::Tensor ret = torch::empty(ids.sizes(), ids.options().dtype(torch::kBool));
  auto ret_ptr = ret.data_ptr<bool>();
  AT_DISPATCH_ALL_TYPES(ids.scalar_type(), "MarkIdAcceptable", [&]() {
    auto ids_ptr = ids.data_ptr<scalar_t>();
    for (auto index : c10::irange(ids.numel())) {
      int64_t mod = (uint64_t)ids_ptr[index] % (uint64_t)slice_size;
      ret_ptr[index] = (mod >= slice_beg && mod < slice_end);
    }
  });
  return ret;
}

void HTIdReadBlock::Read() {
  size_counter_.AddSize(block_info_->Size());
  int64_t ids_num = block_info_->Size() / sizeof(int64_t);
  torch::Tensor ids = torch::empty(
      {ids_num}, at::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  torch::string_view ret;
  auto file = table_reader_->File();
  RECIS_STATUS_COND(file->Read(block_info_->OffsetBeg(), block_info_->Size(),
                               &ret, (char *)ids.data_ptr()));
  accept_indicator_ = MarkIdAcceptable(ids, ht_->SliceInfo()->slice_begin(),
                                       ht_->SliceInfo()->slice_end(),
                                       ht_->SliceInfo()->slice_size());
  index_ = ht_->InsertLookupIndexWithIndicator(ids, accept_indicator_);
}

at::intrusive_ptr<CoalesceHTIDReadBlock> CoalesceHTIDReadBlock::Make(
    HashTablePtr ht, at::intrusive_ptr<BlockInfo> block_info,
    at::intrusive_ptr<TableReader> reader, const std::string &shared_name) {
  auto ret = at::make_intrusive<CoalesceHTIDReadBlock>();
  ret->table_reader_ = reader;
  ret->block_info_ = block_info;
  ret->ht_ = ht;
  ret->child_index_ = ht->ChildrenInfo()->ChildIndex(shared_name);
  return ret;
}

void CoalesceHTIDReadBlock::EncodeIds(at::Tensor ids,
                                      torch::Tensor accept_indicator,
                                      int64_t child_index) {
  auto accept_indicator_ptr = accept_indicator.data_ptr<bool>();
  AT_DISPATCH_ALL_TYPES(ids.scalar_type(), "EncodeIds", [&]() {
    auto ids_ptr = ids.data_ptr<scalar_t>();
    for (auto index : c10::irange(ids.numel())) {
      if (accept_indicator_ptr[index]) {
        ids_ptr[index] = recis::embedding::ChildrenInfo::EncodeId(
            ids_ptr[index], child_index);
      }
    }
  });
}

void CoalesceHTIDReadBlock::PrepareIdsForInsert(at::Tensor ids,
                                                at::Tensor accept_indicator) {
  EncodeIds(ids, accept_indicator, child_index_);
}

void CoalesceHTIDReadBlock::Read() {
  size_counter_.AddSize(block_info_->Size());
  int64_t ids_num = block_info_->Size() / sizeof(int64_t);
  torch::Tensor ids = torch::empty(
      {ids_num}, at::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  torch::string_view ret;
  auto file = table_reader_->File();
  RECIS_STATUS_COND(file->Read(block_info_->OffsetBeg(), block_info_->Size(),
                               &ret, (char *)ids.data_ptr()));
  accept_indicator_ = HTIdReadBlock::MarkIdAcceptable(
      ids, ht_->SliceInfo()->slice_begin(), ht_->SliceInfo()->slice_end(),
      ht_->SliceInfo()->slice_size());
  EncodeIds(ids, accept_indicator_, child_index_);
  index_ = ht_->InsertLookupIndexWithIndicator(ids, accept_indicator_);
}

at::intrusive_ptr<HTSlotReadBlock> HTSlotReadBlock::Make(
    at::intrusive_ptr<embedding::Slot> slot,
    at::intrusive_ptr<BlockInfo> block_info,
    at::intrusive_ptr<TableReader> table_reader) {
  auto ret = at::make_intrusive<HTSlotReadBlock>();
  ret->slot_ = slot;
  ret->block_info_ = block_info;
  ret->table_reader_ = table_reader;
  return ret;
}

void HTSlotReadBlock::Read() {
  TORCH_CHECK(slot_->FullShape(block_info_->Shape()[0]) == block_info_->Shape(),
              "shape not match");
  TORCH_CHECK(slot_->Dtype() == block_info_->Dtype(), "dtype not match");
  TORCH_CHECK(index_.defined(), "index not defined");
  TORCH_CHECK(accept_indicator_.defined(),
              "accept_indicator not defined:", (&accept_indicator_));
  auto flat_nbytes = slot_->FlatSize() * at::elementSize(slot_->Dtype());
  auto read_batch_size = BlockReadSize();
  int64_t read_ids_num = 0;
  int64_t ids_need_to_read = block_info_->Shape()[0];
  auto shape = block_info_->Shape();
  shape[0] = std::min<int64_t>(shape[0], read_batch_size);
  at::Tensor slot_tensor = torch::empty(
      shape,
      at::TensorOptions().device(torch::kCPU).dtype(block_info_->Dtype()));
  while (read_ids_num < ids_need_to_read) {
    int64_t read_size =
        std::min(read_batch_size, ids_need_to_read - read_ids_num);
    size_counter_.AddSize(read_size * flat_nbytes);
    torch::string_view ret;
    auto file = table_reader_->File();
    RECIS_STATUS_COND(file->Read(
        block_info_->OffsetBeg() + read_ids_num * flat_nbytes,
        read_size * flat_nbytes, &ret, (char *)slot_tensor.data_ptr()));
    slot_->IndexInsert(
        recis::embedding::make_lvalue(index_.narrow(0, read_ids_num, read_size))
            .to(slot_->TensorOptions().device()),
        recis::embedding::make_lvalue(slot_tensor.narrow(0, 0, read_size)
                                          .to(slot_->TensorOptions().device())),
        recis::embedding::make_lvalue(
            accept_indicator_.narrow(0, read_ids_num, read_size)
                .to(slot_->TensorOptions().device())));
    read_ids_num += read_size;
  }
}

c10::List<at::intrusive_ptr<at::ivalue::Future>> HTSlotReadBlock::ReadAsync(
    at::ThreadPool *pool) {
  TORCH_CHECK(slot_->FullShape(block_info_->Shape()[0]) == block_info_->Shape(),
              "shape not match");
  TORCH_CHECK(slot_->Dtype() == block_info_->Dtype(), "dtype not match");
  TORCH_CHECK(index_.defined(), "index not defined");
  TORCH_CHECK(accept_indicator_.defined(),
              "accept_indicator not defined:", (&accept_indicator_));
  auto flat_nbytes = slot_->FlatSize() * at::elementSize(slot_->Dtype());
  auto read_batch_size = BlockReadSize();
  int64_t read_ids_num = 0;
  int64_t ids_need_to_read = block_info_->Shape()[0];
  auto shape = block_info_->Shape();
  c10::List<at::intrusive_ptr<at::ivalue::Future>> futures(
      at::FutureType::create(at::NoneType::get()));
  c10::Device device = torch::cuda::is_available()
                           ? c10::Device(c10::kCUDA, at::cuda::current_device())
                           : c10::Device(c10::kCPU);
  while (read_ids_num < ids_need_to_read) {
    int64_t read_size =
        std::min(read_batch_size, ids_need_to_read - read_ids_num);
    auto future = at::make_intrusive<at::ivalue::Future>(at::NoneType::get());
    auto func = [this, read_size, read_ids_num, flat_nbytes, shape, device,
                 future]() mutable {
      try {
        c10::DeviceGuard device_guard(device);
        shape[0] = read_size;
        at::Tensor slot_tensor =
            torch::empty(shape, at::TensorOptions()
                                    .device(torch::kCPU)
                                    .dtype(block_info_->Dtype()));
        torch::string_view ret;
        auto file = table_reader_->File();
        size_counter_.AddSize(read_size * flat_nbytes);
        RECIS_STATUS_COND(file->Read(
            block_info_->OffsetBeg() + read_ids_num * flat_nbytes,
            read_size * flat_nbytes, &ret, (char *)slot_tensor.data_ptr()));

        slot_->IndexInsert(
            recis::embedding::make_lvalue(
                index_.narrow(0, read_ids_num, read_size))
                .to(slot_->TensorOptions().device()),
            recis::embedding::make_lvalue(
                slot_tensor.narrow(0, 0, read_size)
                    .to(slot_->TensorOptions().device())),
            recis::embedding::make_lvalue(
                accept_indicator_.narrow(0, read_ids_num, read_size)
                    .to(slot_->TensorOptions().device())));

        future->markCompleted();
      } catch (std::exception &e) {
        LOG(ERROR) << "exception:" << e.what();
        future->setError(std::current_exception());
      } catch (...) {
        LOG(ERROR) << "unknown exception";
      }
    };
    pool->run(func);
    futures.push_back(future);
    read_ids_num += read_size;
  }
  return futures;
}

at::intrusive_ptr<embedding::Slot> HTSlotReadBlock::Slot() { return slot_; }

void HTSlotReadBlock::ExtractReadInfo(
    at::intrusive_ptr<HTIdReadBlock> id_block) {
  index_ = id_block->index_;
  accept_indicator_ = id_block->accept_indicator_;
}

}  // namespace serialize
}  // namespace recis
