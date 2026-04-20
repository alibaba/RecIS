#include "serialize/ht_read_collection.h"

#include <string>

#include "ATen/core/List.h"
#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/hashtable.h"
#include "embedding/slot_group.h"
#include "serialize/name.h"
#include "serialize/read_block.h"
#include "serialize/table_reader.h"
namespace recis {
namespace serialize {
namespace {
// split into chunks, each chunk is processed concurrently
struct ChunkInfo {
  int64_t slice_id;
  int64_t beg_ids;    // beginning id index (sample number)
  int64_t num_ids;    // number of ids (sample number)
  int64_t beg_bytes;  // beginning byte offset
  int64_t num_bytes;  // number of bytes
  ChunkInfo(int64_t slice_id, int64_t beg_ids, int64_t num_ids)
      : slice_id(slice_id),
        beg_ids(beg_ids),
        num_ids(num_ids),
        beg_bytes(beg_ids * sizeof(int64_t)),
        num_bytes(num_ids * sizeof(int64_t)) {}
};
}  // namespace

at::intrusive_ptr<HTReadCollection> HTReadCollection::Make(
    const std::string &shared_name) {
  return at::make_intrusive<HTReadCollection>(shared_name);
}

HTReadCollection::HTReadCollection(const std::string &shared_name)
    : id_done_(false), share_name_(shared_name) {}

void HTReadCollection::Append(HashTablePtr target_ht,
                              const std::string &slot_name,
                              at::intrusive_ptr<TableReader> reader,
                              at::intrusive_ptr<BlockInfo> block_info) {
  if (IsIdName(slot_name)) {
    if (target_ht->ChildrenInfo()->IsCoalesce()) {
      id_reader_ = CoalesceHTIDReadBlock::Make(target_ht, block_info, reader,
                                               share_name_);

    } else {
      id_reader_ = HTIdReadBlock::Make(reader, block_info, target_ht);
    }
  } else {
    block_reader_.push_back(HTSlotReadBlock::Make(
        target_ht->SlotGroup()->GetSlotByName(slot_name), block_info, reader));
  }
}

std::vector<at::intrusive_ptr<embedding::Slot>> HTReadCollection::ReadSlots() {
  std::vector<at::intrusive_ptr<embedding::Slot>> ret;
  for (auto slot_reader : block_reader_) {
    ret.push_back(slot_reader->Slot());
  }
  return ret;
}

bool HTReadCollection::Valid() {
  return !block_reader_.empty() && id_reader_ != nullptr;
}

bool HTReadCollection::Empty() {
  return block_reader_.empty() && id_reader_ == nullptr;
}

c10::List<at::intrusive_ptr<at::ivalue::Future>>
HTReadCollection::LoadChunksAsync(at::PTThreadPool *pool, int64_t chunk_size) {
  c10::List<at::intrusive_ptr<at::ivalue::Future>> ret(
      at::FutureType::create(at::NoneType::get()));

  // get total ids
  int64_t total_ids = id_reader_ != nullptr ? id_reader_->GetIdShape() : 0;

  if (block_reader_.empty()) {
    return ret;
  }

  auto target_ht = id_reader_->GetHT();
  target_ht->ReserveBlocksForIds(1);

  const int64_t id_bytes = sizeof(int64_t);

  int64_t ids_per_chunk = chunk_size / id_bytes;
  if (ids_per_chunk == 0) {
    ids_per_chunk = 1;  // at least contains 1 id
  }

  std::vector<ChunkInfo> out_slices;
  int64_t slice_id = 0;
  for (int64_t j = 0; j < total_ids; j += ids_per_chunk) {
    int64_t num_ids = std::min(total_ids - j, ids_per_chunk);
    out_slices.emplace_back(slice_id++, j, num_ids);
  }

  for (const auto &chunk : out_slices) {
    auto future = at::make_intrusive<at::ivalue::Future>(at::NoneType::get());

    pool->run([this, chunk, target_ht, future]() mutable {
      try {
        if (chunk.num_ids == 0) {
          future->markCompleted();
          return;
        }

        auto id_block_info = id_reader_->GetBlockInfo();
        int64_t id_block_beg = id_block_info->OffsetBeg();

        torch::Tensor chunk_ids = torch::empty(
            {chunk.num_ids},
            at::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

        auto id_file = id_reader_->GetTableReader()->File();
        torch::string_view ret;

        int64_t id_offset = id_block_beg + chunk.beg_bytes;
        int64_t id_read_size = chunk.num_bytes;

        RECIS_STATUS_COND(id_file->Read(id_offset, id_read_size, &ret,
                                        (char *)chunk_ids.data_ptr()));

        auto chunk_accept = HTIdReadBlock::MarkIdAcceptable(
            chunk_ids, target_ht->SliceInfo()->slice_begin(),
            target_ht->SliceInfo()->slice_end(),
            target_ht->SliceInfo()->slice_size());

        id_reader_->PrepareIdsForInsert(chunk_ids, chunk_accept);

        auto chunk_index =
            target_ht->InsertLookupIndexWithIndicator(chunk_ids, chunk_accept);

        for (auto slot_reader : block_reader_) {
          auto slot_block_info = slot_reader->GetBlockInfo();
          auto flat_nbytes = slot_reader->GetFlatBytes();

          auto slot_file_offset = chunk.beg_ids * flat_nbytes;
          auto read_size_bytes = chunk.num_ids * flat_nbytes;

          int64_t final_file_offset =
              slot_block_info->OffsetBeg() + slot_file_offset;

          auto slot_shape = slot_block_info->Shape();
          slot_shape[0] = chunk.num_ids;
          at::Tensor slot_tensor =
              torch::empty(slot_shape, at::TensorOptions()
                                           .device(torch::kCPU)
                                           .dtype(slot_block_info->Dtype()));
          TORCH_CHECK(
              slot_block_info->Dtype() == slot_reader->GetSlot()->Dtype(),
              "Slot dtype not match", "expected: ", slot_block_info->Dtype(),
              " actual: ", slot_reader->GetSlot()->Dtype(), ";",
              slot_block_info->DebugInfo());
          TORCH_CHECK(
              slot_tensor.sizes().vec() ==
                  slot_reader->GetSlot()->FullShape(slot_tensor.size(0)),
              "shape not match", "expected: ",
              slot_reader->GetSlot()->FullShape(slot_tensor.size(0)),
              " actual: ", slot_tensor.sizes(), ";",
              slot_block_info->DebugInfo());
          auto slot_file = slot_reader->GetTableReader()->File();
          torch::string_view slot_ret;
          RECIS_STATUS_COND(slot_file->Read(final_file_offset, read_size_bytes,
                                            &slot_ret,
                                            (char *)slot_tensor.data_ptr()));

          slot_reader->GetSlot()->IndexInsert(
              chunk_index.to(slot_reader->GetSlot()->TensorOptions().device()),
              slot_tensor.to(slot_reader->GetSlot()->TensorOptions().device()),
              chunk_accept.to(
                  slot_reader->GetSlot()->TensorOptions().device()));
        }

        future->markCompleted();
      } catch (std::exception &e) {
        LOG(ERROR) << "Chunk processing exception: " << e.what();
        future->setError(std::current_exception());
      } catch (...) {
        LOG(ERROR) << "Unknown exception in chunk processing";
        future->setError(std::current_exception());
      }
    });
    ret.push_back(future);
  }
  return ret;
}

}  // namespace serialize
}  // namespace recis
