#pragma once
#include <vector>

#include "ATen/PTThreadPool.h"
#include "ATen/Utils.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/hashtable.h"
#include "embedding/slot_group.h"
#include "serialize/block_info.h"
#include "serialize/read_block.h"
#include "serialize/table_reader.h"
namespace recis {
namespace serialize {

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

class HTReadCollection : public at::intrusive_ptr_target {
 public:
  static at::intrusive_ptr<HTReadCollection> Make(
      const std::string& shared_name);
  HTReadCollection(const std::string& shared_name);
  void Append(HashTablePtr target_ht, const std::string& slot_name,
              at::intrusive_ptr<TableReader> tb_reader,
              at::intrusive_ptr<BlockInfo> block_info);
  void LoadId();
  c10::List<at::intrusive_ptr<at::ivalue::Future>> LoadSlotsAsync(
      at::PTThreadPool* pool);
  void LoadSlots();

  std::vector<at::intrusive_ptr<embedding::Slot>> ReadSlots();
  bool Valid();
  bool Empty();

  c10::List<at::intrusive_ptr<at::ivalue::Future>> LoadChunksAsync(
      at::PTThreadPool* pool, int64_t chunk_size_bytes = 1 << 23);

  AT_DISALLOW_COPY_AND_ASSIGN(HTReadCollection);

 private:
  bool id_done_;
  std::string share_name_;
  at::intrusive_ptr<HTIdReadBlock> id_reader_;
  std::vector<at::intrusive_ptr<HTSlotReadBlock>> block_reader_;
};
}  // namespace serialize
}  // namespace recis
