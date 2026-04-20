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

class HTReadCollection : public at::intrusive_ptr_target {
 public:
  static at::intrusive_ptr<HTReadCollection> Make(
      const std::string& shared_name);

  HTReadCollection(const std::string& shared_name);

  void Append(HashTablePtr target_ht, const std::string& slot_name,
              at::intrusive_ptr<TableReader> tb_reader,
              at::intrusive_ptr<BlockInfo> block_info);

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
