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

void HTReadCollection::LoadId() {
  id_reader_->Read();
  id_done_ = true;
}

void HTReadCollection::LoadSlots() {
  TORCH_CHECK(id_done_, "id not load");
  for (auto slot_reader : block_reader_) {
    slot_reader->ExtractReadInfo(id_reader_);
    slot_reader->Read();
  }
}
c10::List<at::intrusive_ptr<at::ivalue::Future>>
HTReadCollection::LoadSlotsAsync(at::PTThreadPool *pool) {
  TORCH_CHECK(id_done_, "id not load");
  c10::List<at::intrusive_ptr<at::ivalue::Future>> ret(
      at::FutureType::create(at::NoneType::get()));
  for (auto slot_reader : block_reader_) {
    slot_reader->ExtractReadInfo(id_reader_);
    ret.append(slot_reader->ReadAsync(pool));
  }
  return ret;
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
}  // namespace serialize
}  // namespace recis
