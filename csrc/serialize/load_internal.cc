#include "serialize/load_internal.h"

#include <string>

#include "ATen/PTThreadPool.h"
#include "ATen/core/List.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/ivalue_inl.h"
#include "ATen/core/jit_type.h"
#include "c10/cuda/CUDAFunctions.h"
#include "c10/util/Exception.h"
#include "c10/util/flat_hash_map.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/hashtable.h"
#include "embedding/slot_group.h"
#include "serialize/load_summary.h"
#include "serialize/name.h"
#include "serialize/read_block.h"

namespace recis {
namespace {
class HTSlotsEmptyInitializerContext {
 public:
  AT_DISALLOW_COPY_AND_ASSIGN(HTSlotsEmptyInitializerContext);
  HTSlotsEmptyInitializerContext(Hashtable *ht) : ht_(ht) {}
  void AppendSlot(at::intrusive_ptr<embedding::Slot> slot) {
    if (slot_initializer_map_.count(slot.get())) return;
    slot_initializer_map_[slot.get()] = slot->Generator();
    auto empty_generator = embedding::MakeEmptyGenerator(
        slot->Generator()->Shape(), slot->Dtype());
    empty_generator->set_device(
        slot_initializer_map_[slot.get()]->TensorOption().device());
    slot->Generator(empty_generator);
  }

  HTSlotsEmptyInitializerContext(HTSlotsEmptyInitializerContext &&other) {
    ht_ = other.ht_;
    slot_initializer_map_ = other.slot_initializer_map_;
    other.slot_initializer_map_.clear();
  }

  ~HTSlotsEmptyInitializerContext() {
    int64_t last_copy_index = ht_->IdsNum() % ht_->SlotGroup()->BlockSize();
    int64_t rest_num = ht_->SlotGroup()->BlockSize() - last_copy_index;
    for (auto kv : slot_initializer_map_) {
      embedding::Slot *slot = (embedding::Slot *)kv.first;
      slot->Generator(kv.second);
      auto last_block = slot->Values()->back();
      slot->Generator()->Initialize(
          last_block.narrow(0, last_copy_index, rest_num));
    }
  }

 private:
  Hashtable *ht_;
  ska::flat_hash_map<void *, at::intrusive_ptr<embedding::Generator>>
      slot_initializer_map_;
};
}  // namespace
namespace serialize {
at::intrusive_ptr<LoaderInternal> LoaderInternal::Make(
    const LoadInfo &load_info, at::intrusive_ptr<LoadBundle> load_bundle,
    const std::unordered_map<std::string, HashTablePtr> &hts_to_load,
    const std::unordered_map<std::string, at::Tensor> &tensors_to_load,
    int64_t parallel) {
  auto obj = at::make_intrusive<LoaderInternal>();
  obj->load_summary_ = at::make_intrusive<LoadSummary>();
  obj->load_summary_->InitFromLoadInfo(load_info);
  obj->load_bundle_ = load_bundle;
  obj->BuildHTLoadCollection(hts_to_load, load_info);
  obj->BuildTensorLoadCollection(tensors_to_load, load_info);
  obj->parallel_ = parallel;
  return obj;
}

void LoaderInternal::BuildHTLoadCollection(
    const std::unordered_map<std::string, HashTablePtr> &hts_to_load,
    const LoadInfo &load_info) {
  ska::flat_hash_map<std::string, HashTablePtr> name_to_ht;
  ska::flat_hash_map<HashTablePtr, size_t> ht_to_count;
  for (const auto &kv : hts_to_load) {
    ht_to_count[kv.second] = 0;
    for (const auto &name : kv.second->ChildrenInfo()->Children()) {
      name_to_ht[name] = kv.second;
    }
  }
  // 1. calculate coalesced hashtable id num
  for (const auto &info_entry : load_info.Infos()) {
    const auto &shared_name = info_entry.first;
    if (name_to_ht.count(shared_name)) {
      // find the ht in model, to find block in ckpt
      auto target_ht = name_to_ht.at(shared_name);
      for (const auto &load_entry : info_entry.second) {
        const auto &load_shared_name = load_entry.first;
        auto ht_id_name = HTSlotNameEncode(load_shared_name, "id");
        if (load_bundle_->HasTensor(ht_id_name)) {
          auto id_shape = load_bundle_->TensorShape(ht_id_name);
          ht_to_count[target_ht] += id_shape[0];
        }
      }
    }
  }
  // 2. reserve hashtable size for gpu
  for (const auto &kv : hts_to_load) {
    LOG(WARNING) << "not reserve, HashTable name: " << kv.first
                 << ", total id size: " << ht_to_count[kv.second];
    // kv.second->Reserve(ht_to_count[kv.second]);
  }
  // 3. load hashtable
  for (const auto &info_entry : load_info.Infos()) {
    const auto &shared_name = info_entry.first;
    if (name_to_ht.count(shared_name)) {
      // find the ht in model, to find block in ckpt
      auto target_ht = name_to_ht.at(shared_name);
      for (const auto &load_entry : info_entry.second) {
        const auto &load_shared_name = load_entry.first;
        const auto &slot_names = load_entry.second;
        ska::flat_hash_map<std::string, at::intrusive_ptr<HTReadCollection>>
            read_collections;
        for (const auto &slot_name : slot_names) {
          auto tensor_name = HTSlotNameEncode(load_shared_name, slot_name);
          // std::string tensor_name =
          // torch::str(shared_name, TensorSymbolAt(), slot_name);
          if (load_bundle_->HasTensor(tensor_name)) {
            this->load_summary_->MarkVariableLoaded(
                HTSlotNameEncode(shared_name, slot_name), tensor_name);
            // find tensor in ckpt.
            auto slice_infos = load_bundle_->SliceInfos(tensor_name);
            for (const auto &slice_info_str : slice_infos) {
              auto slice_info =
                  embedding::SliceInfo::FromString(slice_info_str);
              if (slice_info->IsEqualRange(target_ht->SliceInfo()) &&
                  slice_info->IsIntersect(target_ht->SliceInfo())) {
                if (read_collections.count(slice_info_str) == 0) {
                  read_collections[slice_info_str] =
                      HTReadCollection::Make(shared_name);
                }
                auto read_collection = read_collections[slice_info_str];
                auto block_name = BlockNameEncode(tensor_name, slice_info_str);
                auto block_info = load_bundle_->GetBlockInfo(block_name);
                auto file = load_bundle_->BlockReadFile(block_name);
                read_collection->Append(target_ht, slot_name, file, block_info);
              }
            }
          } else {
            // not find tensor in ckpt.
            LOG(WARNING) << "Sparse tensor [" << tensor_name
                         << "] not found in safetensors";
          }
        }
        for (auto kv : read_collections) {
          if (kv.second->Empty()) {
            continue;
          }
        }
        for (auto kv : read_collections) {
          auto read_collection = kv.second;
          TORCH_CHECK(read_collection->Valid(), load_shared_name, ", invalid");
          ht_load_collections_[target_ht.get()].push_back(read_collection);
        }
      }
    }
  }
  LOG(WARNING) << " Load Sparse Meta Info Complete";
}

void LoaderInternal::BuildTensorLoadCollection(
    const std::unordered_map<std::string, at::Tensor> &tensors_to_load,
    const LoadInfo &load_info) {
  for (const auto &info_entry : load_info.Infos()) {
    auto tensor_name = info_entry.first;
    if (tensors_to_load.count(tensor_name)) {
      auto target_tensor = tensors_to_load.at(tensor_name);
      const auto &load_entrys = info_entry.second;
      TORCH_CHECK(load_entrys.size() == 1, "currently only support one tensor");
      for (const auto &load_entry : load_entrys) {
        const auto &load_tensor_name = load_entry.first;
        if (load_bundle_->HasTensor(load_tensor_name)) {
          this->load_summary_->MarkVariableLoaded(tensor_name,
                                                  load_tensor_name);
          const auto &load_block_name =
              BlockNameEncode(load_tensor_name, EmptySliceInfo());
          auto block_info = load_bundle_->GetBlockInfo(load_block_name);
          auto file = load_bundle_->BlockReadFile(load_block_name);
          tensor_read_blocks_.push_back(
              TensorReadBlock::Make(target_tensor, block_info, file));
        } else {
          // not find the tensor in ckpt.
          LOG(WARNING) << "Dense tensor [" << load_tensor_name
                       << "] not found in safetensors";
        }
      }
    }
  }
  LOG(WARNING) << " Load Dense Meta Info Complete";
}

void LoaderInternal::Load(int64_t &load_size) {
  ReadBlock::size_counter_.InitCounter();

  if (ht_load_collections_.empty() && tensor_read_blocks_.empty()) {
    return;
  }
  at::PTThreadPool pool(parallel_);
  c10::Device device = torch::cuda::is_available()
                           ? c10::Device(c10::kCUDA, at::cuda::current_device())
                           : c10::Device(c10::kCPU);
  std::vector<HTSlotsEmptyInitializerContext> empty_context;
  for (auto &kv : ht_load_collections_) {
    empty_context.emplace_back((Hashtable *)kv.first);
    for (auto &ht_read_collection : kv.second) {
      for (auto &slot : ht_read_collection->ReadSlots()) {
        empty_context.back().AppendSlot(slot);
      }
    }
  }
  {
    c10::List<at::intrusive_ptr<at::ivalue::Future>> futures(
        at::FutureType::create(at::NoneType::get()));
    for (auto &kv : ht_load_collections_) {
      auto future = at::make_intrusive<at::ivalue::Future>(at::NoneType::get());
      pool.run([&kv, device, future]() {
        try {
          c10::DeviceGuard device_guard(device);
          for (auto &ht_read_collection : kv.second) {
            ht_read_collection->LoadId();
          }
          future->markCompleted();
        } catch (std::exception &e) {
          LOG(ERROR) << e.what();
          future->setError(std::current_exception());
        } catch (...) {
          LOG(ERROR) << "unknown exception";
        }
      });
      futures.push_back(future);
    }
    c10::collectAll(futures)->waitAndThrow();
  }
  pool.waitWorkComplete();
  LOG(WARNING) << " Load Sparse Id Complete";
  {
    c10::List<c10::intrusive_ptr<at::ivalue::Future>> futures(
        at::FutureType::create(at::NoneType::get()));
    for (auto &kv : ht_load_collections_) {
      for (auto &ht_read_collection : kv.second) {
        futures.append(ht_read_collection->LoadSlotsAsync(&pool));
      }
    }
    auto res = c10::collectAll(futures);
    res->waitAndThrow();
  }
  LOG(WARNING) << " Load Sparse Slot Complete";
  {
    c10::List<at::intrusive_ptr<at::ivalue::Future>> futures(
        at::FutureType::create(at::NoneType::get()));
    for (auto &tensor_read_block : tensor_read_blocks_) {
      auto future = at::make_intrusive<at::ivalue::Future>(at::NoneType::get());
      pool.run([tensor_read_block, device, future]() {
        try {
          c10::DeviceGuard device_guard(device);
          tensor_read_block->Read();
          future->markCompleted();
        } catch (std::exception &e) {
          LOG(ERROR) << e.what();
          future->setError(std::current_exception());
        } catch (...) {
          LOG(ERROR) << "unknown exception";
          future->setError(std::current_exception());
        }
      });
      futures.push_back(future);
    }
    c10::collectAll(futures)->waitAndThrow();
  }
  pool.waitWorkComplete();
  LOG(WARNING) << " Load Dense Tensor Complete";
  load_size = ReadBlock::size_counter_.GetSize();
}

at::intrusive_ptr<LoadSummary> LoaderInternal::GetLoadSummary() const {
  return load_summary_;
}

}  // namespace serialize
}  // namespace recis
