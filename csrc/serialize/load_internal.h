#pragma once
#include <string>
#include <unordered_map>

#include "c10/util/intrusive_ptr.h"
#include "serialize/ht_read_collection.h"
#include "serialize/load_bundle.h"
#include "serialize/load_info.h"
#include "serialize/load_summary.h"
#include "serialize/read_block.h"

namespace recis {
namespace serialize {

class LoaderInternal : public at::intrusive_ptr_target {
 public:
  static at::intrusive_ptr<LoaderInternal> Make(
      const LoadInfo &load_info, at::intrusive_ptr<LoadBundle> load_bundle,
      const std::unordered_map<std::string, HashTablePtr> &hts_to_load,
      const std::unordered_map<std::string, at::Tensor> &tensors_to_load,
      int64_t parallel);
  void Load(int64_t &load_size);
  at::intrusive_ptr<LoadSummary> GetLoadSummary() const;

 private:
  void BuildHTLoadCollection(
      const std::unordered_map<std::string, HashTablePtr> &hts_to_load,
      const LoadInfo &load_info);
  void BuildTensorLoadCollection(
      const std::unordered_map<std::string, at::Tensor> &tensors_to_load,
      const LoadInfo &load_info);
  at::intrusive_ptr<LoadBundle> load_bundle_;
  ska::flat_hash_map<void *, std::vector<at::intrusive_ptr<HTReadCollection>>>
      ht_load_collections_;
  std::vector<at::intrusive_ptr<TensorReadBlock>> tensor_read_blocks_;
  at::intrusive_ptr<LoadSummary> load_summary_;
  int64_t parallel_;
};
}  // namespace serialize
}  // namespace recis