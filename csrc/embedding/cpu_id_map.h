#pragma once
#include <unordered_map>
#include <vector>

#include "embedding/id_map.h"
#include "torch/types.h"

namespace recis {
namespace embedding {

class CpuIdMap : public IdMap {
 public:
  CpuIdMap(torch::Device id_device);
  void Lookup(const torch::Tensor &ids, torch::Tensor &output) override;
  void LookupReadOnly(const torch::Tensor &ids, torch::Tensor &output) override;
  void InsertIds(const torch::Tensor &ids, torch::Tensor &output) override;
  torch::Tensor Ids() override;
  torch::Tensor Index() override;
  void DeleteIds(const torch::Tensor &ids, const torch::Tensor &index) override;
  std::pair<torch::Tensor, torch::Tensor> SnapShot() override;
  void Clear() override;
  void Reserve(size_t id_size) override;
  int64_t Size() const override;
  int64_t Capacity() const override;
  ~CpuIdMap();

 private:
  std::mutex mu_;
  ska::flat_hash_map<int64_t, int64_t> ids_map_;
};

}  // namespace embedding
}  // namespace recis
