#pragma once
#include <cuco/dynamic_map.cuh>

#include "embedding/id_map.h"

namespace recis {
namespace embedding {

class GpuIdMap : public IdMap {
 public:
  const int64_t EmptyKey{-1};
  const int64_t ErasedKey{-2};
  const int64_t EmptyValue{-1};
  const size_t MapCapacity{640000}; /* 640'000 * (64 + 64) / 8 = 10MB */
  const float ReserveRatio{0.8};
  const float LoadFactor{0.6};

  GpuIdMap(torch::Device id_device);
  torch::Tensor Lookup(const torch::Tensor &ids) override;
  torch::Tensor LookupReadOnly(const torch::Tensor &ids) override;
  torch::Tensor InsertIds(const torch::Tensor &ids) override;
  torch::Tensor Ids() override;
  torch::Tensor Index() override;
  void DeleteIds(const torch::Tensor &ids, const torch::Tensor &index) override;
  std::pair<torch::Tensor, torch::Tensor> SnapShot() override;
  void Clear() override;
  void Reserve(size_t id_size) override;
  // std::pair<torch::Tensor, torch::Tensor> Reserve(size_t id_size) override;
  ~GpuIdMap();

  using MapType = cuco::dynamic_map<int64_t, int64_t>;

 private:
  std::unique_ptr<cuco::dynamic_map<int64_t, int64_t>> ids_map_;
};

}  // namespace embedding
}  // namespace recis
