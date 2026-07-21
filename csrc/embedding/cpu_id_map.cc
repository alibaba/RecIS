#include "embedding/cpu_id_map.h"

#include "embedding/parallel_util.h"

namespace recis {
namespace embedding {

struct IndexLookupReadOnlyFunctor {
  IndexLookupReadOnlyFunctor(
      const int64_t *ids_vec, int64_t *out_vec,
      const ska::flat_hash_map<int64_t, int64_t> &ids_map,
      int64_t default_value)
      : default_value_(default_value),
        ids_vec_(ids_vec),
        out_vec_(out_vec),
        ids_map_(ids_map) {}

  void operator()(const int64_t begin, const int64_t end) const {
    for (auto index = begin; index < end; index++) {
      auto it = ids_map_.find(ids_vec_[index]);
      out_vec_[index] = (it == ids_map_.end()) ? default_value_ : it->second;
    }
  }

 private:
  int64_t default_value_;
  const int64_t *ids_vec_;
  int64_t *out_vec_;
  const ska::flat_hash_map<int64_t, int64_t> &ids_map_;
};

struct IndexLookupFunctor {
  IndexLookupFunctor(const int64_t *ids_vec, int64_t *out_vec,
                     ska::flat_hash_map<int64_t, int64_t> &ids_map,
                     std::vector<std::pair<int64_t, int64_t *>> &new_ids_pair)
      : ids_vec_(ids_vec),
        out_vec_(out_vec),
        ids_map_(ids_map),
        new_ids_pair_(new_ids_pair) {}

  void operator()(const int64_t begin, const int64_t end) const {
    std::vector<std::pair<int64_t, int64_t *>> new_pairs;
    for (auto index = begin; index < end; index++) {
      auto it = ids_map_.find(ids_vec_[index]);
      if (it == ids_map_.end()) {
        new_pairs.emplace_back(static_cast<int64_t>(ids_vec_[index]),
                               out_vec_ + index);
      } else {
        out_vec_[index] = it->second;
      }
    }
    {
      std::lock_guard<std::mutex> l(insert_lock_);
      for (auto &pair : new_pairs) {
        new_ids_pair_.emplace_back(pair.first, pair.second);
      }
    }
  }

 private:
  const int64_t *ids_vec_;
  int64_t *out_vec_;
  ska::flat_hash_map<int64_t, int64_t> &ids_map_;
  std::vector<std::pair<int64_t, int64_t *>> &new_ids_pair_;
  mutable std::mutex insert_lock_;
};

CpuIdMap::CpuIdMap(torch::Device id_device) {
  id_allocator_ =
      at::make_intrusive<recis::embedding::IdAllocator>(id_device, 0);
}

void CpuIdMap::Lookup(const torch::Tensor &ids, torch::Tensor &output) {
  std::lock_guard<std::mutex> lock(mu_);
  auto index_vec_ptr = output.data_ptr<int64_t>();
  std::vector<std::pair<int64_t, int64_t *>> new_ids_pairs;
  IndexLookupFunctor lookup_functor(ids.data_ptr<int64_t>(), index_vec_ptr,
                                    ids_map_, new_ids_pairs);
  torch::parallel_for(0, ids.numel(), 0, lookup_functor);
  auto new_index = id_allocator_->GenIds(new_ids_pairs.size());
  auto new_index_ptr = new_index.data_ptr<int64_t>();
  for (auto i = 0; i < new_ids_pairs.size(); ++i) {
    auto it = ids_map_.insert({new_ids_pairs[i].first, new_index_ptr[i]});
    *(new_ids_pairs[i].second) = it.first->second;
  }
}

void CpuIdMap::LookupReadOnly(const torch::Tensor &ids, torch::Tensor &output) {
  auto index_vec_ptr = output.data_ptr<int64_t>();
  IndexLookupReadOnlyFunctor lookup_functor(
      ids.data_ptr<int64_t>(), index_vec_ptr, ids_map_, kNullIndex);
  at::parallel_for(0, ids.numel(),
                   recis::CalculateIntraOpGranity(0, ids.numel()),
                   lookup_functor);
}

void CpuIdMap::InsertIds(const torch::Tensor &ids, torch::Tensor &output) {
  Lookup(ids, output);
}

torch::Tensor CpuIdMap::Ids() {
  std::lock_guard<std::mutex> l(mu_);
  int64_t act_size = id_allocator_->GetActiveSize();
  TORCH_CHECK(ids_map_.size() == act_size, "ids_map_size != act_size ",
              ids_map_.size(), " vs ", act_size);
  torch::Tensor ids = torch::empty(
      {act_size},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto ids_vec = ids.data_ptr<int64_t>();
  size_t map_idx = 0;
  for (auto it : ids_map_) {
    ids_vec[map_idx++] = it.first;
  }
  return ids;
}

torch::Tensor CpuIdMap::Index() {
  std::lock_guard<std::mutex> l(mu_);
  int64_t act_size = id_allocator_->GetActiveSize();
  TORCH_CHECK(ids_map_.size() == act_size, "ids_map_size != act_size ",
              ids_map_.size(), " vs ", act_size);

  torch::Tensor index = torch::empty(
      {act_size},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto index_vec = index.data_ptr<int64_t>();
  size_t map_idx = 0;
  for (auto it : ids_map_) {
    index_vec[map_idx++] = it.second;
  }
  return index;
}

std::pair<torch::Tensor, torch::Tensor> CpuIdMap::SnapShot() {
  std::lock_guard<std::mutex> l(mu_);
  int64_t act_size = id_allocator_->GetActiveSize();
  TORCH_CHECK(ids_map_.size() == act_size, "ids_map_size != act_size ",
              ids_map_.size(), " vs ", act_size);

  torch::Tensor ids = torch::empty(
      {act_size},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  torch::Tensor index = torch::empty(
      {act_size},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto ids_vec = ids.data_ptr<int64_t>();
  auto index_vec = index.data_ptr<int64_t>();
  size_t map_idx = 0;
  for (auto it : ids_map_) {
    ids_vec[map_idx] = it.first;
    index_vec[map_idx] = it.second;
    ++map_idx;
  }
  return std::make_pair(ids, index);
}

void CpuIdMap::DeleteIds(const torch::Tensor &ids, const torch::Tensor &index) {
  int64_t num_ids = ids.numel();
  if (num_ids == 0) {
    return;
  }
  std::lock_guard<std::mutex> l(mu_);
  const int64_t *ids_vec = ids.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_ids; ++i) {
    ids_map_.erase(ids_vec[i]);
  }
  id_allocator_->FreeIds(index);
}

void CpuIdMap::Clear() {
  std::lock_guard<std::mutex> l(mu_);
  id_allocator_->Clear();
  ids_map_.clear();
}

void CpuIdMap::Reserve(size_t id_size) { ids_map_.reserve(id_size); }

int64_t CpuIdMap::Size() const { return ids_map_.size(); }

int64_t CpuIdMap::Capacity() const { return ids_map_.bucket_count(); }

CpuIdMap::~CpuIdMap() { Clear(); }

torch::intrusive_ptr<IdMap> MakeCpuIdMap(torch::Device id_device) {
  return torch::make_intrusive<recis::embedding::CpuIdMap>(id_device);
}

torch::intrusive_ptr<IdMap> MakeIdMap(torch::Device id_device) {
  if (id_device.type() == torch::kCPU) {
    return MakeCpuIdMap(id_device);
  } else if (id_device.type() == torch::kCUDA) {
    return MakeGpuIdMap(id_device);
  } else {
    TORCH_CHECK(false,
                "Hashtable only support type [`cpu`|`cuda`], got: ", id_device);
  }
}

}  // namespace embedding
}  // namespace recis
