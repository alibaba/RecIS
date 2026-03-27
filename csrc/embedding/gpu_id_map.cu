#include <ATen/cuda/CUDAContext.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "embedding/gpu_id_map.h"
#include "ops/hashtable_ops.h"

namespace recis {
namespace embedding {

GpuIdMap::GpuIdMap(torch::Device id_device) {
  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  id_allocator_ =
      at::make_intrusive<recis::embedding::IdAllocator>(id_device, 0);
  // TODO(yuhuan.zh) new cuco with torch stream && allocator
  ids_map_.reset(new cuco::flat_hash_map{MapCapacity, cuco::empty_key{EmptyKey},
                                         cuco::empty_value{EmptyValue},
                                         cuco::erased_key{ErasedKey}});
}

torch::Tensor GpuIdMap::Lookup(const torch::Tensor &ids) {
  std::lock_guard<std::mutex> lock(mu_);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor index = torch::empty_like(ids, torch::dtype(torch::kInt64));
  if (ids.numel() == 0) return index;
  torch::Tensor not_find_mask =
      torch::empty_like(ids, torch::dtype(torch::kBool));
  // lookup in cuco map
  ids_map_->find_and_mask(
      ids.data_ptr<int64_t>(), ids.data_ptr<int64_t>() + ids.numel(),
      index.data_ptr<int64_t>(), not_find_mask.data_ptr<bool>(),
      cuco::default_hash_function<int64_t>{}, cuda::std::equal_to<int64_t>{},
      stream);
  // generate index for not exist ids
  torch::Tensor mask_index = torch::cumsum(not_find_mask, 0);
  int64_t gen_num = mask_index[mask_index.numel() - 1].item<int64_t>();
  torch::Tensor new_index = id_allocator_->GenIds(gen_num);
  // fill new index to index result
  auto new_ids_tuple = recis::functional::mask_key_index_op(
      ids, not_find_mask, mask_index, gen_num);
  index = recis::functional::scatter_ids_with_mask_op(
      index, new_index, std::get<1>(new_ids_tuple));
  // insert ids and index not exist into cuco map
  auto pairs = thrust::make_transform_iterator(
      thrust::counting_iterator<std::size_t>{0},
      [new_ids = std::get<0>(new_ids_tuple).data_ptr<int64_t>(),
       new_indices = new_index.data_ptr<int64_t>()] __device__(auto i)
          -> cuco::pair<int64_t, int64_t> {
        return cuco::pair<int64_t, int64_t>{new_ids[i], new_indices[i]};
      });
  ids_map_->insert(pairs, pairs + gen_num,
                   cuco::default_hash_function<int64_t>{},
                   cuda::std::equal_to<int64_t>{}, stream);
  return index;
}

torch::Tensor GpuIdMap::LookupReadOnly(const torch::Tensor &ids) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor index = torch::empty_like(ids, torch::dtype(torch::kInt64));
  if (ids.numel() == 0) return index;
  torch::Tensor not_find_mask =
      torch::empty_like(ids, torch::dtype(torch::kBool));
  // lookup in cuco map
  ids_map_->find_and_mask(
      ids.data_ptr<int64_t>(), ids.data_ptr<int64_t>() + ids.numel(),
      index.data_ptr<int64_t>(), not_find_mask.data_ptr<bool>(),
      cuco::default_hash_function<int64_t>{}, cuda::std::equal_to<int64_t>{},
      stream);
  // generate index for not exist ids
  auto out_index = torch::where(not_find_mask, kNullIndex, index);
  return out_index;
}

torch::Tensor GpuIdMap::InsertIds(const torch::Tensor &ids) {
  auto index = Lookup(ids);
  return index;
}

std::pair<torch::Tensor, torch::Tensor> GpuIdMap::SnapShot() {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t act_size = id_allocator_->GetActiveSize();
  TORCH_CHECK(ids_map_->size() == act_size, " ids_map_size != act_size ",
              ids_map_->size(), " vs ", act_size);
  torch::Tensor ids = torch::empty(
      {act_size},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
  torch::Tensor index = torch::empty(
      {act_size},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
  ids_map_->retrieve_all(ids.data_ptr<int64_t>(), index.data_ptr<int64_t>(),
                         stream);
  return std::make_pair(ids, index);
}

torch::Tensor GpuIdMap::Ids() {
  auto snap_shot = SnapShot();
  return snap_shot.first;
}

torch::Tensor GpuIdMap::Index() {
  auto snap_shot = SnapShot();
  return snap_shot.second;
}

void GpuIdMap::DeleteIds(const torch::Tensor &ids, const torch::Tensor &index) {
  if (ids.numel() == 0) {
    return;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // TODO(sunhechen.shc) return val when erase
  ids_map_->erase(ids.data_ptr<int64_t>(),
                  ids.data_ptr<int64_t>() + ids.numel(),
                  cuco::default_hash_function<typename MapType::key_type>{},
                  cuda::std::equal_to<typename MapType::key_type>{}, stream);
  id_allocator_->FreeIds(index);
}

void GpuIdMap::Clear() {
  id_allocator_->Clear();
  ids_map_.reset(new cuco::flat_hash_map{MapCapacity, cuco::empty_key{EmptyKey},
                                         cuco::empty_value{EmptyValue},
                                         cuco::erased_key{ErasedKey}});
}

void GpuIdMap::Reserve(size_t id_size) {
  // TODO(yuhuan.zh) resize the cuco map
  TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for GpuIdMap::Reserve");
}

int64_t GpuIdMap::Size() const { return ids_map_->size(); }

int64_t GpuIdMap::Capacity() const { return ids_map_->capacity(); }

GpuIdMap::~GpuIdMap() { Clear(); }

torch::intrusive_ptr<IdMap> MakeGpuIdMap(torch::Device id_device) {
  return torch::make_intrusive<recis::embedding::GpuIdMap>(id_device);
}

}  // namespace embedding
}  // namespace recis
