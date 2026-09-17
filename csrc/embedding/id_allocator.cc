#include "embedding/id_allocator.h"

#include "ops/hashtable_ops.h"

namespace recis {
namespace embedding {

IdAllocator::IdAllocator(torch::Device id_device, int64_t init_size,
                         size_t free_block_size)
    : id_device_(id_device),
      cur_size_(init_size),
      free_block_size_(free_block_size),
      free_size_(0) {
  IncreaseBlock(0);
}

torch::Tensor IdAllocator::GenIds(size_t num_ids) {
  std::lock_guard<std::mutex> lock(mu_);
  torch::Tensor out_ids = recis::functional::generate_ids_op(
      num_ids, free_blocks_, free_size_, cur_size_, free_block_size_);
  auto new_id_num = (free_size_ >= num_ids) ? 0 : (num_ids - free_size_);
  cur_size_ = cur_size_ + new_id_num;
  free_size_ = std::max(0, static_cast<int>(free_size_ - num_ids));
  ReduceBlock();
  return out_ids;
}

void IdAllocator::FreeIds(torch::Tensor ids) {
  std::lock_guard<std::mutex> lock(mu_);
  int64_t free_size_back_ = free_size_;
  IncreaseBlock(ids.numel());
  recis::functional::free_ids_op(ids, free_blocks_, free_size_back_,
                                 free_block_size_);
}

void IdAllocator::ReduceBlock() {
  auto reduce_size = free_size_ / free_block_size_ + 1;
  auto cur_block_size = free_blocks_.size();
  while (cur_block_size > reduce_size) {
    free_blocks_.pop_back();
    --cur_block_size;
  }
}

void IdAllocator::IncreaseBlock(int64_t increase_num) {
  free_size_ += increase_num;
  auto inc_size = free_size_ / free_block_size_ + 1;
  auto cur_block_size = free_blocks_.size();
  while (cur_block_size < inc_size) {
    free_blocks_.push_back(torch::empty(
        {static_cast<int64_t>(free_block_size_)},
        torch::TensorOptions().dtype(torch::kInt64).device(id_device_)));
    ++cur_block_size;
  }
}

void IdAllocator::Clear() {
  std::lock_guard<std::mutex> lock(mu_);
  free_size_ = 0;
  cur_size_ = 0;
  free_blocks_.resize(1);
}

IdAllocator::~IdAllocator() { free_blocks_.clear(); }

}  // namespace embedding
}  // namespace recis
