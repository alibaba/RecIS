/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

namespace cuco {

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
dynamic_map<Key, Value, Scope, Allocator>::dynamic_map(std::size_t initial_capacity,
                                                       empty_key<Key> empty_key_sentinel,
                                                       empty_value<Value> empty_value_sentinel,
                                                       Allocator const& alloc,
                                                       cudaStream_t stream)
  : empty_key_sentinel_(empty_key_sentinel.value),
    empty_value_sentinel_(empty_value_sentinel.value),
    erased_key_sentinel_(empty_key_sentinel.value),
    size_(0),
    capacity_(initial_capacity),
    min_insert_size_(1E4),
    max_load_factor_(0.60),
    alloc_{alloc}
{
  submaps_.push_back(std::make_unique<cuco::legacy::static_map<Key, Value, Scope, Allocator>>(
    initial_capacity,
    empty_key<Key>{empty_key_sentinel},
    empty_value<Value>{empty_value_sentinel},
    alloc,
    stream));
  submap_views_.push_back(submaps_[0]->get_device_view());
  submap_mutable_views_.push_back(submaps_[0]->get_device_mutable_view());
  submap_num_successes_.push_back(submaps_[0]->num_successes_);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
dynamic_map<Key, Value, Scope, Allocator>::dynamic_map(std::size_t initial_capacity,
                                                       empty_key<Key> empty_key_sentinel,
                                                       empty_value<Value> empty_value_sentinel,
                                                       erased_key<Key> erased_key_sentinel,
                                                       Allocator const& alloc,
                                                       cudaStream_t stream)
  : empty_key_sentinel_(empty_key_sentinel.value),
    empty_value_sentinel_(empty_value_sentinel.value),
    erased_key_sentinel_(erased_key_sentinel.value),
    size_(0),
    capacity_(initial_capacity),
    min_insert_size_(1E4),
    max_load_factor_(0.60),
    alloc_{alloc}
{
  CUCO_EXPECTS(empty_key_sentinel_ != erased_key_sentinel_,
               "The empty key sentinel and erased key sentinel cannot be the same value.",
               std::runtime_error);

  submaps_.push_back(std::make_unique<cuco::legacy::static_map<Key, Value, Scope, Allocator>>(
    initial_capacity,
    empty_key<Key>{empty_key_sentinel_},
    empty_value<Value>{empty_value_sentinel_},
    erased_key<Key>{erased_key_sentinel_},
    alloc,
    stream));
  submap_views_.push_back(submaps_[0]->get_device_view());
  submap_mutable_views_.push_back(submaps_[0]->get_device_mutable_view());
  submap_num_successes_.push_back(submaps_[0]->num_successes_);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
void dynamic_map<Key, Value, Scope, Allocator>::reserve(std::size_t n, cudaStream_t stream)
{
  int64_t num_elements_remaining = n;
  uint32_t submap_idx            = 0;
  while (num_elements_remaining > 0) {
    std::size_t submap_capacity;

    // if the submap already exists
    if (submap_idx < submaps_.size()) {
      submap_capacity = submaps_[submap_idx]->get_capacity();
    }
    // if the submap does not exist yet, create it
    else {
      submap_capacity = capacity_;
      if (erased_key_sentinel_ != empty_key_sentinel_) {
        submaps_.push_back(std::make_unique<cuco::legacy::static_map<Key, Value, Scope, Allocator>>(
          submap_capacity,
          empty_key<Key>{empty_key_sentinel_},
          empty_value<Value>{empty_value_sentinel_},
          erased_key<Key>{erased_key_sentinel_},
          alloc_,
          stream));
      } else {
        submaps_.push_back(std::make_unique<cuco::legacy::static_map<Key, Value, Scope, Allocator>>(
          submap_capacity,
          empty_key<Key>{empty_key_sentinel_},
          empty_value<Value>{empty_value_sentinel_},
          alloc_,
          stream));
      }
      submap_num_successes_.push_back(submaps_[submap_idx]->num_successes_);
      submap_views_.push_back(submaps_[submap_idx]->get_device_view());
      submap_mutable_views_.push_back(submaps_[submap_idx]->get_device_mutable_view());
      capacity_ *= 2;
    }

    num_elements_remaining -= max_load_factor_ * submap_capacity - min_insert_size_;
    submap_idx++;
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::insert(
  InputIt first, InputIt last, Hash hash, KeyEqual key_equal, cudaStream_t stream)
{
  // TODO: memset an atomic variable is unsafe
  static_assert(sizeof(std::size_t) == sizeof(atomic_ctr_type),
                "sizeof(atomic_ctr_type) must be equal to sizeof(std:size_t).");

  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto constexpr tile_size  = 4;

  std::size_t num_to_insert = std::distance(first, last);

  reserve(size_ + num_to_insert, stream);

  uint32_t submap_idx = 0;
  while (num_to_insert > 0) {
    std::size_t capacity_remaining =
      max_load_factor_ * submaps_[submap_idx]->get_capacity() - submaps_[submap_idx]->get_size();
    // If we are tying to insert some of the remaining keys into this submap, we can insert
    // only if we meet the minimum insert size.
    if (capacity_remaining >= min_insert_size_) {
      CUCO_CUDA_TRY(
        cudaMemsetAsync(submap_num_successes_[submap_idx], 0, sizeof(atomic_ctr_type), stream));

      auto const n         = std::min(capacity_remaining, num_to_insert);
      auto const grid_size = (tile_size * n + stride * block_size - 1) / (stride * block_size);

      detail::insert<block_size, tile_size, cuco::pair<key_type, mapped_type>>
        <<<grid_size, block_size, 0, stream>>>(first,
                                               first + n,
                                               submap_views_.data().get(),
                                               submap_mutable_views_.data().get(),
                                               submap_num_successes_.data().get(),
                                               submap_idx,
                                               submaps_.size(),
                                               hash,
                                               key_equal);

      std::size_t h_num_successes;
      CUCO_CUDA_TRY(cudaMemcpyAsync(&h_num_successes,
                                    submap_num_successes_[submap_idx],
                                    sizeof(atomic_ctr_type),
                                    cudaMemcpyDeviceToHost,
                                    stream));
	  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
      submaps_[submap_idx]->size_ += h_num_successes;
      size_ += h_num_successes;
      first += n;
      num_to_insert -= n;
    }
    submap_idx++;
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::insert_unsafe(
  InputIt first, InputIt last, Hash hash, KeyEqual key_equal, cudaStream_t stream)
{
  // TODO: memset an atomic variable is unsafe
  static_assert(sizeof(std::size_t) == sizeof(atomic_ctr_type),
                "sizeof(atomic_ctr_type) must be equal to sizeof(std:size_t).");

  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto constexpr tile_size  = 4;

  std::size_t num_to_insert = std::distance(first, last);

  reserve(size_ + num_to_insert, stream);

  uint32_t submap_idx = 0;
  while (num_to_insert > 0) {
    std::size_t capacity_remaining =
      max_load_factor_ * submaps_[submap_idx]->get_capacity() - submaps_[submap_idx]->get_size();
    // If we are tying to insert some of the remaining keys into this submap, we can insert
    // only if we meet the minimum insert size.
    if (capacity_remaining >= min_insert_size_) {
      CUCO_CUDA_TRY(
        cudaMemsetAsync(submap_num_successes_[submap_idx], 0, sizeof(atomic_ctr_type), stream));

      auto const n         = std::min(capacity_remaining, num_to_insert);
      auto const grid_size = (tile_size * n + stride * block_size - 1) / (stride * block_size);

      detail::insert_unsafe<block_size, tile_size, cuco::pair<key_type, mapped_type>>
        <<<grid_size, block_size, 0, stream>>>(first,
                                               first + n,
                                               submap_views_.data().get(),
                                               submap_mutable_views_.data().get(),
                                               submap_num_successes_.data().get(),
                                               submap_idx,
                                               submaps_.size(),
                                               hash,
                                               key_equal);

      std::size_t h_num_successes;
      CUCO_CUDA_TRY(cudaMemcpyAsync(&h_num_successes,
                                    submap_num_successes_[submap_idx],
                                    sizeof(atomic_ctr_type),
                                    cudaMemcpyDeviceToHost,
                                    stream));
	  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
      submaps_[submap_idx]->size_ += h_num_successes;
      size_ += h_num_successes;
      first += n;
      num_to_insert -= n;
    }
    submap_idx++;
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename KeyIt, typename ValueIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::insert_key_value_unsafe(
  KeyIt key_first, KeyIt key_last, ValueIt value_first, cudaStream_t stream, Hash hash, KeyEqual key_equal)
{
  // TODO: memset an atomic variable is unsafe
  static_assert(sizeof(std::size_t) == sizeof(atomic_ctr_type),
                "sizeof(atomic_ctr_type) must be equal to sizeof(std:size_t).");

  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto constexpr tile_size  = 4;

  std::size_t num_to_insert = std::distance(key_first, key_last);

  reserve(size_ + num_to_insert, stream);

  uint32_t submap_idx = 0;
  while (num_to_insert > 0) {
    std::size_t capacity_remaining =
      max_load_factor_ * submaps_[submap_idx]->get_capacity() - submaps_[submap_idx]->get_size();
    // If we are tying to insert some of the remaining keys into this submap, we can insert
    // only if we meet the minimum insert size.
    if (capacity_remaining >= min_insert_size_) {
      CUCO_CUDA_TRY(
        cudaMemsetAsync(submap_num_successes_[submap_idx], 0, sizeof(atomic_ctr_type), stream));

      auto const n         = std::min(capacity_remaining, num_to_insert);
      auto const grid_size = (tile_size * n + stride * block_size - 1) / (stride * block_size);

      detail::insert_key_value_unsafe<block_size, tile_size, cuco::pair<key_type, mapped_type>>
        <<<grid_size, block_size, 0, stream>>>(key_first,
                                               key_first + n,
                                               value_first,
                                               submap_views_.data().get(),
                                               submap_mutable_views_.data().get(),
                                               submap_num_successes_.data().get(),
                                               submap_idx,
                                               submaps_.size(),
                                               hash,
                                               key_equal);

      std::size_t h_num_successes;
      CUCO_CUDA_TRY(cudaMemcpyAsync(&h_num_successes,
                                    submap_num_successes_[submap_idx],
                                    sizeof(atomic_ctr_type),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
      submaps_[submap_idx]->size_ += h_num_successes;
      size_ += h_num_successes;
      key_first += n;
      value_first += n;
      num_to_insert -= n;
    }
    submap_idx++;
  }
}
template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::erase(
  InputIt first, InputIt last, Hash hash, KeyEqual key_equal, cudaStream_t stream)
{
  // TODO: memset an atomic variable is unsafe
  static_assert(sizeof(std::size_t) == sizeof(atomic_ctr_type),
                "sizeof(atomic_ctr_type) must be equal to sizeof(std:size_t).");
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto constexpr tile_size  = 4;

  auto const num_keys  = std::distance(first, last);
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  // zero out submap success counters
  for (uint32_t i = 0; i < submaps_.size(); ++i) {
    CUCO_CUDA_TRY(cudaMemsetAsync(submap_num_successes_[i], 0, sizeof(atomic_ctr_type), stream));
  }

  auto const temp_storage_size = submaps_.size() * sizeof(unsigned long long);

  detail::erase<block_size, tile_size>
    <<<grid_size, block_size, temp_storage_size, stream>>>(first,
                                                           first + num_keys,
                                                           submap_mutable_views_.data().get(),
                                                           submap_num_successes_.data().get(),
                                                           submaps_.size(),
                                                           hash,
                                                           key_equal);

  std::vector<std::size_t> h_submap_num_successes_vec(submaps_.size());
  
  for (uint32_t i = 0; i < submaps_.size(); ++i) {
    CUCO_CUDA_TRY(cudaMemcpyAsync(&h_submap_num_successes_vec[i],
                                  submap_num_successes_[i],
                                  sizeof(atomic_ctr_type),
                                  cudaMemcpyDeviceToHost,
                                  stream));
  }
  
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  
  for (uint32_t i = 0; i < submaps_.size(); ++i) {
    submaps_[i]->size_ -= h_submap_num_successes_vec[i];
    size_ -= h_submap_num_successes_vec[i];
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::find(InputIt first,
                                                     InputIt last,
                                                     OutputIt output_begin,
                                                     Hash hash,
                                                     KeyEqual key_equal,
                                                     cudaStream_t stream)
{
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;

  auto const num_keys  = std::distance(first, last);
  auto const grid_size = (num_keys + stride * block_size - 1) / (stride * block_size);

  detail::find<block_size, Value><<<grid_size, block_size, 0, stream>>>(
    first, last, output_begin, submap_views_.data().get(), submaps_.size(), hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename ValueOutIt, typename MaskOutIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::find_value_and_mask(InputIt first,
                                                                    InputIt last,
                                                                    ValueOutIt output_begin,
                                                                    MaskOutIt mask_begin,
                                                                    cudaStream_t stream,
                                                                    Hash hash,
                                                                    KeyEqual key_equal)
{
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;

  auto const num_keys  = std::distance(first, last);
  auto const grid_size = (num_keys + stride * block_size - 1) / (stride * block_size);
  
  detail::find_value_and_mask<block_size, Value><<<grid_size, block_size, 0, stream>>>(
    first, last, output_begin, mask_begin, submap_views_.data().get(), submaps_.size(), hash, key_equal);
}


template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename KeyOut, typename ValueOut>
std::pair<KeyOut, ValueOut> dynamic_map<Key, Value, Scope, Allocator>::retrieve_all_naive(
  KeyOut keys_out, ValueOut values_out, cudaStream_t stream) const
{
  for (auto i = 0; i < submaps_.size(); ++i) {
    std::tie(keys_out, values_out) = submaps_[i]->retrieve_all(keys_out, values_out, stream);
  }
  return {keys_out, values_out};
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename KeyOut, typename ValueOut>
int64_t dynamic_map<Key, Value, Scope, Allocator>::retrieve_all(
  KeyOut keys_out, ValueOut values_out, cudaStream_t stream) const
{
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;

  auto const capacity = get_capacity();
  auto grid_size      = (capacity + stride * block_size - 1) / (stride * block_size);

  std::vector<size_t> submap_cap_prefix(submaps_.size());
  std::inclusive_scan(
    submaps_.begin(),
    submaps_.end(),
    submap_cap_prefix.begin(),
    [](auto const& sum, auto const& submap) { return sum + submap->get_capacity(); },
    size_t{0});
  thrust::device_vector<size_t> submap_cap_prefix_d(submap_cap_prefix);

  using temp_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<char>;
  auto temp_allocator = temp_allocator_type{alloc_};
  auto d_num_out =
    reinterpret_cast<unsigned long long*>(std::allocator_traits<temp_allocator_type>::allocate(
      temp_allocator, sizeof(unsigned long long)));
  CUCO_CUDA_TRY(cudaMemsetAsync(d_num_out, 0, sizeof(unsigned long long), stream));

  detail::retrieve_all<block_size>
    <<<grid_size, block_size, 0, stream>>>(keys_out,
                                           values_out,
                                           submap_views_.data().get(),
                                           submaps_.size(),
                                           capacity,
                                           d_num_out,
                                           submap_cap_prefix_d.data().get());

  size_t h_num_out;
  CUCO_CUDA_TRY(
    cudaMemcpyAsync(&h_num_out, d_num_out, sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  std::allocator_traits<temp_allocator_type>::deallocate(
    temp_allocator, reinterpret_cast<char*>(d_num_out), sizeof(unsigned long long));
  return h_num_out;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename KeyOut, typename ValueOut>
int64_t dynamic_map<Key, Value, Scope, Allocator>::retrieve_all(
  KeyOut keys_out, ValueOut values_out, int64_t start, int64_t range, cudaStream_t stream) const
{
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;

  int64_t const capacity = get_capacity();
  range = std::min(capacity - start, range);
  auto grid_size      = (range + stride * block_size - 1) / (stride * block_size);

  std::vector<size_t> submap_cap_prefix(submaps_.size());
  std::inclusive_scan(
    submaps_.begin(),
    submaps_.end(),
    submap_cap_prefix.begin(),
    [](auto const& sum, auto const& submap) { return sum + submap->get_capacity(); },
    size_t{0});
  thrust::device_vector<size_t> submap_cap_prefix_d(submap_cap_prefix);

  using temp_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<char>;
  auto temp_allocator = temp_allocator_type{alloc_};
  auto d_num_out =
    reinterpret_cast<unsigned long long*>(std::allocator_traits<temp_allocator_type>::allocate(
      temp_allocator, sizeof(unsigned long long)));
  CUCO_CUDA_TRY(cudaMemsetAsync(d_num_out, 0, sizeof(unsigned long long), stream));
  detail::retrieve_all<block_size, Key, Value>
    <<<grid_size, block_size, 0, stream>>>(keys_out,
                                           values_out,
                                           submap_views_.data().get(),
                                           submaps_.size(),
                                           start,
                                           range,
                                           d_num_out,
                                           submap_cap_prefix_d.data().get());

  size_t h_num_out;
  CUCO_CUDA_TRY(
    cudaMemcpyAsync(&h_num_out, d_num_out, sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  std::allocator_traits<temp_allocator_type>::deallocate(
    temp_allocator, reinterpret_cast<char*>(d_num_out), sizeof(unsigned long long));
  return h_num_out;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::contains(InputIt first,
                                                         InputIt last,
                                                         OutputIt output_begin,
                                                         Hash hash,
                                                         KeyEqual key_equal,
                                                         cudaStream_t stream)
{
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto constexpr tile_size  = 4;

  auto const num_keys  = std::distance(first, last);
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  detail::contains<block_size, tile_size><<<grid_size, block_size, 0, stream>>>(
    first, last, output_begin, submap_views_.data().get(), submaps_.size(), hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace cuco
