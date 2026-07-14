/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#ifdef USE_ROCM
// .inl file will not be hipified, so we need to include the hip version of the file
// or we will encounter redefinition of bitwise_compare
#include <cuco/detail/bitwise_compare_hip.cuh>
#include <cuco/detail/error_hip.hpp>
#include <cuco/detail/utils_hip.cuh>

#include <hipcub/device/device_select.hpp>
#include <thrust/tuple.h>

#define cudaStream_t hipStream_t
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#else
#include <cub/device/device_select.cuh>
#include <cuco/detail/bitwise_compare.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/utils.cuh>

#include <cuda/std/tuple>
#endif

#include <cuda/std/utility>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cuco {

#ifdef USE_ROCM
namespace cub = hipcub;
#endif

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
flat_hash_map<Key, Value, Scope, Allocator>::flat_hash_map(std::size_t initial_capacity,
                                                           empty_key<Key> empty_key_sentinel,
                                                           empty_value<Value> empty_value_sentinel,
                                                           double desired_load_factor,
                                                           Allocator const& alloc,
                                                           cudaStream_t stream)
  : capacity_{std::max(initial_capacity, std::size_t{1})},
    size_{0},
    max_load_factor_{desired_load_factor},
    empty_key_sentinel_{empty_key_sentinel.value},
    empty_value_sentinel_{empty_value_sentinel.value},
    erased_key_sentinel_{empty_key_sentinel.value},
    allocator_{alloc}
{
  CUCO_EXPECTS(capacity_ <= std::numeric_limits<std::uint32_t>::max(),
               "flat_hash_map capacity must not exceed UINT32_MAX.",
               std::runtime_error);

  slots_ = std::allocator_traits<slot_allocator_type>::allocate(allocator_, capacity_);

  auto constexpr block_size = 256;
  auto constexpr stride     = 4;
  auto const grid_size      = (capacity_ + stride * block_size - 1) / (stride * block_size);
  detail::initialize<block_size, atomic_key_type, atomic_mapped_type>
    <<<grid_size, block_size, 0, stream>>>(
      slots_, empty_key_sentinel_, empty_value_sentinel_, capacity_);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
flat_hash_map<Key, Value, Scope, Allocator>::flat_hash_map(std::size_t initial_capacity,
                                                           empty_key<Key> empty_key_sentinel,
                                                           empty_value<Value> empty_value_sentinel,
                                                           erased_key<Key> erased_key_sentinel,
                                                           double desired_load_factor,
                                                           Allocator const& alloc,
                                                           cudaStream_t stream)
  : capacity_{std::max(initial_capacity, std::size_t{1})},
    size_{0},
    max_load_factor_{desired_load_factor},
    empty_key_sentinel_{empty_key_sentinel.value},
    empty_value_sentinel_{empty_value_sentinel.value},
    erased_key_sentinel_{erased_key_sentinel.value},
    allocator_{alloc}
{
  CUCO_EXPECTS(empty_key_sentinel_ != erased_key_sentinel_,
               "The empty key sentinel and erased key sentinel cannot be the same value.",
               std::runtime_error);
  CUCO_EXPECTS(capacity_ <= std::numeric_limits<std::uint32_t>::max(),
               "flat_hash_map capacity must not exceed UINT32_MAX.",
               std::runtime_error);

  slots_ = std::allocator_traits<slot_allocator_type>::allocate(allocator_, capacity_);

  auto constexpr block_size = 256;
  auto constexpr stride     = 4;
  auto const grid_size      = (capacity_ + stride * block_size - 1) / (stride * block_size);
  detail::initialize<block_size, atomic_key_type, atomic_mapped_type>
    <<<grid_size, block_size, 0, stream>>>(
      slots_, empty_key_sentinel_, empty_value_sentinel_, capacity_);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
flat_hash_map<Key, Value, Scope, Allocator>::~flat_hash_map()
{
  std::allocator_traits<slot_allocator_type>::deallocate(allocator_, slots_, capacity_);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
void flat_hash_map<Key, Value, Scope, Allocator>::insert(
  InputIt first, InputIt last, Hash hash, KeyEqual key_equal, cudaStream_t stream)
{
  auto const num_keys = cuda::std::distance(first, last);
  if (num_keys == 0) { return; }

  if (num_keys + size() >= static_cast<std::size_t>(capacity() * max_load_factor_)) {
    auto const new_capacity =
      std::max(2 * capacity(),
               2 * static_cast<std::size_t>(
                     std::ceil(static_cast<double>(num_keys + size()) / max_load_factor_)));
    this->rehash(new_capacity, hash, key_equal, stream);
  }

  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_mutable_view();

  using temp_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<char>;
  auto temp_allocator = temp_allocator_type{allocator_};
  auto d_num_out =
    reinterpret_cast<unsigned long long*>(std::allocator_traits<temp_allocator_type>::allocate(
      temp_allocator, sizeof(unsigned long long)));
  CUCO_CUDA_TRY(cudaMemsetAsync(d_num_out, 0, sizeof(unsigned long long), stream));

  detail::insert<block_size, tile_size>
    <<<grid_size, block_size, 0, stream>>>(first, num_keys, d_num_out, view, hash, key_equal);
  size_t h_num_out;
  CUCO_CUDA_TRY(
    cudaMemcpyAsync(&h_num_out, d_num_out, sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  std::allocator_traits<temp_allocator_type>::deallocate(
    temp_allocator, reinterpret_cast<char*>(d_num_out), sizeof(unsigned long long));
  size_ += h_num_out;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
void flat_hash_map<Key, Value, Scope, Allocator>::erase(
  InputIt first, InputIt last, Hash hash, KeyEqual key_equal, cudaStream_t stream)
{
  CUCO_EXPECTS(empty_key_sentinel() != erased_key_sentinel(),
               "You must provide a unique erased key sentinel value at map construction.",
               std::runtime_error);

  auto const num_keys = cuda::std::distance(first, last);
  if (num_keys == 0) { return; }

  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto constexpr tile_size  = 4;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view            = get_device_mutable_view();

  using temp_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<char>;
  auto temp_allocator = temp_allocator_type{allocator_};
  auto d_num_out =
    reinterpret_cast<unsigned long long*>(std::allocator_traits<temp_allocator_type>::allocate(
      temp_allocator, sizeof(unsigned long long)));
  CUCO_CUDA_TRY(cudaMemsetAsync(d_num_out, 0, sizeof(unsigned long long), stream));

  detail::erase<block_size, tile_size>
    <<<grid_size, block_size, 0, stream>>>(first, num_keys, d_num_out, view, hash, key_equal);
  size_t h_num_out;
  CUCO_CUDA_TRY(
    cudaMemcpyAsync(&h_num_out, d_num_out, sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  std::allocator_traits<temp_allocator_type>::deallocate(
    temp_allocator, reinterpret_cast<char*>(d_num_out), sizeof(unsigned long long));
  size_ -= h_num_out;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void flat_hash_map<Key, Value, Scope, Allocator>::find(InputIt first,
                                                       InputIt last,
                                                       OutputIt output_begin,
                                                       Hash hash,
                                                       KeyEqual key_equal,
                                                       cudaStream_t stream)
{
  auto const num_keys = cuda::std::distance(first, last);
  if (num_keys == 0) { return; }

  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  detail::find<block_size, tile_size, Value>
    <<<grid_size, block_size, 0, stream>>>(first, num_keys, output_begin, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename MaskOutIt, typename Hash, typename KeyEqual>
void flat_hash_map<Key, Value, Scope, Allocator>::find_and_mask(InputIt first,
                                                                InputIt last,
                                                                OutputIt output_begin,
                                                                MaskOutIt mask_begin,
                                                                Hash hash,
                                                                KeyEqual key_equal,
                                                                cudaStream_t stream)
{
  auto const num_keys = cuda::std::distance(first, last);
  if (num_keys == 0) { return; }
  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();
  detail::find_and_mask<block_size, tile_size, Value>
    <<<grid_size, block_size, 0, stream>>>(first, num_keys, output_begin, mask_begin, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename KeyOut, typename ValueOut>
std::pair<KeyOut, ValueOut> flat_hash_map<Key, Value, Scope, Allocator>::retrieve_all(
  KeyOut keys_out, ValueOut values_out, cudaStream_t stream) const
{
  static_assert(sizeof(pair_atomic_type) == sizeof(value_type));
  auto slots_begin = reinterpret_cast<value_type*>(slots_);

  auto begin =
    thrust::make_transform_iterator(slots_begin, cuco::detail::slot_to_tuple<Key, Value>{});
  auto const empty_key  = empty_key_sentinel();
  auto const erased_key = erased_key_sentinel();

  auto filled = detail::slot_is_filled<Key>{empty_key, erased_key};

#ifndef NV_PLATFORM
  // ROCm thrust doesn't support cuda::std::tuple, use thrust::tuple
  auto zipped_out_begin = thrust::make_zip_iterator(thrust::make_tuple(keys_out, values_out));
#else
  auto zipped_out_begin = thrust::make_zip_iterator(cuda::std::tuple{keys_out, values_out});
#endif

  std::size_t temp_storage_bytes = 0;
  using temp_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<char>;
  auto temp_allocator = temp_allocator_type{allocator_};
  auto d_num_out      = reinterpret_cast<std::size_t*>(
    std::allocator_traits<temp_allocator_type>::allocate(temp_allocator, sizeof(std::size_t)));
  cub::DeviceSelect::If(
    nullptr, temp_storage_bytes, begin, zipped_out_begin, d_num_out, capacity(), filled, stream);

  // Allocate temporary storage
  auto d_temp_storage =
    std::allocator_traits<temp_allocator_type>::allocate(temp_allocator, temp_storage_bytes);

  cub::DeviceSelect::If(d_temp_storage,
                        temp_storage_bytes,
                        begin,
                        zipped_out_begin,
                        d_num_out,
                        capacity(),
                        filled,
                        stream);

  std::size_t h_num_out;
  CUCO_CUDA_TRY(
    cudaMemcpyAsync(&h_num_out, d_num_out, sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  std::allocator_traits<temp_allocator_type>::deallocate(
    temp_allocator, reinterpret_cast<char*>(d_num_out), sizeof(std::size_t));
  std::allocator_traits<temp_allocator_type>::deallocate(
    temp_allocator, d_temp_storage, temp_storage_bytes);

  return std::make_pair(keys_out + h_num_out, values_out + h_num_out);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void flat_hash_map<Key, Value, Scope, Allocator>::contains(InputIt first,
                                                           InputIt last,
                                                           OutputIt output_begin,
                                                           Hash hash,
                                                           KeyEqual key_equal,
                                                           cudaStream_t stream) const
{
  auto const num_keys = cuda::std::distance(first, last);
  if (num_keys == 0) { return; }

  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  detail::contains<block_size, tile_size>
    <<<grid_size, block_size, 0, stream>>>(first, num_keys, output_begin, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
void flat_hash_map<Key, Value, Scope, Allocator>::rehash(Hash hash,
                                                         KeyEqual key_equal,
                                                         cudaStream_t stream)
{
  this->rehash(this->capacity(), hash, key_equal, stream);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
void flat_hash_map<Key, Value, Scope, Allocator>::rehash(size_t capacity,
                                                         Hash hash,
                                                         KeyEqual key_equal,
                                                         cudaStream_t stream)
{
  auto const old_slots    = this->slots_;
  auto const old_capacity = capacity_;
  capacity                = std::max(std::size_t{1}, capacity);
  // CG bulk probing supports at most the uint32_t slot-index domain.
  CUCO_EXPECTS(capacity <= std::numeric_limits<std::uint32_t>::max(),
               "flat_hash_map capacity must not exceed UINT32_MAX.",
               std::runtime_error);
  slots_    = std::allocator_traits<slot_allocator_type>::allocate(allocator_, capacity);
  capacity_ = capacity;

  auto constexpr init_block_size = 256;
  auto constexpr init_stride     = 4;
  auto const init_grid_size =
    (capacity_ + init_stride * init_block_size - 1) / (init_stride * init_block_size);
  detail::initialize<init_block_size, atomic_key_type, atomic_mapped_type>
    <<<init_grid_size, init_block_size, 0, stream>>>(
      slots_, empty_key_sentinel_, empty_value_sentinel_, capacity_);

  if (size_ == 0) {
    std::allocator_traits<slot_allocator_type>::deallocate(allocator_, old_slots, old_capacity);
    return;
  }

  auto const block_size = 128;
  auto const stride     = 1;
  auto const grid_size  = (old_capacity + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_mutable_view();

  detail::rehash<block_size>
    <<<grid_size, block_size, 0, stream>>>(old_slots, old_capacity, view, hash, key_equal);

  std::allocator_traits<slot_allocator_type>::deallocate(allocator_, old_slots, old_capacity);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename KeyEqual>
__device__ flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::insert_result
flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::packed_cas(
  iterator current_slot,
  value_type const& insert_pair,
  KeyEqual key_equal,
  Key expected_key) noexcept
{
  auto expected_value = this->empty_value_sentinel();

  cuco::detail::pair_converter<value_type> expected_pair{
    cuco::make_pair(expected_key, expected_value)};
  cuco::detail::pair_converter<value_type> new_pair{insert_pair};

  auto slot =
    reinterpret_cast<cuda::atomic<typename cuco::detail::pair_converter<value_type>::packed_type>*>(
      current_slot);

  bool success = slot->compare_exchange_strong(
    expected_pair.packed, new_pair.packed, cuda::std::memory_order_relaxed);
  if (success) {
    return insert_result::SUCCESS;
  }
  // duplicate present during insert
  else if (key_equal(insert_pair.first, expected_pair.pair.first)) {
    return insert_result::DUPLICATE;
  }

  return insert_result::CONTINUE;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename KeyEqual>
__device__ flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::insert_result
flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::back_to_back_cas(
  iterator current_slot,
  value_type const& insert_pair,
  KeyEqual key_equal,
  Key expected_key) noexcept
{
  using cuda::std::memory_order_relaxed;

  auto expected_value = this->empty_value_sentinel();

  // Back-to-back CAS for 8B/8B key/value pairs
  auto& slot_key   = current_slot->first;
  auto& slot_value = current_slot->second;

  bool key_success =
    slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);
  bool value_success =
    slot_value.compare_exchange_strong(expected_value, insert_pair.second, memory_order_relaxed);

  if (key_success) {
    while (not value_success) {
      value_success = slot_value.compare_exchange_strong(
        expected_value = this->empty_value_sentinel(), insert_pair.second, memory_order_relaxed);
    }
    return insert_result::SUCCESS;
  } else if (value_success) {
    slot_value.store(this->empty_value_sentinel(), memory_order_relaxed);
  }

  // our key was already present in the slot, so our key is a duplicate
  if (key_equal(insert_pair.first, expected_key)) { return insert_result::DUPLICATE; }

  return insert_result::CONTINUE;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename KeyEqual>
__device__ flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::insert_result
flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::cas_dependent_write(
  iterator current_slot,
  value_type const& insert_pair,
  KeyEqual key_equal,
  Key expected_key) noexcept
{
  using cuda::std::memory_order_relaxed;

  auto& slot_key = current_slot->first;

  auto const key_success =
    slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);

  if (key_success) {
    auto& slot_value = current_slot->second;
    slot_value.store(insert_pair.second, memory_order_relaxed);
    return insert_result::SUCCESS;
  }

  // our key was already present in the slot, so our key is a duplicate
  if (key_equal(insert_pair.first, expected_key)) { return insert_result::DUPLICATE; }

  return insert_result::CONTINUE;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ bool flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::insert(
  value_type const& insert_pair, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot{this->initial_slot(insert_pair.first, hash)};

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_available =
      cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel()) or
      cuco::detail::bitwise_compare(existing_key, this->erased_key_sentinel());

    // the key we are trying to insert is already in the map, so we return with failure to insert
    if (not slot_is_available and key_equal(existing_key, insert_pair.first)) { return false; }

    if (slot_is_available) {
      auto const status = [&]() {
        // One single CAS operation if `value_type` is packable
        if constexpr (cuco::detail::is_packable<value_type>()) {
          return packed_cas(current_slot, insert_pair, key_equal, existing_key);
        }

        if constexpr (not cuco::detail::is_packable<value_type>()) {
#if (__CUDA_ARCH__ < 700)
          return cas_dependent_write(current_slot, insert_pair, key_equal, existing_key);
#else
          return back_to_back_cas(current_slot, insert_pair, key_equal, existing_key);
#endif
        }
      }();

      // successful insert
      if (status == insert_result::SUCCESS) { return true; }
      // duplicate present during insert
      if (status == insert_result::DUPLICATE) { return false; }
    }

    // if we couldn't insert the key, but it wasn't a duplicate, then there must
    // have been some other key there, so we keep looking for a slot
    current_slot = this->next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ cuda::std::
  pair<typename flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::iterator, bool>
  flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::insert_and_find(
    value_type const& insert_pair, Hash hash, KeyEqual key_equal) noexcept
{
#if __CUDA_ARCH__ < 700
  // Spinning to ensure that the write to the value part took place requires
  // independent thread scheduling introduced with the Volta architecture.
  static_assert(cuco::detail::is_packable<value_type>(),
                "insert_and_find is not supported for unpackable data on pre-Volta GPUs.");
#endif

  auto current_slot{this->initial_slot(insert_pair.first, hash)};

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_available =
      cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel()) or
      cuco::detail::bitwise_compare(existing_key, this->erased_key_sentinel());

    // the key we are trying to insert is already in the map, so we return with failure to insert
    if (not slot_is_available and key_equal(existing_key, insert_pair.first)) {
      // If we cannot use a single CAS operation, ensure that the write to
      // the value part also took place.
      if constexpr (not cuco::detail::is_packable<value_type>()) {
        auto& slot_value       = current_slot->second;
        auto const empty_value = this->empty_value_sentinel();
        while (cuco::detail::bitwise_compare(slot_value.load(cuda::std::memory_order_relaxed),
                                             empty_value)) {
          // spin
        }
      }

      return cuda::std::pair{current_slot, false};
    }

    if (slot_is_available) {
      auto const status = [&]() {
        // One single CAS operation if `value_type` is packable
        if constexpr (cuco::detail::is_packable<value_type>()) {
          return packed_cas(current_slot, insert_pair, key_equal, existing_key);
        }

        if constexpr (not cuco::detail::is_packable<value_type>()) {
          // Only use cas_dependent_write; for back_to_back_cas we cannot
          // guarantee that we get a valid iterator: Consider the case of two
          // threads inserting the same key, and one gets the key while the
          // other gets the value. For a third thread, the entry looks valid,
          // but the second thread will first reset the value to the empty
          // sentinel to signal that the first thread can write its value.
          // This ambiguity cannot be solved for the third thread, so we have
          // to avoid it.
          return cas_dependent_write(current_slot, insert_pair, key_equal, existing_key);
        }
      }();

      // successful insert
      if (status == insert_result::SUCCESS) {
        // This thread did the insertion, so the iterator is guaranteed to be
        // valid without any special care.
        return cuda::std::pair{current_slot, true};
      }
      // duplicate present during insert
      if (status == insert_result::DUPLICATE) {
        // If we cannot use a single CAS operation, ensure that the write to
        // the value part also took place.
        if constexpr (not cuco::detail::is_packable<value_type>()) {
          auto& slot_value       = current_slot->second;
          auto const empty_value = this->empty_value_sentinel();
          while (cuco::detail::bitwise_compare(slot_value.load(cuda::std::memory_order_relaxed),
                                               empty_value)) {
            // spin
          }
        }

        return cuda::std::pair{current_slot, false};
      }
    }

    // if we couldn't insert the key, but it wasn't a duplicate, then there must
    // have been some other key there, so we keep looking for a slot
    current_slot = this->next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ bool flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::insert(
  CG g, value_type const& insert_pair, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot = this->initial_slot(g, insert_pair.first, hash);

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_available =
      cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel()) or
      cuco::detail::bitwise_compare(existing_key, this->erased_key_sentinel());

    // the key we are trying to insert is already in the map, so we return with failure to insert
    if (g.any(not slot_is_available and key_equal(existing_key, insert_pair.first))) {
      return false;
    }

    auto const bucket_contains_available = g.ballot(slot_is_available);

    // we found an empty slot, but not the key we are inserting, so this must
    // be an empty slot into which we can insert the key
    if (bucket_contains_available) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      uint32_t src_lane = __ffs(static_cast<int>(bucket_contains_available)) - 1;

      if (g.thread_rank() == src_lane) {
        // One single CAS operation if `value_type` is packable
        if constexpr (cuco::detail::is_packable<value_type>()) {
          status = packed_cas(current_slot, insert_pair, key_equal, existing_key);
        }
        // Otherwise, two back-to-back CAS operations
        else {
#if (__CUDA_ARCH__ < 700)
          status = cas_dependent_write(current_slot, insert_pair, key_equal, existing_key);
#else
          status = back_to_back_cas(current_slot, insert_pair, key_equal, existing_key);
#endif
        }
      }

      uint32_t res_status = g.shfl(static_cast<uint32_t>(status), src_lane);
      status              = static_cast<insert_result>(res_status);

      // successful insert
      if (status == insert_result::SUCCESS) { return true; }
      // duplicate present during insert
      if (status == insert_result::DUPLICATE) { return false; }
      // if we've gotten this far, a different key took our spot
      // before we could insert. We need to retry the insert on the
      // same bucket
    }
    // if there are no empty slots in the current bucket,
    // we move onto the next bucket
    else {
      current_slot = this->next_slot(g, current_slot);
    }
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ bool flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::erase(
  key_type const& k, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot{this->initial_slot(k, hash)};
  auto const init_slot = current_slot;

  value_type const insert_pair =
    make_pair<Key, Value>(this->erased_key_sentinel(), this->empty_value_sentinel());

  while (true) {
    static_assert(sizeof(Key) == sizeof(atomic_key_type));
    static_assert(sizeof(Value) == sizeof(atomic_mapped_type));
    // TODO: Replace reinterpret_cast with atomic ref when available.
    value_type slot_contents = *reinterpret_cast<value_type const*>(current_slot);
    auto existing_key        = slot_contents.first;
    auto existing_value      = slot_contents.second;

    // Key doesn't exist, return false
    if (cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel())) { return false; }

    // Key exists, return true if successfully deleted
    if (key_equal(existing_key, k)) {
      if constexpr (cuco::detail::is_packable<value_type>()) {
        auto slot = reinterpret_cast<
          cuda::atomic<typename cuco::detail::pair_converter<value_type>::packed_type>*>(
          current_slot);
        cuco::detail::pair_converter<value_type> expected_pair{
          cuco::make_pair(existing_key, existing_value)};
        cuco::detail::pair_converter<value_type> new_pair{insert_pair};

        return slot->compare_exchange_strong(
          expected_pair.packed, new_pair.packed, cuda::std::memory_order_relaxed);
      }
      if constexpr (not cuco::detail::is_packable<value_type>()) {
        current_slot->second.compare_exchange_strong(
          existing_value, insert_pair.second, cuda::std::memory_order_relaxed);
        return current_slot->first.compare_exchange_strong(
          existing_key, insert_pair.first, cuda::std::memory_order_relaxed);
      }
    }

    current_slot = this->next_slot(current_slot);
    // if all keys in this map has been erased, return false
    if (current_slot == init_slot) { return false; }
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ bool flat_hash_map<Key, Value, Scope, Allocator>::device_mutable_view::erase(
  CG g, key_type const& k, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot    = this->initial_slot(g, k, hash);
  auto const init_slot = current_slot;
  value_type const insert_pair =
    make_pair<Key, Value>(this->erased_key_sentinel(), this->empty_value_sentinel());

  while (true) {
    static_assert(sizeof(Key) == sizeof(atomic_key_type));
    static_assert(sizeof(Value) == sizeof(atomic_mapped_type));
    // TODO: Replace reinterpret_cast with atomic ref when available.
    value_type slot_contents = *reinterpret_cast<value_type const*>(current_slot);
    auto existing_key        = slot_contents.first;
    auto existing_value      = slot_contents.second;

    auto const slot_is_empty =
      cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel());

    auto const exists = g.ballot(not slot_is_empty and key_equal(existing_key, k));

    // Key exists, return true if successfully deleted
    if (exists) {
      uint32_t src_lane = __ffs(static_cast<int>(exists)) - 1;

      // All lanes execute shfl below; only src_lane writes the result.
      // Initialize the other lanes' local value so the cast passed to shfl is defined.
      bool status{false};
      if (g.thread_rank() == src_lane) {
        if constexpr (cuco::detail::is_packable<value_type>()) {
          auto slot = reinterpret_cast<
            cuda::atomic<typename cuco::detail::pair_converter<value_type>::packed_type>*>(
            current_slot);
          cuco::detail::pair_converter<value_type> expected_pair{
            cuco::make_pair(existing_key, existing_value)};
          cuco::detail::pair_converter<value_type> new_pair{insert_pair};

          status = slot->compare_exchange_strong(
            expected_pair.packed, new_pair.packed, cuda::std::memory_order_relaxed);
        }
        if constexpr (not cuco::detail::is_packable<value_type>()) {
          current_slot->second.compare_exchange_strong(
            existing_value, insert_pair.second, cuda::std::memory_order_relaxed);
          status = current_slot->first.compare_exchange_strong(
            existing_key, insert_pair.first, cuda::std::memory_order_relaxed);
        }
      }

      uint32_t res_status = g.shfl(static_cast<uint32_t>(status), src_lane);
      return static_cast<bool>(res_status);
    }

    // empty slot found, but key not found, must not be in the map
    if (g.ballot(slot_is_empty)) { return false; }

    current_slot = this->next_slot(g, current_slot);
    if (current_slot == init_slot) { return false; }
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ typename flat_hash_map<Key, Value, Scope, Allocator>::device_view::iterator
flat_hash_map<Key, Value, Scope, Allocator>::device_view::find(Key const& k,
                                                               Hash hash,
                                                               KeyEqual key_equal) noexcept
{
  auto current_slot    = this->initial_slot(k, hash);
  auto const init_slot = current_slot;

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    // Key doesn't exist, return end()
    if (cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel())) {
      return this->end();
    }

    // Key exists, return iterator to location
    if (key_equal(existing_key, k)) { return current_slot; }

    current_slot = this->next_slot(current_slot);
    if (current_slot == init_slot) { return this->end(); }
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ typename flat_hash_map<Key, Value, Scope, Allocator>::device_view::const_iterator
flat_hash_map<Key, Value, Scope, Allocator>::device_view::find(Key const& k,
                                                               Hash hash,
                                                               KeyEqual key_equal) const noexcept
{
  auto current_slot    = this->initial_slot(k, hash);
  auto const init_slot = current_slot;

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    // Key doesn't exist, return end()
    if (cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel())) {
      return this->end();
    }

    // Key exists, return iterator to location
    if (key_equal(existing_key, k)) { return current_slot; }

    current_slot = this->next_slot(current_slot);
    if (current_slot == init_slot) { return this->end(); }
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ typename flat_hash_map<Key, Value, Scope, Allocator>::device_view::iterator
flat_hash_map<Key, Value, Scope, Allocator>::device_view::find(CG g,
                                                               Key const& k,
                                                               Hash hash,
                                                               KeyEqual key_equal) noexcept
{
  auto current_slot    = this->initial_slot(g, k, hash);
  auto const init_slot = current_slot;

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as
    // the sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty =
      cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel());

    // the key we were searching for was found by one of the threads,
    // so we return an iterator to the entry
    auto const exists = g.ballot(not slot_is_empty and key_equal(existing_key, k));
    if (exists) {
      uint32_t src_lane = __ffs(static_cast<int>(exists)) - 1;
      // TODO: This shouldn't cast an iterator to an int to shuffle. Instead, get the index of the
      // current_slot and shuffle that instead.
      intptr_t res_slot = g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane);
      return reinterpret_cast<iterator>(res_slot);
    }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.ballot(slot_is_empty)) { return this->end(); }

    // otherwise, all slots in the current bucket are full with other keys, so we move onto the
    // next bucket
    current_slot = this->next_slot(g, current_slot);
    if (current_slot == init_slot) { return this->end(); }
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ typename flat_hash_map<Key, Value, Scope, Allocator>::device_view::const_iterator
flat_hash_map<Key, Value, Scope, Allocator>::device_view::find(CG g,
                                                               Key const& k,
                                                               Hash hash,
                                                               KeyEqual key_equal) const noexcept
{
  auto current_slot    = this->initial_slot(g, k, hash);
  auto const init_slot = current_slot;

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as
    // the sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty =
      cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel());

    // the key we were searching for was found by one of the threads, so we return an iterator to
    // the entry
    auto const exists = g.ballot(not slot_is_empty and key_equal(existing_key, k));
    if (exists) {
      uint32_t src_lane = __ffs(static_cast<int>(exists)) - 1;
      // TODO: This shouldn't cast an iterator to an int to shuffle. Instead, get the index of the
      // current_slot and shuffle that instead.
      intptr_t res_slot = g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane);
      return reinterpret_cast<const_iterator>(res_slot);
    }

    // we found an empty slot, meaning that the key we're searching
    // for isn't in this submap, so we should move onto the next one
    if (g.ballot(slot_is_empty)) { return this->end(); }

    // otherwise, all slots in the current bucket are full with other keys,
    // so we move onto the next bucket in the current submap

    current_slot = this->next_slot(g, current_slot);
    if (current_slot == init_slot) { return this->end(); }
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename ProbeKey, typename Hash, typename KeyEqual>
__device__ bool flat_hash_map<Key, Value, Scope, Allocator>::device_view::contains(
  ProbeKey const& k, Hash hash, KeyEqual key_equal) const noexcept
{
  auto current_slot    = this->initial_slot(k, hash);
  auto const init_slot = current_slot;

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    if (cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel_)) { return false; }

    if (key_equal(existing_key, k)) { return true; }

    current_slot = this->next_slot(current_slot);
    if (current_slot == init_slot) { return false; }
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename ProbeKey, typename Hash, typename KeyEqual>
__device__ cuda::std::enable_if_t<std::is_invocable_v<KeyEqual, ProbeKey, Key>, bool>
flat_hash_map<Key, Value, Scope, Allocator>::device_view::contains(
  CG g, ProbeKey const& k, Hash hash, KeyEqual key_equal) const noexcept
{
  auto current_slot    = this->initial_slot(g, k, hash);
  auto const init_slot = current_slot;

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as
    // the sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty =
      cuco::detail::bitwise_compare(existing_key, this->empty_key_sentinel());

    // the key we were searching for was found by one of the threads, so we return an iterator to
    // the entry
    if (g.ballot(not slot_is_empty and key_equal(existing_key, k))) { return true; }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.ballot(slot_is_empty)) { return false; }

    // otherwise, all slots in the current bucket are full with other keys, so we move onto the
    // next bucket
    current_slot = this->next_slot(g, current_slot);
    if (current_slot == init_slot) { return false; }
  }
}
}  // namespace cuco

#ifdef USE_ROCM
// Clean up HIP compatibility macros to avoid polluting other files
#undef cudaStream_t
#undef cudaStreamSynchronize
#undef cudaMemsetAsync
#undef cudaMemcpyAsync
#undef cudaMemcpyDeviceToHost
#undef cudaMemcpyHostToDevice
#endif
