#include <mutex>
#include <unordered_map>
#include <queue>

#ifndef CPU_ONLY
#ifdef USE_ROCM
#include <hipcub/util_allocator.hpp>
#else
#include <cub/util_allocator.cuh>
#endif
#endif // CPU_ONLY

#include "absl/log/check.h"
#include "column-io/framework/gpu_runtime.h"
#include "column-io/framework/allocator.h"
#include "column-io/framework/cuda_utils.h"

#ifndef CPU_ONLY
#define CUDA_CHECK(EXPR)                                       \
  do {                                                         \
    const cudaError_t __err = EXPR;                            \
    CHECK_EQ(__err, cudaSuccess) << cudaGetErrorString(__err); \
  } while (0)
#endif

namespace column {

#ifndef CPU_ONLY
#ifdef USE_ROCM
namespace cub = hipcub;
#endif
#endif

namespace {

template <typename T, std::size_t SizeOfT>
struct LeadingZerosCounter {
  static std::size_t count(T Val) {
    if (!Val)
      return std::numeric_limits<T>::digits;

    // Bisection method.
    std::size_t ZeroBits = 0;
    for (T Shift = std::numeric_limits<T>::digits >> 1; Shift; Shift >>= 1) {
      T Tmp = Val >> Shift;
      if (Tmp)
        Val = Tmp;
      else
        ZeroBits |= Shift;
    }
    return ZeroBits;
  }
};

template <typename T>
struct LeadingZerosCounter<T, 4> {
  static std::size_t count(T Val) {
    if (Val == 0)
      return 32;
    return __builtin_clz(Val);
  }
};

template <typename T>
struct LeadingZerosCounter<T, 8> {
  static std::size_t count(T Val) {
    if (Val == 0)
      return 64;
    return __builtin_clzll(Val);
  }
};

/// Return the ceil log base 2 of the specified value, 64 if the value is zero.
/// (64 bit edition.)
inline unsigned Log2_64_Ceil(uint64_t Value) {
  return static_cast<unsigned>(64 - LeadingZerosCounter<uint64_t, sizeof(uint64_t)>::count(Value - 1));
}

/// Returns the next power of two (in 64-bits) that is strictly greater than A.
/// Returns zero on overflow.
inline uint64_t NextPowerOf2(uint64_t A) {
  A |= (A >> 1);
  A |= (A >> 2);
  A |= (A >> 4);
  A |= (A >> 8);
  A |= (A >> 16);
  A |= (A >> 32);
  return A + 1;
}

/// Returns the power of two which is greater than or equal to the given value.
/// Essentially, it is a ceil operation across the domain of powers of two.
inline uint64_t PowerOf2Ceil(uint64_t A) {
  if (!A)
    return 0;
  return NextPowerOf2(A - 1);
}

struct HostBlock {
  // constructor for search key
  HostBlock(size_t size) : size_(size) {}

  HostBlock(size_t size, void* ptr) : size_(size), ptr_(ptr) {}

  std::mutex mutex_;
  size_t size_{0}; // block size in bytes
  void* ptr_{nullptr}; // memory address
  bool allocated_{false}; // in-use flag
};

template <typename B>
struct alignas(64) FreeBlockList {
  std::mutex mutex_;
  std::deque<B*> list_;
};

template <typename B = HostBlock>
class CachingHostAllocatorImpl {
public:
  virtual ~CachingHostAllocatorImpl() {}
  // return data_ptr and block pair.
  virtual std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    // Round up the allocation to the nearest power of two to improve reuse.
    // These power of two sizes are also used to index into the free list.
    size_t roundSize = PowerOf2Ceil(size);

    // First, try to allocate from the free list
    auto* block = get_free_block(roundSize);
    if (block) {
      return {block->ptr_, reinterpret_cast<void*>(block)};
    }

    // Slow path: if we can't allocate from the cached free list, we need
    // to create a new block.
    void* ptr = nullptr;
    allocate_host_memory(roundSize, &ptr);

    // Then, create a new block.
    block = new B(roundSize, ptr);
    block->allocated_ = true;

    return {block->ptr_, reinterpret_cast<void*>(block)};
  }

  virtual void free(void* ctx) {
    if (!ctx) {
      return;
    }

    // Note: we can assume that free is correctly paired with alloc, and thus we
    // do not need to look up the ctx in blocks_.
    auto* block = reinterpret_cast<B*>(ctx);

    {
      std::lock_guard<std::mutex> g(block->mutex_);
      block->allocated_ = false;
    }

    auto index = size_index(block->size_);
    {
      std::lock_guard<std::mutex> g(free_list_[index].mutex_);
      free_list_[index].list_.push_back(block);
    }
  }

  virtual void empty_cache() {
    // Remove all elements from the free list, remove them from the blocks
    // list, and free the associated pinned memory allocation. This requires
    // concurrently holding both the free list mutexes and the blocks mutex, and
    // is the only function that concurrently holds multiple mutexes.
    for (size_t i = 0; i < free_list_.size(); ++i) {
      std::lock(free_list_[i].mutex_, blocks_mutex_);
      std::lock_guard<std::mutex> gf(free_list_[i].mutex_, std::adopt_lock);
      std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

      std::vector<B*> blocks_to_remove(free_list_[i].list_.begin(), free_list_[i].list_.end());
      free_list_[i].list_.clear();
      for (auto* block : blocks_to_remove) {
        free_block(block);
        delete block;
      }
    }
  }

  inline size_t size_index(size_t size) {
    return Log2_64_Ceil(size);
  }

private:
  virtual B* get_free_block(size_t size) {
    auto index = size_index(size);
    std::lock_guard<std::mutex> g(free_list_[index].mutex_);
    if (free_list_[index].list_.size() > 0) {
      B* block = free_list_[index].list_.back();
      free_list_[index].list_.pop_back();
      block->allocated_ = true;
      return block;
    }
    return nullptr;
  }

  /* These following functions are runtime-related. */

  // Allocate page-locked memory on the host.
  virtual void allocate_host_memory(size_t size, void** ptr) = 0;

  // Free block and release the pointer contained in block.
  virtual void free_block(B* block) = 0;

  alignas(64) std::mutex blocks_mutex_;

  // We keep free list as a vector of free lists, one for each power of two
  // size. This allows us to quickly find a free block of the right size.
  // We use deque to store per size free list and guard the list with its own
  // mutex.
  alignas(64) std::vector<FreeBlockList<B>> free_list_ = std::vector<FreeBlockList<B>>(64);
};

#ifndef CPU_ONLY
class CUDACachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<HostBlock> {
public:
  CUDACachingHostAllocatorImpl() {
	unsigned int device_flags;
	CUDA_CHECK(cudaGetDeviceFlags(&device_flags));
	if (device_flags & cudaDeviceMapHost) {
	  flag_ = cudaHostAllocMapped;
	} else {
	  flag_ = cudaHostAllocDefault;
	}
  }
private:
  void allocate_host_memory(size_t size, void** ptr) override {
    // Use cudaHostAlloc for allocating pinned memory (global lock in driver)
    CUDA_CHECK(cudaHostAlloc(ptr, size, flag_));
  }

  void free_block(HostBlock* block) override {
    CUDA_CHECK(cudaFreeHost(block->ptr_));
  }

  unsigned int flag_;
};
#else
class MallocCachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<HostBlock> {
private:
  void allocate_host_memory(size_t size, void** ptr) override {
    *ptr = malloc(size);
  }
  void free_block(HostBlock* block) override {
    ::free(block->ptr_);
  }
};
#endif // CPU_ONLY
}  // namespace

class MallocAllocator : public column::Allocator {
public:
  std::pair<void*, void*> Allocate(size_t size) override {
    void* ptr = malloc(size);
	return {ptr, ptr};
  }
  void Deallocate(void* ctx) override {
	free(ctx);
  }
};

#ifndef CPU_ONLY
class PinnedMemoryAllocator : public Allocator {
public:
  std::pair<void*, void*> Allocate(size_t size) override {
    return impl_.allocate(size);
  }
  void Deallocate(void* ctx) override {
    impl_.free(ctx);
  }
private:
  CUDACachingHostAllocatorImpl impl_;
};
#else
class PinnedMemoryAllocator : public Allocator {
public:
  std::pair<void*, void*> Allocate(size_t size) override {
    return impl_.allocate(size);
  }
  void Deallocate(void* ctx) override {
    impl_.free(ctx);
  }
private:
  MallocCachingHostAllocatorImpl impl_;
};
#endif // CPU_ONLY

Allocator* GetAllocator(bool pin) {
  if (pin) {
    static PinnedMemoryAllocator a;
    return &a;
  } else {
    static MallocAllocator a;
    return &a;
  }
}

#ifndef CPU_ONLY
class CudaAllocator : public Allocator {
 public:
  CudaAllocator(
      cub::CachingDeviceAllocator& a,
      cudaStream_t stream,
      int device_id): a_(a), stream_(stream), device_id_(device_id) {}
  std::pair<void*, void*> Allocate(size_t size) override {
    void* ret;
    GPU_CK(a_.DeviceAllocate(device_id_, &ret, size, stream));
    return {ret, ret};
  }
  void Deallocate(void* ctx) override {
    GPU_CK(a_.DeviceFree(device_id_, ctx));
  }
 private:
  cub::CachingDeviceAllocator& a_;
  cudaStream_t stream_;
  int device_id_;
};

Allocator* CreateCudaAllocator(cudaStream_t stream, int device_id) {
  static cub::CachingDeviceAllocator a((unsigned int)8, (unsigned int)3);
  return new CudaAllocator(a, stream, device_id);
}
#else
Allocator* CreateCudaAllocator(cudaStream_t, int) {
  return GetAllocator(false);
}
#endif // CPU_ONLY
}  // namespace column
