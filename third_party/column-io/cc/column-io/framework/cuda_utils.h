#ifndef _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_CUDA_UTILS_H_
#define _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_CUDA_UTILS_H_

#include <iostream>
#include <chrono>
#include <mutex>
#include "column-io/framework/gpu_runtime.h"

namespace column {

#ifdef CPU_ONLY
#define GPU_CK(call) (void)0

inline int GetCudaDeviceId() { return 0; }

#else

#define GPU_CK(call) {                              \
    cudaError_t err = call;                         \
    if (err != cudaSuccess) {                       \
        printf("CUDA Error at %s:%d code=%d(%s)\n", \
               __FILE__, __LINE__,                  \
               static_cast<int>(err),               \
               cudaGetErrorName(err));              \
        exit(EXIT_FAILURE);                         \
    }                                               \
}

inline int GetCudaDeviceId() {
  const char* local_rank = std::getenv("LOCAL_RANK");
  if (local_rank) {
    return std::stoi(local_rank);
  }
  // LOG(WARNING) << "LOCAL_RANK not set, defaulting to 0"; //TODO: use a internal log library. not absl log
  auto time_chrono = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  char time_buf[64];
  std::strftime(time_buf, sizeof(time_buf), "%Y%m%d %H%M%S", std::localtime(&time_chrono));
  printf("[%s] [WARN] [%s:%d] LOCAL_RANK not set, defaulting to 0\n",
      time_buf, __FILE__, __LINE__);
  return 0;
}
#endif // CPU_ONLY

#define CUDA_LOOP(i, size) \
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x)

template <typename T>
constexpr T Align256(T x) {
  return (x + 256 - 1) / 256 * 256;
}

template <typename T>
constexpr T DivRoundUp(T a, T b) {
  return (a + b - 1) / b;
}

constexpr uint32_t kBlockSize = 256;

inline uint32_t GetNumBlocks(uint32_t size) {
  uint32_t num_blocks = DivRoundUp(size, kBlockSize);
  // cudaDeviceProp prop;
  // GPU_CK(cudaGetDeviceProperties(&prop, 0));
  // uint32_t max_blocks = prop.multiProcessorCount * 8;
  uint32_t max_blocks = 80 * 8;
  return num_blocks > max_blocks ? max_blocks : num_blocks; 
}

static cudaStream_t stream = 0;
static std::mutex mu;
inline cudaStream_t GetCopyStream() {
  std::lock_guard<std::mutex> l(mu);
  if (stream) {
    return stream;
  } else {
    GPU_CK(cudaStreamCreate(&stream));
    return stream;
  }
}

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;

    Timer() : start_time(Clock::now()) {}

    void start() {
      start_time = Clock::now();
    }

    void reset() {
        start_time = Clock::now();
    }

    std::chrono::nanoseconds elapsed_nanoseconds() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(total_duration<std::chrono::nanoseconds>());
    }

    std::chrono::microseconds elapsed_microseconds() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(total_duration<std::chrono::microseconds>());
    }

    std::chrono::milliseconds elapsed_milliseconds() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(total_duration<std::chrono::milliseconds>());
    }

    void print_elapsed(const std::string& msg = "Elapsed time") {
#ifndef NDEBUG
        auto ms = elapsed_milliseconds().count();
        auto us = elapsed_microseconds().count() % 1000;
        auto ns = elapsed_nanoseconds().count() % 1000;
        std::cout << msg << ": "
                  << ms << " ms, "
                  << us << " μs, "
                  << ns << " ns\n";
        reset();
#endif
    }

private:
    Clock::time_point start_time;

    template <typename T>
    T total_duration() const {
      return std::chrono::duration_cast<T>(Clock::now() - start_time);
    }
};


}  // namespace column

#endif   // _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_CUDA_UTILS_H_

