#ifndef _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_GPU_RUNTIME_H_
#define _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_GPU_RUNTIME_H_

// GPU Runtime Abstraction Layer
// This header provides a unified interface for both CUDA and ROCm/HIP

#ifdef CPU_ONLY
// CPU-only stub: provide minimal type definitions so code compiles without GPU
typedef int cudaError_t;
typedef void* cudaStream_t;
struct cudaDeviceProp { int multiProcessorCount; };
constexpr cudaError_t cudaSuccess = 0;

inline const char* cudaGetErrorName(cudaError_t) { return "cudaErrorNoDevice"; }
inline const char* cudaGetErrorString(cudaError_t) { return "GPU disabled (CPU_ONLY build)"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }

// Memory stubs (should not be called in CPU_ONLY mode)
inline cudaError_t cudaMalloc(void**, size_t) { return 1; }
inline cudaError_t cudaMallocHost(void** p, size_t s) { *p = malloc(s); return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return 1; }
inline cudaError_t cudaFreeHost(void* p) { free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return 1; }
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, int, cudaStream_t) { return 1; }
constexpr int cudaMemcpyHostToDevice = 1;
constexpr int cudaMemcpyDeviceToHost = 2;
constexpr int cudaMemcpyDeviceToDevice = 3;
constexpr int cudaMemcpyDefault = 4;
inline cudaError_t cudaMemset(void*, int, size_t) { return 1; }
inline cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t) { return 1; }
constexpr int cudaDeviceMapHost = 0;
constexpr int cudaHostAllocMapped = 0;
constexpr int cudaHostAllocDefault = 0;
inline cudaError_t cudaHostAlloc(void** p, size_t s, unsigned int) { *p = malloc(s); return cudaSuccess; }

// Stream stubs
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

// Device stubs
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int*) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceFlags(unsigned int* f) { *f = 0; return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* c) { *c = 0; return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

#elif defined(USE_ROCM)
// ROCm/HIP includes
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// Map CUDA types to HIP types
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaStream_t hipStream_t
#define cudaDeviceProp hipDeviceProp_t

// Map CUDA functions to HIP functions
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

// Memory management
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaDeviceMapHost hipDeviceMapHost
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostAlloc hipHostAlloc
#define cudaHostAllocDefault hipHostAllocDefault

// Stream management
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamNonBlocking hipStreamNonBlocking

// Device management
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceFlags hipGetDeviceFlags
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset


#else
#include <cuda_runtime_api.h>
#endif


#endif // _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_GPU_RUNTIME_H_
