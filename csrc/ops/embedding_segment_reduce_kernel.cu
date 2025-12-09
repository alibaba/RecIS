#include <ATen/cuda/CUDAContext.h>
#include <c10/core/TensorImpl.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdio>
#include <cub/block/block_reduce.cuh>
#include <vector>

#include "cuda/atomic_fast.cuh"
#include "cuda/cuda_param.cuh"
#include "cuda/element_wise_kernel.cuh"
#include "cuda/packer.cuh"
#include "cuda/utils.cuh"
#include "embedding_segment_reduce.h"
#include "ops/embedding_segment_reduce.h"
namespace recis {
namespace functional {
using namespace recis::cuda;

template <typename scalar_t, typename offset_t, ReduceMode mode,
          bool USE_WEIGHT, int PACK_SIZE>
__global__ void segment_reduce_forward_kernel(
    const scalar_t* __restrict__ unique_emb,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ reverse_indices,
    const offset_t* __restrict__ offsets, scalar_t* output, int64_t B,
    int64_t N, int64_t S, int64_t D) {
  using AP = Packer<scalar_t, PACK_SIZE>;
  using BlockReduce = cub::BlockReduce<scalar_t, 256>;
  // These shared variables will be optimized out by the compiler
  // if we are not computing weighted mean.
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ scalar_t s_weight_sum;

  for (int s = blockIdx.x; s < S - 1; s += gridDim.x) {
    offset_t start = offsets[s];
    offset_t end = offsets[s + 1];
    int64_t length = end - start;
    int64_t total_size = length * D;

    scalar_t weight_sum = 0;
    if constexpr (USE_WEIGHT && mode == ReduceMode::MEAN) {
      // The loop ending condition should be `i / blockDim.x * blockDim.x <
      // length` instead of `i < length`, because the last block may have some
      // threads not doing BlockReduce in the condition `i < length`, which
      // leads to deadlock.
      for (int64_t i = threadIdx.x; i / blockDim.x * blockDim.x < length;
           i += blockDim.x) {
        scalar_t w = 0;
        if (i < length) {
          w = weight[start + i];
        }
        scalar_t res = BlockReduce(temp_storage).Sum(w);
        // Only thread 0 has the reduce sum result.
        if (threadIdx.x == 0) {
          weight_sum += res;
        }
        // NOTE: A subsequent __syncthreads() threadblock barrier should be
        // invoked after calling BlockReduce if the collective’s temporary
        // storage (e.g., temp_storage) is to be reused or repurposed.
        __syncthreads();
      }
      // Copy the result to the shared memory.
      if (threadIdx.x == 0) {
        s_weight_sum = weight_sum;
      }
      __syncthreads();
      // Broadcast the result to all threads.
      weight_sum = s_weight_sum;
    }

    for (int64_t i_base = threadIdx.x; i_base * PACK_SIZE < total_size;
         i_base += blockDim.x) {
      int64_t i = i_base * PACK_SIZE;
      int64_t idx = i / D + start;
      int64_t dp = i % D;

      int64_t raw_idx = reverse_indices[idx];
      scalar_t w = 1;
      if constexpr (USE_WEIGHT) {
        w = weight[idx];
      } else {
        weight_sum = static_cast<scalar_t>(length);
      }
      if constexpr (mode == ReduceMode::MEAN) {
        w = w / weight_sum;
      }

      typename AP::type a_vec;
      typename AP::type b_vec;
      AP::load(unique_emb + raw_idx * D + dp, a_vec);

#pragma unroll
      for (int j = 0; j < PACK_SIZE; j++) {
        auto a_val = AP::get_element(a_vec, j);
        auto res = a_val * w;
        AP::set_element(b_vec, j, res);
      }

      if constexpr (mode == ReduceMode::TILE) {
        AP::store(output + idx * D + dp, b_vec);
      } else {
#pragma unroll
        for (int j = 0; j < PACK_SIZE; j++) {
          scalar_t val = AP::get_element(b_vec, j);
          int64_t index = dp + j;
          atomic_add_custom<scalar_t>(&output[s * D + index], val);
        }
      }
    }
  }
}

template <typename scalar_t, typename offset_t, ReduceMode mode,
          bool USE_WEIGHT, int PACK_SIZE>
__global__ void segment_reduce_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ reverse_indices,
    const offset_t* __restrict__ offsets, scalar_t* grad_unique_emb, int64_t B,
    int64_t N, int64_t S, int64_t D) {
  using AP = Packer<scalar_t, PACK_SIZE>;
  using BlockReduce = cub::BlockReduce<scalar_t, 256>;
  // These shared variables will be optimized out by the compiler
  // if we are not computing weighted mean.
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ scalar_t s_weight_sum;

  for (int64_t s = blockIdx.x; s < S - 1; s += gridDim.x) {
    offset_t start = offsets[s];
    offset_t end = offsets[s + 1];
    int64_t length = end - start;

    scalar_t weight_sum = 0;
    if constexpr (USE_WEIGHT && mode == ReduceMode::MEAN) {
      // The loop ending condition should be `i / blockDim.x * blockDim.x <
      // length` instead of `i < length`, because the last block may have some
      // threads not doing BlockReduce in the condition `i < length`, which
      // leads to deadlock.
      for (int64_t i = threadIdx.x; i / blockDim.x * blockDim.x < length;
           i += blockDim.x) {
        scalar_t w = 0;
        if (i < length) {
          w = weight[start + i];
        }
        scalar_t res = BlockReduce(temp_storage).Sum(w);
        // Only thread 0 has the reduce sum result.
        if (threadIdx.x == 0) {
          weight_sum += res;
        }
        // NOTE: A subsequent __syncthreads() threadblock barrier should be
        // invoked after calling BlockReduce if the collective’s temporary
        // storage (e.g., temp_storage) is to be reused or repurposed.
        __syncthreads();
      }
      // Copy the result to the shared memory.
      if (threadIdx.x == 0) {
        s_weight_sum = weight_sum;
      }
      __syncthreads();
      // Broadcast the result to all threads.
      weight_sum = s_weight_sum;
    }

    for (int64_t i = threadIdx.x; i * PACK_SIZE < (end - start) * D;
         i += blockDim.x) {
      int64_t idx = start + (i * PACK_SIZE / D);
      int64_t dp = (i * PACK_SIZE % D);
      int64_t raw_idx = reverse_indices[idx];
      typename AP::type g_vec;
      if constexpr (mode == ReduceMode::TILE) {
        AP::load(grad_output + idx * D + dp, g_vec);
      } else {
        for (int j = 0; j < PACK_SIZE; ++j) {
          auto g = grad_output[s * D + dp + j];
          AP::set_element(g_vec, j, g);
        }
      }
      scalar_t w_base = 1;
      if constexpr (USE_WEIGHT) {
        w_base = weight[idx];
      } else {
        weight_sum = static_cast<scalar_t>(length);
      }
      if constexpr (mode == ReduceMode::MEAN) {
        w_base /= weight_sum;
      }

      for (int j = 0; j < PACK_SIZE; ++j) {
        atomic_add_custom<scalar_t>(&grad_unique_emb[raw_idx * D + dp + j],
                                    AP::get_element(g_vec, j) * w_base);
      }
    }
  }
}

#ifdef __HIP_PLATFORM_AMD__
template <typename offset_t>
__global__ void compute_segment_ids_kernel(
    const offset_t* __restrict__ offsets, int32_t* __restrict__ segment_ids,
    int64_t B, int64_t S) {
  // Each block handles one segment and fills segment IDs
  int64_t seg = blockIdx.x;
  if (seg >= S - 1) return;

  offset_t start = offsets[seg];
  offset_t end = offsets[seg + 1];

  for (offset_t i = start + threadIdx.x; i < end; i += blockDim.x) {
    segment_ids[i] = seg;
  }
}

template <typename scalar_t, typename offset_t>
__global__ void compute_weight_sums_kernel(
    const scalar_t* __restrict__ weight, const offset_t* __restrict__ offsets,
    scalar_t* __restrict__ weight_sums, int64_t S) {
  int64_t seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= S - 1) return;

  offset_t start = offsets[seg];
  offset_t end = offsets[seg + 1];

  scalar_t sum = 0;
  for (offset_t i = start; i < end; ++i) {
    sum += weight[i];
  }
  weight_sums[seg] = sum;
}

template <typename scalar_t, typename offset_t, ReduceMode mode,
          bool USE_WEIGHT>
__global__ void segment_reduce_backward_kernel_rocm(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ reverse_indices,
    const int32_t* __restrict__ segment_ids,
    const scalar_t* __restrict__ weight_sums,
    const offset_t* __restrict__ offsets, scalar_t* grad_unique_emb, int64_t B,
    int64_t S, int64_t D) {
  // Total work items: B * D / 4 (with PACK_SIZE=4)
  const int64_t total_work = (B * D) >> 2;  // Divide by 4
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = blockDim.x * gridDim.x;
  
  // Process multiple elements per thread for better ILP
  #pragma unroll 1  // Don't unroll outer loop to save registers
  for (int64_t work_idx = tid; work_idx < total_work; work_idx += stride) {
    // Decode flat index to (input_idx, dp)
    const int64_t flat_idx = work_idx << 2;  // Multiply by 4
    const int64_t input_idx = flat_idx / D;
    const int64_t dp = flat_idx - input_idx * D;  // Faster than modulo
    
    // Load segment and unique index (these are per-row, not per-element)
    const int32_t seg = segment_ids[input_idx];
    const int64_t raw_idx = reverse_indices[input_idx];
    
    // Load gradient (4 floats vectorized)
    float4 g_vec;
    if constexpr (mode == ReduceMode::TILE) {
      g_vec = *reinterpret_cast<const float4*>(grad_output + input_idx * D + dp);
    } else {
      g_vec = *reinterpret_cast<const float4*>(grad_output + seg * D + dp);
    }
    
    // Compute weight factor
    scalar_t w_base;
    if constexpr (USE_WEIGHT) {
      w_base = weight[input_idx];
      if constexpr (mode == ReduceMode::MEAN) {
        w_base /= weight_sums[seg];
      }
    } else {
      if constexpr (mode == ReduceMode::MEAN) {
        w_base = static_cast<scalar_t>(1) /
                  static_cast<scalar_t>(offsets[seg + 1] - offsets[seg]);
      } else {
        w_base = static_cast<scalar_t>(1);
      }
    }
    
    // Write results with atomics (4 consecutive floats)
    scalar_t* out_ptr = grad_unique_emb + raw_idx * D + dp;
    atomicAdd(out_ptr + 0, g_vec.x * w_base);
    atomicAdd(out_ptr + 1, g_vec.y * w_base);
    atomicAdd(out_ptr + 2, g_vec.z * w_base);
    atomicAdd(out_ptr + 3, g_vec.w * w_base);
  }
}
#endif

#define FORWARD_LAUNCH_KERNEL(scalar_t, offset_t, mode, use_weight, vec_size) \
  segment_reduce_forward_kernel<scalar_t, offset_t, mode, use_weight,         \
                                vec_size>                                     \
      <<<block_num, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(       \
          unique_emb, weight, reverse_indices, offsets, output, B, N, S, D);  \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

template <typename scalar_t, typename offset_t, ReduceMode mode>
void segment_reduce_forward_kernel_launcher(
    const scalar_t* unique_emb, const scalar_t* weight, bool use_weight,
    const int64_t* reverse_indices, const offset_t* offsets, scalar_t* output,
    int64_t B, int64_t N, int64_t S, int64_t D) {
  int64_t block_size = 256;
  int64_t block_num = 65536;
  block_num = std::min(block_num, S);

  if (D % 4 == 0) {
    if (use_weight) {
      FORWARD_LAUNCH_KERNEL(scalar_t, offset_t, mode, true, 4)
    } else {
      FORWARD_LAUNCH_KERNEL(scalar_t, offset_t, mode, false, 4)
    }
  } else if (D % 2 == 0) {
    if (use_weight) {
      FORWARD_LAUNCH_KERNEL(scalar_t, offset_t, mode, true, 2)
    } else {
      FORWARD_LAUNCH_KERNEL(scalar_t, offset_t, mode, false, 2)
    }
  } else {
    if (use_weight) {
      FORWARD_LAUNCH_KERNEL(scalar_t, offset_t, mode, true, 1)
    } else {
      FORWARD_LAUNCH_KERNEL(scalar_t, offset_t, mode, false, 1)
    }
  }
}

#undef FORWARD_LAUNCH_KERNEL

#define LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, use_weight, vec_size) \
  segment_reduce_backward_kernel<scalar_t, offset_t, mode, use_weight,         \
                                 vec_size>                                     \
      <<<block_num, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(        \
          grad_output, weight, reverse_indices, offsets, grad_unique_emb, B,   \
          N, S, D);                                                            \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef __HIP_PLATFORM_AMD__
#define LAUNCH_BACKWARD_KERNEL_ROCM(scalar_t, offset_t, mode,        \
                                              use_weight)                      \
  segment_reduce_backward_kernel_rocm<scalar_t, offset_t, mode, use_weight>         \
      <<<grid_size, block_size, 0, stream>>>(                                  \
          grad_output, weight, reverse_indices, segment_ids, weight_sums,      \
          offsets, grad_unique_emb, B, S, D);                                  \
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif

template <typename scalar_t, typename offset_t, ReduceMode mode>
void segment_reduce_backward_kernel_launcher(
    const scalar_t* grad_output, const scalar_t* weight, bool use_weight,
    const int64_t* reverse_indices, const offset_t* offsets,
    scalar_t* grad_unique_emb, int64_t B, int64_t N, int64_t S, int64_t D) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int sm_count = get_sm_count();


#ifndef __HIP_PLATFORM_AMD__
  int64_t block_size = 256;
  int64_t block_num = get_sm_count() * 8;
  block_num = std::min(block_num, S);

  if (D % 4 == 0) {
    if (use_weight) {
      LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, true, 4)
    } else {
      LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, false, 4)
    }
  } else if (D % 2 == 0) {
    if (use_weight) {
      LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, true, 2)
    } else {
      LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, false, 2)
    }
  } else {
    if (use_weight) {
      LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, true, 1)
    } else {
      LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, false, 1)
    }
  }
#else
  // Use high-occupancy kernel for D % 4 == 0 (most common case)
  // Fall back to original segment-parallel kernel for other cases
  if (D % 4 == 0) {
    int64_t block_size = 256;
    int64_t total_work = (B * D) / 4;
    // Use larger grid size for better latency hiding
    int64_t grid_size =
        std::min((total_work + block_size - 1) / block_size,
                 static_cast<int64_t>(sm_count * 32));

    // Allocate temporary buffers for segment_ids and weight_sums
    int32_t* segment_ids = cuda_malloc<int32_t>(B * sizeof(int32_t), stream);
    scalar_t* weight_sums = nullptr;
    if constexpr (mode == ReduceMode::MEAN) {
      if (use_weight) {
        weight_sums = cuda_malloc<scalar_t>((S - 1) * sizeof(scalar_t), stream);
      }
    }

    // Precompute segment IDs
    compute_segment_ids_kernel<offset_t>
        <<<S - 1, 256, 0, stream>>>(offsets, segment_ids, B, S);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Precompute weight sums for MEAN mode with weights
    if constexpr (mode == ReduceMode::MEAN) {
      if (use_weight) {
        int64_t ws_grid = (S - 1 + 255) / 256;
        compute_weight_sums_kernel<scalar_t, offset_t>
            <<<ws_grid, 256, 0, stream>>>(weight, offsets, weight_sums, S);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }

    // Launch high-occupancy kernel
    if (use_weight) {
      LAUNCH_HIGH_OCCUPANCY_BACKWARD_KERNEL(scalar_t, offset_t, mode, true)
    } else {
      LAUNCH_HIGH_OCCUPANCY_BACKWARD_KERNEL(scalar_t, offset_t, mode, false)
    }

    // Free temporary buffers
    delete_cuda_ptr(segment_ids);
    if (weight_sums != nullptr) {
      delete_cuda_ptr(weight_sums);
    }
  } else {
    // Fall back to original segment-parallel kernel for D % 4 != 0
    int64_t block_size = 256;
    int64_t block_num = sm_count * 8;
    block_num = std::min(block_num, S);

    if (D % 2 == 0) {
      if (use_weight) {
        LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, true, 2)
      } else {
        LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, false, 2)
      }
    } else {
      if (use_weight) {
        LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, true, 1)
      } else {
        LAUNCH_BACKWARD_KERNEL(scalar_t, offset_t, mode, false, 1)
      }
    }
  }
#endif
}
at::Tensor segment_reduce_forward(at::Tensor unique_emb,
                                  c10::optional<at::Tensor> weight,
                                  at::Tensor reverse_indices,
                                  at::Tensor offsets, std::string mode) {
  TORCH_CHECK(mode == "sum" || mode == "mean" || mode == "tile",
              "Invalid mode: ", mode);
  TORCH_CHECK(unique_emb.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(reverse_indices.is_cuda(),
              "reverse_indices must be a CUDA tensor");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor");

  int64_t B = reverse_indices.size(0);
  int64_t N = unique_emb.size(0);
  int64_t S = offsets.size(0);
  int64_t D = unique_emb.size(1);
  bool use_weight = weight.has_value();
  auto options = unique_emb.options();
  at::Tensor weight_data;
  if (!use_weight) {
    weight_data = torch::ones({1}, options);
  } else {
    weight_data = weight.value();
  }

  at::Tensor output;
  // AT_DISPATCH_FLOATING_TYPES(
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,  // fp64,fp32,fp16,bf16
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Half,  // fp64,fp32,fp16
#else
  AT_DISPATCH_FLOATING_TYPES(  // fp64,fp32
#endif
      unique_emb.scalar_type(), "segmented_reduce", [&] {
        AT_DISPATCH_INDEX_TYPES(
            offsets.scalar_type(), "segmented_reduce_offset", [&] {
              using offset_t = index_t;

              if (mode == "sum") {
                output = torch::zeros({S - 1, D}, options);
                segment_reduce_forward_kernel_launcher<scalar_t, offset_t,
                                                       ReduceMode::SUM>(
                    unique_emb.data_ptr<scalar_t>(),
                    weight_data.data_ptr<scalar_t>(), use_weight,
                    reverse_indices.data_ptr<int64_t>(),
                    offsets.data_ptr<offset_t>(), output.data_ptr<scalar_t>(),
                    B, N, S, D);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              } else if (mode == "mean") {
                output = torch::zeros({S - 1, D}, options);
                segment_reduce_forward_kernel_launcher<scalar_t, offset_t,
                                                       ReduceMode::MEAN>(
                    unique_emb.data_ptr<scalar_t>(),
                    weight_data.data_ptr<scalar_t>(), use_weight,
                    reverse_indices.data_ptr<int64_t>(),
                    offsets.data_ptr<offset_t>(), output.data_ptr<scalar_t>(),
                    B, N, S, D);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              } else if (mode == "tile") {
                output = torch::zeros({B, D}, options);
                segment_reduce_forward_kernel_launcher<scalar_t, offset_t,
                                                       ReduceMode::TILE>(
                    unique_emb.data_ptr<scalar_t>(),
                    weight_data.data_ptr<scalar_t>(), use_weight,
                    reverse_indices.data_ptr<int64_t>(),
                    offsets.data_ptr<offset_t>(), output.data_ptr<scalar_t>(),
                    B, N, S, D);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
            });
      });
  return output;
}

// Backward function dispatcher
at::Tensor segment_reduce_backward(at::Tensor grad_output,
                                   c10::optional<at::Tensor> weight,
                                   at::Tensor reverse_indices,
                                   at::Tensor offsets, int64_t unique_size,
                                   std::string mode) {
  TORCH_CHECK(mode == "sum" || mode == "mean" || mode == "tile",
              "Invalid mode: ", mode);
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(reverse_indices.is_cuda(),
              "reverse_indices must be a CUDA tensor");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor");

  int64_t B = reverse_indices.size(0);
  int64_t N = grad_output.size(0);
  int64_t S = offsets.size(0);
  int64_t D = grad_output.size(1);

  bool use_weight = weight.has_value();
  at::Tensor weight_data;
  if (use_weight) {
    weight_data = weight.value();

  } else {
    weight_data = torch::ones({1}, grad_output.options());
  }
  auto options = grad_output.options();
  auto grad_unique_emb = torch::zeros({unique_size, D}, options);
// AT_DISPATCH_FLOATING_TYPES(
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,  // fp64,fp32,fp16,bf16
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Half,  // fp64,fp32,fp16
#else
  AT_DISPATCH_FLOATING_TYPES(  // fp64,fp32
#endif
      grad_output.scalar_type(), "segmented_reduce_backward", [&] {
        AT_DISPATCH_INDEX_TYPES(
            offsets.scalar_type(), "segmented_reduce_backward_offset", [&] {
              using offset_t = index_t;

              if (mode == "sum") {
                segment_reduce_backward_kernel_launcher<scalar_t, offset_t,
                                                        ReduceMode::SUM>(
                    grad_output.data_ptr<scalar_t>(),
                    weight_data.data_ptr<scalar_t>(), use_weight,
                    reverse_indices.data_ptr<int64_t>(),
                    offsets.data_ptr<offset_t>(),
                    grad_unique_emb.data_ptr<scalar_t>(), B, unique_size, S, D);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              } else if (mode == "mean") {
                segment_reduce_backward_kernel_launcher<scalar_t, offset_t,
                                                        ReduceMode::MEAN>(
                    grad_output.data_ptr<scalar_t>(),
                    weight_data.data_ptr<scalar_t>(), use_weight,
                    reverse_indices.data_ptr<int64_t>(),
                    offsets.data_ptr<offset_t>(),
                    grad_unique_emb.data_ptr<scalar_t>(), B, unique_size, S, D);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              } else if (mode == "tile") {
                segment_reduce_backward_kernel_launcher<scalar_t, offset_t,
                                                        ReduceMode::TILE>(
                    grad_output.data_ptr<scalar_t>(),
                    weight_data.data_ptr<scalar_t>(), use_weight,
                    reverse_indices.data_ptr<int64_t>(),
                    offsets.data_ptr<offset_t>(),
                    grad_unique_emb.data_ptr<scalar_t>(), B, unique_size, S, D);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
            });
      });

  return grad_unique_emb;
}

template <typename T>
struct MergeOffsetsFactory {
  __device__ T operator()(T a, T b) { return a + b; }
};

at::Tensor merge_offsets(const std::vector<at::Tensor>& offsets,
                         torch::Tensor& max_value) {
  int64_t N = offsets.size();
  int64_t total_merge_size = 0;
  std::vector<int64_t> sizes(N);
  auto stream = at::cuda::getCurrentCUDAStream();
  for (int64_t i = 0; i < N; i++) {
    sizes[i] = (i < N - 1) ? (offsets[i].numel() - 1) : offsets[i].numel();
    total_merge_size += sizes[i];
  }
  at::Tensor output = torch::empty({total_merge_size}, offsets[0].options());

  AT_DISPATCH_INTEGRAL_TYPES(
      offsets[0].scalar_type(), "merge_offsets_cuda_impl", ([&] {
        scalar_t* max_value_data = max_value.data_ptr<scalar_t>();
        CudaVecParam<scalar_t> prefix_sum(N, stream);
        CudaVecParam<scalar_t*> offsets_ptrs(N, stream);
        CudaVecParam<scalar_t*> outputs_ptrs(N, stream);
        for (int64_t i = 0; i < N; i++) {
          offsets_ptrs[i] = offsets[i].data_ptr<scalar_t>();
          prefix_sum[i] =
              (i == 0) ? 0 : prefix_sum[i - 1] + max_value_data[i - 1];
        }
        auto output_data = output.data_ptr<scalar_t>();
        for (int64_t i = 0; i < N; i++) {
          outputs_ptrs[i] = output_data;
          output_data += sizes[i];
        }
        fused_element_wise_launcher<scalar_t, scalar_t, scalar_t,
                                    MergeOffsetsFactory<scalar_t>>(
            const_cast<const scalar_t**>(offsets_ptrs.data()),
            prefix_sum.data(), outputs_ptrs.data(), sizes.data(), N,
            MergeOffsetsFactory<scalar_t>(), false, stream);
      }));
  return output;
}

template <typename scalar_t>
__global__ void gen_segment_indices_cuda_kernel(const scalar_t* offset,
                                                scalar_t* output,
                                                const int64_t input_size) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > input_size) return;
  auto start_idx = offset[i - 1];
  for (auto j = 0; j < (offset[i] - offset[i - 1]); ++j) {
    output[start_idx + j] = i - 1;
  }
}

at::Tensor gen_segment_indices_by_offset(torch::Tensor offset) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t segment_size;
  AT_DISPATCH_INTEGRAL_TYPES(
      offset.scalar_type(), "cal_segment_size",
      ([&] { segment_size = offset[offset.numel() - 1].item<scalar_t>(); }));
  at::Tensor output = torch::empty({segment_size}, offset.options());
  AT_DISPATCH_INTEGRAL_TYPES(
      offset.scalar_type(), "gen_segment_indices_by_offset_cuda_impl", ([&] {
        const int64_t threads = 128;
        const int64_t blocks = (offset.numel() - 1 + threads) / threads;
        gen_segment_indices_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(offset.data_ptr<scalar_t>(),
                                             output.data_ptr<scalar_t>(),
                                             offset.numel() - 1);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
  return output;
}
}  // namespace functional
}  // namespace recis
