#include <torch/extension.h>

#include "ATen/cuda/CUDAContext.h"
#include "cuda/atomic_fast.cuh"
#include "cuda/utils.cuh"

namespace recis {
namespace functional {

template <typename T, typename pack_t>
static __global__ void fill_value_2D(T* dst, const T val, const int64_t lens,
                                     const int64_t embedding_dim,
                                     const int64_t id_tile_size,
                                     const int64_t emb_tile_size) {
  int64_t block_idx = blockIdx.x * id_tile_size;
  int64_t emb_idx = threadIdx.x * emb_tile_size;
  int64_t idx = block_idx + threadIdx.y;
  if (idx >= lens || emb_idx >= embedding_dim) return;

  if (emb_idx + emb_tile_size <= embedding_dim) {
    pack_t fill_vec;
    for (auto i = 0; i < emb_tile_size; ++i) {
      *((T*)(&fill_vec) + i) = val;
    }
    *(pack_t*)(dst + idx * embedding_dim + emb_idx) = fill_vec;
  } else {
    for (auto i = 0; i < embedding_dim - emb_idx; ++i) {
      dst[idx * embedding_dim + emb_idx + i] = val;
    }
  }
}

#define FILL_VALUE_2D_LAUNCH_KERNEL(scalar_t, pack_t)           \
  fill_value_2D<scalar_t, pack_t>                               \
      <<<grids, blocks, 0, at::cuda::getCurrentCUDAStream()>>>( \
          dst, val, lens, embedding_dim, id_tile_size, emb_tile_size);

template <typename scalar_t>
void fill_value_2D_kernel_launcher(scalar_t* dst, const scalar_t val,
                                   const int64_t lens,
                                   const int64_t embedding_dim) {
  int64_t emb_tile_size, emb_thread_size, id_tile_size, id_blocks,
      real_pack_size;
  recis::cuda::cal_pack_sizes<scalar_t>(lens, embedding_dim, emb_tile_size,
                                        emb_thread_size, id_tile_size,
                                        id_blocks, real_pack_size);
  dim3 grids(id_blocks);
  dim3 blocks(emb_thread_size, id_tile_size);
  if (real_pack_size == 2) {
    FILL_VALUE_2D_LAUNCH_KERNEL(scalar_t, scalar_t);
  } else if (real_pack_size == 4) {
    FILL_VALUE_2D_LAUNCH_KERNEL(scalar_t, float);
  } else if (real_pack_size == 8) {
    FILL_VALUE_2D_LAUNCH_KERNEL(scalar_t, float2);
  } else if (real_pack_size == 16) {
    FILL_VALUE_2D_LAUNCH_KERNEL(scalar_t, float4);
  } else {
    TORCH_CHECK(false, "fill value 2D cuda kernel error pack size");
  }
}

#undef FILL_VALUE_2D_LAUNCH_KERNEL

template <typename T>
static __global__ void fill_value_1D(T* dst, const T val, const int64_t lens) {
  const int64_t t_id =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (t_id < lens) {
    dst[t_id] = val;
  }
};

template <typename T, typename Index>
static __global__ void segment_weight_sum_kernel(const T* src, bool use_weight,
                                                 const Index* seg_ids, T* dst,
                                                 const int64_t seg_nums,
                                                 const int64_t out_size) {
  const int64_t t_id =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (t_id < out_size) {
    const int64_t out_idx = seg_ids[t_id];
    if (out_idx < 0 || out_idx >= seg_nums) {
      return;
    }
    auto src_id = use_weight ? t_id : 0;
    T src_val = src[src_id];
    atomic_add_custom<T>(dst + out_idx, src_val);
  }
};

template <typename T, typename Index>
static __global__ void segment_weight_norm_kernel(
    const T* weight, bool use_weight, const T* weight_sum, const Index* seg_ids,
    T* output, const int64_t seg_nums, const int64_t weight_size) {
  const int64_t t_id =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (t_id < weight_size) {
    const int64_t sum_idx = static_cast<int64_t>(seg_ids[t_id]);
    auto weight_id = use_weight ? t_id : 0;
    if (sum_idx < 0 || sum_idx >= seg_nums) {
      output[t_id] = weight[weight_id];
      return;
    }
    T sum = weight_sum[sum_idx];
    T norm = weight[weight_id] / sum;
    output[t_id] = norm;
  }
};

template <typename T, typename pack_t>
__device__ void segment_atomic_add(T* dst_gm, T* val_ptr) {
  constexpr int vec_size = sizeof(pack_t) / sizeof(T);
  for (int idx = 0; idx < vec_size; ++idx) {
    atomic_add_custom<T>(dst_gm + idx, val_ptr[idx]);  // atomicAdd
  }
}

template <>
__device__ void segment_atomic_add<float, float2>(float* dst_gm,
                                                  float* val_ptr) {
  float2 val = make_float2(val_ptr[0], val_ptr[1]);
  atomic_add_vec<float2>(reinterpret_cast<float2*>(dst_gm), val);
}

template <>
__device__ void segment_atomic_add<float, float4>(float* dst_gm,
                                                  float* val_ptr) {
  float4 val = make_float4(val_ptr[0], val_ptr[1], val_ptr[2], val_ptr[3]);
  atomic_add_vec<float4>(reinterpret_cast<float4*>(dst_gm), val);
}

template <typename T, typename Index, typename pack_t>
static __global__ void segment_sum_kernel(const T* data, const Index* indices,
                                          const T* weight, bool use_weight,
                                          const Index* seg_ids, T* output,
                                          const int64_t reverse_num,
                                          const int64_t embedding_dim,
                                          const int64_t id_tile_size,
                                          const int64_t emb_tile_size) {
  int64_t block_idx = blockIdx.x * id_tile_size;
  int64_t emb_idx = threadIdx.x * emb_tile_size;
  int64_t idx = block_idx + threadIdx.y;
  if (idx >= reverse_num || emb_idx >= embedding_dim) return;
  const int64_t unique_idx = indices[idx];
  const int64_t out_idx = seg_ids[idx];
  const int64_t weight_idx = use_weight ? idx : 0;
  T weight_val = weight[weight_idx];

  if (emb_idx + emb_tile_size <= embedding_dim) {
    pack_t val = *(pack_t*)(data + unique_idx * embedding_dim + emb_idx);
    for (auto i = 0; i < emb_tile_size; ++i) {
      *((T*)(&val) + i) = *((T*)(&val) + i) * weight_val;
    }
    T* dst_begin =
        output + out_idx * embedding_dim + emb_idx;  // pack start ptr
    segment_atomic_add<T, pack_t>(dst_begin, (T*)(&val));
  } else {
    for (auto i = 0; i < embedding_dim - emb_idx; ++i) {
      T val = data[unique_idx * embedding_dim + emb_idx + i];
      val = val * weight_val;
      atomic_add_custom<T>(output + out_idx * embedding_dim + emb_idx + i, val);
    }
  }
};

#define SEGMENT_SUM_LAUNCH_KERNEL(scalar_t, index_t, pack_t)               \
  segment_sum_kernel<scalar_t, index_t, pack_t>                            \
      <<<grids, blocks, 0, at::cuda::getCurrentCUDAStream()>>>(            \
          data, indices, weight, use_weight, seg_ids, output, reverse_num, \
          embedding_dim, id_tile_size, emb_tile_size);

template <typename scalar_t, typename index_t>
void segment_sum_kernel_launcher(const scalar_t* data, const index_t* indices,
                                 const scalar_t* weight, bool use_weight,
                                 const index_t* seg_ids, scalar_t* output,
                                 const int64_t reverse_num,
                                 const int64_t embedding_dim) {
  int64_t emb_tile_size, emb_thread_size, id_tile_size, id_blocks,
      real_pack_size;
  recis::cuda::cal_pack_sizes<scalar_t>(
      reverse_num, embedding_dim, emb_tile_size, emb_thread_size, id_tile_size,
      id_blocks, real_pack_size);
  if (id_blocks <= 0) return;
  dim3 grids(id_blocks);
  dim3 blocks(emb_thread_size, id_tile_size);
  if (real_pack_size == 2) {
    SEGMENT_SUM_LAUNCH_KERNEL(scalar_t, index_t, scalar_t);
  } else if (real_pack_size == 4) {
    SEGMENT_SUM_LAUNCH_KERNEL(scalar_t, index_t, float);
  } else if (real_pack_size == 8) {
    SEGMENT_SUM_LAUNCH_KERNEL(scalar_t, index_t, float2);
  } else if (real_pack_size == 16) {
    SEGMENT_SUM_LAUNCH_KERNEL(scalar_t, index_t, float4);
  } else {
    TORCH_CHECK(false, "segment sum cuda kernel error pack size");
  }
}

#undef SEGMENT_SUM_LAUNCH_KERNEL

void segment_sum_cuda(torch::Tensor data, torch::Tensor weight, bool use_weight,
                      torch::Tensor indices, torch::Tensor segment_ids,
                      const int64_t num_segments, torch::Tensor& out) {
  int64_t embedding_dim = data.size(1);
  int64_t reverse_num = indices.numel();
  int64_t out_num = num_segments;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // output set
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,  // fp64,fp32,fp16,bf16
      out.scalar_type(), "set_value", ([&] {
        fill_value_2D_kernel_launcher<scalar_t>(
            out.data_ptr<scalar_t>(), scalar_t(0), out_num, embedding_dim);
      }));
  // segment sum
  AT_DISPATCH_INTEGRAL_TYPES(
      indices.scalar_type(), "segment_sum_indices", ([&] {
        using index_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,  // fp64,fp32,fp16,bf16
            data.scalar_type(), "segment_sum_kernel", ([&] {
              segment_sum_kernel_launcher<scalar_t, index_t>(
                  data.data_ptr<scalar_t>(), indices.data_ptr<index_t>(),
                  weight.data_ptr<scalar_t>(), use_weight,
                  segment_ids.data_ptr<index_t>(), out.data_ptr<scalar_t>(),
                  reverse_num, embedding_dim);
            }));
      }));
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(cudaSuccess == err, cudaGetErrorString(err));
};  // end segment_sum_cuda

bool segment_mean_cuda(torch::Tensor weight, bool use_weight,
                       torch::Tensor weight_sum, torch::Tensor& weight_norm,
                       torch::Tensor segment_ids, const int64_t num_segments) {
  cudaError_t err;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t threads = 128;
  const int64_t block_sum = (weight_sum.numel() + threads - 1) / threads;
  const int64_t block_weight = (weight_norm.numel() + threads - 1) / threads;
  // weight set
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,  // fp64,fp32,fp16,bf16
      weight_sum.scalar_type(), "set_value", ([&] {
        fill_value_1D<scalar_t><<<block_sum, threads, 0, stream>>>(
            weight_sum.data_ptr<scalar_t>(), scalar_t(0), weight_sum.numel());
      }));
  if (block_weight <= 0) return true;
  // weight sum
  AT_DISPATCH_INTEGRAL_TYPES(
      segment_ids.scalar_type(), "segment_weight_mean_indices", ([&] {
        using index_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            weight.scalar_type(), "segment_weight_sum_kernel", ([&] {
              segment_weight_sum_kernel<scalar_t, index_t>
                  <<<block_weight, threads, 0, stream>>>(
                      weight.data_ptr<scalar_t>(), use_weight,
                      segment_ids.data_ptr<index_t>(),
                      weight_sum.data_ptr<scalar_t>(), num_segments,
                      weight_norm.numel());
            }));
        err = cudaGetLastError();
        TORCH_CHECK(cudaSuccess == err, cudaGetErrorString(err));
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,  // fp64,fp32,fp16,bf16
            weight.scalar_type(), "segment_weight_norm_kernel", ([&] {
              segment_weight_norm_kernel<scalar_t, index_t>
                  <<<block_weight, threads, 0, stream>>>(
                      weight.data_ptr<scalar_t>(), use_weight,
                      weight_sum.data_ptr<scalar_t>(),
                      segment_ids.data_ptr<index_t>(),
                      weight_norm.data_ptr<scalar_t>(), num_segments,
                      weight_norm.numel());
            }));
        err = cudaGetLastError();
        TORCH_CHECK(cudaSuccess == err, cudaGetErrorString(err));
      }));
  return true;
};  // end segment_mean_cuda
}  // namespace functional
}  // namespace recis
