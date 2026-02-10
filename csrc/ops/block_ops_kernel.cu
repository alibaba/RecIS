#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "cuda/cuda_param.cuh"
#include "cuda/utils.cuh"
#include "compare_ops.h"

namespace recis {
namespace functional {

template <typename T, CompareOp op>
struct CompareFactory;

template <typename T>
struct CompareFactory<T, CompareOp::LT> {
  __host__ __device__ bool operator()(const T a, const T b) const { return a < b; }
};

template <typename T>
struct CompareFactory<T, CompareOp::LE> {
  __host__ __device__ bool operator()(const T a, const T b) const { return a <= b; }
};

template <typename T>
struct CompareFactory<T, CompareOp::GT> {
  __host__ __device__ bool operator()(const T a, const T b) const { return a > b; }
};

template <typename T>
struct CompareFactory<T, CompareOp::GE> {
  __host__ __device__ bool operator()(const T a, const T b) const { return a >= b; }
};

template <typename T>
struct CompareFactory<T, CompareOp::EQ> {
  __host__ __device__ bool operator()(const T a, const T b) const { return a == b; }
};

template <typename scalar_t, typename pack_t>
__global__ void gather_cuda_kernel(const int64_t* ids, const scalar_t* emb,
                                   int64_t num_ids, int64_t embedding_dim,
                                   scalar_t* output, int64_t id_tile_size,
                                   int64_t emb_tile_size) {
  int64_t block_idx = blockIdx.x * id_tile_size;
  int64_t emb_idx = threadIdx.x * emb_tile_size;
  int64_t idx = block_idx + threadIdx.y;
  if (idx >= num_ids || emb_idx >= embedding_dim) return;

  int64_t id = ids[idx];
  if (emb_idx + emb_tile_size <= embedding_dim) {
    *(pack_t*)(output + idx * embedding_dim + emb_idx) =
        *(pack_t*)(emb + id * embedding_dim + emb_idx);
  } else {
    for (auto i = 0; i < embedding_dim - emb_idx; ++i) {
      output[idx * embedding_dim + emb_idx + i] =
          emb[id * embedding_dim + emb_idx + i];
    }
  }
}

#define GATHER_LAUNCH_KERNEL(scalar_t, pack_t)                    \
  gather_cuda_kernel<scalar_t, pack_t>                            \
      <<<grids, blocks, 0, at::cuda::getCurrentCUDAStream()>>>(   \
          ids, emb, num_ids, embedding_dim, output, id_tile_size, \
          emb_tile_size);                                         \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

template <typename scalar_t>
void gather_kernel_launcher(const int64_t* ids, const scalar_t* emb,
                            int64_t num_ids, int64_t embedding_dim,
                            scalar_t* output) {
  int64_t emb_tile_size, emb_thread_size, id_tile_size, id_blocks,
      real_pack_size;
  recis::cuda::cal_pack_sizes<scalar_t>(num_ids, embedding_dim, emb_tile_size,
                                        emb_thread_size, id_tile_size,
                                        id_blocks, real_pack_size);
  dim3 grids(id_blocks);
  dim3 blocks(emb_thread_size, id_tile_size);
  if (real_pack_size == 2) {
    GATHER_LAUNCH_KERNEL(scalar_t, scalar_t);
  } else if (real_pack_size == 4) {
    GATHER_LAUNCH_KERNEL(scalar_t, float);
  } else if (real_pack_size == 8) {
    GATHER_LAUNCH_KERNEL(scalar_t, float2);
  } else if (real_pack_size == 16) {
    GATHER_LAUNCH_KERNEL(scalar_t, float4);
  } else {
    TORCH_CHECK(false, "gather cuda kernel error pack size");
  }
}

#undef GATHER_LAUNCH_KERNEL

template <typename scalar_t, typename pack_t>
__global__ void block_gather_cuda_kernel(
    const int64_t* ids, scalar_t** emb_blocks, int64_t num_ids,
    int64_t embedding_dim, int64_t block_size, int64_t default_key,
    scalar_t* output, bool readonly, int64_t beg, int64_t id_tile_size,
    int64_t emb_tile_size) {
  int64_t block_idx = blockIdx.x * id_tile_size;
  int64_t emb_idx = threadIdx.x * emb_tile_size;
  int64_t idx = block_idx + threadIdx.y;
  if (idx >= num_ids || emb_idx >= embedding_dim) return;

  int64_t id = ids[idx + beg];
  bool fill_zero = (id == default_key && readonly);
  if (fill_zero) {
    if (emb_idx + emb_tile_size <= embedding_dim) {
      pack_t zero_vec;
      for (auto i = 0; i < emb_tile_size; ++i) {
        *((scalar_t*)(&zero_vec) + i) = 0;
      }
      *(pack_t*)(output + idx * embedding_dim + emb_idx) = zero_vec;
    } else {
      for (auto i = 0; i < embedding_dim - emb_idx; ++i) {
        output[idx * embedding_dim + emb_idx + i] = 0;
      }
    }
  } else {
    int64_t block_index = id / block_size;
    int64_t row_index = id % block_size;
    if (emb_idx + emb_tile_size <= embedding_dim) {
      *(pack_t*)(output + idx * embedding_dim + emb_idx) =
          *(pack_t*)(*(emb_blocks + block_index) + row_index * embedding_dim +
                     emb_idx);
    } else {
      for (auto i = 0; i < embedding_dim - emb_idx; ++i) {
        output[idx * embedding_dim + emb_idx + i] =
            emb_blocks[block_index][row_index * embedding_dim + emb_idx + i];
      }
    }
  }
}

#define BLOCK_GATHER_LAUNCH_KERNEL(scalar_t, pack_t)                        \
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();                   \
  block_gather_cuda_kernel<scalar_t, pack_t><<<grids, blocks, 0, stream>>>( \
      ids, emb_blocks, num_ids, embedding_dim, block_size, default_key,     \
      output, readonly, beg, id_tile_size, emb_tile_size);                  \
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

template <typename scalar_t>
void block_gather_kernel_launcher(const int64_t* ids, scalar_t** emb_blocks,
                                  int64_t num_ids, int64_t embedding_dim,
                                  int64_t block_size, int64_t default_key,
                                  scalar_t* output, bool readonly,
                                  int64_t beg) {
  int64_t emb_tile_size, emb_thread_size, id_tile_size, id_blocks,
      real_pack_size;
  recis::cuda::cal_pack_sizes<scalar_t>(num_ids, embedding_dim, emb_tile_size,
                                        emb_thread_size, id_tile_size,
                                        id_blocks, real_pack_size);

  dim3 grids(id_blocks);
  dim3 blocks(emb_thread_size, id_tile_size);
  if (real_pack_size == 2) {
    BLOCK_GATHER_LAUNCH_KERNEL(scalar_t, scalar_t);
  } else if (real_pack_size == 4) {
    BLOCK_GATHER_LAUNCH_KERNEL(scalar_t, float);
  } else if (real_pack_size == 8) {
    BLOCK_GATHER_LAUNCH_KERNEL(scalar_t, float2);
  } else if (real_pack_size == 16) {
    BLOCK_GATHER_LAUNCH_KERNEL(scalar_t, float4);
  } else {
    TORCH_CHECK(false, "block gather cuda kernel error pack size");
  }
}

#undef BLOCK_GATHER_LAUNCH_KERNEL

template <typename scalar_t, typename pack_t, bool is_broadcast>
__global__ void block_insert_cuda_kernel(
    const int64_t* ids, scalar_t** emb_blocks, const bool* mask,
    scalar_t* src_embeddings, int64_t num_ids, int64_t embedding_dim,
    int64_t block_size, bool with_mask, int64_t id_tile_size,
    int64_t emb_tile_size) {
  int64_t block_idx = blockIdx.x * id_tile_size;
  int64_t emb_idx = threadIdx.x * emb_tile_size;
  int64_t idx = block_idx + threadIdx.y;
  if (idx >= num_ids || emb_idx >= embedding_dim) return;
  if (with_mask && !mask[idx]) return;

  int64_t id = ids[idx];
  if (id < 0) {
    CUDA_KERNEL_ASSERT(id == -1);
    return;
  }
  int64_t block_index = id / block_size;
  int64_t row_index = id % block_size;

  if (emb_idx + emb_tile_size <= embedding_dim) {
    const scalar_t* emb = is_broadcast
                              ? src_embeddings + emb_idx
                              : src_embeddings + idx * embedding_dim + emb_idx;
    *(pack_t*)(*(emb_blocks + block_index) + row_index * embedding_dim +
               emb_idx) = *(pack_t*)(emb);
  } else {
    for (auto i = 0; i < embedding_dim - emb_idx; ++i) {
      const scalar_t* emb =
          is_broadcast ? src_embeddings + emb_idx + i
                       : src_embeddings + idx * embedding_dim + emb_idx + i;
      emb_blocks[block_index][row_index * embedding_dim + emb_idx + i] = *emb;
    }
  }
}

#define BLOCK_INSERT_LAUNCH_KERNEL(scalar_t, pack_t, running_is_broadcast)   \
  do {                                                                       \
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();                  \
    auto insert_launch = [&](auto is_broadcast) {                            \
      block_insert_cuda_kernel<scalar_t, pack_t,                             \
                               decltype(is_broadcast)::value>                \
          <<<grids, blocks, 0, stream>>>(                                    \
              ids, emb_blocks, mask, src_embeddings, num_ids, embedding_dim, \
              block_size, with_mask, id_tile_size, emb_tile_size);           \
    };                                                                       \
    if (running_is_broadcast) {                                              \
      insert_launch(std::true_type{});                                       \
    } else {                                                                 \
      insert_launch(std::false_type{});                                      \
    }                                                                        \
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));                           \
  } while (0)

template <typename scalar_t>
void block_insert_kernel_launcher(const int64_t* ids, scalar_t** emb_blocks,
                                  const bool* mask, scalar_t* src_embeddings,
                                  int64_t num_ids, int64_t num_emb,
                                  int64_t embedding_dim, int64_t block_size,
                                  bool with_mask) {
  int64_t emb_tile_size, emb_thread_size, id_tile_size, id_blocks,
      real_pack_size;
  recis::cuda::cal_pack_sizes<scalar_t>(num_ids, embedding_dim, emb_tile_size,
                                        emb_thread_size, id_tile_size,
                                        id_blocks, real_pack_size);
  bool is_broadcast = num_emb == 1;

  dim3 grids(id_blocks);
  dim3 blocks(emb_thread_size, id_tile_size);
  if (real_pack_size == 2) {
    BLOCK_INSERT_LAUNCH_KERNEL(scalar_t, scalar_t, is_broadcast);
  } else if (real_pack_size == 4) {
    BLOCK_INSERT_LAUNCH_KERNEL(scalar_t, float, is_broadcast);
  } else if (real_pack_size == 8) {
    BLOCK_INSERT_LAUNCH_KERNEL(scalar_t, float2, is_broadcast);
  } else if (real_pack_size == 16) {
    BLOCK_INSERT_LAUNCH_KERNEL(scalar_t, float4, is_broadcast);
  } else {
    TORCH_CHECK(false, "block insert cuda kernel error pack size");
  }
}

#undef BLOCK_INSERT_LAUNCH_KERNEL

torch::Tensor gather_cuda(const torch::Tensor ids, const torch::Tensor emb) {
  TORCH_CHECK(ids.device().type() == torch::kCUDA,
              "Input must be on CUDA device");
  TORCH_CHECK(emb.device().type() == torch::kCUDA,
              "Embedding must be on CUDA device");
  int64_t num_ids = ids.numel();
  int embedding_dim = emb.size(1);
  auto output = torch::empty({num_ids, embedding_dim}, emb.options());
  if (num_ids == 0) return output;
  AT_DISPATCH_FLOATING_TYPES_AND3(
      at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Char,
      output.scalar_type(), "gather_cuda_impl", ([&] {
        gather_kernel_launcher<scalar_t>(
            ids.data_ptr<int64_t>(), emb.data_ptr<scalar_t>(), num_ids,
            embedding_dim, output.data_ptr<scalar_t>());
      }));
  return output;
}

torch::Tensor block_gather_cuda(const torch::Tensor ids,
                                std::vector<torch::Tensor>& emb_blocks,
                                int64_t block_size, int64_t default_key,
                                bool readonly, int64_t beg, int64_t end) {
  TORCH_CHECK(ids.device().type() == torch::kCUDA,
              "Input must be on CUDA device");
  TORCH_CHECK(emb_blocks[0].device().type() == torch::kCUDA,
              "Embedding must be on CUDA device");
  int64_t num_ids = end - beg;
  int embedding_dim = emb_blocks[0].size(1);
  auto output = torch::empty({num_ids, embedding_dim}, emb_blocks[0].options());
  if (num_ids == 0) return output;
  auto block_num = emb_blocks.size();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND3(
      at::ScalarType::Half, at::ScalarType::Long, at::ScalarType::BFloat16,
      output.scalar_type(), "block_gather_cuda_impl", ([&] {
        recis::cuda::CudaVecParam<scalar_t*> emb_blocks_ptrs(block_num, stream);
        for (auto i = 0; i < block_num; ++i) {
          emb_blocks_ptrs[i] = emb_blocks[i].data_ptr<scalar_t>();
        }
        block_gather_kernel_launcher<scalar_t>(
            ids.data_ptr<int64_t>(), (scalar_t**)(emb_blocks_ptrs.data()),
            num_ids, embedding_dim, block_size, default_key,
            output.data_ptr<scalar_t>(), readonly, beg);
      }));
  return output;
}

template <typename scalar_t, typename Factory>
__global__ void block_filter_cuda_kernel(
    const int64_t* __restrict__ ids, const scalar_t** __restrict__ emb_blocks,
    int64_t num_ids, int64_t block_size, scalar_t threshold,
    bool* __restrict__ output, Factory factory) {
  int64_t did = blockIdx.x * blockDim.x + threadIdx.x;
  if (did >= num_ids) {
    return;
  }
  int64_t id = ids[did];
  int64_t block_index = id / block_size;
  int64_t row_index = id % block_size;
  scalar_t ft_val = emb_blocks[block_index][row_index];
  bool expired = factory(ft_val, threshold);
  output[did] = expired;
}

template <typename scalar_t, typename Factory>
void block_filter_kernel_launcher(const int64_t* ids,
                                  const scalar_t** emb_blocks, int64_t num_ids,
                                  int64_t block_size, scalar_t threshold,
                                  bool* output, Factory factory) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static const int filter_block_size = 128;
  dim3 grids((num_ids + filter_block_size - 1) / filter_block_size);
  dim3 blocks(filter_block_size);

  block_filter_cuda_kernel<scalar_t, Factory><<<grids, blocks, 0, stream>>>(
      ids, emb_blocks, num_ids, block_size, threshold, output, factory);
}

torch::Tensor block_filter_cuda(const torch::Tensor ids,
                                std::vector<torch::Tensor>& emb_blocks,
                                int64_t threshold, int64_t block_size,
                                int64_t compare_op) {
  TORCH_CHECK(ids.is_cuda(), "Input must be on CUDA device");
  TORCH_CHECK(emb_blocks.size() == 0 || emb_blocks[0].is_cuda(),
              "Embedding must be on CUDA device");
  int64_t num_ids = ids.numel();
  auto output = torch::empty(
      {num_ids},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
  if (num_ids == 0) return output;
  TORCH_CHECK(emb_blocks[0].size(1) == 1,
              "block_filter_cuda only support embedding_dim = 1");
  auto block_num = emb_blocks.size();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_INDEX_TYPES(
      emb_blocks[0].scalar_type(), "block_filter_cuda_impl", ([&] {
        BF_DISPATCH_COMPARE_TYPES(
            compare_op, ([&] {
              using Factory = CompareFactory<index_t, compare_t>;
              recis::cuda::CudaVecParam<index_t*> emb_blocks_ptrs(block_num, stream);
              for (auto i = 0; i < block_num; ++i) {
                emb_blocks_ptrs[i] = emb_blocks[i].data_ptr<index_t>();
              }
              block_filter_kernel_launcher<index_t, Factory>(
                  ids.data_ptr<int64_t>(),
                  const_cast<const index_t**>(emb_blocks_ptrs.data()), num_ids,
                  block_size, static_cast<index_t>(threshold),
                  output.data_ptr<bool>(), Factory());
            }));
      }));
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(cudaSuccess == err, cudaGetErrorString(err));
  return output;
}

void block_insert_cuda(const torch::Tensor ids, const torch::Tensor embedding,
                       std::vector<torch::Tensor>& embedding_blocks,
                       int64_t block_size) {
  TORCH_CHECK(ids.device().type() == torch::kCUDA,
              "Input must be on CUDA device");
  TORCH_CHECK(embedding.device().type() == torch::kCUDA,
              "Embedding must be on CUDA device");
  auto num_ids = ids.numel();
  int64_t num_emb = embedding.size(0);
  int64_t embedding_dim = embedding_blocks[0].size(1);
  auto block_num = embedding_blocks.size();
  if (num_ids == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND3(
      at::ScalarType::Half, at::ScalarType::Long, at::ScalarType::BFloat16,
      embedding.scalar_type(), "block_insert_cuda_impl", ([&] {
        recis::cuda::CudaVecParam<scalar_t*> emb_blocks_ptrs(block_num, stream);
        for (auto i = 0; i < block_num; ++i) {
          emb_blocks_ptrs[i] = embedding_blocks[i].data_ptr<scalar_t>();
        }
        block_insert_kernel_launcher<scalar_t>(
            ids.data_ptr<int64_t>(), (scalar_t**)(emb_blocks_ptrs.data()),
            nullptr, embedding.data_ptr<scalar_t>(), num_ids, num_emb,
            embedding_dim, block_size, false);
      }));
}

void block_insert_with_mask_cuda(const torch::Tensor ids,
                                 const torch::Tensor embedding,
                                 const torch::Tensor mask,
                                 std::vector<torch::Tensor>& embedding_blocks,
                                 int64_t block_size) {
  TORCH_CHECK(ids.device().type() == torch::kCUDA,
              "Input must be on CUDA device");
  TORCH_CHECK(embedding.device().type() == torch::kCUDA,
              "Embedding must be on CUDA device");
  auto num_ids = ids.numel();
  int64_t num_emb = embedding.size(0);
  int64_t embedding_dim = embedding_blocks[0].size(1);
  auto block_num = embedding_blocks.size();
  if (num_ids == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND3(
      at::ScalarType::Half, at::ScalarType::Long, at::ScalarType::BFloat16,
      embedding.scalar_type(), "block_insert_with_mask_cuda_impl", ([&] {
        recis::cuda::CudaVecParam<scalar_t*> emb_blocks_ptrs(block_num, stream);
        for (auto i = 0; i < block_num; ++i) {
          emb_blocks_ptrs[i] = embedding_blocks[i].data_ptr<scalar_t>();
        }
        block_insert_kernel_launcher<scalar_t>(
            ids.data_ptr<int64_t>(), (scalar_t**)(emb_blocks_ptrs.data()),
            mask.data_ptr<bool>(), embedding.data_ptr<scalar_t>(), num_ids,
            num_emb, embedding_dim, block_size, true);
      }));
}

}  // namespace functional
}  // namespace recis
