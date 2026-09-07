#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#include "cuda/cuda_param.cuh"
#include "cuda/utils.cuh"

namespace recis {
namespace functional {

template <typename scalar_t>
void __device__ inline apply_adagrad_kernel(scalar_t& emb, scalar_t& state_sum,
                                            scalar_t grad, scalar_t lr,
                                            const scalar_t eps,
                                            scalar_t weight_decay) {
  grad = grad + weight_decay * emb;
  state_sum += grad * grad;
  emb -= lr * (grad / (sqrtf(state_sum) + eps));
}

template <typename scalar_t, typename pack_t>
__global__ void block_apply_adagrad_cuda_kernel(
    const int64_t* index_vec, scalar_t* grad, scalar_t** emb_blocks,
    scalar_t** state_sum, scalar_t lr, scalar_t eps, scalar_t weight_decay,
    int64_t num_ids, int64_t embedding_dim, int64_t block_size,
    int64_t id_tile_size, int64_t emb_tile_size,
    scalar_t* output_state_sum_values, scalar_t* output_params_before,
    scalar_t* output_params_after, scalar_t* output_grad) {
  int64_t block_idx = blockIdx.x * id_tile_size;
  int64_t emb_idx = threadIdx.x * emb_tile_size;
  int64_t idx = block_idx + threadIdx.y;
  if (idx >= num_ids || emb_idx >= embedding_dim) return;

  auto index = index_vec[idx];
  if (index < 0) {
    CUDA_KERNEL_ASSERT(index == -1);
    return;
  }
  auto block_index = index / block_size;
  auto row_offset = index % block_size * embedding_dim;
  if (emb_idx + emb_tile_size <= embedding_dim) {
    pack_t pack_emb =
        *(pack_t*)(*(emb_blocks + block_index) + row_offset + emb_idx);
    pack_t pack_state_sum =
        *(pack_t*)(*(state_sum + block_index) + row_offset + emb_idx);
    pack_t pack_g = *(pack_t*)(grad + idx * embedding_dim + emb_idx);

    // Store the state_sum, param and grad values before update
    if (output_state_sum_values != nullptr) {
      *(pack_t*)(output_state_sum_values + idx * embedding_dim + emb_idx) =
          pack_state_sum;
    }
    if (output_params_before != nullptr) {
      *(pack_t*)(output_params_before + idx * embedding_dim + emb_idx) =
          pack_emb;
    }
    for (auto i = 0; i < emb_tile_size; ++i) {
      auto& emb = *((scalar_t*)(&pack_emb) + i);
      auto& g = *((scalar_t*)(&pack_g) + i);
      auto grad_elem = g;
      if (output_grad != nullptr) {
        g = grad_elem + weight_decay * emb;
      }
      apply_adagrad_kernel(*((scalar_t*)(&pack_emb) + i),
                           *((scalar_t*)(&pack_state_sum) + i), grad_elem, lr,
                           eps, weight_decay);
    }

    if (output_grad != nullptr) {
      *(pack_t*)(output_grad + idx * embedding_dim + emb_idx) = pack_g;
    }

    // Store the param values after update
    if (output_params_after != nullptr) {
      *(pack_t*)(output_params_after + idx * embedding_dim + emb_idx) =
          pack_emb;
    }

    *(pack_t*)(*(emb_blocks + block_index) + row_offset + emb_idx) = pack_emb;
    *(pack_t*)(*(state_sum + block_index) + row_offset + emb_idx) =
        pack_state_sum;
  } else {
    for (auto i = 0; i < embedding_dim - emb_idx; ++i) {
      scalar_t emb = emb_blocks[block_index][row_offset + emb_idx + i];
      scalar_t state = state_sum[block_index][row_offset + emb_idx + i];
      scalar_t g = grad[idx * embedding_dim + emb_idx + i];

      // Store the state_sum, param and grad values before update
      if (output_state_sum_values != nullptr) {
        output_state_sum_values[idx * embedding_dim + emb_idx + i] = state;
      }
      if (output_params_before != nullptr) {
        output_params_before[idx * embedding_dim + emb_idx + i] = emb;
      }
      if (output_grad != nullptr) {
        output_grad[idx * embedding_dim + emb_idx + i] = g + weight_decay * emb;
      }

      apply_adagrad_kernel(emb, state, g, lr, eps, weight_decay);

      // Store the param values after update
      if (output_params_after != nullptr) {
        output_params_after[idx * embedding_dim + emb_idx + i] = emb;
      }

      emb_blocks[block_index][row_offset + emb_idx + i] = emb;
      state_sum[block_index][row_offset + emb_idx + i] = state;
    }
  }
}

#define BLOCK_APPLY_ADAGRAD_LAUNCH_KERNEL(scalar_t, pack_t)                   \
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();                     \
  block_apply_adagrad_cuda_kernel<scalar_t, pack_t>                           \
      <<<grids, blocks, 0, at::cuda::getCurrentCUDAStream()>>>(               \
          index_vec, grad, emb_blocks, state_sum, lr, eps, weight_decay,      \
          num_ids, embedding_dim, block_size, id_tile_size, emb_tile_size,    \
          output_state_sum_values, output_params_before, output_params_after, \
          output_grad);                                                       \
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

template <typename scalar_t>
void block_apply_adagrad_kernel_launcher(
    const int64_t* index_vec, scalar_t* grad, scalar_t** emb_blocks,
    scalar_t** state_sum, scalar_t lr, scalar_t eps, scalar_t weight_decay,
    int64_t num_ids, int64_t embedding_dim, int64_t block_size,
    scalar_t* output_state_sum_values, scalar_t* output_params_before,
    scalar_t* output_params_after, scalar_t* output_grad) {
  int64_t emb_tile_size, emb_thread_size, id_tile_size, id_blocks,
      real_pack_size;
  recis::cuda::cal_pack_sizes<scalar_t>(num_ids, embedding_dim, emb_tile_size,
                                        emb_thread_size, id_tile_size,
                                        id_blocks, real_pack_size);
  dim3 grids(id_blocks);
  dim3 blocks(emb_thread_size, id_tile_size);
  if (real_pack_size == 2) {
    BLOCK_APPLY_ADAGRAD_LAUNCH_KERNEL(scalar_t, scalar_t);
  } else if (real_pack_size == 4) {
    BLOCK_APPLY_ADAGRAD_LAUNCH_KERNEL(scalar_t, float);
  } else if (real_pack_size == 8) {
    BLOCK_APPLY_ADAGRAD_LAUNCH_KERNEL(scalar_t, float2);
  } else if (real_pack_size == 16) {
    BLOCK_APPLY_ADAGRAD_LAUNCH_KERNEL(scalar_t, float4);
  } else {
    TORCH_CHECK(false, "block_apply_adagrad cuda kernel error pack size");
  }
}

void block_apply_adagrad_gpu(
    const torch::Tensor index, const torch::Tensor grad,
    std::vector<torch::Tensor> emb_blocks, std::vector<torch::Tensor> state_sum,
    double lr, double eps, double weight_decay, int64_t block_size,
    torch::Tensor output_state_sum, torch::Tensor output_params_before,
    torch::Tensor output_params_after, torch::Tensor output_grad) {
  TORCH_CHECK(index.device().type() == torch::kCUDA,
              "Input must be on CUDA device");
  int64_t num_ids = index.numel();
  if (num_ids == 0) {
    return;
  }
  int embedding_dim = emb_blocks[0].size(1);
  auto block_num = emb_blocks.size();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "apply_adagrad_cuda_impl", ([&] {
        recis::cuda::CudaVecParam<scalar_t*> emb_blocks_ptrs(block_num, stream);
        recis::cuda::CudaVecParam<scalar_t*> state_sum_ptrs(block_num, stream);
        for (auto i = 0; i < block_num; ++i) {
          emb_blocks_ptrs[i] = emb_blocks[i].data_ptr<scalar_t>();
          state_sum_ptrs[i] = state_sum[i].data_ptr<scalar_t>();
        }

        scalar_t* output_state_sum_ptr = nullptr;
        if (output_state_sum.defined()) {
          output_state_sum_ptr = output_state_sum.data_ptr<scalar_t>();
        }

        scalar_t* output_params_before_ptr = nullptr;
        if (output_params_before.defined()) {
          output_params_before_ptr = output_params_before.data_ptr<scalar_t>();
        }

        scalar_t* output_params_after_ptr = nullptr;
        if (output_params_after.defined()) {
          output_params_after_ptr = output_params_after.data_ptr<scalar_t>();
        }

        scalar_t* output_grad_ptr = nullptr;
        if (output_grad.defined()) {
          output_grad_ptr = output_grad.data_ptr<scalar_t>();
        }

        block_apply_adagrad_kernel_launcher<scalar_t>(
            index.data_ptr<int64_t>(), grad.data_ptr<scalar_t>(),
            (scalar_t**)(emb_blocks_ptrs.data()),
            (scalar_t**)(state_sum_ptrs.data()), static_cast<scalar_t>(lr),
            static_cast<scalar_t>(eps), static_cast<scalar_t>(weight_decay),
            num_ids, embedding_dim, block_size, output_state_sum_ptr,
            output_params_before_ptr, output_params_after_ptr, output_grad_ptr);
      }));
}

}  // namespace functional
}  // namespace recis

namespace recis {
namespace functional {

// CUDA kernel for adagrad with pre-computed gradient sum of squares
template <typename scalar_t>
void __device__ inline apply_adagrad_sum_kernel(
    scalar_t& emb, scalar_t& state_sum, scalar_t grad, scalar_t grad_sum_sq,
    scalar_t lr, const scalar_t eps) {
  // Use pre-computed gradient sum of squares instead of computing grad * grad
  state_sum += grad_sum_sq;
  emb -= lr * (grad / (sqrtf(state_sum) + eps));
}

template <typename scalar_t, typename pack_t>
__global__ void block_apply_adagrad_sum_cuda_kernel(
    const int64_t* index_vec, scalar_t* grad, scalar_t* grad_sum_sq,
    scalar_t** emb_blocks, scalar_t** state_sum, scalar_t lr, scalar_t eps,
    int64_t num_ids, int64_t embedding_dim, int64_t block_size,
    int64_t id_tile_size, int64_t emb_tile_size,
    scalar_t* output_state_sum_values, scalar_t* output_params_before,
    scalar_t* output_params_after, scalar_t* output_grad) {
  int64_t block_idx = blockIdx.x * id_tile_size;
  int64_t emb_idx = threadIdx.x * emb_tile_size;
  int64_t idx = block_idx + threadIdx.y;
  if (idx >= num_ids || emb_idx >= embedding_dim) return;

  auto index = index_vec[idx];
  if (index < 0) {
    CUDA_KERNEL_ASSERT(index == -1);
    return;
  }
  auto block_index = index / block_size;
  auto row_offset = index % block_size * embedding_dim;
  if (emb_idx + emb_tile_size <= embedding_dim) {
    pack_t pack_emb =
        *(pack_t*)(*(emb_blocks + block_index) + row_offset + emb_idx);
    pack_t pack_state_sum =
        *(pack_t*)(*(state_sum + block_index) + row_offset + emb_idx);
    pack_t pack_g = *(pack_t*)(grad + idx * embedding_dim + emb_idx);
    pack_t pack_g_sq = *(pack_t*)(grad_sum_sq + idx * embedding_dim + emb_idx);

    // Store the state_sum, param and grad values before update
    if (output_state_sum_values != nullptr) {
      *(pack_t*)(output_state_sum_values + idx * embedding_dim + emb_idx) =
          pack_state_sum;
    }
    if (output_params_before != nullptr) {
      *(pack_t*)(output_params_before + idx * embedding_dim + emb_idx) =
          pack_emb;
    }
    if (output_grad != nullptr) {
      *(pack_t*)(output_grad + idx * embedding_dim + emb_idx) = pack_g;
    }

    for (auto i = 0; i < emb_tile_size; ++i) {
      apply_adagrad_sum_kernel(
          *((scalar_t*)(&pack_emb) + i), *((scalar_t*)(&pack_state_sum) + i),
          *((scalar_t*)(&pack_g) + i), *((scalar_t*)(&pack_g_sq) + i), lr, eps);
    }

    // Store the param values after update
    if (output_params_after != nullptr) {
      *(pack_t*)(output_params_after + idx * embedding_dim + emb_idx) =
          pack_emb;
    }

    *(pack_t*)(*(emb_blocks + block_index) + row_offset + emb_idx) = pack_emb;
    *(pack_t*)(*(state_sum + block_index) + row_offset + emb_idx) =
        pack_state_sum;
  } else {
    for (auto i = 0; i < embedding_dim - emb_idx; ++i) {
      scalar_t emb = emb_blocks[block_index][row_offset + emb_idx + i];
      scalar_t state = state_sum[block_index][row_offset + emb_idx + i];
      scalar_t g = grad[idx * embedding_dim + emb_idx + i];
      scalar_t g_sq = grad_sum_sq[idx * embedding_dim + emb_idx + i];

      // Store the state_sum, param and grad values before update
      if (output_state_sum_values != nullptr) {
        output_state_sum_values[idx * embedding_dim + emb_idx + i] = state;
      }
      if (output_params_before != nullptr) {
        output_params_before[idx * embedding_dim + emb_idx + i] = emb;
      }
      if (output_grad != nullptr) {
        output_grad[idx * embedding_dim + emb_idx + i] = g;
      }

      apply_adagrad_sum_kernel(emb, state, g, g_sq, lr, eps);

      // Store the param values after update
      if (output_params_after != nullptr) {
        output_params_after[idx * embedding_dim + emb_idx + i] = emb;
      }

      emb_blocks[block_index][row_offset + emb_idx + i] = emb;
      state_sum[block_index][row_offset + emb_idx + i] = state;
    }
  }
}

#define BLOCK_APPLY_ADAGRAD_SUM_LAUNCH_KERNEL(scalar_t, pack_t)               \
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();                     \
  block_apply_adagrad_sum_cuda_kernel<scalar_t, pack_t>                       \
      <<<grids, blocks, 0, at::cuda::getCurrentCUDAStream()>>>(               \
          index_vec, grad, grad_sum_sq, emb_blocks, state_sum, lr, eps,       \
          num_ids, embedding_dim, block_size, id_tile_size, emb_tile_size,    \
          output_state_sum_values, output_params_before, output_params_after, \
          output_grad);                                                       \
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

template <typename scalar_t>
void block_apply_adagrad_sum_kernel_launcher(
    const int64_t* index_vec, scalar_t* grad, scalar_t* grad_sum_sq,
    scalar_t** emb_blocks, scalar_t** state_sum, scalar_t lr, scalar_t eps,
    int64_t num_ids, int64_t embedding_dim, int64_t block_size,
    scalar_t* output_state_sum_values, scalar_t* output_params_before,
    scalar_t* output_params_after, scalar_t* output_grad) {
  int64_t emb_tile_size, emb_thread_size, id_tile_size, id_blocks,
      real_pack_size;
  recis::cuda::cal_pack_sizes<scalar_t>(num_ids, embedding_dim, emb_tile_size,
                                        emb_thread_size, id_tile_size,
                                        id_blocks, real_pack_size);
  dim3 grids(id_blocks);
  dim3 blocks(emb_thread_size, id_tile_size);
  if (real_pack_size == 2) {
    BLOCK_APPLY_ADAGRAD_SUM_LAUNCH_KERNEL(scalar_t, scalar_t);
  } else if (real_pack_size == 4) {
    BLOCK_APPLY_ADAGRAD_SUM_LAUNCH_KERNEL(scalar_t, float);
  } else if (real_pack_size == 8) {
    BLOCK_APPLY_ADAGRAD_SUM_LAUNCH_KERNEL(scalar_t, float2);
  } else if (real_pack_size == 16) {
    BLOCK_APPLY_ADAGRAD_SUM_LAUNCH_KERNEL(scalar_t, float4);
  } else {
    TORCH_CHECK(false, "block_apply_adagrad_sum cuda kernel error pack size");
  }
}

void block_apply_adagrad_sum_gpu(
    const torch::Tensor index, const torch::Tensor grad,
    const torch::Tensor grad_sum_sq, std::vector<torch::Tensor> emb_blocks,
    std::vector<torch::Tensor> state_sum, double lr, double eps,
    int64_t block_size, torch::Tensor output_state_sum,
    torch::Tensor output_params_before, torch::Tensor output_params_after,
    torch::Tensor output_grad) {
  TORCH_CHECK(index.device().type() == torch::kCUDA,
              "Input must be on CUDA device");
  int64_t num_ids = index.numel();
  if (num_ids == 0) {
    return;
  }
  int embedding_dim = emb_blocks[0].size(1);
  auto block_num = emb_blocks.size();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "apply_adagrad_sum_cuda_impl", ([&] {
        recis::cuda::CudaVecParam<scalar_t*> emb_blocks_ptrs(block_num, stream);
        recis::cuda::CudaVecParam<scalar_t*> state_sum_ptrs(block_num, stream);
        for (auto i = 0; i < block_num; ++i) {
          emb_blocks_ptrs[i] = emb_blocks[i].data_ptr<scalar_t>();
          state_sum_ptrs[i] = state_sum[i].data_ptr<scalar_t>();
        }

        scalar_t* output_state_sum_ptr = nullptr;
        if (output_state_sum.defined()) {
          output_state_sum_ptr = output_state_sum.data_ptr<scalar_t>();
        }

        scalar_t* output_params_before_ptr = nullptr;
        if (output_params_before.defined()) {
          output_params_before_ptr = output_params_before.data_ptr<scalar_t>();
        }

        scalar_t* output_params_after_ptr = nullptr;
        if (output_params_after.defined()) {
          output_params_after_ptr = output_params_after.data_ptr<scalar_t>();
        }

        scalar_t* output_grad_ptr = nullptr;
        if (output_grad.defined()) {
          output_grad_ptr = output_grad.data_ptr<scalar_t>();
        }

        block_apply_adagrad_sum_kernel_launcher<scalar_t>(
            index.data_ptr<int64_t>(), grad.data_ptr<scalar_t>(),
            grad_sum_sq.data_ptr<scalar_t>(),
            (scalar_t**)(emb_blocks_ptrs.data()),
            (scalar_t**)(state_sum_ptrs.data()), static_cast<scalar_t>(lr),
            static_cast<scalar_t>(eps), num_ids, embedding_dim, block_size,
            output_state_sum_ptr, output_params_before_ptr,
            output_params_after_ptr, output_grad_ptr);
      }));
}

}  // namespace functional
}  // namespace recis
