#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include "cuda/cuda_param.cuh"
#include "cuda/element_wise_kernel.cuh"
#include "cuda/packer.cuh"
#include "cuda/utils.cuh"
namespace recis {
namespace functional {
using namespace recis::cuda;
template <typename scalar_t>
__global__ void adam_tf_apply_cuda_kernel(scalar_t* param, scalar_t* grad,
                                          scalar_t* avg, scalar_t* avg_sq,
                                          float step, float lr, float b1,
                                          float b2, float eps,
                                          const int64_t param_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float b1_power = powf(b1, step);
  float b2_power = powf(b2, step);
  float alpha = -lr / (1. - b1_power) * sqrtf(1. - b2_power);
  if (i < param_size) {
    avg[i] = avg[i] * b1 + grad[i] * (1. - b1);
    avg_sq[i] = avg_sq[i] * b2 + (1. - b2) * grad[i] * grad[i];
    param[i] = param[i] + alpha * (avg[i] / (sqrtf(avg_sq[i]) + eps));
  }
}

template <typename scalar_t>
__global__ void fused_adamw_tf_apply_cuda_kernel(
    scalar_t** params, scalar_t** grads, scalar_t** avg, scalar_t** avg_sq,
    float* steps, float weight_decay, float lr, float b1, float b2, float eps,
    int64_t* param_sizes) {
  int64_t vec_id = blockIdx.y;
  int64_t size_local = param_sizes[vec_id];
  int64_t threads_num = blockDim.x * gridDim.x;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  static constexpr int pack_size =
      recis::cuda::select_pack_size<scalar_t, scalar_t>::value;
  using PK = recis::cuda::Packer<scalar_t, pack_size>;
  const auto pack_alignment = alignof(typename PK::type);
  const bool pack_aligned =
      reinterpret_cast<std::uintptr_t>(params[vec_id]) % pack_alignment == 0 &&
      reinterpret_cast<std::uintptr_t>(grads[vec_id]) % pack_alignment == 0 &&
      reinterpret_cast<std::uintptr_t>(avg[vec_id]) % pack_alignment == 0 &&
      reinterpret_cast<std::uintptr_t>(avg_sq[vec_id]) % pack_alignment == 0;
  float b1_power = powf(b1, steps[vec_id]);
  float b2_power = powf(b2, steps[vec_id]);
  float alpha = -lr / (1. - b1_power) * sqrtf(1. - b2_power);
  for (int64_t index = tid; index * pack_size < size_local;
       index += threads_num) {
    int64_t idx = index * pack_size;
    if (pack_aligned && idx + pack_size <= size_local) {
      typename PK::type param_vec, grad_vec, a_vec, a_sq_vec;
      PK::load(params[vec_id] + idx, param_vec);
      PK::load(grads[vec_id] + idx, grad_vec);
      PK::load(avg[vec_id] + idx, a_vec);
      PK::load(avg_sq[vec_id] + idx, a_sq_vec);

#pragma unroll
      for (int64_t j = 0; j < pack_size; ++j) {
        auto a_val = PK::get_element(a_vec, j);
        auto a_sq_val = PK::get_element(a_sq_vec, j);
        auto grad_val = PK::get_element(grad_vec, j);
        auto param_val = PK::get_element(param_vec, j);
        param_val = param_val - weight_decay * param_val;
        a_val = a_val * b1 + grad_val * (1. - b1);
        a_sq_val = a_sq_val * b2 + (1. - b2) * grad_val * grad_val;
        param_val = param_val + alpha * (a_val / (sqrtf(a_sq_val) + eps));
        PK::set_element(a_vec, j, a_val);
        PK::set_element(a_sq_vec, j, a_sq_val);
        PK::set_element(param_vec, j, param_val);
      }

      PK::store(params[vec_id] + idx, param_vec);
      PK::store(avg[vec_id] + idx, a_vec);
      PK::store(avg_sq[vec_id] + idx, a_sq_vec);

    } else {
      const int64_t end = min(idx + pack_size, size_local);
      for (int64_t i = idx; i < end; i++) {
        avg[vec_id][i] = avg[vec_id][i] * b1 + grads[vec_id][i] * (1. - b1);
        avg_sq[vec_id][i] = avg_sq[vec_id][i] * b2 +
                            (1. - b2) * grads[vec_id][i] * grads[vec_id][i];
        params[vec_id][i] =
            params[vec_id][i] * (1. - weight_decay) +
            alpha * (avg[vec_id][i] / (sqrtf(avg_sq[vec_id][i]) + eps));
      }
    }
  }
}

void fused_adamw_tf_apply_cuda(std::vector<torch::Tensor> params,
                               std::vector<torch::Tensor> grads,
                               std::vector<torch::Tensor> avg,
                               std::vector<torch::Tensor> avg_sq,
                               std::vector<torch::Tensor> state_steps,
                               float weight_decay, float lr, float beta1,
                               float beta2, float eps) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int device;
  cudaGetDevice(&device);
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

  int64_t N = params.size();
  int64_t max_size = 0;
  for (int64_t i = 0; i < N; ++i) {
    max_size = max(max_size, params[i].numel());
  }
  int64_t block_num = (max_size + KBLOCK_SIZE - 1) / KBLOCK_SIZE;
  if (block_num > sm_count * 8) block_num = sm_count * 8;
  dim3 grid(block_num, N);
  dim3 block(KBLOCK_SIZE);
  using scalar_t = float;
  recis::cuda::CudaVecParam<scalar_t*> params_ptrs(N, stream);
  recis::cuda::CudaVecParam<scalar_t*> grads_ptrs(N, stream);
  recis::cuda::CudaVecParam<scalar_t*> avg_ptrs(N, stream);
  recis::cuda::CudaVecParam<scalar_t*> avg_sq_ptrs(N, stream);
  recis::cuda::CudaVecParam<float> steps(N, stream);
  recis::cuda::CudaVecParam<int64_t> param_sizes_cuda(N, stream);
  for (int64_t i = 0; i < N; ++i) {
    params_ptrs[i] = params[i].data_ptr<scalar_t>();
    grads_ptrs[i] = grads[i].data_ptr<scalar_t>();
    avg_ptrs[i] = avg[i].data_ptr<scalar_t>();
    avg_sq_ptrs[i] = avg_sq[i].data_ptr<scalar_t>();
    param_sizes_cuda[i] = params[i].numel();
    steps[i] = state_steps[i].item<float>();
  }
  fused_adamw_tf_apply_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
      params_ptrs.data(), grads_ptrs.data(), avg_ptrs.data(),
      avg_sq_ptrs.data(), steps.data(), weight_decay, lr, beta1, beta2, eps,
      param_sizes_cuda.data());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void adam_tf_apply_cuda(torch::Tensor param, torch::Tensor grad,
                        torch::Tensor avg, torch::Tensor avg_sq, float step,
                        float lr, float beta1, float beta2, float eps,
                        int64_t param_size) {
  const int threads = 128;
  const int blocks = (param_size + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      param.scalar_type(), "adam_tf_apply_cuda", ([&] {
        adam_tf_apply_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
            avg.data_ptr<scalar_t>(), avg_sq.data_ptr<scalar_t>(), step, lr,
            beta1, beta2, eps, param_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}

}  // namespace functional
}  // namespace recis
