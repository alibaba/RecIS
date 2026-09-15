#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstring>
#include <cub/block/block_reduce.cuh>

namespace recis {
namespace functional {

constexpr int kRowWiseAdagradThreads = 256;

template <typename scalar_t>
struct RowWiseAdagradBlockPointers {
  scalar_t *embedding;
  scalar_t *state_sum;
};

template <typename scalar_t, typename acc_t>
__global__ void block_apply_row_wise_adagrad_cuda_kernel(
    const int64_t *index_vec, const scalar_t *grad,
    const RowWiseAdagradBlockPointers<scalar_t> *block_ptrs, acc_t lr,
    acc_t eps, acc_t weight_decay, int64_t num_ids, int64_t embedding_dim,
    int64_t block_size) {
  using BlockReduce = cub::BlockReduce<acc_t, kRowWiseAdagradThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ acc_t denominator;

  const int64_t grad_row_index = blockIdx.x;
  if (grad_row_index >= num_ids) {
    return;
  }
  const int64_t index = index_vec[grad_row_index];
  if (index < 0) {
    CUDA_KERNEL_ASSERT(index == -1);
    return;
  }
  const int64_t block_index = index / block_size;
  const int64_t row_index = index % block_size;
  const int64_t emb_offset = row_index * embedding_dim;
  const auto block = block_ptrs[block_index];

  acc_t local_square_sum = 0;
  for (int64_t column = threadIdx.x; column < embedding_dim;
       column += blockDim.x) {
    const acc_t adjusted_grad =
        static_cast<acc_t>(grad[grad_row_index * embedding_dim + column]) +
        weight_decay * static_cast<acc_t>(block.embedding[emb_offset + column]);
    local_square_sum += adjusted_grad * adjusted_grad;
  }
  const acc_t square_sum = BlockReduce(reduce_storage).Sum(local_square_sum);
  if (threadIdx.x == 0) {
    const acc_t new_state = static_cast<acc_t>(block.state_sum[row_index]) +
                            square_sum / static_cast<acc_t>(embedding_dim);
    block.state_sum[row_index] = static_cast<scalar_t>(new_state);
    denominator = sqrt(new_state) + eps;
  }
  __syncthreads();

  for (int64_t column = threadIdx.x; column < embedding_dim;
       column += blockDim.x) {
    const int64_t offset = emb_offset + column;
    const acc_t emb = static_cast<acc_t>(block.embedding[offset]);
    const acc_t adjusted_grad =
        static_cast<acc_t>(grad[grad_row_index * embedding_dim + column]) +
        weight_decay * emb;
    block.embedding[offset] =
        static_cast<scalar_t>(emb - lr * adjusted_grad / denominator);
  }
}

void block_apply_row_wise_adagrad_gpu(const torch::Tensor index,
                                      const torch::Tensor grad,
                                      std::vector<torch::Tensor> emb_blocks,
                                      std::vector<torch::Tensor> state_sum,
                                      double lr, double eps,
                                      double weight_decay, int64_t block_size) {
  TORCH_CHECK(index.device().type() == torch::kCUDA,
              "Input must be on a CUDA device");
  const int64_t num_ids = index.numel();
  if (num_ids == 0) {
    return;
  }
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(!emb_blocks.empty(),
              "embedding blocks must not be empty for a non-empty update");
  TORCH_CHECK(
      emb_blocks.front().dim() > 0 && emb_blocks.front().size(0) == block_size,
      "embedding block leading dimension must equal block_size");
  const int64_t embedding_dim = emb_blocks.front().numel() / block_size;
  TORCH_CHECK(embedding_dim > 0, "embedding row width must be positive");
  TORCH_CHECK(grad.dim() > 0 && grad.size(0) == num_ids &&
                  grad.numel() == num_ids * embedding_dim,
              "gradient shape does not match the embedding row shape");
  const auto block_num = emb_blocks.size();
  const c10::cuda::CUDAGuard device_guard(index.device());
  const auto cuda_stream = at::cuda::getCurrentCUDAStream(index.get_device());
  cudaStream_t stream = cuda_stream;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "apply_row_wise_adagrad_cuda_impl", ([&] {
        using acc_t = at::acc_type<scalar_t, true>;
        using BlockPointers = RowWiseAdagradBlockPointers<scalar_t>;
        const int64_t pointer_bytes =
            static_cast<int64_t>(block_num * sizeof(BlockPointers));
        const auto byte_options = torch::TensorOptions().dtype(torch::kUInt8);
        auto block_ptrs_device =
            torch::empty({pointer_bytes}, byte_options.device(index.device()));
        {
          auto block_ptrs_host = torch::empty(
              {pointer_bytes},
              byte_options.device(torch::kCPU).pinned_memory(true));
          auto *host_bytes = block_ptrs_host.data_ptr<uint8_t>();
          for (size_t i = 0; i < block_num; ++i) {
            const BlockPointers block{emb_blocks[i].data_ptr<scalar_t>(),
                                      state_sum[i].data_ptr<scalar_t>()};
            std::memcpy(host_bytes + i * sizeof(BlockPointers), &block,
                        sizeof(BlockPointers));
          }
          C10_CUDA_CHECK(cudaMemcpyAsync(
              block_ptrs_device.data_ptr(), block_ptrs_host.data_ptr(),
              pointer_bytes, cudaMemcpyHostToDevice, stream));
          TORCH_INTERNAL_ASSERT(at::cuda::CachingHostAllocator_recordEvent(
              block_ptrs_host.data_ptr(),
              block_ptrs_host.storage().data_ptr().get_context(), cuda_stream));
        }
        block_apply_row_wise_adagrad_cuda_kernel<scalar_t, acc_t>
            <<<num_ids, kRowWiseAdagradThreads, 0, stream>>>(
                index.data_ptr<int64_t>(), grad.data_ptr<scalar_t>(),
                reinterpret_cast<const BlockPointers *>(
                    block_ptrs_device.data_ptr()),
                static_cast<acc_t>(lr), static_cast<acc_t>(eps),
                static_cast<acc_t>(weight_decay), num_ids, embedding_dim,
                block_size);
        c10::cuda::CUDACachingAllocator::recordStream(
            block_ptrs_device.storage().data_ptr(), cuda_stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}

template <typename scalar_t, typename acc_t>
__global__ void block_apply_row_wise_adagrad_sum_cuda_kernel(
    const int64_t *index_vec, const scalar_t *grad, const scalar_t *grad_sum_sq,
    const RowWiseAdagradBlockPointers<scalar_t> *block_ptrs, acc_t lr,
    acc_t eps, int64_t num_ids, int64_t embedding_dim, int64_t block_size) {
  using BlockReduce = cub::BlockReduce<acc_t, kRowWiseAdagradThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ acc_t denominator;

  const int64_t grad_row_index = blockIdx.x;
  if (grad_row_index >= num_ids) {
    return;
  }
  const int64_t index = index_vec[grad_row_index];
  if (index < 0) {
    CUDA_KERNEL_ASSERT(index == -1);
    return;
  }
  const int64_t block_index = index / block_size;
  const int64_t row_index = index % block_size;
  const int64_t emb_offset = row_index * embedding_dim;
  const auto block = block_ptrs[block_index];

  acc_t local_square_sum = 0;
  for (int64_t column = threadIdx.x; column < embedding_dim;
       column += blockDim.x) {
    local_square_sum += static_cast<acc_t>(
        grad_sum_sq[grad_row_index * embedding_dim + column]);
  }
  const acc_t square_sum = BlockReduce(reduce_storage).Sum(local_square_sum);
  if (threadIdx.x == 0) {
    const acc_t new_state = static_cast<acc_t>(block.state_sum[row_index]) +
                            square_sum / static_cast<acc_t>(embedding_dim);
    block.state_sum[row_index] = static_cast<scalar_t>(new_state);
    denominator = sqrt(new_state) + eps;
  }
  __syncthreads();

  for (int64_t column = threadIdx.x; column < embedding_dim;
       column += blockDim.x) {
    const int64_t offset = emb_offset + column;
    const acc_t emb = static_cast<acc_t>(block.embedding[offset]);
    const acc_t g =
        static_cast<acc_t>(grad[grad_row_index * embedding_dim + column]);
    block.embedding[offset] = static_cast<scalar_t>(emb - lr * g / denominator);
  }
}

void block_apply_row_wise_adagrad_sum_gpu(const torch::Tensor index,
                                          const torch::Tensor grad,
                                          const torch::Tensor grad_sum_sq,
                                          std::vector<torch::Tensor> emb_blocks,
                                          std::vector<torch::Tensor> state_sum,
                                          double lr, double eps,
                                          int64_t block_size) {
  TORCH_CHECK(index.device().type() == torch::kCUDA,
              "Input must be on a CUDA device");
  const int64_t num_ids = index.numel();
  if (num_ids == 0) {
    return;
  }
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(!emb_blocks.empty(),
              "embedding blocks must not be empty for a non-empty update");
  TORCH_CHECK(
      emb_blocks.front().dim() > 0 && emb_blocks.front().size(0) == block_size,
      "embedding block leading dimension must equal block_size");
  const int64_t embedding_dim = emb_blocks.front().numel() / block_size;
  TORCH_CHECK(embedding_dim > 0, "embedding row width must be positive");
  TORCH_CHECK(grad.dim() > 0 && grad.size(0) == num_ids &&
                  grad.numel() == num_ids * embedding_dim,
              "gradient shape does not match the embedding row shape");
  TORCH_CHECK(grad_sum_sq.dim() > 0 && grad_sum_sq.size(0) == num_ids &&
                  grad_sum_sq.numel() == num_ids * embedding_dim,
              "gradient-square shape does not match the embedding row shape");
  TORCH_CHECK(grad_sum_sq.scalar_type() == grad.scalar_type(),
              "gradient-square dtype must match gradient dtype");
  const auto block_num = emb_blocks.size();
  const c10::cuda::CUDAGuard device_guard(index.device());
  const auto cuda_stream = at::cuda::getCurrentCUDAStream(index.get_device());
  cudaStream_t stream = cuda_stream;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "apply_row_wise_adagrad_sum_cuda_impl", ([&] {
        using acc_t = at::acc_type<scalar_t, true>;
        using BlockPointers = RowWiseAdagradBlockPointers<scalar_t>;
        const int64_t pointer_bytes =
            static_cast<int64_t>(block_num * sizeof(BlockPointers));
        const auto byte_options = torch::TensorOptions().dtype(torch::kUInt8);
        auto block_ptrs_device =
            torch::empty({pointer_bytes}, byte_options.device(index.device()));
        {
          auto block_ptrs_host = torch::empty(
              {pointer_bytes},
              byte_options.device(torch::kCPU).pinned_memory(true));
          auto *host_bytes = block_ptrs_host.data_ptr<uint8_t>();
          for (size_t i = 0; i < block_num; ++i) {
            const BlockPointers block{emb_blocks[i].data_ptr<scalar_t>(),
                                      state_sum[i].data_ptr<scalar_t>()};
            std::memcpy(host_bytes + i * sizeof(BlockPointers), &block,
                        sizeof(BlockPointers));
          }
          C10_CUDA_CHECK(cudaMemcpyAsync(
              block_ptrs_device.data_ptr(), block_ptrs_host.data_ptr(),
              pointer_bytes, cudaMemcpyHostToDevice, stream));
          TORCH_INTERNAL_ASSERT(at::cuda::CachingHostAllocator_recordEvent(
              block_ptrs_host.data_ptr(),
              block_ptrs_host.storage().data_ptr().get_context(), cuda_stream));
        }
        block_apply_row_wise_adagrad_sum_cuda_kernel<scalar_t, acc_t>
            <<<num_ids, kRowWiseAdagradThreads, 0, stream>>>(
                index.data_ptr<int64_t>(), grad.data_ptr<scalar_t>(),
                grad_sum_sq.data_ptr<scalar_t>(),
                reinterpret_cast<const BlockPointers *>(
                    block_ptrs_device.data_ptr()),
                static_cast<acc_t>(lr), static_cast<acc_t>(eps), num_ids,
                embedding_dim, block_size);
        c10::cuda::CUDACachingAllocator::recordStream(
            block_ptrs_device.storage().data_ptr(), cuda_stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}

}  // namespace functional
}  // namespace recis
