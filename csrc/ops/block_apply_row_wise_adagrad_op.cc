#include "ops/block_apply_row_wise_adagrad_op.h"

#include <cmath>

#include "ATen/Dispatch.h"
#include "ATen/OpMathType.h"
#include "ATen/Parallel.h"
#include "c10/util/irange.h"

namespace recis {
namespace functional {

template <class scalar_t>
struct BlocksApplyRowWiseAdagradFunctor {
  using opmath_t = at::opmath_type<scalar_t>;

  BlocksApplyRowWiseAdagradFunctor(int64_t embedding_dim, int64_t block_size,
                                   const int64_t *index_vec,
                                   std::vector<torch::Tensor> &emb_blocks,
                                   scalar_t *grad,
                                   std::vector<torch::Tensor> &state_sum,
                                   double lr, double eps, double weight_decay)
      : embedding_dim_(embedding_dim),
        block_size_(block_size),
        index_vec_(index_vec),
        emb_blocks_(emb_blocks),
        grad_(grad),
        state_sum_(state_sum),
        lr_(lr),
        eps_(eps),
        weight_decay_(weight_decay) {}

  void operator()(int64_t begin, int64_t end) const {
    for (auto i : c10::irange(begin, end)) {
      const auto index = index_vec_[i];
      if (index < 0) {
        TORCH_CHECK(index == -1, "row-wise Adagrad index must be >= -1");
        continue;
      }
      const auto block_index = index / block_size_;
      const auto row_index = index % block_size_;
      auto *emb = emb_blocks_[block_index].data_ptr<scalar_t>() +
                  row_index * embedding_dim_;
      auto *grad_row = grad_ + i * embedding_dim_;
      auto *state = state_sum_[block_index].data_ptr<scalar_t>() + row_index;

      opmath_t square_sum = 0;
      for (auto column : c10::irange(embedding_dim_)) {
        const opmath_t adjusted_grad = static_cast<opmath_t>(grad_row[column]) +
                                       static_cast<opmath_t>(weight_decay_) *
                                           static_cast<opmath_t>(emb[column]);
        square_sum += adjusted_grad * adjusted_grad;
      }
      const opmath_t new_state =
          static_cast<opmath_t>(*state) + square_sum / embedding_dim_;
      *state = static_cast<scalar_t>(new_state);
      const opmath_t denominator = std::sqrt(new_state) + eps_;
      for (auto column : c10::irange(embedding_dim_)) {
        const opmath_t adjusted_grad = static_cast<opmath_t>(grad_row[column]) +
                                       static_cast<opmath_t>(weight_decay_) *
                                           static_cast<opmath_t>(emb[column]);
        emb[column] = static_cast<scalar_t>(static_cast<opmath_t>(emb[column]) -
                                            static_cast<opmath_t>(lr_) *
                                                adjusted_grad / denominator);
      }
    }
  }

 private:
  int64_t embedding_dim_;
  int64_t block_size_;
  const int64_t *index_vec_;
  std::vector<torch::Tensor> &emb_blocks_;
  scalar_t *grad_;
  std::vector<torch::Tensor> &state_sum_;
  double lr_;
  double eps_;
  double weight_decay_;
};

void block_apply_row_wise_adagrad_cpu(const torch::Tensor index,
                                      const torch::Tensor grad,
                                      std::vector<torch::Tensor> emb_blocks,
                                      std::vector<torch::Tensor> state_sum,
                                      double lr, double eps,
                                      double weight_decay, int64_t block_size) {
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
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "apply_row_wise_adagrad_cpu_impl", ([&] {
        BlocksApplyRowWiseAdagradFunctor<scalar_t> functor(
            embedding_dim, block_size, index.data_ptr<int64_t>(), emb_blocks,
            grad.data_ptr<scalar_t>(), state_sum, lr, eps, weight_decay);
        at::parallel_for(0, index.numel(), 0, functor);
      }));
}

void block_apply_row_wise_adagrad(const torch::Tensor index,
                                  const torch::Tensor grad,
                                  std::vector<torch::Tensor> emb_blocks,
                                  torch::Tensor step,
                                  std::vector<torch::Tensor> state_sum,
                                  double lr, double lr_decay, double eps,
                                  double weight_decay, int64_t block_size) {
  step.add_(1);
  const auto step_item = step.item<int64_t>();
  TORCH_CHECK(step_item >= 1);
  lr /= 1 + (step_item - 1) * lr_decay;
  if (index.device().type() == torch::kCUDA) {
    block_apply_row_wise_adagrad_gpu(index, grad, std::move(emb_blocks),
                                     std::move(state_sum), lr, eps,
                                     weight_decay, block_size);
  } else {
    block_apply_row_wise_adagrad_cpu(index, grad, std::move(emb_blocks),
                                     std::move(state_sum), lr, eps,
                                     weight_decay, block_size);
  }
}

template <class scalar_t>
struct BlocksApplyRowWiseAdagradSumFunctor {
  using opmath_t = at::opmath_type<scalar_t>;

  BlocksApplyRowWiseAdagradSumFunctor(int64_t embedding_dim, int64_t block_size,
                                      const int64_t *index_vec,
                                      std::vector<torch::Tensor> &emb_blocks,
                                      scalar_t *grad, scalar_t *grad_sum_sq,
                                      std::vector<torch::Tensor> &state_sum,
                                      double lr, double eps)
      : embedding_dim_(embedding_dim),
        block_size_(block_size),
        index_vec_(index_vec),
        emb_blocks_(emb_blocks),
        grad_(grad),
        grad_sum_sq_(grad_sum_sq),
        state_sum_(state_sum),
        lr_(lr),
        eps_(eps) {}

  void operator()(int64_t begin, int64_t end) const {
    for (auto i : c10::irange(begin, end)) {
      const auto index = index_vec_[i];
      if (index < 0) {
        TORCH_CHECK(index == -1, "row-wise AdagradSum index must be >= -1");
        continue;
      }
      const auto block_index = index / block_size_;
      const auto row_index = index % block_size_;
      auto *emb = emb_blocks_[block_index].data_ptr<scalar_t>() +
                  row_index * embedding_dim_;
      auto *grad_row = grad_ + i * embedding_dim_;
      auto *grad_sum_sq_row = grad_sum_sq_ + i * embedding_dim_;
      auto *state = state_sum_[block_index].data_ptr<scalar_t>() + row_index;

      opmath_t square_sum = 0;
      for (auto column : c10::irange(embedding_dim_)) {
        square_sum += static_cast<opmath_t>(grad_sum_sq_row[column]);
      }
      const opmath_t new_state =
          static_cast<opmath_t>(*state) + square_sum / embedding_dim_;
      *state = static_cast<scalar_t>(new_state);
      const opmath_t denominator = std::sqrt(new_state) + eps_;
      for (auto column : c10::irange(embedding_dim_)) {
        emb[column] = static_cast<scalar_t>(
            static_cast<opmath_t>(emb[column]) -
            static_cast<opmath_t>(lr_) *
                static_cast<opmath_t>(grad_row[column]) / denominator);
      }
    }
  }

 private:
  int64_t embedding_dim_;
  int64_t block_size_;
  const int64_t *index_vec_;
  std::vector<torch::Tensor> &emb_blocks_;
  scalar_t *grad_;
  scalar_t *grad_sum_sq_;
  std::vector<torch::Tensor> &state_sum_;
  double lr_;
  double eps_;
};

void block_apply_row_wise_adagrad_sum_cpu(const torch::Tensor index,
                                          const torch::Tensor grad,
                                          const torch::Tensor grad_sum_sq,
                                          std::vector<torch::Tensor> emb_blocks,
                                          std::vector<torch::Tensor> state_sum,
                                          double lr, double eps,
                                          int64_t block_size) {
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
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "apply_row_wise_adagrad_sum_cpu_impl", ([&] {
        BlocksApplyRowWiseAdagradSumFunctor<scalar_t> functor(
            embedding_dim, block_size, index.data_ptr<int64_t>(), emb_blocks,
            grad.data_ptr<scalar_t>(), grad_sum_sq.data_ptr<scalar_t>(),
            state_sum, lr, eps);
        at::parallel_for(0, index.numel(), 0, functor);
      }));
}

void block_apply_row_wise_adagrad_sum(
    const torch::Tensor index, const torch::Tensor grad,
    const torch::Tensor grad_sum_sq, std::vector<torch::Tensor> emb_blocks,
    torch::Tensor step, std::vector<torch::Tensor> state_sum, double lr,
    double lr_decay, double eps, int64_t block_size) {
  step.add_(1);
  const auto step_item = step.item<int64_t>();
  TORCH_CHECK(step_item >= 1);
  lr /= 1 + (step_item - 1) * lr_decay;
  if (index.device().type() == torch::kCUDA) {
    block_apply_row_wise_adagrad_sum_gpu(
        index, grad, grad_sum_sq, std::move(emb_blocks), std::move(state_sum),
        lr, eps, block_size);
  } else {
    block_apply_row_wise_adagrad_sum_cpu(
        index, grad, grad_sum_sq, std::move(emb_blocks), std::move(state_sum),
        lr, eps, block_size);
  }
}

}  // namespace functional
}  // namespace recis
