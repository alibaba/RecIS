#include "block_apply_adagrad_op.h"

namespace recis {
namespace functional {

template <class TEmb>
struct BlocksApplyAdagradFunctor {
  BlocksApplyAdagradFunctor(int64_t embedding_dim, int64_t block_size,
                            const int64_t *index_vec,
                            std::vector<torch::Tensor> &emb_blocks, TEmb *grad,
                            std::vector<torch::Tensor> &state_sum, double lr,
                            double eps, double weight_decay,
                            TEmb *output_state_sum_values,
                            TEmb *output_params_before,
                            TEmb *output_params_after, TEmb *output_grad)
      : embedding_dim_(embedding_dim),
        block_size_(block_size),
        index_vec_(index_vec),
        emb_blocks_(emb_blocks),
        grad_(grad),
        state_sum_(state_sum),
        lr_(lr),
        eps_(eps),
        weight_decay_(weight_decay),
        output_state_sum_values_(output_state_sum_values),
        output_params_before_(output_params_before),
        output_params_after_(output_params_after),
        output_grad_(output_grad) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto i : c10::irange(beg, end)) {
      auto index = index_vec_[i];  // embedding index
      if (index < 0) {
        TORCH_CHECK(
            index == -1,
            "index of BlocksApplyAdagradFunctor must be >= -1, but get ",
            index);
        continue;
      }
      auto block_index = index / block_size_;
      auto row_index = index % block_size_;
      auto offset = row_index * embedding_dim_;
      auto emb_vec = emb_blocks_[block_index].data_ptr<TEmb>() + offset;
      auto state_sum_vec = state_sum_[block_index].data_ptr<TEmb>() + offset;
      auto grad_vec = grad_ + i * embedding_dim_;
      for (auto element_index : c10::irange(embedding_dim_)) {
        auto &emb_elem = emb_vec[element_index];
        auto grad_elem = grad_vec[element_index];
        auto &state_sum_elem = state_sum_vec[element_index];

        // Store state_sum, param and grad values before update
        if (output_state_sum_values_ != nullptr) {
          output_state_sum_values_[i * embedding_dim_ + element_index] =
              state_sum_elem;
        }
        if (output_params_before_ != nullptr) {
          output_params_before_[i * embedding_dim_ + element_index] = emb_elem;
        }
        grad_elem = grad_elem + weight_decay_ * emb_elem;
        if (output_grad_ != nullptr) {
          output_grad_[i * embedding_dim_ + element_index] = grad_elem;
        }

        state_sum_elem += grad_elem * grad_elem;
        emb_elem -= lr_ * (grad_elem / (sqrtf(state_sum_elem) + eps_));

        // Store param values after update
        if (output_params_after_ != nullptr) {
          output_params_after_[i * embedding_dim_ + element_index] = emb_elem;
        }
        // w_t = w_t-1 - grad * lr / [sqrtf(accum_of_grad_pow) + eps_]
      }
    }
  }

 private:
  const int64_t embedding_dim_;
  const int64_t block_size_;
  const int64_t *index_vec_;
  std::vector<torch::Tensor> &emb_blocks_;
  TEmb *grad_;
  std::vector<torch::Tensor> &state_sum_;
  const double lr_;
  const double eps_;
  const double weight_decay_;
  TEmb *output_state_sum_values_;
  TEmb *output_params_before_;
  TEmb *output_params_after_;
  TEmb *output_grad_;
};

void block_apply_adagrad_cpu(
    const torch::Tensor index, const torch::Tensor grad,
    std::vector<torch::Tensor> emb_blocks, std::vector<torch::Tensor> state_sum,
    double lr, double eps, double weight_decay, int64_t block_size,
    torch::Tensor output_state_sum, torch::Tensor output_params_before,
    torch::Tensor output_params_after, torch::Tensor output_grad) {
  int64_t embedding_dim = emb_blocks[0].size(1);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "apply_adagrad_cpu_impl", ([&] {
        scalar_t *output_state_sum_ptr = nullptr;
        if (output_state_sum.defined()) {
          output_state_sum_ptr = output_state_sum.data_ptr<scalar_t>();
        }

        scalar_t *output_params_before_ptr = nullptr;
        if (output_params_before.defined()) {
          output_params_before_ptr = output_params_before.data_ptr<scalar_t>();
        }

        scalar_t *output_params_after_ptr = nullptr;
        if (output_params_after.defined()) {
          output_params_after_ptr = output_params_after.data_ptr<scalar_t>();
        }

        scalar_t *output_grad_ptr = nullptr;
        if (output_grad.defined()) {
          output_grad_ptr = output_grad.data_ptr<scalar_t>();
        }

        BlocksApplyAdagradFunctor<scalar_t> apply_functor(
            embedding_dim, block_size, index.data_ptr<int64_t>(), emb_blocks,
            grad.data_ptr<scalar_t>(), state_sum, lr, eps, weight_decay,
            output_state_sum_ptr, output_params_before_ptr,
            output_params_after_ptr, output_grad_ptr);
        at::parallel_for(0, index.numel(), 0, apply_functor);
      }));
}

void block_apply_adagrad(const torch::Tensor index, const torch::Tensor grad,
                         std::vector<torch::Tensor> emb_blocks,
                         torch::Tensor step,
                         std::vector<torch::Tensor> state_sum, double lr,
                         double lr_decay, double eps, double weight_decay,
                         int64_t block_size, torch::Tensor output_state_sum,
                         torch::Tensor output_params_before,
                         torch::Tensor output_params_after,
                         torch::Tensor output_grad) {
  step.add_(1);  // step >= 0
  int64_t step_item = step.item<int64_t>();
  TORCH_CHECK(step_item >= 1);
  lr = lr / (1 + (step_item - 1) * lr_decay);
  if (index.device().type() == torch::kCUDA) {
    block_apply_adagrad_gpu(index, grad, emb_blocks, state_sum, lr, eps,
                            weight_decay, block_size, output_state_sum,
                            output_params_before, output_params_after,
                            output_grad);
  } else {
    block_apply_adagrad_cpu(index, grad, emb_blocks, state_sum, lr, eps,
                            weight_decay, block_size, output_state_sum,
                            output_params_before, output_params_after,
                            output_grad);
  }
}

}  // namespace functional
}  // namespace recis

namespace recis {
namespace functional {

// Functor for adagrad with pre-computed gradient sum of squares
template <class TEmb>
struct BlocksApplyAdagradSumFunctor {
  BlocksApplyAdagradSumFunctor(int64_t embedding_dim, int64_t block_size,
                               const int64_t *index_vec,
                               std::vector<torch::Tensor> &emb_blocks,
                               TEmb *grad, TEmb *grad_sum_sq,
                               std::vector<torch::Tensor> &state_sum, double lr,
                               double eps, TEmb *output_state_sum_values,
                               TEmb *output_params_before,
                               TEmb *output_params_after, TEmb *output_grad)
      : embedding_dim_(embedding_dim),
        block_size_(block_size),
        index_vec_(index_vec),
        emb_blocks_(emb_blocks),
        grad_(grad),
        grad_sum_sq_(grad_sum_sq),
        state_sum_(state_sum),
        lr_(lr),
        eps_(eps),
        output_state_sum_values_(output_state_sum_values),
        output_params_before_(output_params_before),
        output_params_after_(output_params_after),
        output_grad_(output_grad) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto i : c10::irange(beg, end)) {
      auto index = index_vec_[i];  // embedding index
      if (index < 0) {
        TORCH_CHECK(
            index == -1,
            "index of BlocksApplyAdagradSumFunctor must be >= -1, but get ",
            index);
        continue;
      }
      auto block_index = index / block_size_;
      auto row_index = index % block_size_;
      auto offset = row_index * embedding_dim_;
      auto emb_vec = emb_blocks_[block_index].data_ptr<TEmb>() + offset;
      auto state_sum_vec = state_sum_[block_index].data_ptr<TEmb>() + offset;
      auto grad_vec = grad_ + i * embedding_dim_;
      auto grad_sum_sq_vec = grad_sum_sq_ + i * embedding_dim_;
      for (auto element_index : c10::irange(embedding_dim_)) {
        auto &emb_elem = emb_vec[element_index];
        auto grad_elem = grad_vec[element_index];
        auto grad_sum_sq_elem = grad_sum_sq_vec[element_index];
        auto &state_sum_elem = state_sum_vec[element_index];

        // Store state_sum, param and grad values before update
        if (output_state_sum_values_ != nullptr) {
          output_state_sum_values_[i * embedding_dim_ + element_index] =
              state_sum_elem;
        }
        if (output_params_before_ != nullptr) {
          output_params_before_[i * embedding_dim_ + element_index] = emb_elem;
        }
        if (output_grad_ != nullptr) {
          output_grad_[i * embedding_dim_ + element_index] = grad_elem;
        }

        // Use pre-computed gradient sum of squares instead of computing grad *
        // grad
        state_sum_elem += grad_sum_sq_elem;
        emb_elem -= lr_ * (grad_elem / (sqrtf(state_sum_elem) + eps_));

        // Store param values after update
        if (output_params_after_ != nullptr) {
          output_params_after_[i * embedding_dim_ + element_index] = emb_elem;
        }
      }
    }
  }

 private:
  const int64_t embedding_dim_;
  const int64_t block_size_;
  const int64_t *index_vec_;
  std::vector<torch::Tensor> &emb_blocks_;
  TEmb *grad_;
  TEmb *grad_sum_sq_;
  std::vector<torch::Tensor> &state_sum_;
  const double lr_;
  const double eps_;
  TEmb *output_state_sum_values_;
  TEmb *output_params_before_;
  TEmb *output_params_after_;
  TEmb *output_grad_;
};

void block_apply_adagrad_sum_cpu(
    const torch::Tensor index, const torch::Tensor grad,
    const torch::Tensor grad_sum_sq, std::vector<torch::Tensor> emb_blocks,
    std::vector<torch::Tensor> state_sum, double lr, double eps,
    int64_t block_size, torch::Tensor output_state_sum,
    torch::Tensor output_params_before, torch::Tensor output_params_after,
    torch::Tensor output_grad) {
  int64_t embedding_dim = emb_blocks[0].size(1);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "apply_adagrad_sum_cpu_impl", ([&] {
        scalar_t *output_state_sum_ptr = nullptr;
        if (output_state_sum.defined()) {
          output_state_sum_ptr = output_state_sum.data_ptr<scalar_t>();
        }

        scalar_t *output_params_before_ptr = nullptr;
        if (output_params_before.defined()) {
          output_params_before_ptr = output_params_before.data_ptr<scalar_t>();
        }

        scalar_t *output_params_after_ptr = nullptr;
        if (output_params_after.defined()) {
          output_params_after_ptr = output_params_after.data_ptr<scalar_t>();
        }

        scalar_t *output_grad_ptr = nullptr;
        if (output_grad.defined()) {
          output_grad_ptr = output_grad.data_ptr<scalar_t>();
        }

        BlocksApplyAdagradSumFunctor<scalar_t> apply_functor(
            embedding_dim, block_size, index.data_ptr<int64_t>(), emb_blocks,
            grad.data_ptr<scalar_t>(), grad_sum_sq.data_ptr<scalar_t>(),
            state_sum, lr, eps, output_state_sum_ptr, output_params_before_ptr,
            output_params_after_ptr, output_grad_ptr);
        at::parallel_for(0, index.numel(), 0, apply_functor);
      }));
}

void block_apply_adagrad_sum(
    const torch::Tensor index, const torch::Tensor grad,
    const torch::Tensor grad_sum_sq, std::vector<torch::Tensor> emb_blocks,
    torch::Tensor step, std::vector<torch::Tensor> state_sum, double lr,
    double lr_decay, double eps, int64_t block_size,
    torch::Tensor output_state_sum, torch::Tensor output_params_before,
    torch::Tensor output_params_after, torch::Tensor output_grad) {
  step.add_(1);  // step >= 0
  int64_t step_item = step.item<int64_t>();
  TORCH_CHECK(step_item >= 1);
  lr = lr / (1 + (step_item - 1) * lr_decay);
  if (index.device().type() == torch::kCUDA) {
    block_apply_adagrad_sum_gpu(index, grad, grad_sum_sq, emb_blocks, state_sum,
                                lr, eps, block_size, output_state_sum,
                                output_params_before, output_params_after,
                                output_grad);
  } else {
    block_apply_adagrad_sum_cpu(index, grad, grad_sum_sq, emb_blocks, state_sum,
                                lr, eps, block_size, output_state_sum,
                                output_params_before, output_params_after,
                                output_grad);
  }
}

}  // namespace functional
}  // namespace recis
