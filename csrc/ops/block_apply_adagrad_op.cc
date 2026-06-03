#include "block_apply_adagrad_op.h"

namespace recis {
namespace functional {

template <class TEmb>
struct BlocksApplyAdagradFunctor {
  BlocksApplyAdagradFunctor(int64_t embedding_dim, int64_t block_size,
                            const int64_t *index_vec,
                            std::vector<torch::Tensor> &emb_blocks, TEmb *grad,
                            std::vector<torch::Tensor> &state_sum, double lr,
                            double eps, double weight_decay)
      : embedding_dim_(embedding_dim),
        block_size_(block_size),
        index_vec_(index_vec),
        emb_blocks_(emb_blocks),
        grad_(grad),
        state_sum_(state_sum),
        lr_(lr),
        eps_(eps),
        weight_decay_(weight_decay) {}
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
        grad_elem = grad_elem + weight_decay_ * emb_elem;
        state_sum_elem += grad_elem * grad_elem;
        emb_elem -= lr_ * (grad_elem / (sqrtf(state_sum_elem) + eps_));
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
};

void block_apply_adagrad_cpu(const torch::Tensor index,
                             const torch::Tensor grad,
                             std::vector<torch::Tensor> emb_blocks,
                             std::vector<torch::Tensor> state_sum, double lr,
                             double eps, double weight_decay,
                             int64_t block_size) {
  int64_t embedding_dim = emb_blocks[0].size(1);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "apply_adagrad_cpu_impl", ([&] {
        BlocksApplyAdagradFunctor<scalar_t> apply_functor(
            embedding_dim, block_size, index.data_ptr<int64_t>(), emb_blocks,
            grad.data_ptr<scalar_t>(), state_sum, lr, eps, weight_decay);
        at::parallel_for(0, index.numel(), 0, apply_functor);
      }));
}

void block_apply_adagrad(const torch::Tensor index, const torch::Tensor grad,
                         std::vector<torch::Tensor> emb_blocks,
                         torch::Tensor step,
                         std::vector<torch::Tensor> state_sum, double lr,
                         double lr_decay, double eps, double weight_decay,
                         int64_t block_size) {
  step.add_(1);  // step >= 0
  int64_t step_item = step.item<int64_t>();
  TORCH_CHECK(step_item >= 1);
  lr = lr / (1 + (step_item - 1) * lr_decay);
  if (index.device().type() == torch::kCUDA) {
    block_apply_adagrad_gpu(index, grad, emb_blocks, state_sum, lr, eps,
                            weight_decay, block_size);
  } else {
    block_apply_adagrad_cpu(index, grad, emb_blocks, state_sum, lr, eps,
                            weight_decay, block_size);
  }
}

}  // namespace functional
}  // namespace recis
