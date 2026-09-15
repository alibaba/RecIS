#pragma once

#include <torch/extension.h>

namespace recis {
namespace functional {

void block_apply_row_wise_adagrad(const torch::Tensor index,
                                  const torch::Tensor grad,
                                  std::vector<torch::Tensor> emb_blocks,
                                  torch::Tensor step,
                                  std::vector<torch::Tensor> state_sum,
                                  double lr, double lr_decay, double eps,
                                  double weight_decay, int64_t block_size);

void block_apply_row_wise_adagrad_gpu(const torch::Tensor index,
                                      const torch::Tensor grad,
                                      std::vector<torch::Tensor> emb_blocks,
                                      std::vector<torch::Tensor> state_sum,
                                      double lr, double eps,
                                      double weight_decay, int64_t block_size);

void block_apply_row_wise_adagrad_sum(
    const torch::Tensor index, const torch::Tensor grad,
    const torch::Tensor grad_sum_sq, std::vector<torch::Tensor> emb_blocks,
    torch::Tensor step, std::vector<torch::Tensor> state_sum, double lr,
    double lr_decay, double eps, int64_t block_size);

void block_apply_row_wise_adagrad_sum_gpu(const torch::Tensor index,
                                          const torch::Tensor grad,
                                          const torch::Tensor grad_sum_sq,
                                          std::vector<torch::Tensor> emb_blocks,
                                          std::vector<torch::Tensor> state_sum,
                                          double lr, double eps,
                                          int64_t block_size);

}  // namespace functional
}  // namespace recis
