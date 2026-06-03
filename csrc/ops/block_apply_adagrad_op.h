#include <torch/extension.h>

namespace recis {
namespace functional {

void block_apply_adagrad(const torch::Tensor index, const torch::Tensor grad,
                         std::vector<torch::Tensor> emb_blocks,
                         torch::Tensor step,
                         std::vector<torch::Tensor> state_sum, double lr,
                         double lr_decay, double eps, double weight_decay,
                         int64_t block_size);

void block_apply_adagrad_gpu(const torch::Tensor index,
                             const torch::Tensor grad,
                             std::vector<torch::Tensor> emb_blocks,
                             std::vector<torch::Tensor> state_sum, double lr,
                             double eps, double weight_decay,
                             int64_t block_size);

}  // namespace functional
}  // namespace recis
