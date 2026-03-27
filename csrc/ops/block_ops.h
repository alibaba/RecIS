#include <torch/extension.h>

namespace recis {
namespace functional {

void block_insert(const torch::Tensor ids, const torch::Tensor embedding,
                  std::vector<torch::Tensor>& emb_blocks, int64_t block_size);

void block_insert_bind(const torch::Tensor ids, const torch::Tensor embedding,
                       std::vector<torch::Tensor> emb_blocks,
                       int64_t block_size);

void block_insert_cuda(const torch::Tensor ids, const torch::Tensor embedding,
                       std::vector<torch::Tensor>& emb_blocks,
                       int64_t block_size);

void block_insert_with_mask(const torch::Tensor ids,
                            const torch::Tensor embedding,
                            const torch::Tensor mask,
                            std::vector<torch::Tensor>& emb_blocks,
                            int64_t block_size);

void block_insert_with_mask_bind(const torch::Tensor ids,
                                 const torch::Tensor embedding,
                                 const torch::Tensor mask,
                                 std::vector<torch::Tensor> emb_blocks,
                                 int64_t block_size);

void block_insert_with_mask_cuda(const torch::Tensor ids,
                                 const torch::Tensor embedding,
                                 const torch::Tensor mask,
                                 std::vector<torch::Tensor>& emb_blocks,
                                 int64_t block_size);

torch::Tensor block_gather(const torch::Tensor ids,
                           std::vector<torch::Tensor>& emb_blocks,
                           int64_t block_size, int64_t default_key,
                           bool readonly);

torch::Tensor block_gather_bind(const torch::Tensor ids,
                                std::vector<torch::Tensor> emb_blocks,
                                int64_t block_size, int64_t default_key,
                                bool readonly);

torch::Tensor gather(const torch::Tensor ids, const torch::Tensor emb);

torch::Tensor block_gather_by_range(const torch::Tensor ids,
                                    std::vector<torch::Tensor>& emb_blocks,
                                    int64_t block_size, int64_t beg,
                                    int64_t end);

torch::Tensor block_gather_cuda(const torch::Tensor ids,
                                std::vector<torch::Tensor>& emb_blocks,
                                int64_t block_size, int64_t default_key,
                                bool readonly, int64_t beg, int64_t end);

torch::Tensor gather_cuda(const torch::Tensor ids, const torch::Tensor emb);

torch::Tensor block_filter_cuda(const torch::Tensor ids,
                                std::vector<torch::Tensor>& emb_blocks,
                                int64_t block_size, int64_t step);

torch::Tensor block_filter(const torch::Tensor ids,
                           std::vector<torch::Tensor>& emb_blocks,
                           // const torch::Tensor mask,
                           int64_t block_size, int64_t step);

torch::Tensor block_filter_bind(const torch::Tensor ids,
                                std::vector<torch::Tensor> emb_blocks,
                                int64_t block_size, int64_t step);

}  // namespace functional
}  // namespace recis
