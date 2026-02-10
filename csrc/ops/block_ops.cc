#include "block_ops.h"
#include "compare_ops.h"

namespace recis {
namespace functional {

template <typename T, CompareOp op>
struct CompareFactory;

template <typename T>
struct CompareFactory<T, CompareOp::LT> {
  bool operator()(const T a, const T b) const { return a < b; }
};

template <typename T>
struct CompareFactory<T, CompareOp::LE> {
  bool operator()(const T a, const T b) const { return a <= b; }
};

template <typename T>
struct CompareFactory<T, CompareOp::GT> {
  bool operator()(const T a, const T b) const { return a > b; }
};

template <typename T>
struct CompareFactory<T, CompareOp::GE> {
  bool operator()(const T a, const T b) const { return a >= b; }
};

template <typename T>
struct CompareFactory<T, CompareOp::EQ> {
  bool operator()(const T a, const T b) const { return a == b; }
};

template <class TEmb>
struct GatherFunctor {
  GatherFunctor(int64_t embedding_dim, const int64_t *index_vec, TEmb *emb_vec,
                TEmb *out_vec)
      : embedding_dim_(embedding_dim),
        index_vec_(index_vec),
        emb_vec_(emb_vec),
        out_vec_(out_vec) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto dst_index : c10::irange(beg, end)) {
      auto src_index = index_vec_[dst_index];
      memcpy(out_vec_ + dst_index * embedding_dim_,
             emb_vec_ + src_index * embedding_dim_,
             sizeof(TEmb) * embedding_dim_);
    }
  }

 private:
  const int64_t embedding_dim_;
  const int64_t *index_vec_;
  TEmb *emb_vec_;
  TEmb *out_vec_;
};

template <class TEmb>
struct BlocksGatherFunctor {
  BlocksGatherFunctor(int64_t embedding_dim, int64_t block_size,
                      const int64_t *index_vec,
                      std::vector<torch::Tensor> &emb_blocks, TEmb *out_vec,
                      int64_t beg)
      : embedding_dim_(embedding_dim),
        block_size_(block_size),
        index_vec_(index_vec),
        emb_blocks_(emb_blocks),
        out_vec_(out_vec),
        beg_(beg) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto dst_index : c10::irange(beg, end)) {
      auto src_index = index_vec_[dst_index];
      auto src_block_index = src_index / block_size_;
      auto src_row_index = src_index % block_size_;
      memcpy(out_vec_ + (dst_index - beg_) * embedding_dim_,
             emb_blocks_[src_block_index].data_ptr<TEmb>() +
                 src_row_index * embedding_dim_,
             sizeof(TEmb) * embedding_dim_);
    }
  }

 private:
  const int64_t embedding_dim_;
  const int64_t block_size_;
  const int64_t *index_vec_;
  std::vector<torch::Tensor> &emb_blocks_;
  TEmb *out_vec_;
  const int64_t beg_;
};

template <class TEmb>
struct ReadOnlyBlocksGatherFunctor {
  ReadOnlyBlocksGatherFunctor(int64_t embedding_dim, int64_t block_size,
                              const int64_t *index_vec,
                              std::vector<torch::Tensor> &emb_blocks,
                              TEmb *out_vec, int64_t default_key, int64_t beg)
      : embedding_dim_(embedding_dim),
        block_size_(block_size),
        default_key_(default_key),
        index_vec_(index_vec),
        emb_blocks_(emb_blocks),
        out_vec_(out_vec),
        beg_(beg) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto dst_index : c10::irange(beg, end)) {
      auto src_index = index_vec_[dst_index];
      if (src_index == default_key_) {
        memset(out_vec_ + dst_index * embedding_dim_, 0,
               embedding_dim_ * sizeof(TEmb));
        continue;
      }
      auto src_block_index = src_index / block_size_;
      auto src_row_index = src_index % block_size_;
      memcpy(out_vec_ + (dst_index - beg_) * embedding_dim_,
             emb_blocks_[src_block_index].data_ptr<TEmb>() +
                 src_row_index * embedding_dim_,
             sizeof(TEmb) * embedding_dim_);
    }
  }

 private:
  const int64_t embedding_dim_;
  const int64_t block_size_;
  const int64_t default_key_;
  const int64_t *index_vec_;
  std::vector<torch::Tensor> &emb_blocks_;
  TEmb *out_vec_;
  const int64_t beg_;
};

template <class TEmb, bool IsBroadcast>
struct BlocksInsertFunctor {
  BlocksInsertFunctor(int64_t embedding_dim, int64_t block_size,
                      const int64_t *index_vec, TEmb *emb_vec,
                      std::vector<torch::Tensor> &emb_blocks)
      : embedding_dim_(embedding_dim),
        block_size_(block_size),
        index_vec_(index_vec),
        emb_vec_(emb_vec),
        emb_blocks_(emb_blocks) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto src_index : c10::irange(beg, end)) {
      const TEmb *emb =
          IsBroadcast ? emb_vec_ : emb_vec_ + src_index * embedding_dim_;
      auto dst_index = index_vec_[src_index];
      if (dst_index < 0) {
        TORCH_CHECK(dst_index == -1,
                    "index of BlocksInsertFunctor must be >= -1, but get ",
                    dst_index);
        continue;
      }
      auto dst_block_index = dst_index / block_size_;
      auto dst_row_index = dst_index % block_size_;

      memcpy(emb_blocks_[dst_block_index].data_ptr<TEmb>() +
                 dst_row_index * embedding_dim_,
             emb, sizeof(TEmb) * embedding_dim_);
    }
  }

 private:
  const int64_t embedding_dim_;
  const int64_t block_size_;
  const int64_t *index_vec_;
  std::vector<torch::Tensor> &emb_blocks_;
  TEmb *emb_vec_;
};

template <class TEmb>
struct BlocksInsertWithMaskFunctor {
  BlocksInsertWithMaskFunctor(int64_t embedding_dim, int64_t block_size,
                              const int64_t *index_vec, TEmb *emb_vec,
                              const bool *mask_vec,
                              std::vector<torch::Tensor> &emb_blocks)
      : embedding_dim_(embedding_dim),
        block_size_(block_size),
        index_vec_(index_vec),
        emb_vec_(emb_vec),
        mask_vec_(mask_vec),
        emb_blocks_(emb_blocks) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto src_index : c10::irange(beg, end)) {
      if (!mask_vec_[src_index]) continue;
      auto dst_index = index_vec_[src_index];
      auto dst_block_index = dst_index / block_size_;
      auto dst_row_index = dst_index % block_size_;
      memcpy(emb_blocks_[dst_block_index].data_ptr<TEmb>() +
                 dst_row_index * embedding_dim_,
             emb_vec_ + src_index * embedding_dim_,
             sizeof(TEmb) * embedding_dim_);
    }
  }

 private:
  const int64_t embedding_dim_;
  const int64_t block_size_;
  const int64_t *index_vec_;
  TEmb *emb_vec_;
  const bool *mask_vec_;
  std::vector<torch::Tensor> &emb_blocks_;
};

torch::Tensor gather_cpu_kernel(const torch::Tensor ids,
                                const torch::Tensor emb) {
  TORCH_CHECK(ids.device().type() == torch::kCPU,
              "CPU version requires CPU input");
  TORCH_CHECK(emb.device().type() == torch::kCPU,
              "CPU version requires CPU emb");
  int64_t num_ids = ids.numel();
  int64_t embedding_dim = emb.size(1);
  auto output = torch::empty({num_ids, embedding_dim}, emb.options());
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, output.scalar_type(), "gather_cpu_impl", ([&] {
        GatherFunctor<scalar_t> gather_functor(
            embedding_dim, ids.data_ptr<int64_t>(), emb.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
        at::parallel_for(0, num_ids, 0, gather_functor);
      }));
  return output;
}

torch::Tensor block_gather_cpu_kernel(const torch::Tensor ids,
                                      std::vector<torch::Tensor> &emb_blocks,
                                      int64_t block_size, int64_t default_key,
                                      bool readonly, int64_t beg, int64_t end) {
  TORCH_CHECK(ids.device().type() == torch::kCPU,
              "CPU version requires CPU input");
  TORCH_CHECK(emb_blocks[0].device().type() == torch::kCPU,
              "CPU version requires CPU emb_blocks");
  int64_t num_ids = end - beg;
  int64_t embedding_dim = emb_blocks[0].size(1);
  auto output = torch::empty({num_ids, embedding_dim}, emb_blocks[0].options());
  if (num_ids == 0) return output;
  if (readonly) {
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, output.scalar_type(),
        "block_gather_readonly_cpu_impl", ([&] {
          ReadOnlyBlocksGatherFunctor<scalar_t> gather_functor(
              embedding_dim, block_size, ids.data_ptr<int64_t>(), emb_blocks,
              output.data_ptr<scalar_t>(), default_key, beg);
          at::parallel_for(beg, end, 0, gather_functor);
        }));
  } else {
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, output.scalar_type(),
                              "block_gather_cpu_impl", ([&] {
                                BlocksGatherFunctor<scalar_t> gather_functor(
                                    embedding_dim, block_size,
                                    ids.data_ptr<int64_t>(), emb_blocks,
                                    output.data_ptr<scalar_t>(), beg);
                                at::parallel_for(beg, end, 0, gather_functor);
                              }));
  }
  return output;
}

template <class TEmb, typename Factory>
struct BlocksFilterFunctor {
  BlocksFilterFunctor(TEmb threshold, int64_t block_size,
                      const int64_t *index_vec,
                      std::vector<torch::Tensor> &emb_blocks, bool *mask_out,
                      Factory factory)
      : threshold_(threshold),
        block_size_(block_size),
        index_vec_(index_vec),
        emb_blocks_(emb_blocks),
        mask_out_(mask_out),
        factory_(factory) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto src_index : c10::irange(beg, end)) {
      auto dst_index = index_vec_[src_index];
      auto dst_block_index = dst_index / block_size_;
      auto dst_row_index = dst_index % block_size_;
      TEmb filter_val =
          emb_blocks_[dst_block_index].data_ptr<TEmb>()[dst_row_index];
      bool to_filter = factory_(filter_val, threshold_);
      mask_out_[src_index] = to_filter;
    }
  }

 private:
  const TEmb threshold_;
  const int64_t block_size_;
  const int64_t *index_vec_;
  std::vector<torch::Tensor> &emb_blocks_;
  bool *mask_out_;
  Factory factory_;
};

torch::Tensor block_filter_cpu_kernel(const torch::Tensor ids,
                                      std::vector<torch::Tensor> &emb_blocks,
                                      int64_t threshold, int64_t block_size,
                                      int64_t compare_op) {
  TORCH_CHECK(ids.device().is_cpu(), "CPU version requires CPU input");
  TORCH_CHECK(emb_blocks.size() == 0 || emb_blocks[0].device().is_cpu(),
              "CPU version requires CPU emb_blocks");
  int64_t num_ids = ids.numel();
  auto output = torch::empty(
      {num_ids},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
  if (num_ids == 0) return output;

  AT_DISPATCH_INDEX_TYPES(
      emb_blocks[0].scalar_type(), "block_filter_cpu_impl", ([&] {
        BF_DISPATCH_COMPARE_TYPES(
            compare_op, ([&] {
              using Factory = CompareFactory<index_t, compare_t>;
              BlocksFilterFunctor<index_t, Factory> filter_functor(
                  static_cast<index_t>(threshold), block_size,
                  ids.data_ptr<int64_t>(), emb_blocks, output.data_ptr<bool>(),
                  Factory());
              at::parallel_for(0, ids.numel(), 0, filter_functor);
            }));
      }));
  return output;
}

torch::Tensor gather(const torch::Tensor ids, const torch::Tensor emb) {
  if (ids.device().type() == torch::kCUDA) {
    return gather_cuda(ids, emb);
  } else {
    return gather_cpu_kernel(ids, emb);
  }
}

torch::Tensor block_gather(const torch::Tensor ids,
                           std::vector<torch::Tensor> emb_blocks,
                           int64_t block_size, int64_t default_key,
                           bool readonly) {
  if (ids.device().type() == torch::kCUDA) {
    return block_gather_cuda(ids, emb_blocks, block_size, default_key, readonly,
                             0, ids.numel());
  } else {
    return block_gather_cpu_kernel(ids, emb_blocks, block_size, default_key,
                                   readonly, 0, ids.numel());
  }
}

torch::Tensor block_filter(const torch::Tensor ids,
                           std::vector<torch::Tensor> emb_blocks,
                           int64_t block_size, int64_t threshold,
                           int64_t compare_op) {
  if (ids.device().type() == torch::kCUDA) {
    return block_filter_cuda(ids, emb_blocks, threshold, block_size, compare_op);
  } else {
    return block_filter_cpu_kernel(ids, emb_blocks, threshold, block_size, compare_op);
  }
}


torch::Tensor block_gather_by_range(const torch::Tensor ids,
                                    std::vector<torch::Tensor> emb_blocks,
                                    int64_t block_size, int64_t beg,
                                    int64_t end) {
  if (ids.device().type() == torch::kCUDA) {
    return block_gather_cuda(ids, emb_blocks, block_size, -1, false, beg, end);
  } else {
    return block_gather_cpu_kernel(ids, emb_blocks, block_size, -1, false, beg,
                                   end);
  }
}

void block_insert_cpu_kernel(const torch::Tensor ids,
                             const torch::Tensor embedding,
                             std::vector<torch::Tensor> embedding_blocks,
                             int64_t block_size) {
  TORCH_CHECK(ids.device().type() == torch::kCPU,
              "Input must be on CPU device");
  TORCH_CHECK(embedding.device().type() == torch::kCPU,
              "Embedding must be on CPU device");
  TORCH_CHECK(
      ids.numel() == embedding.size(0) || embedding.size(0) == 1,
      "ids batch_size must = embedding batch size or embedding batch size = 1");

  auto num_ids = ids.numel();
  if (num_ids == 0) return;
  int64_t embedding_dim = embedding_blocks[0].size(1);
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, embedding.scalar_type(), "block_insert_cpu_impl",
      ([&] {
        auto ids_ptr = ids.data_ptr<int64_t>();
        auto emb_ptr = embedding.data_ptr<scalar_t>();
        auto insert_launcher = [&](auto is_broadcast) {
          using Functor =
              BlocksInsertFunctor<scalar_t, decltype(is_broadcast)::value>;
          Functor insert_functor(embedding_dim, block_size, ids_ptr, emb_ptr,
                                 embedding_blocks);
          at::parallel_for(0, ids.numel(), 0, insert_functor);
        };
        if (embedding.size(0) == 1) {
          insert_launcher(std::true_type{});
        } else {
          insert_launcher(std::false_type{});
        }
      }));
}

void block_insert(const torch::Tensor ids, const torch::Tensor embedding,
                  std::vector<torch::Tensor> emb_blocks, int64_t block_size) {
  if (ids.device().type() == torch::kCUDA) {
    block_insert_cuda(ids, embedding, emb_blocks, block_size);
  } else {
    block_insert_cpu_kernel(ids, embedding, emb_blocks, block_size);
  }
}

void block_insert_with_mask_cpu_kernel(
    const torch::Tensor ids, const torch::Tensor embedding,
    const torch::Tensor mask, std::vector<torch::Tensor> embedding_blocks,
    int64_t block_size) {
  TORCH_CHECK(ids.device().type() == torch::kCPU,
              "Input must be on CPU device");
  TORCH_CHECK(embedding.device().type() == torch::kCPU,
              "Embedding must be on CPU device");
  auto num_ids = ids.numel();
  if (num_ids == 0) return;
  int64_t embedding_dim = embedding_blocks[0].size(1);
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, embedding.scalar_type(),
      "block_insert_with_mask_cpu_impl", ([&] {
        BlocksInsertWithMaskFunctor<scalar_t> insert_functor(
            embedding_dim, block_size, ids.data_ptr<int64_t>(),
            embedding.data_ptr<scalar_t>(), mask.data_ptr<bool>(),
            embedding_blocks);
        at::parallel_for(0, ids.numel(), 0, insert_functor);
      }));
}

void block_insert_with_mask(const torch::Tensor ids,
                            const torch::Tensor embedding,
                            const torch::Tensor mask,
                            std::vector<torch::Tensor> emb_blocks,
                            int64_t block_size) {
  if (ids.device().type() == torch::kCUDA) {
    block_insert_with_mask_cuda(ids, embedding, mask, emb_blocks, block_size);
  } else {
    block_insert_with_mask_cpu_kernel(ids, embedding, mask, emb_blocks,
                                      block_size);
  }
}

}  // namespace functional
}  // namespace recis
