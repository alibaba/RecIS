No-Reduce Embedding（不做 reduce/split 的原始 a2a 输出）
=============================================================

背景
----

``EmbeddingEngine`` 默认的 forward 流程是：

1. **a2a #1**：把 ids 按 owner 分发出去；
2. **a2a #2**：每张 HashTable 查表后把 embedding 发回；
3. **reduce**：按 segment（offsets）做 sum / mean / tile 聚合；
4. **split**：把同一个 runtime group 内 coalesced 在一起的多个 feature 拆回每个 feature 自己的输出。

但有些场景（attention、sequence modeling、MoE 路由等）**不希望框架替你做 reduce**，
而是要拿到 a2a 之后还没聚合的原始 embedding，自己做后续处理。

``EmbeddingOption.no_reduce=True`` 提供了这条 fast-path：
框架在第 2 步 a2a 完成之后，直接把结果以 :class:`recis.nn.NoReduceEmbedding`
的形式交还给用户，**跳过 reduce 和 split**。

约束
----

* ``no_reduce=True`` 的 feature 在 ``EmbeddingEngine`` 里 **不会和任何其他
  feature 做样本级 coalesce**（即不会和别的 feature 共用同一条 a2a tensor）。
  框架在 ``HashTableCoalescedGroup.add_option`` 中给它生成唯一的 runtime key
  来强制这个隔离。
* 它仍然可以和别的 feature **共享同一张 HashTable**（``shared_name`` 相同）：
  即使一个 ``no_reduce=True`` 的 feature 和一个普通 reduce 的 feature 共用
  ``shared_name``，二者也只是共表（参数共享），样本级 a2a 仍然独立。框架不会
  对此报错——这是允许的设计；如果你不希望共表，给它独立的 ``shared_name`` 即可。
* 设了 ``no_reduce=True`` 时，``combiner`` / ``combiner_kwargs`` 会被忽略。
* 反向传播自动可用：``EmbeddingExchange`` 的 backward 仍然会做 a2a 把梯度送回
  owner 端，所以你在拿到 ``NoReduceEmbedding.emb`` 之后做的任何可微运算都能
  正常回传梯度。**注意**：no_reduce 路径反向时，梯度按 unique-id 维度搬运，
  字节量与正常 reduce 路径不一样（reduce 路径反向是 ``[batch_size, dim]`` 量级，
  no_reduce 路径是 ``[N_unique_after_a2a, dim]`` 量级）。序列长、unique id 多
  的场景，建议留意带宽开销。

API
---

详细文档：:class:`recis.nn.NoReduceEmbedding` 、 :class:`recis.nn.EmbeddingOption` 。

返回的 dataclass 字段含义：

* ``emb`` (``torch.Tensor``)：a2a 完成后的 unique embedding，形状 ``[N_unique, dim]``。
* ``reverse_index`` (``torch.Tensor``)：把原始 ids 映射到 ``emb`` 的索引；
  ``emb[reverse_index]`` 即每个原始 id 对应的 embedding。
* ``offsets`` (``torch.Tensor``)：原始输入的 segment 边界，形状 ``[batch_size+1]``。

原始 dense shape 不在 dataclass 里返回；需要 batch 维时用
``offsets.numel() - 1`` 推，或直接从你自己手里的输入张量取。

用法示例
--------

.. code-block:: python

    import torch
    from recis.nn import EmbeddingEngine, EmbeddingOption, NoReduceEmbedding
    from recis.ragged.tensor import RaggedTensor

    emb_options = {
        # 普通 feature：走默认 sum reduce
        "user_id": EmbeddingOption(
            embedding_dim=64,
            shared_name="user_embedding",
            combiner="sum",
            device=torch.device("cuda"),
        ),
        # 序列特征：拿原始 a2a 输出，自己做 attention
        "click_seq": EmbeddingOption(
            embedding_dim=64,
            shared_name="item_embedding",
            no_reduce=True,
            device=torch.device("cuda"),
        ),
    }

    engine = EmbeddingEngine(emb_options)

    # 32 个 sample，每条序列长 6 -> 总 ids = 192
    bs, seq_len = 32, 6
    features = {
        "user_id": torch.randint(0, 10000, (bs, 1), device="cuda"),
        "click_seq": RaggedTensor(
            values=torch.randint(0, 50000, (bs * seq_len,), device="cuda"),
            offsets=torch.arange(0, bs + 1, dtype=torch.int32, device="cuda") * seq_len,
        ),
    }

    out = engine(features)

    # user_id：常规 reduce 后的 embedding tensor，shape [32, 64]
    user_emb: torch.Tensor = out["user_id"]

    # click_seq：未做 reduce 的 NoReduceEmbedding
    seq: NoReduceEmbedding = out["click_seq"]

    # 自己按需展开
    per_id_emb = seq.emb[seq.reverse_index]   # [N_total_ids, 64]
    # seq.offsets 给出 batch 内每个 sample 的边界，可以用 padded sequence + attention
    # 等任意自定义聚合处理 per_id_emb。
