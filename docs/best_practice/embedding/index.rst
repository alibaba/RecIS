Embedding嵌入使用
====================

.. toctree::
   :maxdepth: 2

   define_embedding
   engine
   no_reduce
   filter

简单示例
----------

.. code-block:: python

    def get_emb_conf():
        emb_options = {}
        emb_options["item_id"] = EmbeddingOption(
           embedding_dim=8,
           shared_name="item_id",
           combiner="mean",
           device=torch.device("cuda"),
        )
        emb_options["user_id"] = EmbeddingOption(
           embedding_dim=8,
           shared_name="user_id",
           combiner="mean",
           device=torch.device("cuda"),
        )
        emb_options["rate"] = EmbeddingOption(
           embedding_dim=8,
           shared_name="item_id",
           combiner="mean",
           device=torch.device("cuda"),
        )
    return emb_options