Quick Start
===========

This section will help you get started with RecIS quickly through a simple CTR model example.

Basic Concepts
--------------

Before starting, understand several core concepts of RecIS:

- **Feature**: Input features
- **FeatureEngine**: Feature processing engine
- **EmbeddingOption**: Embedding table configuration options
- **EmbeddingEngine**: Embedding processing engine
- **SparseOptimizer**: Sparse parameter optimizer
- **Trainer**: Training manager

First Model
-----------

Generate Sample ORC Data
~~~~~~~~~~~~~~~~~~~~~~~~

**1. Import ORC-related modules**

.. code-block:: python

    import os

    import numpy as np
    import pyarrow as pa
    import pyarrow.orc as orc

**2. Prepare data**

.. code-block:: python

    # Data output directory
    file_dir = "./fake_data/"
    # Number of samples per file
    bs = 2048
    # Total number of files
    file_num = 10

    # Features
    label = np.floor(np.random.rand(bs, 1) + 0.5, dtype=np.float32)
    user_id = np.arange(bs, dtype=np.int64).reshape(bs, 1)
    item_id = np.arange(bs, dtype=np.int64).reshape(bs, 1)

    data = {
        "label": label.tolist(),
        "user_id": user_id.tolist(),
        "item_id": item_id.tolist(),
    }

    table = pa.Table.from_pydict(data)
    # Write files
    for i in range(file_num):
        orc.write_table(table, os.path.join(file_dir, "data_{}.orc".format(i)))

Create Simple CTR Prediction Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Import necessary modules**

.. code-block:: python

    import os
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    
    from recis.io.orc_dataset import OrcDataset
    from recis.features.feature import Feature
    from recis.features.op import (
        SelectField,
        Mod,
    )
    from recis.features.feature_engine import FeatureEngine
    from recis.nn.modules.embedding import EmbeddingOption
    from recis.nn import EmbeddingEngine
    from recis.optim import SparseAdamW
    from recis.nn.modules.hashtable import filter_out_sparse_param
    from recis.framework.trainer import Trainer, TrainingArguments

**2. Define feature engineering**

.. code-block:: python

    # Map user_id feature to range within 10000
    user_fea = Feature("user_id") \
        .add_op(SelectField("user_id")) \
        .add_op(Mod(10000))
    # Map item_id feature to range within 20000
    item_fea = Feature("item_id") \
        .add_op(SelectField("item_id")) \
        .add_op(Mod(20000))
    fea_options = [user_fea, item_fea]
    

**3. Define model**

.. code-block:: python

    class SimpleCTR(nn.Module):
        def __init__(self):
            super().__init__()

            # Feature processing
            self.feature_engine = FeatureEngine(fea_options)

            # Sparse Embedding
            user_emb_opt = EmbeddingOption(
                embedding_dim=16,
                shared_name="user_emb",
            )
            item_emb_opt = EmbeddingOption(
                embedding_dim=16, 
                shared_name="item_emb"
            )
            self.embedding_engine = EmbeddingEngine(
                {"user_emb": user_emb_opt, "item_emb": item_emb_opt}
            )
            
            # Dense layers
            self.dnn = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            self.loss_fn = nn.BCELoss()
        
        def forward(self, batch):
            # Feature processing
            batch = self.feature_engine(batch)
            # Embedding lookup
            batch = self.embedding_engine(batch)
            labels = batch.pop("label")
            

            # Feature concatenation
            user_emb = batch["user_emb"]
            item_emb = batch["item_emb"]
            features = torch.cat([user_emb, item_emb], dim=-1)
            
            # Prediction
            logits = self.dnn(features)
            loss = self.loss_fn(logits.squeeze(), labels.float())
            
            return loss

**4. Define dataset**

.. code-block:: python

    def get_dataset():
        worker_idx = int(os.environ.get("RANK", 0))
        worker_num = int(os.environ.get("WORLD_SIZE", 1))
        dataset = OrcDataset(
            1024, # batch size
            worker_idx=worker_idx,
            worker_num=worker_num,
            read_threads_num=2, # Number of data reading threads
            prefetch=1, # Number of prefetched data
            is_compressed=False,
            drop_remainder=True, # Remove data that doesn't fill a batch
            transform_fn=[lambda x: x[0]],
            dtype=torch.float32,
            device="cuda", # Dataset output directly to cuda
            save_interval=None,
        )
        data_paths = ["./fake_data/"]
        for path in data_paths:
            dataset.add_path(path)
        dataset.fixedlen_feature("label", [0.0])
        dataset.varlen_feature("user_id")
        dataset.varlen_feature("item_id")
        return dataset

**5. Train model**

.. code-block:: python

    def train():
        # Create model
        model = SimpleCTR()
        
        # Separate sparse and dense parameters
        sparse_params = filter_out_sparse_param(model)
        
        # Create optimizers
        sparse_optimizer = SparseAdamW(sparse_params, lr=0.001)
        dense_optimizer = AdamW(model.parameters(), lr=0.001)
        
        # Create dataset
        train_dataset = get_dataset()
        
        # Training configuration
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            train_steps=100,
            log_steps=10,
            save_steps=50
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            dense_optimizers=(dense_optimizer, None),
            sparse_optimizer=sparse_optimizer
        )
        
        # Start training
        trainer.train()

    if __name__ == "__main__":
        train()

Advanced Features
-----------------

**Distributed Training**

.. code-block:: python

    import torch.distributed as dist
    
    # Initialize distributed environment
    dist.init_process_group()

**Enable GPU HashTable**

.. code-block:: python

    user_emb_opt = EmbeddingOption(
        embedding_dim=16,
        shared_name="user_emb",
        device=torch.device("cuda"),
    )
    item_emb_opt = EmbeddingOption(
        embedding_dim=16, 
        shared_name="item_emb",
        device=torch.device("cuda"),
    )

**Single-machine Multi-GPU Concurrency Tuning**

.. code-block:: python

    from multiprocessing import cpu_count
    def set_num_threads():
        cpu_num = cpu_count() // 16
        os.environ["OMP_NUM_THREADS"] = str(cpu_num)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
        os.environ["MKL_NUM_THREADS"] = str(cpu_num)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
        torch.set_num_interop_threads(cpu_num)
        torch.set_num_threads(cpu_num)
        # set device for local run
        torch.cuda.set_device(int(os.getenv("RANK", "-1")))

**How to read saved model data**

.. code-block:: python

    from recis.serialize.checkpoint_reader import CheckpointReader
    
    # create ckpt reader
    reader = CheckpointReader("./model_dir")
    
    # get all tensor names
    print(reader.tensor_names())


**Performance monitoring**

The built-in ``Trainer`` automatically reports QPS, FLOPS, and MFU. Use
``monitor_report_args`` to configure the interval, eval/train FLOPS ratio, and
MFU quality threshold:

.. code-block:: python

    from recis.hooks.monitor_report_hook import ReportArguments

    report_args = ReportArguments(
        interval_step=100,
        eval_flops_ratio=1.0 / 3.0,
        min_peak_coverage=0.99,
    )

    trainer = Trainer(
        # Other arguments are omitted
        monitor_report_args=report_args,
    )

Add ``ProfilerHook`` separately only when an operator-level Timeline is needed:

.. code-block:: python

    from recis.hooks import ProfilerHook

    trainer.add_hooks([
        ProfilerHook(
            wait=1,
            warmup=28,
            active=2,
            repeat=1,
            output_dir="./timeline/",
        )
    ])

See :doc:`monitoring_en` for automatic metrics, mixed-precision MFU, the
startup sampling lifecycle, and how to disable monitoring.

Next Steps
----------

Now that you've mastered the basics of RecIS, you can:

1. See :doc:`api/index` for detailed API documentation
2. See :doc:`examples/index` for more examples
3. Read :doc:`faq_en` to troubleshoot common issues

If you encounter problems, you can:

- Check the project's `Issues <https://github.com/alibaba/RecIS/issues>`
- Join the technical discussion group for help
