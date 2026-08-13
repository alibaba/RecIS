Frequently Asked Questions
==========================

This section collects common questions and solutions encountered when using RecIS.

Installation and Environment
----------------------------

**Q: How to verify if RecIS installation is successful?**

A: Run the following verification script:

.. code-block:: python

   import recis
   import torch
   
   print(f"RecIS version: {recis.__version__}")
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   
   # Test core functionality
   from recis.nn import DynamicEmbedding, EmbeddingOption
   
   try:
       emb_opt = EmbeddingOption(embedding_dim=16)
       embedding = DynamicEmbedding(emb_opt)
       
       ids = torch.LongTensor([1, 2, 3])
       output = embedding(ids)
       print(f"Embedding output shape: {output.shape}")
       print("✅ RecIS installation verification successful!")
   except Exception as e:
       print(f"❌ RecIS installation verification failed: {e}")

Data Processing
---------------

**Q: How to handle variable-length sequence data?**

A: Use RaggedTensor or sequence processing operations:

.. code-block:: python

   from recis.ragged.tensor import RaggedTensor
   from recis.features.op import SequenceTruncate
   
   # Method 1: Using RaggedTensor
   values = torch.LongTensor([1, 2, 3, 4, 5, 6])
   offsets = torch.LongTensor([0, 2, 4, 6])  # Three sequences: [1,2], [3,4], [5,6]
   ragged_tensor = RaggedTensor(values, offsets)
   
   # Method 2: Using sequence processing operations
   from recis.features.feature import Feature
   from recis.features.op import SelectField
   
   sequence_feature = Feature("user_history", [
       SelectField("history_ids", dtype=torch.long, from_dict=True),
       SequenceTruncate(        
        seq_len=64,
        check_length=True,
        truncate=True,
        truncate_side="left",
        n_dims=2)
   ])

**Q: How to customize data preprocessing?**

A: Through the transform_fn parameter:

.. code-block:: python

   def custom_transform(batch):
       # Custom preprocessing logic
       batch['processed_feature'] = process_feature(batch['raw_feature'])
       
       # Data type conversion
       for key in ['user_id', 'item_id']:
           if key in batch:
               batch[key] = batch[key].long()
       
       # Data normalization
       if 'score' in batch:
           batch['score'] = (batch['score'] - batch['score'].mean()) / batch['score'].std()
       
       return batch
   
   dataset = OdpsDataset(
       batch_size=1024,
       transform_fn=custom_transform
   )

Model Training
--------------

**Q: What to do when NaN or Inf appears during training?**

A: Common causes and solutions:

1. **Learning rate too high**:

   .. code-block:: python

      # Reduce learning rate
      sparse_optimizer = SparseAdamW(sparse_params, lr=0.0001)  # From 0.001 to 0.0001
      dense_optimizer = AdamW(model.parameters(), lr=0.0001)

2. **Gradient explosion**:

   .. code-block:: python

      # Add gradient clipping
      import torch.nn as nn
      
      # Add after backward propagation
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

3. **Numerical instability**:

   .. code-block:: python

      # Check input data
      def check_tensor(tensor, name):
          if torch.isnan(tensor).any():
              print(f"NaN detected in {name}")
          if torch.isinf(tensor).any():
              print(f"Inf detected in {name}")
      
      # Add checks in model
      def forward(self, batch):
          for key, value in batch.items():
              check_tensor(value, key)
          # ... model computation

**Q: How to handle class imbalance problems?**

A: Several solutions:

1. **Weighted loss function**:

   .. code-block:: python

      import torch.nn as nn
      
      # Calculate class weights
      pos_weight = (negative_samples / positive_samples)
      loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

2. **Sampling strategies**:

   .. code-block:: python

      from torch.utils.data import WeightedRandomSampler
      
      # Create sampling weights
      sample_weights = [1.0 if label == 0 else 5.0 for label in labels]
      sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

3. **Evaluation metric adjustment**:

   .. code-block:: python

      # Focus on F1, precision, recall instead of just accuracy
      from recis.metrics import AUROC
      
      auc_metric = AUROC(num_thresholds=200)
      # Also calculate precision, recall, F1

Distributed Training
--------------------

**Q: How to configure multi-node multi-GPU training?**

A: Complete distributed training configuration:

1. **Environment variable setup**:

   .. code-block:: bash

      # Master node
      export MASTER_ADDR="192.168.1.100"
      export MASTER_PORT="12355"
      export WORLD_SIZE=8
      export RANK=0
      export LOCAL_RANK=0
      
      # Other nodes
      export RANK=1  # Increment sequentially

2. **Code configuration**:

   .. code-block:: python

      import torch.distributed as dist
      import os
      
      def setup_distributed():
          # Initialize distributed environment
          dist.init_process_group(backend='nccl')
          
          # Set device
          local_rank = int(os.environ.get('LOCAL_RANK', 0))
          torch.cuda.set_device(local_rank)
          
          return local_rank
      
      # Wrap model
      local_rank = setup_distributed()
      model = model.cuda(local_rank)

3. **Launch script**:

   .. code-block:: bash

      # Launch with torchrun
      torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
               --master_addr="192.168.1.100" --master_port=12355 \
               train.py

**Q: How to synchronize metrics in distributed training?**

A: Use metrics that support distributed synchronization:

.. code-block:: python

   from recis.metrics import AUROC
   
   # Enable distributed synchronization
   auc_metric = AUROC(
       num_thresholds=200,
       dist_sync_on_step=True  # Sync every step
   )
   
   # Or manual synchronization
   def sync_tensor(tensor):
       if dist.is_initialized():
           dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
       return tensor

Performance Issues
------------------

**Q: How to optimize slow training speed?**

A: Performance tuning recommendations:

1. **Profile analysis**:

   .. code-block:: python

      from recis.hooks import ProfilerHook
      
      profiler_hook = ProfilerHook(
          wait=1,
          warmup=1,
          active=3,
          repeat=2,
          output_dir="./profile_logs",
      )
      trainer.add_hooks([profiler_hook])

   ``wait/warmup/active/repeat`` form a relative schedule after the public
   profiler is created at global step 10. See :doc:`monitoring_en` for the
   distinction between automatic QPS/FLOPS/MFU metrics and Timeline profiling.

2. **Compute Optimization**:

    .. code-block:: python

        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True

        # Use compilation optimization
        model = torch.compile(model) # PyTorch 2.0+

Troubleshooting
---------------

**Q: What should I do if I encounter a CUDA-related error?**

A: Common CUDA Errors and Solutions:

1. **CUDA out of memory**:

    .. code-block:: python

        # Reduce batch size
        batch_size = 512 # Reduce from 1024 to 512

        # Clear GPU cache
        torch.cuda.empty_cache()

2. **CUDA device mismatch**:

    .. code-block:: python

        # Ensure all tensors are on the same device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device)

**Q: How to debug the problem of model not converging?**

A: Debugging steps:

1. **Check the data**:

    .. code-block:: python

        # Check the data distribution
        print("Label distribution:", torch.bincount(labels))
        print("Feature statistics:", features.mean(), features.std())

2. **Check the model**:

    .. code-block:: python

        # Check the gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                print(f"{name}: grad_norm={grad_norm:.6f}")

3. **Adjust hyperparameters**:

    .. code-block:: python

        # Try different learning rates
        learning_rates = [0.1, 0.01, 0.001, 0.0001]

        # Try different optimizers
        optimizers = [
                torch.optim.Adam(params, lr=0.001),
                torch.optim.AdamW(params, lr=0.001),
                torch.optim.SGD(params, lr=0.01, momentum=0.9)
        ]

Need Help ?
-----------

If none of the above solutions resolve your issue:

1. **View Logs**: Carefully review the error log and stack trace.
2. **Search Documentation**: Search for relevant keywords in the documentation.
3. **View Examples**: Refer to similar example code.
4. **Submit Issue**: Submit a detailed description of the issue on GitHub.
5. **Community Help**: Join the technical discussion group for help.

**Question Template**

When seeking help, please provide the following information:

.. code-block:: text

    **Environment Information**
    - RecIS Version:
    - PyTorch Version:
    - CUDA Version:
    - Operating System:

    **Problem Description**
    - Specific Issue:
    - Expected Behavior:
    - Actual Behavior:

    **Reproduction Steps**
    1. Step 1
    2. Step 2
    3. ...

    **Error Message**
    ```
    Full Error Log
    ```

    **Relevant Code**
    ```python
    Minimal Reproducible Code Example
    ```
