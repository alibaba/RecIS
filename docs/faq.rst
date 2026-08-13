常见问题
========

本章节收集了 RecIS 使用过程中的常见问题和解决方案。

安装和环境
----------

**Q: 如何验证 RecIS 安装是否成功？**

A: 运行以下验证脚本：

.. code-block:: python

   import recis
   import torch
   
   print(f"RecIS 版本: {recis.__version__}")
   print(f"PyTorch 版本: {torch.__version__}")
   print(f"CUDA 可用: {torch.cuda.is_available()}")
   
   # 测试核心功能
   from recis.nn import DynamicEmbedding, EmbeddingOption
   
   try:
       emb_opt = EmbeddingOption(embedding_dim=16)
       embedding = DynamicEmbedding(emb_opt)
       
       ids = torch.LongTensor([1, 2, 3])
       output = embedding(ids)
       print(f"Embedding 输出形状: {output.shape}")
       print("✅ RecIS 安装验证成功！")
   except Exception as e:
       print(f"❌ RecIS 安装验证失败: {e}")

数据处理
--------

**Q: 如何处理变长序列数据？**

A: 使用 RaggedTensor 或序列处理操作：

.. code-block:: python

   from recis.ragged.tensor import RaggedTensor
   from recis.features.op import SequenceTruncate
   
   # 方法1: 使用 RaggedTensor
   values = torch.LongTensor([1, 2, 3, 4, 5, 6])
   offsets = torch.LongTensor([0, 2, 4, 6])  # 三个序列: [1,2], [3,4], [5,6]
   ragged_tensor = RaggedTensor(values, offsets)
   
   # 方法2: 使用序列处理操作
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

**Q: 如何自定义数据预处理？**

A: 通过 transform_fn 参数：

.. code-block:: python

   def custom_transform(batch):
       # 自定义预处理逻辑
       batch['processed_feature'] = process_feature(batch['raw_feature'])
       
       # 数据类型转换
       for key in ['user_id', 'item_id']:
           if key in batch:
               batch[key] = batch[key].long()
       
       # 数据归一化
       if 'score' in batch:
           batch['score'] = (batch['score'] - batch['score'].mean()) / batch['score'].std()
       
       return batch
   
   dataset = OdpsDataset(
       batch_size=1024,
       transform_fn=custom_transform
   )

模型训练
--------

**Q: 训练过程中出现 NaN 或 Inf 怎么办？**

A: 常见原因和解决方案：

1. **学习率过大**：

   .. code-block:: python

      # 降低学习率
      sparse_optimizer = SparseAdamW(sparse_params, lr=0.0001)  # 从 0.001 降到 0.0001
      dense_optimizer = AdamW(model.parameters(), lr=0.0001)

2. **梯度爆炸**：

   .. code-block:: python

      # 添加梯度裁剪
      import torch.nn as nn
      
      # 在反向传播后添加
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

3. **数值不稳定**：

   .. code-block:: python

      # 检查输入数据
      def check_tensor(tensor, name):
          if torch.isnan(tensor).any():
              print(f"NaN detected in {name}")
          if torch.isinf(tensor).any():
              print(f"Inf detected in {name}")
      
      # 在模型中添加检查
      def forward(self, batch):
          for key, value in batch.items():
              check_tensor(value, key)
          # ... 模型计算

**Q: 如何处理类别不平衡问题？**

A: 几种解决方案：

1. **加权损失函数**：

   .. code-block:: python

      import torch.nn as nn
      
      # 计算类别权重
      pos_weight = (negative_samples / positive_samples)
      loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

2. **采样策略**：

   .. code-block:: python

      from torch.utils.data import WeightedRandomSampler
      
      # 创建采样权重
      sample_weights = [1.0 if label == 0 else 5.0 for label in labels]
      sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

3. **评估指标调整**：

   .. code-block:: python

      # 关注 F1、精确率、召回率而不只是准确率
      from recis.metrics import AUROC
      
      auc_metric = AUROC(num_thresholds=200)
      # 同时计算精确率、召回率、F1


分布式训练
----------

**Q: 如何配置多机多卡训练？**

A: 完整的分布式训练配置：

1. **环境变量设置**：

   .. code-block:: bash

      # 主节点
      export MASTER_ADDR="192.168.1.100"
      export MASTER_PORT="12355"
      export WORLD_SIZE=8
      export RANK=0
      export LOCAL_RANK=0
      
      # 其他节点
      export RANK=1  # 依次递增

2. **代码配置**：

   .. code-block:: python

      import torch.distributed as dist
      import os
      
      def setup_distributed():
          # 初始化分布式环境
          dist.init_process_group(backend='nccl')
          
          # 设置设备
          local_rank = int(os.environ.get('LOCAL_RANK', 0))
          torch.cuda.set_device(local_rank)
          
          return local_rank
      
      # 包装模型
      local_rank = setup_distributed()
      model = model.cuda(local_rank)

3. **启动脚本**：

   .. code-block:: bash

      # 使用 torchrun 启动
      torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
               --master_addr="192.168.1.100" --master_port=12355 \
               train.py

**Q: 分布式训练中如何同步指标？**

A: 使用支持分布式的指标：

.. code-block:: python

   from recis.metrics import AUROC
   
   # 启用分布式同步
   auc_metric = AUROC(
       num_thresholds=200,
       dist_sync_on_step=True  # 每步同步
   )
   
   # 或者手动同步
   def sync_tensor(tensor):
       if dist.is_initialized():
           dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
       return tensor

性能问题
--------

**Q: 训练速度慢怎么优化？**

A: 性能调优建议：

1. **Profile 分析**：

   .. code-block:: python

      from recis.hooks import ProfilerHook
      
      profiler_hook = ProfilerHook(
          wait=1,
          warmup=1,
          active=3,
          repeat=2,
          output_dir="./profiler_output",
      )
      trainer.add_hooks([profiler_hook])

   ``wait/warmup/active/repeat`` 是从全局 step 10 创建公开 profiler 后开始的
   相对 schedule。QPS、FLOPS、MFU 自动监控与 Timeline 的区别详见
   :doc:`monitoring`。

2. **计算优化**：

   .. code-block:: python

      # 启用 cuDNN 基准测试
      torch.backends.cudnn.benchmark = True
      
      # 使用编译优化
      model = torch.compile(model)  # PyTorch 2.0+

错误排查
--------

**Q: 遇到 CUDA 相关错误怎么办？**

A: 常见 CUDA 错误及解决方案：

1. **CUDA out of memory**：

   .. code-block:: python

      # 减少批次大小
      batch_size = 512  # 从 1024 减少到 512
      
      # 清理 GPU 缓存
      torch.cuda.empty_cache()

2. **CUDA device mismatch**：

   .. code-block:: python

      # 确保所有张量在同一设备
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model = model.to(device)
      
      for key, value in batch.items():
          if torch.is_tensor(value):
              batch[key] = value.to(device)

**Q: 如何调试模型不收敛的问题？**

A: 调试步骤：

1. **检查数据**：

   .. code-block:: python

      # 检查数据分布
      print("标签分布:", torch.bincount(labels))
      print("特征统计:", features.mean(), features.std())

2. **检查模型**：

   .. code-block:: python

      # 检查梯度
      for name, param in model.named_parameters():
          if param.grad is not None:
              grad_norm = param.grad.norm()
              print(f"{name}: grad_norm={grad_norm:.6f}")

3. **调整超参数**：

   .. code-block:: python

      # 尝试不同的学习率
      learning_rates = [0.1, 0.01, 0.001, 0.0001]
      
      # 尝试不同的优化器
      optimizers = [
          torch.optim.Adam(params, lr=0.001),
          torch.optim.AdamW(params, lr=0.001),
          torch.optim.SGD(params, lr=0.01, momentum=0.9)
      ]

获取帮助
--------

如果以上解决方案都无法解决您的问题：

1. **查看日志**：仔细阅读错误日志和堆栈跟踪
2. **搜索文档**：在文档中搜索相关关键词
3. **查看示例**：参考相似的示例代码
4. **提交 Issue**：在 GitHub 上提交详细的问题描述
5. **社区求助**：加入技术交流群获取帮助

**提问模板**

在寻求帮助时，请提供以下信息：

.. code-block:: text

   **环境信息**
   - RecIS 版本：
   - PyTorch 版本：
   - CUDA 版本：
   - 操作系统：
   
   **问题描述**
   - 具体问题：
   - 期望行为：
   - 实际行为：
   
   **复现步骤**
   1. 步骤一
   2. 步骤二
   3. ...
   
   **错误信息**
   ```
   完整的错误日志
   ```
   
   **相关代码**
   ```python
   最小可复现的代码示例
   ```
