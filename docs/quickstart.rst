快速开始
========

本章节将通过一个简单的 CTR 模型示例，帮助您快速上手 RecIS。

基础概念
--------

在开始之前，了解 RecIS 的几个核心概念：

- **Feature**: 输入特征
- **FeatureEngine**: 特征处理引擎
- **EmbeddingOption**: Embedding表配置选项
- **EmbeddingEngine**: Embedding处理引擎
- **SparseOptimizer**: 稀疏参数优化器
- **Trainer**: 训练管理器

第一个模型
----------

生成示例ORC数据
~~~~~~~~~~~~~~~

**1. 导入ORC相关模块**

.. code-block:: python

    import os

    import numpy as np
    import pyarrow as pa
    import pyarrow.orc as orc

**2. 准备数据**

.. code-block:: python

    # 数据产出地址
    file_dir = "./fake_data/"
    # 每个文件的样本数目
    bs = 2048
    # 文件总数
    file_num = 10

    # 特征
    label = np.floor(np.random.rand(bs, 1) + 0.5, dtype=np.float32)
    user_id = np.arange(bs, dtype=np.int64).reshape(bs, 1)
    item_id = np.arange(bs, dtype=np.int64).reshape(bs, 1)

    data = {
        "label": label.tolist(),
        "user_id": user_id.tolist(),
        "item_id": item_id.tolist(),
    }

    table = pa.Table.from_pydict(data)
    # 写出文件
    for i in range(file_num):
        orc.write_table(table, os.path.join(file_dir, "data_{}.orc".format(i)))

创建简单的 CTR 预估模型
~~~~~~~~~~~~~~~~~~~~~~~~

**1. 导入必要的模块**

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

**2. 定义特征工程**

.. code-block:: python

    # user_id 特征映射到10000范围内
    user_fea = Feature("user_id") \
        .add_op(SelectField("user_id")) \
        .add_op(Mod(10000))
    # item_id 特征映射到20000范围内
    item_fea = Feature("item_id") \
        .add_op(SelectField("item_id")) \
        .add_op(Mod(20000))
    fea_options = [user_fea, item_fea]
    

**3. 定义模型**

.. code-block:: python

    class SimpleCTR(nn.Module):
        def __init__(self):
            super().__init__()

            # 特征处理
            self.feature_engine = FeatureEngine(fea_options)

            # 稀疏Embedding
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
            
            # 稠密层
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
            # 特征处理
            batch = self.feature_engine(batch)
            # Embedding 查找
            batch = self.embedding_engine(batch)
            labels = batch.pop("label")
            

            # 特征拼接
            user_emb = batch["user_emb"]
            item_emb = batch["item_emb"]
            features = torch.cat([user_emb, item_emb], dim=-1)
            
            # 预测
            logits = self.dnn(features)
            loss = self.loss_fn(logits.squeeze(), labels.float())
            
            return loss

**4. 定义数据集**

.. code-block:: python

    def get_dataset():
        worker_idx = int(os.environ.get("RANK", 0))
        worker_num = int(os.environ.get("WORLD_SIZE", 1))
        dataset = OrcDataset(
            1024, # batch size
            worker_idx=worker_idx,
            worker_num=worker_num,
            read_threads_num=2, # 读取数据线程数
            prefetch=1, # 预取数据个数
            is_compressed=False,
            drop_remainder=True, # 删除不满batch的数据
            transform_fn=[lambda x: x[0]],
            dtype=torch.float32,
            device="cuda", # dataset数据结果直接输出到cuda上
            save_interval=None,
        )
        data_paths = ["./fake_data/"]
        for path in data_paths:
            dataset.add_path(path)
        dataset.fixedlen_feature("label", [0.0])
        dataset.varlen_feature("user_id")
        dataset.varlen_feature("item_id")
        return dataset

**5. 训练模型**

.. code-block:: python

    def train():
        # 创建模型
        model = SimpleCTR()
        
        # 分离稀疏和稠密参数
        sparse_params = filter_out_sparse_param(model)
        
        # 创建优化器
        sparse_optimizer = SparseAdamW(sparse_params, lr=0.001)
        dense_optimizer = AdamW(model.parameters(), lr=0.001)
        
        # 创建数据集
        train_dataset = get_dataset()
        
        # 训练配置
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            train_steps=100,
            log_steps=10,
            save_steps=50
        )
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            dense_optimizers=(dense_optimizer, None),
            sparse_optimizer=sparse_optimizer
        )
        
        # 开始训练
        trainer.train()

    if __name__ == "__main__":
        train()

高级特性
--------

**分布式训练**

.. code-block:: python

    import torch.distributed as dist
    
    # 初始化分布式环境
    dist.init_process_group()

**启用GPU HashTable**

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

**单机多卡并发数调优**

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

**查看保存的模型数据**

.. code-block:: python

    from recis.serialize.checkpoint_reader import CheckpointReader
    
    # 创建ckpt reader
    reader = CheckpointReader("./model_dir")
    
    # 查看所有参数
    print(reader.tensor_names())

**性能监控**

内置 ``Trainer`` 会自动上报 QPS、FLOPS 和 MFU。可以通过
``monitor_report_args`` 配置汇报周期、eval/train FLOPS 比例和 MFU 质量门槛：

.. code-block:: python

    from recis.hooks.monitor_report_hook import ReportArguments

    report_args = ReportArguments(
        interval_step=100,
        eval_flops_ratio=1.0 / 3.0,
        min_peak_coverage=0.99,
    )

    trainer = Trainer(
        # 其他参数省略
        monitor_report_args=report_args,
    )

如需分析具体算子，再额外添加 ``ProfilerHook`` 生成 Timeline：

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

自动指标、混合精度 MFU、启动采样机制和关闭方式详见
:doc:`monitoring`。

下一步
------

现在您已经掌握了 RecIS 的基础用法，可以：

1. 查看 :doc:`api/index` 了解详细的 API 文档
2. 参考 :doc:`examples/index` 学习更多示例
3. 阅读 :doc:`faq` 解决常见问题

如果遇到问题，可以：

- 查看项目 `Issues <https://github.com/alibaba/RecIS/issues>`_
- 加入技术交流群获取帮助
