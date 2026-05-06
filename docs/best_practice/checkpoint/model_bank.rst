Model Bank
================

1. 用户不提供 model bank 或者 model bank 为空时，不启用任何 model bank 的解析功能。
2. 用户提供 model bank 后，首先检查 model bank 中字段是否合法。若合法，将按 model_bank 中提供的模型地址加载模型。

model bank 基础规则
---------------------

model_bank 内含有各个字段，一个简单的 model_bank 示例如下所示，之后会详细介绍各个字段的含义。

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_1,
            "load": ["*"],
            "exclude": ["io_state"],
        }
    ]


字段及含义
~~~~~~~~~~~~~~~~~

.. list-table:: model bank 字段说明
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - key
     - value
     - 含义
     - 示例
     - 备注
   * - path
     - str
     - 加载模型路径
     - "model.xx.yy.v1"
     - 必写，且不能为空字符串。
   * - load
     - List: [str]
     - 用来单独加载模型中的变量，* 表示全部加载
     - ["var1", "var2"]
     - 可选项，默认值 ["*"]，表示加载这个模型。
   * - exclude
     - List: [str]
     - 和 load 相反，表示不加载模型中的这些变量
     - ["var1", "var2"]
     - 可选项，默认值为 [""]，表示没有不加载的变量。
   * - is_dynamic
     - bool
     - 为 True 时，下一次加载的时候（例如间隔step或者间隔 window），检查 version 是否更新，更新后加载新模型。否则不加载。
     - True 或者 False
     - 可选项，默认值 False
   * - hashtable_clear
     - bool
     - 为 True 时，清空当前 tensor，load 新模型。为 False 时，增量加载。
     - True 或者 False
     - 可选项，默认值为 True
   * - oname
     - dict
     - A 模型加载 B 模型时，两个模型中的 tensor 名可能不一致，使用 oname 进行名称映射
     - "oname": { "name_A1": "name_B1", "name_A2": "name_B2"}
     - 可选项，默认值为空字典
   * - ignore_error
     - bool
     - 为 True 时，打印模型名不匹配等错误日志。为 False 时，报错退出。
     - True 或者 False
     - 详细用法参考下文示例
   * - skip
     - bool
     - 为 True 时，跳过当前 bank 的解析，常用于搜推中的冷启/热启等场景。
     - True 或者 False
     - 默认为 False。

通配符
~~~~~~~

- 使用 * 可以匹配任意数量的任意字符。可用于 load、exclude 和 oname 字段中。如 model_s1\* 可以匹配 model_s1_var1
- 在 oname 中同样可以使用 \*。{dense.\*.weight : layer.\*.weight}，表示逐层匹配权重。dense.1.weight 匹配 layer.1.weight，dense.2.weight 匹配 layer.2.weight，依次类推。

额外字段
~~~~~~~~~~~~
load 和 exclude 除了写 tensor name 外，默认添加以下内容：

.. code-block:: python

    global_step = "global_step"
    train_io = "train_io"
    eval_io = "eval_io"
    train_window_io = "train_window_io"
    eval_window_io = "eval_window_io"
    io_state = "io_state"

- 声明为 io_state 时，默认包含 train_io、eval_io、train_window_io、eval_window_io 和 train_epoch 训练步数。
- global_step，全局步数。
- recis.dense.optim，写在 exclude 时，不加载 dense 模型的优化器。

覆盖规则
~~~~~~~~~~~~~~

会倒序解析 model_bank，靠后的 model_bank 配置会覆盖前面的 model_bank，越靠后的优先级越高：

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_2,
            "load": ["table_1*"],
        },
        {
            "path": ckpt_3,
            "load": ["*"],
        },
        {
            "path": ckpt_4,
            "load": ["table_1*"],
        }
    ]

在这个配置中共有三条 model_bank 配置：

- 第三条优先级最高，所以 table_1 相关的稀疏表内容从 ckpt_4 加载
- 而后是第二条 model_bank，从 ckpt_3 加载 model 的其余内容
- 第一条优先级最低，model 的所有内容已经从 ckpt_3 和 ckpt_4 加载，所以不会在 ckpt_2 中加载任何内容

额外的，model_bank 位于 checkpoint 字段下。当 checkpoint 字段中 output_dir 不为空时，recis 会自动为 model_bank 添加一条配置，优先加载 output_dir 中的所有模型：

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_2,
            "load": ["table_1*"],
        },
        {
            "path": ckpt_3,
            "load": ["*"],
        },
        {
            "path": ckpt_4,
            "load": ["table_1*"],
        },
        {
            "path": "${checkpoint.output_dir}",
            "load": ["*"],
        }
    ]

其次才会从用户提供的 model_bank 中加载模型，达到断点续训的目的。

一定会报错
~~~~~~~~~~~~~~

1. model_bank 必须指定 path，否则不知道从哪里加载，一定会报错。报错信息为：ValueError: path must be provided。错误示例：

.. code-block:: python

    "model_bank": [
        {
            "load": ["*"],
            "exclude": ["io_state"],
            "is_dynamic": True,
        },
    ]

2. oname 匹配数量不同，会报错。table_f*会正则匹配很多东西，不能只是 table_e，可以写为 {"table_f*": "table_e*"}。错误示例：

.. code-block:: python

    "model_bank": [
        "oname": [
            {"table_f*": "table_e"}
        ],
    ]

3. 加载 dense 优化器时，param_group 数量不一致时报错退出。错误示例：

.. code-block:: python

    optimizer2 = optim.AdamW(
        [
            {
                "params": model2.classifier.named_parameters(),
                "lr": 0.01,
            }
        ]
    )
    optimizer2.add_param_group(
        {
            "params": model2.feature_extractor.named_parameters(),
            "lr": 0.15,
        }
    )
    self._model_run(model2, optimizer2)
    torch.save(optimizer2.state_dict(), "optimizer2.pth")

    new_optimizer = wrapped_named_optimizer(optim.AdamW)(model2.named_parameters())
    new_optimizer.load_state_dict(torch.load("optimizer2.pth"))

上述代码中 load_state_dict 将会报错退出，param_group 数量不一致，加载会导致歧义。

dense 优化器
~~~~~~~~~~~~

熟悉 torch 优化器的朋友一定了解，torch 优化器的加载完全依赖参数的注册顺序和 add_param_group 的顺序，和模型中的 layer 名完全无关。这种依赖顺序的加载很容易出错，所以 torch 原生优化器会检查 param_group 长度（模型 layer 的数量），不一致时直接报错退出。

但模型结构会经常发生变化，如 shape 改动、模型层数扩充等。当模型改动后，为了能顺利加载优化器，我们开发了 named_optimizer 优化器，更建议把优化器写成：

.. code-block:: python

    from recis.optim import wrapped_named_optimizer
    import torch.optim as optim

    # 继承 optim.AdamW，传入模型参数
    optimizer = wrapped_named_optimizer(optim.AdamW)(dense_model.named_parameters())

这样无论模型结构如何变化，优化器都能顺利加载。

常见示例
-------------

model_abc 这个 ckpt 中含有 table_a, table_b, table_c 三张表：

.. code-block:: python

    self.table_a = HashTable(
        [1024], name="table_a", slice=gen_slice(shard_idx, shard_num)
    )
    self.table_b = HashTable(
        [1024], name="table_b", slice=gen_slice(shard_idx, shard_num)
    )
    self.table_c = HashTable(
        [1024], name="table_c", slice=gen_slice(shard_idx, shard_num)
    )
    self.dense1 = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.Linear(512, 1),
    )
    self.dense2 = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.Linear(512, 1),
    )

model_abcde 这个 ckpt 中含有 table_a, table_b, table_c，table_d，table_e 五张表：

.. code-block:: python

    self.table_a = HashTable(
        [1024], name="table_a", slice=gen_slice(shard_idx, shard_num)
    )
    self.table_b = HashTable(
        [1024], name="table_b", slice=gen_slice(shard_idx, shard_num)
    )
    self.table_c = HashTable(
        [1024], name="table_c", slice=gen_slice(shard_idx, shard_num)
    )
    self.table_d = HashTable(
        [1024], name="table_d", slice=gen_slice(shard_idx, shard_num)
    )
    self.table_e = HashTable(
        [1024], name="table_e", slice=gen_slice(shard_idx, shard_num)
    )
    self.dense1 = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.Linear(512, 1),
    )
    self.dense2 = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.Linear(512, 1),
    )


加载部分模型
~~~~~~~~~~~~~

如果 model 只含有 a, b。a，b 从 model_abc 加载：

.. code-block:: python

    "model_bank": [
        {
            path : model_abc,
            load: ["*"]
        }
    ]

从两个不同位置分别加载部分 sparse 参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果 model 含有 a, b，c，d，e。a，b，c 从 model_abc 加载，model_abc 无法提供 d，e，所以 d，e 从 model_abcde 加载：

.. code-block:: python

    "model_bank": [
        {
            path : model_abcde,
            load: ["*"]
        },
        {
            path : model_abc,
            load: ["*"]
        }
    ]

ckpt 和模型中名字不一致
~~~~~~~~~~~~~~~~~~~~~~~~~

如果 model 只含有 a, b，c，d，f。可以使用 oname 功能，table_f 去加载model_abcde 的 table_e。

.. code-block:: python

    "model_bank": [
        {
            path : model_abcde,
            load: ["*"],
            oname: [{"table_f*": "table_e*"}]
        },
        {
            path : model_abc,
            load: ["*"]
        }
    ]


单个字段详解
------------

当前训练的模型称为 model，需要加载的远端模型称为 ckpt。下方示例所用的 model 结构：

.. code-block:: python

    class Model(torch.nn.Module):
        def __init__(self, shard_idx=0, shard_num=1):
            super().__init__()
            self.shard_idx = shard_idx
            self.shard_num = shard_num
            self.table_1 = HashTable(
                [1024], name="table_1", slice=gen_slice(shard_idx, shard_num)
            )
            self.table_2 = HashTable(
                [1024], name="table_2", slice=gen_slice(shard_idx, shard_num)
            )

            # self.dense = DenseModel(num_layers=1000)
            self.dense1 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Linear(512, 1),
            )
            self.dense2 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Linear(512, 1),
            )

        def forward(self, x):
            return self.dense1(self.table_1(x) + self.table_2(x)) + self.dense2(
                self.table_2(x)
            )

全部加载
~~~~~~~~

从 ckpt_1 加载除 io_state 外的所有 tensor。

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_1,
            "load": ["*"],
            "exclude": ["io_state"],
            "is_dynamic": False,
        }
    ]

只加载稀疏、不加载稠密
~~~~~~~~~~~~~~~~~~~~~~~~~

从 ckpt_2 加载所有 table 开头的 tensor。

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_2,
            "load": ["table*"],
        }
    ]

稀疏、稠密分别加载
~~~~~~~~~~~~~~~~~~~

从 ckpt_4 加载系数表 table_2，从 ckpt_3 加载 dense 模型。

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_4,
            "load": ["table_2*"],
        },
        {
            "path": ckpt_3,
            "load": ["dense*"],
        },
        {
            "path": ckpt_4,
            "load": ["table_1*"],
        },
    ]

oname 用法
~~~~~~~~~~~~~~~~~~~

从 ckpt_10 加载 table_1 和 table_2 的 tensor，以及 dense 模型的 tensor。其中 table_1 和 table_2 的 tensor 名称不一致，需要使用 oname 进行映射。

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_10,
            "load": ["table_1*", "table_2*", "dense*"],
            "exclude": ["io_state"],
            "oname": [
                {"table_1*": "table_2*"},
                {"table_2*": "table_1*"},
                {"dense1*": "dense2*"},
                {"dense2*": "dense1*"},
            ],
        }
    ]

is_dynamic 用法
~~~~~~~~~~~~~~~~~~~

为 True 时，下一次加载的时候（例如间隔step或者间隔 window），加载更新后的新模型。

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_10,
            "load": ["dense*"],
            "exclude": ["io_state"],
            "is_dynamic": False,
        },
        {
            "path": ckpt_10,
            "load": ["table_1*", "table_2*"],
            "exclude": ["io_state"],
            "is_dynamic": True,
        },
    ]

复杂示例
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_3,
            "load": ["*"],
            "exclude": ["table_1@*avg*", "io_state"],
        },
        {
            "path": ckpt_4,
            "load": ["table_1*"],
            "exclude": ["table_1@*avg*", "io_state"],
            "is_dynamic": False,
            "hashtable_clear": True,
        },
        {
            "path": ckpt_4,
            "load": ["table_2*"],
            "exclude": ["io_state"],
            "is_dynamic": False,
            "hashtable_clear": True,
        },
        {
            "path": ckpt_6,
            "load": ["dense*"],
            "exclude": ["io_state"],
        },
    ]

1. dense 模型、优化器参数从 ckpt_6 加载。
2. table_2 的 id、embedding、优化器状态从 ckpt_4 加载。
3. table_1 的 id 和 embedding 从 ckpt_4 加载。不加载优化器。
4. 除上述内容外，其余所有相关内容从 ckpt_3 加载。

ignore_error 示例
~~~~~~~~~~~~~~~~~~~~~~~~

适用范围：

1. oname 的 key 不在 model 中，或者 value 不在 ckpt 中
2. 用户声明的 tensor name，不在 model 中
3. 用户声明的 tensor name，不在 ckpt 中


oname 映射失败
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_10,
            "load": ["table_1"],
            "exclude": ["io_state"],
            "oname": [
                {"table_1@id": "table_7@id"},
                {"table_1@embedding": "table_7@embedding"},
            ],
            "is_dynamic": True,
            "ignore_error": False,
        },
    ]

ignore_error 为 false 时会报错并退出：Bad oname, Dst table table\_7\@id not found in dst_names。为 True 时，会打印警告信息，继续执行。

tensor name 不在 model 中
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    "model_bank": [
        {
            "path": ckpt_10,
            "load": ["table_3"],
            "exclude": ["io_state"],
            "is_dynamic": False,
            "ignore_error": False,           # ignore 为 false
        },
    ]

ignore_error 为 false 时会报错并退出，ValueError: Variable table_3 not found in model names。为 True 时，会打印警告信息，继续执行。

tensor name 不在 ckpt 中
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- model A 含有 table_1 和 table_3，运行保存到 ckpt
- model B 含有 table_1 和 table_2，加载 ckpt。table_1 可以被加载，如果 load 中写明要加载 table_2，如果 ignore_error 为 True，输出警告日志；否则报错退出。


常见警告
~~~~~~~~~

1. [WARNING] [recis.framework.checkpoint_manager] Load dense optimizer from /tmp/tmppnfd5cgw/ckpt_10 may cause error, please upgrade to PyTorch>=2.6.0 and use named optimizer。建议用户升级 torch，使用 recis 提供的 named_optimizer。
2. No var dense2.0.weight found in dst_names, ckpt path: ckpt_10。常见于用户写了 load: ["*"]，但是模型名 dense2.0.weight 不在 ckpt 中。
