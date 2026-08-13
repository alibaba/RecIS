Hooks
============

详细API文档： :doc:`../../api/hooks`

训练指标统计大盘（阿里内部功能）
----------------------------------------

参考文档：:ref:`nebula-mltracker`

.. code-block:: python

    # trainer: Trainer
    # project_name: 指标统计项目名称
    # experiment_name: 指标统计实验名称
    # track_config: 指标统计其他配置

    if os.environ['RANK'] == '0':
        ml_tracker_hook = MLTrackerHook(
            project_name,
            experiment_name,
            track_config,
        )
        trainer.add_hooks([ml_tracker_hook])

预测结果Trace到ODPS（阿里内部功能）
--------------------------------------------

.. code-block:: python

    # trainer: Trainer
    # fields: 预测结果列名
    # fields = ["id", "preds", "labels"]
    # types: 预测结果列类型
    # types = ["string", "string", "string"]

    trace_hook_config = {
        "access_id": "xxx",
        "access_key": "xxx",
        "end_point": "xxx",
        "project": "xxx",
        "table_name": "xxx",
        "partition": "ds1=xxx,ds2=xxx",
    }
    trace_hook = TraceToOdpsHook(
        config=trace_hook_config, fields=fields, types=types, worker_num=8
    )
    trainer.add_hooks([trace_hook])

运行指标监控
------------

内置 ``Trainer`` 会自动注册运行指标 hook，上报 QPS、FLOPS 和 MFU；无需手工
添加 ``MetricReportHook``。配置方式、启动采样机制和混合精度口径详见
:doc:`../../monitoring`。

Timeline分析
------------------

.. code-block:: python

    # trainer: Trainer
    # output_dir: 输出目录

    if int(os.environ.get("RANK", 0)) == 0:
        hooks = ProfilerHook(
            wait=1,
            warmup=249,
            active=1,
            repeat=2,
            output_dir=output_dir,
        )
        trainer.add_hook(hooks)

公开 profiler 在全局 step 10 创建，``wait/warmup/active/repeat`` 从该位置开始
按相对 schedule 执行。
