训练性能监控
============

RecIS 提供两类用途不同的性能观测能力：

* ``MetricReportHook`` 由内置 ``Trainer`` 自动注册，持续上报 QPS、FLOPS 和
  MFU 等运行指标。用户通常只需要配置口径，不需要手工添加 hook。
* ``ProfilerHook`` 是用户按需添加的公开 Timeline 工具，用于生成
  ``torch.profiler`` Chrome trace，定位具体算子和调度问题。

自动指标监控
------------

默认情况下，使用 ``recis.framework.Trainer`` 即会启用运行指标监控。
可以通过 ``monitor_report_args`` 调整汇报周期和 MFU 口径：

.. code-block:: python

   from recis.framework import Trainer
   from recis.hooks.monitor_report_hook import ReportArguments

   report_args = ReportArguments(
       interval_step=100,
       eval_flops_ratio=1.0 / 3.0,
       min_peak_coverage=0.99,
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       dense_optimizers=(dense_optimizer, lr_scheduler),
       sparse_optimizer=sparse_optimizer,
       monitor_report_args=report_args,
   )

主要参数如下：

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - 参数
     - 默认值
     - 作用
   * - ``interval_step``
     - ``100``
     - 每多少个实际 step 汇报一次；``None`` 表示关闭该 hook 的周期指标汇报。
   * - ``eval_flops_ratio``
     - ``1/3``
     - 一个 eval step 相对一个 train step 的 FLOPS 比例。
   * - ``min_peak_coverage``
     - ``0.99``
     - 具有已知 dtype 峰值的 FLOPS 覆盖率门槛；不足时不汇报 MFU。
   * - ``tflops_peak``
     - 自动估计
     - 显式指定统一的设备峰值；只应在整个统计口径使用同一精度时设置。
   * - ``compute_precision``
     - 自动估计
     - 按指定精度查询设备峰值，并使用统一标量峰值计算 MFU。

默认的混合精度路径会根据 profiler 报告的算子输入 dtype 分类 FLOPS，并按设备
对应 dtype 的峰值组合出混合峰值。只有确定整个统计口径使用统一精度时，才建议
设置 ``tflops_peak`` 或 ``compute_precision`` 覆盖该行为。

关闭监控
~~~~~~~~

设置 ``RECIS_MONITOR_ON=0`` 可以关闭监控：

.. code-block:: bash

   RECIS_MONITOR_ON=0 python train.py

内部启动 profiler 会在创建 ``torch.profiler`` 之前返回，因此不会产生 Kineto、
``record_shapes`` 或 event tree 采集成本。

上报指标
~~~~~~~~

主要指标使用 ``recis.framework`` 前缀：

.. list-table::
   :header-rows: 1
   :widths: 30 28 42

   * - 指标
     - 关键 tag
     - 含义
   * - ``recis.framework.qps``
     - ``recis_qps_type``
     - 窗口总 QPS，以及 train/eval QPS。
   * - ``recis.framework.flops``
     - ``recis_flops_type=flops``
     - profiler 可计数算子的实际 FLOPS/s。
   * - ``recis.framework.flops``
     - ``recis_flops_type=flops_peak``
     - MFU 使用的设备混合峰值或显式标量峰值。
   * - ``recis.framework.mfu``
     - ``recis_mfu_type=mfu``
     - profiler 可见算子的理想执行时间与实际窗口时间之比。

作用机制
--------

启动采样只执行一次，后续汇报不会重复运行 profiler：

1. step 1--5 跳过；step 6 用于 profiler warmup；连续采样 step 7--9。
2. step 9 的 ``after_step`` 先调用 ``prof.step()`` 完整关闭 trace，再读取事件。
3. 三步 FLOPS 均值按照采样 step 类型归一为 train-equivalent FLOPS/step，写入
   ``StartupFlopsState``。
4. ``MetricReportHook`` 与内部 profiler 共享同一个 state。每次汇报只读取缓存，
   再根据窗口内真实的 train/eval step 数换算，因此热路径为常数时间。

令 ``r`` 为 ``eval_flops_ratio``，则：

.. math::

   window\_scale = train\_steps + r \times eval\_steps

.. math::

   FLOPS/s = F_{train} \times window\_scale / elapsed

其中 ``F_train`` 是缓存的 train-equivalent FLOPS/step。该口径同时支持
train 采样/train 汇报、train 采样/eval 汇报、eval 采样/train 汇报和 eval
采样/eval 汇报；不依赖对训练相位的猜测。

混合精度 MFU
~~~~~~~~~~~~~

对每种 profiler 输入 dtype，先计算其理想执行时间，再求和：

.. math::

   T_{ideal} = \sum_d F_d / P_d

.. math::

   MFU = T_{ideal} \times window\_scale / elapsed

``T_ideal`` 和混合峰值只在启动采样关窗时计算一次，周期汇报不会重复遍历事件。

质量边界
~~~~~~~~

* ``torch.profiler(with_flops=True)`` 主要覆盖矩阵乘和二维卷积；融合、自定义、
  sparse、embedding/gather 等算子可能不计入。因此这里的 MFU 是 profiler 可见
  算子集合的 lower-bound 指标，不等同于论文中的模型全量 MFU。
* dtype 分类依据是 profiler 报告的算子输入 dtype，不推断后端实际数学模式。
  FP32 输入可能实际使用 TF32 或其他内部算法；启动日志会标记可疑情况，但
  不会猜测并改写精度。
* ``input_dtype_coverage`` 或 ``peak_coverage`` 不足时仍可汇报成立的 FLOPS，
  但启动日志会记录无效原因并停止汇报 MFU，避免制造偏高的假值。
* 启动有效采样窗口内如果 train/eval 类型混合，或训练在 step 9 前结束，启动
  估计会保守失效。此时 QPS 仍可正常工作。

Timeline 分析
-------------

需要分析具体算子时，额外添加公开 ``ProfilerHook``：

.. code-block:: python

   import os
   from recis.hooks import ProfilerHook

   if int(os.environ.get("RANK", "0")) == 0:
       trainer.add_hooks([
           ProfilerHook(
               wait=1,
               warmup=28,
               active=2,
               repeat=1,
               output_dir="./timeline/",
           )
       ])

公开 profiler 在全局 step 10 创建，以确保内部 FLOPS profiler 已经完全释放
Kineto。``wait/warmup/active/repeat`` 从该创建点开始按相对 schedule 执行，不再
叠加历史全局 offset；``wait`` 必须大于 0。例如 ``wait=1, warmup=1`` 会从全局
step 12 开始记录。

升级与兼容
----------

* 已经运行的任务继续使用启动时安装的 wheel，不会被分支或新 wheel 热替换。
* 纯 RecIS ``Trainer`` 在框架内部共享启动估计 state，无需额外 wiring。
* 使用 rank-util Trainer 时，需要使用包含共享 state 适配器的 rank-util 版本。
  该适配器会检查已安装的 RecIS：老 RecIS 继续走原无参构造，新 RecIS 才启用
  共享 state 和混合精度估计。
* 升级后的公开 ``ProfilerHook`` 从全局 step 10 开始执行其相对 schedule。这是
  为确保 Kineto 生命周期隔离而保留的一项有意行为变化。
