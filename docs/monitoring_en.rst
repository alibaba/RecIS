Training Performance Monitoring
===============================

RecIS provides two performance-observation features with different purposes:

* ``MetricReportHook`` is registered automatically by the built-in ``Trainer``
  and continuously reports runtime metrics such as QPS, FLOPS, and MFU. Users
  normally configure its accounting policy instead of adding the hook manually.
* ``ProfilerHook`` is a public, opt-in Timeline tool that writes
  ``torch.profiler`` Chrome traces for operator- and schedule-level analysis.

Automatic Runtime Metrics
-------------------------

Runtime monitoring is enabled by default when ``recis.framework.Trainer`` is
used. Pass ``monitor_report_args`` to configure the reporting interval and MFU
accounting policy:

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

The main options are:

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Option
     - Default
     - Purpose
   * - ``interval_step``
     - ``100``
     - Report every N actual steps; ``None`` disables this hook's periodic metrics.
   * - ``eval_flops_ratio``
     - ``1/3``
     - FLOPS ratio of one eval step relative to one train step.
   * - ``min_peak_coverage``
     - ``0.99``
     - Required fraction of FLOPS with a known dtype peak before MFU is reported.
   * - ``tflops_peak``
     - automatic
     - Use an explicit uniform device peak. Set this only when one precision
       applies to the complete accounting basis.
   * - ``compute_precision``
     - automatic
     - Look up a device peak for one explicit precision and use a scalar peak.

By default, the mixed-precision path groups profiler-countable FLOPS by the
operator input dtype reported by the profiler and combines the corresponding
device peaks. Override it with ``tflops_peak`` or ``compute_precision`` only
when a uniform precision is known to represent the complete accounting basis.

Disabling monitoring
~~~~~~~~~~~~~~~~~~~~

Set ``RECIS_MONITOR_ON=0`` to disable monitoring:

.. code-block:: bash

   RECIS_MONITOR_ON=0 python train.py

The internal startup hook returns before creating ``torch.profiler``, so it
incurs no Kineto, ``record_shapes``, or event-tree collection cost.

Reported metrics
~~~~~~~~~~~~~~~~

The primary metrics use the ``recis.framework`` prefix:

.. list-table::
   :header-rows: 1
   :widths: 30 28 42

   * - Metric
     - Key tag
     - Meaning
   * - ``recis.framework.qps``
     - ``recis_qps_type``
     - Total window QPS and the train/eval QPS breakdown.
   * - ``recis.framework.flops``
     - ``recis_flops_type=flops``
     - Actual profiler-countable FLOPS per second.
   * - ``recis.framework.flops``
     - ``recis_flops_type=flops_peak``
     - Mixed or explicit scalar device peak used by MFU.
   * - ``recis.framework.mfu``
     - ``recis_mfu_type=mfu``
     - Ideal time of profiler-visible operators divided by actual window time.

How It Works
------------

Startup profiling runs once; periodic reporting never reruns the profiler:

1. Steps 1--5 are skipped, step 6 warms up the profiler, and steps 7--9 are
   sampled continuously.
2. At step 9, ``after_step`` calls ``prof.step()`` to close the trace before
   reading events.
3. The three-step mean is normalized to train-equivalent FLOPS/step according to
   the sampled step type and published to ``StartupFlopsState``.
4. ``MetricReportHook`` shares that state with the internal profiler. Each
   report reads the cached estimate and scales it using the actual train/eval
   step counts in the window, making the hot path constant time.

Let ``r`` be ``eval_flops_ratio``:

.. math::

   window\_scale = train\_steps + r \times eval\_steps

.. math::

   FLOPS/s = F_{train} \times window\_scale / elapsed

``F_train`` is the cached train-equivalent FLOPS/step. This formulation supports
all four sample/report combinations (train/train, train/eval, eval/train, and
eval/eval) without guessing the current training phase.

Mixed-precision MFU
~~~~~~~~~~~~~~~~~~~

Ideal time is computed per profiler input dtype and then summed:

.. math::

   T_{ideal} = \sum_d F_d / P_d

.. math::

   MFU = T_{ideal} \times window\_scale / elapsed

``T_ideal`` and the mixed peak are computed once when startup profiling closes;
periodic reports do not traverse events again.

Quality boundaries
~~~~~~~~~~~~~~~~~~

* ``torch.profiler(with_flops=True)`` mainly covers matrix multiplication and
  2-D convolution. Fused, custom, sparse, embedding/gather, and other operators
  may be absent. The resulting MFU is therefore a lower-bound metric for the
  profiler-visible operator set, not a paper-style model-wide MFU.
* Classification uses the operator input dtype reported by the profiler, not
  the backend's actual math mode. FP32 inputs may execute with TF32 or another
  internal algorithm. Startup logs identify suspicious cases, but RecIS does
  not guess and rewrite the precision.
* When ``input_dtype_coverage`` or ``peak_coverage`` is insufficient, valid
  FLOPS may still be reported. The startup log records the invalid reason and
  MFU is omitted rather than manufacturing an inflated value.
* If train and eval step types are mixed inside the active startup window, or
  training ends before step 9, the startup estimate is conservatively invalid.
  QPS remains available.

Timeline Profiling
------------------

Add the public ``ProfilerHook`` when an operator-level trace is needed:

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

The public profiler is created at global step 10, after the internal FLOPS
profiler has fully released Kineto. ``wait/warmup/active/repeat`` form a relative
schedule from that creation point and no historical global offset is added.
``wait`` must be greater than zero. For example, ``wait=1, warmup=1`` starts
recording at global step 12.

Upgrade and Compatibility
-------------------------

* Running jobs keep the wheel installed at startup; a branch or new wheel is
  never hot-swapped into an existing container.
* The pure RecIS ``Trainer`` wires the shared startup-estimate state internally.
* A rank-util Trainer must use a rank-util release containing the shared-state
  compatibility adapter. The adapter preserves the original no-argument path
  for old RecIS releases and enables shared state only when the installed RecIS
  exposes the capability.
* After upgrading, the public ``ProfilerHook`` starts its relative schedule at
  global step 10. This is the one intentional behavior change retained to make
  the Kineto lifecycle handoff deterministic.
