"""A recis Hook that profiles per-module time and FLOPs during real training.

recis already ships ProfilerHook, which exports a Chrome trace. That is the right
tool for looking at a timeline, but it does not answer "which module costs what",
and the deepspeed FlopsProfiler that would answer it was measured to inflate
latency 55-311x on this platform while reporting 0 MACs for einsum.

This hook runs recis.utils.profiler over a few real training steps and
prints the per-module breakdown, so no separate profiling script or synthetic
input is needed.

Usage in runner.py, next to the existing ProfilerHook:

    from recis.hooks import ModuleFlopsHook

    if int(os.environ.get("RANK", 0)) == 0:
        hooks = [
            ModuleFlopsHook(model=graph, start_step=20, steps=3, max_depth=2,
                            output_dir=trainer.output_dir),
        ]
        trainer.add_hooks(hooks)

    Passing the whole model is enough: the towers are detected from it. dense= and
    sparse= remain available to override the detection or to profile only one.

Design notes:
  - one-shot: it profiles a window of steps once and then fully un-instruments the
    model, so the rest of training runs at full speed
  - the steps before start_step act as warmup, which is required here (Triton JIT
    and allocator growth otherwise land inside the measurement); start_step should
    be at least a few steps in
  - rank-0 only by default, so a distributed job does not print N copies
  - the model is restored in a finally block; if anything goes wrong the training
    job must not be left running with instrumented forwards
"""

import os
import traceback

from recis.hooks import Hook
from recis.hooks.initial_profiler_hook import _InitialProfilerHook
from recis.utils.logger import Logger
from recis.utils.profiler.combined_profiler import CombinedProfiler


class ModuleFlopsHook(Hook):
    """Profile per-module device time and FLOPs for a window of training steps.

    Args:
        model: the whole model; the dense and sparse towers are then detected
            automatically, so the caller does not have to split them.
        dense: the dense tower, or None. Overrides detection when given.
        sparse: the sparse tower, or None. Overrides detection when given.
        start_step: step index at which to begin profiling. Everything before it
            serves as warmup.
        steps: how many steps to profile.
        max_depth: module-tree depth to instrument. Keep it small; instrumenting
            every leaf of a wide model (this one has 173 FC blocks) adds a CUDA
            event pair per call and skews the shallow numbers.
        include: optional name prefixes to instrument beyond max_depth, for
            drilling into one subtree, e.g. ["main_net.transformer"].
        output_dir: if given, the report is also written to
            {output_dir}/module_profile-{rank}-step{N}.txt
        rank0_only: only profile on rank 0.
        report_kwargs: forwarded to the report (depth, top_kernels).
    """

    def __init__(
        self,
        model=None,
        dense=None,
        sparse=None,
        start_step=20,
        steps=3,
        max_depth=2,
        include=None,
        output_dir=None,
        rank0_only=True,
        **report_kwargs,
    ):
        self.logger = Logger("ModuleFlopsHook")
        self.model = model
        self.dense = dense
        self.sparse = sparse
        min_start = _InitialProfilerHook.StepDuration + 1
        self.start_step = max(int(start_step), 1)
        if self.start_step < min_start:
            self.logger.warning(
                f"start_step={self.start_step} overlaps with _InitialProfilerHook "
                f"(exits at step {_InitialProfilerHook.StepDuration}), "
                f"clamping to {min_start} to avoid torch.profiler singleton conflict"
            )
            self.start_step = min_start
        self.steps = max(int(steps), 1)
        self.max_depth = max_depth
        self.include = include
        self.output_dir = output_dir
        self.rank0_only = rank0_only
        self.report_kwargs = report_kwargs

        self._rank = int(os.environ.get("RANK", 0))
        self._step = 0
        self._prof = None
        self._done = False

    def _enabled(self):
        if self.model is None and self.dense is None and self.sparse is None:
            return False
        return not (self.rank0_only and self._rank != 0)

    def before_step(self, is_train=True, *args, **kwargs):
        if self._done or not is_train or not self._enabled():
            return
        self._step += 1
        if self._step != self.start_step:
            return
        try:
            self._prof = CombinedProfiler(
                model=self.model,
                dense=self.dense,
                sparse=self.sparse,
                max_depth=self.max_depth,
                include=self.include,
            )
            self._prof.start_profile()
            towers = ",".join(
                t
                for t, m in (("dense", self._prof.dense), ("sparse", self._prof.sparse))
                if m
            )
            self.logger.info(
                f"combined profiling ({towers}) started at step {self._step} "
                f"for {self.steps} steps"
            )
        except Exception:
            self.logger.error(
                f"failed to start module profiling:\n{traceback.format_exc()}"
            )
            self._teardown()

    def after_step(self, is_train=True, *args, **kwargs):
        if self._done or self._prof is None or not is_train:
            return
        try:
            self._prof.step()
            if self._step - self.start_step + 1 < self.steps:
                return
        except Exception:
            self.logger.error(
                f"module profiling step failed:\n{traceback.format_exc()}"
            )
            self._teardown()
            return

        # window complete
        try:
            self._prof.stop_profile()
            self._report()
        except Exception:
            self.logger.error(
                f"module profiling report failed:\n{traceback.format_exc()}"
            )
        finally:
            self._teardown()

    def _emit(self, **kw):
        self._prof.print_report(**kw, **self.report_kwargs)

    def _report(self):
        self._emit()
        if not self.output_dir:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(
                self.output_dir, f"combined_profile-{self._rank}-step{self._step}.txt"
            )
            with open(path, "w") as f:
                self._emit(file=f)
            self.logger.info(f"module profile written to {path}")
        except Exception:
            self.logger.error(
                f"failed to write module profile:\n{traceback.format_exc()}"
            )

    def _teardown(self):
        """Un-instrument the model. Must run even on failure."""
        if self._prof is not None:
            try:
                self._prof.end_profile()
            except Exception:
                self.logger.error(
                    f"failed to restore model forwards:\n{traceback.format_exc()}"
                )
        self._prof = None
        self._done = True

    def after_train(self, *args, **kwargs):
        if self._prof is not None:
            self.logger.warning(
                "training ended while profiling was still active; restoring"
            )
            self._teardown()

    def end(self, *args, **kwargs):
        if self._prof is not None:
            self._teardown()
