"""Per-module performance profilers for RecIS models.

Three profilers, all zero-touch (they wrap module.forward, no model edits):

    ModuleProfiler   dense tower: per-module wall / kernel / gemm time, FLOPs and
                     TFLOP/s, plus process-wide kernel and backward breakdowns.
    SparseProfiler   sparse tower: per-stage host / kernel / memory / all2all
                     time and communication volume, keyed on traffic not FLOPs.
    CombinedProfiler both towers in one profiling window, so their numbers are
                     directly comparable, with a step-composition overview.

For training, ModuleFlopsHook (recis.hooks) drives a CombinedProfiler over a few
real steps and then un-instruments the model.
"""

from recis.utils.profiler.combined_profiler import CombinedProfiler
from recis.utils.profiler.module_profiler import ModuleProfiler, profile_steps
from recis.utils.profiler.ncu_profiler import run_ncu_profiling
from recis.utils.profiler.sparse_profiler import SparseProfiler


__all__ = [
    "ModuleProfiler",
    "SparseProfiler",
    "CombinedProfiler",
    "profile_steps",
    "run_ncu_profiling",
]
