from recis.info import is_internal_enabled

from .clip_grad_norm_hook import ClipGradNormHook
from .filter_hook import HashTableFilterHook
from .hook import Hook
from .initial_profiler_hook import _InitialProfilerHook
from .logger_hook import LoggerHook
from .monitor_report_hook import MetricReportHook
from .profiler_hook import ProfilerHook


__all__ = [
    "Hook",
    "LoggerHook",
    "_InitialProfilerHook",
    "ProfilerHook",
    "HashTableFilterHook",
    "MetricReportHook",
    "ClipGradNormHook",
]

if is_internal_enabled():
    from .ml_tracker_hook import (
        MLTrackerHook as MLTrackerHook,
        add_to_ml_tracker as add_to_ml_tracker,
    )
    from .trace_to_odps_hook import (  # noqa: F401
        TraceToOdpsHook as TraceToOdpsHookV1,
        add_to_trace as add_to_trace,
    )
    from .trace_to_odps_hook_v2 import TraceToOdpsHookV2

    TraceToOdpsHook = TraceToOdpsHookV2

    __all__.extend(
        ["MLTrackerHook", "add_to_ml_tracker",
         "TraceToOdpsHook", "TraceToOdpsHookV1", "TraceToOdpsHookV2",
         "add_to_trace"]
    )
