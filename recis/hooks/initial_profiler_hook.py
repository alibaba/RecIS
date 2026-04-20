import os

from torch.profiler import ProfilerActivity, profile, schedule

from recis.hooks.hook import Hook
from recis.hooks.monitor_report_hook import MetricReportHook
from recis.monitor.monitor_reporter import FLOPS_NAME
from recis.utils.logger import Logger


class _InitialProfilerHook(Hook):
    """Hook for performance profiling during training.
    The _InitialProfilerHook is automatically used by the built-in data collection module of `recis`.
    Note that _InitialProfilerHook CANNOT BE USED EXTERNALLY!!!

    Due to the singleton limitation of `torch.profiler`,
    the runtime of `_InitialProfilerHook` is earlier than that of `ProfilerHook`,
    and the two do not overlap. Therefore, the `wait` period of `ProfilerHook` is longer than
    the runtime of `_InitialProfilerHook`.

    StepDuration is the midpoint step between _InitialProfilerHook and ProfilerHook.
    the start-up steps are dropped, due to the noise of prof events. so wait step cannot be too short.
    """

    StepDuration = 5

    def __init__(self, scheduler=None):
        self.logger = Logger("_InitialProfilerHook")
        self.enable = True if os.environ.get("RECIS_MONITOR_ON", "1") == "1" else False
        # make sure the finish step in model.train, is less than StepDuration
        self.wait = 4
        self.warmup = 0
        self.active = 1
        self.repeat = 1
        self.skip_first = 0

        if not scheduler:
            # none scheduler will cause Kineto singleton error when ProfilerHook works
            scheduler = schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
                skip_first=self.skip_first,
            )
        self.prof_step_count = 0
        self.prof = None
        if not self.enable:
            return
        self.prof = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=scheduler,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
            with_modules=False,
        )
        self.prof.__enter__()
        self.prof.step()

    def before_step(self, is_train=True, *args, **kwargs):
        if not self.enable:
            return
        if not self.prof:
            return
        self.prof.step()

    def after_step(self, is_train=True, *args, **kwargs):
        if not self.enable:
            return
        self.prof_step_count += 1
        effective_step_count_end = _InitialProfilerHook.StepDuration - 1
        if not self.prof:
            return

        self.prof.step()
        if self.prof_step_count == effective_step_count_end:
            total_flops = 0
            for event in self.prof.key_averages():
                if not hasattr(event, "flops") or event.flops is None:
                    continue
                total_flops += int(event.flops)
            step_flops = total_flops / (self.active * self.repeat)
            MetricReportHook._internal_profs[FLOPS_NAME] = step_flops
            self.prof.__exit__(None, None, None)
            self.prof = None
