import os
from pathlib import Path

import torch
from packaging.version import Version
from torch.profiler import ProfilerActivity, profile, schedule

from recis.framework.filesystem import get_file_system
from recis.hooks.hook import Hook
from recis.hooks.initial_profiler_hook import _InitialProfilerHook
from recis.info import is_internal_enabled
from recis.utils.logger import Logger


if is_internal_enabled():
    from recis.utils.mos import Mos
else:
    Mos = None


class ProfilerHook(Hook):
    """Hook for performance profiling during training.

    The ProfilerHook uses PyTorch's profiler to collect detailed performance metrics
    during training. It captures CPU and GPU activities, memory usage, operation shapes,
    and FLOP counts. The profiling results are saved as Chrome trace files for
    visualization in Chrome's tracing tool.

    Args:
        wait (int): Number of steps to wait before starting profiling. Defaults to 1.
        warmup (int): Number of warmup steps before active profiling. Defaults to 48.
        active (int): Number of active profiling steps. Defaults to 1.
        repeat (int): Number of profiling cycles to repeat. Defaults to 4.
        output_dir (str): Directory to save profiling results. Defaults to "./".

    Attributes:
        prof (torch.profiler.profile): PyTorch profiler instance.
        logger (Logger): Logger instance for outputting messages.
        output_dir (str): Output directory for profiling results.

    Example:
        >>> from recis.hooks import ProfilerHook
        >>> # Create profiler hook with custom settings
        >>> profiler_hook = ProfilerHook(
        ...     wait=1, warmup=28, active=2, repeat=1, output_dir="./timeline/"
        ... )
        >>> trainer.add_hook(profiler_hook)
        >>> # The hook will automatically profile training and save results
        >>> # Results will be saved as Chrome trace files (.json)

    Note:
        The profiling results can be visualized by opening the generated .json files
        in Chrome's tracing tool (chrome://tracing/).
    """

    def __init__(self, wait=1, warmup=48, active=1, repeat=4, output_dir="./"):
        self.logger = Logger("ProfilerHook")
        assert wait > 0, "ProfilerHook wait must be greater than 0"
        if output_dir.startswith("model"):
            assert Mos is not None, "Cannot import mos, check interneal version."
            output_dir = Mos(output_dir).real_physical_path
        self.output_dir = output_dir

        # leave the previous wait step for _InitialProfilerHook
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat

        self.prof = None
        self.prof_step_count = 0

    def get_trace_handler(self):
        """Creates and returns a trace handler function for profiling results.

        The trace handler is called when profiling data is ready to be saved.
        It generates a unique filename based on the application ID and step number,
        then saves the Chrome trace file to the specified output directory.

        Returns:
            callable: A function that handles trace saving when profiling is complete.

        Note:
            The generated filename format is: {APP_ID}-timeline-{step_num}.json
            where APP_ID comes from environment variables (defaults to 'local').
        """

        def default_trace_handler(prof):
            rank = os.environ.get("RANK", "0")
            local_save_file = f"{os.environ.get('APP_ID', 'local')}-{rank}-timeline-{prof.step_num}.json"

            try:
                output_dir_path = Path(self.output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(
                    f"Failed to create output directory {self.output_dir}: {e}"
                )
                return

            fs = get_file_system(self.output_dir)
            remote_save_file = os.path.join(self.output_dir, local_save_file)
            prof.export_chrome_trace(local_save_file)
            fs.put_file(local_save_file, remote_save_file)
            self.logger.info(f"Save profiler result : {remote_save_file}")

        return default_trace_handler

    def get_prof_schedule(self):
        version_current = Version(torch.__version__.split("+", 1)[0])
        # torch 2.6.0 has skip_first_wait, could align skipped step with real step
        version_with_smart_skip = Version("2.6.0")
        if version_current >= version_with_smart_skip:
            skip_first = self.wait + _InitialProfilerHook.StepDuration
            scheduler = schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
                skip_first=skip_first,
                skip_first_wait=1,
            )
        else:
            skip_first = _InitialProfilerHook.StepDuration
            scheduler = schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
                skip_first=skip_first,
            )
        prof = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=scheduler,
            on_trace_ready=self.get_trace_handler(),
            with_stack=True,
            profile_memory=True,
            record_shapes=True,
            with_flops=True,
        )
        return prof

    def before_step(self, is_train=True, *args, **kwargs):
        before_step_count = self.prof_step_count + 1
        effective_step_count_begin = _InitialProfilerHook.StepDuration + 1

        if before_step_count < effective_step_count_begin:
            pass
        elif before_step_count > effective_step_count_begin:
            pass
        else:
            # before_step_count == general_step_count, initiallize profiler
            self.prof = self.get_prof_schedule()

    def after_step(self, is_train=True, *args, **kwargs):
        """Called after each training step to advance the profiler.

        This method is invoked after each training step to advance the profiler's
        internal step counter. The profiler uses this information to determine
        when to start/stop profiling based on the configured schedule.

        Args:
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).

        Note:
            The profiler automatically handles the profiling schedule based on
            the wait, warmup, active, and repeat parameters provided during
            initialization.
        """
        self.prof_step_count += 1
        effective_step_count_begin = _InitialProfilerHook.StepDuration + 1

        if self.prof_step_count < effective_step_count_begin:
            return  # not my stage of after_steps
        if self.prof is None:
            return  # my stage, but not initialized yet
        self.prof.step()
