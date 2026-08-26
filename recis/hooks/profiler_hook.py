import os
import threading
import traceback

from torch.profiler import ProfilerActivity, profile, schedule

from recis.hooks.hook import Hook
from recis.hooks.initial_profiler_hook import (
    _InitialProfilerHook as _InitialProfilerHook,  # Compatibility re-export.
)
from recis.hooks.initial_profiler_hook import USER_PROFILER_CREATE_STEP
from recis.info import is_internal_enabled
from recis.utils.logger import Logger


# Seconds a trace save (export/upload/MOS RPCs) may block before the trace
# is abandoned, so a stuck filesystem or RPC cannot stall the training step.
_TRACE_SAVE_TIMEOUT = 600


def run_with_timeout(action, timeout, what, logger):
    """Run action() with a bounded wait; abandon it after ``timeout`` seconds.
    """
    outcome = {}

    def _run():
        try:
            action()
        except BaseException:  # noqa: BLE001 - reported back for logging
            outcome["error"] = traceback.format_exc()

    worker = threading.Thread(target=_run, name="recis-trace-save", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.error(
            f"{what} timed out after {timeout}s; abandoning this trace "
            "(the blocked worker thread is left running)"
        )
        return False
    if "error" in outcome:
        logger.error(f"{what} failed:\n{outcome['error']}")
        return False
    return True


class ProfilerHook(Hook):
    """Hook for performance profiling during training.

    The ProfilerHook uses PyTorch's profiler to collect detailed performance metrics
    during training. It captures CPU and GPU activities, memory usage, operation shapes,
    and FLOP counts. The profiling results are saved as Chrome trace files for
    visualization in Chrome's tracing tool.

    Args:
        wait (int): Number of steps to wait before starting profiling. Must be
            greater than 0 for the step-driven profiler lifecycle. The public
            profiler is created at global step 10; wait/warmup/active are local
            to that point (for example wait=1, warmup=1 starts recording global
            step 12). Defaults to 1.
        warmup (int): Number of warmup steps before active profiling. Defaults to 48.
        active (int): Number of active profiling steps. Defaults to 1.
        repeat (int): Number of profiling cycles to repeat. Defaults to 4.
        output_dir (str): Directory to save profiling results. Defaults to "./".

    Raises:
        ValueError: If wait is not greater than zero. Older releases used an
            AssertionError for this invalid public argument.

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
        if wait <= 0:
            raise ValueError("ProfilerHook wait must be greater than 0")
        if output_dir.startswith("model"):
            assert is_internal_enabled(), "Cannot import mos, check internal version."
            # Lazy import: avoids the recis.hooks -> recis.framework import
            # cycle.
            from recis.utils.mos import Mos

            output_dir = Mos(output_dir).real_physical_path
        self.output_dir = output_dir

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
            # Lazy import: avoids the recis.hooks -> recis.framework import
            # cycle.
            from recis.framework.filesystem import get_file_system

            rank = os.environ.get("RANK", "0")
            local_save_file = f"{os.environ.get('APP_ID', 'local')}-{rank}-timeline-{prof.step_num}.json"

            def _save():
                # the output dir may be a remote uri, so let fsspec create it
                fs = get_file_system(self.output_dir)
                fs.makedirs(self.output_dir, exist_ok=True)
                remote_save_file = os.path.join(self.output_dir, local_save_file)
                prof.export_chrome_trace(local_save_file)
                fs.put_file(local_save_file, remote_save_file)
                self.logger.info(f"Save profiler result : {remote_save_file}")

            run_with_timeout(
                _save,
                _TRACE_SAVE_TIMEOUT,
                f"Save profiler result {local_save_file}",
                self.logger,
            )

        return default_trace_handler

    def get_prof_schedule(self):
        scheduler = schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
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

        # Constructing a profile object never acquires Kineto. Start the public
        # profiler's local schedule at step 10, after the internal profiler has
        # fully exited at step 9. wait > 0 keeps schedule(0) at NONE, so the
        # first prof.step() begins from a valid inactive state.
        if before_step_count == USER_PROFILER_CREATE_STEP:
            # This hook intentionally keeps the existing step-driven lifecycle:
            # schedule transitions prepare/start Kineto when prof.step() moves
            # out of NONE. Adding start() without a paired stop/error lifecycle
            # would change the behavior of this public hook.
            try:
                self.prof = self.get_prof_schedule()
            except Exception:
                self.logger.error(
                    f"Failed to create profiler:\n{traceback.format_exc()}"
                )

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

        if self.prof_step_count < USER_PROFILER_CREATE_STEP:
            return  # not my stage of after_steps
        if self.prof is None:
            return  # my stage, but not initialized yet
        try:
            self.prof.step()
        except Exception:
            # another profiler may hold the kineto session; dropping this
            # timeline keeps the training loop alive
            self.prof = None
            self.logger.error(f"Profiler step failed:\n{traceback.format_exc()}")
