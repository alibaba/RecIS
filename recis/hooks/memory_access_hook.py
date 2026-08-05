"""Memory access tracking hook for profiling custom operators."""

from recis.hooks.hook import Hook
from recis.utils.logger import Logger
from recis.utils.profiler.memory_access import MemoryAccessTracker
from recis.utils.profiler.memory_access_setup import setup_memory_access_tracking


class MemoryAccessHook(Hook):
    """Hook for tracking memory access during training steps.

    This hook enables memory access tracking for specific steps and prints
    statistics after each tracked step. It uses the MemoryAccessTracker's
    dynamic patch/unpatch mechanism to ensure zero overhead when not tracking.

    Args:
        profile_step: The step number to track (default: 13).
        print_summary: Whether to print summary after each tracked step (default: True).
        summary_detail: Whether to print detailed execution info (default: False).
            If True, prints each execution record; if False, prints aggregated stats only.

    Example:
        >>> from recis.hooks import MemoryAccessHook
        >>> # Track memory access at step 10 with detailed info
        >>> hook = MemoryAccessHook(profile_step=10, summary_detail=True)
        >>> trainer.register_hook(hook)
        >>> # Track step 3 with default aggregated summary
        >>> hook = MemoryAccessHook()
        >>> trainer.register_hook(hook)
    """

    def __init__(
        self,
        profile_step: int = 13,
        print_summary: bool = True,
        summary_detail: bool = False,
    ):
        # regist all memory access function
        setup_memory_access_tracking()
        self.logger = Logger("MemoryAccessHook")
        self.profile_step = profile_step
        self.print_summary = print_summary
        self.summary_detail = summary_detail

        self._step_count = 0
        self._is_tracking = False
        self._current_context = None

        self.logger.info(
            f"MemoryAccessHook initialized: profile_step={profile_step}, "
            f"summary_detail={summary_detail}"
        )

    def _should_track(self, is_train: bool) -> bool:
        """Check if we should track the current step."""
        # Check step range
        if self._step_count == self.profile_step:
            return True

        return False

    def before_step(self, is_train: bool = True, *args, **kwargs):
        """Called before each training step.

        Starts memory access tracking if the current step is within the tracking range.
        """
        if not self._should_track(is_train):
            return

        self._is_tracking = True
        self._current_context = MemoryAccessTracker.start_context(
            f"step_{self._step_count}"
        )
        self.logger.info(f"Started memory access tracking for step {self._step_count}")

    def after_step(self, is_train: bool = True, *args, **kwargs):
        """Called after each training step.

        Stops memory access tracking and prints summary if tracking was enabled.
        """
        # Always increment step count
        self._step_count += 1

        if not self._is_tracking:
            return

        # End tracking and get context
        ctx = MemoryAccessTracker.end_context()
        self._is_tracking = False

        if ctx is None:
            return

        # Print summary
        if self.print_summary:
            self._print_summary(ctx)

        self._current_context = None

    def _print_summary(self, ctx):
        """Print memory access summary in a formatted way."""
        summary = ctx.summary()

        self.logger.info("=" * 60)
        self.logger.info(f"Memory Access Summary: {summary['name']}")
        self.logger.info("=" * 60)
        self.logger.info(
            f"Total: {summary['total_bytes']:,} bytes "
            f"({summary['total_bytes'] / (1024**3):.3f} GB)"
        )
        self.logger.info(f"Time: {summary['total_time_ms']:.3f} ms")
        self.logger.info(f"Average Bandwidth: {summary['avg_bandwidth_gbps']:.2f} GB/s")
        self.logger.info("-" * 60)

        if summary["ops"]:
            self.logger.info("Per-operation statistics:")
            for op_name, stats in summary["ops"].items():
                self.logger.info(
                    f"  {op_name}:\n"
                    f"    count: {stats['count']}\n"
                    f"    bytes: {stats['bytes']:,} ({stats['bytes'] / (1024**3):.3f} GB)\n"
                    f"    time: {stats['time_ms']:.3f} ms\n"
                    f"    bandwidth: {stats['bandwidth_gbps']:.2f} GB/s"
                )
        else:
            self.logger.info("No operations tracked in this step.")

        # Print detailed execution info if summary_detail is True
        if self.summary_detail and summary["execution_details"]:
            self.logger.info("-" * 60)
            self.logger.info("Execution details:")
            for detail in summary["execution_details"]:
                self.logger.info(
                    f"  [{detail['index']}] {detail['op_name']}:\n"
                    f"      bytes: {detail['bytes']:,} ({detail['bytes'] / (1024**3):.6f} GB)\n"
                    f"      time: {detail['time_ms']:.3f} ms"
                )

        self.logger.info("=" * 60)

    def end(self, is_train: bool = True, *args, **kwargs):
        """Called at the end of training.

        Ensures tracking is stopped and unpatches all operators.
        """
        if self._is_tracking:
            MemoryAccessTracker.end_context()
            self._is_tracking = False
            self._current_context = None

        self.logger.info(
            f"MemoryAccessHook finished. Total steps processed: {self._step_count}"
        )
