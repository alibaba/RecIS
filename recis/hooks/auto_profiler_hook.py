import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from recis.hooks.profiler_hook import (
    _TRACE_SAVE_TIMEOUT,
    ProfilerHook,
    run_with_timeout,
)
from recis.utils.logger import Logger


# Collected by the platform loguploader.
_AUTO_PROFILER_OUTPUT_DIR = "/var/log/recis_auto_profiler_traces"
_auto_profiler_logger = Logger("AutoProfiler")


@dataclass
class AutoProfilerArguments:
    """Configuration for the profiler automatically registered by Trainer.

    Attributes:
        enabled (bool): Whether Trainer registers the auto profiler. The
            ``RECIS_PROFILER_ON`` environment variable is checked first and
            wins when set to anything other than ``"1"``.
        ranks (Optional[List[int]]): Global ranks to profile. ``None`` means
            profile every rank. Defaults to ``[0]``. Ranks outside
            ``[0, world_size)`` are dropped with a warning logged on rank 0.
        wait (int): ``torch.profiler.schedule`` wait steps. Defaults to 100.
        warmup (int): ``torch.profiler.schedule`` warmup steps. Defaults to 1.
        active (int): ``torch.profiler.schedule`` active steps. Defaults to 1.
        repeat (int): ``torch.profiler.schedule`` cycles. Defaults to 1.
    """

    enabled: bool = True
    ranks: Optional[List[int]] = field(default_factory=lambda: [0])
    wait: int = 100
    warmup: int = 1
    active: int = 1
    repeat: int = 1

    def __post_init__(self):
        if self.wait <= 0:
            raise ValueError("AutoProfilerArguments.wait must be greater than 0")
        if self.warmup < 0:
            raise ValueError("AutoProfilerArguments.warmup must not be negative")
        if self.active < 1:
            raise ValueError("AutoProfilerArguments.active must be at least 1")
        if self.repeat < 1:
            raise ValueError("AutoProfilerArguments.repeat must be at least 1")
        if self.ranks is not None:
            invalid_types = [
                rank
                for rank in self.ranks
                if isinstance(rank, bool) or not isinstance(rank, int)
            ]
            if invalid_types:
                raise ValueError(
                    "AutoProfilerArguments.ranks must contain only integers"
                )


class _LocalTimelineProfilerHook(ProfilerHook):
    """Auto profiler writing gzip timelines into the task log directory.

    The platform log collector uploads this directory after the trace is
    complete. RecIS deliberately performs no remote filesystem or MOS I/O.
    The output directory is fixed to ``/var/log/recis_auto_profiler_traces``.
    """

    def __init__(
        self,
        rank,
        wait=100,
        warmup=1,
        active=1,
        repeat=1,
    ):
        self._rank = rank
        self._app_id = os.environ.get("APP_ID", "local")
        self._run_datetime = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        super().__init__(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            output_dir=_AUTO_PROFILER_OUTPUT_DIR,
        )

    def _build_file_name(self, step_num):
        return (
            f"{self._app_id}-{self._rank:03d}-timeline-{step_num}-"
            f"{self._run_datetime}.json.gz"
        )

    def get_trace_handler(self):
        def local_timeline_handler(prof):
            local_save_file = os.path.join(
                self.output_dir, self._build_file_name(prof.step_num)
            )

            def _save():
                os.makedirs(self.output_dir, exist_ok=True)
                prof.export_chrome_trace(local_save_file)
                self.logger.info(f"Save profiler result : {local_save_file}")

            run_with_timeout(
                _save,
                _TRACE_SAVE_TIMEOUT,
                f"Save profiler result {local_save_file}",
                self.logger,
            )

        return local_timeline_handler


def _resolve_auto_profiler_ranks(ranks, world_size, rank):
    # Every rank resolves this independently; warn only on rank 0.
    if ranks is None:
        effective_ranks = list(range(world_size))
    else:
        invalid_ranks = sorted({r for r in ranks if r < 0 or r >= world_size})
        if invalid_ranks and rank == 0:
            _auto_profiler_logger.warning(
                f"Ignore invalid auto profiler global ranks {invalid_ranks}; "
                f"world_size={world_size}"
            )
        effective_ranks = sorted({r for r in ranks if 0 <= r < world_size})

    if len(effective_ranks) > 8 and rank == 0:
        _auto_profiler_logger.warning(
            f"Auto profiler is enabled on {len(effective_ranks)} ranks; "
            "profiling and storage overhead grows with rank count"
        )
    return effective_ranks


def _build_auto_profiler(args, rank, world_size):
    """Build the profiler Trainer registers when the user configured none.

    Returns ``None`` when ``RECIS_PROFILER_ON`` is set to anything other than
    ``"1"``, which is how a task opts out of automatic collection. Otherwise
    returns an independent profiler for the configured global rank.
    """
    if os.environ.get("RECIS_PROFILER_ON", "1") != "1":
        return None
    if not args.enabled:
        return None

    effective_ranks = _resolve_auto_profiler_ranks(args.ranks, world_size, rank)
    if rank not in effective_ranks:
        return None

    profiler_kwargs = {
        "wait": args.wait,
        "warmup": args.warmup,
        "active": args.active,
        "repeat": args.repeat,
    }
    return _LocalTimelineProfilerHook(rank=rank, **profiler_kwargs)
