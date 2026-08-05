"""NVTX profiling hook for ncu profiler integration.

This hook automatically wraps all torch.ops.recis calls with NVTX range
markers on a specific training step, enabling precise ncu profiling with
minimal overhead.

NVTX ranges and torch.cuda.profiler.start/stop are controlled via the
before_step/after_step callbacks, tightly wrapping the training step.

Usage:
    from recis.hooks import NvtxProfileHook

    nvtx_hook = NvtxProfileHook(profile_step=23)
    trainer.add_hooks([nvtx_hook])

    # Then run with ncu:
    # ncu --nvtx --nvtx-include "op:xxx/" --profile-from-start off \
    #     --metrics dram__bytes.sum,gpu__time_duration.sum python train.py
"""

import os
import sys

import torch

from recis.hooks.hook import Hook
from recis.utils.logger import Logger


class NvtxProfileHook(Hook):
    """Hook for NVTX profiling of recis ops with ncu.

    Uses before_step/after_step callbacks to enable NVTX range markers
    for all torch.ops.recis calls on a specific training step. Uses
    torch.cuda.profiler.start/stop to control ncu profiling when used
    with --profile-from-start off.

    Args:
        profile_step: The 1-indexed training step to profile. Defaults to 23.

    .. warning::
        Must not monitor the same step as ``MemoryAccessHook``. Both
        hooks patch ops via the ``nvtx_wrapper._targets`` mutable
        reference mechanism: MemoryAccessHook redirects targets to
        MemoryAccessTracker, inserting memory-access tracking logic
        before and after each op. This extra overhead distorts ncu
        timing and bandwidth measurements. Additionally,
        MemoryAccessTracker's patch/unpatch interleaving with NVTX
        range push/pop may cause kernel attribution errors. Ensure the
        two hooks use different ``profile_step`` values (defaults:
        NVTX=23, MemoryAccess=13).

    Example:
        >>> from recis.hooks import NvtxProfileHook
        >>> hook = NvtxProfileHook(profile_step=23)
        >>> trainer.add_hooks([hook])
    """

    def __init__(self, profile_step: int = 23):
        # Import nvtx_wrapper to trigger the module-level patching:
        #   - _OpNamespace.__getattr__ is patched to serve NVTX-wrapped
        #     recis ops (stored in _wrapped_ops).
        #   - torch.unique and HashTable._embedding_lookup_internal are
        #     directly replaced with NVTX-wrapped versions.
        #   - _rewrap_cached_ops() is called at import time to wrap
        #     any ops cached earlier (e.g. by _wrap_torch_ops).
        #
        # The NVTX wrappers use a mutable target reference (_targets)
        # that can be redirected by MemoryAccessTracker, enabling both
        # hooks to coexist.
        from recis.utils.profiler.nvtx_wrapper import _ProfileState, _rewrap_cached_ops

        # Re-run in case new ops were cached between import and now.
        _rewrap_cached_ops()

        self._profile_state = _ProfileState
        self.logger = Logger("NvtxProfileHook")
        self.profile_step = profile_step

        if self.profile_step <= 0:
            self.logger.warning(
                f"NvtxProfileHook: profile_step={self.profile_step} <= 0, "
                f"NVTX profiling disabled."
            )
            self.profile_step = None
            return

        self._step_counter = 0
        self._profiling_active = False
        self.logger.info(f"NVTX profiling: will profile step {self.profile_step}")

    def before_step(self, is_train=True, *args, **kwargs):
        """Enable NVTX ranges and start ncu capture at the profile step."""
        if not is_train or self.profile_step is None:
            return

        self._step_counter += 1
        if self._step_counter == self.profile_step:
            self._profiling_active = True
            self._profile_state.enabled = True
            torch.cuda.profiler.start()  # for --profile-from-start off
            self.logger.info(
                f"NVTX enabled at step {self._step_counter}/{self.profile_step}"
            )

    def after_step(self, is_train=True, *args, **kwargs):
        """Synchronize and stop ncu capture after the profile step."""
        if not is_train or not self._profiling_active:
            return

        torch.cuda.synchronize()
        self._profiling_active = False
        self._profile_state.enabled = False
        torch.cuda.profiler.stop()
        self.logger.info(
            f"NVTX disabled after step {self._step_counter}/{self.profile_step}"
        )

        # Exit training after profiling to avoid running to convergence.
        # ncu has already captured all data between start() and stop();
        # it will finalize and write the CSV when the child process exits.
        self.logger.info("Profile step completed, exiting training")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def end(self, is_train=True, *args, **kwargs):
        """Log summary and warn if profiling was never triggered."""
        if self.profile_step is not None and self.profile_step > self._step_counter:
            self.logger.warning(
                f"NvtxProfileHook: profile_step={self.profile_step} but only "
                f"{self._step_counter} training steps were executed. "
                f"NVTX profiling was never triggered!"
            )

        self.logger.info(
            f"NvtxProfileHook finished. Total training steps processed: "
            f"{self._step_counter}"
        )
