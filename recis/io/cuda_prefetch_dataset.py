import queue
import threading
from collections import deque
from enum import IntEnum
from typing import Callable, Iterator

import torch
from torch.utils.data import IterableDataset

from recis.utils.logger import Logger


logger = Logger(__name__)

# How long the consumer waits for a batch before releasing the gate itself
# (see ``_get_output``) so a missing notify can never deadlock training.
_GATE_FALLBACK_TIMEOUT = 5.0


class StreamPriority(IntEnum):
    """CUDA stream priority levels for the prefetch side stream.

    Lower integer value = higher GPU scheduling priority.
    The default stream has priority 0; negative values preempt it.

    Attributes:
        HIGH (-2): Highest priority. Side stream preempts all others.
        DEFAULT (-1): Higher than the default stream.
        LOW (0): Same priority as the default stream, no preemption.
    """

    HIGH = -2
    DEFAULT = -1
    LOW = 0


class _PrefetchWorkerError:
    """Sentinel carrying a worker-side exception to the consumer thread."""

    def __init__(self, exc: BaseException):
        self.exc = exc


# Generous bound: real batches nest ~6 levels deep (batch tuple -> dict ->
# precomputed tuple -> per-hashtable dict -> per-group dict -> group object).
# Exceeding it would silently skip protection, so keep ample margin.
_MAX_COLLECT_DEPTH = 16


def _collect_holdables(obj, out, _depth: int = 0):
    """Collect the tensors reachable from a batch into ``out``.

    Holding the batch container is not sufficient for cross-stream safety:
    consumers drop inner references mid-step (``RecISModel.forward`` pops
    ``_precomputed_group_features``, ``RuntimeGroupFeature.clear_ids`` drops
    coalesced ids). Once the last Python reference dies, the block returns to
    the *producing* stream's allocator pool and the next transform may reuse
    and overwrite it while the consumer stream still has kernels reading it.
    Holding the tensor objects themselves keeps those blocks reserved until
    the release event fires, with no per-tensor CUDA calls (unlike
    ``record_stream``, which also defers allocator reuse).
    """
    if _depth > _MAX_COLLECT_DEPTH:
        return
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_holdables(value, out, _depth + 1)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_holdables(value, out, _depth + 1)
    elif hasattr(obj, "holdable_tensors"):
        for tensor in obj.holdable_tensors():
            if torch.is_tensor(tensor):
                out.append(tensor)
    elif hasattr(obj, "values") and hasattr(obj, "offsets"):  # RaggedTensor
        _collect_holdables(obj.values(), out, _depth + 1)
        _collect_holdables(obj.offsets(), out, _depth + 1)
        if hasattr(obj, "weight"):
            _collect_holdables(obj.weight(), out, _depth + 1)


def _holdables_of(obj):
    holdables = []
    _collect_holdables(obj, holdables)
    return holdables


def _drain_inflight(inflight):
    """Release batches whose guarding event has completed.

    Called both from ``notify`` (mid-step) and ``__next__`` so held tensors
    are returned to the allocator as early as possible: draining only once
    per step would keep an extra batch alive and raise peak memory.
    """
    while inflight and inflight[0][1].query():
        inflight.popleft()


class _CudaPrefetchIterator:
    """Run ``transform_fn`` on a side CUDA stream in a background thread.

    Raw batches are fetched either on the **consumer (main) thread** inside
    ``__next__`` — right before the trainer starts forward, the default — or
    in the worker thread when ``fetch_in_thread`` is set. Main-thread fetch
    keeps the Python-heavy IO/convert path from contending for the GIL with
    the forward pass; worker-thread fetch moves it off the main thread
    entirely, where it is gated to run after ``notify`` so it competes with
    backward rather than forward.

    Cross-stream memory safety uses one CUDA event per batch plus holding
    the batch's tensors (see ``_collect_holdables``), instead of per-tensor
    ``record_stream``:

    - a raw batch's tensors stay referenced by the worker until the side
      stream has finished reading them (its transform ``done_event``);
    - a transformed batch's tensors stay referenced here until the consumer
      stream has finished the step that used it (release event).

    Work the producer enqueues on the caller's current stream (H2D copies,
    dtype casts) is ordered via ``ready_event``; work done on other streams
    must already be materialised at hand-off, which the dataset pack stage
    guarantees.
    """

    def __init__(
        self,
        input_iterator: Iterator,
        transform_fn: Callable,
        buffer_size: int,
        lazy_start: bool = False,
        stream_priority: int = StreamPriority.LOW,
        fetch_in_thread: bool = False,
    ):
        self._input_iterator = input_iterator
        self._transform_fn = transform_fn
        self._out_queue = queue.Queue(maxsize=buffer_size)
        # Unbounded is safe: pushes are paced 1:1 by ``__next__`` (the
        # bootstrap adds one extra), so at most two batches are staged.
        # Unused when fetching in the worker thread.
        self._raw_queue = queue.Queue()
        self._fetch_in_thread = fetch_in_thread
        self._device = torch.cuda.current_device()
        self._stream = torch.cuda.Stream(self._device, priority=int(stream_priority))
        self._stop = False
        self._input_exhausted = False
        self._bootstrapped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self._lazy_start = lazy_start
        self._go_event = threading.Event()
        # (tensors, event) pairs kept alive until the GPU is done with them.
        self._raw_inflight = deque()  # accessed by worker thread only
        self._out_inflight = deque()  # accessed by consumer thread only
        self._prev_holdables = None

    def _fetch_raw(self):
        """Fetch one raw batch on the caller thread and stage it for the worker.

        Records an event on the caller's current stream so the side stream
        can order after any GPU work enqueued while producing the batch.
        """
        if self._input_exhausted:
            return
        try:
            data = next(self._input_iterator)
        except StopIteration:
            self._input_exhausted = True
            self._raw_queue.put(None)
            return
        ready_event = torch.cuda.current_stream().record_event()
        self._raw_queue.put((data, ready_event))

    def _put_output(self, item) -> bool:
        while not self._stop:
            try:
                self._out_queue.put(item, timeout=1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self):
        torch.cuda.set_device(self._device)
        try:
            self._run_loop()
            self._put_output(None)
        except Exception as exc:  # propagate instead of hanging the consumer
            self._put_output(_PrefetchWorkerError(exc))

    def _run_loop(self):
        first = True
        while not self._stop:
            if self._fetch_in_thread:
                # Gate before fetching so the fetch's Python work also lands
                # in the post-notify window (overlapping backward) instead of
                # competing with forward for the GIL.
                if self._lazy_start and not first:
                    self._go_event.wait()
                    self._go_event.clear()
                    if self._stop:
                        return
                first = False
                try:
                    data = next(self._input_iterator)
                except StopIteration:
                    return
                ready_event = torch.cuda.current_stream().record_event()
            else:
                try:
                    item = self._raw_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:  # input exhausted
                    return
                data, ready_event = item
                if self._lazy_start and not first:
                    self._go_event.wait()
                    self._go_event.clear()
                    if self._stop:
                        return
                first = False
            # Collect before transforming: the transform pops entries from
            # the raw batch (labels, sample ids), dropping references while
            # its own kernels may still be reading them.
            raw_holdables = _holdables_of(data)
            with torch.cuda.stream(self._stream):
                self._stream.wait_event(ready_event)
                transformed = self._transform_fn(data)
            done_event = self._stream.record_event()
            # Keep the raw tensors referenced until the side stream is done
            # reading them; only then may the allocator recycle their blocks.
            self._raw_inflight.append((raw_holdables, done_event))
            del data, raw_holdables
            _drain_inflight(self._raw_inflight)
            if not self._put_output(
                (transformed, done_event, _holdables_of(transformed))
            ):
                return

    def notify(self):
        self._go_event.set()
        # Also drain here: freeing the previous batch mid-step instead of at
        # the next __next__ keeps one less batch of tensors alive.
        _drain_inflight(self._out_inflight)

    def _get_output(self):
        """Wait for the next transformed batch, self-releasing the gate.

        The consumer only blocks here when it actually needs the batch, so if
        the notify for it never arrives (e.g. the configured notify position
        does not fire for this model), releasing the gate here is both safe
        and required for progress: otherwise the worker would wait for a
        notify that never comes while the consumer waits for the worker.
        """
        while True:
            try:
                return self._out_queue.get(timeout=_GATE_FALLBACK_TIMEOUT)
            except queue.Empty:
                if not self._thread.is_alive():
                    raise RuntimeError(
                        "cuda prefetch worker thread died without reporting "
                        "an error; cannot continue"
                    ) from None
                if not self._lazy_start:
                    continue
                logger.warning(
                    "Prefetch transform was not released within "
                    f"{_GATE_FALLBACK_TIMEOUT}s; releasing it inline. "
                    "Check that the configured notify position actually "
                    "fires (a forward hook does not run when a module is "
                    "invoked as module.forward(...) instead of "
                    "module(...)). Overlap is degraded until then."
                )
                self._go_event.set()

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop:
            raise StopIteration
        if not self._started:
            self._thread.start()
            self._started = True
        if not self._bootstrapped:
            # Stage two raw batches up-front: one for the ungated first
            # transform, one pending so the post-forward notify() already
            # has work to release. The worker stages its own when fetching
            # in the thread.
            if not self._fetch_in_thread:
                self._fetch_raw()
            self._bootstrapped = True
        if not self._fetch_in_thread:
            self._fetch_raw()
        item = self._get_output()
        if item is None:
            self._stop = True
            raise StopIteration
        if isinstance(item, _PrefetchWorkerError):
            self._stop = True
            raise RuntimeError("cuda prefetch worker failed") from item.exc
        transformed, done_event, holdables = item
        stream = torch.cuda.current_stream()
        stream.wait_event(done_event)
        # Hold the previous batch's tensors until the consumer stream
        # finishes the step that used them: one event per batch replaces the
        # per-tensor record_stream calls and their deferred-reuse cost.
        if self._prev_holdables is not None:
            self._out_inflight.append((self._prev_holdables, stream.record_event()))
        _drain_inflight(self._out_inflight)
        self._prev_holdables = holdables
        return transformed

    def __del__(self):
        self._stop = True
        self._go_event.set()
        if self._started:
            self._thread.join(timeout=5)


class _InlinePrefetchIterator:
    """Run ``transform_fn`` on a side CUDA stream from the consumer thread.

    No worker thread is created: ``notify()`` (called at the configured
    position of the train step) enqueues the next batch's transform on the
    side stream and returns once kernels are issued; overlap comes from CUDA
    async execution. This avoids all GIL contention with the train loop at
    the cost of running the transform's Python part serially on it.

    Cross-stream memory safety follows the same one-event-per-batch plus
    tensor-holding scheme as ``_CudaPrefetchIterator``.
    """

    def __init__(
        self,
        input_iterator: Iterator,
        transform_fn: Callable,
        buffer_size: int = 1,
        lazy_start: bool = False,
        stream_priority: int = StreamPriority.LOW,
    ):
        # buffer_size / lazy_start are accepted for signature compatibility;
        # inline mode is notify-driven with a fixed depth of one batch.
        del buffer_size, lazy_start
        self._input_iterator = input_iterator
        self._transform_fn = transform_fn
        self._device = torch.cuda.current_device()
        self._stream = torch.cuda.Stream(self._device, priority=int(stream_priority))
        self._stop = False
        self._input_exhausted = False
        self._bootstrapped = False
        self._staged_raw = None  # (data, ready_event) awaiting transform
        self._pending = None  # (transformed, done_event, holdables) to consume
        self._raw_inflight = deque()
        self._out_inflight = deque()
        self._prev_holdables = None

    def _fetch_raw(self):
        if self._input_exhausted:
            return
        try:
            data = next(self._input_iterator)
        except StopIteration:
            self._input_exhausted = True
            return
        ready_event = torch.cuda.current_stream().record_event()
        self._staged_raw = (data, ready_event)

    def _transform_staged(self):
        if self._staged_raw is None:
            return
        data, ready_event = self._staged_raw
        self._staged_raw = None
        # Collect before transforming: the transform pops entries from the
        # raw batch while its own kernels may still be reading them.
        raw_holdables = _holdables_of(data)
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(ready_event)
            transformed = self._transform_fn(data)
        done_event = self._stream.record_event()
        self._raw_inflight.append((raw_holdables, done_event))
        del data, raw_holdables
        _drain_inflight(self._raw_inflight)
        self._pending = (transformed, done_event, _holdables_of(transformed))

    def notify(self):
        """Launch the next-batch transform so its GPU work overlaps the
        remaining phases of the current step."""
        if self._pending is None:
            self._transform_staged()
        # Free the previous batch mid-step rather than at the next __next__.
        _drain_inflight(self._out_inflight)

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop:
            raise StopIteration
        if not self._bootstrapped:
            self._fetch_raw()
            self._bootstrapped = True
        if self._pending is None:
            # First batch, or notify() was not called (e.g. eval loop).
            self._transform_staged()
        if self._pending is None:
            self._stop = True
            raise StopIteration
        transformed, done_event, holdables = self._pending
        self._pending = None
        # Stage the next raw batch for the coming notify().
        self._fetch_raw()
        stream = torch.cuda.current_stream()
        stream.wait_event(done_event)
        if self._prev_holdables is not None:
            self._out_inflight.append((self._prev_holdables, stream.record_event()))
        _drain_inflight(self._out_inflight)
        self._prev_holdables = holdables
        return transformed


class CudaPrefetchDataset(IterableDataset):
    """Prefetch dataset that executes a transform on a separate CUDA stream.

    The transform runs on an independent CUDA stream, either in a background
    thread (``enable_thread=True``) or inline on the consuming thread,
    allowing overlap with computation on the default stream. Raw batches are
    fetched on the consuming thread to avoid GIL contention with training.
    """

    def __init__(
        self,
        input_dataset: IterableDataset,
        transform_fn: Callable,
        buffer_size: int = 1,
        lazy_start: bool = False,
        stream_priority: int = StreamPriority.LOW,
        enable_thread: bool = True,
        fetch_in_thread: bool = False,
    ):
        self._input_dataset = input_dataset
        self._transform_fn = transform_fn
        self._buffer_size = buffer_size
        self._lazy_start = lazy_start
        self._stream_priority = stream_priority
        self._enable_thread = enable_thread
        self._fetch_in_thread = fetch_in_thread

    def __iter__(self) -> Iterator:
        return cuda_prefetch_iterator(
            iter(self._input_dataset),
            self._transform_fn,
            self._buffer_size,
            lazy_start=self._lazy_start,
            stream_priority=self._stream_priority,
            enable_thread=self._enable_thread,
            fetch_in_thread=self._fetch_in_thread,
        )


def cuda_prefetch_iterator(
    input_iterator: Iterator,
    transform_fn: Callable,
    buffer_size: int = 1,
    lazy_start: bool = False,
    stream_priority: int = StreamPriority.LOW,
    enable_thread: bool = True,
    fetch_in_thread: bool = False,
) -> Iterator:
    """Wrap an existing iterator with CUDA stream prefetch and transform.

    Args:
        enable_thread (bool): When True (default), the transform runs in a
            background worker thread. When False, it runs inline on the
            consuming thread at ``notify()`` time (no GIL contention, at the
            cost of serialising the transform's Python part).
        fetch_in_thread (bool): When True, the worker thread also pulls raw
            batches from ``input_iterator``; otherwise the consuming thread
            fetches them in ``__next__``. Requires ``enable_thread``.

    Raises:
        ValueError: If ``fetch_in_thread`` is set without ``enable_thread``.
    """
    if fetch_in_thread and not enable_thread:
        raise ValueError(
            "fetch_in_thread requires enable_thread: inline mode has no "
            "worker thread to fetch data on."
        )
    if not enable_thread:
        return _InlinePrefetchIterator(
            input_iterator,
            transform_fn,
            buffer_size,
            lazy_start=lazy_start,
            stream_priority=stream_priority,
        )
    return _CudaPrefetchIterator(
        input_iterator,
        transform_fn,
        buffer_size,
        lazy_start=lazy_start,
        stream_priority=stream_priority,
        fetch_in_thread=fetch_in_thread,
    )
