import queue
import threading
from typing import Iterator

from torch.utils.data import IterableDataset

from recis.utils.logger import Logger


logger = Logger(__name__)
_QUEUE_TIMEOUT = 0.1
_CLOSE_TIMEOUT = 5.0
_END = object()


class _StopFlag:
    def __init__(self):
        self._event = threading.Event()

    def stop(self):
        return self._event.is_set()

    def set_stop(self, flag: bool):
        if flag:
            self._event.set()
        else:
            self._event.clear()


class _WorkerError:
    def __init__(self, exc):
        self.exc = exc


class _DataPrefetcher(threading.Thread):
    def __init__(self, queue: queue.Queue, iterator: Iterator, stop_flag: _StopFlag):
        super().__init__(daemon=True)
        self._queue = queue
        self._iterator = iterator
        self._stop_flag = stop_flag

    def _put(self, item):
        while not self._stop_flag.stop():
            try:
                self._queue.put(item, timeout=_QUEUE_TIMEOUT)
                return
            except queue.Full:
                continue

    def run(self):
        terminal = _END
        try:
            while not self._stop_flag.stop():
                try:
                    input_data = next(self._iterator)
                except StopIteration:
                    break
                self._put(input_data)
                del input_data
        except BaseException as exc:  # Worker boundary: relay to the consumer.
            terminal = _WorkerError(exc)
        finally:
            # Only the producer closes a started input: a generator may still
            # be executing next() when the consumer requests shutdown.
            try:
                close = getattr(self._iterator, "close", None)
                if callable(close):
                    close()
                self._iterator = None
            except BaseException as exc:
                if terminal is _END:
                    terminal = _WorkerError(exc)
                else:
                    logger.error(  # noqa: G201 - Logger has no exception() method.
                        "Input cleanup failed", exc_info=True
                    )
        if self._stop_flag.stop():
            if isinstance(terminal, _WorkerError):
                logger.error("Prefetch worker failed during shutdown: %r", terminal.exc)
        else:
            self._put(terminal)


class _PrefetchIterator:
    """Single-consumer iterator with explicit, retryable shutdown.

    A blocking input next() cannot be interrupted. close() then raises
    TimeoutError and retains ownership; retry close() after the input returns.
    Calls to next() and close() must be serialized by the consumer.
    """

    def __init__(self, dataset, input_iterator: Iterator, buffer_size=1):
        self._dataset = dataset
        self._input_iterator = input_iterator
        self._queue = queue.Queue(maxsize=buffer_size)
        self._stop_flag = _StopFlag()
        self._runner = _DataPrefetcher(
            self._queue, self._input_iterator, self._stop_flag
        )
        self._started = False
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop_flag.stop():
            raise StopIteration
        if not self._started:
            self._runner.start()
            self._started = True
        while not self._stop_flag.stop():
            try:
                ret = self._queue.get(timeout=_QUEUE_TIMEOUT)
            except queue.Empty:
                if self._runner.is_alive():
                    continue
                # The producer may have queued its final results between
                # the timeout and exiting. Recheck after it can no longer put.
                try:
                    ret = self._queue.get_nowait()
                except queue.Empty:
                    self.close()
                    raise RuntimeError(
                        "prefetch worker exited without a result"
                    ) from None
            self._queue.task_done()
            if ret is _END:
                self.close()
                raise StopIteration
            if isinstance(ret, _WorkerError):
                try:
                    raise RuntimeError("prefetch worker failed") from ret.exc
                finally:
                    try:
                        self.close()
                    except Exception:
                        logger.error(  # noqa: G201 - Logger has no exception() method.
                            "Prefetch cleanup failed", exc_info=True
                        )
            return ret
        raise StopIteration

    def close(self):
        """Stop production and release queued data once the worker exits."""
        if self._closed:
            return
        self._stop_flag.set_stop(True)
        if self._started:
            if self._runner is threading.current_thread():
                raise RuntimeError("prefetch worker cannot close its own iterator")
            self._runner.join(timeout=_CLOSE_TIMEOUT)
            if self._runner.is_alive():
                raise TimeoutError("prefetch worker is still running; retry close()")
        # A failed producer-side close retains the input for a safe retry
        # here, after join confirms that no next() is still executing.
        close = getattr(self._runner._iterator, "close", None)
        if callable(close):
            close()
        self._runner._iterator = None
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._input_iterator = None
        self._dataset = None
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Explicit close reports failures; destruction is only a fallback.
            logger.warning("Prefetch iterator cleanup was incomplete", exc_info=True)


class PrefetchDataset(IterableDataset):
    def __init__(self, input_dataset: IterableDataset, buffer_size=1):
        self._input_dataset = input_dataset
        self._buffer_size = buffer_size

    def __iter__(self) -> Iterator:
        return _PrefetchIterator(self, iter(self._input_dataset), self._buffer_size)
