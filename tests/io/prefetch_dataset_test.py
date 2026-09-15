import queue
import threading
import weakref

import pytest

from recis.io import prefetch_dataset


class _Source:
    def __init__(self, values):
        self.values = iter(values)
        self.close_count = 0
        self.close_thread = None

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)

    def close(self):
        self.close_count += 1
        self.close_thread = threading.current_thread()
        close = getattr(self.values, "close", None)
        if callable(close):
            close()
        self.values = None


def test_close_before_start_is_idempotent():
    source = _Source(range(3))
    iterator = iter(prefetch_dataset.PrefetchDataset(source))
    iterator.close()
    iterator.close()
    assert source.close_count == 1
    assert not iterator._runner.is_alive()
    assert iterator._input_iterator is None
    with pytest.raises(StopIteration):
        next(iterator)


def test_close_stops_worker_with_full_queue():
    full = threading.Event()
    closed = threading.Event()

    def produce():
        try:
            yield 0
            yield 1
            full.set()
            yield 2
        finally:
            closed.set()

    source = _Source(produce())
    iterator = iter(prefetch_dataset.PrefetchDataset(source, buffer_size=1))
    try:
        assert next(iterator) == 0
        assert full.wait(5)
    finally:
        iterator.close()
    assert not iterator._runner.is_alive()
    assert closed.is_set()
    assert source.close_thread is iterator._runner
    assert iterator._queue.empty()
    iterator.close()
    assert source.close_count == 1


@pytest.mark.parametrize("values", [[], [None], [1, 2, 3]])
def test_exhaustion_closes_input(values):
    source = _Source(values)
    iterator = iter(prefetch_dataset.PrefetchDataset(source))
    try:
        assert list(iterator) == values
        assert iterator._closed
        assert not iterator._runner.is_alive()
        assert source.close_count == 1
    finally:
        iterator.close()


def test_input_error_reaches_consumer_and_closes_worker():
    def produce():
        yield 7
        raise ValueError("input failed")

    iterator = iter(prefetch_dataset.PrefetchDataset(_Source(produce())))
    try:
        assert next(iterator) == 7
        with pytest.raises(RuntimeError, match="prefetch worker failed") as caught:
            next(iterator)
        assert isinstance(caught.value.__cause__, ValueError)
        assert iterator._closed
        assert not iterator._runner.is_alive()
    finally:
        iterator.close()


def test_close_timeout_retains_input_and_can_be_retried(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def produce():
        yield 0
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test input was not released")
        yield 1

    source = _Source(produce())
    iterator = iter(prefetch_dataset.PrefetchDataset(source))
    monkeypatch.setattr(prefetch_dataset, "_CLOSE_TIMEOUT", 0.01)
    try:
        assert next(iterator) == 0
        assert entered.wait(5)
        with pytest.raises(TimeoutError, match="still running"):
            iterator.close()
        assert iterator._runner.is_alive()
        assert not iterator._closed
        assert iterator._input_iterator is source
        assert source.close_count == 0
    finally:
        release.set()
        iterator._runner.join(timeout=5)
        iterator.close()
    assert source.close_count == 1
    assert source.close_thread is iterator._runner
    assert iterator._closed


def test_repeated_early_close_releases_queued_objects():
    class Payload:
        pass

    for _ in range(20):
        refs = []

        def produce():
            while True:
                value = Payload()
                refs.append(weakref.ref(value))
                yield value

        iterator = iter(prefetch_dataset.PrefetchDataset(produce()))
        try:
            value = next(iterator)
            del value
        finally:
            iterator.close()
        assert not iterator._runner.is_alive()
        assert iterator._dataset is None
        assert iterator._input_iterator is None
        assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("result", ["data", "end", "error", "missing"])
def test_worker_exits_between_queue_timeout_and_liveness_check(result, monkeypatch):
    release = threading.Event()

    def produce():
        if not release.wait(5):
            raise TimeoutError("producer was not released")
        if result == "data":
            yield 7
        elif result == "error":
            raise ValueError("input failed")

    iterator = iter(prefetch_dataset.PrefetchDataset(produce(), buffer_size=2))
    if result == "missing":
        monkeypatch.setattr(iterator._runner, "_put", lambda item: None)
    original_get = iterator._queue.get
    rechecked = False

    def get_with_exit_after_timeout(*args, **kwargs):
        nonlocal rechecked
        try:
            return original_get(*args, **kwargs)
        except queue.Empty:
            if not rechecked:
                rechecked = True
                release.set()
                iterator._runner.join(timeout=5)
                assert not iterator._runner.is_alive()
                expected = {"data": 2, "end": 1, "error": 1, "missing": 0}
                assert iterator._queue.qsize() == expected[result]
            raise

    monkeypatch.setattr(iterator._queue, "get", get_with_exit_after_timeout)
    monkeypatch.setattr(prefetch_dataset, "_QUEUE_TIMEOUT", 0.01)
    try:
        if result == "error":
            with pytest.raises(RuntimeError, match="prefetch worker failed") as caught:
                next(iterator)
            assert isinstance(caught.value.__cause__, ValueError)
        elif result == "missing":
            with pytest.raises(RuntimeError, match="exited without a result"):
                next(iterator)
        else:
            if result == "data":
                assert next(iterator) == 7
            with pytest.raises(StopIteration):
                next(iterator)
        assert rechecked
        assert iterator._closed
    finally:
        release.set()
        iterator.close()
