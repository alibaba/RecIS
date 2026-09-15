import queue
import threading
import weakref
from collections import UserDict
from dataclasses import dataclass

import pytest
import torch

from recis.io import cuda_prefetch_dataset as prefetch
from recis.ragged import tensor as ragged_tensor


@dataclass(frozen=True)
class _PreparedBatch:
    prepared: object
    labels: dict


class _PreparedEmbedding:
    def __init__(self, tensor):
        self.tensor = tensor

    def holdable_tensors(self):
        return [self.tensor]


class _LeaseMapping(UserDict):
    def __init__(self, tensor):
        super().__init__({"must_not_materialize": object()})
        self.tensor = tensor

    def values(self):
        raise AssertionError("physical lease protocol must precede Mapping")

    def holdable_tensors(self):
        return [self.tensor]


def test_collect_holdables_traverses_mapping_subclasses():
    value = torch.tensor([1])

    held = prefetch._holdables_of(UserDict({"feature": value}))

    assert len(held) == 1
    assert held[0] is value


def test_collect_holdables_traverses_dataclass_payloads():
    prepared_value = torch.tensor([1])
    label = torch.tensor([2])
    batch = _PreparedBatch(
        prepared=_PreparedEmbedding(prepared_value),
        labels={"label": label},
    )

    held = prefetch._holdables_of(batch)

    assert len(held) == 2
    assert held[0] is prepared_value
    assert held[1] is label


def test_collect_holdables_prefers_group_lease_protocol():
    value = torch.tensor([3])

    held = prefetch._holdables_of(_LeaseMapping(value))

    assert held == [value]


def test_collect_holdables_handles_cycles_and_dataclass_classes():
    value = torch.tensor([4])
    batch = [value]
    batch.append(batch)
    assert prefetch._holdables_of(batch) == [value]
    assert prefetch._holdables_of(_PreparedBatch) == []


def test_collect_holdables_rejects_excessive_nesting():
    batch = torch.tensor([1])
    for _ in range(prefetch._MAX_COLLECT_DEPTH + 1):
        batch = [batch]
    with pytest.raises(ValueError, match="nesting limit"):
        prefetch._holdables_of(batch)


def test_collect_holdables_keeps_protocol_tensor_filtering():
    value = torch.tensor([1])

    class Lease:
        def holdable_tensors(self):
            return [None, "metadata", value]

    assert prefetch._holdables_of(Lease()) == [value]


@pytest.mark.parametrize("result", [None, 3])
def test_collect_holdables_rejects_invalid_protocol_result(result):
    class Lease:
        def holdable_tensors(self):
            return result

    with pytest.raises(TypeError):
        prefetch._holdables_of(Lease())


def _make_iterator(source, mode, **kwargs):
    return prefetch.cuda_prefetch_iterator(
        source,
        kwargs.pop("transform", lambda value: value + 1),
        enable_thread=mode != "inline",
        fetch_in_thread=mode == "worker_fetch",
        **kwargs,
    )


def _assert_closed(iterator):
    assert iterator._closed
    assert iterator._input_iterator is None
    assert iterator._prev_holdables is None
    assert not iterator._raw_inflight
    assert not iterator._out_inflight
    if isinstance(iterator, prefetch._CudaPrefetchIterator):
        assert not iterator._thread.is_alive()
        assert iterator._raw_queue.empty()
        assert iterator._out_queue.empty()
    else:
        assert iterator._staged_raw is None
        assert iterator._pending is None


_GPU_MODES = ["inline", "thread", "worker_fetch"]
_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a real CUDA device"
)


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", _GPU_MODES)
def test_cuda_close_before_start(mode):
    iterator = _make_iterator(iter([]), mode)
    iterator.close()
    iterator.close()
    iterator.notify()
    with pytest.raises(StopIteration):
        next(iterator)
    _assert_closed(iterator)


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", _GPU_MODES)
@pytest.mark.parametrize("notify", [False, True])
def test_cuda_values_and_exhaustion(mode, notify, monkeypatch):
    monkeypatch.setattr(prefetch, "_GATE_FALLBACK_TIMEOUT", 0.05)
    closed = threading.Event()

    def produce():
        try:
            for value in range(4):
                yield torch.full((32,), value, device="cuda")
        finally:
            closed.set()

    iterator = _make_iterator(produce(), mode, lazy_start=True)
    try:
        for expected in range(1, 5):
            value = next(iterator)
            torch.testing.assert_close(value, torch.full_like(value, expected))
            if notify:
                iterator.notify()
        with pytest.raises(StopIteration):
            next(iterator)
        assert closed.is_set()
        _assert_closed(iterator)
    finally:
        iterator.close()


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", _GPU_MODES)
@pytest.mark.parametrize("failure", ["input", "transform", "notify"])
def test_cuda_errors_close_input_and_storage(mode, failure):
    closed = threading.Event()

    def produce():
        try:
            for index in range(4):
                if failure == "input" and index == 1:
                    raise ValueError("input failed")
                yield torch.full((8,), index, device="cuda")
        finally:
            closed.set()

    calls = 0

    def transform(value):
        nonlocal calls
        calls += 1
        result = value + 1
        if failure == "transform" or (failure == "notify" and calls == 2):
            raise ValueError("transform failed")
        return result

    iterator = _make_iterator(produce(), mode, transform=transform, lazy_start=True)
    try:
        with pytest.raises((ValueError, RuntimeError), match="failed"):
            if failure == "notify":
                next(iterator)
                iterator.notify()
            for _ in range(4):
                next(iterator)
        assert closed.is_set()
        _assert_closed(iterator)
    finally:
        iterator.close()


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", _GPU_MODES)
def test_cuda_close_waits_for_original_consumer_stream(mode):
    consumer = torch.cuda.Stream()
    unrelated = torch.cuda.Stream()
    refs = []

    def transform(value):
        result = torch.full((4096,), value, device="cuda")
        refs.append(weakref.ref(result))
        return {"value": result}

    iterator = _make_iterator(
        iter(range(4)), mode, transform=transform, lazy_start=True
    )
    try:
        with torch.cuda.stream(consumer):
            batch = next(iterator)
            tensor = batch.pop("value")
            # Delay a real consumer read so close must honor this stream.
            torch.cuda._sleep(100_000_000)
            result = tensor + 5
            finished = consumer.record_event()
            del tensor, batch
        assert iterator._prev_stream == consumer
        assert refs[0]() is not None
        with torch.cuda.stream(unrelated):
            iterator.close()
        assert finished.query()
        torch.testing.assert_close(result, torch.full_like(result, 5))
        _assert_closed(iterator)
        assert all(ref() is None for ref in refs)
    finally:
        iterator.close()


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", _GPU_MODES)
def test_cuda_next_retires_previous_on_its_own_stream(mode):
    first = torch.cuda.Stream()
    second = torch.cuda.Stream()
    iterator = _make_iterator(iter(range(4)), mode, lazy_start=True)
    try:
        with torch.cuda.stream(first):
            assert next(iterator) == 1
            torch.cuda._sleep(100_000_000)
            finished = first.record_event()
        with torch.cuda.stream(second):
            iterator.notify()
            assert next(iterator) == 2
            # Even an empty lease needs the previous stream's event.
            if not finished.query():
                assert iterator._out_inflight
                assert not iterator._out_inflight[0][1].query()
        iterator.close()
        assert finished.query()
    finally:
        iterator.close()


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", ["thread", "worker_fetch"])
def test_cuda_close_timeout_is_retryable(mode, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    refs = []
    calls = 0

    def transform(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test transform was not released")
        tensor = torch.full((8,), value, device="cuda")
        refs.append(weakref.ref(tensor))
        return tensor

    iterator = _make_iterator(
        iter(range(5)), mode, transform=transform, lazy_start=True
    )
    monkeypatch.setattr(prefetch, "_CLOSE_TIMEOUT", 0.01)
    try:
        value = next(iterator)
        del value
        iterator.notify()
        assert entered.wait(5)
        with pytest.raises(TimeoutError, match="still running"):
            iterator.close()
        assert not iterator._closed
        assert iterator._thread.is_alive()
        assert refs[0]() is not None
    finally:
        release.set()
        iterator._thread.join(timeout=5)
        iterator.close()
    _assert_closed(iterator)
    assert all(ref() is None for ref in refs)


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", _GPU_MODES)
def test_cuda_repeated_close_releases_storage(mode):
    refs = []

    def transform(value):
        result = torch.full((4096,), value, device="cuda")
        refs.append(weakref.ref(result))
        return result

    for _ in range(20):
        iterator = _make_iterator(iter(range(5)), mode, transform=transform)
        try:
            value = next(iterator)
            del value
        finally:
            iterator.close()
        _assert_closed(iterator)
        assert all(ref() is None for ref in refs)


def test_collect_holdables_preserves_ragged_and_nested_containers():
    values = torch.tensor([1, 2])
    offsets = torch.tensor([0, 2])
    weights = torch.tensor([0.5, 0.5])
    batch = ragged_tensor.RaggedTensor(
        values, [offsets], weight=weights, dense_shape=(1, 2)
    )
    held = prefetch._holdables_of(({"ragged": batch},))
    assert len(held) == 3
    assert held[0] is values
    assert held[1] is offsets
    assert held[2] is weights


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", ["thread", "worker_fetch"])
def test_cuda_close_with_full_output_queue(mode, monkeypatch):
    full = threading.Event()
    iterator = _make_iterator(iter(range(8)), mode, transform=lambda value: value)
    original_put = iterator._out_queue.put

    def observe_put(item, *args, **kwargs):
        original_put(item, *args, **kwargs)
        if isinstance(item, tuple) and item[0] == 1:
            full.set()

    monkeypatch.setattr(iterator._out_queue, "put", observe_put)
    try:
        assert next(iterator) == 0
        assert full.wait(5)
        assert iterator._out_queue.full()
    finally:
        iterator.close()
    _assert_closed(iterator)


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", _GPU_MODES)
def test_transform_failure_keeps_popped_raw_tensor_until_gpu_finishes(mode):
    finished = []
    results = []

    def produce():
        for _ in range(3):
            yield {"raw": torch.full((4096,), 7, device="cuda")}

    def transform(batch):
        raw = batch.pop("raw")
        torch.cuda._sleep(100_000_000)
        results.append(raw + 1)
        finished.append(torch.cuda.current_stream().record_event())
        raise ValueError("transform failed after launching GPU work")

    iterator = _make_iterator(produce(), mode, transform=transform)
    try:
        with pytest.raises((ValueError, RuntimeError), match="failed"):
            next(iterator)
        assert finished[0].query()
        torch.testing.assert_close(results[0], torch.full_like(results[0], 8))
        _assert_closed(iterator)
    finally:
        iterator.close()


@_REQUIRES_CUDA
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("mode", _GPU_MODES)
def test_close_after_current_device_changes(mode):
    with torch.cuda.device(0):
        iterator = _make_iterator(iter(range(3)), mode)
        next(iterator)
    try:
        with torch.cuda.device(1):
            iterator.close()
        _assert_closed(iterator)
    finally:
        iterator.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("mode", ["thread", "worker_fetch"])
@pytest.mark.parametrize("result", ["data", "end", "error", "missing"])
def test_cuda_worker_exits_after_queue_timeout(mode, result, monkeypatch):
    release = threading.Event()

    def transform(value):
        if result == "error":
            raise ValueError("transform failed")
        return torch.full((8,), value + 1, device="cuda")

    values = [6] if result in ("data", "error") else []
    iterator = _make_iterator(iter(values), mode, transform=transform, buffer_size=2)
    original_put = iterator._put_output

    def put_after_timeout(item):
        if not release.wait(5):
            raise TimeoutError("producer was not released")
        return True if result == "missing" else original_put(item)

    original_get = iterator._out_queue.get
    rechecked = False

    def get_with_exit_after_timeout(*args, **kwargs):
        nonlocal rechecked
        try:
            return original_get(*args, **kwargs)
        except queue.Empty:
            if not rechecked:
                rechecked = True
                release.set()
                iterator._thread.join(timeout=5)
                assert not iterator._thread.is_alive()
                expected = {"data": 2, "end": 1, "error": 1, "missing": 0}
                assert iterator._out_queue.qsize() == expected[result]
            raise

    monkeypatch.setattr(iterator, "_put_output", put_after_timeout)
    monkeypatch.setattr(iterator._out_queue, "get", get_with_exit_after_timeout)
    monkeypatch.setattr(prefetch, "_GATE_FALLBACK_TIMEOUT", 0.01)
    try:
        if result == "error":
            with pytest.raises(
                RuntimeError, match="cuda prefetch worker failed"
            ) as caught:
                next(iterator)
            assert isinstance(caught.value.__cause__, ValueError)
        elif result == "missing":
            with pytest.raises(RuntimeError, match="died without reporting"):
                next(iterator)
        else:
            if result == "data":
                value = next(iterator)
                torch.testing.assert_close(value, torch.full_like(value, 7))
            with pytest.raises(StopIteration):
                next(iterator)
        assert rechecked
        _assert_closed(iterator)
    finally:
        release.set()
        iterator.close()
