import threading

import pytest

from recis.framework.pipeline_utils import close_prefetch
from recis.io import prefetch_dataset
from recis.io.prefetch_dataset import PrefetchDataset
from recis.io.wrap_end_dataset import WrapEndDataset


class _Source:
    def __init__(self):
        self.values = iter([1, 2])
        self.close_count = 0
        self.fail_close = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)

    def close(self):
        self.close_count += 1
        if self.fail_close:
            raise TimeoutError("input still running")


@pytest.mark.parametrize("consume", [0, 1, 3])
def test_close_forwards_before_start_after_consumption_and_exhaustion(consume):
    source = _Source()
    iterator = iter(WrapEndDataset(source))
    expected = [(False, 1), (False, 2), (True, None)]
    for index in range(consume):
        assert next(iterator) == expected[index]
    close_prefetch(iterator)
    close_prefetch(iterator)
    assert source.close_count == 1
    assert iterator._closed
    assert iterator._input_iterator is None
    assert iterator._dataset is None
    assert next(iterator) == (True, None)


def test_failed_close_retains_input_for_retry():
    source = _Source()
    source.fail_close = True
    dataset = WrapEndDataset(source)
    iterator = iter(dataset)
    with pytest.raises(TimeoutError, match="input still running"):
        close_prefetch(iterator)
    assert not iterator._closed
    assert iterator._input_iterator is source
    assert iterator._dataset is dataset
    assert next(iterator) == (True, None)
    source.fail_close = False
    close_prefetch(iterator)
    close_prefetch(iterator)
    assert source.close_count == 2
    assert iterator._closed
    assert iterator._input_iterator is None


def test_close_accepts_input_without_close():
    iterator = iter(WrapEndDataset([1, 2]))
    close_prefetch(iterator)
    assert next(iterator) == (True, None)


def test_close_propagates_prefetch_timeout_and_allows_retry(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def produce():
        yield 1
        entered.set()
        if not release.wait(5):
            raise TimeoutError("producer was not released")
        yield 2

    iterator = iter(WrapEndDataset(PrefetchDataset(produce())))
    inner = iterator._input_iterator
    monkeypatch.setattr(prefetch_dataset, "_CLOSE_TIMEOUT", 0.01)
    try:
        assert next(iterator) == (False, 1)
        assert entered.wait(5)
        with pytest.raises(TimeoutError, match="still running"):
            close_prefetch(iterator)
        assert iterator._input_iterator is inner
        assert not iterator._closed
        assert inner._runner.is_alive()
    finally:
        release.set()
        inner._runner.join(timeout=5)
        close_prefetch(iterator)
        inner.close()
    assert iterator._closed
    assert inner._closed
    assert iterator._input_iterator is None
