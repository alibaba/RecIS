from types import SimpleNamespace
from unittest import mock

import pytest

from recis.framework import trainer as trainer_module
from recis.framework.pipeline_utils import wrap_with_prefetch
from recis.io.map_dataset import MapDataset
from recis.io.prefetch_dataset import PrefetchDataset
from recis.io.wrap_end_dataset import WrapEndDataset


_ENTRYPOINTS = [
    "_train_loop",
    "_eval_loop",
    "_train_eval_loop",
    "_train_loop_by_window",
    "_eval_loop_by_window",
    "_train_eval_loop_by_window",
]


class _Iterator:
    def __init__(self):
        self.close_count = 0
        self.close_error = None

    def __iter__(self):
        return self

    def __next__(self):
        return True, None

    def close(self):
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


class _Dataset:
    def __init__(self):
        self.iterators = []

    def __iter__(self):
        iterator = _Iterator()
        self.iterators.append(iterator)
        return iterator


def _make_trainer(monkeypatch):
    trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
    trainer.model = mock.Mock()
    trainer.args = SimpleNamespace(window_iter=1)
    trainer.hooks = [mock.Mock()]
    trainer.train_dataset = _Dataset()
    trainer.eval_dataset = _Dataset()
    trainer._pipeline_prefetch_transform = object()
    trainer._prefetch_buffer_size = 1
    trainer._prefetch_stream_priority = 0
    trainer._prefetch_enable_thread = True
    trainer._prefetch_fetch_in_thread = False
    trainer._active_prefetch_iter = None
    trainer._epoch = 5
    trainer.sync_exit_flag = lambda flag: flag
    trainer.get_new_window_iter = iter
    monkeypatch.setattr(
        trainer_module, "wrap_with_prefetch", lambda iterator, *args, **kwargs: iterator
    )
    return trainer


def _assert_released(trainer):
    acquired = trainer.train_dataset.iterators + trainer.eval_dataset.iterators
    assert acquired
    assert all(iterator.close_count == 1 for iterator in acquired)
    assert trainer._active_prefetch_iter is None
    assert trainer._epoch == 5


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_trainer_closes_on_input_exhaustion(entrypoint, monkeypatch):
    trainer = _make_trainer(monkeypatch)
    getattr(trainer, entrypoint)()
    _assert_released(trainer)


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_trainer_closes_at_max_steps(entrypoint, monkeypatch):
    trainer = _make_trainer(monkeypatch)
    if "train_eval" in entrypoint:
        getattr(trainer, entrypoint)(train_steps=0, eval_steps=0)
    else:
        getattr(trainer, entrypoint)(max_steps=0)
    _assert_released(trainer)
    trainer.hooks[0].before_step.assert_not_called()


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
@pytest.mark.parametrize("hook_name", ["before_step", "out_off_data"])
def test_trainer_closes_on_step_hook_error(entrypoint, hook_name, monkeypatch):
    trainer = _make_trainer(monkeypatch)
    getattr(trainer.hooks[0], hook_name).side_effect = ValueError("hook failed")
    with pytest.raises(ValueError, match="hook failed"):
        getattr(trainer, entrypoint)()
    _assert_released(trainer)


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_trainer_closes_raw_iterator_if_wrapping_fails(entrypoint, monkeypatch):
    trainer = _make_trainer(monkeypatch)
    monkeypatch.setattr(
        trainer_module,
        "wrap_with_prefetch",
        mock.Mock(side_effect=ValueError("wrap failed")),
    )
    with pytest.raises(ValueError, match="wrap failed"):
        getattr(trainer, entrypoint)()
    _assert_released(trainer)


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS[3:])
@pytest.mark.parametrize("hook_name", ["before_window", "after_window"])
def test_window_hook_errors_close_acquired_iterators(
    entrypoint, hook_name, monkeypatch
):
    trainer = _make_trainer(monkeypatch)
    getattr(trainer.hooks[0], hook_name).side_effect = ValueError("window hook failed")
    with pytest.raises(ValueError, match="window hook failed"):
        getattr(trainer, entrypoint)()
    _assert_released(trainer)


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS[3:])
@pytest.mark.parametrize("sync_error", [False, True])
def test_distributed_window_exit_closes_local_iterator(
    entrypoint, sync_error, monkeypatch
):
    trainer = _make_trainer(monkeypatch)
    trainer.sync_exit_flag = mock.Mock(return_value=True)
    if sync_error:
        trainer.sync_exit_flag.side_effect = ValueError("sync failed")
        with pytest.raises(ValueError, match="sync failed"):
            getattr(trainer, entrypoint)()
    else:
        getattr(trainer, entrypoint)()
    _assert_released(trainer)


@pytest.mark.parametrize("entrypoint", ["_train_loop", "_train_loop_by_window"])
def test_after_train_hook_cannot_notify_finished_iterator(entrypoint, monkeypatch):
    trainer = _make_trainer(monkeypatch)

    def after_train():
        assert trainer._active_prefetch_iter is None
        raise ValueError("after train failed")

    trainer.hooks[0].after_train.side_effect = after_train
    with pytest.raises(ValueError, match="after train failed"):
        getattr(trainer, entrypoint)()
    _assert_released(trainer)


def test_training_error_survives_close_error(monkeypatch):
    trainer = _make_trainer(monkeypatch)

    def before_step(**kwargs):
        del kwargs
        trainer.train_dataset.iterators[0].close_error = RuntimeError("close failed")
        raise ValueError("training failed")

    trainer.hooks[0].before_step.side_effect = before_step
    with pytest.raises(ValueError, match="training failed"):
        trainer._train_loop()
    _assert_released(trainer)


@pytest.mark.parametrize(
    "entrypoint",
    ["_train_loop", "_eval_loop", "_train_loop_by_window", "_eval_loop_by_window"],
)
def test_training_or_model_error_closes_iterator(entrypoint, monkeypatch):
    trainer = _make_trainer(monkeypatch)
    monkeypatch.setattr(_Iterator, "__next__", lambda self: (False, object()))
    trainer._global_step = 0
    trainer.data_to_cuda = False
    if "train_loop" in entrypoint:
        trainer.accelerator = mock.MagicMock()
        trainer._train_step = mock.Mock(side_effect=ValueError("training failed"))
    else:
        trainer.model.side_effect = ValueError("training failed")
    with pytest.raises(ValueError, match="training failed"):
        getattr(trainer, entrypoint)()
    _assert_released(trainer)


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
@pytest.mark.parametrize("exit_kind", ["exhaustion", "max_steps", "hook_error"])
def test_trainer_closes_real_cpu_prefetch_chain(entrypoint, exit_kind, monkeypatch):
    trainer = _make_trainer(monkeypatch)
    acquired = []

    class Dataset:
        def __iter__(self):
            mapped = MapDataset(range(4), map_funcs=[lambda value: value + 1])
            iterator = iter(WrapEndDataset(PrefetchDataset(mapped)))
            # Keep strong references so GC cannot hide missing close forwarding.
            acquired.append((iterator, iterator._input_iterator))
            return iterator

    trainer.train_dataset = Dataset()
    trainer.eval_dataset = Dataset()
    trainer._pipeline_prefetch_transform = None
    trainer.data_to_cuda = False
    trainer._global_step = 0
    trainer.accelerator = mock.MagicMock()
    trainer._train_step = mock.Mock()
    monkeypatch.setattr(trainer_module, "wrap_with_prefetch", wrap_with_prefetch)
    monkeypatch.setattr(trainer_module, "get_log_metrics", dict)
    kwargs = {}
    if exit_kind == "max_steps":
        kwargs = (
            {"train_steps": 1, "eval_steps": 1}
            if "train_eval" in entrypoint
            else {"max_steps": 1}
        )
    try:
        if exit_kind == "hook_error":
            trainer.hooks[0].after_data.side_effect = ValueError("hook failed")
            with pytest.raises(ValueError, match="hook failed"):
                getattr(trainer, entrypoint)(**kwargs)
        else:
            getattr(trainer, entrypoint)(**kwargs)
        assert acquired
        for outer, inner in acquired:
            assert outer._input_iterator is None
            assert inner._closed
            assert not inner._runner.is_alive()
            assert inner._queue.empty()
        assert trainer._active_prefetch_iter is None
    finally:
        for _, inner in acquired:
            inner.close()
