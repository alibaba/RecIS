from unittest import mock

import pytest
import torch

from recis.framework import pipeline_utils


def test_prefetch_fails_at_setup_when_model_has_no_capability():
    with pytest.raises(TypeError, match="prefetch_step"):
        pipeline_utils.setup_pipeline_prefetch(
            model=torch.nn.Linear(2, 2),
            data_to_cuda=False,
            enable_pipeline_prefetch=True,
        )


def test_explicit_pipeline_transform_is_supported_without_model_method():
    transform, size = pipeline_utils.setup_pipeline_prefetch(
        model=torch.nn.Linear(2, 2),
        data_to_cuda=False,
        enable_pipeline_prefetch=True,
        pipeline_prefetch_fn=lambda value: value + 1,
    )

    assert size == 1
    assert transform((False, 2)) == (False, 3)
    assert transform((True, 2)) == (True, 2)


def test_close_prefetch_accepts_missing_and_noncallable_close():
    class NoClose:
        close = None

    pipeline_utils.close_prefetch(None)
    pipeline_utils.close_prefetch(iter([1]))
    pipeline_utils.close_prefetch(NoClose())


def test_close_prefetch_calls_close():
    iterator = mock.Mock()
    pipeline_utils.close_prefetch(iterator)
    iterator.close.assert_called_once_with()


def test_close_prefetch_preserves_original_error():
    iterator = mock.Mock()
    iterator.close.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(ValueError, match="training failed"):
        try:
            raise ValueError("training failed")
        finally:
            pipeline_utils.close_prefetch(iterator)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        pipeline_utils.close_prefetch(iterator)
