"""Unit tests for MapDatasetSampleFilter Python-side validation.

These tests mock the C++ pybind layer (column_io.lib.py_interface) so they can
run without `tools/build_and_develop.sh`. They cover only the Python-side
parameter validation in MapDatasetSampleFilter.__init__; end-to-end behavior
(actual row drop, denylist hit correctness) is covered by the integration test
at tests/integration/sample_filter/packer_dataset_sample_filter_test.py.
"""

import sys
from unittest.mock import MagicMock

# Mock the pybind layer BEFORE importing dataset.py, mirroring the pattern in
# tests/unit/dataset_test.py:1-9 (the C++ extension is not loadable in pure
# Python unit envs).
_mock_py_interface = MagicMock()
_mock_interface = MagicMock()
sys.modules.setdefault("column_io.lib.py_interface", _mock_py_interface)
sys.modules.setdefault("column_io.lib.interface", _mock_interface)

import pytest

from column_io.dataset import dataset as dataset_io


def _make_fake_input(schema_dict):
    """Construct a fake upstream Dataset with a controlled schema.

    Schema layout follows column-io's convention: a list whose first element
    is a dict mapping feature_name -> list[list[int]] (positions in tensor
    vector). For unit tests we only need the keys() to be probed.
    """
    fake = MagicMock(spec=dataset_io.Dataset)
    fake.schema = [schema_dict]
    # nest_seq_leaf_num / pack_nest_sequence will be called on schema; the
    # mocked C++ interface accepts anything, so we don't need to match shapes.
    fake.impl.return_value = MagicMock()
    return fake


def test_filter_dict_must_be_dict():
    fake_input = _make_fake_input({"sample_id": [[0]]})
    with pytest.raises(ValueError, match="filter_dict must be a dict"):
        dataset_io.MapDatasetSampleFilter(fake_input, filter_dict=["not", "a", "dict"])


def test_filter_dict_values_must_be_list_of_str():
    fake_input = _make_fake_input({"sample_id": [[0]]})
    # 非 list/tuple 值
    with pytest.raises(ValueError, match="must be list/tuple"):
        dataset_io.MapDatasetSampleFilter(
            fake_input, filter_dict={"pid": "single_value_not_a_list"}
        )
    # list 内含非 str
    with pytest.raises(ValueError, match="must contain only str values"):
        dataset_io.MapDatasetSampleFilter(
            fake_input, filter_dict={"pid": ["a", 123]}
        )


def test_filter_dict_keys_must_be_str():
    fake_input = _make_fake_input({"sample_id": [[0]]})
    with pytest.raises(ValueError, match="filter_dict keys must be str"):
        dataset_io.MapDatasetSampleFilter(fake_input, filter_dict={42: ["a"]})


def test_missing_sample_id_column_raises():
    """E2: schema 不含 sample_id 列时, 必须在 Python 层 fail-fast."""
    fake_input = _make_fake_input({"some_other_col": [[0]]})
    with pytest.raises(ValueError, match="'sample_id'"):
        dataset_io.MapDatasetSampleFilter(
            fake_input, filter_dict={"pid": ["x"]}
        )


def test_non_dict_schema_raises_value_error():
    """E2 hardened: schema 不是 list[dict] 时也必须给 ValueError, 而不是
    TypeError. (例如 SliceListStringDataset.schema 返回 kPlaceHolder=None)."""
    fake_input = MagicMock(spec=dataset_io.Dataset)
    fake_input.schema = None  # 模拟 schema 没初始化的退化情形
    fake_input.impl.return_value = MagicMock()
    with pytest.raises(ValueError, match="list\\[dict\\] input.schema"):
        dataset_io.MapDatasetSampleFilter(fake_input, filter_dict={"pid": ["x"]})

    fake_input2 = MagicMock(spec=dataset_io.Dataset)
    fake_input2.schema = [None]  # list[None] — schema[0] 不是 dict
    fake_input2.impl.return_value = MagicMock()
    with pytest.raises(ValueError, match="list\\[dict\\] input.schema"):
        dataset_io.MapDatasetSampleFilter(fake_input2, filter_dict={"pid": ["x"]})


def test_empty_filter_dict_is_accepted():
    """E1: 空 filter_dict 应当被接受 (no-op), 不报错."""
    fake_input = _make_fake_input({"sample_id": [[0]]})
    # 不应抛异常; C++ ClassifyBatch fast-path 全填 0
    ds = dataset_io.MapDatasetSampleFilter(fake_input, filter_dict={})
    # schema 应已经注入 _sample_group_id
    assert any("_sample_group_id" in entry for entry in ds.schema)


def test_registry_resolves_sample_filter():
    """sample_filter 应能通过 MapDatasetRegistry 解析."""
    cls = dataset_io.map_dataset_registry.get_class("sample_filter")
    assert cls is dataset_io.MapDatasetSampleFilter


def test_map_dispatch_via_ds_dot_map():
    """用户期望调用方式: ds.map(name='sample_filter', kargs={...})."""
    fake_input = _make_fake_input({"sample_id": [[0]]})
    # Dataset.map 是 instance 方法; 这里直接调 base Dataset.map
    ds = dataset_io.Dataset.map(
        fake_input,
        name="sample_filter",
        kargs={"filter_dict": {"pid": ["x"]}},
    )
    assert isinstance(ds, dataset_io.MapDatasetSampleFilter)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
