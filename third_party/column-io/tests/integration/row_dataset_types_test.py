# -*- coding: utf-8 -*-
"""
Tests for OdpsOpenStorageRowDataset Arrow->Python type conversion.

Exercises the C++ ``ArrowCellToPyObject`` path via the ``test_convert_ipc_file``
helper exposed from the row_dataset_test pybind module. This lets us verify
every Arrow type mapping end-to-end without ODPS network access.

Prerequisites:
    - column_io built with BUILD_TESTING=ON and INTERNAL_VERSION=1
    - pyarrow installed

Usage:
    pytest tests/integration/row_dataset_types_test.py -vs
"""

import math
import os
import tempfile

import pyarrow as pa
import pytest

from column_io.lib import odps_open_storage_row_dataset_test as row_dataset_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ipc_file(batches, path: str):
    """Write one or more RecordBatches to an Arrow IPC file."""
    if isinstance(batches, pa.RecordBatch):
        batches = [batches]
    schema = batches[0].schema
    with pa.OSFile(path, "wb") as sink:
        writer = pa.ipc.new_file(sink, schema)
        for batch in batches:
            writer.write_batch(batch)
        writer.close()


def _build_full_batch() -> pa.RecordBatch:
    """Build a RecordBatch with one column per supported Arrow type.

    3 rows: row 0 & 1 have values, row 2 is all-null.
    """
    struct_type = pa.struct([("x", pa.int32()), ("y", pa.string())])
    map_type = pa.map_(pa.string(), pa.int64())
    nested_type = pa.list_(pa.struct([("id", pa.int32()), ("val", pa.float64())]))
    map_list_type = pa.map_(pa.string(), pa.list_(pa.int32()))

    arrays = [
        pa.array([True, False, None], type=pa.bool_()),
        pa.array([1, -1, None], type=pa.int8()),
        pa.array([256, -256, None], type=pa.int16()),
        pa.array([100000, -100000, None], type=pa.int32()),
        pa.array([2**40, -(2**40), None], type=pa.int64()),
        pa.array([0, 255, None], type=pa.uint8()),
        pa.array([0, 65535, None], type=pa.uint16()),
        pa.array([0, 2**32 - 1, None], type=pa.uint32()),
        pa.array([0, 2**63 - 1, None], type=pa.uint64()),
        pa.array([1.5, -1.5, None], type=pa.float32()),
        pa.array([3.14, -2.71, None], type=pa.float64()),
        pa.array(["hello", "世界", None], type=pa.string()),
        pa.array([b"\x00\x01\x02", b"\xff", None], type=pa.binary()),
        pa.array(["large_str", "", None], type=pa.large_string()),
        pa.array([b"\xab\xcd", b"", None], type=pa.large_binary()),
        pa.array([[1, 2, 3], [4], None], type=pa.list_(pa.int64())),
        pa.array([["a", "b"], ["c"], None], type=pa.large_list(pa.string())),
        pa.array([[10, 20], [30, 40], None], type=pa.list_(pa.int32(), 2)),
        pa.array(
            [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, None], type=struct_type
        ),
        pa.array(
            [[("k1", 10), ("k2", 20)], [("k3", 30)], None], type=map_type
        ),
        pa.array(
            [
                [{"id": 1, "val": 1.1}, {"id": 2, "val": 2.2}],
                [{"id": 3, "val": 3.3}],
                None,
            ],
            type=nested_type,
        ),
        pa.array(
            [
                [("scores", [90, 80]), ("ids", [1, 2])],
                [("empty", [])],
                None,
            ],
            type=map_list_type,
        ),
        pa.array([1577836800000, 0, None], type=pa.timestamp("ms")),
    ]

    schema = pa.schema([
        ("col_bool", pa.bool_()),
        ("col_int8", pa.int8()),
        ("col_int16", pa.int16()),
        ("col_int32", pa.int32()),
        ("col_int64", pa.int64()),
        ("col_uint8", pa.uint8()),
        ("col_uint16", pa.uint16()),
        ("col_uint32", pa.uint32()),
        ("col_uint64", pa.uint64()),
        ("col_float", pa.float32()),
        ("col_double", pa.float64()),
        ("col_string", pa.string()),
        ("col_binary", pa.binary()),
        ("col_large_string", pa.large_string()),
        ("col_large_binary", pa.large_binary()),
        ("col_list", pa.list_(pa.int64())),
        ("col_large_list", pa.large_list(pa.string())),
        ("col_fixed_list", pa.list_(pa.int32(), 2)),
        ("col_struct", struct_type),
        ("col_map", map_type),
        ("col_nested_list_struct", nested_type),
        ("col_map_list_val", map_list_type),
        ("col_timestamp_ms", pa.timestamp("ms")),
    ])

    return pa.RecordBatch.from_arrays(arrays, schema=schema)


# Expected values for row 0 and row 1 (col_name, row0_val, row1_val, pytype).
# For floats where fp precision matters, row*_val is None and only type is checked.
ROW_EXPECTATIONS = [
    ("col_bool",               True,              False,             bool),
    ("col_int8",               1,                 -1,                int),
    ("col_int16",              256,               -256,              int),
    ("col_int32",              100000,            -100000,           int),
    ("col_int64",              2**40,             -(2**40),          int),
    ("col_uint8",              0,                 255,               int),
    ("col_uint16",             0,                 65535,             int),
    ("col_uint32",             0,                 2**32 - 1,         int),
    ("col_uint64",             0,                 2**63 - 1,         int),
    ("col_float",              None,              None,              float),
    ("col_double",             3.14,              -2.71,             float),
    ("col_string",             "hello",           "世界",            str),
    ("col_binary",             b"\x00\x01\x02",   b"\xff",           bytes),
    ("col_large_string",       "large_str",       "",                str),
    ("col_large_binary",       b"\xab\xcd",       b"",               bytes),
    ("col_list",               [1, 2, 3],         [4],               list),
    ("col_large_list",         ["a", "b"],        ["c"],             list),
    ("col_fixed_list",         [10, 20],          [30, 40],          list),
    ("col_struct",             {"x": 1, "y": "a"}, {"x": 2, "y": "b"}, dict),
    ("col_map",                {"k1": 10, "k2": 20}, {"k3": 30},    dict),
    ("col_nested_list_struct",
     [{"id": 1, "val": 1.1}, {"id": 2, "val": 2.2}],
     [{"id": 3, "val": 3.3}],                                       list),
    ("col_map_list_val",
     {"scores": [90, 80], "ids": [1, 2]},
     {"empty": []},                                                  dict),
    ("col_timestamp_ms",   1577836800000,         0,                  int),
]


@pytest.fixture
def ipc_file():
    """Write the full test batch to a temp IPC file, clean up afterwards."""
    batch = _build_full_batch()
    fd, path = tempfile.mkstemp(suffix=".arrow")
    os.close(fd)
    try:
        _write_ipc_file(batch, path)
        yield path
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestArrowCellToPyObject:
    """Test the C++ ArrowCellToPyObject conversion via row_dataset_test."""

    def test_row_count(self, ipc_file):
        rows = row_dataset_test.test_convert_ipc_file(ipc_file)
        assert len(rows) == 3

    def test_column_count(self, ipc_file):
        rows = row_dataset_test.test_convert_ipc_file(ipc_file)
        assert len(rows[0]) == len(ROW_EXPECTATIONS)

    @pytest.mark.parametrize(
        "col_idx, col_name, expected_r0, expected_r1, expected_type",
        [
            (i, name, r0, r1, t)
            for i, (name, r0, r1, t) in enumerate(ROW_EXPECTATIONS)
        ],
        ids=[e[0] for e in ROW_EXPECTATIONS],
    )
    def test_row0_types_and_values(
        self, ipc_file, col_idx, col_name, expected_r0, expected_r1, expected_type
    ):
        rows = row_dataset_test.test_convert_ipc_file(ipc_file)
        actual = rows[0][col_idx]
        assert isinstance(actual, expected_type), (
            f"row0[{col_name}]: expected {expected_type.__name__}, "
            f"got {type(actual).__name__}"
        )
        if expected_r0 is not None:
            assert actual == expected_r0, (
                f"row0[{col_name}]: expected {expected_r0!r}, got {actual!r}"
            )

    @pytest.mark.parametrize(
        "col_idx, col_name, expected_r0, expected_r1, expected_type",
        [
            (i, name, r0, r1, t)
            for i, (name, r0, r1, t) in enumerate(ROW_EXPECTATIONS)
        ],
        ids=[e[0] for e in ROW_EXPECTATIONS],
    )
    def test_row1_types_and_values(
        self, ipc_file, col_idx, col_name, expected_r0, expected_r1, expected_type
    ):
        rows = row_dataset_test.test_convert_ipc_file(ipc_file)
        actual = rows[1][col_idx]
        assert isinstance(actual, expected_type), (
            f"row1[{col_name}]: expected {expected_type.__name__}, "
            f"got {type(actual).__name__}"
        )
        if expected_r1 is not None:
            assert actual == expected_r1, (
                f"row1[{col_name}]: expected {expected_r1!r}, got {actual!r}"
            )

    def test_row2_all_nulls(self, ipc_file):
        rows = row_dataset_test.test_convert_ipc_file(ipc_file)
        r2 = rows[2]
        for col_idx, (col_name, *_) in enumerate(ROW_EXPECTATIONS):
            assert r2[col_idx] is None, (
                f"row2[{col_name}]: expected None, got {r2[col_idx]!r}"
            )

    def test_float_precision(self, ipc_file):
        """Verify float32 values are within expected precision."""
        rows = row_dataset_test.test_convert_ipc_file(ipc_file)
        float_idx = 9  # col_float
        assert math.isclose(rows[0][float_idx], 1.5, rel_tol=1e-6)
        assert math.isclose(rows[1][float_idx], -1.5, rel_tol=1e-6)


class TestSelectedColumns:
    """Test column selection / filtering via the selected_columns parameter."""

    def test_select_subset(self, ipc_file):
        selected = ["col_string", "col_int64", "col_bool"]
        rows = row_dataset_test.test_convert_ipc_file(ipc_file, selected)
        assert len(rows) == 3
        assert len(rows[0]) == 3
        assert rows[0][0] == "hello"
        assert rows[0][1] == 2**40
        assert rows[0][2] is True

    def test_select_single_column(self, ipc_file):
        rows = row_dataset_test.test_convert_ipc_file(ipc_file, ["col_double"])
        assert len(rows[0]) == 1
        assert rows[0][0] == 3.14

    def test_nonexistent_column_returns_none(self, ipc_file):
        rows = row_dataset_test.test_convert_ipc_file(
            ipc_file, ["col_string", "no_such_column"]
        )
        assert len(rows[0]) == 2
        assert rows[0][0] == "hello"
        assert rows[0][1] is None

    def test_empty_selected_returns_all(self, ipc_file):
        rows = row_dataset_test.test_convert_ipc_file(ipc_file, [])
        assert len(rows[0]) == len(ROW_EXPECTATIONS)


class TestMultiBatch:
    """Test that test_convert_ipc_file handles IPC files with multiple batches."""

    def test_two_batches(self):
        batch = _build_full_batch()
        fd, path = tempfile.mkstemp(suffix=".arrow")
        os.close(fd)
        try:
            _write_ipc_file([batch, batch], path)
            rows = row_dataset_test.test_convert_ipc_file(path)
            assert len(rows) == 6
            assert rows[0][0] is True  # col_bool row 0 batch 1
            assert rows[3][0] is True  # col_bool row 0 batch 2
        finally:
            os.unlink(path)


class TestEmptyBatch:
    """Test behavior with an empty RecordBatch (0 rows)."""

    def test_empty_batch_returns_empty(self):
        schema = pa.schema([("col_int", pa.int32())])
        empty = pa.RecordBatch.from_arrays(
            [pa.array([], type=pa.int32())], schema=schema
        )
        fd, path = tempfile.mkstemp(suffix=".arrow")
        os.close(fd)
        try:
            _write_ipc_file(empty, path)
            rows = row_dataset_test.test_convert_ipc_file(path)
            assert len(rows) == 0
        finally:
            os.unlink(path)
