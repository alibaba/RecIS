# -*- coding: utf-8 -*-
"""
Corner-case tests for ArrowCellToPyObject data conversion.

Focuses on data-level edge cases: NULL in various positions, UINT64 overflow,
non-UTF8 strings, irregular nested arrays, MAP with null keys, etc.

Prerequisites:
    - column_io built with BUILD_TESTING=ON and INTERNAL_VERSION=1
    - pyarrow installed

Usage:
    pytest tests/integration/row_dataset_corner_cases_test.py -vs
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


@pytest.fixture
def _tmp_ipc():
    """Context manager that provides a temp path for IPC files."""
    fd, path = tempfile.mkstemp(suffix=".arrow")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def _convert(batch, selected_columns=None):
    """Helper: write batch to IPC file, convert via C++, return rows."""
    fd, path = tempfile.mkstemp(suffix=".arrow")
    os.close(fd)
    try:
        _write_ipc_file(batch, path)
        cols = selected_columns or []
        return row_dataset_test.test_convert_ipc_file(path, cols)
    finally:
        os.unlink(path)


# ===========================================================================
# 1. NULL values in various positions
# ===========================================================================

class TestNullScalars:
    """NULL values for scalar types."""

    def test_int_null(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([None, 1, None], type=pa.int64())})
        rows = _convert(batch)
        assert rows[0][0] is None
        assert rows[1][0] == 1
        assert rows[2][0] is None

    def test_float_null(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([None, 3.14, None], type=pa.float64())})
        rows = _convert(batch)
        assert rows[0][0] is None
        assert rows[1][0] == 3.14
        assert rows[2][0] is None

    def test_bool_null(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([None, True, False], type=pa.bool_())})
        rows = _convert(batch)
        assert rows[0][0] is None
        assert rows[1][0] is True
        assert rows[2][0] is False

    def test_string_null(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([None, "hello", None], type=pa.string())})
        rows = _convert(batch)
        assert rows[0][0] is None
        assert rows[1][0] == "hello"
        assert rows[2][0] is None

    def test_binary_null(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([None, b"\x01\x02", None], type=pa.binary())})
        rows = _convert(batch)
        assert rows[0][0] is None
        assert rows[1][0] == b"\x01\x02"
        assert rows[2][0] is None

    def test_all_rows_null(self):
        """Entire column is NULL."""
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([None, None, None], type=pa.int64())})
        rows = _convert(batch)
        assert all(r[0] is None for r in rows)


class TestNullInArrays:
    """NULL values inside Array/List types."""

    def test_array_element_null(self):
        """Array with NULL elements: [1, None, 3]."""
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[1, None, 3]], type=pa.list_(pa.int64()))})
        rows = _convert(batch)
        assert rows[0][0] == [1, None, 3]

    def test_array_all_elements_null(self):
        """Array where all elements are NULL: [None, None]."""
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[None, None]], type=pa.list_(pa.int64()))})
        rows = _convert(batch)
        assert rows[0][0] == [None, None]

    def test_array_column_null(self):
        """Entire array cell is NULL (not elements, the whole list)."""
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([None, [1, 2]], type=pa.list_(pa.int64()))})
        rows = _convert(batch)
        assert rows[0][0] is None
        assert rows[1][0] == [1, 2]

    def test_nested_array_with_null_sublists(self):
        """Nested list where some sub-lists are NULL."""
        typ = pa.list_(pa.list_(pa.int32()))
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[[1, 2], None, [3]]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == [[1, 2], None, [3]]

    def test_array_of_strings_with_null(self):
        """Array<String> with NULL elements."""
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([["a", None, "c"]], type=pa.list_(pa.string()))})
        rows = _convert(batch)
        assert rows[0][0] == ["a", None, "c"]

    def test_multi_row_array_null_patterns(self):
        """6 rows covering every NULL pattern combination for List<Int64>."""
        typ = pa.list_(pa.int64())
        batch = pa.RecordBatch.from_pydict({"v": pa.array([
            [1, 2, 3],
            [None, 2, None],
            [],
            None,
            [None, None, None],
            [1],
        ], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == [1, 2, 3]
        assert rows[1][0] == [None, 2, None]
        assert rows[2][0] == []
        assert rows[3][0] is None
        assert rows[4][0] == [None, None, None]
        assert rows[5][0] == [1]

    def test_multi_column_arrays_null_patterns(self):
        """3 array columns x 4 rows, each column/row with different NULL positions."""
        batch = pa.RecordBatch.from_pydict({
            "int_list": pa.array(
                [[1, 2], None, [None], [3, 4, 5]],
                type=pa.list_(pa.int64())),
            "str_list": pa.array(
                [["a", "b"], ["c"], None, [None, "d"]],
                type=pa.list_(pa.string())),
            "nested": pa.array(
                [[[1, 2], [3]], None, [[None]], [[], [4, 5]]],
                type=pa.list_(pa.list_(pa.int32()))),
        })
        rows = _convert(batch)
        assert rows[0] == ([1, 2], ["a", "b"], [[1, 2], [3]])
        assert rows[1] == (None, ["c"], None)
        assert rows[2] == ([None], None, [[None]])
        assert rows[3] == ([3, 4, 5], [None, "d"], [[], [4, 5]])


class TestNullInMaps:
    """NULL values in Map types."""

    def test_map_value_null(self):
        """Map with NULL value: {"a": None}."""
        typ = pa.map_(pa.string(), pa.int64())
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[("a", None)]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"a": None}

    def test_map_all_values_null(self):
        """Map where all values are NULL."""
        typ = pa.map_(pa.string(), pa.int64())
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[("k1", None), ("k2", None)]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"k1": None, "k2": None}

    def test_map_column_null(self):
        """Entire map cell is NULL."""
        typ = pa.map_(pa.string(), pa.int64())
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([None, [("k", 1)]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] is None
        assert rows[1][0] == {"k": 1}

    def test_multi_row_map_null_patterns(self):
        """2 map columns x 5 rows with varying NULL patterns."""
        simple_type = pa.map_(pa.string(), pa.int64())
        nested_type = pa.map_(pa.string(), pa.list_(pa.int32()))
        batch = pa.RecordBatch.from_pydict({
            "simple_map": pa.array([
                [("a", 1), ("b", 2)],
                [("c", None)],
                None,
                [],
                [("d", 4), ("e", 5), ("f", 6)],
            ], type=simple_type),
            "nested_map": pa.array([
                [("x", [1, 2])],
                None,
                [("y", None)],
                [("z", [])],
                [("w", [None, 1])],
            ], type=nested_type),
        })
        rows = _convert(batch)
        assert rows[0] == ({"a": 1, "b": 2}, {"x": [1, 2]})
        assert rows[1] == ({"c": None}, None)
        assert rows[2] == (None, {"y": None})
        assert rows[3] == ({}, {"z": []})
        assert rows[4] == ({"d": 4, "e": 5, "f": 6}, {"w": [None, 1]})


class TestNullInStructs:
    """NULL values in Struct types."""

    def test_struct_field_null(self):
        """Struct with some fields NULL."""
        typ = pa.struct([("x", pa.int32()), ("y", pa.string())])
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([{"x": 1, "y": None}], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"x": 1, "y": None}

    def test_struct_all_fields_null(self):
        """Struct where all fields are NULL (but struct itself is not NULL)."""
        typ = pa.struct([("x", pa.int32()), ("y", pa.string())])
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([{"x": None, "y": None}], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"x": None, "y": None}

    def test_struct_column_null(self):
        """Entire struct cell is NULL."""
        typ = pa.struct([("x", pa.int32()), ("y", pa.string())])
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([None, {"x": 1, "y": "a"}], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] is None
        assert rows[1][0] == {"x": 1, "y": "a"}

    def test_multi_row_struct_null_patterns(self):
        """5 rows with different fields being None in each row."""
        typ = pa.struct([
            ("x", pa.int32()), ("y", pa.string()), ("z", pa.list_(pa.int32()))
        ])
        batch = pa.RecordBatch.from_pydict({"v": pa.array([
            {"x": 1, "y": "a", "z": [1, 2]},
            {"x": None, "y": "b", "z": [3]},
            {"x": 2, "y": None, "z": None},
            {"x": None, "y": None, "z": None},
            None,
        ], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"x": 1, "y": "a", "z": [1, 2]}
        assert rows[1][0] == {"x": None, "y": "b", "z": [3]}
        assert rows[2][0] == {"x": 2, "y": None, "z": None}
        assert rows[3][0] == {"x": None, "y": None, "z": None}
        assert rows[4][0] is None


# ===========================================================================
# 2. UINT64 large values (overflow regression test)
# ===========================================================================

class TestUint64Overflow:
    """UINT64 values that exceed INT64_MAX must not overflow to negative."""

    def test_uint64_max(self):
        val = 2**64 - 1  # 18446744073709551615
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([val], type=pa.uint64())})
        rows = _convert(batch)
        assert rows[0][0] == val
        assert rows[0][0] > 0

    def test_uint64_just_above_int64_max(self):
        val = 2**63  # 9223372036854775808, first value that overflows int64
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([val], type=pa.uint64())})
        rows = _convert(batch)
        assert rows[0][0] == val
        assert rows[0][0] > 0

    def test_uint64_various_large_values(self):
        values = [2**63, 2**63 + 1, 2**64 - 2, 2**64 - 1]
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array(values, type=pa.uint64())})
        rows = _convert(batch)
        for i, expected in enumerate(values):
            assert rows[i][0] == expected, (
                f"row {i}: expected {expected}, got {rows[i][0]}")
            assert rows[i][0] > 0

    def test_uint64_zero_and_small(self):
        """Ensure small UINT64 values still work correctly."""
        values = [0, 1, 2**32, 2**63 - 1]
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array(values, type=pa.uint64())})
        rows = _convert(batch)
        for i, expected in enumerate(values):
            assert rows[i][0] == expected

    def test_uint64_null(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([2**64 - 1, None], type=pa.uint64())})
        rows = _convert(batch)
        assert rows[0][0] == 2**64 - 1
        assert rows[1][0] is None


# ===========================================================================
# 3. Array / List edge cases
# ===========================================================================

class TestArrayEdgeCases:
    """Edge cases for Array/List columns."""

    def test_empty_array(self):
        """Empty list []."""
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[]], type=pa.list_(pa.int64()))})
        rows = _convert(batch)
        assert rows[0][0] == []

    def test_single_element_array(self):
        """Single-element list [42]."""
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[42]], type=pa.list_(pa.int64()))})
        rows = _convert(batch)
        assert rows[0][0] == [42]

    def test_ragged_arrays(self):
        """Irregular-length arrays in different rows."""
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[1, 2], [3, 4, 5, 6], [7]], type=pa.list_(pa.int32()))})
        rows = _convert(batch)
        assert rows[0][0] == [1, 2]
        assert rows[1][0] == [3, 4, 5, 6]
        assert rows[2][0] == [7]

    def test_nested_ragged_arrays(self):
        """Nested irregular arrays: [[1,2], [3,4,5]]."""
        typ = pa.list_(pa.list_(pa.int32()))
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[[1, 2], [3, 4, 5]]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == [[1, 2], [3, 4, 5]]

    def test_deeply_nested_arrays(self):
        """3 levels of nesting."""
        typ = pa.list_(pa.list_(pa.list_(pa.int32())))
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[[[1, 2], [3]], [[4]]]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == [[[1, 2], [3]], [[4]]]

    def test_large_array(self):
        """Array with many elements (performance boundary)."""
        large_list = list(range(10000))
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([large_list], type=pa.list_(pa.int64()))})
        rows = _convert(batch)
        assert rows[0][0] == large_list

    def test_array_of_empty_strings(self):
        """Array<String> with empty strings."""
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([["", "", ""]], type=pa.list_(pa.string()))})
        rows = _convert(batch)
        assert rows[0][0] == ["", "", ""]

    def test_fixed_size_list_with_nulls(self):
        """Fixed-size list with NULL elements."""
        typ = pa.list_(pa.int32(), 3)
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[1, None, 3]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == [1, None, 3]

    def test_multi_row_ragged_with_nulls(self):
        """7 rows: ragged lengths + null elements + null rows combined."""
        typ = pa.list_(pa.int64())
        batch = pa.RecordBatch.from_pydict({"v": pa.array([
            [1, 2, 3, 4, 5],
            [],
            [None],
            [6, 7],
            None,
            [None, 8, None, 9, None],
            [10],
        ], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == [1, 2, 3, 4, 5]
        assert rows[1][0] == []
        assert rows[2][0] == [None]
        assert rows[3][0] == [6, 7]
        assert rows[4][0] is None
        assert rows[5][0] == [None, 8, None, 9, None]
        assert rows[6][0] == [10]


# ===========================================================================
# 5. Float special values
# ===========================================================================

class TestFloatSpecialValues:
    """NaN, Inf, -Inf, subnormals for float/double."""

    def test_nan(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([float("nan")], type=pa.float64())})
        rows = _convert(batch)
        assert math.isnan(rows[0][0])

    def test_inf(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([float("inf")], type=pa.float64())})
        rows = _convert(batch)
        assert rows[0][0] == float("inf")

    def test_negative_inf(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([float("-inf")], type=pa.float64())})
        rows = _convert(batch)
        assert rows[0][0] == float("-inf")

    def test_negative_zero(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([-0.0], type=pa.float64())})
        rows = _convert(batch)
        assert rows[0][0] == 0.0
        assert math.copysign(1.0, rows[0][0]) == -1.0

    def test_float32_nan_inf(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([float("nan"), float("inf"), float("-inf")],
                           type=pa.float32())})
        rows = _convert(batch)
        assert math.isnan(rows[0][0])
        assert rows[1][0] == float("inf")
        assert rows[2][0] == float("-inf")

    def test_subnormal_float(self):
        """Subnormal (denormalized) float value."""
        import sys
        val = sys.float_info.min * sys.float_info.epsilon
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([val], type=pa.float64())})
        rows = _convert(batch)
        assert rows[0][0] == val


# ===========================================================================
# 6. String edge cases
# ===========================================================================

class TestStringEdgeCases:
    """Edge cases for string data (valid UTF-8)."""

    def test_empty_string(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([""], type=pa.string())})
        rows = _convert(batch)
        assert rows[0][0] == ""
        assert isinstance(rows[0][0], str)

    def test_unicode_multibyte(self):
        """Multi-byte UTF-8 characters."""
        vals = ["日本語", "中文", "العربية", "한국어"]
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array(vals, type=pa.string())})
        rows = _convert(batch)
        for i, expected in enumerate(vals):
            assert rows[i][0] == expected

    def test_emoji(self):
        """4-byte UTF-8 codepoints (emoji)."""
        vals = ["🎉🎊", "👨‍👩‍👧‍👦", "🏴󠁧󠁢󠁳󠁣󠁴󠁿"]
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array(vals, type=pa.string())})
        rows = _convert(batch)
        for i, expected in enumerate(vals):
            assert rows[i][0] == expected

    def test_string_with_embedded_null(self):
        """String with \\0 in the middle."""
        val = "before\x00after"
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([val], type=pa.string())})
        rows = _convert(batch)
        assert rows[0][0] == val
        assert len(rows[0][0]) == len(val)

    def test_very_long_string(self):
        """String with 100k characters."""
        val = "x" * 100000
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([val], type=pa.string())})
        rows = _convert(batch)
        assert rows[0][0] == val
        assert len(rows[0][0]) == 100000


# ===========================================================================
# 7. Map edge cases
# ===========================================================================

class TestMapEdgeCases:
    """Edge cases for Map columns."""

    def test_empty_map(self):
        """Empty map {}."""
        typ = pa.map_(pa.string(), pa.int64())
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {}

    def test_map_with_empty_string_key(self):
        """Map with empty string as key."""
        typ = pa.map_(pa.string(), pa.int64())
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[("", 42)]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"": 42}

    def test_map_with_int_keys(self):
        """Map with integer keys."""
        typ = pa.map_(pa.int64(), pa.string())
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[(1, "a"), (2, "b")]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {1: "a", 2: "b"}

    def test_map_with_nested_value(self):
        """Map with list values."""
        typ = pa.map_(pa.string(), pa.list_(pa.int32()))
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[("k", [1, 2, 3])]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"k": [1, 2, 3]}

    def test_map_many_entries(self):
        """Map with many entries."""
        entries = [(f"key_{i}", i) for i in range(100)]
        typ = pa.map_(pa.string(), pa.int64())
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([entries], type=typ)})
        rows = _convert(batch)
        expected = {f"key_{i}": i for i in range(100)}
        assert rows[0][0] == expected

    def test_multi_row_maps_varying_sizes(self):
        """5 rows with different entry counts and null patterns."""
        typ = pa.map_(pa.string(), pa.int64())
        batch = pa.RecordBatch.from_pydict({"v": pa.array([
            [("a", 1)],
            [("b", 2), ("c", 3), ("d", 4)],
            [],
            None,
            [("e", None), ("f", 5)],
        ], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"a": 1}
        assert rows[1][0] == {"b": 2, "c": 3, "d": 4}
        assert rows[2][0] == {}
        assert rows[3][0] is None
        assert rows[4][0] == {"e": None, "f": 5}


# ===========================================================================
# 8. Struct edge cases
# ===========================================================================

class TestStructEdgeCases:
    """Edge cases for Struct columns."""

    def test_struct_with_nested_struct(self):
        """Struct containing another struct."""
        inner = pa.struct([("a", pa.int32())])
        outer = pa.struct([("inner", inner), ("b", pa.string())])
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([{"inner": {"a": 42}, "b": "hello"}], type=outer)})
        rows = _convert(batch)
        assert rows[0][0] == {"inner": {"a": 42}, "b": "hello"}

    def test_struct_with_list_field(self):
        """Struct containing a list field."""
        typ = pa.struct([("ids", pa.list_(pa.int64())), ("name", pa.string())])
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([{"ids": [1, 2, 3], "name": "test"}], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"ids": [1, 2, 3], "name": "test"}

    def test_struct_with_map_field(self):
        """Struct containing a map field."""
        typ = pa.struct([
            ("props", pa.map_(pa.string(), pa.string())),
            ("id", pa.int32()),
        ])
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([{"props": [("color", "red")], "id": 1}], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"props": {"color": "red"}, "id": 1}

    def test_multi_row_nested_struct(self):
        """4 rows of nested struct with varying NULL at different levels."""
        inner = pa.struct([("a", pa.int32()), ("b", pa.string())])
        outer = pa.struct([("inner", inner), ("tag", pa.string())])
        batch = pa.RecordBatch.from_pydict({"v": pa.array([
            {"inner": {"a": 1, "b": "x"}, "tag": "t1"},
            {"inner": None, "tag": "t2"},
            {"inner": {"a": None, "b": "y"}, "tag": None},
            None,
        ], type=outer)})
        rows = _convert(batch)
        assert rows[0][0] == {"inner": {"a": 1, "b": "x"}, "tag": "t1"}
        assert rows[1][0] == {"inner": None, "tag": "t2"}
        assert rows[2][0] == {"inner": {"a": None, "b": "y"}, "tag": None}
        assert rows[3][0] is None


# ===========================================================================
# 9. Integer boundary values
# ===========================================================================

class TestIntegerBoundaries:
    """Boundary values for integer types."""

    def test_int8_boundaries(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([-128, 127, 0], type=pa.int8())})
        rows = _convert(batch)
        assert rows[0][0] == -128
        assert rows[1][0] == 127
        assert rows[2][0] == 0

    def test_int16_boundaries(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([-32768, 32767], type=pa.int16())})
        rows = _convert(batch)
        assert rows[0][0] == -32768
        assert rows[1][0] == 32767

    def test_int32_boundaries(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([-2**31, 2**31 - 1], type=pa.int32())})
        rows = _convert(batch)
        assert rows[0][0] == -2**31
        assert rows[1][0] == 2**31 - 1

    def test_int64_boundaries(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([-2**63, 2**63 - 1], type=pa.int64())})
        rows = _convert(batch)
        assert rows[0][0] == -2**63
        assert rows[1][0] == 2**63 - 1

    def test_uint8_boundaries(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([0, 255], type=pa.uint8())})
        rows = _convert(batch)
        assert rows[0][0] == 0
        assert rows[1][0] == 255

    def test_uint16_boundaries(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([0, 65535], type=pa.uint16())})
        rows = _convert(batch)
        assert rows[0][0] == 0
        assert rows[1][0] == 65535

    def test_uint32_boundaries(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([0, 2**32 - 1], type=pa.uint32())})
        rows = _convert(batch)
        assert rows[0][0] == 0
        assert rows[1][0] == 2**32 - 1


# ===========================================================================
# 10. Binary edge cases
# ===========================================================================

class TestBinaryEdgeCases:
    """Edge cases for Binary columns."""

    def test_empty_binary(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([b""], type=pa.binary())})
        rows = _convert(batch)
        assert rows[0][0] == b""

    def test_binary_all_zeros(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([b"\x00\x00\x00"], type=pa.binary())})
        rows = _convert(batch)
        assert rows[0][0] == b"\x00\x00\x00"
        assert len(rows[0][0]) == 3

    def test_binary_all_ff(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([b"\xff" * 256], type=pa.binary())})
        rows = _convert(batch)
        assert rows[0][0] == b"\xff" * 256

    def test_large_binary(self):
        """Large binary data (1MB)."""
        val = bytes(range(256)) * 4096  # 1MB
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([val], type=pa.large_binary())})
        rows = _convert(batch)
        assert rows[0][0] == val


# ===========================================================================
# 11. Multi-column interactions
# ===========================================================================

class TestMultiColumnInteractions:
    """Corner cases involving multiple columns in the same row."""

    def test_mixed_null_and_non_null(self):
        """Row where some columns are NULL and others are not."""
        batch = pa.RecordBatch.from_pydict({
            "a": pa.array([None], type=pa.int64()),
            "b": pa.array(["hello"], type=pa.string()),
            "c": pa.array([None], type=pa.list_(pa.int32())),
            "d": pa.array([3.14], type=pa.float64()),
        })
        rows = _convert(batch)
        assert rows[0] == (None, "hello", None, 3.14)

    def test_all_columns_null_single_row(self):
        """Single row where every column is NULL."""
        batch = pa.RecordBatch.from_pydict({
            "a": pa.array([None], type=pa.int64()),
            "b": pa.array([None], type=pa.string()),
            "c": pa.array([None], type=pa.float64()),
            "d": pa.array([None], type=pa.bool_()),
        })
        rows = _convert(batch)
        assert rows[0] == (None, None, None, None)

    def test_selected_columns_ordering(self):
        """Column selection should return data in requested order."""
        batch = pa.RecordBatch.from_pydict({
            "x": pa.array([1], type=pa.int32()),
            "y": pa.array([2], type=pa.int32()),
            "z": pa.array([3], type=pa.int32()),
        })
        rows = _convert(batch, selected_columns=["z", "x"])
        assert rows[0] == (3, 1)

    def test_many_typed_columns_multi_row(self):
        """8 columns of all major types x 5 rows, NULL columns differ per row."""
        batch = pa.RecordBatch.from_pydict({
            "c_int": pa.array([1, None, 3, None, 5], type=pa.int64()),
            "c_float": pa.array([None, 2.0, None, 4.0, None], type=pa.float64()),
            "c_bool": pa.array([True, None, False, None, True], type=pa.bool_()),
            "c_str": pa.array([None, "b", None, "d", None], type=pa.string()),
            "c_bin": pa.array([b"\x01", None, b"\x03", None, b"\x05"],
                              type=pa.binary()),
            "c_list": pa.array([None, [1, 2], None, [3], None],
                               type=pa.list_(pa.int32())),
            "c_map": pa.array(
                [[("a", 1)], None, [("c", 3)], None, [("e", 5)]],
                type=pa.map_(pa.string(), pa.int64())),
            "c_struct": pa.array(
                [{"x": 1}, None, {"x": None}, None, {"x": 5}],
                type=pa.struct([("x", pa.int32())])),
        })
        rows = _convert(batch)
        # row 0: int=1, float=None, bool=True, str=None, bin=\x01, list=None, map={"a":1}, struct={"x":1}
        assert rows[0] == (1, None, True, None, b"\x01", None, {"a": 1}, {"x": 1})
        # row 1: int=None, float=2.0, bool=None, str="b", bin=None, list=[1,2], map=None, struct=None
        assert rows[1] == (None, 2.0, None, "b", None, [1, 2], None, None)
        # row 2: int=3, float=None, bool=False, str=None, bin=\x03, list=None, map={"c":3}, struct={"x":None}
        assert rows[2] == (3, None, False, None, b"\x03", None, {"c": 3}, {"x": None})
        # row 3: int=None, float=4.0, bool=None, str="d", bin=None, list=[3], map=None, struct=None
        assert rows[3] == (None, 4.0, None, "d", None, [3], None, None)
        # row 4: int=5, float=None, bool=True, str=None, bin=\x05, list=None, map={"e":5}, struct={"x":5}
        assert rows[4] == (5, None, True, None, b"\x05", None, {"e": 5}, {"x": 5})


# ===========================================================================
# 12. Complex nested combinations
# ===========================================================================

class TestComplexNested:
    """Complex nested type combinations that stress the recursive converter."""

    def test_list_of_structs_with_nulls(self):
        """List<Struct> where some struct elements are NULL."""
        typ = pa.list_(pa.struct([("id", pa.int32()), ("val", pa.float64())]))
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array(
                [[{"id": 1, "val": 1.0}, None, {"id": 3, "val": 3.0}]],
                type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == [{"id": 1, "val": 1.0}, None, {"id": 3, "val": 3.0}]

    def test_map_with_list_values_containing_nulls(self):
        """Map<String, List<Int>> where list values contain NULLs."""
        typ = pa.map_(pa.string(), pa.list_(pa.int32()))
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([[("k", [1, None, 3])]], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"k": [1, None, 3]}

    def test_struct_with_null_list_and_null_map(self):
        """Struct where list field and map field are both NULL."""
        typ = pa.struct([
            ("ids", pa.list_(pa.int64())),
            ("props", pa.map_(pa.string(), pa.string())),
            ("name", pa.string()),
        ])
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([{"ids": None, "props": None, "name": "test"}],
                           type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"ids": None, "props": None, "name": "test"}

    def test_list_of_maps(self):
        """List<Map<String, Int>>."""
        typ = pa.list_(pa.map_(pa.string(), pa.int32()))
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array(
                [[[("a", 1), ("b", 2)], [("c", 3)]]],
                type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == [{"a": 1, "b": 2}, {"c": 3}]

    def test_empty_nested_containers(self):
        """All nested containers are empty but not NULL."""
        typ = pa.struct([
            ("list_f", pa.list_(pa.int32())),
            ("map_f", pa.map_(pa.string(), pa.int32())),
        ])
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([{"list_f": [], "map_f": []}], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"list_f": [], "map_f": {}}

    def test_list_of_structs_multi_row(self):
        """List<Struct> across 6 rows with varying NULL at element/field/row level."""
        typ = pa.list_(pa.struct([("id", pa.int32()), ("val", pa.float64())]))
        batch = pa.RecordBatch.from_pydict({"v": pa.array([
            [{"id": 1, "val": 1.0}, {"id": 2, "val": 2.0}],
            [None, {"id": 3, "val": 3.0}],
            [],
            None,
            [{"id": 4, "val": None}],
            [{"id": None, "val": 5.0}, {"id": 6, "val": None}, None],
        ], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == [{"id": 1, "val": 1.0}, {"id": 2, "val": 2.0}]
        assert rows[1][0] == [None, {"id": 3, "val": 3.0}]
        assert rows[2][0] == []
        assert rows[3][0] is None
        assert rows[4][0] == [{"id": 4, "val": None}]
        assert rows[5][0] == [{"id": None, "val": 5.0}, {"id": 6, "val": None}, None]

    def test_map_nested_values_multi_row(self):
        """Map<String, List<Int32>> across 5 rows with varying NULL patterns."""
        typ = pa.map_(pa.string(), pa.list_(pa.int32()))
        batch = pa.RecordBatch.from_pydict({"v": pa.array([
            [("a", [1, 2]), ("b", [3])],
            [("c", None)],
            [],
            None,
            [("d", [None, 4]), ("e", [])],
        ], type=typ)})
        rows = _convert(batch)
        assert rows[0][0] == {"a": [1, 2], "b": [3]}
        assert rows[1][0] == {"c": None}
        assert rows[2][0] == {}
        assert rows[3][0] is None
        assert rows[4][0] == {"d": [None, 4], "e": []}

    def test_full_complex_multi_column_multi_row(self):
        """4 complex-type columns x 6 rows: every row has a different NULL pattern."""
        list_struct_type = pa.list_(
            pa.struct([("id", pa.int32()), ("val", pa.string())]))
        map_list_type = pa.map_(pa.string(), pa.list_(pa.int32()))
        struct_map_type = pa.struct([
            ("name", pa.string()),
            ("props", pa.map_(pa.string(), pa.string())),
        ])
        nested_list_type = pa.list_(pa.list_(pa.int64()))

        batch = pa.RecordBatch.from_pydict({
            "col_ls": pa.array([
                [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}],
                [None, {"id": 3, "val": "c"}],
                None,
                [],
                [{"id": None, "val": "d"}, {"id": 5, "val": None}],
                None,
            ], type=list_struct_type),
            "col_ml": pa.array([
                [("k1", [1, 2]), ("k2", [3])],
                None,
                [("k3", None)],
                [],
                [("k4", []), ("k5", [None, 4])],
                None,
            ], type=map_list_type),
            "col_sm": pa.array([
                {"name": "n1", "props": [("color", "red")]},
                {"name": "n2", "props": None},
                None,
                {"name": None, "props": None},
                {"name": "n3", "props": [("size", "big"), ("weight", "heavy")]},
                None,
            ], type=struct_map_type),
            "col_nl": pa.array([
                [[1, 2], [3, 4, 5]],
                [None, [6]],
                [],
                None,
                [[None, 7], []],
                None,
            ], type=nested_list_type),
        })
        rows = _convert(batch)
        # row 0: all columns have full values
        assert rows[0] == (
            [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}],
            {"k1": [1, 2], "k2": [3]},
            {"name": "n1", "props": {"color": "red"}},
            [[1, 2], [3, 4, 5]],
        )
        # row 1: mixed None at various levels
        assert rows[1] == (
            [None, {"id": 3, "val": "c"}],
            None,
            {"name": "n2", "props": None},
            [None, [6]],
        )
        # row 2: col_ls=None, col_ml has None value, col_sm=None, col_nl=[]
        assert rows[2] == (None, {"k3": None}, None, [])
        # row 3: empty containers and all-None struct fields
        assert rows[3] == ([], {}, {"name": None, "props": None}, None)
        # row 4: partial None in nested fields
        assert rows[4] == (
            [{"id": None, "val": "d"}, {"id": 5, "val": None}],
            {"k4": [], "k5": [None, 4]},
            {"name": "n3", "props": {"size": "big", "weight": "heavy"}},
            [[None, 7], []],
        )
        # row 5: all columns None
        assert rows[5] == (None, None, None, None)


# ===========================================================================
# 13. Timestamp types
# ===========================================================================

class TestTimestampValues:
    """TIMESTAMP type corner cases (ms and ns units)."""

    def test_timestamp_ms_epoch(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([0], type=pa.timestamp("ms"))})
        rows = _convert(batch)
        assert rows[0][0] == 0
        assert isinstance(rows[0][0], int)

    def test_timestamp_ms_typical(self):
        """2020-01-01T00:00:00 in milliseconds."""
        val = 1577836800000
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([val], type=pa.timestamp("ms"))})
        rows = _convert(batch)
        assert rows[0][0] == val

    def test_timestamp_ms_negative(self):
        """Before epoch."""
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([-1000], type=pa.timestamp("ms"))})
        rows = _convert(batch)
        assert rows[0][0] == -1000

    def test_timestamp_null(self):
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array([1577836800000, None, 0], type=pa.timestamp("ms"))})
        rows = _convert(batch)
        assert rows[0][0] == 1577836800000
        assert rows[1][0] is None
        assert rows[2][0] == 0


    def test_timestamp_multi_row_mixed_nulls(self):
        """Multiple rows with ms timestamps and varying NULL positions."""
        values = [1577836800000, None, 0, None, -86400000, 1609459200000]
        batch = pa.RecordBatch.from_pydict(
            {"v": pa.array(values, type=pa.timestamp("ms"))})
        rows = _convert(batch)
        for i, expected in enumerate(values):
            if expected is None:
                assert rows[i][0] is None
            else:
                assert rows[i][0] == expected
