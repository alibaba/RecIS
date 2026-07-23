# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy
from typing import List, Tuple

from column_io.lib import interface
from column_io.dataset.log_util import logger
from column_io.dataset.odps_env_setup import (
    ensure_standard_path_format,
    init_odps_open_storage_session,
)
from odps import types as odps_types
from column_io.dataset.open_storage_row_reader import (
    DefaultReadBatch,
    Distributor,
    OpenstorageClient,
    OutOfRangeException,
    TableSchemaType,
    _try_get_table_range,
)

# Extended type mapping for V2. Strictly aligned with the Arrow types that
# C++ ArrowCellToPyObject actually handles (see odps_open_storage_row_dataset.cc).
odps_type_to_pytype_v2 = {
    # ---- Boolean (Arrow BOOL) ----
    odps_types.Boolean:   TableSchemaType(bool, "boolean"),
    odps_types.Tinyint:   TableSchemaType(int, "tinyint"),
    odps_types.Smallint:  TableSchemaType(int, "smallint"),
    odps_types.Int:       TableSchemaType(int, "int"),
    odps_types.Bigint:    TableSchemaType(int, "bigint"),
    odps_types.Float:     TableSchemaType(float, "float"),
    odps_types.Double:    TableSchemaType(float, "double"),
    odps_types.String:    TableSchemaType(object, "string"),
    odps_types.Binary:    TableSchemaType(object, "binary"),
    odps_types.Array:     TableSchemaType(object, "array"),
    odps_types.Map:       TableSchemaType(object, "map"),
    odps_types.Struct:    TableSchemaType(object, "struct"),
    odps_types.Datetime:  TableSchemaType(int, "datetime")
}

class OpenStorageRowReaderV2(OpenstorageClient):
    _internal = interface._OdpsOpenStorageRowDataset

    def __init__(
        self,
        table_name: str,
        selected_cols: str = "",
        excluded_cols: str = "",
        slice_id: int = 0,
        slice_count: int = 1,
        num_threads: int = 1,
        capacity: int = 2048,
        batch_size: int = DefaultReadBatch,
    ):
        logger.debug(
            "OpenStorageRowReaderV2 create, table=%s slice=%s/%s "
            "num_threads=%s capacity=%s batch_size=%s",
            table_name, slice_id, slice_count, num_threads, capacity, batch_size,
        )

        super(OpenStorageRowReaderV2, self).__init__(table_name)

        if slice_id < 0 or slice_id >= slice_count:
            raise ValueError(
                "slice_id and slice_count are invalid: {}, {}".format(
                    slice_id, slice_count))
        if num_threads < 0 or capacity <= 0:
            raise ValueError(
                "num_threads ({}) should be >=0 and capacity ({}) should be > 0"
                .format(num_threads, capacity))
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0, got {}".format(batch_size))

        self._odps_table_paths = [
            p for p in table_name.split(",") if p.startswith("odps://")
        ]
        if len(self._odps_table_paths) != 1:
            raise ValueError(
                "expected exactly 1 odps:// path, got {} from: {}".format(
                    len(self._odps_table_paths), table_name))
        self._is_close: bool = False
        self._batch_size: int = batch_size
        self._reader_cache: List[Tuple] = []

        self._schema_all_column = None
        self._get_schema_all_column_from_odps()

        self._select_column: List[str] = []
        self._schema_select_column: List[Tuple[str, type]] = []
        self._schema = None
        self._apply_select_columns(selected_cols, excluded_cols)

        # Open Storage session (creates / registers as appropriate).
        standard_paths = ensure_standard_path_format(self._odps_table_paths)
        init_odps_open_storage_session(
            standard_paths, required_data_columns=self._select_column)

        self._table_size: int = sum(
            self._internal.get_table_size(path)
            for path in self._odps_table_paths
        )
        start, end, has_explicit_range = _try_get_table_range(
            self._odps_table_paths[0])
        if has_explicit_range:
            self._start_pos: int = start
            self._end_pos: int = end
        else:
            dist = Distributor(self._table_size, slice_id, slice_count)
            self._start_pos = dist.start
            self._end_pos = dist.end
        self._offset_pos: int = self._start_pos

        # Build the underlying dataset on the (possibly sliced) path. The C++
        # class accepts a list of paths; here we always feed it
        # exactly one entry that already encodes the slice via `?start=&end=`.
        self._reader = self._build_reader_for_range(
            self._start_pos, self._end_pos)

        self._batch_size = min(
            self._batch_size, max(1, self._end_pos - self._start_pos))
        logger.debug(
            "OpenStorageRowReaderV2 init done, batch_size=%d range=[%d,%d)",
            self._batch_size, self._start_pos, self._end_pos)

    def _get_schema_all_column_from_odps(self):
        tmp_schema = []
        table = self._odps.get_table(self._odps_table_names[0])
        columns = table.table_schema.columns
        self._odps_partitions = table.table_schema.partitions
        for col in columns:
            mapping = odps_type_to_pytype_v2.get(type(col.type))
            if mapping is None:
                logger.warning(
                    "unknown odps type %s for column %s, falling back to string/object",
                    type(col.type).__name__, col.name)
                col_type_name = "string"
                col_py_type = object
            else:
                col_type_name = mapping.typestr
                col_py_type = mapping.pytype
            tmp_schema.append((col.name, col_type_name, col_py_type))
        self._schema_all_column = numpy.array(
            tmp_schema,
            dtype=[("colname", object), ("typestr", object), ("pytype", type)])

    def _apply_select_columns(self, selected_cols: str, excluded_cols: str):
        partition_names = [p.name for p in self._odps_partitions]
        if selected_cols and excluded_cols:
            raise ValueError(
                "selected_cols and excluded_cols cannot both be set")
        all_cols = self._schema_all_column["colname"].tolist()

        if not selected_cols and not excluded_cols:
            self._select_column = [c for c in all_cols if c not in partition_names]
        elif selected_cols:
            requested = [c.strip() for c in selected_cols.split(",") if c.strip()]
            seen = set()
            resolved = []
            for col in requested:
                if col in seen:
                    logger.debug("selected_cols %s duplicated, skipping", col)
                    continue
                if col not in all_cols:
                    logger.warning(
                        "selected_cols %s is invalid for %s/%s",
                        col, self._odps_project, self._odps_table_names[0])
                    continue
                if col in partition_names:
                    logger.warning(
                        "selected_cols %s is partition column, skipping", col)
                    continue
                seen.add(col)
                resolved.append(col)
            self._select_column = resolved
        else:  # excluded_cols only
            excluded_set = {c.strip() for c in excluded_cols.split(",") if c.strip()}
            excluded_set.update(partition_names)
            self._select_column = [c for c in all_cols if c not in excluded_set]

        col_lookup = {col[0]: col[2] for col in self._schema_all_column}
        self._schema_select_column = [
            (name, col_lookup[name]) for name in self._select_column
        ]
        rm_index_list = [
            i for i, col in enumerate(self._schema_all_column)
            if col[0] not in self._select_column
        ]
        self._schema = numpy.delete(self._schema_all_column, rm_index_list, 0)
        logger.debug(
            "OpenStorageRowReaderV2 columns resolved: %s", self._select_column)

    def _build_reader_for_range(self, start: int, end: int):
        """Create a fresh ``_OdpsOpenStorageRowDataset`` covering [start,end).

        The C++ class is multi-path capable (it walks ``file_cur_`` through
        the supplied list and transparently advances on ``OutOfRange``). We
        only ever feed it a single path with the slice range baked into the
        URL so that the open-storage session lookup remains stable.
        """
        base_path = self._odps_table_paths[0].split("?")[0]
        sliced_path = "{}?start={}&end={}".format(base_path, start, end)
        reader_name = "OpenStorageRowReaderV2[{}]".format(
            self._odps_table_names[0])
        return self._internal.make(
            [sliced_path],
            self._select_column,
            self._batch_size,
            reader_name,
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._reader_cache = []
        self._reader = None
        self._is_close = True

    def _check_status(self):
        if self._is_close:
            raise Exception("Table is closed!")

    def get_row_count(self) -> int:
        self._check_status()
        return self._end_pos - self._start_pos

    def get_schema(self):
        self._check_status()
        return self._schema

    @property
    def start_pos(self) -> int:
        self._check_status()
        return self._start_pos

    @property
    def end_pos(self) -> int:
        self._check_status()
        return self._end_pos

    @property
    def offset_pos(self) -> int:
        self._check_status()
        return self._offset_pos

    @property
    def selected_columns(self) -> List[str]:
        self._check_status()
        return list(self._select_column)

    def seek(self, offset: int):
        """Seek to an absolute row offset within ``[start_pos, end_pos)``."""
        self._check_status()
        if offset < self._start_pos or offset >= self._end_pos:
            raise ValueError(
                "offset:{} out of valid range:[{},{})".format(
                    offset, self._start_pos, self._end_pos))
        self._reader.seek(offset)
        self._reader_cache = []
        self._offset_pos = offset

    def save_state(self) -> str:
        self._check_status()
        return self._reader.save_state()

    def restore_state(self, state: str):
        self._check_status()
        self._reader.restore_state(state)
        self._reader_cache = []
        # Keep the user-facing offset in sync with the underlying reader.
        self._offset_pos = self._reader.tell()

    _MAX_READ_RETRIES = 3

    def _rebuild_reader(self):
        """Rebuild the underlying C++ reader from current offset."""
        self._reader = self._build_reader_for_range(
            self._offset_pos, self._end_pos)
        self._reader_cache = []

    def read(self, num_records=1, allow_smaller_final_batch=False,
             to_ndarray=False):
        self._check_status()
        if num_records <= 0:
            raise ValueError("num_records must be > 0")

        records: List = self._reader_cache
        self._reader_cache = []
        retries = 0

        while len(records) < num_records:
            try:
                batch = self._reader.read_batch()
                if not batch:
                    if records:
                        break
                    raise OutOfRangeException(
                        "End of table reached at pos:{}/{}".format(
                            self._offset_pos, self._end_pos))
                self._offset_pos += len(batch)
                records.extend(batch)
                retries = 0
            except OutOfRangeException:
                raise
            except RuntimeError as e:
                retries += 1
                if retries > self._MAX_READ_RETRIES:
                    raise
                logger.warning(
                    "read_batch failed (retry %d/%d): %s",
                    retries, self._MAX_READ_RETRIES, e)
                time.sleep(1)
                self._rebuild_reader()

        if len(records) > num_records:
            self._reader_cache = records[num_records:]
            records = records[:num_records]
        if len(records) < num_records and not allow_smaller_final_batch:
            raise OutOfRangeException(
                "End of table reached: returned {} < requested {} at pos:{}/{}"
                .format(len(records), num_records, self._offset_pos,
                        self._end_pos))
        if to_ndarray:
            return numpy.array(records, dtype=self._schema_select_column)
        return records
