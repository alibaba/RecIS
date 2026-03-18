import os

import torch

from recis.io.dataset_base import DatasetBase
from recis.io.odps_dataset import (
    get_table_size as odps_get_table_size,
    init_odps_session,
)
from recis.utils.logger import Logger


logger = Logger(__name__)

try:
    # In BUILD_DOCUMENT mode, NO code will be really executed. So libs in IO needn't be imported
    if os.environ.get("BUILD_DOCUMENT", None) == "1":
        raise ImportError("column_io is not installed")

    from column_io.dataset.dataset import Dataset as ColumnIO_Dataset
    from column_io.dataset.file_sharding import OdpsComboTableSharding
except ImportError:
    pass


# TODO: 抽取Combo base class, 用于之后的lake
class OdpsComboDataset(DatasetBase):
    """ODPS Dataset for reading Open Data Processing Service tables.

    This class provides functionality to read ODPS tables efficiently with support for
    both sparse (variable-length) and dense (fixed-length) features. It extends
    DatasetBase to provide ODPS-specific optimizations including hash feature processing,
    table sharding, and batch processing.

    The OdpsDataset automatically detects and uses ODPS Open Storage when available,
    which provides better performance for large-scale data processing. It supports
    distributed training by allowing multiple workers to process different shards
    of the data concurrently.

    Attributes:
        hash_types (List[str]): List of hash algorithms used for features.
        hash_buckets (List[int]): List of hash bucket sizes for features.
        hash_features (List[str]): List of feature names that use hashing.

    Example:
        Creating and configuring an ODPS dataset:

        ```python
        # Initialize dataset
        dataset = OdpsDataset(
            batch_size=512, worker_idx=0, worker_num=4, shuffle=True, ragged_format=True
        )

        # Add ODPS tables
        dataset.add_paths(
            ["recommendation.user_features", "recommendation.item_features"]
        )

        # Configure sparse features with hashing
        dataset.varlen_feature(
            "user_clicked_items", hash_type="farm", hash_bucket=1000000
        )
        dataset.varlen_feature("item_categories", hash_type="murmur", hash_bucket=10000)

        # Configure dense features
        dataset.fixedlen_feature("user_age", default_value=25.0)
        dataset.fixedlen_feature("item_price", default_value=0.0)
        ```
    """

    def __init__(
        self,
        batch_size,
        worker_idx=0,
        worker_num=1,
        combo_group_size=0,
        read_threads_num=4,
        pack_threads_num=None,
        prefetch=1,
        is_compressed=False,
        drop_remainder=False,
        worker_slice_batch_num=None,
        shuffle=False,
        ragged_format=True,
        transform_fn=None,
        save_interval=100,
        check_data=True,
        primary_key="",
        dtype=torch.float32,
        device="cpu",
        prefetch_transform=None,
        user_define_module=None,
        read_batch_size=None,
    ) -> None:
        """Initialize ComboDataset with configuration parameters.

        Args:
            batch_size (int): Number of samples per batch.
            worker_idx (int, optional): Index of current worker. Defaults to 0.
            worker_num (int, optional): Total number of workers. Defaults to 1.
            read_threads_num (int, optional): Number of reading threads. Defaults to 4.
            pack_threads_num (int, optional): Number of packing threads. Defaults to None.
            prefetch (int, optional): Number of batches to prefetch. Defaults to 1.
            is_compressed (bool, optional): Whether data is compressed. Defaults to False.
            drop_remainder (bool, optional): Whether to drop incomplete batches. Defaults to False.
            worker_slice_batch_num (int, optional): Number of batches per worker slice. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            ragged_format (bool, optional): Whether to use ragged tensor format. Defaults to True.
            transform_fn (callable, optional): Data transformation function. Defaults to None.
            save_interval (int, optional): Interval for saving checkpoints. Defaults to 100.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
            device (str, optional): Device for tensor operations. Defaults to "cpu".
            prefetch_transform (int, optional): Number of batches to prefetch for transform. Defaults to None. A python thread will be used to prefetch data.
            user_define_module (callable, optional): User-defined module for data processing. Defaults to None.
            read_batch_size (int, optional): Read batch size set for source dataset, if not specified, batch_size will be used.

        Note:
            The dataset automatically detects ODPS Open Storage availability and
            configures the appropriate backend for optimal performance.
        """
        if not isinstance(combo_group_size, int) or combo_group_size < 2:
            raise ValueError("combo_group_size must be int and larger than 1")

        super().__init__(
            batch_size,
            worker_idx=worker_idx,
            worker_num=worker_num,
            read_threads_num=read_threads_num,
            pack_threads_num=pack_threads_num,
            prefetch=prefetch,
            is_compressed=is_compressed,
            drop_remainder=drop_remainder,
            worker_slice_batch_num=worker_slice_batch_num,
            ragged_format=ragged_format,
            transform_fn=transform_fn,
            save_interval=save_interval,
            dtype=dtype,
            device=device,
            prefetch_transform=prefetch_transform,
            user_define_module=user_define_module,
            read_batch_size=read_batch_size,
        )
        self._shuffle = shuffle
        self._total_row_count = 0
        self.hash_types = []
        self.hash_buckets = []
        self.hash_features = []
        self._table_ids = []
        self._combo_group_size = combo_group_size
        self._paths: list[list[str]] = []  # not extend list[str] from DatasetBase
        self._start_groups: list[list[int]] = []
        self._end_groups: list[list[int]] = []
        self._table_sizes: list[list[int]] = []
        self._primary_key = primary_key
        self._check_data = check_data
        self._dataset_batch_size = batch_size

    @classmethod
    def _parse_input_path(cls, path: str) -> tuple[str, int, int]:
        """Parse the full input path into base_path and start_offset+end_offset(optional).
        Args:
            path: The input path. E.g. odps://project_foo/tables/table_bar/part=p1?start=0&end=100

        Returns:
            path: The parsed path. E.g. odps://project_foo/tables/table_bar/part=p1
            begin: The begin offset. E.g. 0
            end: The end offset. E.g. 100
        """
        assert "odps" in path, f"input path must be one odps path, but get {path}"
        begin, end = None, None
        if "?" not in path:
            return path, begin, end

        path_query = path.split("?")
        if len(path_query) != 2:
            raise ValueError(f"Add path error, input_path prefix invalid: {path_query}")
        path = path_query[0]
        kvs = path_query[1].split("&")
        if len(kvs) != 2:
            raise ValueError(
                f"Add path error, input_path offset invalid: {path_query[1]}, kv is {kvs}"
            )
        for idx in kvs:
            kv = idx.split("=")
            if len(kv) != 2:
                raise ValueError(
                    f"Add path error, input_path offset invalid: {path_query[1]}, kv is {kv}"
                )
            if kv[0] == "start":
                begin = int(kv[1])
            elif kv[0] == "end":
                end = int(kv[1])
        return path, begin, end

    def add_path(self, table_group: list[str]):
        """Add a table group

        Args:
        table_group: A list of string. The table group.
        """
        if len(table_group) != self._combo_group_size:
            raise ValueError(
                f"Number of tables in a table group not correct. Expect {self._combo_group_size}, get {len(table_group)}."
            )
        start_group = []
        end_group = []
        path_group = []
        for table_path in table_group:
            path, begin, end = self._parse_input_path(table_path)
            path_group.append(path)
            start_group.append(begin)
            end_group.append(end)
        if len(set(start_group)) > 1 or len(set(end_group)) > 1:
            raise ValueError(
                f"Add path error: tables offset unaligned, `{table_group}` is not valid table group"
            )
        self._paths.append(path_group)
        self._start_groups.append(start_group)
        self._end_groups.append(end_group)

    def add_paths(self, table_groups: list[list[str]]):
        """Add a list of table groups.

        Args:
        table_groups: A list of table_group. The table groups, ordered by groups.
        """
        if not isinstance(table_groups, (list, tuple)) or any(
            not isinstance(x, (list, tuple)) for x in table_groups
        ):
            raise ValueError("Add path error: table_groups should be a list of lists")
        for table_group in table_groups:
            self.add_path(table_group)

    def varlen_feature(
        self, name: str, hash_type=None, hash_bucket=0, trans_int8=False, table_id=-1
    ):
        """Configure a variable-length (sparse) feature with optional hashing.

        Variable-length features are columns that contain sequences or lists of values
        with varying lengths across samples. These features can optionally be processed
        with hash functions for dimensionality reduction and categorical encoding.

        Args:
            name (str): Name of the feature column in the ODPS tables.
            hash_type (str, optional): Hash algorithm to use for the feature.
                Supported values are "farm" (FarmHash) and "murmur" (MurmurHash).
                If None, no hashing is applied. Defaults to None.
            hash_bucket (int, optional): Size of the hash bucket (vocabulary size).
                Only used when hash_type is specified. Defaults to 0.
            trans_int8 (bool, optional): Whether to convert string data directly to
                int8 tensors without hashing. Only effective when hash_type is None.
                Defaults to False.

        Example:
            ```python
            # Sparse feature with FarmHash for large vocabularies
            dataset.varlen_feature(
                "user_clicked_items", hash_type="farm", hash_bucket=1000000
            )

            # Sparse feature with MurmurHash for smaller vocabularies
            dataset.varlen_feature(
                "item_categories", hash_type="murmur", hash_bucket=50000
            )

            # Raw sparse feature without hashing (for pre-processed IDs)
            dataset.varlen_feature("user_behavior_sequence")

            # String feature converted to int8 (for text processing)
            dataset.varlen_feature("review_tokens", trans_int8=True)
            ```

        Raises:
            AssertionError: If hash_type is not "farm" or "murmur" when specified.

        Note:
            Hash functions are useful for handling large categorical vocabularies
            by mapping them to a fixed-size space. FarmHash generally provides
            better distribution properties, while MurmurHash is faster for smaller
            vocabularies.
        """
        if name in self._select_column:
            return

        self._select_column.append(name)
        self._table_ids.append(table_id)
        if hash_type:
            assert hash_type in [
                "farm",
                "murmur",
            ], "hash_type must be farm / murmur"
            self.hash_features.append(name)
            self.hash_buckets.append(hash_bucket)
            self.hash_types.append(hash_type)
        elif trans_int8:
            self.hash_features.append(name)
            self.hash_buckets.append(hash_bucket)
            self.hash_types.append("no_hash")

    def fixedlen_feature(self, name: str, default_value: float, table_id=-1):
        """Configure a fixed-length (dense) feature.

        Fixed-length features are typically used for numerical data where each
        sample has exactly one value, such as user age, item price, or ratings.

        Args:
            name (str): Name of the feature column in the ODPS table.
            default_value (float): Default value to use when feature is missing.

        Example:
            ```python
            # Numerical features with default values
            dataset.fixedlen_feature("user_age", default_value=25.0)
            dataset.fixedlen_feature("item_price", default_value=0.0)
            dataset.fixedlen_feature("rating", default_value=3.0)
            ```

        Note:
            Default values are important for handling missing data gracefully
            and ensuring consistent tensor shapes across batches.
        """
        if name not in self._select_column:
            self._select_column.append(name)
            self._table_ids.append(table_id)
        if name not in self._dense_column:
            self._dense_column.append(name)
            self._dense_default_value.append(default_value)

    def _shard_path(self, sub_id: int, sub_num: int):
        """Create table shards for distributed processing.

        This method partitions the input ODPS tables across multiple workers and threads
        to enable parallel data loading. It uses OdpsTableSharding to ensure
        balanced distribution of data and initializes ODPS Open Storage session
        when available.

        Args:
            sub_id (int): Sub-process identifier within the worker.
            sub_num (int): Total number of sub-processes per worker.

        Note:
            This is an internal method used by the dataset creation process.
            When ODPS Open Storage is available, it initializes the session
            with standardized paths and required columns for optimal performance.
        """
        combo_paths = [path for group in self._paths for path in group]
        init_odps_session(combo_paths, select_column=self._select_column)
        file_shard = OdpsComboTableSharding()
        file_shard.add_paths(self._paths, self._start_groups, self._end_groups)
        self._shard_paths = file_shard.partition(
            self._worker_idx * sub_num + sub_id,
            self._worker_num * sub_num,
            self._read_threads_num,
            slice_size=(
                self._worker_slice_batch_num * self._batch_size
                if self._worker_slice_batch_num
                else None
            ),
            shuffle=self._shuffle,
        )
        logger.debug(f"odps combo::_shard_paths get {self._shard_paths}")

    def make_dataset_fn(self):
        return lambda table_group: ColumnIO_Dataset.from_odps_combo_source(
            [
                [
                    table.decode("utf-8") if isinstance(table, bytes) else table
                    for table in table_group
                ]
            ],
            self._combo_group_size,
            self._is_compressed,
            self._batch_size,
            self._select_column,
            self.hash_features,
            self.hash_types,
            self.hash_buckets,
            self._dense_column,
            self._dense_default_value,
            self._table_ids,
            self._check_data,
            self._primary_key,
        )

    def get_table_size(self):
        """Calculate and return the sizes of all configured ODPS tables.

        This method iterates through all added ODPS tables and retrieves their sizes,
        updating the internal tracking of total row count for the dataset.
        When ODPS Open Storage is available, it initializes the session before
        querying table sizes.

        Returns:
            List[int]: List of table sizes (number of rows) corresponding to each table.

        Example:
            ```python
            dataset.add_paths(["project.table1", "project.table2"])
            sizes = dataset.get_table_size()
            print(f"Table sizes: {sizes}")
            ```

        Note:
            This method may take some time to execute as it queries the ODPS
            service for actual table statistics.
        """
        # maybe do it in column_io ?
        # NOTE: hechen do it here. but I think leave it in columnIO dataset init is simpler.
        combo_paths = [path for group in self._paths for path in group]
        init_odps_session(combo_paths, select_column=self._select_column)

        self._table_sizes = [[0] * self._combo_group_size for _ in self._paths]
        self._total_row_count = 0
        for i, table_group in enumerate(self._paths):
            # for table_name in table_group:
            for j, table_name in enumerate(table_group):
                table_size = odps_get_table_size(table_name)
                self._table_sizes[i][j] = table_size
            self._total_row_count += table_size
        return self._table_sizes
