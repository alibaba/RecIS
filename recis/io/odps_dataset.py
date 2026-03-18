import os

import torch

from recis.io.dataset_base import DatasetBase
from recis.utils.logger import Logger


logger = Logger(__name__)


def get_table_size(table_name: str) -> int:
    """Get the size (number of rows) of an ODPS table.

    Args:
        table_name (str): The name of the ODPS table in format 'project.table'.

    Returns:
        int: The number of rows in the specified table.

    Example:

    .. code-block:: python

        size = get_table_size("my_project.user_behavior_table")
        print(f"Table has {size} rows")

    """
    raise NotImplementedError


def init_odps_session(paths: list[str], select_column: list[str]):
    """
    Initialize ODPS session

    Args:
        paths: list of paths
        select_column: list of column_name to read

    Note:
        The session contains two type of information:
            1. authorization
            2. odps data location

        The session could be create by three internal ways, thus three types of session:
            1. odps-algo-sdk
            2. odps-open-storage-direct
            3. odps-open-storage-tunnel
        And each type of session are NOT ABLE TO SHARE INFOMATION with other types.

        The session WILL NOT be created by every worker, only the creator of whole task could do it.
        So if the creator does not know the new dataset to be read, we need to call this function forcely.
    """
    raise NotImplementedError


try:
    if os.environ.get("BUILD_DOCUMENT", None) == "1":
        # In BUILD_DOCUMENT mode, NO code will be really executed. So libs in IO needn't be imported
        raise ImportError("column_io is needn't in BUILD_DOCUMENT mode")

    import column_io.dataset.dataset as column_io_dataset
    from column_io.dataset.file_sharding import OdpsTableSharding
    from column_io.dataset.odps_env_setup import (
        ensure_standard_path_format,
        init_odps_open_storage_session,
        is_turn_on_odps_open_storage,
    )

    # Automatically select the appropriate ODPS dataset backend
    if is_turn_on_odps_open_storage():
        odps_dataset_func = column_io_dataset.OdpsOpenStorageDataset
    else:
        odps_dataset_func = column_io_dataset.OdpsTableColumnDataset
    odps_dataset_func.load_plugin()

    def get_table_size(table_name):
        table_size = odps_dataset_func.get_table_size(table_name)
        return table_size

    def init_odps_session(paths: list[str], select_column: list[str]):
        if not is_turn_on_odps_open_storage():
            return  # TODO: refresh_odps_io_config() for  odps algo-sdk
        standard_paths = ensure_standard_path_format(paths)
        init_odps_open_storage_session(
            standard_paths, required_data_columns=select_column
        )

except ImportError:
    pass


class OdpsDataset(DatasetBase):
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

    .. code-block:: python

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

    """

    def __init__(
        self,
        batch_size,
        worker_idx=0,
        worker_num=1,
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
        dtype=torch.float32,
        device="cpu",
        prefetch_transform=None,
        user_define_module=None,
        read_batch_size=None,
    ) -> None:
        """Initialize OdpsDataset with configuration parameters.

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
        super().__init__(
            batch_size,
            worker_idx,
            worker_num,
            read_threads_num,
            pack_threads_num,
            prefetch,
            is_compressed,
            drop_remainder,
            worker_slice_batch_num,
            ragged_format,
            transform_fn,
            save_interval,
            dtype,
            device,
            prefetch_transform,
            user_define_module,
            read_batch_size,
        )
        self._shuffle = shuffle
        self._table_sizes = []
        self._total_row_count = 0

    def add_path(self, odps_table):
        """Add a single ODPS table to the dataset.

        Args:
            odps_table (str): ODPS table name in format 'project.table'.

        Example:

        .. code-block:: python

            dataset.add_path("my_project.user_behavior_table")

        """
        self._paths.append(odps_table)
        self._table_sizes.append(0)

    def add_paths(self, odps_tables):
        """Add multiple ODPS tables to the dataset.

        Args:
            odps_tables (List[str]): List of ODPS table names in format 'project.table'.

        Example:

        .. code-block:: python

            dataset.add_paths(
                [
                    "recommendation.user_features",
                    "recommendation.item_features",
                    "recommendation.interaction_logs",
                ]
            )

        """
        for table in odps_tables:
            self.add_path(table)

    def _shard_path(self, sub_id, sub_num):
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
        init_odps_session(paths=self._paths, select_column=self._select_column)
        file_shard = OdpsTableSharding()
        file_shard.add_paths(self._paths)
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

    def make_dataset_fn(self):
        """Create a dataset factory function for ODPS table processing.

        This method returns a lambda function that creates a column_io Dataset
        from ODPS tables with the configured features and processing parameters.

        Returns:
            callable: A function that takes a table name and returns a Dataset object.

        Note:
            The returned function is used internally by the data loading pipeline
            to create dataset instances for each shard of data. It automatically
            uses the appropriate ODPS backend (Open Storage or traditional).
        """
        return lambda x: column_io_dataset.Dataset.from_odps_source(
            [x.decode() if isinstance(x, bytes) else x],
            self._is_compressed,
            self._read_batch_size,
            self._select_column,
            self.hash_features,
            self.hash_types,
            self.hash_buckets,
            self._dense_column,
            self._dense_default_value,
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

        .. code-block:: python

            dataset.add_paths(["project.table1", "project.table2"])
            sizes = dataset.get_table_size()
            print(f"Table sizes: {sizes}")


        Note:
            This method may take some time to execute as it queries the ODPS
            service for actual table statistics.
        """
        init_odps_session(paths=self._paths, select_column=self._select_column)
        for i, table_name in enumerate(self._paths):
            table_size = get_table_size(table_name)
            self._table_sizes[i] = table_size
            self._total_row_count += table_size
        return self._table_sizes
