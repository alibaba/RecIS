import os
import time
from datetime import datetime
from typing import Optional, Union

import torch

from recis.info import is_internal_enabled
from recis.io.dataset_base import DatasetBase


if not os.environ.get("BUILD_DOCUMENT", None) == "1":
    import column_io.dataset.dataset as column_io_dataset
    from column_io.dataset.file_sharding import LakeStreamSharding

    if is_internal_enabled():
        from column_io.dataset.config import (
            LakeBatchConfig as OriginalLakeBatchConfig,
            LakeConfig as OriginalLakeConfig,
        )

        class LakeConfig:
            """Lake Config for reading streaming data from Lake sources.

            Attributes:
                storageName (str): Storage name.
                projectName (str): Project name.
                tableName (str): Table name.
                columnFamilyName (str): Column family name.
                partitionSpec (str): Partition specification.
                startTime (int): Start time.
                endTime (int): End time.
                useService (bool): Whether to use service.
                serviceName (str): Service name.
                batch_table (bool): Whether to use batch table.
            """

            def __init__(
                self,
                storageName: str,
                projectName: str,
                tableName: str,
                columnFamilyName: str,
                partitionSpec: str,
                startTime: Optional[Union[int, str]] = None,
                endTime: Optional[Union[int, str]] = None,
                useService: Optional[bool] = True,
                serviceName: Optional[str] = None,
                batch_table: Optional[bool] = False,
            ):
                if isinstance(startTime, str):
                    startTime = str2timestamp(startTime)
                if isinstance(endTime, str):
                    endTime = str2timestamp(endTime)
                if batch_table:
                    self.config = OriginalLakeBatchConfig(
                        storageName,
                        projectName,
                        tableName,
                        columnFamilyName,
                        partitionSpec,
                    )
                else:
                    self.config = OriginalLakeConfig(
                        storageName,
                        projectName,
                        tableName,
                        columnFamilyName,
                        partitionSpec,
                        startTime,
                        endTime,
                        useService,
                        serviceName,
                    )

            def get_v1_path(self):
                return self.config.get_v1_path()

            def get_v2_path(self, withTime=True):
                return self.config.get_v2_path(withTime)


def str2timestamp(s, format="%Y%m%d%H%M%S"):
    s = str(s)
    ts = time.mktime(datetime.strptime(s, format).timetuple())
    return int(ts) * 1000000


class LakeStreamDataset(DatasetBase):
    """Lake Stream Dataset for reading streaming data from Lake sources.

    This class provides functionality to read streaming data from Lake sources
    efficiently with support for time-range based data selection and configurable
    prefetching. It extends DatasetBase to provide Lake-specific optimizations
    including stream sharding and real-time data processing.

    The LakeStreamDataset supports distributed streaming by allowing multiple workers
    to process different shards of the stream concurrently. It provides flexible
    configuration for prefetching and buffering to optimize streaming performance.

    Attributes:
        _lake_use_prefetch (bool): Whether to enable prefetching for streams.
        _lake_prefetch_thread_num (int): Number of threads for prefetching.
        _lake_prefetch_buffer_size (int): Size of prefetch buffer.
        _begins (List[int]): List of begin timestamps for each stream.
        _ends (List[int]): List of end timestamps for each stream.

    Example:
        Creating and configuring a Lake stream dataset:

    .. code-block:: python

        # Initialize dataset with prefetching
        dataset = LakeStreamDataset(
            batch_size=512,
            worker_idx=0,
            worker_num=4,
            shuffle=True,
            shuffle_seed=42,
            lake_use_prefetch=True,
            lake_prefetch_thread_num=2,
            lake_prefetch_buffer_size=1024,
        )

        # Add stream sources with time ranges
        dataset.add_path("/lake/user_events", begin=1640995200, end=1641081600)
        dataset.add_path("/lake/item_updates", begin=1640995200, end=1641081600)

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
        shuffle=False,
        shuffle_seed=0,
        lake_use_prefetch=False,
        lake_prefetch_thread_num=1,
        lake_prefetch_buffer_size=1024,
        worker_slice_batch_num=500,
        ragged_format=True,
        transform_fn=None,
        save_interval=100,
        dtype=torch.float32,
        device="cpu",
        prefetch_transform=None,
        user_define_module=None,
    ) -> None:
        """Initialize LakeStreamDataset with configuration parameters.

        Args:
            batch_size (int): Number of samples per batch.
            worker_idx (int, optional): Index of current worker. Defaults to 0.
            worker_num (int, optional): Total number of workers. Defaults to 1.
            read_threads_num (int, optional): Number of reading threads. Defaults to 4.
            pack_threads_num (int, optional): Number of packing threads. Defaults to None.
            prefetch (int, optional): Number of batches to prefetch. Defaults to 1.
            is_compressed (bool, optional): Whether data is compressed. Defaults to False.
            drop_remainder (bool, optional): Whether to drop incomplete batches. Defaults to False.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            shuffle_seed (int, optional): Seed for shuffling randomization. Defaults to 0.
            lake_use_prefetch (bool, optional): Whether to enable Lake prefetching. Defaults to False.
            lake_prefetch_thread_num (int, optional): Number of prefetch threads. Defaults to 1.
            lake_prefetch_buffer_size (int, optional): Size of prefetch buffer. Defaults to 1024.
            worker_slice_batch_num (int, optional): Number of batches per worker slice. Defaults to 500.
            ragged_format (bool, optional): Whether to use ragged tensor format. Defaults to True.
            transform_fn (callable, optional): Data transformation function. Defaults to None.
            save_interval (int, optional): Interval for saving checkpoints. Defaults to 100.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
            device (str, optional): Device for tensor operations. Defaults to "cpu".
            prefetch_transform (int, optional): Number of batches to prefetch for transform. Defaults to None.

        Note:
            Lake-specific parameters (lake_use_prefetch, lake_prefetch_thread_num,
            lake_prefetch_buffer_size) are used to optimize streaming performance
            by enabling asynchronous data loading and buffering.
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
        )
        self._lake_use_prefetch = lake_use_prefetch
        self._lake_prefetch_thread_num = lake_prefetch_thread_num
        self._lake_prefetch_buffer_size = lake_prefetch_buffer_size
        self._begins = []
        self._ends = []
        self._shuffle = shuffle
        self._shuffle_seed = shuffle_seed

    def make_dataset_fn(self):
        """Create a dataset factory function for Lake stream processing.

        This method returns a lambda function that creates a column_io Dataset
        from Lake stream sources with the configured processing parameters.

        Returns:
            callable: A function that takes stream paths and returns a Dataset object.

        Note:
            The returned function is used internally by the data loading pipeline
            to create dataset instances for each shard of streaming data.
            It configures Lake-specific prefetching parameters for optimal performance.
        """
        return lambda x: column_io_dataset.Dataset.from_lake_source(
            paths=x,
            is_compressed=self._is_compressed,
            batch_size=self._batch_size,
            selected_columns=self._select_column,
            hash_features=self.hash_features,
            hash_types=self.hash_types,
            hash_buckets=self.hash_buckets,
            dense_columns=self._dense_column,
            dense_defaults=self._dense_default_value,
            use_prefetch=self._lake_use_prefetch,
            prefetch_thread_num=self._lake_prefetch_thread_num,
            prefetch_buffer_size=self._lake_prefetch_buffer_size,
        )

    def add_path(self, path, begin, end):
        """Add a Lake stream path with time range to the dataset.

        Args:
            path (str): Path to the Lake stream source.
            begin (int): Begin timestamp for data selection (Unix timestamp).
            end (int): End timestamp for data selection (Unix timestamp).

        Example:

        .. code-block:: python

            # Add stream for last 24 hours
            dataset.add_path(
                "/lake/user_behavior_stream",
                begin=1640995200,  # 2022-01-01 00:00:00
                end=1641081600,  # 2022-01-02 00:00:00
            )


        Note:
            The time range allows for precise control over which portion of
            the streaming data to process, enabling both historical analysis
            and real-time processing scenarios.
        """
        self._paths.append(path)
        self._begins.append(begin)
        self._ends.append(end)

    def _shard_path(self, sub_id, sub_num):
        """Create stream shards for distributed processing.

        This method partitions the input Lake streams across multiple workers and threads
        to enable parallel data loading. It uses LakeStreamSharding to ensure
        balanced distribution of streaming data with proper time range handling.

        Args:
            sub_id (int): Sub-process identifier within the worker.
            sub_num (int): Total number of sub-processes per worker.

        Note:
            This is an internal method used by the dataset creation process.
            The sharding strategy ensures that each worker processes a unique
            subset of the streaming data while maintaining temporal consistency
            and load balance. Shuffle is applied with the configured seed for
            reproducible randomization.
        """
        sharder = LakeStreamSharding()
        for path, begin, end in zip(self._paths, self._begins, self._ends):
            sharder.add_path(path, begin, end)
        self._shard_paths = sharder.partition(
            self._worker_idx * sub_num + sub_id,
            self._worker_num * sub_num,
            self._read_threads_num,
            shuffle=self._shuffle,
            seed=self._shuffle_seed,
        )
