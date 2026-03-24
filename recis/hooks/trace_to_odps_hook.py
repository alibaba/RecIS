import os
import time
from functools import wraps
from multiprocessing import Process, Queue
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from recis.hooks import Hook
from recis.info import is_internal_enabled
from recis.utils.cluster import is_external_cluster
from recis.utils.logger import Logger


if is_internal_enabled():
    from recis.utils.cluster import X_CLUSTER_TUNNEL_ENDPOINT


if not os.environ.get("BUILD_DOCUMENT", None) == "1":
    from odps import ODPS
    from odps.models import Schema
    from odps.tunnel.io.writer import ArrowWriter
    from odps.tunnel.tabletunnel import TableTunnel


logger = Logger(__name__)

TRACE_MAP = {}

rank = int(os.environ.get("RANK", 0))


def retry(retry_count, interval):
    """Decorator for adding retry logic to functions.

    This decorator automatically retries a function if it raises an exception,
    with configurable retry count and interval between attempts.

    Args:
        retry_count (int): Maximum number of retry attempts.
        interval (float): Time interval (in seconds) between retry attempts.

    Returns:
        callable: Decorated function with retry logic.

    Example:
        >>> @retry(retry_count=3, interval=1.0)
        ... def unreliable_function():
        ...     # Function that might fail
        ...     pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retry_count):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retry_count - 1:
                        raise e
                    time.sleep(interval)

        return wrapper

    return decorator


def add_to_trace(name: str, tensor: Union[torch.Tensor, np.ndarray, list] = None):
    """Adds data to the trace map for ODPS logging.

    This function adds training data to the global trace map that will be
    uploaded to ODPS tables. Supports tensors, numpy arrays, and lists.

    Args:
        name (str): Name/key for the data being traced.
        tensor (Union[torch.Tensor, np.ndarray, list]): Data to be traced.
            Must be one of the supported types.

    Raises:
        ValueError: If the tensor type is not supported.

    Example:
        >>> import torch
        >>> import numpy as np
        >>> # Add tensor data
        >>> embeddings = torch.randn(100, 64)
        >>> add_to_trace("user_embeddings", embeddings)
        >>> # Add numpy array
        >>> features = np.random.rand(100, 32)
        >>> add_to_trace("item_features", features)
        >>> # Add list data
        >>> user_ids = [1, 2, 3, 4, 5]
        >>> add_to_trace("user_ids", user_ids)

    Note:
        Tensor data is automatically converted to numpy arrays for compatibility
        with ODPS. A warning is logged if data with the same name already exists.
    """
    if not isinstance(tensor, (torch.Tensor, np.ndarray, list)):
        raise ValueError(
            f"Trace data must be torch.Tensor or np.ndarray or list, not now {type(tensor)}"
        )
    global TRACE_MAP
    if name in TRACE_MAP:
        logger.warning(f"Trace data {name} already exists")
    if isinstance(tensor, torch.Tensor):
        TRACE_MAP[name] = tensor.detach().cpu().numpy()
    else:
        TRACE_MAP[name] = tensor


def get_trace_map():
    """Gets the global trace map containing data to be uploaded to ODPS.

    Returns:
        Dict: Global trace map containing key-value pairs of data to be traced.
    """
    global TRACE_MAP
    return TRACE_MAP


def clear_trace_map():
    """Clears the global trace map.

    This function is typically called after uploading data to ODPS
    to prepare for the next batch of trace data.
    """
    global TRACE_MAP
    TRACE_MAP = {}


def patch_flush(self):
    """Patches the flush method for optimized ODPS uploads.

    This function modifies the default flush behavior to support
    chunked uploads for large data transfers to ODPS.

    Args:
        self: The writer instance to patch.
    """
    checksum = self._crccrc.getvalue()
    self._write_unint32(checksum)
    self._crccrc.reset()
    chunk_size = 1 << 27

    def gen():
        # synchronize chunk upload
        data = self._out.getvalue()
        while data:
            to_send = data[:chunk_size]
            data = data[chunk_size:]
            yield to_send

    self._request_callback(gen())


class TraceWriter(Process):
    """Multiprocess writer for uploading trace data to ODPS tables.

    The TraceWriter runs as a separate process to handle ODPS uploads without
    blocking the main training process. It supports buffering, batching, and
    automatic retry mechanisms for reliable data transfer.

    Args:
        config (Dict): ODPS configuration containing access credentials and table info.
            Required keys: access_id, access_key, project, end_point, table_name.
            Optional keys: partition.
        fields (List[str]): List of field names for the ODPS table schema.
        types (List[str]): List of field types corresponding to the fields.
        writer_id (int): Unique identifier for this writer process.
        queue (Queue): Multiprocessing queue for receiving data to write.
        size_threshold (int): Buffer size threshold in bytes for triggering flushes.
            Defaults to 50 MiB.

    Attributes:
        table_name (str): Name of the ODPS table to write to.
        fields (List[str]): Field names for the table schema.
        types (List[str]): Field types for the table schema.
        partition (str): Partition specification for the table.
        write_count (int): Total number of rows written.
        buffer (List[Dict]): Internal buffer for batching data.
        buffered_size (int): Current size of buffered data in bytes.

    Example:
        >>> config = {
        ...     "access_id": "your_access_id",
        ...     "access_key": "your_access_key",
        ...     "project": "your_project",
        ...     "end_point": "your_endpoint",
        ...     "table_name": "training_traces",
        ...     "partition": "dt=20231201",
        ... }
        >>> fields = ["user_id", "item_id", "score"]
        >>> types = ["bigint", "bigint", "double"]
        >>> queue = Queue()
        >>> writer = TraceWriter(config, fields, types, 0, queue)
        >>> writer.start()
    """

    def __init__(
        self,
        config: Dict,
        fields: List[str],
        types: List[str],
        writer_id: int,
        queue: Queue,
        size_threshold: int = 50 * 1024 * 1024,  # 50 MiB
    ):
        super().__init__()
        required = {"access_id", "access_key", "project", "table_name"}
        if missing := required - config.keys():
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")

        self._block_id = 0
        self.table_name = config["table_name"]
        self.partition = config.get(
            "partition", None
        )  # if None, the table is non-partitioned
        self.fields = fields
        self.types = types

        # Handling ODPS
        if is_internal_enabled():
            end_point = (
                X_CLUSTER_TUNNEL_ENDPOINT
                if is_external_cluster()
                else config.get("end_point")
                or "http://service.odps.aliyun-inc.com/api"  # TODO: AIDC singapore/korea compability
            )
            odps_args = [config["access_id"], config["access_key"], config["project"]]
            odps_kwargs = (
                {"tunnel_endpoint": X_CLUSTER_TUNNEL_ENDPOINT}
                if is_external_cluster()
                else {"endpoint": end_point}
            )
        else:
            odps_args = [
                config["access_id"],
                config["access_key"],
                config["project"],
                config["end_point"],
            ]
            odps_kwargs = {}
        odps = ODPS(*odps_args, **odps_kwargs)

        # Create table or not by is_external_cluster
        if is_external_cluster():
            logger.info(
                f"In external cluster, skip create table and partition: {self.table_name}, {self.partition}"
            )
        else:
            partitions = []
            part_types = []
            if self.partition:
                for s in self.partition.split(","):
                    k = s.split("=")[0]
                    partitions.append(k)
                    part_types.append("string")
            table = odps.create_table(
                self.table_name,
                schema=Schema.from_lists(
                    self.fields, self.types, partitions, part_types
                ),
                if_not_exists=True,
                lifecycle=365,
                table_properties={"columnar.nested.type": "true"},
            )
            if self.partition:
                table.create_partition(self.partition, if_not_exists=True)

        # Create upload session
        self._tunnel_client = TableTunnel(odps)
        self._writer_session = self._tunnel_client.create_upload_session(
            self.table_name, partition_spec=self.partition
        )

        self.write_count = 0
        self.write_id = writer_id
        self._block_id = 0
        self.daemon = True
        self.queue = queue
        # Buffering
        self.buffer = []  # list of dicts
        self.buffered_size = 0
        self.size_threshold = size_threshold

    def run(self) -> None:
        """Main process loop for handling data writes.

        Continuously processes data from the queue until a None sentinel
        value is received, indicating shutdown.
        """
        while True:
            data = self.queue.get()
            if data is None:
                self.flush(force=True)
                break
            self.write(data)

    @retry(retry_count=3, interval=10)
    def write(self, data: Dict[str, np.ndarray]):
        """Writes data to the internal buffer and flushes when threshold is reached.

        Args:
            data (Dict[str, np.ndarray]): Dictionary mapping field names to data arrays.

        Note:
            Data is automatically converted to lists for consistency and buffered
            until the size threshold is reached, at which point it's flushed to ODPS.
        """
        # Convert all values to lists for consistency
        for key in data.keys():
            if not isinstance(data[key], list):
                data[key] = data[key].tolist()
        self.buffer.append(data)
        # Estimate size using DataFrame
        df = pd.DataFrame(data)
        self.buffered_size += df.memory_usage(deep=True).sum()
        # Check threshold
        if self.buffered_size >= self.size_threshold:
            self._flush_buffer()

    def _flush_buffer(self):
        """Flushes the internal buffer to ODPS.

        Merges all buffered data into a single batch and uploads it to ODPS
        using Arrow format for efficient transfer.
        """
        if not self.buffer:
            return
        # Concatenate all dicts in buffer
        merged = {k: [] for k in self.buffer[0].keys()}
        for d in self.buffer:
            for k in merged:
                merged[k].extend(d[k])
        ArrowWriter._flush = patch_flush
        df = pd.DataFrame(merged)
        row_num = len(df)
        write_data = pa.RecordBatch.from_pandas(df)
        writer = self._writer_session.open_arrow_writer(self._block_id)
        writer.write(write_data)
        writer.close()
        self.write_count += row_num
        self._block_id += 1
        self.buffer = []
        self.buffered_size = 0
        # flush ODPS every 2 blocks
        if self._block_id == 2:
            self.flush()

    @retry(retry_count=3, interval=10)
    def flush(self, force=False):
        """Flushes data to ODPS and commits the upload session.

        Args:
            force (bool): If True, forces flushing of any remaining buffer data.
                Defaults to False.

        Note:
            This method commits the current upload session and creates a new one
            for subsequent writes. It's called automatically every 2 blocks or
            when forced during shutdown.
        """
        # Flush any remaining buffer first
        if force:
            self._flush_buffer()
        # update writer_session
        if self._block_id > 0:
            self._writer_session.commit(list(range(self._block_id)))
            self._writer_session = self._tunnel_client.create_upload_session(
                self.table_name, partition_spec=self.partition
            )
            self._block_id = 0

    def __del__(self):
        """Cleanup method called when the writer process is destroyed.

        Logs the total write count and ensures any remaining data is flushed.
        """
        logger.info(
            f"[rank-{rank}] [writer-{self.write_id}] write_count = {self.write_count}"
        )
        if self._block_id > 0 or self.buffer:
            self.flush(force=True)


class TraceToOdpsHook(Hook):
    """Hook for tracing training data to ODPS tables.

    The TraceToOdpsHook provides high-performance data collection and upload
    capabilities for training traces. It uses multiprocessing to avoid blocking
    the main training process and supports configurable batching and buffering.

    Args:
        config (Dict): ODPS configuration dictionary containing connection details.
            Required keys: access_id, access_key, project, end_point, table_name.
            Optional keys: partition.
        fields (List[str]): List of field names for the ODPS table schema.
        types (List[str]): List of field types corresponding to the fields.
        worker_num (int): Number of worker processes for parallel uploads.
            Defaults to 1.
        size_threshold (int): Buffer size threshold in bytes for triggering flushes.
            Defaults to 50 MiB.

    Attributes:
        queue (Queue): Multiprocessing queue for data transfer.
        writer_num (int): Number of writer processes.
        writers (List[TraceWriter]): List of writer process instances.

    Example:
        >>> from recis.hooks import TraceToOdpsHook, add_to_trace
        >>> # Configure ODPS connection
        >>> config = {
        ...     "access_id": "your_access_id",
        ...     "access_key": "your_access_key",
        ...     "project": "your_project",
        ...     "end_point": "your_endpoint",
        ...     "table_name": "training_traces",
        ...     "partition": "dt=20231201",
        ... }
        >>> # Define table schema
        >>> fields = ["user_id", "item_id", "embedding", "score"]
        >>> types = ["bigint", "bigint", "string", "double"]
        >>> # Create hook
        >>> odps_hook = TraceToOdpsHook(
        ...     config=config, fields=fields, types=types, worker_num=2
        ... )
        >>> trainer.add_hook(odps_hook)
        >>> # During training, add data to be traced
        >>> add_to_trace("user_embeddings", user_embeddings)
        >>> add_to_trace("item_scores", item_scores)
        >>> # The hook will automatically upload data after each step

    Note:
        This hook is only available in internal environments where ODPS
        access is configured. Use add_to_trace() to add data that should
        be uploaded to ODPS tables.
    """

    def __init__(
        self,
        config: Dict,
        fields: List[str],
        types: List[str],
        worker_num: int = 1,
        size_threshold: int = 50 * 1024 * 1024,
    ) -> None:
        super().__init__()
        self.queue = Queue(maxsize=worker_num)
        self.writer_num = worker_num
        self.writers = []
        for i in range(self.writer_num):
            self.writers.append(
                TraceWriter(
                    config, fields, types, i, self.queue, size_threshold=size_threshold
                )
            )
        for writer in self.writers:
            writer.start()

    def check_alive(self):
        """Checks if all writer processes are still alive.

        Returns:
            bool: True if all writer processes are alive, False otherwise.

        Raises:
            ValueError: If any writer process has died unexpectedly.
        """
        alive = True
        for writer in self.writers:
            if not writer.is_alive():
                alive = False
        return alive

    def after_step(self, is_train=True, *args, **kw):
        """Called after each training step to upload accumulated trace data.

        This method retrieves all data from the trace map and sends it to
        the writer processes for upload to ODPS. After sending, the trace
        map is cleared to prepare for the next step.

        Args:
            *args: Variable length argument list (unused).
            **kw: Arbitrary keyword arguments (unused).

        Raises:
            ValueError: If any writer subprocess has encountered an error.

        Note:
            The method checks that all writer processes are still alive before
            sending data. If any process has died, an error is raised.
        """
        if not self.check_alive():
            raise ValueError("TraceToOdpsHook sub-process raise error")
        data = get_trace_map()
        self.queue.put(data)
        clear_trace_map()

    def end(self, is_train=True, *args, **kwargs):
        """Called at the end of training to properly shutdown writer processes.

        This method sends shutdown signals to all writer processes and waits
        for them to complete their work and terminate gracefully.
        """
        for writer in self.writers:
            self.queue.put(None)
        for writer in self.writers:
            writer.join()
