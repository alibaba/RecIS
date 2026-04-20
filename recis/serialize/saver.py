from typing import Optional

import torch

from recis.monitor.monitor_reporter import SAVE_TIME_NAME, MonitorReporter


class Saver:
    """Saves model state dictionaries by sharding and parallel processing.

    This class handles both sparse (hashtable-based) and dense (tensor-based) state
    dictionaries, applying filtering and sharding logic before saving to disk.

    Examples:
    Typical usage example for saving a sharded checkpoint:

    >>> sparse_state_dict_copy = sparse_state_dict.copy()
    >>> sparse_state_dict, dense_state_dict = split_sparse_dense_state_dict(
    ...     sparse_state_dict_copy
    ... )
    >>> saver = Saver(
    ...     shard_index=shard_id,
    ...     shard_num=shard_num,
    ...     parallel=concurrent,
    ...     hashtables=sparse_state_dict,
    ...     tensors=dense_state_dict,
    ...     path=ckpt_path,
    ... )
    >>> saver.save()


    """

    def __init__(
        self,
        shard_index: int = 0,
        shard_num: int = 1,
        parallel: int = 8,
        path: str = ".",
        hashtables: Optional[dict] = None,
        tensors: Optional[list] = None,
        filter_func=lambda x: x,
    ) -> None:
        """Initializes the Saver with configuration and state data.

        Args:
            shard_index: The index of the current shard (0-based). Defaults to 0.
            shard_num: The total number of shards to create. Defaults to 1 (no sharding).
            parallel: The degree of parallelism for write operations. Defaults to 8.
            path: The output directory for saved files. Defaults to current directory.
            hashtables: A dictionary of sparse state (hashtables). Defaults to empty dict.
            tensors: A list of dense state (tensors). Defaults to empty list.
            filter_func: A callable to filter write blocks. Defaults to identity function.
        """
        if tensors is None:
            tensors = {}
        if hashtables is None:
            hashtables = {}
        self._shard_index = shard_index
        self._shard_num = shard_num
        self._parallel = parallel
        self._path = path
        self._hashtables = hashtables
        self._tensors = tensors
        self._filter_func = filter_func
        self._saver_impl = torch.classes.recis.Saver(
            self._shard_index, self._shard_num, self._parallel, self._path
        )

    @MonitorReporter.report_time_wrapper(SAVE_TIME_NAME, force=True)
    def save(self):
        """Executes the saving process.

        Generates write blocks from the state data, applies the filter function,
        and delegates to the internal saver implementation for actual I/O operations.
        """
        write_blocks = self._saver_impl.make_write_blocks(
            self._hashtables, self._tensors
        )
        write_blocks = self._filter_func(write_blocks)
        self._saver_impl.save(write_blocks)
