import json
from typing import Optional

import torch

from recis.monitor.monitor_reporter import (
    LOAD_SIZE_NAME,
    LOAD_TIME_NAME,
    MonitorReporter,
)
from recis.utils.logger import Logger


logger = Logger(__name__)


class Loader:
    """Loads model state dictionaries from checkpoint files with parallel processing.

    This class handles loading both sparse (hashtable-based) and dense (tensor-based)
    state dictionaries from disk, applying filtering logic to the load configuration.

    Examples:
        Typical usage example for loading a checkpoint:

        >>> loader = Loader(
        ...     checkpoint_path="/path/to/checkpoint",
        ...     hashtables=sparse_state_dict,
        ...     tensors=dense_state_dict,
        ...     parallel=16,
        ... )
        >>> loader.load()

    """

    def __init__(
        self,
        checkpoint_path: str,
        hashtables: Optional[dict] = None,
        tensors: Optional[dict] = None,
        parallel: int = 16,
        filter_func=lambda x: x,
    ) -> None:
        """Initializes the Loader with configuration and target state dictionaries.

        Args:
            checkpoint_path: The directory path containing checkpoint files to load.
            hashtables: A dictionary to receive loaded sparse state data.
                If None, an empty dictionary will be created.
            tensors: A dictionary to receive loaded dense state data.
                If None, an empty dictionary will be created.
            parallel: The degree of parallelism for read operations. Defaults to 16.
            filter_func: A callable to filter load information. Defaults to identity function.
        """
        self._checkpoint_path = checkpoint_path
        self._hashtables = hashtables
        if self._hashtables is None:
            self._hashtables = {}
        self._tensors = tensors
        if self._tensors is None:
            self._tensors = {}
        self._impl = torch.classes.recis.Loader(
            self._checkpoint_path,
            parallel,
            self._hashtables,
            self._tensors,
        )
        self._filter_func = filter_func

    @MonitorReporter.report_time_wrapper(LOAD_TIME_NAME, force=True)
    def load(self, print_load_summary=False):
        """Executes the loading process.

        Retrieves default load information from the checkpoint, applies the filter function
        to modify the load configuration, and delegates to the internal loader implementation
        for actual I/O operations.

        The load operation involves:
        1. Retrieving default load information from the checkpoint metadata;
        2. Applying the filter function to modify the load configuration;
        3. Loading the state data into the provided hashtables and tensors dictionaries using parallel processing;

        The actual file reading and data reconstruction are handled by the torch.classes.recis.Loader class.
        """
        load_info = json.loads(self._impl.default_load_info())
        if load_info is None:
            logger.warning(
                f"No load info found in {self._checkpoint_path}, skip loading"
            )
            return

        load_info = self._filter_func(load_info)
        load_summary, load_size = self._impl.load(json.dumps(load_info))
        MonitorReporter.report(LOAD_SIZE_NAME, load_size, force=True)
        if print_load_summary:
            self._print_load_summary(load_summary)

    def _print_load_summary(self, load_summary):
        missing_info = load_summary.missing_info()
        match_info = load_summary.match_info()
        logger.info(f"{'*' * 8} Load Summary {'*' * 8}")
        logger.info(f"load path: {self._checkpoint_path}")
        logger.info(f"{'*' * 8} Match Info   {'*' * 8}")
        dst_max_length = 0
        src_max_length = 0
        for dst, src in match_info.items():
            dst_max_length = max(dst_max_length, len(dst))
            src_max_length = max(src_max_length, len(src))
        for dst, src in match_info.items():
            logger.info(f"{dst:>{dst_max_length}} <- {src:<{src_max_length}}")
        logger.info(f"{'*' * 8} Missing Info  {'*' * 8}")
        logger.info(missing_info)
