import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import torch

from recis.io.lake_dataset import LakeStreamDataset
from recis.io.odps_dataset import OdpsDataset, get_table_size
from recis.utils.logger import Logger


if not os.environ.get("BUILD_DOCUMENT", None) == "1":
    from column_io.dataset.file_sharding import LakeStreamSharding


logger = Logger(__name__)


@dataclass
class TableSheet:
    """Data structure representing a segment of an ODPS table.

    TableSheet encapsulates the information needed to access a specific
    portion of an ODPS table, including the table name and row range.

    Attributes:
        table (str): The ODPS table name in format 'project.table'.
        start (int): Starting row index (inclusive).
        end (int): Ending row index (exclusive).

    Example:

    .. code-block:: python

        # Create a table sheet for rows 1000-2000
        sheet = TableSheet()
        sheet.table = "my_project.user_data"
        sheet.start = 1000
        sheet.end = 2000

    """

    table: str
    start: int
    end: int

    def __init__(self, table, start, end):
        self.table = table
        self.start = start
        self.end = end


def _parse_proportion(thread_num, proportions):
    """Parse proportion configuration and calculate thread distribution.

    This function validates proportion configurations and calculates how
    many threads should be allocated to each proportion group.

    Args:
        thread_num (int): Total number of threads available.
        proportions (List[tuple]): List of (parts, total) tuples defining
            the proportion of data each table should contribute.

    Returns:
        tuple: A tuple containing:
            - threads (List[int]): Number of threads for each proportion group
            - total_threads (int): Total threads after adjustment

    Raises:
        ValueError: If proportion configurations are inconsistent or invalid.

    Example:

    .. code-block:: python

        # Two tables with 2:1 ratio, 6 total threads
        threads, total = _parse_proportion(6, [(2, 3), (1, 3)])
        # Result: threads=[4, 2], total=6

    """
    total = {t for _, t in proportions}
    if len(total) != 1:
        raise ValueError(f"total split num should consistency, but get {total}")
    total = total.pop()
    part_thread_num = int(math.ceil(float(thread_num) / total))
    acc = 0
    threads = []
    for t, _ in proportions:
        acc += t
        threads.append(part_thread_num * t)
        if acc > total:
            raise ValueError(f"invalid proportions: {proportions}")
        elif acc == total:
            break
    return threads, total * part_thread_num


def _build_shard(begin, length, num, extra_offset=0):
    """Build data shards by dividing a range into approximately equal parts.

    This function divides a data range into the specified number of shards,
    distributing any remainder rows as evenly as possible across shards.

    Args:
        begin (int): Starting index of the range.
        length (int): Total length of the range to shard.
        num (int): Number of shards to create.
        extra_offset (int, optional): Offset for distributing remainder. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - works (List[tuple]): List of (start, end) tuples for each shard
            - extra_offset (int): Updated offset for next sharding operation

    Example:

    .. code-block:: python

        # Divide 100 rows into 3 shards starting from row 0
        shards, offset = _build_shard(0, 100, 3)
        # Result: [(0, 34), (34, 67), (67, 100)]

    """
    shards = [length // num] * num
    for i in range(length % num):
        shards[(extra_offset + i) % num] += 1
    extra_offset = (extra_offset + length) % num
    works = []
    for i in range(num):
        works.append((begin, begin + shards[i]))
        begin += shards[i]
    return works, extra_offset


def _shard_sheets_by_split_num(
    table, task_id, task_num, thread_num, split_num, extra_offset
):
    """Create table sheets by dividing table into fixed number of splits.

    This function creates TableSheet objects by dividing an ODPS table into
    a fixed number of splits, with each worker getting an equal share.

    Args:
        table (str): ODPS table name.
        task_id (int): Current worker/task identifier.
        task_num (int): Total number of workers/tasks.
        thread_num (int): Number of threads per worker.
        split_num (int): Number of splits to create.
        extra_offset (int): Offset for load balancing.

    Returns:
        tuple: A tuple containing:
            - sheets (List[TableSheet]): Table sheets for this worker
            - extra_offset (int): Updated offset for next operation

    Example:

    .. code-block:: python

        sheets, offset = _shard_sheets_by_split_num("project.table", 0, 4, 2, 10, 0)

    """
    shard_num = task_num * thread_num * split_num
    table_size = get_table_size(table)
    shards, extra_offset = _build_shard(0, table_size, shard_num, extra_offset)
    sheets = [TableSheet(table=table, start=start, end=end) for start, end in shards]
    return sheets[task_id::task_num], extra_offset


def _shard_sheets_by_row_num(
    table, task_id, task_num, thread_num, row_num, extra_offset
):
    """Create table sheets by dividing table into fixed-size row chunks.

    This function creates TableSheet objects by dividing an ODPS table into
    chunks of approximately the specified row count.

    Args:
        table (str): ODPS table name.
        task_id (int): Current worker/task identifier.
        task_num (int): Total number of workers/tasks.
        thread_num (int): Number of threads per worker.
        row_num (int): Target number of rows per chunk.
        extra_offset (int): Offset for load balancing.

    Returns:
        tuple: A tuple containing:
            - sheets (List[TableSheet]): Table sheets for this worker
            - extra_offset (int): Updated offset for next operation

    Example:

    .. code-block:: python

        sheets, offset = _shard_sheets_by_row_num("project.table", 0, 4, 2, 10000, 0)

    """
    shard_num = task_num * thread_num
    table_size = get_table_size(str(table))
    sheets = []
    for shard_offset in range(0, table_size, row_num * task_num):
        shard_size = min(row_num * task_num, table_size - shard_offset)
        shards, extra_offset = _build_shard(
            shard_offset, shard_size, shard_num, extra_offset
        )
        sheets.extend(
            TableSheet(table=table, start=start, end=end) for start, end in shards
        )
    return sheets[task_id::task_num], extra_offset


def _shard_table_by_subprocess(table_sheets, sub_id, sub_num):
    """Shard table sheets across subprocesses within a worker.

    This function further divides table sheets among subprocesses within
    a single worker, creating ODPS query strings with start/end parameters.

    Args:
        table_sheets (List[TableSheet]): Table sheets to shard.
        sub_id (int): Subprocess identifier.
        sub_num (int): Total number of subprocesses.

    Returns:
        List[str]: List of ODPS query strings for this subprocess.

    Example:

    .. code-block:: python

        queries = _shard_table_by_subprocess(sheets, 0, 2)
        # Result: ['project.table?start=0&end=500', ...]

    """
    sub_sheets = []
    for table_sheet in table_sheets:
        table = table_sheet.table
        start = table_sheet.start
        end = table_sheet.end
        slice_size = end - start
        extra = slice_size % sub_num
        slice_size = slice_size // sub_num
        end = start + (sub_id + 1) * slice_size + min(sub_id + 1, extra)
        start = start + sub_id * slice_size + min(sub_id, extra)
        sub_sheets.append(f"{table}?start={start}&end={end}")
    return sub_sheets


def _shard_lake_by_subprocess(lake_sheets, sub_id, sub_num):
    """Shard Lake stream sheets across subprocesses within a worker.

    This function divides Lake stream configurations among subprocesses,
    updating worker indices and counts appropriately.

    Args:
        lake_sheets (List[str]): Lake stream configuration strings.
        sub_id (int): Subprocess identifier.
        sub_num (int): Total number of subprocesses.

    Returns:
        List[str]: Updated Lake stream configurations for this subprocess.

    Note:
        Lake shard configuration format:
        'main_dir|start_time;end_time|hash|worker_idx;worker_num'

    Example:

    .. code-block:: python

        configs = _shard_lake_by_subprocess(lake_sheets, 0, 2)

    """
    if sub_num == 1:
        return lake_sheets
    sub_sheets = []
    for lake_sheet in lake_sheets:
        lake_splits = lake_sheet.rsplit("|", 1)
        worker_conf = lake_splits[-1].split(";")
        index = int(worker_conf[0]) * sub_num + sub_id
        total_num = sub_num * int(worker_conf[1])
        sub_sheets.append(f"{lake_splits[0]}|{index};{total_num}")
    return sub_sheets


def make_odps_window_io(split_num=None, row_num=None):
    """Create an ODPS dataset class with windowed access patterns.

    This factory function creates a specialized ODPS dataset class that supports
    windowed data processing with configurable sharding strategies.

    Args:
        split_num (int, optional): Number of splits to divide each table into.
        row_num (int, optional): Target number of rows per data chunk.

    Returns:
        class: A specialized OdpsDataset class with windowed capabilities.

    Raises:
        ValueError: If both or neither split_num and row_num are specified.

    Example:

    .. code-block:: python

        # Create dataset class with split-based sharding
        WindowDataset = make_odps_window_io(split_num=8)

        dataset = WindowDataset(batch_size=1024)
        dataset.add_path("project.table1", proportion=(2, 3))
        dataset.add_path("project.table2", proportion=(1, 3))

        # Process data in windows
        while True:
            try:
                if dataset.next_window():
                    continue  # Skip empty window
                for batch in dataset:
                    process_batch(batch)
            except StopIteration:
                break

    Note:
        Exactly one of split_num or row_num must be specified to define
        the sharding strategy.
    """
    if not ((split_num is None) ^ (row_num is None)):
        raise ValueError("only one of split num and row num should be set!")

    class _OdpsDataset(OdpsDataset):
        """Specialized ODPS dataset with windowed access capabilities.

        This class extends OdpsDataset to provide windowed data processing,
        allowing large tables to be processed in manageable chunks with
        configurable sharding strategies and proportional sampling.

        Class Attributes:
            _split_num (int): Number of splits for table sharding.
            _total_row_num (int): Target rows per chunk for row-based sharding.
            _path_proportion (List[tuple]): Proportion configuration for each table.
            _window_paths (List): Current window's data paths.
            _read_offset (torch.LongTensor): Current reading offset in windows.
            _shard_sheets (List): Current shard's table sheets.
        """

        _split_num = split_num
        _total_row_num = row_num
        _path_proportion = []
        _window_paths = []
        _read_offset = torch.LongTensor([0])
        _shard_sheets = []

        def __init__(self, *args, **kwargs):
            # for window io, save_interval should be None,
            # because read offset is saved after each window.
            kwargs["save_interval"] = None
            super().__init__(*args, **kwargs)

        def add_path(self, odps_table, proportion: tuple = (1, 1)):
            """Add an ODPS table path with proportional configuration.

            Args:
                odps_table (str): The ODPS table name to read.
                proportion (tuple): The portion of data this table contributes to batches.
                    Format: (parts, total_parts) where parts/total_parts defines
                    the proportion of this table in each batch.

            Raises:
                ValueError: If proportion format is invalid.

            Example:

            .. code-block:: python

                # Table contributes 2/3 of each batch
                dataset.add_path("project.main_table", proportion=(2, 3))

                # Table contributes 1/3 of each batch
                dataset.add_path("project.aux_table", proportion=(1, 3))

            Note:
                All tables should have the same total_parts value for consistency.
                The sum of all parts should equal total_parts.
            """
            if len(proportion) != 2:
                raise ValueError("porportion size must be 2")
            if not all(isinstance(v, int) for v in proportion):
                raise ValueError("element in proportion should be int")
            self._paths.append(odps_table)
            self._path_proportion.append(proportion)

        def _shard_window_paths(self):
            """Create windowed paths by sharding tables according to configuration.

            This method processes all added tables and creates windowed access
            patterns based on the configured sharding strategy (split_num or row_num).
            It handles proportional thread allocation and creates balanced shards.

            Raises:
                RuntimeError: If no paths have been added to the dataset.

            Note:
                This method is called internally when windowed access is first needed.
                It calculates thread distribution based on proportions and creates
                appropriate table sheets for each window.
            """
            if len(self._paths) == 0:
                raise RuntimeError("No paths are added to this dataset.")
            thread_nums, self._read_threads_num = _parse_proportion(
                self._read_threads_num, self._path_proportion
            )
            self._batch_size = int(
                math.ceil(self._batch_size / self._path_proportion[0][1])
            )
            if self._split_num:
                splits = [[] for _ in range(self._split_num)]
            shard_paths = []
            for index, table in enumerate(self._paths):
                extra_offset = 0
                if self._split_num:
                    thread_num = thread_nums[index % len(thread_nums)]
                    sheets, extra_offset = _shard_sheets_by_split_num(
                        table,
                        self._worker_idx,
                        self._worker_num,
                        thread_num,
                        self._split_num,
                        extra_offset,
                    )
                    for i in range(self._split_num):
                        splits[i].extend(sheets[i * thread_num : (i + 1) * thread_num])
                    if index % len(thread_nums) == len(thread_nums) - 1:
                        for i in range(self._split_num):
                            shard_paths.extend(splits[i])
                            splits[i] = []
                else:
                    sheets, extra_offset = _shard_sheets_by_row_num(
                        table,
                        self._worker_idx,
                        self._worker_num,
                        self._read_threads_num,
                        self._total_row_num,
                        extra_offset,
                    )
                    shard_paths.extend(sheets)
            self._window_paths = shard_paths

        def _shard_path(self, sub_id, sub_num):
            """Shard current window's paths across subprocesses.

            Args:
                sub_id (int): Subprocess identifier.
                sub_num (int): Total number of subprocesses.

            Note:
                This method is called internally to distribute the current
                window's table sheets among available subprocesses.
            """
            self._shard_paths = _shard_table_by_subprocess(
                self._shard_sheets, sub_id, sub_num
            )

        def state_dict(self):
            """Get the current state of the windowed IO system.

            Returns:
                dict: State dictionary containing:
                    - read_offset: Current reading offset tensor

            Example:

            .. code-block:: python

                state = dataset.state_dict()
                # Save state for resumption
                torch.save(state, "dataset_state.pt")

            """
            return {"read_offset": self._read_offset}

        def load_state_dict(self, state_dict):
            """Load previously saved state to resume windowed processing.

            Args:
                state_dict (dict): State dictionary from previous state_dict() call.

            Example:

            .. code-block:: python

                # Load saved state
                state = torch.load("dataset_state.pt")
                dataset.load_state_dict(state)

                # Continue from where we left off
                while True:
                    try:
                        if dataset.next_window():
                            continue
                        for batch in dataset:
                            process_batch(batch)
                    except StopIteration:
                        break

            """
            self._read_offset = state_dict["read_offset"]

        def next_window(self) -> bool:
            """Advance to the next window and reinitialize the dataset.

            This method moves to the next window of data and prepares the dataset
            for processing that window. It handles window initialization and
            empty window detection.

            Returns:
                bool: True if the current window should be skipped (empty),
                    False if the window contains data to process.

            Raises:
                StopIteration: When all windows have been processed.

            Example:

            .. code-block:: python

                while True:
                    try:
                        if dataset.next_window():
                            print("Skipping empty window")
                            continue

                        print("Processing window...")
                        for batch in dataset:
                            process_batch(batch)

                    except StopIteration:
                        print("All windows processed")
                        break

            Note:
                This method should be called before processing each window.
                It automatically handles window boundaries and data availability.
            """
            if len(self._window_paths) == 0:
                self._shard_window_paths()
            read_offset = self._read_offset[0]
            if read_offset == len(self._window_paths):
                raise StopIteration("Finish IO")
            self._shard_sheets = self._window_paths[
                read_offset : read_offset + self._read_threads_num
            ]
            self._read_offset += self._read_threads_num
            return False

        def reset(self):
            """Reset the windowed dataset to initial state.

            This method clears all windowed paths and resets the reading offset,
            allowing the dataset to be reused from the beginning.

            Example:

            .. code-block:: python

                # Process data once
                while True:
                    try:
                        if dataset.next_window():
                            continue
                        for batch in dataset:
                            process_batch(batch)
                    except StopIteration:
                        break

                #
                dataset.reset()
                # Can now call next_window() again from the beginning

            """
            self._window_paths = []
            self._read_offset = self._read_offset.fill_(0)

    return _OdpsDataset


def _sample_sheets(table, task_id, thread_num, row_num, seed=0):
    """Create randomly sampled table sheets for experimentation.

    This function creates TableSheet objects by randomly sampling portions
    of an ODPS table, which is useful for experimentation and testing.

    Args:
        table (str): ODPS table name.
        task_id (int): Task identifier for seeding randomization.
        thread_num (int): Number of threads to create sheets for.
        row_num (int): Target number of rows to sample.
        seed (int, optional): Random seed for reproducible sampling. Defaults to 0.

    Returns:
        List[TableSheet]: List of randomly sampled table sheets.

    Example:

    .. code-block:: python

        # Sample 10000 rows across 4 threads
        sheets = _sample_sheets("project.table", 0, 4, 10000, seed=42)

    Note:
        This function uses random sampling, so results will vary unless
        the same seed is used. It's primarily intended for experimentation
        and testing scenarios.
    """
    random.seed(seed + task_id)
    table_size = get_table_size(str(table))
    row_num = min(row_num, table_size)
    row_per_thread = row_num // thread_num + 1
    sheets = []
    for i in range(thread_num):
        if i == row_num % thread_num:
            row_per_thread -= 1
        start = random.randint(0, table_size - row_per_thread)
        end = start + row_per_thread
        # sheets.append('{}?start={}&end={}'.format(table, start, end))
        sheets.append(TableSheet(table=table, start=start, end=end))
    return sheets


def make_odps_sample_io(row_num=500000, seed=0):
    """Make ODPS sample io class.

    Args:
      row_num: Number of rows per worker.
      seed: Initial seed for randomization.
    """

    class _OdpsSampleIO(OdpsDataset):
        _total_row_num = row_num
        _initial_seed = seed
        _window_paths = []
        _read_offset = torch.LongTensor([0])
        _shard_sheets = []

        def _shard_window_paths(self):
            self._window_paths = []
            for table in self._paths:
                sheets = _sample_sheets(
                    table,
                    self._worker_idx,
                    self._read_threads_num,
                    self._total_row_num,
                    self._initial_seed,
                )
                self._window_paths.extend(sheets)

        def _shard_path(self, sub_id, sub_num):
            self._shard_paths = _shard_table_by_subprocess(
                self._shard_sheets, sub_id, sub_num
            )

        def state_dict(self):
            """Get state of window io\n
            Return:
              {"read_offset": tensor_state}
            """
            return {"read_offset": self._read_offset}

        def load_state_dict(self, state_dict):
            """Load state of window io\n
            Args:
              state_dict: the state dict to load.
            """
            self._read_offset = state_dict["read_offset"]

        def next_window(self) -> bool:
            """Move the window forward.

            Args:
              sess: Session to move the window.

            Returns:
              Whether the window is empty.
            """
            self._shard_window_paths()
            self._shard_sheets = []
            self._shard_sheets.extend(self._window_paths)

            return False

        def reset(self):
            pass

    return _OdpsSampleIO


_US = int(1e6)


def _str_to_timestamp(s, format="%Y%m%d%H%M%S"):
    s = str(s)
    ts = time.mktime(datetime.strptime(s, format).timetuple())
    return int(ts) * _US


def _ts_to_string(ts, format="%Y%m%d%H%M%S"):
    ts = ts // _US
    return datetime.strftime(datetime.fromtimestamp(ts), format)


def make_lake_stream_window_io(step_mins=60, repeat_mins=None, name="train"):
    """Make lake stream window io class.

    Args:
      step_mins: Step length in minutes move window forward.
      repeat_mins: The small step length used to read repeatedly.
        When it is None, it will be set to the whole read length.
    """
    if not isinstance(step_mins, int):
        raise ValueError("The step mins must be int")
    if repeat_mins is not None:
        if not isinstance(repeat_mins, int):
            raise ValueError("the repeat_mins should be None or int")

    class _LakeStreamWindowIO(LakeStreamDataset):
        _step = step_mins * 60 * _US
        _repeat = None
        if repeat_mins is not None:
            _repeat = repeat_mins * 60 * _US
        _path_proportion = []
        _window_paths = []
        _read_offset = torch.LongTensor([0])
        _shard_sheets = []
        _epochs = 1
        _name = name
        _loaded = False

        def __init__(self, *args, **kwargs):
            # for window io, save_interval should be None,
            # because read offset is saved after each window.
            kwargs["save_interval"] = None
            super().__init__(*args, **kwargs)
            self._current_window = []

        def add_path(self, lake_path, start_time, interval_mins):
            self.add_paths([(lake_path, start_time, interval_mins)])

        def add_paths(self, lake_configs, proportions=None):
            """a list of lake config.

            Args:
              lake_configs: list of lake config, each item is tuple of
                (topic, start time, interval mins).
              proportions: List of portion of each lake source, each item is
                tuple of (portion, total portion).
            """
            if proportions is None:
                proportions = len(lake_configs) * [(1, 1)]

            if len(proportions) != len(lake_configs):
                raise ValueError("The len(lake path) must equal len(proportions)")
            self._path_proportion.extend(proportions)
            for lake_path, start_time, interval_min in lake_configs:
                end_time = start_time + interval_min * 60 * _US
                path = lake_path
                super().add_path(path, start_time, end_time)

        def _shard_window_data(self):
            logger.info("shard window data")
            start_ts = self._begins[0]
            if not self._loaded:
                self._read_offset[0] = start_ts
            self._interval = self._ends[0] - start_ts
            if self._interval < 0:
                raise ValueError("interval should be greater than 0")
            if self._repeat is not None:
                if self._interval % self._repeat != 0:
                    raise ValueError(
                        "the interval should be an integer multiple of repeat_min"
                    )
            else:
                self._repeat = self._interval

            self._ends = [-1] * len(self._ends)
            self._thread_nums, self._read_threads_num = _parse_proportion(
                self._read_threads_num, self._path_proportion
            )
            self._window_paths = self._paths

        def _shard_path(self, sub_id, sub_num):
            self._shard_paths = _shard_lake_by_subprocess(
                self._shard_sheets, sub_id, sub_num
            )

        def state_dict(self):
            """Get state of window io\n
            Return:
              {"read_offset": tensor_state}
            """
            return {"read_offset": self._read_offset}

        def load_state_dict(self, state_dict):
            """Load state of window io\n
            Args:
              state_dict: the state dict to load.
            """
            self._read_offset = state_dict["read_offset"]
            self._loaded = True

        def reset(self):
            self._window_paths = []
            self._loaded = False

        def current_window(self):
            return self._current_window

        def next_window(self):
            """Move the window forward.

            Returns:
              Whether the window is empty.
            """
            if len(self._window_paths) == 0:
                self._shard_window_data()

            start_ts = self._read_offset.item()
            self._read_offset = self._read_offset + self._step
            end_ts = start_ts + self._interval
            self._current_window = [_ts_to_string(start_ts), _ts_to_string(end_ts)]
            self._shard_sheets = []
            total_step = 1
            try:
                total_step = int(self._interval / self._repeat)
            except ZeroDivisionError:
                pass
            for step in range(total_step):
                step_paths = []
                for index, config in enumerate(self._paths):
                    sharding = LakeStreamSharding()
                    sharding.add_path(
                        config,
                        start_ts + step * self._repeat,
                        start_ts + (step + 1) * self._repeat,
                    )
                    paths = sharding.partition(
                        self._worker_idx,
                        self._worker_num,
                        self._thread_nums[index % len(self._thread_nums)],
                        self._shuffle,
                    )
                    step_paths.extend(paths)
                self._shard_sheets.extend(step_paths * self._epochs)
            logging_str = f"Initialize IO[{self._name}] to: start[{_ts_to_string(start_ts)}] end[{_ts_to_string(end_ts)}]"
            if self._epochs != 1:
                logging_str += f" repeat_mins[{self._repeat / (60 * _US):.0f}] repeat_times[{self._epochs}]"
            logging_str += "."
            logger.info(logging_str)
            try:
                next(iter(self))
            except StopIteration:
                return True
            return False

    return _LakeStreamWindowIO
