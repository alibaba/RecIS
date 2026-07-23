import os
from column_io.dataset.log_util import logger
import random
try:
    from column_io.dataset.odps_env_setup import is_turn_on_odps_open_storage
    if is_turn_on_odps_open_storage():
        from column_io.lib.interface import _OdpsOpenStorageDataset as OdpsTableDataset
    else:
        from column_io.lib.interface import _OdpsTableColumnDataset as OdpsTableDataset
    from column_io.dataset.dataset import OdpsComboDataset
except:
    OdpsTableDataset = None
    OdpsComboDataset = None

class OrcFileSharding(object):
    def __init__(self):
        self._file_dirs = list()
        self._file_names = list()
        self._file_sizes = list()
        self._total_row_count = 0

    def get_paths(self):
        return self._path_dirs

    def list_file(self, file_dir, prefix):
        files = []
        for dirname, dirs, filenames in os.walk(file_dir):
            files.extend([os.path.join(dirname, filename) for filename in filenames if filename.endswith(prefix)])
        return files

    def add_path(self, file_dir, prefix=".orc"):
        self._file_dirs.append(file_dir)
        files = self.list_file(file_dir, prefix)
        file_count = len(files)
        self._file_names.extend(files)
        self._file_sizes.append(file_count)
        # TODO(yuhuan.zh) partition file by row count
        self._total_row_count += 0

    def add_paths(self, paths, prefix=".orc"):
        for path in paths:
            self.add_path(path, prefix)

    def get_row_count(self):
        return self._total_row_count

    def partition(
        self,
        worker_idx,
        worker_num,
        slice_per_worker,
        shuffle=False,
        seed=10,
        epochs_num=1,
    ):
        tables_to_read = []
        for i, file_sheet in enumerate(self._file_names):
            if i % worker_num == worker_idx:
                tables_to_read.append(file_sheet)

        if shuffle:
            if seed:
                random.seed(seed)
            ret = []
            for i in range(0, epochs_num):
                paths = tables_to_read[:]
                random.shuffle(paths)
                ret.extend(paths)
            return ret
        else:
            return tables_to_read * epochs_num


class OdpsTableSharding(object):
    def __init__(self):
        self._tables = list()
        self._table_sizes = list()
        self._total_row_count = 0

    def get_paths(self):
        return self._tables

    def add_path(self, table_name):
        table_size = OdpsTableDataset.get_table_size(table_name)
        if table_size < 0:
            raise ValueError("get table size failed: " + table_name)
        elif table_size == 0:
            logger.warning("table size is 0, skip table: " + table_name)
            return
        logger.info("table size is " + str(table_size) + ", table name: " + table_name)
        self._tables.append(table_name)
        self._table_sizes.append(table_size)
        self._total_row_count += table_size

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def get_row_count(self):
        return self._total_row_count

    def partition(
        self,
        worker_idx,
        worker_num,
        slice_per_worker,
        shuffle=False,
        seed=10,
        epochs_num=1,
        slice_size=None,
    ):
        table_sheets = []
        compute_slice_size = False if slice_size else True
        for table_name, table_size in zip(self._tables, self._table_sizes):
            if compute_slice_size:
                slice_size = max(1, int(table_size / (worker_num * slice_per_worker)))
            num_slices = int(table_size / slice_size)
            num_slices += table_size > num_slices * slice_size
            for i in range(0, num_slices):
                end = min((i + 1) * slice_size, table_size)
                table_sheet = "{}?start={}&end={}".format(
                    table_name, i * slice_size, end
                )
                table_sheets.append(table_sheet)

        tables_to_read = []
        for i, table_sheet in enumerate(table_sheets):
            if i % worker_num == worker_idx:
                tables_to_read.append(table_sheet)

        if shuffle:
            if seed:
                random.seed(seed)
            ret = []
            for i in range(0, epochs_num):
                paths = tables_to_read[:]
                random.shuffle(paths)
                ret.extend(paths)
            return ret
        else:
            return tables_to_read * epochs_num

class OdpsComboTableSharding(object):
    """Table sharding for combo data."""

    def __init__(self):
        """Initiate an OdpsComboTableSharding object."""
        self._table_groups = []
        self._rows_per_table_group = []
        self._begin_groups = []
        self._end_groups = []
        self._total_row_count = 0

    def partition(self, worker_idx, worker_num, slice_per_worker, shuffle=False, seed=None, epochs_num=1, slice_size=None):
        """Get the combo paths partitions for the given worker.

        Args:
            worker_idx: integer. The index of the worker. Should be set to 0 when work queue is enabled.
            worker_num: integer. The number of workers. Should be set to 1 when work queue is enabled.
            slice_per_worker: integer. Slice per path per worker. Suggested to be set to reader thread num.
            shuffle: boolean. Whether to shuffle the paths.
            seed: integer. Seed when executing shuffle.
            epochs_num: integer. The number of epochs to read data.
            slice_size: integer. Preset size of slice.

        Returns:
            List of lists of string. Each item is a sharding of table group.
        """
        if len(self._table_groups) == 0:
            raise ValueError('table_groups should not be empty when `partition` is called.')
        table_num_per_group = len(self._table_groups[0])
        for table_group in self._table_groups:
            if len(table_group) != table_num_per_group:
                raise ValueError('table_groups length should be all the same, expect {}, got {}'
                            .format(table_num_per_group, len(table_group)))

        # Phase 1: create ALL slice tuples from all groups (no worker filtering)
        all_slice_tuples = []
        configured_slice_size = slice_size
        for table_names_per_group, table_size, begin_group in zip(self._table_groups, self._rows_per_table_group, self._begin_groups):
            if configured_slice_size is None:
                cur_slice_size = max(1, int(table_size / (worker_num * slice_per_worker)))
            else:
                cur_slice_size = configured_slice_size
            cur_sharding_groups = []
            for table_name, begin in zip(table_names_per_group, begin_group):
                cur_table_sharding = []
                num_slices = int(table_size / cur_slice_size)
                num_slices += table_size > num_slices * cur_slice_size
                for i in range(0, num_slices):
                    end = min((i + 1) * cur_slice_size, table_size)
                    table_sheet = "{}?start={}&end={}".format(
                        table_name, begin + i * cur_slice_size, begin + end)
                    cur_table_sharding.append(table_sheet)
                cur_sharding_groups.append(cur_table_sharding)
            all_slice_tuples.extend(list(zip(*cur_sharding_groups)))

        # Phase 2: global round-robin across the pooled tuples
        res = [t for i, t in enumerate(all_slice_tuples) if i % worker_num == worker_idx]

        if not shuffle:
            return res * epochs_num
        if seed:
            random.seed(seed)
        shuffled_table_sheets = []
        for i in range(0, epochs_num):
            paths = res[:]
            random.shuffle(paths)
            shuffled_table_sheets.extend(paths)
        return shuffled_table_sheets
        

    def get_paths(self):
      """Return the table groups. Each group of paths would assembled together.
         For example, suppose we have two odps tables, with three table groups:
         [
           ["odps://xxx/ds=aaa", "odps://yyy/ds=aaa"],
           ["odps://xxx/ds=bbb", "odps://yyy/ds=bbb"],
           ["odps://xxx/ds=ccc", "odps://yyy/ds=ccc"]
         ]
      """
      return self._table_groups

    def add_paths(self, table_groups, begin_groups, end_groups):
        if not (len(table_groups) == len(begin_groups) == len(end_groups)):
            raise ValueError("table_groups, begin_groups, end_groups must have same length")
        for path, begin, end in zip(table_groups, begin_groups, end_groups):
            self.add_path(path, begin, end)

    def add_path(self, table_group, begin_group, end_group):
        """Add a table group.

        Args:
            table_group: list. List of odps paths within a table group.
        """
        if not isinstance(table_group, (list, tuple)):
            raise ValueError('Table groups for OdpsComboTableSharding should be '
                            'list of odps paths, but got {}'.format(table_group))
        if len(self._table_groups) > 0 and len(table_group) != len(self._table_groups[0]):
            raise ValueError('Table groups for OdpsComboTableSharding must have '
                            'the same number of partitions, but got '
                            '{} and {}'.format(len(self._table_groups[0]), len(table_group)))
        if len(table_group) != len(begin_group):
            raise ValueError('Table groups for OdpsComboTableSharding must have the same number of rows {} {}'.format(len(table_group), len(begin_group)))
        real_table_group = []
        real_begin_group = []
        real_end_group = []
        target_row_count = -1
        for part, begin, end in zip(table_group, begin_group, end_group):
            real_path = part.encode('utf-8')
            if not (begin or end):
                end = OdpsComboDataset.get_table_size(real_path)
                begin = 0
            part_size = end - begin
            if part_size < 0:
                raise ValueError("Get table size failed: " + real_path)
            if target_row_count != -1 and part_size != target_row_count:
                raise ValueError("Rows not match for {} and {}: {} vs {}".format(
                    table_group[0], part, target_row_count, part_size))
            target_row_count = part_size
            real_table_group.append(part)
            real_begin_group.append(begin)
            real_end_group.append(end)
        if target_row_count == 0:
            logger.info("Table sizes are all 0, skip table group: " + real_table_group)
            return

        # logger.info("DEBUG [combodataset] Table size is {}, table group: {}; begin group {}; end group {}".format(target_row_count, real_table_group, real_begin_group, real_end_group))
        self._rows_per_table_group.append(target_row_count)
        self._table_groups.append(real_table_group)
        self._begin_groups.append(real_begin_group)
        self._end_groups.append(real_end_group)
        self._total_row_count += target_row_count

    def get_row_count(self):
      """Get total row count of the combo data.

      Returns:
        An integer, the total number of rows.
      """
      return self._total_row_count




class LakeStreamSharding(object):
    def __init__(self, max_range=65536):
        self._lake_paths = list()
        self._begin_times = list()
        self._end_times = list()
        self._max_range = max_range

    def add_path(self, lake_path, begin_time, end_time=None):
        self._lake_paths.append(lake_path)
        self._begin_times.append(begin_time)
        self._end_times.append(end_time)
        return self

    def partition(
        self, worker_idx, worker_num, slice_per_worker=1, shuffle=False, seed=10
    ):
        assert (
            worker_num > 0
            and slice_per_worker > 0
            and worker_idx >= 0
            and worker_idx < worker_num
        )
        res = []
        for lake_path, begin_time, end_time in zip(
            self._lake_paths, self._begin_times, self._end_times
        ):
            for i in range(slice_per_worker):
                slice_index = slice_per_worker * worker_idx + i
                """
        lake_shard_dir_config is like
        main dir|start time;end time|hash|worker_idx;worker_num
        the complete path is: main dir + shard path
        """
                lake_shard_dir_config = (
                    lake_path + "|begin=" + str(begin_time) + ";end="
                )
                if end_time is not None:
                    lake_shard_dir_config = lake_shard_dir_config + str(end_time) + "|"
                else:
                    lake_shard_dir_config = lake_shard_dir_config + "-1|"
                lake_shard_dir_config = (
                    lake_shard_dir_config + str(self._max_range) + "|"
                )
                lake_shard_dir_config = (
                    lake_shard_dir_config
                    + str(slice_index)
                    + ";"
                    + str(slice_per_worker * worker_num)
                )
                res.append(lake_shard_dir_config)
        if shuffle:
            if seed:
                random.seed(seed)
            random.shuffle(res)
        return res


class LakeBatchSharding(object):
    def __init__(self):
        self._lake_paths = list()

    def add_path(self, lake_path):
        self._lake_paths.append(lake_path)
        return self

    def partition(
        self, worker_idx, worker_num, slice_per_worker=1, shuffle=False, seed=10
    ):
        assert (
            worker_num > 0
            and slice_per_worker > 0
            and worker_idx >= 0
            and worker_idx < worker_num
        )
        res = []
        for lake_path in self._lake_paths:
            for i in range(slice_per_worker):
                slice_index = slice_per_worker * worker_idx + i
                """
                lake_shard_dir_config is like
                main dir|slice_idx;slice_num
                the complete path is: main dir + shard path
                """
                lake_shard_dir_config = lake_path + "|"
                lake_shard_dir_config = lake_shard_dir_config + str(slice_index) + ";" + str(slice_per_worker * worker_num)
                res.append(lake_shard_dir_config)
        if shuffle:
            if seed:
                random.seed(seed)
            random.shuffle(res)
        return res
