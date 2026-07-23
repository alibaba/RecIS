# -*- coding: utf8 -*-
import os
from typing import Any, Callable
import multiprocessing
import json
import copy

from column_io.lib import interface
from column_io.dataset.nest import pack_nest_sequence, _pack_nest_sequence_internal
from column_io.dataset.nest import nest_seq_leaf_num
from column_io.dataset.config import LakeConfig
from column_io.dataset.job_info import get_odps_endpoint
from column_io.dataset.log_util import logger, varlogger, init_openstorage_logger
try:
    from column_io.dataset.odps_env_setup import ensure_standard_path_format, \
        is_turn_on_odps_open_storage, init_odps_open_storage_session
except:
    pass

kPlaceHolder = None


class Dataset:
    def __init__(self, impl) -> None:
        self._impl = impl

    def __iter__(self):
        iterator = interface.MakeIterator(self.impl())
        return Iterator(iterator, self)

    def impl(self):
        """
        get c++ reference of Dataset,
          user should not access this.
        """
        return self._impl

    @property
    def schema(self):
        raise NotImplemented("out_names not implemented")


    @property
    def schema_type(self):
        """
        Returns a dict of {name: type}, or empty dict if not applicable.
        """
        return {}



    @staticmethod
    def from_list_string(array):
        """
        Args:
          array: a list of filenames.
        """
        if array and all(isinstance(group, tuple) or isinstance(group, list) for group in array):
            return SliceListStringComboDataset(array)
        else:
            return SliceListStringDataset(array)
    """ 
    @staticmethod
    def from_list_string_combo(array):
        return SliceListStringComboDataset(array)
    """

    @staticmethod
    def from_rb_files(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return LocalRBStreamDataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def from_orc_files(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return LocalOrcDataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )


    @staticmethod
    def from_odps_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_xrec=False,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          hash_types: hash functions such as: farm, murmur.
          hash_buckets: buckets to hash.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
          use_xrec: flag suggesting the user is xrec, print slice
            opening and out of range info.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        if is_turn_on_odps_open_storage():
            odps_dataset_func = OdpsOpenStorageDataset
        else:
            odps_dataset_func = OdpsTableColumnDataset
        return odps_dataset_func(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_xrec,
        )
    
    @staticmethod
    def from_open_storage_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_xrec=False,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          hash_types: hash functions such as: farm, murmur.
          hash_buckets: buckets to hash.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return OdpsOpenStorageDataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_xrec,
        )
    
    @staticmethod
    def from_common_io_odps_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        dense_columns,
        dense_defaults,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return Dataset.from_odps_source(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            [], [],
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def from_odps_combo_source(
        paths,
        combo_group_size,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        table_ids,
        check_data,
        primary_key,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          hash_types: hash functions such as: farm, murmur.
          hash_buckets: buckets to hash.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return OdpsComboDataset(
            paths,
            combo_group_size,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            table_ids,
            check_data,
            primary_key,
        )

    @staticmethod
    def from_lake_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features=None,
        hash_types=None,
        hash_buckets=None,
        dense_columns=None,
        dense_defaults=None,
        use_prefetch=False,
        prefetch_thread_num=1,
        prefetch_buffer_size=1024,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          hash_types: hash functions such as: farm, murmur.
          hash_buckets: buckets to hash.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
          use_prefetch: use lake prefetch or not.
          prefetch_thread_num: lake prefetch thread num.
          prefetch_buffer_size: lake prefetch buffer size.
        """
        if hash_features is None:
            hash_features = []
        if hash_types is None:
            hash_types = []
        if hash_buckets is None:
            hash_buckets = []
        if dense_columns is None:
            dense_columns = []
        if dense_defaults is None:
            dense_defaults = []
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        if len(paths) == 0:
            raise ValueError("paths must not be empty")
        IoClass = LakeStreamColumnDataset
        if LakeConfig.is_inc_path(paths if isinstance(paths, (str, bytes)) else paths[0]):
            IoClass = LakeMultiCFStreamColumnDataset 
        return IoClass(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_prefetch,
            prefetch_thread_num,
            prefetch_buffer_size,
        )
    
    @staticmethod
    def from_lake_batch_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch=False,
        prefetch_thread_num=1,
        prefetch_buffer_size=1024,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          hash_types: hash functions such as: farm, murmur.
          hash_buckets: buckets to hash.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
          use_prefetch: use lake prefetch or not.
          prefetch_thread_num: lake prefetch thread num.
          prefetch_buffer_size: lake prefetch buffer size.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return LakeBatchColumnDataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_prefetch,
            prefetch_thread_num,
            prefetch_buffer_size,
        )
    
    def parallel(
        self,
        transfunc,
        cycle_length,
        block_length=1,
        sloppy=True,
        buffer_output_elements=1,
        prefetch_input_elements=0,
    ):
        """
        Args:
          transfunc: a sample map function accept a path as arg
            and return a dataset like lambda x: Dataset.from_odps_source([x],....),
            now only support Dataset.from_odps_source.
          cycle_length: The number of input `Dataset`s to interleave from in parallel.
          block_length: The number of consecutive elements to pull from an input
            `Dataset` before advancing to the next input `Dataset`.
          sloppy: If false, elements are produced in deterministic order. Otherwise,
            the implementation is allowed, for the sake of expediency, to produce
            elements in a non-deterministic order.
          buffer_output_elements: The number of elements each iterator being
            interleaved should buffer (similar to the `.prefetch()` transformation for
            each interleaved iterator).
          prefetch_input_elements: The number of input elements to transform to
            iterators before they are needed for interleaving.
        """
        return ParallelDataset(
            self,
            transfunc,
            cycle_length,
            block_length,
            sloppy,
            buffer_output_elements,
            prefetch_input_elements,
        )

    def pack(self, batch_size, drop_remainder, parallel=None, pinned_result=False, gpu_result=False,
             user_define_module=None, dense_columns=None, dense_default_value=None, compress=True):
        """
        pack will pack the output of dataset to
          specified `batch_size`.
        Args:
          batch_size: the needed batch size.
          drop_remainder: specify if the reset of data should be dropped
            when the reset of data is not enough to build a output with `batch_size`
          parallel: size of thread pool to process, if None if will be set to
            number of cpu cores
          pinned_result: If true, the packed result will be on pinned memory.
          gpu_result: If true, the packed result will be on gpu memory.
            This argument will overwrite `pinned_result`.
          compress: If true, The compressed table will be treated as a compressed table.
        """
        if parallel is None:
            parallel = multiprocessing.cpu_count()
        if parallel <= 0:
            raise ValueError("parallel must > 0 but get [{}]".format(parallel))
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return PackDataset(self, batch_size, drop_remainder, parallel, pinned_result,
                           gpu_result, user_define_module, dense_columns, dense_default_value, compress=compress)

    def repeat(self, take_num=1, repeat=-1):
        """
        take `take_num` batch from source dataset,
          and repeat `repeat` times.
        Args:
          take_num: the number of batch to take
            from source dataset.
          repeat: the repeat times on cached
            dataset, `-1` means repeat infinitly.
        """
        return RepeatDataset(self, take_num, repeat)

    def prefetch(self, buffer_size=1):
        """
        prefetch `buffer_size` batch from source dataset
          to make full use of cpu.
        Args:
          buffer_size: number of batch to take from
            source dataset.
        """
        return PrefetchDataset(self, buffer_size)

    def map(self, name="map_dataset_rank", kargs=None):
        if kargs is None:
            kargs = {}
        kargs['input'] = self
        return MapDatasetRegistry().create(name, **kargs)

class Iterator:
    def __init__(self, iterator_impl, dataset: Dataset) -> None:
        self._iterator_impl = iterator_impl
        self._iterator_row_mode :bool = os.environ.get("ODPS_DATASET_ROW_MODE", "0") == "1" # TODO: support arg-style configuration
        self._dataset = dataset

    def __next__(self):
        # combo_mode
        if isinstance(self._dataset, SliceListStringComboDataset):
            return _pack_nest_sequence_internal(self.schema, interface.GetNextFromIterator(self._iterator_impl, self._iterator_row_mode), lambda x: x, 0)[0]

        # row_mode needn't pack array output according to schema, just keep list of row format
        # col_mode,however. need pack array output according to schema, reorder into map dict from name to col-batch
        if self._iterator_row_mode:
            # type: list[tuple[object]]
            return interface.GetNextFromIterator(self._iterator_impl, self._iterator_row_mode)
        else:
            # type: map[string, array[object]]
            return pack_nest_sequence(
                self.schema, interface.GetNextFromIterator(self._iterator_impl, self._iterator_row_mode)
            )

    @property
    def schema(self):
        return self._dataset.schema

    def serialize(self):
        """
        serialize the states of iterator to string.
        Retruns:
          a string
        NOTE: protobuf is used, the size of state should not be too large
          or a empty string will returned.
        """
        return interface.SerializeIteraterStateToString(self._iterator_impl)

    def deserialize(self, states):
        """
        deserialize the states of iterator from string
        Args:
          states: a string contain the state of iterator.
        """
        interface.DerializeIteraterStateFromString(self._iterator_impl, states)


class SliceListStringDataset(Dataset):
    _internal = interface._ListStringDataset

    def __init__(self, array) -> None:
        super().__init__(self._internal.make_dataset(array))

    def schema(self):
        return kPlaceHolder

class SliceListStringComboDataset(Dataset):
    _internal = interface._ListStringComboDataset

    def __init__(self, array) -> None:
        super().__init__(self._internal.make_dataset(array))

    def schema(self):
        return kPlaceHolder

class ParallelDataset(Dataset):
    _internal = interface._ParallelDataset

    def __init__(
        self,
        input: Dataset,
        transfunc,
        cycle_length,
        block_length,
        sloppy,
        buffer_output_elements,
        prefetch_input_elements,
    ) -> None:
        self._input = input
        next_iter_input = next(iter(input))
        # logger.debug(f"ParallelDataset::next_iter_input is {next_iter_input}")
        out_dataset = transfunc(next_iter_input) 
        self._input_builder = out_dataset.builder
        if not issubclass(type(out_dataset), Dataset):
            raise RuntimeError(
                "output type of transfunc must be of type {}".format(type(Dataset))
            )
        self._schema = out_dataset.schema
        self._schema_type = out_dataset.schema_type
        super().__init__(
            self._internal.make_dataset(
                input.impl(),
                self._input_builder,
                cycle_length,
                block_length,
                sloppy,
                buffer_output_elements,
                prefetch_input_elements,
            )
        )

    @property
    def schema(self):
        return self._schema

    @property
    def schema_type(self):
        return self._schema_type


class PackDataset(Dataset):
    _internal = interface._PackerDataset
    _internal_map_dataset = interface._MapDataSet

    def __init__(
        self, input: Dataset, batch_size, drop_remainder=False, parallel=None, pinned_result=False, gpu_result=False,
        user_define_module = None,
        dense_columns = None, 
        dense_default_value = None,
        compress=True
        ) -> None:
        self._input = input
        self._preorder_dataset_impl = None
        self._compress = compress
        if user_define_module is not None: 
          from column_io.aot_compile.aot_module import UserDefineModule
          new_input_schema = copy.deepcopy(self._input.schema)
          old_input_schema = copy.deepcopy(self._input.schema)
          new_input_schema[0]['_sample_group_id'] = [['Placeholder']]
          old_elem_num = nest_seq_leaf_num(old_input_schema)
          old_elem_indice = list(range(old_elem_num))
          new_elem_num = nest_seq_leaf_num(new_input_schema)
          new_elem_indice = list(range(new_elem_num))
          schema_map_with_pos_new = pack_nest_sequence(new_input_schema, new_elem_indice)
          schema_map_with_pos_old = pack_nest_sequence(old_input_schema, old_elem_indice)
          logger.debug(f'schema_map_with_pos_new is {schema_map_with_pos_new}, schema_map_with_pos_old is {schema_map_with_pos_old}, self._input.schema is {self._input.schema} ')
          user_define_module.fill_compiler(self._input.schema, self._input.schema_type, dense_columns, dense_default_value, batch_size)
          self._preorder_dataset_impl = self._internal_map_dataset.make_dataset(
              self._input.impl(), schema_map_with_pos_new, schema_map_with_pos_old, 
              user_define_module.get_user_module_columns(), user_define_module.get_aot_so_path()
          )
          self._input.schema[0]['_sample_group_id'] = [['Placeholder']]
        else:
          self._preorder_dataset_impl = self._input.impl()
        self._make_reorder_info()
        self._preorder_dataset = self._internal.make_reorder_dataset(
            self._preorder_dataset_impl, self._new_indice
        )
        super().__init__(
            self._internal.make_dataset(
                self._preorder_dataset,
                batch_size,
                drop_remainder,
                self._pack_tables,
                self._num_tables,
                self._ragged_ranks,
                parallel,
                pinned_result,
                gpu_result,
                self._do_classify
            )
        )
        self._postorder_dataset = self._internal.make_reorder_dataset(
            self._impl, self._reverse_indice
        )

    def _make_reorder_info(self):
        input_schema = self._input.schema
        elem_num = nest_seq_leaf_num(input_schema)
        elem_indice = list(range(elem_num))
        schema_map_with_pos = pack_nest_sequence(input_schema, elem_indice)
        self._pack_tables = []
        self._num_tables = len(schema_map_with_pos)
        self._values_t = []
        self._splits_t = []
        self._names_t = []
        self._ragged_ranks = []
        self._indicators_t = []
        self._group_id_t = []
        self._do_classify = False
        for table_idx, dic in enumerate(schema_map_with_pos):
            for feature in sorted(dic):
                positions_tuple = dic[feature]
                for positions in positions_tuple:
                    if feature.startswith("_indicator") and self._compress:
                        self._indicators_t.append(positions[0])
                        continue
                    if feature.startswith("_sample_group_id"):
                        self._group_id_t.append(positions[0])
                        self._do_classify = True
                        continue
                    self._names_t.append(feature)
                    self._values_t.extend(positions[:1])
                    self._splits_t.extend(positions[1:])
                    self._ragged_ranks.append(len(positions) - 1)
                    self._pack_tables.append(table_idx)
        if self._do_classify:
          self._new_indice = self._group_id_t + self._indicators_t + self._values_t + self._splits_t
        else:
          self._new_indice = self._indicators_t + self._values_t + self._splits_t
        indice_map = {new: ori for ori, new in enumerate(self._new_indice)}
        self._reverse_indice = [
            indice_map[index] for index in range(len(self._new_indice))
        ]
        logger.debug(f'self._new_indice is {self._new_indice}, do_classify is {self._do_classify}, '
              f'schema_map_with_pos is {schema_map_with_pos}'
              f'self._ragged_ranks is {self._ragged_ranks}'
              f'self._pack_tables is  {self._pack_tables}')
    def impl(self):
        return self._postorder_dataset

    @property
    def schema(self):
        return self._input.schema


class RepeatDataset(Dataset):
    _internal = interface._RepeatDataset

    def __init__(self, input: Dataset, take_num=1, repeat=-1):
        self._input = input
        super().__init__(self._internal.make_dataset(input.impl(), take_num, repeat))

    @property
    def schema(self):
        return self._input.schema


class PrefetchDataset(Dataset):
    _internal = interface._PrefetchDataset

    def __init__(self, input: Dataset, buffer_size=1) -> None:
        self._input = input
        super().__init__(self._internal.make_dataset(input.impl(), buffer_size))

    @property
    def schema(self):
        return self._input.schema


class OdpsTableColumnDataset(Dataset):
    _internal = interface._OdpsTableColumnDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_xrec,
    ) -> None:
        self._input_columns, self._schema, self._schema_type = self._internal.parse_schema(
            paths,
            is_compressed,
            set(selected_columns),
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )
        super().__init__(
            self._internal.make_dataset(
                paths,
                is_compressed,
                batch_size,
                selected_columns,
                self._input_columns,
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            self._input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def schema_type(self):
        return self._schema_type

    @property
    def builder(self):
        return self._builder

    @staticmethod
    def get_table_size(path):
        # type: (str) -> int
        # TODO: call refresh_odps_io_config
        return OdpsTableColumnDataset._internal.get_table_size(path)
    
    @staticmethod
    def load_plugin():
      interface._OdpsTableColumnDataset.load_plugin()

class OdpsOpenStorageDataset(Dataset):
    _internal = interface._OdpsOpenStorageDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_xrec,
    ) -> None:
        init_openstorage_logger()
        standard_paths = ensure_standard_path_format(paths)
        init_odps_open_storage_session(standard_paths, required_data_columns=selected_columns)
        self._input_columns, self._schema, self._schema_type = self._internal.parse_schema(
            paths,
            is_compressed,
            set(selected_columns),
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )
        super().__init__(
            self._internal.make_dataset(
                paths,
                is_compressed,
                batch_size,
                selected_columns,
                self._input_columns,
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults,
                use_xrec,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            self._input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_xrec,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def schema_type(self):
        return self._schema_type

    @property
    def builder(self):
        return self._builder

    @staticmethod
    def get_table_size(path):
        # type: (str) -> int
        # NOTE: init session will use full-column session to get table size, not always same as reader's session
        standard_paths = ensure_standard_path_format([path])
        init_odps_open_storage_session(standard_paths)
        return interface._OdpsOpenStorageDataset.get_table_size(path)

    @staticmethod
    def get_session_expire_timestamp(session_id):
        # type: (str)->int
        return interface._OdpsOpenStorageDataset.get_session_expire_timestamp(session_id)

    @staticmethod
    def load_plugin():
      interface._OdpsOpenStorageDataset.load_plugin()


class OdpsComboDataset(Dataset):
    _internal = interface._OdpsComboDataset
    _helper_internal = None
    
    @classmethod
    def helper_internal(cls):
        if cls._helper_internal is None:
            cls._helper_internal = interface._OdpsOpenStorageDataset if is_turn_on_odps_open_storage() else \
                              interface._OdpsTableColumnDataset
        return cls._helper_internal

    def __init__(
        self,
        paths: list[list[str]],
        combo_group_size: int,
        is_compressed: bool,
        batch_size: int,
        selected_columns: list[str],
        hash_features: list[str],
        hash_types: list[str],
        hash_buckets: list[int],
        dense_columns: list[str],
        dense_defaults: list[list[float]],
        table_ids=None,
        check_data=True,
        primary_key='',
    ) -> None:
        if not isinstance(combo_group_size, int) or combo_group_size < 2:
            raise ValueError("combo_group_size must be int and larger than 1")

        if not (isinstance(paths, list) and len(paths) > 0 and
                all(isinstance(tg, list) for tg in paths) and
                all(len(tg) == combo_group_size for tg in paths)):
            raise ValueError(
                f"input paths for OdpsComboDataset must be list of list "
                f"with each sublist having length {combo_group_size}, but got {paths}"
            )

        if check_data and (not primary_key or not primary_key.strip()):
           raise ValueError(
               "When 'check_data' is True, 'primary_key' must be a non-empty string. "
               "Got primary_key='{}'".format(primary_key)
           )

        table_ids = [-1] * len(selected_columns) if table_ids is None else table_ids 
        if len(table_ids) != len(selected_columns):
            raise ValueError(
                    f"table_ids length ({len(table_ids)}) does not match "
                    f"selected_columns length ({len(selected_columns)})"
                )
        logger.debug(f"[OdpsComboDataset] table_ids-len:{len(table_ids)}")

        if is_turn_on_odps_open_storage():
            tables = [path for group in paths for path in group]
            standard_paths = ensure_standard_path_format(tables)
            init_odps_open_storage_session(standard_paths)

        table_group = paths[0]
        table_select_columns = [[] for _ in range(len(table_group))]
        table_input_columns =  [[] for _ in range(len(table_group))]
        table_schemas = []
        # output_schema = {}
        # output_schema_type = {}
        table_schemas = self._fetch_schema(table_group, is_compressed)
        logger.debug(f"OdpsComboDataset: table_group = {table_group}, table_select_columns = {table_select_columns}")
        logger.debug(f"table_schemas = {table_schemas}")
        ### assign and check features
        def is_feature_in_schema(feature, schema_set, compressed):
            if feature in schema_set:
                return True
            if compressed:
                if f'{feature}_0' in schema_set or f'{feature}_1' in schema_set:
                    return True
            return False

        for i, schema in enumerate(table_schemas):
            if not is_feature_in_schema(primary_key, schema, is_compressed):
                raise ValueError(None, None, f"Primary key {primary_key} not found in table {i}")
        
        for feature, table_id in zip(selected_columns, table_ids):
            if table_id != -1:
                if not is_feature_in_schema(feature, table_schemas[table_id], is_compressed):
                    raise ValueError(f"cannot find feature {feature} in table {table_id} according to the user")
            else:
                for i, schema in enumerate(table_schemas):
                    if is_feature_in_schema(feature, schema, is_compressed):
                        table_id = i
                        break
                if table_id == -1:
                    raise ValueError("cannot find feature: {} from all tables".format(feature))
            table_select_columns[table_id].append(feature)
        logger.debug(f"table_select_columns = {table_select_columns}")
        
        # 保证用户输入的每个表都有效
        for table_id, table_column in enumerate(table_select_columns):
            if not table_column or (len(table_column) == 1 and table_column[0] == primary_key):
                raise ValueError(
                    f"Table {table_id} ({paths}): "
                    f"no valid features selected (only contains primary key '{primary_key}')"
                )

        # 加入primary key
        table_select_columns = [sublist if primary_key in sublist else sublist + [primary_key] for sublist in table_select_columns]

        # 依次对每个表进行parse schema, 证明可读
        self._schema = [] # type: list[dict[str, list[list[object]] ]]
        self._schema_type = {}
        for i, table_path in enumerate(table_group): # for i in range(0, len(table_group)):
            input_column, schema, schema_type = self.helper_internal().parse_schema(
                [table_group[i]],
                is_compressed,
                set(table_select_columns[i]),
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults
            )
            # schema_type: some str like
            #   '{"uniq_id":"string","_indicator":"int64","141_1":"{k:int64, v:float}",,,"ds":"string"}'
            for group_idx, schema_item in enumerate(schema):
                # output_schema.update(schema[0])
                if len(self._schema) <= group_idx:
                    self._schema.append({})
                self._schema[group_idx].update(schema_item)
                self._schema_type.update( {table_group[i]: schema_type}) # TODO: Make sure the use. needn't now
            table_input_columns[i] = input_column
            assert len(input_column) == (len(set(table_select_columns[i])) + is_compressed)
        # self._schema = [output_schema]
        # self._schema_type = {}
        
        super().__init__(
            self._internal.make_dataset(
                paths,
                is_compressed,
                batch_size,
                table_select_columns,           # 用户输入的所有要读的feature, 分配给各个表了. 但是还是别名
                table_input_columns,            # 刚才分配给各个表的feature, 如果是compressed, 那么加上_0 or _1 
                hash_features,                  # 其中需要进行hash的feature
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults,
                check_data,
                primary_key if not is_compressed else primary_key + "_0",
                is_turn_on_odps_open_storage(),
            )
        )
        logger.debug(f"table_select_columns is {table_select_columns} \ntable_input_columns is {table_input_columns}")
        self._builder = self._internal.make_builder(
            is_compressed,
            batch_size,
            table_select_columns,
            table_input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            check_data,
            primary_key if not is_compressed else primary_key + "_0",
            is_turn_on_odps_open_storage(),
        )

    #TODO: 重构以使用parse_schema, 或odps_endpoint直接走tunnel
    def _fetch_schema(self, table_group, _is_compressed):
        # logger.info(f"DEBUG (f"OdpsComboDataset: _fetch_schema, table_group is {table_group}, _is_compressed is {_is_compressed}")
        table_schemas = []
        import odps
        from odps import ODPS
        from column_io.dataset.secret_util import decode
        ENCODED_ODPS_ACCESS_ID = os.environ.get('ENCODED_ODPS_ACCESS_ID')
        ENCODED_ODPS_ACCESS_KEY = os.environ.get('ENCODED_ODPS_ACCESS_KEY')
        access_id = os.getenv('access_id', decode(ENCODED_ODPS_ACCESS_ID))
        access_key = os.getenv('access_key', decode(ENCODED_ODPS_ACCESS_KEY))
        if access_id is None or str(access_id) == '':
          raise ValueError('access_id and ENCODED_ODPS_ACCESS_ID cannot be both None !')
        if access_key is None or str(access_key) == '':
          raise ValueError('access_key and ENCODED_ODPS_ACCESS_KEY cannot be both None !')
        access_id = str(access_id)
        access_key = str(access_key)
        odps_endpoint = get_odps_endpoint()
        for table in table_group:
            parts = table.strip().split("//")[1].strip().strip('/').split('/') 
            project, table = parts[0], parts[2] 
            logger.info(f"[OdpsComboDataset] fetching-schema project:{project}, table:{table}")
            o = ODPS(access_id=access_id,
                     secret_access_key=access_key,
                     project=project,
                     endpoint=odps_endpoint)
            t = o.get_table(table)
            table_schemas.append([column.name for column in t.schema.columns])
        # logger.info(f"DEBUG OdpsComboDataset: table_schemas = {table_schemas}")
        return table_schemas

    @property
    def schema(self):
        return self._schema

    @property
    def schema_type(self):
        return self._schema_type

    @property
    def builder(self):
        return self._builder

    @staticmethod
    def get_table_size(path):
        # type: (str) -> int
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        if not isinstance(path, str):
            varlogger.info(f"get_table_size with path-type: {type(path)}, value: {path}")
        if is_turn_on_odps_open_storage():
            standard_paths = ensure_standard_path_format([path])
            init_odps_open_storage_session(standard_paths)
        return OdpsComboDataset.helper_internal().get_table_size(path)
    
    @staticmethod
    def load_plugin():
      OdpsComboDataset.helper_internal().load_plugin()

class LocalRBStreamDataset(Dataset):
    _internal = interface._LocalRBStreamDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ) -> None:
        self._input_columns, self._schema, self._schema_type = self._internal.parse_schema(
            paths,
            is_compressed,
            set(selected_columns),
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

        super().__init__(
            self._internal.make_dataset(
                paths,
                is_compressed,
                batch_size,
                selected_columns,
                self._input_columns,
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            self._input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def schema_type(self):
        return self._schema_type

    @property
    def builder(self):
        return self._builder


class LocalOrcDataset(Dataset):
    _internal = interface._LocalOrcDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ) -> None:
        self._input_columns, self._schema, self._schema_type = self._internal.parse_schema(
            paths,
            is_compressed,
            set(selected_columns),
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

        super().__init__(
            self._internal.make_dataset(
                paths,
                is_compressed,
                batch_size,
                selected_columns,
                self._input_columns,
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            self._input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def schema_type(self):
        return self._schema_type

    @property
    def builder(self):
        return self._builder


class LakeStreamColumnDataset(Dataset):
    _internal = interface._LakeStreamColumnDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        self._input_columns, self._schema, self._schema_type = self._internal.parse_schema(
            paths=paths,
            is_compressed=is_compressed,
            selected_columns=set(selected_columns),
            hash_features=hash_features,
            hash_types=hash_types,
            hash_buckets=hash_buckets,
            dense_columns=dense_columns,
            dense_defaults=dense_defaults,
        )

        super().__init__(
            self._internal.make_dataset(
                paths=paths,
                is_compressed=is_compressed,
                batch_size=batch_size,
                selected_columns=selected_columns,
                input_columns=self._input_columns,
                hash_features=hash_features,
                hash_types=hash_types,
                hash_buckets=hash_buckets,
                dense_columns=dense_columns,
                dense_defaults=dense_defaults,
                use_prefetch=use_prefetch,
                prefetch_thread_num=prefetch_thread_num,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed=is_compressed,
            batch_size=batch_size,
            selected_columns=selected_columns,
            input_columns=self._input_columns,
            hash_features=hash_features,
            hash_types=hash_types,
            hash_buckets=hash_buckets,
            dense_columns=dense_columns,
            dense_defaults=dense_defaults,
            use_prefetch=use_prefetch,
            prefetch_thread_num=prefetch_thread_num,
            prefetch_buffer_size=prefetch_buffer_size,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def schema_type(self):
        return self._schema_type

    @property
    def builder(self):
        return self._builder

class LakeMultiCFStreamColumnDataset(Dataset):
    _internal = interface._LakeMultiCFStreamColumnDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        self._input_columns, self._schema, self._schema_type = self._internal.parse_schema(
            paths=paths,
            is_compressed=is_compressed,
            selected_columns=set(selected_columns),
            hash_features=hash_features,
            hash_types=hash_types,
            hash_buckets=hash_buckets,
            dense_columns=dense_columns,
            dense_defaults=dense_defaults,
        )

        super().__init__(
            self._internal.make_dataset(
                paths=paths,
                is_compressed=is_compressed,
                batch_size=batch_size,
                selected_columns=selected_columns,
                input_columns=self._input_columns,
                hash_features=hash_features,
                hash_types=hash_types,
                hash_buckets=hash_buckets,
                dense_columns=dense_columns,
                dense_defaults=dense_defaults,
                use_prefetch=use_prefetch,
                prefetch_thread_num=prefetch_thread_num,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed=is_compressed,
            batch_size=batch_size,
            selected_columns=selected_columns,
            input_columns=self._input_columns,
            hash_features=hash_features,
            hash_types=hash_types,
            hash_buckets=hash_buckets,
            dense_columns=dense_columns,
            dense_defaults=dense_defaults,
            use_prefetch=use_prefetch,
            prefetch_thread_num=prefetch_thread_num,
            prefetch_buffer_size=prefetch_buffer_size,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def schema_type(self):
        return self._schema_type

    @property
    def builder(self):
        return self._builder

class LakeBatchColumnDataset(Dataset):
    _internal = interface._LakeBatchColumnDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        iterator_row_mode :bool = os.environ.get("ODPS_DATASET_ROW_MODE", "0") == "1" # TODO: support arg-style configuration
        parse_schema_func : callable # type: Callable[[str], tuple[str, dict[str, str]] ]
        if not iterator_row_mode:
            parse_schema_func = self._internal.parse_schema
            schema_selected_columns=set(selected_columns)
        else:
            parse_schema_func = self._internal.parse_schema_by_rows
            schema_selected_columns=selected_columns
        
        self._input_columns, self._schema, self._schema_type = parse_schema_func(
            paths=paths,
            is_compressed=is_compressed,
            selected_columns=schema_selected_columns,
            hash_features=hash_features,
            hash_types=hash_types,
            hash_buckets=hash_buckets,
            dense_columns=dense_columns,
            dense_defaults=dense_defaults,
        )

        # Allow empty to select all columns
        if len(selected_columns) == 0:
            selected_columns = self._input_columns

        super().__init__(
            self._internal.make_dataset(
                paths=paths,
                is_compressed=is_compressed,
                batch_size=batch_size,
                selected_columns=selected_columns,
                input_columns=self._input_columns,
                hash_features=hash_features,
                hash_types=hash_types,
                hash_buckets=hash_buckets,
                dense_columns=dense_columns,
                dense_defaults=dense_defaults,
                use_prefetch=use_prefetch,
                prefetch_thread_num=prefetch_thread_num,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed=is_compressed,
            batch_size=batch_size,
            selected_columns=selected_columns,
            input_columns=self._input_columns,
            hash_features=hash_features,
            hash_types=hash_types,
            hash_buckets=hash_buckets,
            dense_columns=dense_columns,
            dense_defaults=dense_defaults,
            use_prefetch=use_prefetch,
            prefetch_thread_num=prefetch_thread_num,
            prefetch_buffer_size=prefetch_buffer_size,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def schema_type(self):
        return self._schema_type

    @property
    def builder(self):
        return self._builder
class MapDatasetRank(Dataset):
    _internal = interface._MapDatasetRank

    def __init__(self, input: Dataset, scene_map) -> None:
        self._input = input
        new_input_schema = copy.deepcopy(self._input.schema)
        old_input_schema = copy.deepcopy(self._input.schema)
        new_input_schema[0]['_sample_group_id'] = [['Placeholder']]
        new_input_schema[0]['scene_flag'] = [['Placeholder']]
        old_elem_num = nest_seq_leaf_num(old_input_schema)
        old_elem_indice = list(range(old_elem_num))
        new_elem_num = nest_seq_leaf_num(new_input_schema)
        new_elem_indice = list(range(new_elem_num))
        schema_map_with_pos_new = pack_nest_sequence(new_input_schema, new_elem_indice)
        schema_map_with_pos_old = pack_nest_sequence(old_input_schema, old_elem_indice)
        super().__init__(
            self._internal.make_dataset(
                input.impl(),
                schema_map_with_pos_new,
                schema_map_with_pos_old,
                scene_map
            )
        )
        self._schema = schema_map_with_pos_new

    @property
    def schema(self):
        return self._schema

class MapDatasetRankCanXi(Dataset):
    _internal = interface._MapDatasetRankCanXi

    def __init__(self, input: Dataset, odl_mode) -> None:
        self._input = input
        new_input_schema = copy.deepcopy(self._input.schema)
        old_input_schema = copy.deepcopy(self._input.schema)
        new_input_schema[0]['_sample_group_id'] = [['Placeholder']]
        old_elem_num = nest_seq_leaf_num(old_input_schema)
        old_elem_indice = list(range(old_elem_num))
        new_elem_num = nest_seq_leaf_num(new_input_schema)
        new_elem_indice = list(range(new_elem_num))
        schema_map_with_pos_new = pack_nest_sequence(new_input_schema, new_elem_indice)
        schema_map_with_pos_old = pack_nest_sequence(old_input_schema, old_elem_indice)
        super().__init__(
            self._internal.make_dataset(
                input.impl(),
                schema_map_with_pos_new,
                schema_map_with_pos_old,
                odl_mode
            )
        )
        self._schema = schema_map_with_pos_new

    @property
    def schema(self):
        return self._schema

class MapDatasetSampleFilter(Dataset):
    """基于 sample_id 字符串字段做行级黑名单过滤的 map dataset.

    Args:
      input: 上游 Dataset, schema 必须含 "sample_id" 字段.
      filter_dict: dict[str, list[str]]. 黑名单语义 (in -> drop) +
        OR-across-keys (任一 key 命中即丢). 空字典表示不过滤.

    Returns:
      新的 Dataset, 输出 schema 比 input 多一个 `_sample_group_id` int64 列
      (-1 = 该行需丢弃, 0 = 保留).

    Raises:
      ValueError: filter_dict 类型不合法, 或 input.schema 不含 "sample_id".

    Note:
      本 dataset 仅 inject `_sample_group_id` 列, 实际的行 drop 在下游
      `.pack()` 内完成 (packer.cc:459-470 自动跳过 group_id<0 的行,
      由 dataset.py:_make_reorder_info 中 `_sample_group_id` -> do_classify
      触发). 因此调用方必须在本 dataset 之后链 `.pack(...)`, 否则过滤静默
      失效, `_sample_group_id` 列将作为普通列穿透至下游.

    sample_id 支持的格式 (生产者侧契约, V1 解析器实现):

      "<prefix>\x01<k1>:<v1>,<k2>:<v2>,...,<kN>:<vN>"

      - 一段 prefix (例 "DIANTAO"), 紧跟一个 \x01 (SOH, ASCII 0x01) 字节
        作为分隔符, 之后是若干 "key:value" 条目, 条目之间用 "," 分隔,
        key 与 value 之间用第一个 ":" 分隔.
      - prefix 段会被解析器忽略, 不参与匹配.
      - value 内允许出现 ":" (例 timestamp "2026-01-01T12:34:56"), 因为
        解析器对每个 token 只在第一个 ":" 处切分一次.
      - value 内 **不允许** 出现 "," (会被错误切成两个 token);
        亦不支持反斜杠转义.
      - sample_id 中不含 "\x01" 时, 整行被视作格式异常, 默认 **保留**
        (denylist 安全侧, 避免误删).

      合法示例:
        "DIANTAO\x01pvid:xxx,entityId:760,productType:H,pid:431451_1007,"
        "subProductType:200^22001^1^11,timestamp:1778168872,..."

      filter_dict 的 key 应当严格匹配 sample_id 中的 key 名 (大小写敏感);
      key 在 sample_id 中不存在则该 key 不参与判定 (该行该 key 不会触发
      drop, 仍由其他命中的 key 决定结果).

    Usage example (lake 数据流, 压缩表):

        from column_io.dataset import dataset as dataset_io
        from column_io.dataset.config import LakeConfig
        from column_io.dataset.file_sharding import LakeStreamSharding

        # 1) 源 dataset: lake 压缩表, 必须把 sample_id 加入 select_column.
        #    以下 LakeConfig 字段为占位符, 替换为业务方实际值.
        lake_config = LakeConfig(
            storageName="<YOUR_LAKE_STORAGE>",      # e.g. "naXX-Y"
            projectName="<YOUR_PROJECT_NAME>",
            tableName="<YOUR_TABLE_NAME>",
            columnFamilyName="<YOUR_COLUMN_FAMILY>",  # 通常 "default_column_family"
            partitionSpec="<YOUR_PARTITION_SPEC>",    # e.g. "current/data"
        )
        sharding = LakeStreamSharding()
        sharding.add_path(
            lake_config.get_v1_path(),
            <START_TIMESTAMP_US>,   # 起始时间, 微秒
            <END_TIMESTAMP_US>,     # 结束时间, 微秒
        )
        ds = dataset_io.Dataset.from_lake_source(
            sharding.partition(0, 1)[0],
            True,              # is_compressed=True (压缩表必须传 True)
            64,                # batch_size
            ["sample_id"],     # selected_columns, 必含 "sample_id"
            [], [], [], [], [],
        )

        # 2) 接 sample_filter: 通过 registry 调用, name 固定 "sample_filter"
        ds = ds.map(
            name="sample_filter",
            kargs={
                "filter_dict": {
                    "pid": ["<DENYLIST_PID_1>", "<DENYLIST_PID_2>"],
                    "subProductType": ["<DENYLIST_SUBTYPE_1>"],
                }
            },
        )

        # 3) 必须接 .pack() 才能让过滤生效 (实际的 row drop 发生在 packer)
        ds = ds.pack(batch_size=64, drop_remainder=True)

        # 4) 迭代: 输出 batch 中的每一行都已保证不命中 filter_dict
        for batch in ds:
            ...  # batch 中含 _sample_group_id 列 (全 0, 可忽略)
    """

    _internal = interface._MapDatasetSampleFilter

    def __init__(self, input: Dataset, filter_dict: dict) -> None:
        self._input = input
        # ---- 防御性参数校验 ----
        # 这些校验放在 Python 层而不是 C++, 是因为 C++ 端的 .at() / std::map
        # 抛 out_of_range 时 stack trace 对业务定位不友好; Python 层 fail-fast
        # 给出清晰的错误信息.
        if not isinstance(filter_dict, dict):
            raise ValueError(
                "filter_dict must be a dict, got {}".format(type(filter_dict))
            )
        for k, v in filter_dict.items():
            if not isinstance(k, str):
                raise ValueError(
                    "filter_dict keys must be str, got key {!r} of type {}".format(
                        k, type(k)
                    )
                )
            if not isinstance(v, (list, tuple)):
                raise ValueError(
                    "filter_dict[{!r}] must be list/tuple, got {}".format(k, type(v))
                )
            if not all(isinstance(x, str) for x in v):
                raise ValueError(
                    "filter_dict[{!r}] must contain only str values".format(k)
                )
        # input.schema 应该是 list[dict], schema[0] 是主表的列名 -> 位置列表.
        # 防御性地校验形状: 某些 Dataset (例如 SliceListStringDataset) 把
        # schema 实现成 method 或返回 kPlaceHolder=None, 直接 `in` 会抛
        # TypeError 而不是清晰的 ValueError, 这里提前 fail-fast.
        schema = self._input.schema
        if (
            not schema
            or not isinstance(schema, (list, tuple))
            or not isinstance(schema[0], dict)
            or "sample_id" not in schema[0]
        ):
            if isinstance(schema, (list, tuple)) and schema and isinstance(schema[0], dict):
                available = sorted(schema[0].keys())
            else:
                available = "<not a list[dict] schema: {!r}>".format(type(schema))
            raise ValueError(
                "MapDatasetSampleFilter requires a list[dict] input.schema with "
                "'sample_id' in schema[0]; got available columns: {}".format(available)
            )

        # 友好提示: 空 filter_dict 不会过滤任何行 (C++ 端走 fast-path)
        if len(filter_dict) == 0:
            try:
                logger.warning(
                    "[MapDatasetSampleFilter] filter_dict is empty; this is a no-op."
                )
            except Exception:
                # logger 在某些 import 环境下未初始化, 不要因为日志失败影响功能
                pass

        # 转 dict[str, list[str]]: pybind11/stl.h 自动转换为 C++
        # std::map<std::string, std::vector<std::string>> (与 K2 scene_map
        # 的 std::map<std::string, int32_t> 转换路径同源).
        normalized_filter = {k: list(v) for k, v in filter_dict.items()}

        # ---- schema 注入 _sample_group_id ----
        # 这一段 1:1 抄自 MapDatasetRankCanXi.__init__ (本文件上方 ~30 行处).
        # 关键点: 在 input.schema[0] 上插一个 '_sample_group_id' 哨兵列,
        # 这样 pack_nest_sequence 算出来的位置就是 C++ Iterator 需要的
        # group_id_t_pos.
        new_input_schema = copy.deepcopy(self._input.schema)
        old_input_schema = copy.deepcopy(self._input.schema)
        new_input_schema[0]["_sample_group_id"] = [["Placeholder"]]
        old_elem_num = nest_seq_leaf_num(old_input_schema)
        old_elem_indice = list(range(old_elem_num))
        new_elem_num = nest_seq_leaf_num(new_input_schema)
        new_elem_indice = list(range(new_elem_num))
        schema_map_with_pos_new = pack_nest_sequence(new_input_schema, new_elem_indice)
        schema_map_with_pos_old = pack_nest_sequence(old_input_schema, old_elem_indice)

        super().__init__(
            self._internal.make_dataset(
                input.impl(),
                schema_map_with_pos_new,
                schema_map_with_pos_old,
                normalized_filter,
            )
        )
        self._schema = schema_map_with_pos_new

    @property
    def schema(self):
        return self._schema


class MapDatasetRegistry(object):
    _instance = None
    _registry = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MapDatasetRegistry, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def register(self, key, cls_target):
        if key in self._registry:
            raise ValueError(f"Key '{key}' duplicated in registry!")
        print(f"[INFO] [dataset.py] register {key} ---> {cls_target.__name__}")
        self._registry[key] = cls_target

    def get_class(self, key):
        cls_target = self._registry.get(key)
        if not cls_target:
            raise ValueError(f"Key '{key}' not found in registry!")
        return cls_target

    def create(self, key, *args, **kwargs):
        cls_target = self.get_class(key)
        print(f"[INFO] [dataset.py] create map dataset [{key}] with args: ", args, kwargs)
        return cls_target(*args, **kwargs)

    def list_all(self):
        return list(self._registry.keys())


map_dataset_registry = MapDatasetRegistry()
map_dataset_registry.register("k2_prerank", MapDatasetRank)
map_dataset_registry.register("k2_rank_canxi", MapDatasetRankCanXi)
map_dataset_registry.register("sample_filter", MapDatasetSampleFilter)
