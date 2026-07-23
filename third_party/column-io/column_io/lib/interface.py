import os
from column_io.lib import py_interface
import threading

# setup environ for odps plugin
# this kind of code may move to other directory.
os.environ["LIB_ODPS_PLUGIN"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "libodps_plugin.so"
)
os.environ["OPENSTORAGEso"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "libopen_storage_plugin.so"
)
os.environ["LAKERUNTIMEso"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lib_lake_IO.so"
)
os.environ["LOCAL_ORC_so"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "liblocal_orc_plugin.so"
)
#aot runner plugin
os.environ["LIB_PLUGIN_SO"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "libplugin.so"
)
py_interface._global_init()

#open storage
def GetOdpsOpenStorageTableSize(path):
    return py_interface.GetOdpsOpenStorageTableSize(path)

def GetSessionExpireTimestamp(session_id):
    return py_interface.GetSessionExpireTimestamp(session_id)

def InitOdpsOpenStorageSessions(access_id, access_key, tunnel_endpoint, odps_endpoint,
                                projects, tables, partition_specs, physical_partitions,
                                required_data_columns, sep, mode, default_project, connect_timeout, rw_timeout):
    return py_interface.InitOdpsOpenStorageSessions(
                          access_id, access_key, tunnel_endpoint, odps_endpoint,
                          projects, tables, partition_specs, physical_partitions,
                          required_data_columns, sep, mode, default_project, connect_timeout, rw_timeout)

def RegisterOdpsOpenStorageSession(access_id, access_key, tunnel_endpoint, odps_endpoint, project, table, partition, 
                                   required_data_columns, sep, mode, default_project, connect_timeout, rw_timeout,
                                   register_light, session_id, expiration_time, record_count, session_def_str):
    return py_interface.RegisterOdpsOpenStorageSession(access_id, access_key, tunnel_endpoint, odps_endpoint, project, table, partition,
                                                       required_data_columns, sep, mode, default_project, connect_timeout, rw_timeout,
                                                       register_light, session_id, expiration_time, record_count, session_def_str)

def ExtractLocalReadSession(access_id, access_key, project, table, partition):
    return py_interface.ExtractLocalReadSession(access_id, access_key, project, table, partition)

def RefreshReadSessionBatch():
    return py_interface.RefreshReadSessionBatch() 

def GetHaloAgentMetric(halo_endpoint, type):
    # type: (str, str) -> int
    return py_interface.GetHaloAgentMetric(halo_endpoint, type) 

def GetOdpsOpenStorageTableFeatures(str_path, is_compressed):
    return py_interface.GetOdpsOpenStorageTableFeatures(str_path, is_compressed)

def FreeBuffer(ptr):
    return py_interface.FreeBuffer(ptr)

def GetNextFromIterator(iterator, row_mode):
    return py_interface.GetNextFromIterator(iterator, row_mode)


def SerializeIteraterStateToString(iterator):
    return py_interface.SerializeIteraterStateToString(iterator)


def DerializeIteraterStateFromString(iterator, state):
    return py_interface.DerializeIteraterStateFromString(iterator, state)


def MakeIterator(dataset):
    return py_interface.MakeIterator(dataset)


class _ListStringDataset:
    @staticmethod
    def make_dataset(inputs):
        return py_interface._ListStringDataset.make_dataset(inputs)
    
class _ListStringComboDataset:
    @staticmethod
    def make_dataset(inputs):
        return py_interface._ListStringComboDataset.make_dataset(inputs)

class _PackerDataset:
    @staticmethod
    def make_dataset(
        input_dataset,
        batch_size,
        drop_remainder,
        pack_tables,
        num_tables,
        ragged_ranks,
        parallel,
        pinned_result,
        gpu_result,
        do_classify
    ):
        return py_interface._PackerDataset.make_dataset(
            input_dataset,
            batch_size,
            drop_remainder,
            pack_tables,
            num_tables,
            ragged_ranks,
            parallel,
            pinned_result,
            gpu_result,
            do_classify
        )

    @staticmethod
    def make_reorder_dataset(input_dataset, new_order):
        return py_interface._PackerDataset.make_reorder_dataset(
            input_dataset, new_order
        )


class _ParallelDataset:
    @staticmethod
    def make_dataset(
        input_dataset,
        map_fn,
        cycle_length,
        block_length,
        sloppy,
        buffer_output_elements,
        prefetch_input_elements,
    ):
        return py_interface._ParallelDataset.make_dataset(
            input_dataset,
            map_fn,
            cycle_length,
            block_length,
            sloppy,
            buffer_output_elements,
            prefetch_input_elements,
        )


class _PrefetchDataset:
    @staticmethod
    def make_dataset(input, buffer_size):
        return py_interface._PrefetchDataset.make_dataset(input, buffer_size)


class _RepeatDataset:
    @staticmethod
    def make_dataset(input, take_num=1, repeat=-1):
        return py_interface._RepeatDataset.make_dataset(input, take_num, repeat)

class _MapDataSet:
    @staticmethod
    def make_dataset(input_dataset, new_input_schema, old_input_schema, user_module_columns, module_so_path):
        return py_interface._MapDataSet.make_dataset(input_dataset, new_input_schema, old_input_schema, 
                user_module_columns, 
                module_so_path)


class _LocalRBStreamDataset:
    @staticmethod
    def make_dataset(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._LocalRBStreamDataset.make_dataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def make_builder(
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._LocalRBStreamDataset.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def parse_schema(
        paths,
        is_compressed,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._LocalRBStreamDataset.parse_schema(
            paths,
            is_compressed,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

class _OdpsOpenStorageDataset:
    @staticmethod
    def make_dataset(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_xrec,
    ):
        return py_interface._OdpsOpenStorageDataset.make_dataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_xrec,
        )

    @staticmethod
    def make_builder(
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_xrec,
    ):
        return py_interface._OdpsOpenStorageDataset.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_xrec,
        )

    @staticmethod
    def parse_schema(
        paths,
        is_compressed,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._OdpsOpenStorageDataset.parse_schema(
            paths,
            is_compressed,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def get_table_size(path):
        return py_interface._OdpsOpenStorageDataset.get_table_size(path)
    
    # @staticmethod
    # def get_session_expire_timestamp(session_id):
    #     # type: (str)->int
    #     return py_interface._OdpsOpenStorageDataset.get_session_expire_timestamp(session_id)
    @staticmethod
    def load_plugin():
        py_interface._OdpsOpenStorageDataset.load_open_storage_plugin()

    @staticmethod
    def get_table_features(path, is_compressed):
        # logger.debug(f"invoke py_interface._OdpsOpenStorageDataset.get_table_features")
        return py_interface._OdpsOpenStorageDataset.get_table_features(path, is_compressed)


class _OdpsOpenStorageRowDataset:
    @staticmethod
    def make(paths, selected_columns, batch_size, reader_name):
        return py_interface._OdpsOpenStorageRowDataset.make(
            paths, selected_columns, batch_size, reader_name)

    @staticmethod
    def get_table_size(path):
        return py_interface._OdpsOpenStorageRowDataset.get_table_size(path)


class _LocalOrcDataset:
    @staticmethod
    def make_dataset(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._LocalOrcDataset.make_dataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def make_builder(
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._LocalOrcDataset.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def parse_schema(
        paths,
        is_compressed,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._LocalOrcDataset.parse_schema(
            paths,
            is_compressed,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )


class _OdpsTableColumnDataset:
    @staticmethod
    def make_dataset(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._OdpsTableColumnDataset.make_dataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def make_builder(
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._OdpsTableColumnDataset.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def parse_schema(
        paths,
        is_compressed,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._OdpsTableColumnDataset.parse_schema(
            paths,
            is_compressed,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def get_table_size(path):
        return py_interface._OdpsTableColumnDataset.get_table_size(path)
    
    @staticmethod
    def load_plugin():
        py_interface._OdpsTableColumnDataset.load_odps_plugin()

    @staticmethod
    def get_table_features(path, is_compressed):
        return py_interface._OdpsTableColumnDataset.get_table_features(path, is_compressed)


class _OdpsComboDataset:
    @staticmethod
    def make_dataset(
        paths,
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
        primary_key,
        turn_on_odps_open_storage,
    ):
        return py_interface._OdpsComboDataset.make_dataset(
            paths,
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
            primary_key,
            turn_on_odps_open_storage,
        )
    
    @staticmethod
    def make_builder(
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
        primary_key,
        turn_on_odps_open_storage,
    ):
        return py_interface._OdpsComboDataset.make_builder(
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
            primary_key,
            turn_on_odps_open_storage,
        )

    @staticmethod
    def parse_schema(
        paths,
        is_compressed,
        selected_columns,
        hash_features,
        dense_columns,
        dense_defaults,
    ):
        return py_interface._OdpsComboDataset.parse_schema(
            paths,
            is_compressed,
            selected_columns,
            hash_features,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def get_table_size(path):
        return py_interface._OdpsComboDataset.get_table_size(path)
    
    @staticmethod
    def load_plugin():
        py_interface._OdpsComboDataset.load_odps_plugin()

    @staticmethod
    def get_table_features(path, is_compressed):
        return py_interface._OdpsComboDataset.get_table_features(path, is_compressed)



_register_lock = threading.Lock()
_first_resiter = False


def _register_exit_hook(fn):
    import atexit

    global _register_lock
    global _first_resiter
    _register_lock.acquire_lock()
    if not _first_resiter:
        _first_resiter = True

        @atexit.register
        def close_pangu():
            fn()

    _register_lock.release()


class _LakeStreamColumnDataset:
    @staticmethod
    def make_dataset(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        return py_interface._LakeStreamColumnDataset.make_dataset(
            paths,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            is_compressed,
            batch_size,
            use_prefetch,
            prefetch_thread_num,
            prefetch_buffer_size,
        )

    @staticmethod
    def make_builder(
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        return py_interface._LakeStreamColumnDataset.make_builder(
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            is_compressed,
            batch_size,
            use_prefetch,
            prefetch_thread_num,
            prefetch_buffer_size,
        )

    @staticmethod
    def parse_schema(
        paths,
        is_compressed,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        _register_exit_hook(_LakeStreamColumnDataset.close_pangu)
        return py_interface._LakeStreamColumnDataset.parse_schema(
            paths,
            is_compressed,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def close_pangu():
        return py_interface._LakeStreamColumnDataset.close_pangu()
    
class _LakeMultiCFStreamColumnDataset:
    @staticmethod
    def make_dataset(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        return py_interface._LakeMultiCFStreamColumnDataset.make_dataset(
            paths,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            is_compressed,
            batch_size,
            use_prefetch,
            prefetch_thread_num,
            prefetch_buffer_size,
        )

    @staticmethod
    def make_builder(
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        return py_interface._LakeMultiCFStreamColumnDataset.make_builder(
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            is_compressed,
            batch_size,
            use_prefetch,
            prefetch_thread_num,
            prefetch_buffer_size,
        )

    @staticmethod
    def parse_schema(
        paths,
        is_compressed,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        _register_exit_hook(_LakeMultiCFStreamColumnDataset.close_pangu)
        return py_interface._LakeMultiCFStreamColumnDataset.parse_schema(
            paths,
            is_compressed,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def close_pangu():
        return py_interface._LakeMultiCFStreamColumnDataset.close_pangu()

class _LakeBatchColumnDataset:
    @staticmethod
    def make_dataset(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        return py_interface._LakeBatchColumnDataset.make_dataset(
            paths,
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            is_compressed,
            batch_size,
            use_prefetch,
            prefetch_thread_num,
            prefetch_buffer_size,
        )

    @staticmethod
    def make_builder(
        is_compressed,
        batch_size,
        selected_columns,
        input_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        return py_interface._LakeBatchColumnDataset.make_builder(
            selected_columns,
            input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            is_compressed,
            batch_size,
            use_prefetch,
            prefetch_thread_num,
            prefetch_buffer_size,
        )

    @staticmethod
    def parse_schema(
        paths,
        is_compressed,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        # type: (list[str], bool, set[str], set[str], set[str], dict[str, any]) -> tuple[list[str], dict[str, any]]
        _register_exit_hook(_LakeBatchColumnDataset.close_pangu)
        return py_interface._LakeBatchColumnDataset.parse_schema(
            paths,
            is_compressed,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )
    @staticmethod
    def parse_schema_by_rows(
        paths,
        is_compressed,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        # type: (list[str], bool, list[str], set[str], set[str], dict[str, any]) -> tuple[list[str], dict[str, any]]
        _register_exit_hook(_LakeBatchColumnDataset.close_pangu)
        return py_interface._LakeBatchColumnDataset.parse_schema_by_rows(
            paths,
            is_compressed,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def close_pangu():
        return py_interface._LakeBatchColumnDataset.close_pangu()

class _MapDatasetRank:
    @staticmethod
    def make_dataset(input_dataset, new_input_schema, old_input_schema, scene_map):
        return py_interface._MapDataSetRank.make_dataset(input_dataset, new_input_schema, old_input_schema, scene_map)


class _MapDatasetRankCanXi:
    @staticmethod
    def make_dataset(input_dataset, new_input_schema, old_input_schema, odl_mode):
        return py_interface._MapDataSetRankCanXi.make_dataset(input_dataset, new_input_schema, old_input_schema, odl_mode)


class _MapDatasetSampleFilter:
    @staticmethod
    def make_dataset(input_dataset, new_input_schema, old_input_schema, filter_dict):
        return py_interface._MapDataSetSampleFilter.make_dataset(
            input_dataset, new_input_schema, old_input_schema, filter_dict
        )
