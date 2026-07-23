#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A wrapper of Openstorage Python SDK, for reading/writing ODPS table."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import time
import logging
import datetime
import numpy
import urllib3
from typing import Any,Tuple
from io import open

# from torch import Tensor
from odps import ODPS,models,types
from odps import options
from odps.errors import ODPSError
from column_io.dataset.dataset import Dataset
from column_io.dataset.config import is_external_cluster
from column_io.dataset.log_util import logger
from column_io.dataset.open_storage_utils import decode


options.connect_timeout = 300 # TODO(xyh): 调整halo超时. 但是这个参数在本文件里实际没用到
DefaultReadBatch = 1024 # [1, 20000], 1024 is experienced recommend

# Just fork the belowing typedict to odps_type_to_pytype in common_io.table.typedict, for compatibility
'''
typedict = {
    'bigint': int,
    'double': float,
    'boolean': bool,
    'string': object,
    'datetime': int,
    'map': object,
    'array': object, # NOTE(siran.ysr): TableReader may not support read array yet.
    'unknown': object
}
const std::map<ColumnType, std::string> kTypeStrMap({
    {kInt, "bigint"},
    {kBool, "boolean"},
    {kString, "string"},
    {kDateTime, "datetime"},
    {kDouble, "double"},
    {kMap, "map"},
    {kArray, "array"},
    {kUnknown, "unknown"},
});
'''
class TableSchemaType:
    def __init__(self, t, t_str):
        # type: (type, str) -> TableSchemaType
        self.pytype = t # type: type
        self.typestr = t_str # type: str

odps_type_to_pytype = {
    types.Tinyint:  TableSchemaType(int, "bigint"),
    types.Smallint: TableSchemaType(int, "bigint"),
    types.Int:      TableSchemaType(int, "bigint"),
    types.Bigint:   TableSchemaType(int, "bigint"),
    types.Float:    TableSchemaType(float, "double"),
    types.Double:   TableSchemaType(float, "double"),
    types.Boolean:  TableSchemaType(bool, "boolean"),
    types.String:   TableSchemaType(str, "string"),
    
    types.Array:    TableSchemaType(list, "array"),
    # 未测试类型:
    types.Binary:   TableSchemaType(object, "string"), # default string, incase for some unrecognized codec

    # types.Datetime: (int, "bigint"),
    # types.Timestamp: (int, "bigint"),
} # type: dict[ types.ColumnType , TableSchemaType ]

def _parse_table_path(odps_table_path):
    """Method that parse odps table path.
    """
    str_list = odps_table_path.split("/")
    if len(str_list) < 5 or str_list[3] != "tables":
        raise ValueError(
            "'%s' is invalid, please refer: 'odps://${your_projectname}/"
            "tables/${table_name}/${pt_1}/${pt_2}/...'" % odps_table_path)

    table_partition = ",".join(str_list[5:])
    if not table_partition:
        table_partition = None
    table_info = {
        "odps project": str_list[2],
        "table name": str_list[4],
        "table partition": table_partition}
    logger.debug("%s -> %s", odps_table_path, str(table_info))
    return (str_list[2], str_list[4], table_partition)

def _try_get_table_range(table_path : str) -> Tuple[int, int, bool]:
    sep_count = table_path.count("?")
    if sep_count == 1:
        if not re.match(r'(.)*(?:\?start=\d+&end=\d+)$', table_path):
            raise ValueError('cannot parse range since invalid table range format: {}'.format(table_path))
        slice_range = table_path.split('?')[-1]
        start_end = slice_range.split('&')
        slice_start = int(start_end[0].split('=')[-1])
        slice_end = int(start_end[1].split('=')[-1])
        return (slice_start, slice_end, True)
    elif sep_count > 1:
        raise ValueError("Invalid table path {}".format(table_path))
        # logger.error("skip range since invalid table range format: {}".format(table_path))
        # return ( 0, 0, False)
    return ( 0, 0, False)

class Distributor(object):  # entirely copied from common_io.table_tunnel.py
  """Determine the table reading position according to the slice information
  """
  def __init__(self, row_count, slice_id, slice_count):
    size = int(row_count / slice_count)
    split_point = row_count % slice_count
    if slice_id < split_point:
      self._start = slice_id * (size + 1)
      self._end = self._start  + (size + 1)
    else:
      self._start = split_point * (size + 1) + (slice_id - split_point) * size
      self._end = self._start + size

  @property
  def start(self):
    return self._start

  @property
  def end(self):
    return self._end

def StrSplit(text, sep, ignore_empty):
    result = []
    old = 0
    while True:
        n = text.find(sep, old)
        if n == -1:
            break
        if not ignore_empty or old != n:
            result.append(text[old:n])
        old = n + len(sep)

    if not ignore_empty or old < len(text):
        result.append(text[old:])
    return result

kProjectNameTag = "project_name"
kTableNameTag = "talbe_name"
KPartitionSpecTag = "partition_spec"
kStartTag = "start"
kEndTag = "end"
def odps_table_path_parse_impl(odps_table_path: str):
    ret = dict()

    # extract project_name
    new_pos = odps_table_path.find("//")
    new_pos += 2
    old_pos = new_pos
    new_pos = odps_table_path.find("/", new_pos)
    ret[kProjectNameTag] = odps_table_path[old_pos:new_pos]

    # extract table_name
    new_pos = odps_table_path.find("tables")
    new_pos += 7
    old_pos = new_pos
    new_pos = odps_table_path.find("/", new_pos)
    tmp_pos = odps_table_path.find("?")
    is_partition_set = False
    if new_pos == -1:  # for none-partition table
        if tmp_pos == -1:  # for none-partition table without ?start=xx&end=xx
            ret[kTableNameTag] = odps_table_path[old_pos:]
            ret[KPartitionSpecTag] = ""
            is_partition_set = True
            return ret
        else:  # for none-partition table with ?start=xx&end=xx
            ret[kTableNameTag] = odps_table_path[old_pos:tmp_pos]
            ret[KPartitionSpecTag] = ""
            is_partition_set = True
    else:
        ret[kTableNameTag] = odps_table_path[old_pos:new_pos]
    
    # 兼容tunnel传参格式
    parts = [p.strip() for p in ret[kTableNameTag].split(".")]  # 过滤空片段（如 ".."）
    n = len(parts)

    if n == 1:
        ret[kTableNameTag] = parts[0]
    elif n == 2:
        ret[kProjectNameTag] = parts[0] if parts[0] else ret[kProjectNameTag]
        ret[kTableNameTag] = parts[1] if parts[1] else ret[kTableNameTag]
    elif n==3:
        ret[kProjectNameTag] = parts[0] if parts[0] else ret[kProjectNameTag]
        ret[kTableNameTag] = parts[2] if parts[2] else ret[kTableNameTag]
    else:
        raise ValueError("Invalid table path {}".format(odps_table_path))

    # extract partitions
    new_pos += 1
    old_pos = new_pos
    if not is_partition_set:
        new_pos = odps_table_path.find("?", new_pos)
        if new_pos == -1:
            ret[KPartitionSpecTag] = odps_table_path[old_pos:]
        else:
            ret[KPartitionSpecTag] = odps_table_path[old_pos:new_pos]

    # extract param
    if new_pos == -1:
        return ret
    new_pos = odps_table_path.find("?")
    new_pos += 1
    old_pos = new_pos
    param_str = odps_table_path[old_pos:]
    param_str_vec = StrSplit(param_str, "&", False)
    for param in param_str_vec:
        single_param_vec = StrSplit(param, "=", False)
        ret[single_param_vec[0]] = single_param_vec[1]
    return ret

'''
odps_table_path_parse(odps_table_path :str) -> (project : str, table : str, parts : str)
'''
def odps_table_path_parse(odps_table_path: str):
    try:
        path = odps_table_path.split(",")[0]
        parts_dict = odps_table_path_parse_impl(path)
        return parts_dict[kProjectNameTag], parts_dict[kTableNameTag], "/".join(parts_dict[KPartitionSpecTag])  # project, table, parts
    except Exception as e:
        logger.error("failed to parse path: {}".format(path))
        raise e
'''
exist_table(odps : ODPS, project, table_name) -> None
'''
def exist_table(odps : ODPS, project, table_name):
    retry = 3
    while retry > 0:    # for non_partitioned table, maybe get cluster meta file error in concurrent mode
        try:
            exist = odps.exist_table(table_name, project)
            if not exist:
                raise ValueError(f"odps_project_table {table_name} not exist.")
            return
        except ODPSError as err:
            retry -= 1
            logger.warning(f"exist_table error: {err}")
            time.sleep(1)

class OutOfRangeException(Exception):
    '''
    OutOfRangeException, This exception is raised in "end-of-file" conditions
    '''
    def __init__(self, err_msg):
        info = "Out of range exception: {}".format(err_msg)
        Exception.__init__(self, info)
'''
override OutOfRangeException to EOFError
1. To ensure compatibility between OutOfRangeException in common_io and column_io
2. Why not import OutOfRangeException from columnio?
   For avoiding issues due to load order dependencies.
3. EOFError does not cause unintended catches and also carries the semantics of 'end of data'.
   Ref:  https://docs.python.org/zh-cn/3.10/library/exceptions.html#EOFError

Related CR:
   common_io:
      https://code.alibaba-inc.com/algo/common-io/codereview/23255507
   column_io:
      https://code.alibaba-inc.com/alimama-data-infrastructure/column-io/codereview/23255515
'''
OutOfRangeException = EOFError
            
class OpenStorageConf(object):
    """Configuration information of OpenStorage.
    """
    def __init__(self):
        self._parse_odps_config_file()
        ENCODED_ODPS_ACCESS_ID = os.environ.get('ENCODED_ODPS_ACCESS_ID')
        ENCODED_ODPS_ACCESS_KEY = os.environ.get('ENCODED_ODPS_ACCESS_KEY')
        _ACCESS_ID = os.getenv('ACCESS_ID')
        _ACCESS_KEY = os.getenv('ACCESS_KEY')
        access_id = decode(ENCODED_ODPS_ACCESS_ID) if ENCODED_ODPS_ACCESS_ID else _ACCESS_ID
        access_key = decode(ENCODED_ODPS_ACCESS_KEY) if ENCODED_ODPS_ACCESS_KEY else _ACCESS_KEY
        self.account_id = access_id
        if not self.account_id:
            raise ValueError("access_id is None.")
        self.account_key = access_key
        if not self.account_key:
             raise ValueError("access_key is None.")
        self.account_type = os.getenv('ACCOUNT_TYPE', 'aliyun')
        self.odps_endpoint = os.getenv('ODPS_ENDPOINT', "")
        if not self.odps_endpoint:
            raise ValueError("odps_endpoint is None.")

    def _parse_odps_config_file(self):
        default_odps_config_path = os.getenv("HOME", "/home/admin") + "/.odps_config.ini"
        odps_config_path = os.getenv("ODPS_CONFIG_FILE_PATH", default_odps_config_path)
        if not os.path.exists(odps_config_path):
            return
        odps_config = {}
        with open(odps_config_path, 'r') as f:
            for line in f.readlines():
                values = line.split('=', 1)
                if len(values) == 2:
                    odps_config[values[0]] = values[1].strip()
        try:
            os.environ['ACCESS_ID'] = odps_config['access_id']
            os.environ['ACCESS_KEY'] = odps_config['access_key']
            os.environ['ODPS_ENDPOINT'] = odps_config['end_point']
        except KeyError as err:
            raise IOError("'%s' does not exist in the %s file." \
                                        % (err.message, odps_config_path))

class OpenstorageClient(object):
    """A OpenStorage client that can be used to read or write odps tables"""
    def __init__(self, table_path, mode="input"):
        self._odps_table_path = table_path # type: str
        self._odps_project = "" # type: str
        self._odps_table_names = [""] # type: list[str]
        # self._odps_table_partitions : list[str] = [""] # ["part1=val1,part2=val2", "part3=val3"]
        self._odps_partitons = [] # type: list[list[models.table.TableSchema.TablePartition]]
        self._odps = None # type: ODPS
        # get partition column name from self._odps schema
        # self._odps_partition_column : str = ""

        self._init_from_ak()

    def _init_from_ak(self):
        self.openstorage_conf = OpenStorageConf()
        # self._odps_project, self._odps_table_names[0], self._odps_table_partitions[0] = odps_table_path_parse(self._odps_table_path)
        self._odps_project, self._odps_table_names[0], _ = odps_table_path_parse(self._odps_table_path)
        self._odps = ODPS(access_id=self.openstorage_conf.account_id, secret_access_key=self.openstorage_conf.account_key,
                project=self._odps_project, endpoint=self.openstorage_conf.odps_endpoint)
        if is_external_cluster():
            logger.info("Here we are in external cluster, skip odps meta check.") # IS_EXTERNAL_CLUSTER on
        else:
            if not self._odps.exist_project(self._odps_project):
                raise ValueError(f"odps_project_name {self._odps_project} not exist.")
            for table_name in self._odps_table_names:
                exist_table(self._odps, self._odps_project, table_name)

class OpenStorageRowReader(OpenstorageClient):
    """A reader that reads table data through tunnel."""
    def __init__(self, table_name :str, selected_cols="", excluded_cols="",
                        slice_id=0, slice_count=1, num_threads=1, capacity=2048):
        ''' Args:
            table_name: a string of the table to be opened.
            selected_cols: a string indicates the columns selected to be read,
                            with the delimiter ',', cannot be used together with
                            'excluded_cols'.
            excluded_cols: a string indicates the columns not to be read,
                            with the delimiter ',', cannot be used together with
                            'selected_cols'.
            slice_id: slice index when table was read under distributed cases,
                        should be used together with 'slice_count'.
            slice_count: total number slices (or workers) under distributed cases,
                        should be used together with 'slice_id'.
            num_threads: unused, keep compatible
            capacity: unused, keep compatible
            refresh_interval: (not yet implemented) session rebuild time
        '''
        logger.debug(f"real reader create, args table_name:{table_name}, slice_id{slice_id}, slice_count{slice_count}, num_threads{num_threads}, capacity{capacity} ")

        # table_name should be single-path. but multi-path is support here, for some case(such as from_common_io_odps_source), not really needed
        super(OpenStorageRowReader, self).__init__(table_name)
        if slice_id < 0 or slice_id >= slice_count:
            raise Exception(f"slice_id and slice_count are invalid: {slice_id}, {slice_count}")
        if num_threads < 0 or capacity <= 0:
            raise Exception(f"num_threads ({num_threads}) should be >=0 and capacity ({capacity}) should be > 0")
        
        ## dataset-table related variables ##
        self._odps_table_paths = [ p for p in table_name.split(",") if p.startswith("odps://")] # type: list[str] 
        if len(self._odps_table_paths) != 1:
            logger.error("extract table-paths:{} in count from table:{}, which may uncompatible!".format(len(self._odps_table_paths), table_name))
        self._is_compressed = False # type: bool
        self._batch_size = DefaultReadBatch # type: int # TODO(xyh):这里col是创建时传入, common是read时随时传入
        self._is_close = False # type: bool

        ## schema related variables ##
        # Begin getting original raw schema from odps.ODPS.table
        self._schema_all_column = None # type: numpy.recarray[(tuple["colname", str], tuple['typestr', str], tuple['pytype', type])]
        self._get_schema_all_column_from_odps()
        # Begin getting selected column list and format(in-numpy, to read for )
        self._select_column = [] # type: list
        self.__hash_features = [] # type: list # NOTE(XYH): hash_features 列, common_io场景应该不存在
        self.__dense_column = [] # type: list # NOTE(XYH): fixedlen_feature 列, common_io场景应该不存在
        self.__dense_defaults = [] # type: list # NOTE(XYH): fixedlen_feature 列, common_io场景应该不存在需要填默认值的数据源(from lizhendong)
        self._schema_select_column = None # type: numpy.recarray[(tuple["colname", str], tuple['pytype', type])] # 用于read格式转换
        self._schema = None # type: numpy.recarray[(tuple["colname", str], tuple['typestr', str], tuple['pytype', type])] # 用于get_schema
        self._apply_select_columns(selected_cols, excluded_cols) # must before self._reader create since _select_column needed

        ## reader related variables ##
        if os.environ.get("ODPS_DATASET_ROW_MODE") is None:
            os.environ["ODPS_DATASET_ROW_MODE"] = "1"
        self._reader_row_mode = os.environ["ODPS_DATASET_ROW_MODE"] == "1" # type: bool # whether to get read result in row-mode
        self._reader = Dataset.from_common_io_odps_source(self._odps_table_paths, self._is_compressed, self._batch_size,
                self._select_column, self.__hash_features, self.__dense_column, self.__dense_defaults) # type: Dataset
        self._reader_iter = iter(enumerate(self._reader)) # type: iter
        self._reader_cache = []
        self._start_pos = 0 # type: int
        self._offset_pos = 0 # type: int
        self._end_pos = 0 # type: int
        self._table_size = sum([self._reader.get_table_size(path) for path in self._odps_table_paths]) # type: int
        
        ## rearrange offset ##
        # rearrange start&end according to table_name/slice_id/slice_count args
        (start, end, specify_range) = _try_get_table_range(self._odps_table_paths[0])
        if not specify_range:
            dist = Distributor(self._table_size, slice_id, slice_count)
            self._start_pos = dist.start
            self._offset_pos = dist.start
            self._end_pos = dist.end
            self.seek(offset=self._start_pos)
            # logger.debug("rearrange offset in [{},{}) and reader according to slice:{}/{}".format(self._start_pos, self._end_pos, slice_id, slice_count))
        else:
            self._start_pos = start
            self._offset_pos = start
            self._end_pos = end
        
        self._batch_size = min(self._batch_size, self._end_pos - self._start_pos)
        logger.debug("OpenStorageRowReader init done, batch_size:{}, slice/tablesize:{}".format(self._batch_size, self._end_pos - self._start_pos))

        ''' E.g. for self._schema:
        ipdb> self._schema
        array([('uniq_id', 'string', <class 'object'>),
            ('content', 'string', <class 'object'>),
            ('related_ids', 'string', <class 'object'>),
            ('rnd1', 'bigint', <class 'int'>),
            ('rnd2', 'double', <class 'float'>)],
            dtype=[('colname', 'O'), ('typestr', 'O'), ('pytype', 'O')])

        ipdb> paths ==> [b'odps://project_name/tables/table_name/ds=ds_name?start=0&end=?']
        ipdb> is_compressed  ==> False
        ipdb> batch_size  ==> 1024
        ipdb> selected_columns  ==> ['uniq_id', 'content', 'related_ids', 'rnd1', 'rnd2']
        '''

    def _get_schema_all_column_from_odps(self):
        # get original raw schema from odps.ODPS.table
        tmp_schema = []
        table = self._odps.get_table(self._odps_table_names[0]) # type: models.Table
        columns = table.table_schema.columns # type: list[types.Column]
        self._odps_partitons = table.table_schema.partitions  # E.g. [<partition ds, type string>]
        for col in columns:
            col_name = col.name
            # if col_name not in self._select_column:
            #     logger.info(f"skip schema parsing since column {col_name} not in selected_cols")
            col_type_name = odps_type_to_pytype[type(col.type)].typestr
            col_py_type = odps_type_to_pytype[type(col.type)].pytype 
            tmp_schema.append((col_name, col_type_name, col_py_type))
        self._schema_all_column = numpy.array(
            [item for item in tmp_schema],
            dtype=[('colname', object), ('typestr', object), ('pytype', type)])

    def _apply_select_columns(self, selected_cols :str, excluded_cols: str):
        partition_names = [ part.name for part in self._odps_partitons]
        if selected_cols != "" and excluded_cols != "":
            raise Exception("selected_cols and excluded_cols cannot both be set")
        elif selected_cols == "" and excluded_cols == "":
            self._select_column : list  = self._schema_all_column['colname'].tolist()
        elif selected_cols != "":
            self._select_column : list  = [ col.strip() for col in selected_cols.split(",")]
            for col in self._select_column:
                if col in self._select_column:
                    logger.debug(f"selected_cols {col} is duplicate added. skip adding")
                elif col not in self._schema_all_column['colname'].tolist():
                    logger.warning(f"selected_cols {col} is invalid for project/table: {self._odps_project}/{self._odps_table_names[0]}")
                elif col in partition_names:
                    logger.warning(f"selected_cols {col} is partition column, skip adding")
                else:
                    self._select_column.append(col)
        elif excluded_cols != "":
            excluded_col_list = [col.strip() for col in excluded_cols.split(",")]
            excluded_col_list += partition_names
            self._select_column : list  = [col for col in self._schema_all_column['colname'].tolist() if col not in excluded_col_list]
        else: # actualy never happens
            logger.error(f"unexpected situation in columns adding!")
        # partition_sets = self._odps_table_partitions[0].split("/")
        # partition_names = [ part.split("=")[0] for part in partition_sets if "=" in part ]
        self._select_column = [col for col in self._select_column if col not in partition_names]
        self._schema_select_column = [ (col[0], col[2]) for col in self._schema_all_column if col[0] in self._select_column]
        rm_index_list = []
        for col_idx, col in enumerate(self._schema_all_column):
            if col[0] not in self._select_column:
                rm_index_list.append(col_idx)
        self._schema = numpy.delete(self._schema_all_column, rm_index_list, 0)
        logger.debug(f"OpenStorageRowReader get columns done, odps project/table: {self._odps_project}/{self._odps_table_names[0]}, selected_cols: {self._select_column}")


    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        self._reader_cache = None
        self._reader_iter = None
        self._reader = None
        self._is_close = True

    def _check_status(self):
        if self._is_close:
            raise Exception("Table is closed!")


    def get_row_count(self):
        """Return total row count could be read from this table."""
        self._check_status()
        return self._end_pos - self._start_pos
        # return self._reader.get_table_size(self.odps_paths[0])

    def get_schema(self):
        self._check_status()
        return self._schema
    
    def seek(self, offset):
        """Seek to the relative position of the rows to be read.

        Args:
            offset: an integer indicates the position to be seeked.
        """
        self._check_status()
        if offset < 0 or offset < self._start_pos or offset >= self._end_pos :
            raise ValueError("offset:{} out of valid range:[{},{})".format(offset, self._start_pos, self._end_pos))

        seek_paths = self._odps_table_paths[0].split("?")[0] # no ? is also fine
        seek_paths = [ "{}?start={}&end={}".format(seek_paths, offset, self._end_pos) ]
        # NOTE: 考虑到底层seek和重试已耦合, columnIO不透传出来c++层的seek 方法, 这里转用table_path重置reader的方式实现seek
        self._reader = Dataset.from_common_io_odps_source(seek_paths, self._is_compressed, self._batch_size,
                self._select_column, self.__hash_features, self.__dense_column, self.__dense_defaults)
        self._reader_iter = iter(enumerate(self._reader))
        self._reader_cache = []
        self._offset_pos = offset

    @property
    def start_pos(self):
        """Get table reader start position."""
        self._check_status()
        return self._start_pos

    @property
    def end_pos(self):
        """Get table reader end position."""
        self._check_status()
        return self._end_pos

    @property
    def offset_pos(self):
        """Get table reader offset position."""
        self._check_status()
        return self._offset_pos

    def close(self):
        """Close the table."""
        # self._reader.close()
        self._is_close = True

    def _transpose_col_to_row(self, batch_col_based, row_idx) -> tuple:
        # type: (dict[str, list], int) -> tuple
        ''' 按行转置(单行)
            batch_col_based: 列存数据
            row_idx: 行索引
            return: 行
        '''
        row = []
        for col_idx, col_name in enumerate(self._select_column):
            col_data = batch_col_based[col_name] # type: numpy.ndarray
            col_data = col_data[0] # 不知道为什么多套了一层list 可能是为了方便pack
            # list element in column_list
            if self._schema_select_column[col_idx][1] == list:
                # what if row_idx +1 >= len(col_data) ? needn't deal with this, just leave it out of range
                #     logger.info(f"column data out of range. set default val:{data} for col:{col_name}, idx:{row_idx},len:{len(col_data)}, content: {col_data}")
                data = col_data[0] # or type(col_data[0])() 
                indice_start = col_data[1][row_idx] # type: int
                indice_end = col_data[1][row_idx+1] # type: int
                data = data[indice_start: indice_end] # type: numpy.ndarray
                data = data.tolist() # type: numpy.ndarray
                if len(data) > 0 and \
                    (isinstance(data[0], bytes) or isinstance(data[0], numpy.bytes_) ): # if elements need decode
                    data = [ val.decode("utf-8") for val in data ]
            # primitive element in column_list
            else: 
                data = col_data[0] # type: numpy.ndarray
                data = data.item(row_idx)
                # try to decode bytes or timestamp...
                if isinstance(data, bytes) or isinstance(data, numpy.bytes_) :
                    data = data.decode("utf-8") # excetion will be catch outside this func
                elif isinstance(data, datetime.datetime): # TODO(xyh): type not verified yet. however no such type in real nebula tasks. just leave it
                    data = int(time.mktime(data.timetuple()))
                else:
                    pass # other types such as `object`, so keep raw data
            row.append(data)
        return tuple(row)

    def read(self, num_records=1, allow_smaller_final_batch=False,
                     to_ndarray=False) -> list[tuple]:
        stop_flag : bool = False
        def _read_next() -> tuple:
            try:
                iter_idx, batch_col_based  = next(self._reader_iter)
                return batch_col_based, False
            except StopIteration as e: # Can be raise from _read in column_io
                logger.debug(f"table reader reach StopIteration of {self._odps_table_paths[0]} ")
                return [None], True
        
        # def _read_iter():
        '''Read the table and return the rows as a recarray.

        Args:
            num_records: an integer indicates the rows to be read.
            allow_smaller_final_batch: return last partial batch when true,
                                                                 otherwise throw OutOfRangeError.
            to_ndarray: when setting as true, output will be set as a numpy ndarray.
        '''
        self._check_status()
        records : list[tuple] = self._reader_cache
        self._reader_cache = []
        while len(records) < num_records:
            try:
                if not self._reader_row_mode:
                    batch_row_based : list[tuple] = [] # list<tuple>
                    batch_col_based, stop_flag = _read_next()
                    if stop_flag:
                        ''' badcase: https://code.alibaba-inc.com/algo/common-io/commit/fffaa49e60c588418d1ff78ada395f719b244008
                        1. 表读完时, 如果多次while循环, 需要返还读取到的尾部 (通常是 num_records > records 的情况)
                        2. 表读完时, 如果执行了单次循环(即未读到任何records就触发stop_flag), 则理应抛出OutOfRange
                        e.g. num_records==10, table_size=15
                        step1 read(): _read_next 5, _read_next 5 ==> return 10; 
                        step2 read(): _read_next 5 ==> break;
                        step3 read(): _read_next 0 ==> raise;
                        '''
                        if len(records) > 0:
                            break
                        else:
                            raise OutOfRangeException("End of table reached! batch stop on pos:{}/{}".format(self.offset_pos, self.end_pos))
                    if len(batch_col_based) == 0 or len(batch_col_based[0]) == 0:
                        raise OutOfRangeException("End of table reached! batch empty on pos:{}/{}".format(self.offset_pos, self.end_pos))
                    batch_col_based = batch_col_based[0] # type: dict[str, numpy.ndarray] # wonder why list in list. maybe for pack a batch
                    # batch_col_based:  { 
                    #     "sample_id":      array([b'0d15303146_0', b'0d169237_0', ...... ]),
                    #     "urb_seq_length": array([19.], dtype=float64), 
                    #     "urb_seq_preitem":array([-666078, -5307881817759276931, ...... ]), 
                    #     ...... } 
                    head_col_data = next(iter(batch_col_based.values())) # type: numpy.ndarray
                    head_col_data = head_col_data[0]
                    if len(head_col_data) == 1: # [ [primitive] ]
                        batch_size = len(head_col_data[0])
                    elif len(head_col_data) == 2: # [ [primitive], [indice]]
                        batch_size = len(head_col_data[1]) - 1
                    else:
                        logger.warning("unexpected column length of col_name:{}, col_data len:{}".format(batch_col_based.keys()[0], len(head_col_data)))
                        col_len = [ len(col_data[0]) for col_data in batch_col_based.values() ]
                        batch_size = min(self._batch_size, max(col_len)) 
                    
                    self._offset_pos += batch_size
                    row_idx = -1
                    try:
                        for row_idx in range(batch_size):
                            row = self._transpose_col_to_row(batch_col_based, row_idx)
                            batch_row_based.append(row)
                    except Exception as e:
                        logger.error(f"common_io transpose_col_to_row fail in idx:{row_idx} because:{e}")
                else:
                    batch_row_based, stop_flag = _read_next()
                    if stop_flag:
                        if len(records) > 0:
                            break
                        else:
                            raise OutOfRangeException("End of table reached! batch stop on pos:{}/{}".format(self.offset_pos, self.end_pos))
                    if len(batch_row_based) == 0 or len(batch_row_based[0]) == 0:
                        raise OutOfRangeException("End of table reached! batch empty on pos:{}/{}".format(self.offset_pos, self.end_pos))
                    self._offset_pos += len(batch_row_based)
                    # TODO: the self._offset_pos is noy really user get pos (a small gap <= batch_pack_size)
                records.extend(batch_row_based)
            except urllib3.exceptions.ProtocolError as uep: # OpenstorageV1 use httpcall, OpenstorageV2 use sharemem
                logger.warning("read_batch catch ProtocolErrorException: {}".format(uep))
                self.seek(self._offset_pos)
            except urllib3.exceptions.ReadTimeoutError as uer: # OpenstorageV1 use httpcall, OpenstorageV2 use sharemem
                logger.warning("read_batch catch ReadTimeoutErrorException: {}".format(uer))
                self.seek(self._offset_pos)

        if len(records) > num_records:
            self._reader_cache = records[num_records: ]
            records = records[0: num_records]
            # logger.debug("num_records {} not multiple of reader_batch {}, maybe performance loss!".format(num_records, self._batch_size))
        if len(records) < num_records and not allow_smaller_final_batch:
            raise OutOfRangeException("End of table reached! batch smaller than num_records on pos:{}/{}".format(self.offset_pos, self.end_pos))
        if to_ndarray:
            record_arr = numpy.array(records, self._schema_select_column)
            return record_arr
        else:
            return records

