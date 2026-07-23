# -*- coding: utf8 -*-
import json
import os
import time
from odps import ODPS
from collections import defaultdict
from column_io.dataset.log_util import logger, varlogger, LOG_DIR
from column_io.dataset.secret_util import decode

ODPS_PARTITION_MODE_KEY_NAME = "ODPS_PARTITION_MODE"

class TablePartitionStruct:
    def __init__(self, project, table, logical_partition):
        # type: (str, str, str) -> None
        # open storage允许N个物理分区组合成一个逻辑分区
        # 因session粒度是逻辑分区, N个物理分区也只是1个session, hash参数也逻辑分区
        # logical_partition: 逻辑分区, 如果不使用多分区, 逻辑分区即为物理分区
        # physical_partitions: 逻辑分区对应的物理分区, 如果不使用多分区, physical_partitions为空list
        ENCODED_ODPS_ACCESS_ID = os.environ.get('ENCODED_ODPS_ACCESS_ID')
        ENCODED_ODPS_ACCESS_KEY = os.environ.get('ENCODED_ODPS_ACCESS_KEY')
        self.access_id = decode(ENCODED_ODPS_ACCESS_ID) if ENCODED_ODPS_ACCESS_ID else os.getenv('access_id')
        self.access_key = decode(ENCODED_ODPS_ACCESS_KEY) if ENCODED_ODPS_ACCESS_KEY else os.getenv('access_key')
        self.project = project
        self.table = table
        self.logical_partition = logical_partition
        self.physical_partitions = list()
        self.partition_prefix_list = list()  # 用于暂存前缀匹配的list
    
    def set_physical_partitions(self, physical_partitions):
        # type: (list) -> None
        self.physical_partitions = physical_partitions

    def get_physical_partitions(self):
        # type: () -> list
        return self.physical_partitions

    def get_logical_partition(self):
        # type: () -> list
        return self.logical_partition

    def set_partition_prefix_list(self, prefix_list):
        # type: (list) -> None
        self.partition_prefix_list = prefix_list

    def get_partition_prefix_list(self):
        # type: () -> list
        return self.partition_prefix_list


    def __str__(self):
        return json.dumps({
            "project": self.project,
            "table": self.table,
            "logical_partition": self.logical_partition,
            "physical_partitions": self.physical_partitions,
            "partition_prefix_list": self.partition_prefix_list
        }, indent=2)


def dump_to_file(ret_table_partition_struct_list):
    pass

def get_from_file():
    return False, []

def _normalize_partition_name_to_path(partition_name):
    # convert partition.name
    # 将 oyodps partition(class odps.models.partition.Partition) 属性: name
    # 由于格式:   ds='20250517',tag='train',nation='CH'
    # 转化为格式: ds=20250517/tag=train/nation=CH
    if not partition_name.strip():
        return ""
    pairs = [p.strip() for p in partition_name.split(',')]
    path_parts = []
    for pair in pairs:
        if '=' not in pair:
            continue
        k, v = pair.split('=', 1)
        k = k.strip()
        v = v.strip()
        # 去掉首尾单引号或双引号
        if v.startswith("'") and v.endswith("'"):
            v = v[1:-1]
        elif v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        path_parts.append("%s=%s" % (k, v))
    return "/".join(path_parts)


def _parse_prefix_path(prefix_path):
    # 将输入的前缀路径转为标准 k=v 列表
    # 输入: ds=20250517/tag=train/nation=CH
    # 输出: ['ds=20250517', 'tag=train', 'nation=CH']
    if not prefix_path or prefix_path == "":
        return []
    if prefix_path.endswith("/"):
        prefix_path = prefix_path[:-1]
    return [p.strip() for p in prefix_path.split('/') if p.strip()]


def prefixed_match_rule(table_partition_struct_list, table_obj):
    # 匹配满足前缀的分区，并返回路径格式的分区路径（非 partition.name 原始格式）
    # :param table_partition_struct_list: list of TablePartitionStuct
    # :return: None

    # 预处理所有前缀 -> 转为有序 k=v 列表
    for table_partition_struct in table_partition_struct_list:
        p = table_partition_struct.logical_partition
        if p == "":
            # 空前缀：返回所有分区的路径格式
            result = []
            try:
                for part in table_obj.partitions:
                    path = _normalize_partition_name_to_path(part.name)
                    result.append(path)
                return result
            except Exception as e:
                logger.warning("Failed to iterate partitions: %s" % str(e))
                return []
        table_partition_struct.set_partition_prefix_list(_parse_prefix_path(p))

    def is_match(part_list, prefix_list):
        # (list<str>, list<str>) -> bool
        match = True
        for i in range(len(prefix_list)):
            if part_list[i] != prefix_list[i]:
                match = False
                break
        return match

    try:
        for part in table_obj.partitions:
            # 将当前分区转为有序 k=v 列表
            # e.g.   odps 分区表示方式 ds='20250729',hh='14' ->  ['ds=20250729', 'hh=14']
            part_list = _parse_prefix_path(_normalize_partition_name_to_path(part.name))
            # 检查是否匹配任意一个前缀（前缀匹配）
            for table_partition_struct in table_partition_struct_list:
                prefix_list = table_partition_struct.get_partition_prefix_list()
                if len(part_list) < len(prefix_list):  # 分区层级低于输入, 必然不是子分区
                    continue
                match = is_match(part_list, prefix_list)
                if match:
                    table_partition_struct.get_physical_partitions().append("/".join(part_list))
    except Exception as e:
        logger.warning("Warning: failed to iterate partitions: %s" % str(e))


class PartitionMode:
    # 标准值
    ORIGIN = "origin"                   # 不做任何partition处理
    PREFIX_MERGE = "prefix_merge"       # 相同前缀的分区逻辑上组合为一个分区, 合建一个session
    PREFIX_NO_MERGE = "prefix_no_merge" # 相同前缀的分区各自建session, 用于帮用户免于写很多partition
    # 默认值（兜底）
    DEFAULT = ORIGIN

    # Map digits to modes
    _value_map = {
        0: ORIGIN,
        1: PREFIX_MERGE,
        2: PREFIX_NO_MERGE,
        "0": ORIGIN,
        "1": PREFIX_MERGE,
        "2": PREFIX_NO_MERGE,
    }
    _surpported_value_map = {
        0: ORIGIN,
        1: PREFIX_MERGE,
        "0": ORIGIN,
        "1": PREFIX_MERGE,
    }

    # Allowed string values
    _partition_handle_mode = {
        PREFIX_MERGE,
        PREFIX_NO_MERGE,
    }
    _surpported_strings = {
      ORIGIN,
      PREFIX_MERGE,
    }

    @classmethod
    def use_partition_handle(cls, partition_mode):
        partition_mode = cls.parse(partition_mode)
        return partition_mode in cls._partition_handle_mode

    @classmethod
    def parse(cls, value):
        if value is None:
            return cls._surpported_value_map[0]   # PartitionMode None means ORIGIN

        # Handle int or string digit
        if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
            try:
                num = int(value)
                if num in cls._surpported_value_map:
                    return cls._surpported_value_map[num]
                else:
                    raise ValueError()
            except Exception:
                raise ValueError(
                    "Invalid number: '{}'. Supported: 0=ORIGIN 1=PREFIX_MERGE".format(value))
        # Handle string
        if isinstance(value, str):
            if value.lower() in cls._surpported_strings:
                return value.lower()
            else:
                raise ValueError(
                    "Unsupported string: '{}'. Supported: {}".format(
                        value, ", ".join(sorted(cls._surpported_strings)) ) )
        # Unsupported type
        raise ValueError(
            "Unsupported type: {}. Value must be str or int.".format(
                type(value).__name__ ) )
    
    @classmethod
    def gen_full_partition_if_need(cls, proj_tbl_part, partition_mode, odps_endpoint):
        # 根据输入的 (project, table, partition_prefix) 三元组集合，
        # 查询 ODPS 获取每个非完整表达式所代表分区下的所有完整分区路径，
        # 将结果维护到list of TablePartitionStruct
        #
        # :param proj_tbl_part: set of (project_name, table_name, partition_prefix)
        # :return: table_partition_struct
        #     e.g
        #     [
        #         TablePartitionStruct<project, table, logical_partition, physical_partitions>,
        #         ...
        #         TablePartitionStruct<project, table, logical_partition, physical_partitions>
        #     ]
        # NOTE/TODO:
        #   1. 注意防御dataloader shuffle
        #   2. 考虑兼容是正在做的做分区合并需求， 和 未来可能得前缀建分区 但不多分区合并为一个session的需求

        cache_exists, ret_table_partition_struct_list = get_from_file()
        if cache_exists:
            return ret_table_partition_struct_list

        # 按 (project, table) 分组 partition_prefix
        grouped = defaultdict(list)
        for item in proj_tbl_part:
            try:
                project_name, table_name, partition_prefix = item
            except Exception as e:
                err_msg = "[ERROR] Unpack proj_tbl_part item got exception: {}, item: {}".format(e, item)
                logger.error(err_msg)
                raise ValueError(err_msg)
            table_partition_struct = TablePartitionStruct(project_name, table_name,
                                                          logical_partition=partition_prefix)
            grouped[(project_name, table_name)].append(table_partition_struct)

        # 缓存 table 对象: key -> (table_obj, is_partitioned)
        table_cache = dict()
        # 缓存odps project
        odps_client_cache = dict()
        for (project_name, table_name), grouped_table_partition_structs in grouped.items():
            cache_key = (project_name, table_name)

            # 获取 ODPS 客户端
            if project_name in odps_client_cache:
                odps_client = odps_client_cache[project_name]
            else:
                odps_client = ODPS(access_id=grouped_table_partition_structs[0].access_id,
                                secret_access_key=grouped_table_partition_structs[0].access_key,
                                project=project_name,
                                endpoint=odps_endpoint)
                odps_client_cache[project_name] = odps_client

            # 获取 table 对象（缓存）
            if cache_key not in table_cache:
                try:
                    table_obj = odps_client.get_table(table_name)
                    # grouped[(project_name, table_name)] 仅有1个元素且为空字符串则为无分区表
                    is_partitioned = not (len(grouped[(project_name, table_name)]) == 1 and \
                                    grouped[(project_name, table_name)][0].get_logical_partition() == "")
                    table_cache[cache_key] = (table_obj, is_partitioned)
                except Exception as e:
                    logger.warning("Failed to get table {}: {}".format(table_name, str(e)))
                    table_cache[cache_key] = None  # 标记失败
                    continue
            else:
                cached = table_cache[cache_key]
                if cached is None:
                    continue
                table_obj, is_partitioned = cached

            if is_partitioned:
                # 一次调用将同一个表的分区全查出来, 减少odps api调用开销
                if partition_mode == cls.PREFIX_MERGE:
                    prefixed_match_rule(grouped_table_partition_structs, table_obj)
            ret_table_partition_struct_list.extend(grouped_table_partition_structs)
        dump_to_file(ret_table_partition_struct_list)
        return ret_table_partition_struct_list

    @classmethod
    def gen_partition_struct(cls, proj_tbl_part_set, partition_mode, ODPS_ENDPOINT):
        # set<tuple<str, str, str>>, str, str
        partition_mode = cls.parse(partition_mode)
        res = cls.gen_full_partition_if_need(proj_tbl_part_set, partition_mode, ODPS_ENDPOINT)
        cls.check_result(res)
        return res

    @classmethod
    def check_result(cls, table_partition_struct_list):
        if not table_partition_struct_list:
            return
        empty_partition_structs = []
        for struct in table_partition_struct_list:
            if not struct.physical_partitions:  # len(physical_partitions) == 0
                empty_partition_structs.append(struct)
        if empty_partition_structs:
            header_msg = "Found {} partitions Not Exist or no partitions START with them !".format(len(empty_partition_structs))
            logger.error(header_msg)
            body_msg = ""
            for struct in empty_partition_structs:
                body_msg += "    - odps://{}/tables/{}/{}\n".format(struct.project, struct.table, struct.logical_partition)
            logger.error(body_msg)
            raise ValueError(header_msg + "\n" + body_msg)

def get_odps_partition_modified_timestamp(is_odps_endpoint_reachable, access_id, access_key, project_name, table_name, partition_name, odps_endpoint):
        if is_odps_endpoint_reachable:
            try:
                odps = ODPS(access_id=access_id,
                        secret_access_key=access_key,
                        project=project_name,
                        endpoint=odps_endpoint)
                table = odps.get_table(table_name)
                partition = table.get_partition(partition_name.replace("/", ","))
                if hasattr(partition, "last_data_modified_time") and partition.last_data_modified_time:
                    last_modified_time = partition.last_data_modified_time
                elif hasattr(partition, "last_modified_time") and partition.last_modified_time:
                    last_modified_time = partition.last_modified_time
                timestamp = str(int(time.mktime(last_modified_time.timetuple())))
            except Exception as e:
                logger.warning("error when get partition modified timestamp, the message is: {}".format(e))
                # 如果 last_data_modified_time 取值异常，就用当前时间
                timestamp = str(int(time.time()))
        else:
          timestamp = str(int(time.time()))
          msg = "odps service: [{}] is not reachable, and going to use now time as partition last modified timestamp!".format(odps_endpoint)
          logger.warning(msg)
          varlogger.warning(msg)
        return timestamp


def test_prefixed_match_rule():
    t = ("LTA*****N8m", "QJD*****1Yw", "palgo_fpage", "generanking_eg_sample_mergeclk2k_opt_hh_v3")
    ak, sk, project_name, table_name = t

    odps_endpoint = os.getenv("ODPS_ENDPOINT", "")
    odps_client = ODPS(access_id=ak,
                       secret_access_key=sk,
                       project=project_name,
                       endpoint=odps_endpoint)
    table_obj = odps_client.get_table(table_name)

    os.environ["access_id"] = ak
    os.environ["access_key"] = sk
    table_partition_struct_list = list()
    p1 = TablePartitionStruct(project_name, table_name, "ds=20250722")
    table_partition_struct_list.append(p1)
    print(p1)
    prefixed_match_rule(table_partition_struct_list, table_obj)
    for p in table_partition_struct_list:
      print(p)

def test_gen_partition_struct():
    # None-partition table
    #t = ("LTA*****FBg", "rZJf*****si", "autonavi_base_sp_dev", "dwd_gd_use_action_wide_poi_filter_session_beijing_6month_train_filtered_26plus", "")
    '''
{
  "project": "autonavi_base_sp_dev",
  "table": "dwd_gd_use_action_wide_poi_filter_session_beijing_6month_train_filtered_26plus",
  "logical_partition": "",
  "physical_partitions": []
}
    '''

    # Partition table with multipartition-as-one
    t = ("LTA*****N8m", "QJD*****1Yw", "palgo_fpage", "generanking_eg_sample_mergeclk2k_opt_hh_v3", "ds=20250722")
    '''
{
  "project": "palgo_fpage",
  "table": "generanking_eg_sample_mergeclk2k_opt_hh_v3",
  "logical_partition": "ds=20250722",
  "physical_partitions": [
    "ds=20250722/hh=00",
    "ds=20250722/hh=01",
    "ds=20250722/hh=02",
    "ds=20250722/hh=03",
    "ds=20250722/hh=04",
    "ds=20250722/hh=05",
    "ds=20250722/hh=06",
    "ds=20250722/hh=07",
    "ds=20250722/hh=08",
    "ds=20250722/hh=09",
    "ds=20250722/hh=10",
    "ds=20250722/hh=11",
    "ds=20250722/hh=12",
    "ds=20250722/hh=13",
    "ds=20250722/hh=14",
    "ds=20250722/hh=15",
    "ds=20250722/hh=16",
    "ds=20250722/hh=17",
    "ds=20250722/hh=18",
    "ds=20250722/hh=19",
    "ds=20250722/hh=20",
    "ds=20250722/hh=21",
    "ds=20250722/hh=22",
    "ds=20250722/hh=23"
  ]
}
    '''

    # Partition table with logical_partition == physical_partitions
    t = ("LTA*****N8m", "QJD*****1Yw", "palgo_fpage", "generanking_eg_sample_mergeclk2k_opt_hh_v3", "ds=20250722/hh=23")
    '''
{
  "project": "palgo_fpage",
  "table": "generanking_eg_sample_mergeclk2k_opt_hh_v3",
  "logical_partition": "ds=20250722/hh=23",
  "physical_partitions": [
    "ds=20250722/hh=23"
  ]
}
    '''
    # Partition table with multipartition-as-one
    t = ("LTA*****9Yc", "k39*****hyL", "palgo_fpage", "generanking_eg_sample_mergeclk2k_growvideo_hh_v3__omega", "ds=20231012")
    '''
[2025-10-23 16:26:13,683] [ERROR] [partition_util.py:311] Found 3 partitions Not Exist or no partitions START with them !
[2025-10-23 16:26:13,683] [ERROR] [partition_util.py:315]     - odps://palgo_fpage/tables/generanking_eg_sample_mergeclk2k_growvideo_hh_v3__omega/ds=20231010

[2025-10-23 16:26:13,683] [ERROR] [partition_util.py:315]     - odps://palgo_fpage/tables/generanking_eg_sample_mergeclk2k_growvideo_hh_v3__omega/ds=20231010
    - odps://palgo_fpage/tables/generanking_eg_sample_mergeclk2k_growvideo_hh_v3__omega/ds=20231011

[2025-10-23 16:26:13,683] [ERROR] [partition_util.py:315]     - odps://palgo_fpage/tables/generanking_eg_sample_mergeclk2k_growvideo_hh_v3__omega/ds=20231010
    - odps://palgo_fpage/tables/generanking_eg_sample_mergeclk2k_growvideo_hh_v3__omega/ds=20231011
    - odps://palgo_fpage/tables/generanking_eg_sample_mergeclk2k_growvideo_hh_v3__omega/ds=20251013

ValueError: Found 3 partitions Not Exist or no partitions START with them !
    - odps://palgo_fpage/tables/generanking_eg_sample_mergeclk2k_growvideo_hh_v3__omega/ds=20231010
    - odps://palgo_fpage/tables/generanking_eg_sample_mergeclk2k_growvideo_hh_v3__omega/ds=20231011
    - odps://palgo_fpage/tables/generanking_eg_sample_mergeclk2k_growvideo_hh_v3__omega/ds=20251013
    '''

    ak, sk, prj_name, tbl_name, partition_prefix = t
    os.environ["access_id"] = ak
    os.environ["access_key"] = sk
    odps_endpoint = os.getenv("ODPS_ENDPOINT", "")

    proj_tbl_part = set()
    proj_tbl_part.add((prj_name, tbl_name, partition_prefix))
    proj_tbl_part.add((prj_name, tbl_name, "ds=20231010"))
    proj_tbl_part.add((prj_name, tbl_name, "ds=20231011"))
    res = PartitionMode.gen_partition_struct(proj_tbl_part, PartitionMode.PREFIX_MERGE, odps_endpoint)
    for r in res:
        print("Finally output: {}".format(r))


def test_partition_mode():
    # Valid cases
    print("====  Valid cases ====")
    cases = [
        (None, PartitionMode.ORIGIN),
        ("origin", PartitionMode.ORIGIN),
        ("prefix_merge", PartitionMode.PREFIX_MERGE),
        ("ORIGIN", PartitionMode.ORIGIN),
        ("PREFIX_MERGE", PartitionMode.PREFIX_MERGE),
        (0, PartitionMode.ORIGIN),
        ("0", PartitionMode.ORIGIN),
        (1, PartitionMode.PREFIX_MERGE),
        ("1", PartitionMode.PREFIX_MERGE),
    ]
    passed = 0
    for inp, expected in cases:
        try:
            result = PartitionMode.parse(inp)
            if result == expected:
                passed += 1
            else:
                print("Failed: input={}, got={}, expected={}".format(inp, result, expected))
        except Exception as e:
            print("Error: input={}, exception={}".format(inp, str(e)))

    # Invalid cases
    print("====  Invalid cases ====")
    invalid_cases = [
        "xyz",
        "4",
        "",
        [],
        "prefix_merge ",
        "prefix_no_merge",
        " PREFIX_MERGE",
        "regex_merge\n",
        4,
        -1,
    ]

    for inp in invalid_cases:
        try:
            PartitionMode.parse(inp)
            print("Error: should fail for input: {0}".format(inp))
        except ValueError as ve:
            passed += 1
            print(ve)
        except Exception as e:
            print("Unexpected error for input {0}: {1}".format(inp, str(e)))
    print("Passed {0}/{1}".format(passed, len(cases) + len(invalid_cases)))


def run_test():
    #test_prefixed_match_rule()
    #test_gen_partition_struct()
    test_partition_mode()

if __name__ == '__main__':
    run_test()

