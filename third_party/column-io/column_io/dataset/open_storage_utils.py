# -*- coding: utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import base64
import copy
import datetime
import fcntl
import json
import os
import errno
import psutil
import random
import requests
import socket
import six
import subprocess
import sys
import time
import uuid
import threading
import traceback

from binascii import b2a_hex, a2b_hex
from collections import OrderedDict
from odps import ODPS
from requests import RequestException
from column_io.lib import interface
from column_io._version import __odps_sdk_version__
from column_io.dataset.log_util import logger, varlogger, LOG_DIR
from column_io.dataset.metric_util import metric_factory,MetricPoints,MetricStatus
from column_io.dataset.metric_util import NebulaIOFatalError
from column_io.dataset.job_info import JobInfo,get_app_id,get_job_info,get_child_order,get_task_name,get_work_id,is_notebook, get_odps_endpoint
from column_io.dataset.secret_util import decode
from column_io.dataset.partition_util import get_odps_partition_modified_timestamp

COLUMN_IO = "column_io"
COLUMN_IO_CPU = "column_io_cpu"
PAIIO = "paiio"
PYTHONPATH = "PYTHONPATH"
ODPS_ACCESS_INFO_DUMP = "odps_access_info_dump"
SESSION_EXPIRATION_THRESHOLD = int(0.5 * 3600 * 1000)  # in ms, should be shorter than the session expiration time on the service side.
INDICATOR_PREFIX = "_indicator_"
AUTH_MODE_READ = "read"
AUTH_MODE_WRITE = "write"
OPEN_STORAGE = "open_storage"
NEBULA_OPEN_STORAGE_CACHE_SERVER = os.getenv("NEBULA_OPEN_STORAGE_CACHE_SERVER", "http://odps.nebula.alibaba-inc.com/authorize/odps_open_storage/cache")
ODPS_ENDPOINT = get_odps_endpoint()
CHECK_STATUS = "check_status"
SESSION_ID = "session_id"
EXPIRATION_TIME = "expiration_time"
RECORD_COUNT = "record_count"
SESSION_DEF_DICT = "session_def_dict"
ORDERED_REQUIRED_DATA_COLUMNS = "ordered_required_data_columns"
CHECK_AUTH_PREFIX = "check_auth"
LOCAL_CACHE_PREFIX = "local_cache"
NEBULA_FORCE_CREATE_SESSION = "NEBULA_FORCE_CREATE_SESSION"

_is_session_creator_override = None  # type: Optional[bool]


# def is_external_cluster():
#     is_external = os.environ.get('IS_EXTERNAL_CLUSTER', None)
#     return str(is_external).lower() == 'true'

def is_service_reachable(service_url, timeout_seconds=3, max_retries=2, retry_delay=2):
    # try:
    #   # Python3
    #   from urllib.parse import urlparse, urlunparse
    # except ImportError:
    #   # Python2
    #   from urlparse import urlparse, urlunparse
    from six.moves.urllib.parse import urlparse, urlunparse
    parsed = urlparse(service_url)
    # (protocal, host+port, empty parth, empty args, empty query, emptry segment)
    base_tuple = (parsed.scheme, parsed.netloc, '', '', '', '')
    service_base_url = urlunparse(base_tuple)
    # let this request more like from a brower or curl
    headers = {
        #"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "User-Agent": "curl/7.64.1",  # curl UA
        "Accept": "*/*",
    }
    # 重试逻辑
    last_failure = None  # 记录最后一次失败信息，循环结束统一打日志
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(service_base_url,
                                    headers=headers,
                                    allow_redirects=False,
                                    verify=True,
                                    timeout=timeout_seconds)
            return True
        except requests.RequestException as e:
            # RequestException 已覆盖 ConnectionError / Timeout / HTTPError / SSLError 等所有 requests 异常
            last_failure = "exception={}".format(e)
        except Exception as e:
            # 兜底：极少数底层异常（如 socket.timeout、OSError 等）不被 requests 包装
            last_failure = "unexpected_exception={}".format(e)

        # 还有剩余重试机会才 sleep；不打中间过程的日志
        if attempt < max_retries:
            time.sleep(retry_delay)

    # 全部尝试完仍失败，统一打一条日志
    logger.error("Service reachability check failed after {} attempts for {}, last_failure: {}".format(
        max_retries + 1, service_base_url, last_failure))
    return False
    # if is_external_cluster():
    #     return False
    # return True

def get_tunnel_endpoint():
    for key in ["tunnel_end_point", "TUNNEL_ENDPOINT"]:
        value = os.getenv(key)
        if value:
            return value
    return "ipc:///home/admin/halo/socket/halo.sock"


# 提前查好相关service可达性cache住结果
# 防御shuffle场景冲击service和日志刷屏
IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE = is_service_reachable(NEBULA_OPEN_STORAGE_CACHE_SERVER)
IS_ODPS_ENDPOINT_REACHABLE = is_service_reachable(ODPS_ENDPOINT)
logger.warning("Service reachable: {}, NEBULA_OPEN_STORAGE_CACHE_SERVER: {}"\
       .format(IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE, NEBULA_OPEN_STORAGE_CACHE_SERVER))
logger.warning("Service reachable: {}, ODPS_ENDPOINT: {}".format(IS_ODPS_ENDPOINT_REACHABLE, ODPS_ENDPOINT))

def get_pkg_name():
    """
    return: paiio / column_io
    """
    return str(__name__).split(".")[0]

def is_column_io():
    pkg = get_pkg_name()
    return pkg == COLUMN_IO or pkg == COLUMN_IO_CPU

def is_paiio():
    return get_pkg_name() == PAIIO

def set_session_creator_override(value):
    # type: (Optional[bool]) -> None
    if value is not None and not isinstance(value, bool):
        raise TypeError("is_session_creator_override must be bool or None, got: {}".format(type(value).__name__))
    global _is_session_creator_override
    _is_session_creator_override = value

def is_session_creator():
    if _is_session_creator_override is not None:
        return _is_session_creator_override
    if not IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE:
        return True
    if is_notebook():
        return True
    task_name = get_task_name()
    worker_id = get_work_id()
    if is_column_io():
        return task_name == "scheduler"
    elif is_paiio():
        if task_name == "worker" and worker_id == 0:
            return True
        else:
            return False
    else:
        raise NebulaIOFatalError("unsupported io module: {}, task_name: {}, worker_id: {}".format(get_pkg_name(), task_name, worker_id))
        # raise ValueError("unsupported io module: {}".format(get_pkg_name()))

def makedirs_if_not_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        # EEXIST: 路径已存在（可能是并发其它进程刚创建的）
        if e.errno == errno.EEXIST and os.path.isdir(path):
            return
        raise NebulaIOFatalError("makedirs failed for path={}, errno={}, original={}"
                                    .format(path, getattr(e, "errno", None), e))

def force_recreate_session():
  default_v = "1" if is_column_io() else "0"
  env_v = os.environ.get(NEBULA_FORCE_CREATE_SESSION, default_v)
  return str(env_v) == "1"

def is_restarted_version_inplace():
  # Check the RESTART_VERSION environment variable
  #   an empty string indicates no global restart has occurred.
  #
  #  - name: RESTART_VERSION
  #    valueFrom:
  #      fieldRef:
  #        apiVersion: v1
  #        fieldPath: 'metadata.labels[''kubedl.io/expected-restart-version'']'
  #
  restart_version = os.getenv("RESTART_VERSION", "0")
  try:
    restart_version = int(restart_version)
  except Exception as e:
    restart_version = 0
  return restart_version == 0

def get_local_ip(): # TODO: make sure ipv6 support
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # UDP won`t really connect to this endpoint
        ip = s.getsockname()[0]
        s.close()
    except socket.error:
        ip = "127.0.0.1"
    return ip

def local_cache_file_name(prefix, project, table, partition):
    # type: (str, str, str, str) -> str
    # NOTE: multi level partition, such as: ds=20250218/type=train
    full_name = "{}.{}.{}.{}.txt".format(prefix, project, table, partition.replace('/', '_'))
    if len(full_name) < 255:
        return full_name
    short_full_name = "{}.{}.{}.{}.{}.txt".format(prefix, project[:32], table[:32], partition.replace('/', '_')[:32], hash(full_name))
    return short_full_name

def check_file(target_file):
    try:
        modi_time = os.path.getmtime(target_file)
    except OSError as e:
        # 文件不存在/不可访问：直接认为缓存无效
        return False
    # --------  NOTEBOOK, WITHOUT REFRESH DAEMON THREAD, SO CHECK EXP MORE EXTREMLY ---------- #
    if is_notebook() and time.time() - modi_time >= 12*3600: # why 12h?  because when cache file create, the remain time is >=12h
        modi_time = int(modi_time)
        try:
            os.remove(target_file)
        except Exception as e:
            logger.warning("failed to remove local_cache:{}, exception: {}".format(target_file, e))
        finally:
            varlogger.info("local_cache:{} is at exp time:{}, del and new one".format(target_file, modi_time))
            return False
    return os.path.exists(target_file)

def check_or_mkdir(relative_path):
    if os.path.exists(LOG_DIR):
        target_dir = os.path.join(LOG_DIR, relative_path)
    else:
        work_dir = os.getenv(PYTHONPATH, "./")
        target_dir = os.path.join(work_dir, relative_path)
    makedirs_if_not_exists(target_dir)
    return target_dir

def dump_debug_access_info_batch(session_struct_set):
    # type: (list[HashableSessionStruct]) -> None
    for i, session_struct in enumerate(session_struct_set):
        try:
            ordered_dict = OrderedDict([
                ("access_id", session_struct.access_id),
                ("access_key", session_struct.access_key),
                ("project", session_struct.project),
                ("table", session_struct.table),
                ("tunnel_endpoint", session_struct._tunnel_endpoint),
                ("default_project", ""),
                ("hash_table", "")
            ])
            TARGET_FILE = "test.conf." + session_struct.project + "." + session_struct.table
            target_dir = check_or_mkdir(ODPS_ACCESS_INFO_DUMP)
            target_file = os.path.join(target_dir, TARGET_FILE)
            if os.path.exists(target_file):
                return
            with open(target_file, "w") as f:
                json.dump(ordered_dict, f, indent=4)
        except Exception as e:
            logger.info("dump access info exception: {}".format(e))
            varlogger.info("dump access info exception format_exec: {}".format(traceback.format_exc()))

def check_auth_and_data(session_struct_set):
    # type: (list[HashableSessionStruct]) -> None
    target_dir = check_or_mkdir(ODPS_ACCESS_INFO_DUMP)
    # 打散各个worker退出此函数的时间点，避免各个worker同时请求open storage session cache，概率性降低请求总量
    if not is_session_creator():
        sleep_time = round(random.uniform(0, 5 * len(session_struct_set)), 2)
        time.sleep(sleep_time)
    for idx, session_strcut in enumerate(session_struct_set):
        try:
            target_file = local_cache_file_name(LOCAL_CACHE_PREFIX, session_strcut.project, session_strcut.table, session_strcut.partition)
            target_file = os.path.join(target_dir, target_file)
            if check_file(target_file):
                try:
                    with open(target_file, "r") as f:
                        try:
                            fcntl.flock(f, fcntl.LOCK_EX)
                            file_content = json.load(f)
                            check_status = file_content.get(CHECK_STATUS, False)
                            if check_status:
                                continue
                        finally:
                            fcntl.flock(f, fcntl.LOCK_UN)
                except (IOError, OSError) as e:  # Py2/3 兼容：open 失败会落到这里
                    # 只兜住“文件不存在”(ENOENT / 2)；其它打开失败原因继续抛给外层
                    if getattr(e, "errno", None) != errno.ENOENT:
                        raise
        except Exception as e:
            traceback.print_exc()
            logger.warning("check partition ({}, {}, {}) failed.".format(session_strcut.project, session_strcut.table, session_strcut.partition))
            if not (IS_ODPS_ENDPOINT_REACHABLE and IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE):
                # 弹内: meta网络已通，可继续检查所读分区是否存在
                # (暂时)特殊弹内: 如韩国, meta集群在韩国搭建, 故meta通, 但 cache server因vip网关ssl问题暂时未通
                # 弹外: 多云网络 只有tunnel通, meta及cache server均不通
                # 汇总: 只有tun-ep / cache_server(代表meta) 均通时才进行存在性检查
                return
            # else分支逻辑: 查看所读分区是否存在, 提前抛出不存在错误 以免跑很久才报错消耗卡时
            partition_name = session_strcut.partition.replace("/", ",")
            o = ODPS(access_id=session_strcut.access_id,
                    secret_access_key=session_strcut.access_key,
                    project=session_strcut.project,
                    endpoint=session_strcut._odps_endpoint)
            table = o.get_table(session_strcut.table)

            is_partitioned_table = False
            if hasattr(table, "table_schema"):
                schema = table.table_schema
            elif hasattr(table, "schema"):
                schema = table.schema
            else:
                return
            if hasattr(schema, "partitions") and isinstance(schema.partitions, list) and len(schema.partitions) > 0:
                is_partitioned_table = True
            else:
                is_partitioned_table = False
            if is_partitioned_table:
                partition = table.get_partition(partition_name)
                partition.size  # It doesn`t matter what to query; it does matter to query to determine the presense of table.



# HashableSessionStruct: odps session最小化定义的信息元组(from p+t+p+cols)
class HashableSessionStruct:
    def __init__(self, project, table,
                 logical_partition,
                 physical_partitions = [],
                 select_columns = []):
        # type: (str, str, str, list, list) -> None
        # open storage允许N个物理分区组合成一个逻辑分区
        # 因session粒度是逻辑分区, N个物理分区也只是1个session, hash参数也逻辑分区
        # logical_partition: 逻辑分区, 如果不使用多分区, 逻辑分区即为物理分区
        # physical_partitions: 逻辑分区对应的物理分区, 如果不使用多分区, physical_partitions为空list
        ENCODED_ODPS_ACCESS_ID = os.environ.get('ENCODED_ODPS_ACCESS_ID')
        ENCODED_ODPS_ACCESS_KEY = os.environ.get('ENCODED_ODPS_ACCESS_KEY')
        self.access_id = decode(ENCODED_ODPS_ACCESS_ID) if ENCODED_ODPS_ACCESS_ID else os.getenv('access_id')
        self.access_key = decode(ENCODED_ODPS_ACCESS_KEY) if ENCODED_ODPS_ACCESS_KEY else os.getenv('access_key')
        self.access_id = str(self.access_id)
        self.access_key = str(self.access_key)
        self.project = str(project)
        self.table = str(table)
        self.partition = str(logical_partition)
        self.physical_partitions = [str(phy_part) for phy_part in physical_partitions]
        self.select_columns = copy.deepcopy(select_columns)
        self.halo_worker_docker_image = os.environ.get("HALO_WORKER_DOCKER_IMAGE", "")
        self.halo_worker_docker_image_tag = self.halo_worker_docker_image.rsplit(":", 1)[1] if ":" in self.halo_worker_docker_image else ""
        suffix = "_accelerated"
        if self.halo_worker_docker_image_tag.endswith(suffix):
            self.halo_worker_docker_image_tag = self.halo_worker_docker_image_tag[:-len(suffix)]
        # 以下成员为衍生值, 逻辑上唯一确定(但不实际保证)
        self._hash_id = "" # type: str
        self._ordered_select_columns = None # type: list[str]
        self._partition_modified_timestamp = None
        self._odps_endpoint = get_odps_endpoint()
        self._tunnel_endpoint = get_tunnel_endpoint()
        
        # 以下成员为业务关联字段, 逻辑上不唯一确定
        # ...

    def set_part_size(self, part_size, part_id, akey):
        self.part_id = part_id
        self.part_size = part_size

    def _get_ordered_columns(self, select_columns):
        if len(select_columns) == 0:
            return select_columns
        ordered_columns = list()
        # combo 
        if isinstance(select_columns[0], tuple):
            select_columns = [item[0] for item in select_columns] 
        logger.info("required_data_columns is {}".format(select_columns))
        target_dir = check_or_mkdir(ODPS_ACCESS_INFO_DUMP)
        target_file = local_cache_file_name(LOCAL_CACHE_PREFIX, self.project, self.table, self.partition)
        target_file = os.path.join(target_dir, target_file)
        if os.path.exists(target_file):
            with open(target_file, "r") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX) # Maybe no need to acquire the file lock
                    dump_dict = json.load(f)
                    ordered_columns = dump_dict[ORDERED_REQUIRED_DATA_COLUMNS]
                    return ordered_columns
                except Exception as e:
                    pass
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN) # Release the file lock
        if IS_ODPS_ENDPOINT_REACHABLE:
            # TODO: ODPS init and get_table may occur exception. catch it
            o = ODPS(access_id=self.access_id,
                    secret_access_key=self.access_key,
                    project=self.project,
                    endpoint=self._odps_endpoint)
            t = o.get_table(self.table)
            columns = t.schema.columns
            deepcopied_select_columns = copy.deepcopy(select_columns)
            for c in columns:
                if c.name.startswith(INDICATOR_PREFIX):
                    deepcopied_select_columns.append(c.name)
            for c in columns:
                for cc in deepcopied_select_columns:
                    if c.name.startswith(cc):
                        ordered_columns.append(c.name)
                        break
        else:
            # reutrn empty list and log warnning, reminding if all columns for create session
            # create session for all columns in lest
            # ODPS-0420061: Invalid parameter in HTTP request - The requested columns should be in order with table schema
            msg = "odps service: [{}] is not reachable, and going to creat session for all columns !".format(self._odps_endpoint)
            logger.info(msg)
            varlogger.warning(msg)
        return ordered_columns

    def get_ordered_select_columns(self):
        if not self._ordered_select_columns:
            self._ordered_select_columns = self._get_ordered_columns(self.select_columns)
        return self._ordered_select_columns

    def get_partition_modified_timestamp(self):
        if not self._partition_modified_timestamp:
            self._partition_modified_timestamp = get_odps_partition_modified_timestamp(IS_ODPS_ENDPOINT_REACHABLE, self.access_id, self.access_key, 
                                                                                       self.project, self.table, self.partition, self._odps_endpoint)
        return self._partition_modified_timestamp

    def get_hash_id(self):
        #  type: () -> str
        if self._hash_id != "":
            return self._hash_id
        # NOTE: when(proj,table,parts, cols) same and auth(access_id&key) insame, cannot share session_id
        def gen_hash_key(*strings):
            import hashlib
            hash_object = hashlib.sha256()
            for string in strings:
                hash_object.update(string.encode('utf-8'))
            hashkey = hash_object.hexdigest()
            return hashkey  # length: 64 Bytes
        ordered_columns = self.get_ordered_select_columns()
        hash_id_args = [
          self.access_id,
          self.access_key,
          self.project,
          self.table,
          self.partition,
          self._tunnel_endpoint,
          __odps_sdk_version__,
          self.halo_worker_docker_image_tag,
        ]
        hash_id_args.extend(ordered_columns)
        self._hash_id = gen_hash_key(*hash_id_args)
        logger.info("gen_hash_id:{}, {}, {}".format(self._hash_id, self.to_str_basic(), self.to_str_extended()))
        varlogger.info("gen_hash_id:{}, {}".format(self._hash_id, self.to_str_all()))
        return self._hash_id
    def to_str_extended(self):
        return "tunnel endpoint:{}, odps sdk version:{}, halo worker docker image tag: {}".format(
            self._tunnel_endpoint, __odps_sdk_version__, self.halo_worker_docker_image_tag
        )    
    def to_str_basic(self):
        return "project:{}, table:{}, partition:{}".format(self.project, self.table, self.partition)
    def to_str_full(self):
        return "access_id:{}, access_key:{}, {}, {}".format(self.access_id, self.access_key, self.to_str_basic(), self.to_str_extended())
    def to_str_all(self):
        return "{}, columns:\n{}".format(self.to_str_full(), json.dumps(self.get_ordered_select_columns(), indent=4))
    #TODO: @staticmethod def dumps_to_local_file()/ loads_from_local_file()

__report_init_done = False # type: bool # TODO: make-sure multiprocess spawn is safe here in COLUMN-IO
def try_report_metric_init_done(odps_table_len):
    global __report_init_done
    if __report_init_done:
        return
    # try_report_metric_init_done. The local `metric_status` file is used to handle multi-process scenarios
    # when multiple processes are created concurrently, it's impossible to avoid duplicate reporting actions
    # so file synchronization is used instead.
    target_dir = check_or_mkdir(ODPS_ACCESS_INFO_DUMP)
    
    cache_files = [f for f in os.listdir(target_dir) if f.startswith(LOCAL_CACHE_PREFIX)]
    if odps_table_len < len(cache_files):
        varlogger.info("cache count:{} > odps_table len:{}, please check config".format(len(cache_files), odps_table_len))
    if odps_table_len > len(cache_files):
        return # not init ready
    # ↓↓ heavy operation, should not frequently called
    def _probe_reported_from_file(target_file):
        if not os.path.exists(target_file):
            with open(target_file, 'w') as f:
                json.dump({}, f)
        with open(target_file, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                try:
                    metric_status = json.load(f)
                except Exception:
                    metric_status = {} # 文件为空或损坏，重置为默认值
                if metric_status.get("openstorage_session_init") is True:
                    return True
                metric_status["openstorage_session_init"] = True
                f.seek(0)
                f.truncate()
                json.dump(metric_status, f)
                f.flush()
                return False
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return True
    reported = True
    try:
        TARGET_FILE = "metric_status.json"
        target_file = os.path.join(target_dir, TARGET_FILE)
        reported = _probe_reported_from_file(target_file=target_file)
    except Exception as e:
        varlogger.info("fail to load metric status file: %s", str(e))
    if reported:
        return
    metric_client = metric_factory.get("openstorage_session_init")
    metric_client.try_start()
    metric_tag_map = {
        "code": "0",
        "status": "success",
    }
    metric_client.report(MetricPoints.init_qps, 1, metric_tag_map)
    __report_init_done = True
    return

def get_session_cache_from_local(session_struct, rw_mode):
    session_id = ""
    expiration_time = -1
    record_count = -1
    session_def_dict = dict()
    target_file = local_cache_file_name(LOCAL_CACHE_PREFIX, session_struct.project, session_struct.table, session_struct.partition)
    target_dir = check_or_mkdir(ODPS_ACCESS_INFO_DUMP)
    target_file = os.path.join(target_dir, target_file)
    status = MetricStatus.SUCCESS  
    if not check_file(target_file):
        status = MetricStatus.LOCAL_CACHE_FILE_NOT_EXISTS_ERROR
        return status, session_id, expiration_time, record_count, session_def_dict
    with open(target_file, "r") as f:
        try:
            # Acquire the file lock
            fcntl.flock(f, fcntl.LOCK_EX)
            d = json.load(f)
            session_id = d.get(SESSION_ID, session_id)
            expiration_time = d.get(EXPIRATION_TIME, expiration_time)
            record_count = d.get(RECORD_COUNT, record_count)
            session_def_dict = d.get(SESSION_DEF_DICT, session_def_dict)
        except Exception as e:
            logger.error("fail to get session from local cache: {}, exception: {}".format(target_file, e))
            status = MetricStatus.LOCAL_CACHE_ERROR
        finally:
            # Release the file lock
            fcntl.flock(f, fcntl.LOCK_UN)
    MemSessionCache4Refresh[session_struct] = SessionCache(hash_id=session_struct.get_hash_id(), session_id=session_id,
                                                           expiration_time=expiration_time, rw_mode=rw_mode, record_count=1)
    return status, session_id, expiration_time, record_count, session_def_dict


# 创建(初始化session的函数任务)线程池
def launch_future_task_init_session_list(session_struct_list, sep, rw_mode, 
                                        default_project, connect_timeout, rw_timeout):
    # type: (set[HashableSessionStruct], str, str, str, int, int) -> None
    for i, session_struct in enumerate(session_struct_list):
        varlogger.info("CAPI InitOdpsOpenStorageSessions {: >3}/{: >3} for {}"
                       .format(i, len(session_struct_list), session_struct.to_str_basic()))
        ordered_select_columns = session_struct.get_ordered_select_columns()
        ret = interface.InitOdpsOpenStorageSessions(
                session_struct.access_id, session_struct.access_key,
                session_struct._tunnel_endpoint, session_struct._odps_endpoint,
                session_struct.project, session_struct.table, session_struct.partition,
                sep.join(session_struct.physical_partitions),
                sep.join(ordered_select_columns),
                sep, rw_mode, default_project, connect_timeout, rw_timeout)
        if ret != 0 :
            raise NebulaIOFatalError("CAPI InitOdpsOpenStorageSessions failed")
    varlogger.info("call CAPI InitOdpsOpenStorageSessions end")

def extract_local_read_session(session_struct):  
    # type: (HashableSessionStruct) -> tuple[int, str, int, int]
    ''' e.g. Typical session def
    {   "DataSchema": {
            "DataColumns": [
                {   "Comment": "",  "Name": "sample_id",    "Nullable": true,   "Type": "string"},
                {   "Comment": "",  "Name": "label",        "Nullable": true,   "Type": "array<double>"},
                {   "Comment": "",  "Name": "pay_seq_is_p4p",   "Nullable": true,   "Type": "array<array<bigint>>"}],
            "PartitionColumns": [
                {   "Comment": "",  "Name": "ds",   "Nullable": true,   "Type": "string"},
                {   "Comment": "",  "Name": "tag",  "Nullable": true,   "Type": "string"}]},
        "ExpirationTime": 1713465184134,
        "Message": "",
        "RecordCount": 1568854778,
        "SessionId": "202404180233044c8b580b00001ea902",
        "SessionStatus": "NORMAL",
        "SessionType": "BATCH_READ",
        "SplitsCount": -1,
        "SupportedDataFormat": [
            {   "Type": "ARROW",    "Version": "V5"}],
    } '''
    status = MetricStatus.SUCCESS
    session_id = ""
    expiration_time = -1
    record_count = -1
    session_def = dict()
    try:
        session_def_str = interface.ExtractLocalReadSession(session_struct.access_id, session_struct.access_key,session_struct.project, session_struct.table, session_struct.partition)
        session_def = json.loads(str(session_def_str))
    except json.decoder.JSONDecodeError as e:
        status = MetricStatus.JSON_ERROR
        msg = "fail in json decode, err:{}".format(e)
        return status, session_id, expiration_time, record_count, session_def
    except Exception as e:
        status = MetricStatus.UNKNOWN_ERROR
        msg = "fail in extract session, err:{}".format(e)
        varlogger.error("fail in extract session, err:{}, traceback:{}".format(e, traceback.format_exc()))
        return status, session_id, expiration_time, record_count, session_def

    # if no session_id get
    if not session_def.get("SessionId"):
        status = MetricStatus.FIELD_ERROR
        msg = "session_def or Id empty, def:{}".format(session_def)
        return status, session_id, expiration_time, record_count, session_def
    if "ExpirationTime" not in session_def or "RecordCount" not in session_def:
        status = MetricStatus.FIELD_ERROR
        msg = "ExpirationTime or RecordCount , def:{}".format(session_def)
        return status, session_id, expiration_time, record_count, session_def
    # success. status = 0
    session_id = session_def["SessionId"]
    expiration_time = session_def["ExpirationTime"]
    record_count = session_def["RecordCount"]
    return status, session_id, expiration_time, record_count, session_def

# MemSessionCache4Refresh: 保存session cache post结构对象(hash_id/session_id -> post body)
MemSessionCache4Refresh = dict() # type: dict[HashableSessionStruct, SessionCache]
class SessionCache(object):
    def __init__(self, hash_id, session_id, expiration_time, rw_mode, record_count):
        self.hash_id = hash_id
        self.session_id = session_id
        self.expiration_time = expiration_time
        self.rw_mode = rw_mode
        self.record_count = record_count

def post_session_cache_to_local(session_id, expiration_time, record_count, session_def_dict, session_struct):
    # type: (str, int, int, dict, HashableSessionStruct) -> MetricStatus
    # NOTE: each partition owns one cache file
    status = MetricStatus.SUCCESS
    TARGET_FILE = local_cache_file_name(LOCAL_CACHE_PREFIX, session_struct.project, session_struct.table, session_struct.partition)
    target_dir = check_or_mkdir(ODPS_ACCESS_INFO_DUMP)
    target_file = os.path.join(target_dir, TARGET_FILE)
    if os.path.exists(target_file):
        return status
    dump_dict = dict()
    dump_dict[CHECK_STATUS] = True
    dump_dict[SESSION_ID] = session_id
    dump_dict[EXPIRATION_TIME] = expiration_time
    dump_dict[RECORD_COUNT] = record_count
    dump_dict[SESSION_DEF_DICT] = session_def_dict
    with open(target_file, "w") as f:
        try:
            dump_dict[ORDERED_REQUIRED_DATA_COLUMNS] = session_struct.get_ordered_select_columns()
            # Acquire the file lock
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(dump_dict, f, indent=4)
        except Exception as e:
            logger.error("fail to insert session to local cache: {}, exception: {}".format(target_file, e))
            status = MetricStatus.LOCAL_CACHE_ERROR
        finally:
            # Release the file lock
            fcntl.flock(f, fcntl.LOCK_UN)
    return status

def post_session_cache_to_remote(job_info, session_struct, rw_mode, session_id, expiration_time):
    # type: (JobInfo, HashableSessionStruct, str, str, int) -> tuple[int, str] # noexcept. 
    # @rw_mode:  "row" / "size"; @auth_mode:  "read" / "write";   @session_type: "open_storage" / "tunnel"
    # @return: (0, None) or (1+, msg)
    status, err_msg, ret  = MetricStatus.SUCCESS, "", {}
    if not IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE:
        return status, err_msg
    ordered_required_data_columns = session_struct.get_ordered_select_columns()
    hash_id = session_struct.get_hash_id()
    request_ip = get_local_ip()
    cluster = os.environ.get("CLUSTER_NAME", os.environ.get("CALCULATE_CLUSTER"))
    site = os.environ.get("SIGMA_APP_SITE")
    data = {
        "request_time": time.time(),
        "user_id": job_info._user_id,
        "task_id": job_info._task_id,
        "app_id": job_info._app_id,
        "nebula_project": job_info._nebula_project,
        "scheduler_queue": job_info._scheduler_queue,
        "docker_image": job_info._docker_image,
        "access_id":  session_struct.access_id,
        "access_key": session_struct.access_key,
        "odps_end_point": session_struct._odps_endpoint,
        "is_foreign": job_info._is_foreign,
        "odps_quota_name": None,
        "odps_project": session_struct.project,
        "odps_table": session_struct.table,
        "table_partition": session_struct.partition,
        "required_columns": ordered_required_data_columns,
        "request_ip": request_ip,
        "rw_mode": rw_mode,
        "auth_mode": AUTH_MODE_READ,
        "session_type": OPEN_STORAGE,
        "hash_id": hash_id,
        "session_id": session_id,
        "expiration_time": expiration_time,
        "halo_worker_docker_image": job_info._halo_worker_docker_image,
        "cluster": cluster,
        "site": site,
    }
    retry_cnt = 3 # simple retry for post error or json parse err(for 502, e.g.)
    for idx in range(retry_cnt):
        try:
            resp = requests.post(url=NEBULA_OPEN_STORAGE_CACHE_SERVER, json=data) # type: requests.Response
            resp.encoding = resp.apparent_encoding or 'utf-8'
            resp.raise_for_status()
            ret = resp.json() # type: dict
            status, err_msg = MetricStatus.SUCCESS, "" # overwrite from previous except err if exist
            break
        except requests.exceptions.HTTPError as e: # raise_for_status error
            status = MetricStatus.REQUEST_ERROR
            err_msg = "fail in http get status, resp:{}, err:{}".format(resp, e)
        except requests.exceptions.RequestException as e: # requests.get error
            status = MetricStatus.REQUEST_ERROR
            err_msg = "fail in http get conn, err:{}".format(e)
        except requests.exceptions.JSONDecodeError as e: # json() format error
            leading_resp_text = resp.text[:256] if ( resp.text and len(resp.text) > 0) else "None"
            status = MetricStatus.JSON_ERROR
            err_msg = "fail in get json parsing, err:{}. leading 256 resp: {}".format(e, leading_resp_text)
        except Exception as e:
            status = MetricStatus.UNKNOWN_ERROR
            err_msg = "fail in get unexpected err:{}".format(e)
            varlogger.error("fail in get unexpected err:{}, data:{}".format(e, data))
        if not ret:
            time.sleep(idx)
    # retry fail
    if status != MetricStatus.SUCCESS:
        data_print = [ {k: v}  for k, v in data.items() if k not in ["access_id", "access_key"]]
        logger.error("post_session_cache_to_remote error {}, data: {}".format(err_msg, data_print) )
        return status, err_msg
    # is_ok is None or False
    if not ret.get('is_ok'):
        status = MetricStatus.FIELD_ERROR
        err_msg = "post_session_cache_to_remote fail with is_ok False, resp: {}".format(ret)
        return status, err_msg
    MemSessionCache4Refresh[session_struct] = SessionCache(hash_id=hash_id, session_id=session_id,
                                                        expiration_time=expiration_time, rw_mode=rw_mode, record_count=1)
    # success
    return status, err_msg


def repeat_refresh_read_session_batch(interval):
    job_info = get_job_info()
    metric_tag_map = {
        "code": "0",
        "status": "success",
    }
    metric_client = metric_factory.get("openstorage_session_refresh")
    metric_client.try_start()
    logger.debug("repeat_refresh_read_session_batch metric_tag_map: {}".format(metric_tag_map))
    def iter_to_post_refreshed_session():
        expire_time_valid_start = (int(time.time()) - 30*24*3600) * 1000 # get timestamp of last month in ms. the expire time MUST bigger than it
        expire_time_valid_end = (int(time.time()) + 30*24*3600) * 1000 # get timestamp of next month in ms. the expire time MUST small than it
        MemSessionCache4RefreshList = list(MemSessionCache4Refresh.items())
        for session_struct, cache in MemSessionCache4RefreshList:
            hash_id,session_id = cache.hash_id, cache.session_id
            expire_time_ms = interface.GetSessionExpireTimestamp(session_id) # type: int
            if expire_time_valid_start < expire_time_ms * 1000 < expire_time_valid_end: # sec timestamp
                expire_time_ms = expire_time_ms * 1000 # check if for second timestamp format
            # check for expired timestamp value
            if expire_time_ms < int(time.time() * 1000) + SESSION_EXPIRATION_THRESHOLD:
                varlogger.info("openstorage neednot refresh session_id:{} with old expire_ms:{}".format(session_id, expire_time_ms))
                continue # if expire_time_ms is nearly expired, it must not be refreshed
            status, err_msg = post_session_cache_to_remote(job_info=job_info, session_struct=session_struct, rw_mode=cache.rw_mode, 
                                    session_id=cache.session_id, expiration_time=expire_time_ms )
            varlogger.info("openstorage refreshed session_id:{} with new expire_ms:{}, post-status:{}, post-msg:{}".format(session_id, expire_time_ms, status, err_msg))
    pass
    while True:
        varlogger.info("openstorage refresh_read_session begin to work for {} sessions".format(len(MemSessionCache4Refresh)))
        metric_time = time.time()
        metric_client.try_start()
        status = interface.RefreshReadSessionBatch()
        # TODO(xyh): metric_tag_map["session_id"] in future
        if status == MetricStatus.SUCCESS:
            metric_tag_map["code"] = "0"
            metric_tag_map["status"] = "success"
        else:
            metric_tag_map["code"] = "1"
            metric_tag_map["status"] = "fail"
            logger.warning("Unable to refresh read session occurred, please check the cause.")
        metric_tag_map["code"] = str(status)
        metric_time = round((time.time() - metric_time) * 1000)
        metric_client.report(MetricPoints.latency_ipc_ms, metric_time, metric_tag_map )
        metric_client.report(MetricPoints.refresh_qps, 1, metric_tag_map )
        # TODO: rm me when bug fixed. log to make sure is posted. no need to worry log flush. it's one log for a hour
        # TODO: finger out what happens if only partial success refreshed when multiple sessions exist
        iter_to_post_refreshed_session()
        # metric_client.try_close()
        time.sleep(interval)

def repeat_refresh_halo_metrics(interval):
    metric_tag_map = {
        "ds_name": "odps_openstorage",
        "multiprocess_seq": str(get_child_order()),
    }
    metric_client = metric_factory.get("openstorage_session_refresh")
    metric_client.try_start()
    logger.debug("repeat_refresh_halo_metrics metric_tag_map: {}".format(metric_tag_map))
    halo_endpoint = get_tunnel_endpoint()
    if halo_endpoint.startswith("http://") or halo_endpoint.startswith("https://"):
        varlogger.info("repeat_refresh_halo_metrics stop as endpoint:{} not support".format(halo_endpoint))
        return
    elif not halo_endpoint.startswith("ipc://"):
        varlogger.info("repeat_refresh_halo_metrics stop as endpoint:{} unrecognized".format(halo_endpoint))
        return
    while True:
        resp = 0
        try:
            ''' metric_type:
            - 1: get_read_bytes
            - 2: get_reset_read_bytes
            '''
            metric_type = 2
            resp = interface.GetHaloAgentMetric(halo_endpoint, metric_type) # type: int
            if resp < 0:
                varlogger.warning("repeat_refresh_halo_metrics get {} from endpoint:{}, type:{}".format(resp, halo_endpoint, metric_type))
            elif resp > 0:
                metric_client.report(MetricPoints.read_byte, resp, metric_tag_map )
            else:
                pass # collect zero value
        except Exception as e:
            varlogger.info("repeat_refresh_halo_metrics get halo_endpoint fail, err:{}, trace:{}".format(e, traceback.format_exc()))
        # TODO: rm me when bug fixed. log to make sure is posted. no need to worry log flush. it's one log for a hour
        # TODO: finger out what happens if only partial success refreshed when multiple sessions exist
        # metric_client.try_close()
        remain_interval = interval - (time.time() % interval)
        time.sleep(max(remain_interval, 1)) # at least 1 second every report


__has_started_refresh_session_thread = False
def try_start_refresh_session_thread():
    global __has_started_refresh_session_thread
    if not __has_started_refresh_session_thread:
        __has_started_refresh_session_thread = True
        interval_seconds = 1 * 60 * 60
        refresh_thread = threading.Thread(target=repeat_refresh_read_session_batch, args=(interval_seconds,))
        refresh_thread.daemon = True
        refresh_thread.start()

__has_started_refresh_halo_metrics = False
def try_start_refresh_halo_metrics():
    global __has_started_refresh_halo_metrics
    if not __has_started_refresh_halo_metrics:
        __has_started_refresh_halo_metrics = True
        interval_seconds = 1 * 4 # half of report interval, keep counter increase smoother
        halo_metric_thread = threading.Thread(target=repeat_refresh_halo_metrics, args=(interval_seconds,))
        halo_metric_thread.daemon = True
        halo_metric_thread.start()

def create_and_post_read_session_list(session_struct_set, sep, rw_mode, default_project,
                                    connect_timeout, rw_timeout):
    # type: ( set[HashableSessionStruct], str, str, str, int, int ) -> any
    # 1. Try to get session from remote, in case of failover restart,  worker0 uses different session_id from other workers.
    # NOTE:
    # Retry less times (now try 3 times), because very poossibley fail with no entry instead of query timeout. 
    # all sessions in same session map, only need one fresh thread try_start_refresh_session_thread()
    if len(session_struct_set) == 0:
        return
    logger.info("Going to create new sessions in count:{}".format(len(session_struct_set)))
    retry_times = 1
    
    launch_future_task_init_session_list(session_struct_set, sep, rw_mode, default_project, connect_timeout, rw_timeout)

    job_info = get_job_info()
    metric_tag_map_success = {
        "code": "0",
        "status": "success",
    }
    metric_client = metric_factory.get("openstorage_session_create")
    metric_client.try_start()
    def create_and_post_read_session_struct(tid, session_struct, result_group):
        # type: (int, HashableSessionStruct, list[str]) -> None
        project, table, partition = session_struct.project, session_struct.table, session_struct.partition
        retry_wait_seconds = 10
        metric_tag_map = copy.deepcopy(metric_tag_map_success)
        metric_tag_map["odps_project"],metric_tag_map["odps_table"],metric_tag_map["odps_partition"] = project,table,partition # odps-proj, not nebula's
        metric_time_start = time.time()
        status = MetricStatus.SUCCESS
        err_msg = ""
        for retry_cnt in range(retry_times + 1):
            if retry_cnt > 0:
                metric_tag_map["code"] = str(status)
                metric_tag_map["status"] = "fail"
                metric_tag_map["session_id"] = "null"
                metric_tag_map["hash_id"] = "null"
                metric_client.report(MetricPoints.create_qps, 1, metric_tag_map )
                logger.warning("{} fail in create_and_post retry_cnt. status:{}, session_id:{}".format(session_struct.to_str_basic(), status, session_id))
                time.sleep(retry_wait_seconds)
            status, session_id, expiration_time, record_count, session_def_dict = extract_local_read_session(session_struct)
            if status != MetricStatus.SUCCESS:
                varlogger.warning("{} fail in extract. status:{}, msg:{}".format(session_struct.to_str_basic(), status, session_id))
                continue
            # insert session def to local cache
            status = post_session_cache_to_local(session_id, expiration_time, record_count, session_def_dict, session_struct)
            if status != MetricStatus.SUCCESS:
                varlogger.warning("{} fail in insert. status:{}, msg:{}".format(session_struct.to_str_basic(), status, session_id))
                continue
            status, err_msg = post_session_cache_to_remote(job_info, session_struct, rw_mode, session_id, expiration_time)
            if status != MetricStatus.SUCCESS:
                varlogger.warning("{} fail in post. status:{}, session_id:{}, msg:{}".format(session_struct.to_str_basic(), status, session_id, err_msg))
                continue
            # success
            logger.info("{} success in extract and post session.".format(session_struct.to_str_basic()))
            result_group[tid] = session_id
            metric_tag_map["code"] = str(status)
            metric_tag_map["status"] = "success"
            metric_tag_map["session_id"] = session_id
            latency_post_ms = round((time.time() - metric_time_start) * 1000)
            metric_client.report(MetricPoints.latency_post_ms, latency_post_ms, metric_tag_map )
            metric_client.report(MetricPoints.create_qps, 1, metric_tag_map )
            return
        raise NebulaIOFatalError("{} fail in create_and_post retry_cnt:{}, status:{}, msg:{}".format(session_struct.to_str_basic(), retry_cnt, status, session_id))
        # return
    # launch thread get_and_regist_read_session_struct
    thread_group = [] # type: list[threading.Thread]
    session_id_group = [ "" for i in range(len(session_struct_set))] # type: list[str] #擦python2的屁股, Thread无法返回值 用队列传递
    for i, session_struct in enumerate(session_struct_set):
        thread_group.append(threading.Thread(target=create_and_post_read_session_struct, args=(i, session_struct, session_id_group)))
        thread_group[i].start()
    for i, session_struct in enumerate(session_struct_set):
        thread_group[i].join()
        # if session_id_group[i] == "":
        #     raise NebulaIOFatalError("here no need to raise, Any subthread fial 任意子线程一旦失败 会主动抛出")
    try_report_metric_init_done(len(session_struct_set))
    return

def get_session_cache_from_remote(job_info, session_struct, rw_mode):
    # type: (JobInfo, HashableSessionStruct, str ) -> tuple[int, str, str]
    # noexcept. return: (0, hash_id, session_id) or (1+, hash_id, msg)
    hash_id = session_struct.get_hash_id()
    partition_modified_timestamp = session_struct.get_partition_modified_timestamp()
    request_ip = get_local_ip()
    cluster = os.environ.get("CLUSTER_NAME", os.environ.get("CALCULATE_CLUSTER"))
    site = os.environ.get("SIGMA_APP_SITE")
    data = {
        "request_time": time.time(),
        "hash_id": hash_id,
        "task_id": job_info._task_id,
        "app_id": job_info._app_id,
        "odps_project": session_struct.project,
        "odps_table": session_struct.table,
        "table_partition": session_struct.partition,
        "worker_id": job_info._rank,
        "request_ip": request_ip,
        "cluster": cluster,
        "site": site,
    }
    status = MetricStatus.SUCCESS
    err_msg = ""
    if not IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE:
      return MetricStatus.REQUEST_ERROR, "session cache service not reachable."
    try:
        resp = requests.get(url=NEBULA_OPEN_STORAGE_CACHE_SERVER, params=data) # type: requests.Response
        resp.encoding = resp.apparent_encoding or 'utf-8'
        resp.raise_for_status()
        ret = resp.json() # type: dict
    except requests.exceptions.HTTPError as e: # raise_for_status error
        status = MetricStatus.REQUEST_ERROR
        err_msg = "fail in http get status, code:{}, resp:{}".format(resp, e)
    except requests.exceptions.JSONDecodeError as e: # json() format error
        leading_resp_text = resp.text[:256] if ( resp.text and len(resp.text) > 0) else "None"
        status = MetricStatus.JSON_ERROR
        err_msg = "fail in get json parsing, err:{}. leading 256 resp: {}".format(e, leading_resp_text)
    except requests.exceptions.RequestException as e: # requests.get error
        status = MetricStatus.REQUEST_ERROR
        err_msg = "fail in http get conn, err:{}".format(e)
    except Exception as e:
        status = MetricStatus.UNKNOWN_ERROR
        err_msg = "fail in get unexpected err:{}".format(e)
        varlogger.error("fail in get unexpected err:{}, traceback:{}".format(e, traceback.format_exc()))
    finally:
        if status != 0:
            return status, err_msg
    if not ret or not ret.get("is_ok", False) or not ret.get("session_id"):
        status = MetricStatus.FIELD_ERROR
        err_msg = "get cache remote is_ok or session_id False, resp: {}".format(ret)
        return status, err_msg
    # success get session_id, but may not valid
    session_id = ret["session_id"]
    expiration_time = ret.get("expiration_time", 0)
    session_create = ret.get("session_create")
    app_id = ret.get("app_id")
    valid_start = (int(time.time()) - 30*24*3600) * 1000 # get timestamp of last month in ms. the expire time MUST bigger than it
    valid_end = (int(time.time()) + 30*24*3600) * 1000 # get timestamp of next month in ms. the expire time MUST small than it

    # check for second timestamp format
    if valid_start < expiration_time * 1000 < valid_end: # sec timestamp
        expiration_time = expiration_time * 1000
    # check for expired timestamp value
    if expiration_time < int(time.time() * 1000) + SESSION_EXPIRATION_THRESHOLD:
        status = MetricStatus.OUTDATE_ERROR
        err_msg = "session_id expired, expiration_time:{}, resp:{} ".format(expiration_time, ret)
        return status, err_msg
    logger.info("cache app_id: {}, task app_id: {}, session_create: {}, partition_modified_timestamp: {}".format(app_id, job_info._app_id, session_create, partition_modified_timestamp))
    if not is_session_creator() and str(job_info._app_id) != str(app_id):
        status = MetricStatus.WAITING
        err_msg = "wait session creator"
        return status, err_msg
    if session_create and app_id:
        if int(partition_modified_timestamp) > int(session_create) and str(job_info._app_id) != str(app_id):
            status = MetricStatus.PATITION_MODIDIFIED
            err_msg = "odps partition modified, session_create:{}, partition_modified_timestamp:{}".format(session_create, partition_modified_timestamp)
            return status, err_msg
    if force_recreate_session():
        if str(job_info._app_id) != str(app_id):
            status = MetricStatus.FORCE_RECREATE_SESSION
            err_msg = "force_recreate_session is True, do not use other task's session."
            return status, err_msg
    # success, status = 0
    MemSessionCache4Refresh[session_struct] = SessionCache(hash_id=hash_id, session_id=session_id,
                                                        expiration_time=expiration_time, rw_mode=rw_mode, record_count=1)
    logger.info("cache ok session_id: {}".format(session_id))
    return status, session_id

def local_register_read_session(session_id, session_struct, 
                                sep, rw_mode, default_project,
                                connect_timeout, rw_timeout):
  # type: (str, HashableSessionStruct, str, str, str, int, int) -> tuple[int, str]
  # noexcept. return: (0, None) or (1+, msg)
  status = MetricStatus.SUCCESS
  try:
    ordered_required_data_columns = session_struct.get_ordered_select_columns()
  except Exception as e:
    status = MetricStatus.UNKNOWN_ERROR
    varlogger.error("fail in local_register_read_session err:{}".format(e))
    return status, str(e)
  try:
    session_id = str(session_id)
    register_light = False
    expiration_time = -1
    record_count = -1
    session_def_str = ""
    status = interface.RegisterOdpsOpenStorageSession(
                        str(session_struct.access_id), str(session_struct.access_key),
                        str(session_struct._tunnel_endpoint), str(session_struct._odps_endpoint),
                        str(session_struct.project), str(session_struct.table), str(session_struct.partition),
                        str(sep.join(ordered_required_data_columns)),
                        str(sep), str(rw_mode), str(default_project),
                        connect_timeout, rw_timeout,
                        register_light, str(session_id),
                        expiration_time, record_count, str(session_def_str))
  except Exception as e:
    status = MetricStatus.UNKNOWN_ERROR
    varlogger.error("fail in local_register_read_session err:{}".format(e))
    return status, str(e)
  if status != MetricStatus.SUCCESS :
    status = MetricStatus.CALL_ERROR
    return status, "RegisterOdpsOpenStorageSession_fail" #Locally register OpenStorageSession failed
  return status, None

def local_register_read_session_light(session_struct, sep, rw_mode, default_project,
                                    connect_timeout, rw_timeout,
                                    session_id, expiration_time,
                                    record_count, session_def_dict):
    # type: (HashableSessionStruct, str, str, str, int, int, str, int, int, dict) -> tuple[int, str]
    # noexcept. return: (0, None) or (1+, msg)
    status = MetricStatus.SUCCESS
    register_light = True
    try:
        ordered_select_columns = session_struct.get_ordered_select_columns()
        if isinstance(session_def_dict, six.string_types):
            session_def_str = session_def_dict
        elif isinstance(session_def_dict, dict):
            session_def_str = json.dumps(session_def_dict)
        else:
            raise ValueError("session_def unsurported type: {}".format(type(session_def_dict)))
        status = interface.RegisterOdpsOpenStorageSession(
                            str(session_struct.access_id), str(session_struct.access_key),
                            str(session_struct._tunnel_endpoint), str(session_struct._odps_endpoint),
                            str(session_struct.project), str(session_struct.table), str(session_struct.partition),
                            str(ordered_select_columns),
                            str(sep), str(rw_mode), str(default_project),
                            connect_timeout, rw_timeout,
                            register_light, str(session_id),
                            expiration_time, record_count, str(session_def_str))
    except Exception as e:
        logger.error("fail to RegisterOdpsOpenStorageSession light, exception: {}".format(e))
        status = MetricStatus.UNKNOWN_ERROR
        return status, str(e)
    if status != MetricStatus.SUCCESS :
        status = MetricStatus.CALL_ERROR
        return status, "RegisterOdpsOpenStorageSession_light_fail" #Locally register OpenStorageSession failed
    return status, None

def get_and_register_read_session_list(session_struct_set,
                                  sep, rw_mode, default_project,
                                  connect_timeout, rw_timeout):
    # type: (set[HashableSessionStruct], str, str, str, int, int) -> set[HashableSessionStruct]
    retry_times = 1 if is_session_creator() else 60
    retry_wait_seconds = 1 if is_session_creator() else 10
    job_info = get_job_info()
    metric_tag_map_success = {
            "code": "0",
            "status": "success",
    }
    metric_client = metric_factory.get("openstorage_session_cache")
    metric_client.try_start()

    def get_and_regist_read_session_struct(tid, session_struct, result_group):
        # type: (int, HashableSessionStruct, list[str]) -> None
        project, table, partition = session_struct.project, session_struct.table, session_struct.partition
        hash_id = session_struct.get_hash_id()
        status, session_id = MetricStatus.SUCCESS, "null"
        metric_tag_map = copy.deepcopy(metric_tag_map_success)
        metric_tag_map["odps_project"],metric_tag_map["odps_table"],metric_tag_map["odps_partition"] = project,table,partition # odps-proj, not nebula's
        for retry_cnt in range(retry_times + 1):
            if retry_cnt > 0:
                metric_tag_map["code"] = str(status)
                metric_tag_map["status"] = "fail"
                metric_tag_map["session_id"] = "null"
                metric_tag_map["hash_id"] = hash_id
                metric_client.report(MetricPoints.cache_qps, 1, metric_tag_map )
                time.sleep(retry_wait_seconds)
            varlogger.info("GetSess tid {:0>4}: get and register for {}".format(tid, session_struct.to_str_basic()))
            metric_time_start = time.time()
            dump_debug_access_info_batch([session_struct])
            try:
                # https://code.alibaba-inc.com/algo/paiio/commit/8db1440594d1a09a91f1041a83f58493ddcaa68d
                # 1. what does local_cache store for?
                #     a. sessionId, recordCount, required_data_columns, expirationTime, sessionDef
                #     b. OR,  sessionDef only, other info can be fetched from server
                # 2. After session refresh, who will update local_cache.expirationTime? 
                #     a. local_cache.expirationTime no one will update
                #     b. However, now expirationTime in C++runtime won't used, so needn't update for now, temporaly
                status, session_id, expiration_time, record_count, session_def_dict = get_session_cache_from_local(session_struct, rw_mode)
                if status == MetricStatus.SUCCESS:
                    # This API will be called frequently and requires low overhead.
                    # 1. Override status = interface.RegisterOdpsOpenStorageSession to accept several key pieces of information, or the entire sessiondef.
                    # 2. Separately, write local_register_read_session / RegisterOdpsOpenStorageSession:
                    #     a. Do not query the Odps metadata and do not sort the columns (as seen in the previous implementation of RegisterOdpsOpenStorageSession,
                    #        the sorted columns are only used for C++ GetReadSession; if C++ doesn't do this, the column information won't be used).
                    #     b. As mentioned above, C++ does not call GetReadSession, reducing one query.
                    #     c. All information required by C++ is obtained and passed from Python, ensuring C++ DO NOT make any additional queries at session level.
                    status, err_msg = local_register_read_session_light(session_struct, sep, rw_mode, default_project,
                                                                connect_timeout, rw_timeout,
                                                                session_id, expiration_time,
                                                                record_count, session_def_dict)
                    if is_session_creator():
                        status, err_msg = post_session_cache_to_remote(job_info, session_struct, rw_mode, session_id, expiration_time)
                        if status != MetricStatus.SUCCESS:
                            varlogger.warning("{} fail in post. status:{}, session_id:{}, msg:{}".format(session_struct.to_str_basic(), status, session_id, err_msg))
                            continue
            except Exception as e:
                logger.error("GetSess tid {:0>4}: local_register exception: {}".format(tid, e))
                varlogger.error("GetSess tid {:0>4}: local_register exception:{}, traceback:{}".format(tid, e, traceback.format_exc()))
                status = MetricStatus.UNKNOWN_ERROR
            
            if status != MetricStatus.SUCCESS:
                # when reached here, means local_cache DO NOT CONTAINS VALID session
                # suppose for each <project, table, partition> set, in task pod lifecycle, this func only called once, so allow the heavy cost before
                status, session_id = get_session_cache_from_remote(job_info, session_struct, rw_mode)
                if status != MetricStatus.SUCCESS :
                    logger.info("GetSess tid {:0>4}: Unable to get session_id for {} from cache, msg:{}".format(tid, session_struct.to_str_basic(), session_id))
                    continue
                varlogger.info("GetSess tid {:0>4}: get_and_regist session for {} from remote, session_id:{}".format(tid, session_struct.to_str_basic(), session_id))

                status, msg = local_register_read_session(session_id, session_struct, sep, rw_mode, default_project, connect_timeout, rw_timeout)
                if status != MetricStatus.SUCCESS:
                    varlogger.info("GetSess tid {:0>4}: unable to regist session for {}, retry_cnt: {}, session_id:{} and sleep 10s.".format(tid, session_struct.to_str_basic(), retry_cnt, session_id))
                    continue
                status, session_id, expiration_time, record_count, session_def_dict = extract_local_read_session(session_struct) # 此处 extract local read session 是为了将session def str 取出存入 local cache
                if status != MetricStatus.SUCCESS:
                    varlogger.info("GetSess tid {:0>4}: unable to regist session for {}, retry_cnt: {}, session_id:{} and sleep 10s.".format(tid, session_struct.to_str_basic(), retry_cnt, session_id))
                    continue
                logger.info("GetSess tid {:0>4}: successfully get and register session for {}".format(tid, session_struct.to_str_basic()))
                status = post_session_cache_to_local(session_id, expiration_time, record_count, session_def_dict, session_struct)# insert session def to local cache
                if status != MetricStatus.SUCCESS:
                    varlogger.warning("{} fail in insert. status:{}, msg:{}".format(session_struct.to_str_basic(), status, session_id))
                    continue
                if is_session_creator():
                    status, err_msg = post_session_cache_to_remote(job_info, session_struct, rw_mode, session_id, expiration_time)
                    if status != MetricStatus.SUCCESS:
                        varlogger.warning("{} fail in post. status:{}, session_id:{}, msg:{}".format(session_struct.to_str_basic(), status, session_id, err_msg))
                        continue
            metric_tag_map["code"] = str(status)
            metric_tag_map["status"] = "success"
            metric_tag_map["session_id"] = str(session_id)
            metric_tag_map["hash_id"] = str(hash_id)
            latency_get_ms = round((time.time() - metric_time_start) * 1000)
            metric_client.report(MetricPoints.cache_qps, 1, metric_tag_map )
            metric_client.report(MetricPoints.latency_get_ms, latency_get_ms, metric_tag_map )
            result_group[tid] = session_id
            return
        
        if not is_session_creator():  # session creator not fail here, go to next session create
            raise NebulaIOFatalError("GetSess tid {:0>4}: query get session failed for {} after enouph retries".format(tid, session_struct.to_str_basic()))
        logger.info("GetSess tid {:0>4}: Unable to query get session for {} from remote, prepair to create new one".format(tid, session_struct.to_str_basic()))
        result_group[tid] = ""
        return

    # launch thread get_and_regist_read_session_struct
    thread_group = [] # type: list[threading.Thread]
    session_id_group = [ "" for i in range(len(session_struct_set))] # type: list[str] # for py2, Thread cannot return, have to use a queue
    for i, session_struct in enumerate(session_struct_set):
        thread_group.append(threading.Thread(target=get_and_regist_read_session_struct, args=(i, session_struct, session_id_group)))
        thread_group[i].start()
    
    create_session_struct_set = set() # type: set[HashableSessionStruct]
    for i, session_struct in enumerate(session_struct_set):
        thread_group[i].join()
        if session_id_group[i] == "":
            create_session_struct_set.add(session_struct)
    varlogger.info("Get and register odps open storage session done, unccessful session len: {}.".format(len(create_session_struct_set)))
    try_report_metric_init_done(len(session_struct_set))
    return create_session_struct_set
