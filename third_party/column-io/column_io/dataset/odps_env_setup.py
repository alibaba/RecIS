# -*- coding: utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import time
import subprocess
import psutil
import six
import getpass
import uuid
import argparse
import contextlib
import errno
import tempfile
import fcntl
import requests
from collections import OrderedDict
from binascii import b2a_hex, a2b_hex
from Crypto.Cipher import AES
from requests import RequestException

from column_io.dataset.log_util import logger, varlogger
from column_io.dataset.open_storage_utils import decode
from column_io.dataset.open_storage_utils import check_auth_and_data
from column_io.dataset.open_storage_utils import dump_debug_access_info_batch
from column_io.dataset.open_storage_utils import create_and_post_read_session_list
from column_io.dataset.open_storage_utils import get_and_register_read_session_list
from column_io.dataset.open_storage_utils import try_start_refresh_session_thread,try_start_refresh_halo_metrics
from column_io.dataset.open_storage_utils import HashableSessionStruct
from column_io.dataset.open_storage_utils import is_session_creator, is_column_io, set_session_creator_override
from column_io.dataset.open_storage_utils import ODPS_ENDPOINT, IS_ODPS_ENDPOINT_REACHABLE
from column_io.dataset.job_info import get_app_id, get_work_id, is_notebook, get_app_config
from column_io.dataset.partition_util import ODPS_PARTITION_MODE_KEY_NAME, PartitionMode


ODPS_OPEN_STORAGE_printed = False
ODPS_OPEN_STORAGE_ENV = "ODPS_OPEN_STORAGE_MDL" # for torch in mdl/xrec
def is_turn_on_odps_open_storage():
    global ODPS_OPEN_STORAGE_printed
    if ODPS_OPEN_STORAGE_ENV not in os.environ:
        return False
    val = os.environ[ODPS_OPEN_STORAGE_ENV]
    if not ODPS_OPEN_STORAGE_printed:
        logger.info("turn on OdpsOpenStorage")
        ODPS_OPEN_STORAGE_printed = True
    return val == "1"

def ensure_standard_path_format(paths):
    if not isinstance(paths, (set, tuple, list)):
        raise ValueError("Input paths must in (set, tuple, list), got type: {}".format(type(paths)))
    standard_paths = set()
    for p in paths:
        if p is None or p == "" or p == b"":
            raise ValueError("path: {} invalid".format(p))
        if isinstance(p, bytes):
            p = p.decode("utf-8")
        standard_p = p.split("?")[0]
        standard_paths.add(standard_p)
    res_list = list(standard_paths)
    return res_list

def _encrypt(message, key):
    while len(key) < 32:
        key += "1"
    mode = AES.MODE_OFB
    cryptor = AES.new(key.encode("utf-8"), mode, b"0000000000000000")
    length = 16
    count = len(message)
    if count % length != 0:
        add = length - (count % length)
    else:
        add = 0
    message = message + ("\0" * add)
    ciphertext = cryptor.encrypt(message.encode("utf-8"))
    result = b2a_hex(ciphertext)
    return result.decode("utf-8")


def _get_userinfo_from_env():
    project = os.environ["project_name"] if os.environ.get("project_name", "") else None
    access_id = os.environ["access_id"] if os.environ.get("access_id", "") else None
    access_key = os.environ["access_key"] if os.environ.get("access_key", "") else None
    end_point = os.environ["end_point"] if os.environ.get("end_point", "") else None
    foreign = True if os.environ.get("foreign", "").lower() == "true" else False
    refresh = True if os.environ.get("refresh", "").lower() == "true" else False
    return project, access_id, access_key, end_point, foreign, refresh


def _get_userinfo_from_odpscmd_home(odpscmd_home=None):
    odpscmd_home = os.getenv("ODPSCMD_HOME") if odpscmd_home is None else odpscmd_home
    if not odpscmd_home:
        return None, None, None, None, False, False
    odpsconf = "%s/conf/odps_config.ini" % odpscmd_home
    if not os.path.exists(odpsconf):
        raise RuntimeError("%s not exists!" % odpsconf)
    project = None
    access_id = None
    access_key = None
    end_point = None
    foreign = False
    refresh = False
    with open(odpsconf, "rb") as f:
        for line in f:
            line = line.lstrip()
            if line.startswith(b"#"):
                continue
            strlist = line.split(b"=", 1)
            if len(strlist) < 2:
                continue
            key = strlist[0].strip().decode("utf-8")
            value = strlist[1].strip().decode("utf-8")
            if key == "project_name":
                project = value
            if key == "access_id":
                access_id = value
            if key == "access_key":
                access_key = value
            if key == "end_point":
                end_point = value
            if key == "foreign":
                foreign = value.lower() == "true"
            if key == "refresh":
                refresh = value.lower() == "true"
    return project, access_id, access_key, end_point, foreign, refresh


def _get_userinfo_from_config(conf=None):
    if conf is None:
        return None, None, None, None, False, False
    if "odps_perm" in conf:
        conf = conf["odps_perm"]
    access_id = _get_config(conf, "access_id")
    access_key = _get_config(conf, "access_key")
    project = _get_config(conf, "project")
    if not project:
        project = _get_config(conf, "project_name")
    end_point = _get_config(conf, "end_point")
    foreign = _get_config(conf, "foreign", False)
    refresh = _get_config(conf, "refresh", False)
    return project, access_id, access_key, end_point, foreign, refresh


def _get_config(conf, key, default=None):
    if key in conf:
        return conf[key]
    else:
        return default


SUCCESS = 0
NOT_EXIST = 1
PERM_TASK_FAIL = -1
REQUEST_FAIL = -2
ODPS_TABLES = []
HTTP_SUCCESS_CODE = 200


def _get_request_data(
        task_id,
        project_name,
        access_id,
        access_key,
        end_point,
        foreign=False,
        table_name=None,
        volume_name=None,
        refresh=False,
    ):
    # type: (str,str,str,str,str,bool,str,str,bool) -> str
    if table_name:
        define_names = table_name
    elif volume_name:
        define_names = volume_name
    else:
        raise RuntimeError("Please specify table or volume name")
    data = {
        "task_id": task_id,
        "table_names": define_names,
        "user_name": "kaixu.rkx",
        "project_name": project_name,
        "access_id": access_id,
        "access_key": access_key,
        "end_point": end_point,
        "foreign": str(foreign).lower(),
        "docker_image": "reg.docker.alibaba-inc.com/alimama/xdl:2.1-2106-sdk39",
        "refresh": str(refresh).lower(),
    }
    return data


# only handle odps authorize response
def _handle_response(response):
    if response.status_code != HTTP_SUCCESS_CODE:
        raise RuntimeError(
            "http request failed, status code: %s, reason: %s."
            % (response.status_code, response.reason)
        )
    rsp = json.loads(response.text)
    rsp_check = rsp.get("rsp_code", 0)
    if rsp_check != 0 or "data" not in rsp:
        logger.error("odps auth failed:", rsp.get("message"))
        return None
    return rsp.get("data")


def _send_request_to_server(method, url, data=None, count=1, error=None, max_count=3):
    if count > max_count:
        raise RuntimeError("http request %s failed, msg %s" % (url, error))
    method = method.upper()
    try:
        if "GET" == method:
            response = requests.get(url)
        elif "POST" == method:
            response = requests.post(url, json=data)
        else:
            raise ValueError("request method %s error" % (method))
        return _handle_response(response)
    except RequestException as e:
        count += 1
        time.sleep(10)
        return _send_request_to_server(method, url, data, count, e, max_count)


def _save_odps_io_config(result, output_dir="odps_io_config"):
    output_dir = os.environ.get("ODPS_IO_CONFIG_FILE", "odps_io_config")
    if result["status"] == "failed":
        raise RuntimeError(result["error"])
    elif result["status"] != "finished":
        logger.error(result["status"], result["error"])
        return PERM_TASK_FAIL
    if "odps_io_config" not in result or not result["odps_io_config"]:
        logger.error("result has not odps_io_config.", result["error"])
        return PERM_TASK_FAIL
    filename = output_dir
    if os.path.isdir(output_dir):
        filename = os.path.join(output_dir, "odps_io_config")
    fd, tmp_filename = tempfile.mkstemp(dir=".")
    os.environ["ODPS_IO_CONFIG_FILE"] = filename
    with open(tmp_filename, "w") as f:
        f.write(result["odps_io_config"])
    os.close(fd)
    import shutil
    shutil.move(tmp_filename, filename)
    logger.info("odps authority certification success.")
    return SUCCESS


def _get_cache(data, output_dir="odps_io_config"):
    cache_url = "https://odps.nebula.alibaba-inc.com/authorize/cache"
    result = None
    try:
        result = _send_request_to_server("post", cache_url, data)
    except Exception as e:
        print(e)
    if result:
        return _save_odps_io_config(result, output_dir)
    return PERM_TASK_FAIL


def _get_task_status(url, count=1, max_count=120):
    if count > max_count:
        return None
    result = None
    try:
        result = _send_request_to_server("get", url)
    except Exception as e:
        logger.error(e)
    if not result or result["status"] in ["running", "ready", "pending"]:
        time.sleep(10)
        if result:
            logger.error( "odps authorize job is {},\tcheck status count is {}".format(str(result["status"]), str(count)))
        count += 1
        return _get_task_status(url, count, max_count)
    return result


def _sync_auth(data, output_dir="odps_io_config", max_count=120):
    logger.error("Start running odps authority certification job for get odps_io_config, please waiting for a few minutes!")
    sync_url = "https://odps.nebula.alibaba-inc.com/authorize/sync"
    result = None
    try:
        result = _send_request_to_server("post", sync_url, data)
    except Exception as e:
        logger.error(e)
    if not result:
        return PERM_TASK_FAIL
    logview = result.get("logview")
    if not logview:
        logger.error(result["error"])
        return PERM_TASK_FAIL
    logger.info("=" * 100)
    logger.info(logview)
    logger.info("=" * 100)
    get_url = os.path.join(sync_url, data.get("task_id"))
    result = _get_task_status(get_url, max_count=max_count)
    if not result:
        return PERM_TASK_FAIL
    return _save_odps_io_config(result, output_dir)


_NUWA_SYNC_PROCESS_LOCK_FILE = "/tmp/nuwa_sync_process_lock"


@contextlib.contextmanager
def _lock_nuwa_sync_process():
    try:
        fd = os.open(
            _NUWA_SYNC_PROCESS_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
        fd = os.open(_NUWA_SYNC_PROCESS_LOCK_FILE, os.O_WRONLY, 0o600)
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX)
        yield
        fcntl.lockf(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def run_perm(data, output_dir="odps_io_config", refresh=False):
    if refresh:
        return _sync_auth(data, output_dir)
    ret = _get_cache(data, output_dir)
    if ret != SUCCESS:
        return _sync_auth(data, output_dir)
    return ret


def _get_nuwa_process():
    nuwa_process_name = "/usr/local/nuwa_config_sync/nuwa_config_sync.py"
    procs = []
    for proc in psutil.process_iter(["cmdline"]):
        cmdline = proc.info["cmdline"]
        if len(cmdline) > 1 and cmdline[1] == nuwa_process_name:
            procs.append(proc)
    return procs


def is_asi_cluster():
    cluster_name = os.environ.get("CLUSTER_NAME", None)
    if cluster_name and cluster_name.startswith("asi"):
        return True
    return False


def setup_nuwa_process():
    if is_asi_cluster():
        return
    with _lock_nuwa_sync_process():
        if _get_nuwa_process():
            logger.info("nuwa process already exists.")
            return
        logger.info("set up nuwa process")
        subprocess.Popen([
                "python2 /usr/local/nuwa_config_sync/nuwa_config_sync.py --mode prod --cluster AY118F --mount_tables_config cn-mongolia-in:ODPS,cn-zhangbei-in:ODPS,cn-shanghai-corp:ODPS --interval 300"
            ],
            shell=True,
        )
        while len(_get_nuwa_process()) == 0:
            time.sleep(2)


def close_nuwa_process():
    with _lock_nuwa_sync_process():
        for proc in _get_nuwa_process():
            proc.terminate()
            logger.info("nuwa process close success.")


def setup_local_odps_env(
        conf=None,
        odpscmd_home=None,
        table_name=None,
        volume_name=None,
        output_dir="odps_io_config",
    ):
    app_id = "local_" + str(uuid.uuid4()).replace("-", "")[:12]
    logger.info("setup_local_odps_env app_id:".format(app_id))
    foreign = False
    refresh = False
    prj, access_id, access_key, end_point, foreign, refresh = _get_userinfo_from_env()
    if None in (prj, access_id, access_key, end_point) or conf:
        prj, access_id, access_key, end_point, foreign, refresh = (
            _get_userinfo_from_config(conf)
        )
    if None in (prj, access_id, access_key, end_point):
        prj, access_id, access_key, end_point, foreign, refresh = (
            _get_userinfo_from_odpscmd_home(odpscmd_home)
        )
    if None in (prj, access_id, access_key, end_point):
        raise ValueError(
            "get odps userinfo failed", (prj, access_id, access_key, end_point)
        )
    data = _get_request_data(
        app_id,
        prj,
        access_id,
        access_key,
        end_point,
        foreign,
        table_name,
        volume_name,
        refresh,
    )
    ret = run_perm(data, output_dir, refresh)
    if ret == SUCCESS:
        if not os.path.exists("/usr/local/nuwa_config_sync/nuwa_config_sync.py"):
            raise RuntimeError(
                "/usr/local/nuwa_config_sync/nuwa_config_sync.py not found, please use xdl release image"
            )
        setup_nuwa_process()
    else:
        raise RuntimeError("get odps_io_config failed, error code:", ret)




def disable_local_odps_perm(): # type: () -> bool
    disable_local_odps_perm = (
        True
        if os.environ.get("PAITF_DISABLE_LOCAL_ODPS_PERM", "").lower() == "true"
        else False
    )
    return disable_local_odps_perm


def _odps_table_path_parse(paths): # type: (list[str]) -> list[str]
    origin_paths = paths
    if isinstance(paths, six.string_types):
        paths = [paths]
    results = []
    for path in paths:
        try:
            path = path.strip().split("//")[1].strip().strip("/")
            parts = path.split("/")

            # 兼容tunnel传参格式
            split_parts = [p.strip() for p in parts[2].split(".")]  # 过滤空片段（如 ".."）
            n = len(split_parts)

            if n == 1:
                parts[2] = split_parts[0]
            elif n == 2:
                parts[0] = split_parts[0] if split_parts[0] else parts[0]
                parts[2] = split_parts[1] if split_parts[1] else parts[2]
            elif n==3:
                parts[0] = split_parts[0] if split_parts[0] else parts[0]
                parts[2] = split_parts[2] if split_parts[2] else parts[2]
            else:
                raise ValueError("Invalid table path {}".format(path))
            results.append((parts[0], parts[2], "/".join(parts[3:])))
        except Exception as e:
            logger.error("failed to parse paths: {}".format(origin_paths))
            raise e
    return results


def get_odps_io_config(paths, output_dir="odps_io_config"):
    if is_turn_on_odps_open_storage():
        logger.info("open storage session is initialized, return.")
        return
    setup_nuwa_process()
    if disable_local_odps_perm():
        return
    parser = argparse.ArgumentParser(description="xdl arguments")
    parser.add_argument("--zk_addr", default=None) # TODO(not nessessary): in mdl worker, get it by by "zfs://${ZK_HOSTS}/psplus/${APP_ID}"
    args, unknown = parser.parse_known_args()
    kvaddr = args.zk_addr
    if kvaddr is None:
        if paths is None or len(paths) == 0:
            return
        if not isinstance(paths, list) or not all(
            isinstance(_, six.string_types) for _ in paths
            ):
            raise ValueError("paths error: paths should be a filled list of strings")
        def add_odps_paths(paths): # more than one data_io uses the same odps_io_config
            for path in paths:
                if path not in ODPS_TABLES:
                    ODPS_TABLES.append(path)
        add_odps_paths(paths)
        table_name = ",".join(ODPS_TABLES)
        setup_local_odps_env(table_name=table_name, output_dir=output_dir)


def try_get_conf(data, output_dir="odps_io_config", try_cnt=150):
    ret = SUCCESS
    while try_cnt > 0:
        ret = _get_cache(data, output_dir)
        if ret == SUCCESS:
            break
        time.sleep(20)
        try_cnt -= 1
    return ret


def try_run_pai(data, output_dir="odps_io_config", try_cnt=3):
    ret = SUCCESS
    while try_cnt > 0:
        ret = _sync_auth(data, output_dir)
        if ret == SUCCESS:
            break
        time.sleep(2)
        try_cnt -= 1
    return ret


def try_run_perm(data, output_dir="odps_io_config", refresh=False, try_cnt=3):
    ret = run_perm(data, output_dir, refresh)
    if ret == SUCCESS:
        return ret
    else:
        return try_run_pai(data, output_dir, try_cnt - 1)


def refresh_odps_io_config(
        project,
        access_id,
        access_key,
        end_point,
        foreign=False,
        table_name=None,
        volume_name=None,
        output_dir="odps_io_config",
        try_conf_cnt=150,
        try_pai_cnt=3,
        refresh=False,
    ):
    # type: (str, str, str, str, bool, str, str, str, int, int, bool) -> int
    worker_id = get_work_id()
    app_id = get_app_id(worker_id)
    logger.info("refresh_odps_io_config app_id:", app_id, "worker_id:", worker_id)
    setup_nuwa_process()
    data = _get_request_data(
        app_id,
        project,
        access_id,
        access_key,
        end_point,
        foreign,
        table_name,
        volume_name,
        refresh,
    )
    if worker_id != 0:
        return try_get_conf(data, output_dir, try_conf_cnt)
    else:
        return try_run_perm(data, output_dir, refresh, try_pai_cnt)


kSep=","

def init_odps_open_storage_session(paths,
                                   required_data_columns=[],
                                   mode="row", default_project="",
                                   connect_timeout=300, rw_timeout=1800,
                                   is_create_session = None):
    # paths: paths or odps table urls
    # required_data_columns: columns to read, if len(required_data_columns) == 0,
    #        read all columns.
    if str(os.getenv("OPEN_STORAGE_SESSION_TRACE", "0")) == "1":
        logger.warning("Printing traceback since env OPEN_STORAGE_SESSION_TRACE in init_odps_open_storage_session which is NOT AN ERR")
        import traceback
        traceback.print_stack()
        logger.warning("Printing traceback ends and keep running")

    #Note: some customized business wants to get new data at runtime, not from odps_table
    if is_create_session is not None:
        varlogger.info("is_create_session param set to: {} for paths: {}".format(is_create_session, paths))
        set_session_creator_override(is_create_session)

    assert(isinstance(required_data_columns, list))
    try:
        odps_token_envs = ["access_id", "access_key", "ACCESS_ID", "ACCESS_KEY", "ENCODED_ODPS_ACCESS_ID", "ENCODED_ODPS_ACCESS_KEY"]
        varlogger.info("envs: {}, envs_values: {}".format(odps_token_envs, [os.getenv(env) for env in odps_token_envs]))
        ENCODED_ODPS_ACCESS_ID = os.environ.get("ENCODED_ODPS_ACCESS_ID")
        ENCODED_ODPS_ACCESS_KEY = os.environ.get("ENCODED_ODPS_ACCESS_KEY")
        access_id = decode(ENCODED_ODPS_ACCESS_ID) if ENCODED_ODPS_ACCESS_ID else os.getenv("access_id")
        access_key = decode(ENCODED_ODPS_ACCESS_KEY) if ENCODED_ODPS_ACCESS_KEY else os.getenv("access_key")
        if access_id is None or str(access_id) == "":
            logger.error("access_id and ENCODED_ODPS_ACCESS_ID cannot be both None")
            raise ValueError("access_id and ENCODED_ODPS_ACCESS_ID cannot be both None !")
        if access_key is None or str(access_key) == "":
            logger.error("access_key and ENCODED_ODPS_ACCESS_KEY cannot be both None")
            raise ValueError("access_key and ENCODED_ODPS_ACCESS_KEY cannot be both None !")
        access_id = str(access_id)
        access_key = str(access_key)

        connect_timeout = os.environ.get("connect_timeout", connect_timeout)
        connect_timeout = int(connect_timeout)
        rw_timeout = os.environ.get("rw_timeout", rw_timeout)
        rw_timeout = int(rw_timeout)
    except Exception as e:
        raise ValueError("The following environment must be set while using odps reader " +
                       "tunnel or openstorage : tunnel_end_point, end_point. error: {}".format(e))

    session_struct_set = set()
    if not isinstance(paths, list):
        if isinstance(paths, six.string_types):
            paths = str(paths).split(",")
        else:
            msg = "paths must be string or list, but got type: {}, {}".format(type(paths), paths)
            raise ValueError(msg)

    '''
    In mdl jobs, session is created by scheduler, which won`t know which columns to read, required_data_columns is [].
    Howere, workers know which columns to read, so required_data_columns is filled with columns.
    Here making scheduler and worekr gen difference hashid, making worker cannot find session.
    Therefore, we force make required_data_columns = [] in mdl jobs.
    '''
    if is_column_io():
        required_data_columns = []
    proj_tbl_part_set = set()
    for path in paths:
        rets = _odps_table_path_parse(path)
        for r in rets:
            proj_tbl_part_set.add((r[0], r[1], r[2]))
    if is_column_io():
        odps_partition_mode = os.getenv(ODPS_PARTITION_MODE_KEY_NAME, None)
    else:
        app_config, _ = get_app_config()
        odps_partition_mode = app_config.get(ODPS_PARTITION_MODE_KEY_NAME, None)
    if IS_ODPS_ENDPOINT_REACHABLE and PartitionMode.use_partition_handle(odps_partition_mode):
        table_partition_struct_list = PartitionMode.gen_partition_struct(proj_tbl_part_set, odps_partition_mode, ODPS_ENDPOINT)
        for table_partition_struct in table_partition_struct_list:
            session_struct_set.add(
                HashableSessionStruct(table_partition_struct.project, table_partition_struct.table,
                                      table_partition_struct.logical_partition, table_partition_struct.get_physical_partitions(),
                                      required_data_columns) )
    else:
        session_struct_set.update(
            HashableSessionStruct(ret[0], ret[1], ret[2], ret[2] if isinstance(ret[2], list) else [ret[2]],
                                  required_data_columns) for ret in proj_tbl_part_set)

    #  Because session with NORMAL status may encounter project/table/partition deletion,
    #  we must check the presence of each partition at the very beginning.
    dump_debug_access_info_batch(session_struct_set)
    check_auth_and_data(session_struct_set)  # check or cache
    # check or cache
    create_session_struct_set = get_and_register_read_session_list(session_struct_set,
                                kSep, mode, default_project, connect_timeout, rw_timeout)

    if not is_session_creator():
        try_start_refresh_halo_metrics()
        varlogger.info("init odps openstorage session finish for size of path:{}, session_set:{}, create_set:{}".format(
            len(paths), len(session_struct_set), len(create_session_struct_set)))
        return

    if len(create_session_struct_set) > 0:
        create_and_post_read_session_list(create_session_struct_set,
                                kSep, mode, default_project, connect_timeout, rw_timeout)
    if not is_notebook():
        try_start_refresh_session_thread()
    try_start_refresh_halo_metrics()
    varlogger.info("init odps openstorage session success for size of path:{}, session_set:{}, create_set:{}".format(
            len(paths), len(session_struct_set), len(create_session_struct_set)))
