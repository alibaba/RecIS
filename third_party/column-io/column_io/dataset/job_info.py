# -*- coding: utf-8 -*-
# This file is part of the PAI-XDL project. Used to get run time arg&config of the full task
import os,sys,time,json,argparse,uuid

class JobConfigKey:
    UNKNOWN = "kUnknown"
    # nebula config key
    NEBULA_USER_ID = "_NEBULA_USER_ID"
    NEBULA_PROJECT = "NEBULA_PROJECT"
    SCHEDULER_QUEUE = "scheduler_queue"
    TASK_ID = "TASK_ID"
    APP_ID = "APP_ID"
    # general config key
    DOCKER_IMAGE = "docker_image"
    HALO_WORKER_DOCKER_IMAGE = "halo_worker_docker_image"
    USER_ID = "USER_ID"
    TASK_NAME = "TASK_NAME"
    TASK_INDEX = "TASK_INDEX"
    RANK = "RANK"

def get_app_config(): # TODO add cache variable
    parser = argparse.ArgumentParser(description="xdl arguments")
    parser.add_argument('--config', default=None)
    parser.add_argument('--app_id', default=None)
    args, unknown = parser.parse_known_args()
    try:
        app_config = json.load(open(args.config))
        if JobConfigKey.SCHEDULER_QUEUE not in app_config and JobConfigKey.DOCKER_IMAGE not in app_config:
            # 说明 --config 被用户的config覆盖, 通过参数检查排除错误加载
            app_config = {}
    except Exception as e:
        app_config = {}
    app_id = args.app_id if args.app_id else os.getenv(JobConfigKey.APP_ID, None)
    return app_config, app_id

def get_work_id():
    # mdl style
    worker_id = os.environ.get(JobConfigKey.RANK, None)
    if worker_id is not None:
        return int(worker_id)
    # xdl style
    parser = argparse.ArgumentParser(description="xdl arguments")
    parser.add_argument("--task_index", default=None)
    # parser.add_argument("--zk_addr", default=None) # some occasion use coreDNS, not a universal solution
    args, unknown = parser.parse_known_args()
    if args.task_index is None: # and args.zk_addr is None:
        os.environ[JobConfigKey.RANK] = str(0)
        return 0
    # assert args.task_index, "task_index can't be None in distributed mode"
    task_index = int(args.task_index)
    os.environ[JobConfigKey.RANK] = str(task_index)
    return task_index

def get_app_id(worker_id = get_work_id()):
    app_id = str(os.environ.get(JobConfigKey.APP_ID, "")) # both mdl and xdl style contains this env
    if app_id != "":
        return app_id
    parser = argparse.ArgumentParser(description="xdl arguments")
    parser.add_argument("--app_id", default=None)
    args, unknown = parser.parse_known_args()
    app_id = args.app_id if args.app_id else str(uuid.uuid4()).replace("-", "")[:12]
    app_id = (
        "local_"
        + app_id
        + "_"
        + str(worker_id)
        + "_"
        + str(uuid.uuid4()).replace("-", "")[:12]
    )
    return app_id

def get_task_name():
    task_name = os.environ.get(JobConfigKey.TASK_NAME, "")
    if task_name != "":
        return task_name
    parser = argparse.ArgumentParser(description="xdl arguments")
    parser.add_argument("--task_name", default=None)
    args, unknown = parser.parse_known_args()
    task_name = str(args.task_name) if args.task_name else "worker"
    os.environ[JobConfigKey.TASK_NAME] = task_name
    return task_name


class JobInfo(object):
    def __init__(self, user_id, task_id, task_name, app_id, nebula_project=JobConfigKey.UNKNOWN, 
        scheduler_queue=JobConfigKey.UNKNOWN, is_foreign=0, rank=get_work_id(), 
        docker_image=JobConfigKey.UNKNOWN, halo_worker_docker_image=JobConfigKey.UNKNOWN):
        self._user_id = user_id
        self._task_id = task_id
        self._task_name = task_name
        self._app_id = app_id or "unknown"
        self._nebula_project = nebula_project
        self._scheduler_queue = scheduler_queue
        self._docker_image = docker_image
        self._is_foreign = is_foreign
        self._rank = rank
        self._halo_worker_docker_image = halo_worker_docker_image

def get_job_info():
    app_config, app_id = get_app_config()
    task_name = get_task_name()
    task_id = os.getenv(JobConfigKey.TASK_ID, None)
    user_id = app_config.get(JobConfigKey.NEBULA_USER_ID, os.getenv(JobConfigKey.USER_ID.upper(), JobConfigKey.UNKNOWN))
    nebula_project = os.getenv(JobConfigKey.NEBULA_PROJECT, os.getenv(JobConfigKey.NEBULA_PROJECT.upper(), JobConfigKey.UNKNOWN))
    scheduler_queue = app_config.get(JobConfigKey.SCHEDULER_QUEUE, os.getenv(JobConfigKey.SCHEDULER_QUEUE.upper(), JobConfigKey.UNKNOWN))
    docker_image = app_config.get(JobConfigKey.DOCKER_IMAGE, os.getenv(JobConfigKey.DOCKER_IMAGE.upper(), JobConfigKey.UNKNOWN))  # FIXME: by 20251014 星云任务暂缺该env
    halo_worker_docker_image = app_config.get(JobConfigKey.HALO_WORKER_DOCKER_IMAGE, os.getenv(JobConfigKey.HALO_WORKER_DOCKER_IMAGE.upper(), JobConfigKey.UNKNOWN))  # FIXME: by 20251014 星云任务暂缺该env
    # worker_id = os.getenv(JobConfigKey.TASK_INDEX, 0) # actully it is pod INDEX, not worker-process index
    job_info = JobInfo(user_id=user_id, task_id=task_id, task_name=task_name,app_id=app_id,
                    nebula_project=nebula_project, scheduler_queue=scheduler_queue,
                    docker_image=docker_image, halo_worker_docker_image=halo_worker_docker_image)
    return job_info

def is_notebook():
    return str(os.environ.get("NOTEBOOK_CONTAINER", "0")) == "1"

def get_odps_endpoint():
    # type: () -> str
    DEFAULT = os.getenv("ODPS_ENDPOINT", "")
    value = os.getenv("ODPS_ENDPOINT") or os.getenv("odps_endpoint") or os.getenv("end_point") or DEFAULT
    if not value.startswith(("http://", "https://")):
        return DEFAULT
    return value

def get_child_order():
    # type: () -> int
    my_pid = str(os.getpid())
    parent_pid = os.getppid()
    path = os.path.join("/proc", str(parent_pid), "task", str(parent_pid), "children")
    with open(path, "r") as f:
        count = 0
        child_pid_list = []
        for line in f:
            child_pid_list += line.split()
        for child_pid in child_pid_list:
            count += 1
            if child_pid == my_pid or count > 1024: # 1024 is only for dealing dead loop. no special meaning
                return count
    return 0
