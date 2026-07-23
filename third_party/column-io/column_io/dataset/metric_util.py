# -*- coding: utf-8 -*-
import os,sys,time,datetime
import threading
import subprocess
from column_io.dataset.job_info import get_work_id,get_job_info,is_notebook

# ========== 确保在第三方模块`kmonitor` import 前设好这些环境变量 ==========
def _ensure_hippo_env():
    if "HIPPO_APP" not in os.environ:
        os.environ["HIPPO_APP"] = os.getenv("APP_ID", "null")
    if "HIPPO_SERVICE_NAME" not in os.environ:
        os.environ["HIPPO_SERVICE_NAME"] = os.getenv(
            "CALCULATE_CLUSTER", "null"
        ).upper()
    if "HIPPO_ROLE" not in os.environ:
        os.environ["HIPPO_ROLE"] = os.getenv("TASK_NAME", "worker")
        print(
            "[INFO] HIPPO_ROLE not set, use TASK_NAME:{} instead".format(
                os.environ["HIPPO_ROLE"]
            )
        )
    if "HIPPO_SLAVE_IP" in os.environ:
        return
    def _get_k8s_node_ip_internal():
        # 1. if env "RequestedIP" exist and is ipv6, then use /etc/hostinfo-ipv6
        if ":" in os.getenv("RequestedIP", "") and os.path.exists("/etc/hostinfo-ipv6"):
            # read file content of /etc/hostinfo-ipv6,
            with open("/etc/hostinfo-ipv6", "r") as f:
                # file content is like: "alicloud-alpha-bj-a\n1.2.3.4"
                for line in f:
                    if line.count(":") > 0:
                        return str(line.strip())  # line is ipv6
            return "localhost"
        # 2. if env "RequestedIP" not exist or not ipv6, then use /etc/hostinfo
        if os.path.exists("/etc/hostinfo"):
            with open("/etc/hostinfo", "r") as f:
                for line in f:
                    if line.count(".") == 3 and \
                        all(0 <= int(num) < 256 for num in line.rstrip().split(".") ):
                        return str(line.strip())  # line is ipv4
            return "localhost"
        # 3. use default env "KUBERNETES_NODE_IP" or "localhost"
        return os.getenv("KUBERNETES_NODE_IP", "localhost")
    os.environ["HIPPO_SLAVE_IP"] = _get_k8s_node_ip_internal()

# ensure HIPPO_* env for `kmonitor` can recognize the container
_ensure_hippo_env()

class MetricStatus():
    # expected status
    SUCCESS = 0 # ok
    WAITING = 1001 # waiting status

    # cross-processes  errors
    REQUEST_ERROR = 2001 # http, rpc request error
    CALL_ERROR = 2002 # ipc, internal-func call error
    LOCAL_CACHE_ERROR = 2003 # local cache error
    LOCAL_CACHE_FILE_NOT_EXISTS_ERROR = 2004 # local cache file not exists

    # logic error in process
    JSON_ERROR = 3001 #
    CODEC_ERROR = 3002 # e.g. utf-8 error
    FIELD_ERROR = 3003 # field missing, type-wrong or other relative error
    OUTDATE_ERROR = 3011  # context(session, token,,,) is expired

    # not to use old session
    PATITION_MODIDIFIED = 4001
    FORCE_RECREATE_SESSION = 4002


    # un classified errors
    UNKNOWN_ERROR = 9999

    pass

# This Exception aims to replace tf.errors or absl error so as to re-used in columnio
# Strictly speaking, this should be an independent Python module or file. 
# Also, check if tf/abseil errors provides other useful actions()
# However, in order to quickly launch, it is temporarily placed here :)
class NebulaIOFatalError(Exception):
    def __init__(self, msg):        
        self.msg = msg
        # metric report
        self._report_metric()
    def _report_metric(self):
        global metric_factory
        metric_client = metric_factory.get("openstorage_session_init")
        metric_client.try_start()
        metric_tag_map = {
            "code": "1",
            "status": "fail",
        }
        metric_client.report(MetricPoints.init_qps, 1, metric_tag_map)
        time.sleep(1.5)
        # metric_client.try_close()
    def __str__(self):
        # return "NebulaIOFatalError: {}".format(self.msg)
        return self.msg
    
class VirtualMonitor():
    # Why need me here? to silent thrift exception, when need not metric(e.g. notebook)
    def __init__(self, *args, **kwargs):
        self.is_alive_ = True
    def is_alive(self):
        return self.is_alive_
    def close(self):
        self.is_alive_ = False
    def run(self):
        pass
    def report(self, metric_name, immutable_metrics_tags, value):
        pass
    def report_metric(self, metric_name, value, extra_tags):
        pass
    def register(self, name, metric_type, priority, statistics_type=None):
        pass
    def register_metric(self, type, name, tags):
        pass

class VirtualKmonitor():
    class MetricTypes:
        GAUGE_METRIC = None
        ACC_METRIC = None
        COUNTER_METRIC = None
    class KMonitor(VirtualMonitor):
        pass

try:
    from kmonitor import kmonitor
except ImportError as e:
    print("[WARN] import kmonitor error: {}. A mock monitor will be used".format(e), file=sys.stderr)
    class kmonitor(VirtualKmonitor):
        pass

class MetricPoints():
    class Point():
        def __init__(self, name, metric_type=kmonitor.MetricTypes.GAUGE_METRIC, priority=None, statistics_type=None):
            self.name = name # type: str # point name 
            self.metric_type = metric_type # type: kmonitor.MetricTypes # the point static type 

    # TODO: v1/v2 prefix in future if need
    # GAUGE
    # E.g. xdl.metric.openstorage_session_refresh.latency_ipc_ms/xdl.metric.openstorage_session_cache.cache_qps
    latency_get_ms = Point("latency_get_ms", metric_type=kmonitor.MetricTypes.GAUGE_METRIC, statistics_type=None)
    latency_post_ms = Point("latency_post_ms", metric_type=kmonitor.MetricTypes.GAUGE_METRIC, statistics_type=None)
    latency_ipc_ms = Point("latency_ipc_ms", metric_type=kmonitor.MetricTypes.GAUGE_METRIC, statistics_type=None)
    
    cache_qps = Point("cache_qps", metric_type=kmonitor.MetricTypes.COUNTER_METRIC, statistics_type=None)
    create_qps = Point("create_qps", metric_type=kmonitor.MetricTypes.COUNTER_METRIC, statistics_type=None)
    refresh_qps = Point("refresh_qps", metric_type=kmonitor.MetricTypes.COUNTER_METRIC, statistics_type=None)

    init_qps = Point("init_qps", metric_type=kmonitor.MetricTypes.COUNTER_METRIC, statistics_type=None)

    # COUNTER
    read_byte = Point("read_byte", metric_type=kmonitor.MetricTypes.ACC_METRIC, statistics_type=None)

    # QPS
    pass

    @staticmethod
    def _get_points():
        # type: () -> list[Point] # return all available points
        return [
            MetricPoints.latency_get_ms, # for cache get
            MetricPoints.latency_post_ms, # for post new session cache
            MetricPoints.latency_ipc_ms, # for refresh call with ipc halo-worker
            MetricPoints.cache_qps,
            MetricPoints.create_qps,
            MetricPoints.refresh_qps,
            MetricPoints.init_qps,
            MetricPoints.read_byte,
        ]
  
    @staticmethod
    def regist_all_metric_points(_kmonitor, prefix=""):
        # type: (kmonitor.KMonitor, str) -> None # type: ignore
        for p in MetricPoints._get_points():
            real_name = "{}.{}".format(prefix, p.name)
            _kmonitor.register_metric(p.metric_type, real_name, {})
            # logger.debug("successfully regist metric {}".format(name))


class KMonitorClient():
    def __init__(self, namespace, module, tenant, global_tag_map):
        # self._metric_namespace = namespace
        self._status_mutex = threading.Lock()
        self.namespace_prefix = "{}.{}".format(namespace, module)
        self._global_tag_map = global_tag_map #type: dict
        # self._tenant = tenant

        # host_ip = global_tag_map["host_ip"]
        # host_port = global_tag_map["host_port"]
        # host_ip = "[{}]".format(host_ip) if ":" in host_ip else host_ip
        # sink_address = "{}:{}".format(host_ip, host_port)
        if "host_ip" in self._global_tag_map:
            del self._global_tag_map["host_ip"]
        if "host_port" in self._global_tag_map:
            del self._global_tag_map["host_port"]

        # print("[{}] [INFO] start init pykmonitor.KMonitorFactory, service_name:{}, sink_address:{}, global_map count:{}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"), namespace, sink_address, len(global_tag_map)), file=sys.stderr)
        self._kmonitor = None # type: kmonitor.KMonitor
        if is_notebook():
            self._kmonitor = VirtualMonitor()
        else:
            self._kmonitor = kmonitor.KMonitor(self._global_tag_map)
            MetricPoints.regist_all_metric_points(self._kmonitor, self.namespace_prefix)


    def try_start(self):
        pass

    def running(self):
        return True

    def try_close(self):
        time.sleep(1.5) # wait for metric agent report
        return

    """
    Args:
        metric_name: metric point name, must regist first, full name is {namespace}.{module}.{metric_name}, DO NOT use char except a-z A-Z 0-9 . - _
        metric_value: metric point value, must be numeric.
        tag_map: dict format {tag_name: tag_value}
    """
    def report(self, point, metric_value, tag_map={}):
        # type: (MetricPoints.Point, float, dict) -> None
        real_name = "{}.{}".format(self.namespace_prefix, point.name)
        # print("[{}] [INFO] report kmonitor metric name:{}, value:{}, tag_map:{}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"), real_name, metric_value, tag_map))
        self._kmonitor.report_metric(metric_name=real_name, value=metric_value, extra_tags=tag_map, )


_k8s_node_ip = None
def get_k8s_node_ip():
    # type: () -> str
    global _k8s_node_ip
    if _k8s_node_ip:
        return _k8s_node_ip
    _k8s_node_ip = os.getenv("HIPPO_SLAVE_IP", "localhost")
    # KMONITOR_SINK_ADDRESS tell c++ to get new style NODE_IP. C++ no need to imple again(also more complex)
    os.environ["KMONITOR_SINK_ADDRESS"] = _k8s_node_ip
    return _k8s_node_ip


class KMonitorClientFactory():
    job_info = get_job_info()
    metric_kwargs = {
        "namespace": "xdl.metrics",
        # "module": "openstorage",
        "tenant": "default",
        "global_tag_map": {
            "io_type": __name__.split('.')[0], # e.g. paiio/column/common => paiio,
            "host_ip": get_k8s_node_ip(),
            "host_port": str(4141),
            "task_id": str(job_info._task_id),
            "app_id": str(job_info._app_id),
            "user_id": str(job_info._user_id),
            "task_name": str(job_info._task_name),
            "rank": str(job_info._rank),
            "docker_image": str(job_info._docker_image).split(":")[-1],
            "calculate_cluster": str(os.environ.get("CALCULATE_CLUSTER", "null")),
            "nebula_project": str(os.environ.get("NEBULA_PROJECT", "null")),
            "scheduler_queue": str(os.environ.get("SCHEDULER_QUEUE", "null")),
            "openstorage_backend": str(os.environ.get("OPEN_STORAGE_BACKEND", "null")),
            "tunnel_endpoint": str(os.environ.get("OPEN_STORAGE_TUNNEL_ENDPOINT", "null")),
            "sigma_app_site": str(os.environ.get("SIGMA_APP_SITE", "null")),
            # "ip": pod_ip, # auto added by kmonitor sdk
        },
    }
    def __init__(self):
        self._client_map = {} # type: dict[str, KMonitorClient] # {module: KMonitorClient}
        self._mutex = threading.Lock()
        self._launch_collector_daemon()

    @staticmethod
    def _launch_collector_daemon():
        try:
            import recis # recis framework provide collector daemon already
            return
        except Exception:
            pass
        collector_dir = os.path.dirname(os.path.abspath(__file__))
        collector_path = os.path.join(collector_dir, "collector.py")
        _ = subprocess.Popen(
            f"python {collector_path}",
            shell=True,
            stdout=subprocess.DEVNULL,  # drop stdout
            stderr=subprocess.DEVNULL,  # drop stderr
            stdin=subprocess.DEVNULL,  # drop stdin
            start_new_session=True,  # detach from self process
        )
        # logger.debug(f"launch collector daemon pid: {proc.pid}")

    def get(self, module):
        # type : (str) -> KMonitorClient
        if module in self._client_map:
            return self._client_map[module]
        with self._mutex:
            if module in self._client_map:
                return self._client_map[module]
            kwargs = KMonitorClientFactory.metric_kwargs
            self._client_map[module] = KMonitorClient(
                namespace=kwargs["namespace"],
                module=module,
                tenant=kwargs["tenant"],
                global_tag_map=kwargs["global_tag_map"],
            )
            return self._client_map[module]

metric_factory = KMonitorClientFactory()

def test_metric_point():
    metric_client = metric_factory.get("openstorage_session_cache")
    metric_client.try_start()

    appid = os.environ.get("APP_ID", "xdl-{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M")) )
    print("\t\t metric_client submit for {} begin...".format(appid))

    metric_client.report(MetricPoints.cache_qps, 1, { "app_id": appid } )
    time.sleep(1.5)
    print("[{}] [INFO] metric_client closing...".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")))
    metric_client.try_close()
    print("[{}] [INFO] metric_client close done".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")))

def test_metric(times = 5):
    for i in range(times):
        test_metric_point()
        time.sleep(1)

# Test me if need. I work fine
if __name__ == "__main__":
    test_metric()
    time.sleep(11)
