#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ruff: noqa: G001,UP032,UP009,UP015,RUF013
import asyncio
import logging
import os
import random
import signal
import sys
import threading
import time
from asyncio.queues import Queue
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

import psutil


if TYPE_CHECKING:
    from _io import TextIOWrapper


flogger: logging.Logger = None


def SetupLogger():
    global flogger
    flogger = logging.getLogger(__name__)
    flogger.propagate = False
    flogger.setLevel(logging.INFO)
    logfile_path = os.path.join(
        os.environ.get("STD_LOG_DIR", "./log/"), "recis_daemon_metric.log"
    )  # os.path.dirname("/var/log/nebulaio_logs/")
    if not os.path.exists(os.path.dirname(logfile_path)):
        try:
            os.makedirs(os.path.dirname(logfile_path))
        except OSError:
            pass
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(process)d#%(thread)d] [%(filename)s:%(lineno)d] %(message)s"
    )
    log_handler: logging.FileHandler = RotatingFileHandler(
        filename=logfile_path, maxBytes=100 * 1024 * 1024, backupCount=3
    )
    log_handler.setFormatter(formatter)
    flogger.addHandler(log_handler)
    # std_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    # std_handler.setFormatter(formatter)
    # flogger.addHandler(std_handler)
    flogger.debug("flogger:{} init finish".format(flogger.name))


class Event:
    class PointType:
        kGauge: int = 100
        kGaugeSticky: int = 110  # NOTE: sticky only accumulate in sdk, not in collector
        kCounter: int = 200
        kSummary: int = 300

    def __init__(
        self,
        timestamp: int,
        name: str,
        tags: dict[str, str],
        type: int,
        content: str,
        extra_tags: dict[str, str] = None,
    ):
        self.timestamp: int = timestamp  # in milliseconds
        self.name: str = name
        self.tags: dict[str, str] = tags
        self.type: int = type
        self.content: str = content  # E.g. {1,7,2} -> "val=2,num=3,max=7,min=1,sum=10"
        self.extra_tags: dict[str, str] = (
            extra_tags if extra_tags is not None else {}
        )  # one-time report tags

    def __repr__(self):
        attrs = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({attrs})"

    def lag_ms(self) -> int:
        now_ms = round(time.time() * 1000)
        return now_ms - self.timestamp


class EventSubmitter:
    @staticmethod
    def GetSubmitter(is_internal: bool) -> "EventSubmitter":
        if is_internal:
            return InternalEventSubmitter()
        else:
            return OpensourceEventSubmitter()

    def __init__(self):
        pass

    def _generate_default_tags(self):
        default_tags = {
            "app": "recis",
            # e.g. hub.alibaba.com/repo/recis_1:cuda12.4->"recis_1:cuda12.4"
            "docker_image": os.getenv("DOCKER_IMAGE", "null").split("/")[-1],
        }
        return default_tags

    def submit(self, event_list: list[Event]):
        pass


class InternalEventSubmitter(EventSubmitter):
    def __init__(self):
        super().__init__()
        InternalEventSubmitter._kmonitor_env_helper()
        self.default_tags = self._generate_default_tags()
        from kmonitor.kmonitor import KMonitor

        self.kmonitor = KMonitor(default_tags=self.default_tags)
        # registered_metrics: set of (name,aggre,tags_hash) E.g. ("metric_a", "gauge", hash{"tag1": "val1"})
        #   - used as 'regist exactly once' for kmonitor to avoid un(duplicate)regist exception
        # self.registered_metrics: set[str, int] = set()
        self.registered_metrics: set[str] = set()

    @staticmethod
    def _kmonitor_env_helper():
        if "HIPPO_APP" not in os.environ:
            os.environ["HIPPO_APP"] = os.getenv("APP_ID", "null")
        if "HIPPO_SERVICE_NAME" not in os.environ:
            os.environ["HIPPO_SERVICE_NAME"] = os.getenv(
                "CALCULATE_CLUSTER", "null"
            ).upper()
        if "HIPPO_ROLE" not in os.environ:
            os.environ["HIPPO_ROLE"] = os.getenv("TASK_NAME", "worker")
            flogger.info(
                "HIPPO_ROLE not set, use TASK_NAME:{} instead".format(
                    os.environ["HIPPO_ROLE"]
                )
            )
        if "HIPPO_SLAVE_IP" not in os.environ:

            def get_node_ip_internal():
                # 1. if env "RequestedIP" exist and is ipv6, then use /etc/hostinfo-ipv6
                if ":" in os.getenv("RequestedIP", "") and os.path.exists(
                    "/etc/hostinfo-ipv6"
                ):
                    # read file content of /etc/hostinfo-ipv6,
                    with open(
                        "/etc/hostinfo-ipv6", "r"
                    ) as f:  # file content is like: "alicloud-alpha-bj-a\n1.2.3.4"
                        for line in f:
                            if line.count(":") > 0:
                                return str(line.strip())  # line is ipv6
                    return "localhost"
                # 2. if env "RequestedIP" not exist or not ipv6, then use /etc/hostinfo
                if os.path.exists("/etc/hostinfo"):
                    with open("/etc/hostinfo", "r") as f:
                        for line in f:
                            if line.count(".") == 3 and all(
                                0 <= int(num) < 256 for num in line.rstrip().split(".")
                            ):
                                return str(line.strip())  # line is ipv4
                    return "localhost"
                # 3. use default env "KUBERNETES_NODE_IP" or "localhost"
                return os.getenv("KUBERNETES_NODE_IP", "localhost")

            os.environ["HIPPO_SLAVE_IP"] = get_node_ip_internal()

    def _generate_default_tags(self):
        default_tags = super()._generate_default_tags()
        default_tags["app_id"] = os.getenv("APP_ID", "null")
        default_tags["user_id"] = os.getenv("_NEBULA_USER_ID", "null")
        default_tags["task_id"] = os.getenv("TASK_ID", "null")
        default_tags["task_name"] = os.getenv("TASK_NAME", "worker")
        default_tags["scheduler_queue"] = os.getenv("SCHEDULER_QUEUE", "null")
        default_tags["calculate_cluster"] = os.getenv("CALCULATE_CLUSTER", "null")
        default_tags["sigma_app_site"] = os.getenv("SIGMA_APP_SITE", "null")
        return default_tags

    def _cast_to_ktype(self, type: Event.PointType) -> str:
        from kmonitor.kmonitor import MetricTypes

        if type == Event.PointType.kGauge:
            return MetricTypes.GAUGE_METRIC
        elif type == Event.PointType.kGaugeSticky:
            return MetricTypes.GAUGE_METRIC
        elif type == Event.PointType.kCounter:
            return MetricTypes.COUNTER_METRIC
        elif type == Event.PointType.kSummary:
            return MetricTypes.GAUGE_METRIC
        else:
            # sdk has no other types, so use gauge as a substitute
            return MetricTypes.GAUGE_METRIC

    def submit(self, event_list: list[Event]):
        for event in event_list:
            if event.lag_ms() > 60 * 1000:
                flogger.warning(
                    f"event {event.name} lag_ms:{event.lag_ms()} > 60 * 1000, skip"
                )
                continue
            self._submit_event(event)

    def _submit_event(self, event: Event):
        metric_name = event.name
        metric_values = event.content
        # E.g. "val=2.1,num=3,max=7.999,min=1.0,sum=10.0999999"
        aggre_value_tup = [
            blob.strip() for blob in metric_values.split(",") if blob != ""
        ]
        for aggre_value in aggre_value_tup:
            aggr, val = aggre_value.split("=")
            if aggr not in {
                "val",
                "sum",
                "max",
                "min",
                "p10",
                "p50",
                "p90",
                "p99",
                "avg",
            }:
                continue
            try:
                val_float = float(val)
            except ValueError:
                continue
            # f"{metric_name}.{aggr}" if aggr != "val" else  # 对周期性聚合指标加入后缀
            real_name = metric_name if aggr == "val" else f"{metric_name}.{aggr}"
            real_value = val_float
            # real_tags = hash(frozenset(event.tags.items()))
            # if (real_name, real_tags) not in self.registered_metrics:
            #     self.registered_metrics.add((real_name, real_tags))
            #     self.kmonitor.register_metric(self._cast_to_ktype(event.type), real_name, event.tags)
            if real_name not in self.registered_metrics:
                self.registered_metrics.add(real_name)
                self.kmonitor.register_metric(
                    self._cast_to_ktype(event.type), real_name, event.tags
                )
            real_tags = event.tags.copy()
            real_tags.update(event.extra_tags)
            self.kmonitor.report_metric(real_name, real_value, real_tags)
            flogger.debug(
                "submit event name: {}, value: {}, tags: {}+{}.".format(
                    real_name, real_value, event.tags, event.extra_tags
                )
            )
        # finish one event


class OpensourceEventSubmitter(EventSubmitter):
    # TODO: implement InfluxdbEventSubmitter, PrometheusEventSubmitter
    def __init__(self):
        super().__init__()

    def submit(self, event_list: list[Event]):
        pass


class GPUInfo:
    _nvidia_smi_exist: bool = True

    # TODO: support rocm-smi. alixpu-smi
    def detect_supported_fields() -> list[str]:
        import subprocess

        try:
            # according --help-query-gpu to get all available fields
            # TODO: support `rocm-smi` command for future
            out = subprocess.check_output(
                ["nvidia-smi", "--help-query-gpu"],
                encoding="utf-8",
                stderr=subprocess.STDOUT,
            )
            return [
                line.strip().split()[0] for line in out.splitlines() if line.strip()
            ]
        except Exception:
            return []

    SUPPORTED_FIELDS = detect_supported_fields()

    def __init__(
        self,
        id: int,
        name: str,
        gpu_util: float,
        gpu_sm_util: float,
        mem_band_util: float,
        mem_total_MB: int,
        mem_used_MB: int,
        temperature: float,
        power_draw: float,
    ):
        self.id = str(id)
        self.gpu_util = f"{gpu_util:.4f}"
        self.gpu_sm_util = f"{gpu_sm_util:.4f}"
        self.mem_total_MB = str(mem_total_MB)
        self.mem_used_MB = str(mem_used_MB)
        self.mem_band_util = f"{mem_band_util:.4f}"
        self.temperature = f"{temperature:.4f}"
        self.power_draw = f"{power_draw:.4f}"

    @staticmethod
    def QueryGPUInfo() -> list["GPUInfo"]:
        if not GPUInfo._nvidia_smi_exist:
            return []
        import subprocess

        """调用 nvidia-smi 获取多卡信息，驱动不支持的字段会自动回退"""
        fields = [
            "index",
            "name",
            "utilization.gpu",
            # 如果驱动支持 SM，则用SM，否则后面代码用gpu_util近似
            "utilization.sm"
            if "utilization.sm" in GPUInfo.SUPPORTED_FIELDS
            else "utilization.gpu",
            "utilization.memory",
            "memory.total",
            "memory.used",
            "temperature.gpu",
            "power.draw",
        ]
        cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
        output = ""
        try:
            output = subprocess.check_output(
                cmd, encoding="utf-8", stderr=subprocess.STDOUT, timeout=5.0
            )
        except subprocess.CalledProcessError as e:
            flogger.warning("nvidia-smi exec fail: {}".format(e.output))
            return []
        except subprocess.TimeoutExpired:
            flogger.warning("nvidia-smi exec timeout")
            return []
        except TimeoutError:
            flogger.warning("nvidia-smi exec timeout")
            return []
        except Exception as e:
            flogger.warning("nvidia-smi exec error: {}".format(e))
            return []

        def static_cast(t: any, value: str) -> any:
            try:
                return t(value)
            except (ValueError, TypeError):
                return t()
            except Exception:
                return 0.0

        gpus_info_list = []
        for line in output.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                flogger.warning(
                    "nvidia-smi output has fewer fields than expected: {}".format(
                        line[:256]
                    )
                )
                continue
            gpu_info = GPUInfo(
                id=static_cast(int, parts[0]),
                name=parts[1],
                gpu_util=static_cast(float, parts[2]) / 100,
                gpu_sm_util=static_cast(float, parts[3])
                / 100,  # 如果无SM字段，这里就是GPU Util
                mem_band_util=static_cast(float, parts[4]) / 100,
                mem_total_MB=static_cast(int, parts[5]),
                mem_used_MB=static_cast(int, parts[6]),
                temperature=static_cast(float, parts[7]),
                power_draw=static_cast(float, parts[8]),
            )
            gpus_info_list.append(gpu_info)
        return gpus_info_list

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({attrs})"


class HardwareInfo:
    _info: "HardwareInfo" = None
    _sample_thread: threading.Thread = None
    _lock = threading.Lock()
    _initialized = False

    def __init__(
        self,
        cpu_percent: float,
        mem_percent: float,
        mem_total_byte: float,
        mem_used_byte: float,
        gpus_info: list[GPUInfo],
        disk_total_byte: float,
        disk_used_byte: float,
        disk_percent: float,
        net_tx_byte: float,
        net_rx_byte: float,
    ):
        self.cpu_percent: str = f"{cpu_percent:.4f}"
        self.mem_percent: str = f"{mem_percent:.4f}"
        self.mem_total_byte: str = f"{mem_total_byte:.0f}"
        self.mem_used_byte: str = f"{mem_used_byte:.0f}"
        self._gpus_info: list[GPUInfo] = gpus_info
        # self.disk_total_byte: str = f"{disk_total_byte:.0f}"
        # self.disk_used_byte: str = f"{disk_used_byte:.0f}"
        # self.disk_percent: str = f"{disk_percent:.4f}"
        self.net_tx_byte: str = f"{net_tx_byte:.0f}"
        self.net_rx_byte: str = f"{net_rx_byte:.0f}"

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({attrs})"

    @staticmethod
    def get() -> "HardwareInfo":
        if not HardwareInfo._initialized:
            with HardwareInfo._lock:
                if not HardwareInfo._initialized:
                    HardwareInfo._sample_thread = threading.Thread(
                        target=HardwareInfo._async_sample, args=(7.0,), daemon=True
                    )
                    HardwareInfo._sample_thread.start()
                    HardwareInfo._initialized = True
        while HardwareInfo._info is None:
            time.sleep(0.5)
        return HardwareInfo._info

    @staticmethod
    def _async_sample(interval_ms: int):
        workdir = psutil.Process().cwd()
        interval_sec = interval_ms / 1000.0

        last_net_io = None
        while True:
            cpu_percent = psutil.cpu_percent(interval=None)
            disk_usage = psutil.disk_usage(workdir)
            mem = psutil.virtual_memory()
            net_io = psutil.net_io_counters()
            # 计算网络速率
            if last_net_io is not None:
                net_bytes_sent_rate = (
                    net_io.bytes_sent - last_net_io.bytes_sent
                ) / interval_sec
                net_bytes_recv_rate = (
                    net_io.bytes_recv - last_net_io.bytes_recv
                ) / interval_sec
            else:
                net_bytes_sent_rate = 0.0
                net_bytes_recv_rate = 0.0
            last_net_io = net_io
            gpus_info = GPUInfo.QueryGPUInfo()
            info = HardwareInfo(
                cpu_percent=cpu_percent / 100.0,
                mem_percent=mem.percent / 100.0,
                mem_total_byte=mem.total,
                mem_used_byte=mem.used,
                gpus_info=gpus_info,
                disk_total_byte=disk_usage.total,
                disk_used_byte=disk_usage.used,
                disk_percent=disk_usage.percent,
                net_tx_byte=net_bytes_sent_rate,
                net_rx_byte=net_bytes_recv_rate,
            )
            with HardwareInfo._lock:
                HardwareInfo._info = info
            time.sleep(interval_sec)


def _generate_pod_events() -> list[Event]:
    event_list: list[Event] = []
    hardware_info = HardwareInfo.get()
    # iter basic hardware_info fields
    for field in hardware_info.__dict__:
        if field.startswith("_"):
            continue
        name = "recis.hardware.{}".format(field)
        tags = {}  # {"pod": "k8s_" + os.getenv("APP_ID", "null")}
        value = getattr(hardware_info, field)
        event = Event(
            timestamp=int(time.time() * 1000),
            name=name,
            tags=tags,
            type=Event.PointType.kGauge,
            content="val={}".format(value),
        )
        event_list.append(event)
    # iter hardware_info.gpu fields with id=0,1,2,...
    for gpu_info in hardware_info._gpus_info:
        for field in gpu_info.__dict__:
            if field.startswith("_") or field == "id":
                continue
            name = "recis.hardware.gpu.{}".format(field)
            tags = {"gpu_id": "0"}
            extra_tags = {
                "gpu_id": str(gpu_info.id)
            }  # {"pod": "k8s_" + os.getenv("APP_ID", "null"), "gpu_id": str(gpu_info.id)}
            value = getattr(gpu_info, field)
            event = Event(
                timestamp=int(time.time() * 1000),
                name=name,
                tags=tags,
                type=Event.PointType.kGauge,
                content="val={}".format(value),
                extra_tags=extra_tags,
            )
            event_list.append(event)
    return event_list


class EventActor:
    # 输出时被过滤的key. 不允许上报
    ExcludeKeys: dict[str, bool] = {"__Pid": True, "__FactInst": True, "__TP": True}
    # 输出时被替换为别名的ke
    OnameKeys: dict[str, str] = {
        "__Framework": "framework",
        "__Rank": "rank",
        "__WorldSize": "world_size",
        "__TaskName": "task_name",
    }

    @staticmethod
    def _parse_line_group(
        line: str, parser_global_tags: dict[str, str]
    ) -> dict[str, str]:
        tuples: list[str] = line[1:].split()
        tuples = [blob.strip() for blob in tuples if blob != ""]
        if len(tuples) < 1:
            return parser_global_tags
        if tuples[0] == "BeginGroup" and len(tuples) >= 2:
            parser_global_tags = {}
            tags = tuples[1].split(",")
            # group_name = tags[0]
            tags = tags[1:]
            for tag in tags:
                k, v = tag.split("=")
                if k in EventActor.ExcludeKeys:
                    continue
                if k in EventActor.OnameKeys:
                    k = EventActor.OnameKeys[k]
                parser_global_tags[k] = v
        elif tuples[0] == "EndGroup":
            pass
        else:
            pass
        return parser_global_tags

    @staticmethod
    def _parse_line_event(line: str, parser_global_tags: dict[str, str]) -> Event:
        # parse "name,tags   k1=v1,k2=v2    timestamp" . __TP is in tags key as Event.PointType
        tuples = line.split()
        if len(tuples) != 3:
            flogger.warning("parse error for: {}".format(line[:256]))
            return None
        timestamp_ms = int(tuples[2])
        name_tags = [blob.strip() for blob in tuples[0].split(",") if blob != ""]
        if len(name_tags) < 1:
            flogger.warning("parse error for: {}".format(line[:256]))
            return None
        name: str = name_tags[0]
        point_type: int = Event.PointType.kGauge
        tags = parser_global_tags.copy()
        extra_tags = {}
        for tag in name_tags[1:]:
            k, v = tag.split("=")
            if k == "__TP":
                if v == "count":
                    point_type = Event.PointType.kCounter
                elif v == "sumry":
                    point_type = Event.PointType.kSummary
                else:
                    point_type = Event.PointType.kGauge
                continue
            if k in EventActor.ExcludeKeys:
                continue
            if k in EventActor.OnameKeys:
                k = EventActor.OnameKeys[k]
                tags[k] = v
            else:
                extra_tags[k] = v
        event = Event(
            timestamp=timestamp_ms,
            name=name,
            tags=tags,
            type=point_type,
            content=tuples[1],
            extra_tags=extra_tags,
        )
        return event

    @staticmethod
    async def EventLogParser(
        filename: str, output_queue: asyncio.Queue[Event], interval: int = 2
    ):
        r"""
        async tail -F filename
        Args:
            filename (str):   the filename to tail
            output_queue (asyncio.Queue[Event]):  the output queue to store metric event
            interval (int):   the interval to check EOF
        """
        flogger.info(f"start EventLogParser for {filename}")
        current_inode: int = None
        file_: TextIOWrapper = None
        partial_line = ""

        def open_file():
            nonlocal file_, current_inode, partial_line
            file_ = open(filename, "r")
            current_inode = os.fstat(file_.fileno()).st_ino
            file_.seek(0, os.SEEK_SET)
            partial_line = ""
            # file_.seek(0, os.SEEK_END) # 从末尾开始, 则行为视为自动丢弃非最新数据

        # 等待文件出现
        while True:
            try:
                await asyncio.to_thread(open_file)
                break
            except FileNotFoundError:
                if int(time.time()) % 60 <= interval:
                    flogger.info(
                        f"open filename EOF: {filename} , retry in {interval}s"
                    )
                await asyncio.sleep(interval)

        parser_global_tags: dict[str, str] = {}
        while True:
            # 读取当前周期全部新行
            while True:
                line = await asyncio.to_thread(file_.readline)
                if not line:
                    flogger.debug(
                        f"read filename EOF: {filename} , retry in {interval}s"
                    )
                    break
                if not line.endswith("\n"):
                    partial_line += line
                    continue
                if partial_line != "":
                    line = partial_line + line
                    partial_line = ""
                line = str(line).strip()
                if len(line) <= 1:  # empty line
                    continue
                if line.startswith("#"):
                    parser_global_tags = EventActor._parse_line_group(
                        line, parser_global_tags
                    )
                    continue
                else:  # real data
                    event = EventActor._parse_line_event(line, parser_global_tags)
                    if event is None:
                        continue
                    if event.lag_ms() > 10 * 60 * 1000:
                        flogger.debug(
                            f"event {event.name} lag_ms:{event.lag_ms()} too old, skip"
                        )
                        continue
                    await output_queue.put(event)

            await asyncio.sleep(interval)
            try:  # Check file rotate
                st = os.stat(filename)
                if st.st_ino != current_inode:
                    flogger.info(
                        f"detect filename: {filename} rotated, reopen new file"
                    )
                    file_.close()
                    await asyncio.sleep(0.5)  # sleep shortly for new file to be ready
                    await asyncio.to_thread(open_file)
            except FileNotFoundError:
                flogger.warning(
                    f"rotate filename EOF: {filename} not found. retry in {interval}s"
                )

    @staticmethod
    async def EventSockParser(filename: str, output_queue: asyncio.Queue[Event]):
        r"""
        async tail -F filename
        Args:
            filename (str):   the filename to listen
            output_queue (asyncio.Queue[Event]):  the output queue to store metric event
            interval (int):   the interval to check EOF
        """
        flogger.info(f"start EventSockParser for {filename}")
        try:
            os.unlink(filename)
        except FileNotFoundError:
            pass
        except Exception as e:
            flogger.warning(f"unlink {filename} error: {e}")

        import queue
        import socketserver
        from http.server import BaseHTTPRequestHandler

        body_queue: queue.Queue[bytes] = queue.Queue(maxsize=1024)

        class myHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"Pong!\n")
                # TODO: internal status analysis

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    try:
                        body_queue.put(body, timeout=1.0)
                    except queue.Full:
                        flogger.warning("body_queue full, drop msg")
                    # set response, note that client(in monitor sdk) need't this resp
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"OK\n")
                except Exception as e:
                    flogger.error(f"POST handler error: {e}")
                    self.send_response(500)
                    self.end_headers()

        class UnixSocketHttpServer(socketserver.UnixStreamServer):
            def get_request(self):
                request, client_address = super().get_request()
                return (request, ["local", 0])

        def run_server():
            try:
                server = UnixSocketHttpServer(filename, myHandler)
                flogger.info(f"Unix socket server start on {filename}")
                server.serve_forever()
            except Exception as e:
                flogger.error(f"Socket server error: {e}")
            finally:
                flogger.info(f"Unix socket server finish on {filename}")

        def get_metric_record_logger():
            # This logger is used to dump metric record
            # since sock msg cannot directly store as log msg
            dump_logger = logging.getLogger(f"{__name__}-dump")
            dump_logger.propagate = False
            dump_logger.setLevel(logging.INFO)
            logfile_path = os.path.join(
                os.environ.get("STD_LOG_DIR", "./log/"), "recis_sock_metric.log"
            )
            log_handler: logging.FileHandler = RotatingFileHandler(
                filename=logfile_path, maxBytes=100 * 1024 * 1024, backupCount=3
            )
            log_handler.setFormatter(logging.Formatter("%(message)s"))
            dump_logger.addHandler(log_handler)
            return dump_logger

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        metric_logger = get_metric_record_logger()

        while True:
            try:
                body = await asyncio.to_thread(body_queue.get, timeout=0.5)
            except queue.Empty:
                flogger.info(f"wait filename EOF: {filename} , retry in 2s")
                await asyncio.sleep(2)
                continue
            except Exception as e:
                flogger.error(f"SockParser body_queue get error: {e}")
                continue

            try:
                resp = body.decode(errors="ignore")
                metric_logger.info(resp)
                lines = resp.strip().split("\n")
                parser_global_tags = {}
                for line in lines:
                    line = str(line).strip()
                    if len(line) <= 1:  # empty line
                        continue
                    if line.startswith("#"):
                        parser_global_tags = EventActor._parse_line_group(
                            line, parser_global_tags
                        )
                        continue
                    event = EventActor._parse_line_event(line, parser_global_tags)
                    if event is None:
                        continue
                    if event.lag_ms() > 10 * 60 * 1000:
                        flogger.debug(
                            f"event {event.name} lag_ms:{event.lag_ms()} too old, skip"
                        )
                        continue
                    await output_queue.put(event)
            except Exception:
                flogger.exception("Handle body error")
        # flogger.debug(f"finish EventSockParser for {filename}")

    @staticmethod
    def EventDispatcher(queue: asyncio.Queue[Event], interval: int = 2):
        r"""
        Event Queue Consumer. All evnets are send by this dispatcher.
        """
        try:
            from recis.info import is_internal_enabled
        except Exception:
            # When recis not exist, it MUST NOT be Opensource.
            #   So in internal_version, launch daemon is always necessary
            #   Hope we can get `is_recis` flag from env or somewhere
            def is_internal_enabled():
                return True

        submitter = EventSubmitter.GetSubmitter(is_internal=is_internal_enabled())
        last_snapshot_: float = time.time()
        last_snapshot_ = last_snapshot_ - int(last_snapshot_) % interval  # round

        while True:
            last_snapshot_ += interval
            sleep_time = last_snapshot_ - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -interval:  # 防止时间回拨或严重滞后，重置对齐
                last_snapshot_ = time.time()
                last_snapshot_ = last_snapshot_ - int(last_snapshot_) % interval
                continue
            pod_list = _generate_pod_events()
            event_list = []
            while True:
                try:
                    event = queue.get_nowait()
                    event_list.append(event)
                except asyncio.QueueEmpty:
                    break
            flogger.info(
                f"submit events {len(event_list)} + {len(pod_list)} for this period"
            )
            try:
                submitter.submit(event_list + pod_list)
            except Exception:
                flogger.exception("EventDispatcher submit with error")
        flogger.info("EventQueueConsumer exit from event loop")


class DaemonProcess:
    @staticmethod
    def FirstOrDie():
        # Single-machine processes preemptive coordination starter
        my_pid = os.getpid()
        my_cmd = " ".join(psutil.Process(my_pid).cmdline())
        my_start_time = psutil.Process(my_pid).create_time()
        stat = os.stat(__file__)
        my_version_time = max(stat.st_ctime, stat.st_mtime)
        # launch delay to allow psutil iter process fully
        time.sleep(0.5 + random.random())
        for p in psutil.process_iter(["pid", "cmdline", "create_time"]):
            this_pid = int(p.info["pid"])
            this_cmd = " ".join(p.info.get("cmdline", []))
            this_create_time = p.info.get("create_time", float("inf"))
            try:
                if my_pid == this_pid:
                    continue
                if my_cmd != this_cmd:
                    continue
                if my_start_time < this_create_time:
                    continue
                if my_version_time > this_create_time:
                    p.send_signal(signal.SIGTERM)
                    continue
                if my_pid < this_pid:
                    continue
                flogger.info("exit since not earliest process")
                sys.exit(0)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
        flogger.info("start as no same and earlier command processes found")

    running_ = True
    dispatcher: threading.Thread = None
    parsers: threading.Thread = None
    event_queue: Queue[Event] = asyncio.Queue()
    interval = 2

    @classmethod
    def StartDispatcher(cls):
        cls.dispatcher = threading.Thread(
            target=EventActor.EventDispatcher, args=(cls.event_queue, cls.interval)
        )
        cls.dispatcher.start()

    @classmethod
    def JoinDispatcher(cls):
        cls.dispatcher.join()
        cls.running_ = False

    @classmethod
    def StartParser(cls):
        async def _start():
            world_size = int(
                os.environ.get("WORLD_SIZE", "1")
            )  # TODO dynamic world_size and it's parser
            parser_tasks: list[asyncio.Task[None]] = []
            # launch every file's tail parser. Non-localhost file coroutine does not need to handler with
            for i in range(world_size):
                read_logname = os.environ.get("STD_LOG_DIR", "./log/")
                read_logname = os.path.join(read_logname, str(i), "recis_metric.log")
                task = asyncio.create_task(
                    EventActor.EventLogParser(
                        read_logname, cls.event_queue, cls.interval
                    )
                )
                parser_tasks.append(task)
            # launch a single unix domain socket parser.
            read_sockname = os.environ.get("STD_LOG_DIR", "./log/")
            read_sockname = os.path.join(read_sockname, "recis_metric.sock")
            parser_tasks.append(
                asyncio.create_task(
                    EventActor.EventSockParser(read_sockname, cls.event_queue)
                )
            )
            try:
                while cls.running_:
                    await asyncio.sleep(cls.interval)
            except asyncio.CancelledError:
                flogger.info(
                    "EventActor.EventLogParser canceled from event loop normally"
                )
            finally:
                flogger.info("EventLogParser cancelling parser_tasks...")
                for task in parser_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*parser_tasks, return_exceptions=True)
                flogger.info("EventLogParser cancelled all parser_tasks")

        cls.parsers = threading.Thread(
            target=lambda: asyncio.run(_start()), daemon=True
        )
        cls.parsers.start()
        flogger.info("Started parser_tasks thread.")

    @classmethod
    def JoinParser(cls):
        cls.parsers.join()
        flogger.info("finish parser_tasks")


if __name__ == "__main__":
    SetupLogger()
    try:
        DaemonProcess.FirstOrDie()
        DaemonProcess.StartDispatcher()
        DaemonProcess.StartParser()
        # Wait ...
        DaemonProcess.JoinDispatcher()
        DaemonProcess.JoinParser()
    except Exception:
        flogger.exception("unexpected error during daemon exec")
    sys.exit(-1)
