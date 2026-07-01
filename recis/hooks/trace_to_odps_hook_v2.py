import atexit
import os
import queue
import threading
import time
import traceback
from decimal import Decimal
from functools import wraps
from typing import Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pac

from recis.hooks import Hook

# 复用老模块的全局函数,用户调用 API 不变
from recis.hooks.trace_to_odps_hook import TRACE_MAP  # noqa: F401  仅引用,确保单一全局实例
from recis.hooks.trace_to_odps_hook import add_to_trace  # noqa: F401
from recis.hooks.trace_to_odps_hook import clear_trace_map, get_trace_map, rank
from recis.info import is_internal_enabled
from recis.monitor.monitor_reporter import MonitorReporter
from recis.utils.logger import Logger


if is_internal_enabled():
    from recis.utils.cluster import get_odps_access_info


if not os.environ.get("BUILD_DOCUMENT", None) == "1":
    from odps import ODPS, options as odps_options
    from odps.apis.storage_api import (
        SessionRequest,
        StorageApiArrowClient,
        TableBatchWriteRequest,
        WriteRowsRequest,
    )
    from odps.models import Schema
    from odps.tunnel.io.types import odps_schema_to_arrow_schema
    from odps.types import (
        Column as OdpsColumn,
        OdpsSchema,
        timestamp_ntz as ODPS_TIMESTAMP_NTZ,
        validate_data_type as validate_odps_type,
    )


logger = Logger(__name__)


def retry(retry_count: int, interval: float, retryable=Exception):
    """带 warning log 的重试装饰器。

    - 中间失败打 warning log(便于排查瞬时错误)
    - `raise` 不带参数,保留完整 traceback
    - 可选只对指定异常类型重试,业务错误立即抛出
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retry_count):
                try:
                    return func(*args, **kwargs)
                except retryable as e:
                    if i == retry_count - 1:
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {i + 1}/{retry_count} failed: "
                        f"{e!r}, retrying in {interval}s"
                    )
                    time.sleep(interval)

        return wrapper

    return decorator


def _to_pa_array(arr: np.ndarray) -> "pa.Array":
    """numpy ndarray -> pyarrow Array,支持任意维度。

    - 1D:基础 Array(数值 dtype 零拷贝)
    - 2D:List<numeric>
    - 3D+:递归构造嵌套 ListArray
    """
    if arr.ndim == 1:
        return pa.array(arr)
    N = arr.shape[0]
    inner_size = arr.shape[1]
    inner = _to_pa_array(arr.reshape(-1, *arr.shape[2:]))
    offsets = pa.array(np.arange(0, N * inner_size + 1, inner_size, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, inner)


_HALO_SOCKET_URL = "http+unix://%2Fhome%2Fadmin%2Fhalo%2Fsocket%2Fhalo.sock"


def _resolve_tunnel_endpoint(odps_kwargs: dict, config: dict) -> str:
    """根据环境变量决定 tunnel endpoint:halo socket 或普通 tunnel。

    选择逻辑:
        ODPS_PANGU_ACCESS_MODE == 'tunnel' → 只能走 tunnel
        否则 OPEN_STORAGE_BACKEND == 'halo-worker' → 走 halo socket
        否则 → 走 tunnel

    requests-unixsocket 作为 recis 的声明依赖随 pip install 自动安装,
    pyodps rest.py 在首次 import 时即可识别,无需运行时 patch。
    """
    odps_pangu_access_mode = os.getenv("ODPS_PANGU_ACCESS_MODE", None)
    open_storage_backend = os.getenv("OPEN_STORAGE_BACKEND", None)
    can_use_halo = (
        odps_pangu_access_mode != "tunnel"
        and open_storage_backend == "halo-worker"
    )
    if not can_use_halo:
        return odps_kwargs.get("tunnel_endpoint") or config.get("tunnel_endpoint")

    logger.info("Using halo socket as tunnel endpoint")
    return _HALO_SOCKET_URL


class TraceWriterV2(threading.Thread):
    """新版 trace writer。

    - threading.Thread,daemon=False
    - V3 数据转换(write 时立即转 pa.Array,累积 nbytes)
    - storage API 写入(替代 tunnel)
    - error_queue 主动报告异常
    - run() try/finally 保证 flush
    """

    # 自适应阈值常量
    _AUTO_MIN_THRESHOLD = 128 * 1024 * 1024   # 128 MiB
    _AUTO_FLUSH_INTERVAL = 1800               # 目标每 30 分钟 flush 一次
    _AUTO_WARMUP_FLUSHES = 3                 # 前 3 次 flush 后自适应(采样更稳定)
    _MAX_FLUSH_INTERVAL = 6 * 3600            # 最长 6 小时必须 flush 一次(保底)

    def __init__(
        self,
        config: Dict,
        fields: List[str],
        types: List[str],
        writer_id: int,
        in_queue: "queue.Queue",
        error_queue: "queue.Queue",
        partition_ready: "threading.Event",
        partition_lock: "threading.Lock",
        size_threshold: Optional[int] = None,
        max_size_threshold: int = 4 * 1024 * 1024 * 1024,
    ):
        # daemon=True:主进程意外死亡时 writer 跟着死,确保进程能退出。
        # 优雅 cleanup 由 hook.end() / atexit 兜底,best-effort 不强求。
        super().__init__(daemon=True)

        required = {"access_id", "access_key", "project", "table_name"}
        if missing := required - config.keys():
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")

        self.table_name = config["table_name"]
        self.partition = config.get("partition", None)
        self.fields = fields
        self.types = types
        self.write_id = writer_id
        self.in_queue = in_queue
        self.error_queue = error_queue
        self._max_size_threshold = max_size_threshold

        # size_threshold: None = 自适应, int = 固定值
        if size_threshold is None:
            self._auto_threshold = True
            self.size_threshold = self._AUTO_MIN_THRESHOLD
        else:
            self._auto_threshold = False
            self.size_threshold = min(size_threshold, self._max_size_threshold)

        # 自适应统计
        self._flush_count = 0
        self._total_flushed_nbytes = 0
        self._first_write_time: Optional[float] = None

        # 跨 writer 共享:第一次 commit 时串行化,避免多个 writer 并发触发 DDL race
        # 一旦任意 writer commit 成功(分区已建),partition_ready 被 set,
        # 后续所有 writer 直接并行 commit 不再走 lock。
        self._partition_ready = partition_ready
        self._partition_lock = partition_lock

        # 服务端真实 arrow schema(_open_write_session 时填充,用于 flush 时类型对齐)
        self._arrow_schema = None
        self._odps_col_types = {}  # 列名(小写) → ODPS 类型对象,用于 timestamp_ntz 判断

        # V3 buffer:按列累积 pa.Array
        self.columns: Dict[str, List[pa.Array]] = {}
        self.buffered_size = 0
        self.write_count = 0
        self._block_number = 0   # 同一 session 内的 block 编号,commit 后随新 session 重置
        self._commit_messages: List[str] = []
        self._last_flush_time = time.time()  # 上次 flush 时间(用于时间保底)

        # ODPS client
        if is_internal_enabled():
            odps_args, odps_kwargs, need_create_table = get_odps_access_info(config)
        else:
            odps_args = [
                config["access_id"],
                config["access_key"],
                config["project"],
                config["end_point"],
            ]
            odps_kwargs = {}
            need_create_table = True
        self._odps = ODPS(*odps_args, **odps_kwargs)

        # 多云环境跳过 schema namespace 查询,避免走 REST endpoint(后者在多云环境不通)
        self._odps.is_schema_namespace_enabled = lambda settings=None: False

        # 建表(如需)。分区由 storage API 在 create_write_session 时自动创建,无需显式建。
        if need_create_table:
            partitions, part_types = [], []
            if self.partition:
                for s in self.partition.split(","):
                    partitions.append(s.split("=")[0])
                    part_types.append("string")
            self._odps.create_table(
                self.table_name,
                schema=Schema.from_lists(
                    self.fields, self.types, partitions, part_types
                ),
                if_not_exists=True,
                lifecycle=365,
                table_properties={"columnar.nested.type": "true"},
            )
        else:
            logger.info(
                f"Skip create table: {self.table_name}, {self.partition}"
            )

        # storage API client + 首个 write session
        self._table = self._odps.get_table(self.table_name)

        # tunnel endpoint 选择(halo / tunnel)
        tunnel_endpoint = _resolve_tunnel_endpoint(odps_kwargs, config)
        quota_name = config.get("quota_name", "")
        self._client = StorageApiArrowClient(
            odps=self._odps,
            table=self._table,
            rest_endpoint=tunnel_endpoint,
            quota_name=quota_name,
        )

        self._ensure_partition()
        self._open_write_session()

    def _get_arrow_schema(self, write_resp) -> "pa.Schema":
        """从 storage API 响应中提取 arrow schema,同时缓存 ODPS 列类型用于 timestamp_ntz 判断。"""
        data_cols = write_resp.data_schema.data_columns
        odps_cols = [OdpsColumn(c.name, c.type) for c in data_cols]
        self._odps_col_types = {
            c.name.lower(): validate_odps_type(c.type) for c in data_cols
        }
        return odps_schema_to_arrow_schema(OdpsSchema(odps_cols))

    @staticmethod
    def _localize_timezone(col, tz=None):
        """为无时区的 timestamp 列添加时区信息(移植自 V1 tunnel writer)。"""
        if col.type.tz is not None:
            return col
        if tz is None:
            if odps_options.local_timezone is True or odps_options.local_timezone is None:
                from odps.lib import tzlocal
                tz = str(tzlocal.get_localzone())
            elif odps_options.local_timezone is False:
                tz = "UTC"
            else:
                tz = str(odps_options.local_timezone)
        if hasattr(pac, "assume_timezone") and isinstance(tz, str):
            return pac.assume_timezone(col, timezone=tz)
        else:
            pd_col = col.to_pandas().dt.tz_localize(tz)
            return pa.Array.from_pandas(pd_col)

    @staticmethod
    def _str_to_decimal_array(col, dec_type):
        """字符串列转 decimal 数组(移植自 V1 tunnel writer)。"""
        dec_col = col.to_pandas().map(Decimal)
        return pa.Array.from_pandas(dec_col, type=dec_type)

    def _ensure_partition(self):
        """rank 0 的第一个 writer 通过空写入预建分区,避免多 writer 并发 commit 时撞 DDL race。

        成功或失败都不影响后续流程:成功则分区已存在,失败则靠现有 lock/retry 兜底。
        """
        rank = int(os.environ.get("RANK", 0))
        if rank != 0 or self.write_id != 0:
            return
        try:
            write_req = TableBatchWriteRequest()
            write_req.partition_spec = self.partition
            write_req.overwrite = False
            write_resp = self._client.create_write_session(write_req)

            arrow_schema = self._get_arrow_schema(write_resp)
            empty_arrays = [pa.array([], type=field.type) for field in arrow_schema]
            empty_batch = pa.RecordBatch.from_arrays(
                empty_arrays, schema=arrow_schema
            )

            write_rows_req = WriteRowsRequest(
                session_id=write_resp.session_id,
                block_number=0,
            )
            arrow_writer = self._client.write_rows_arrow(write_rows_req)
            arrow_writer.write(empty_batch)
            commit_msg, ok = arrow_writer.finish()
            if not ok:
                logger.warning(
                    f"[writer-{self.write_id}] empty write finish returned ok=False, "
                    f"partition may not be created"
                )
                return

            session_req = SessionRequest(session_id=write_resp.session_id)
            self._client.commit_write_session(session_req, [commit_msg])
            logger.info(
                f"[writer-{self.write_id}] partition pre-created: {self.partition}"
            )
        except Exception as e:
            logger.warning(
                f"[writer-{self.write_id}] ensure_partition failed: {e!r}, "
                f"will rely on lock/retry fallback"
            )

    def _open_write_session(self):
        """创建 storage write session,准备接收数据。"""
        write_req = TableBatchWriteRequest()
        write_req.partition_spec = self.partition
        write_req.overwrite = False
        write_resp = self._client.create_write_session(write_req)
        self._session_id = write_resp.session_id
        self._block_number = 0

        # 首次创建 session 时获取真实表 schema,后续 commit 重开 session 不重复获取
        if self._arrow_schema is None:
            self._arrow_schema = self._get_arrow_schema(write_resp)

    @retry(retry_count=3, interval=10)
    def write(self, data: Dict[str, Union[np.ndarray, list]]):
        """V3 的 write:立即转 pa.Array 累积,精确累计 nbytes。"""
        if self._first_write_time is None:
            self._first_write_time = time.time()
        for k, v in data.items():
            if isinstance(v, list):
                v = np.asarray(v)
            arr = _to_pa_array(v)
            self.columns.setdefault(k, []).append(arr)
            self.buffered_size += arr.nbytes
        # flush 条件:buffer 大小达到阈值,或距上次 flush 超过 _MAX_FLUSH_INTERVAL
        if self.buffered_size >= self.size_threshold:
            self._flush_buffer()
        elif time.time() - self._last_flush_time >= self._MAX_FLUSH_INTERVAL:
            logger.info(
                f"[writer-{self.write_id}] time-based flush: "
                f"{time.time() - self._last_flush_time:.0f}s since last flush, "
                f"buffered_size={self.buffered_size / 1024 / 1024:.1f} MiB"
            )
            self._flush_buffer()

    def _align_types(self, write_data: "pa.RecordBatch") -> "pa.RecordBatch":
        """将 RecordBatch 各列对齐到 ODPS 表真实 schema。

        storage API 不像 V1 tunnel 那样在底层自动 cast,
        这里手动处理 timestamp 时区、decimal 字符串转换、通用类型 cast。
        逻辑移植自 V1 tunnel writer (odps/tunnel/io/writer.py)。
        """
        if self._arrow_schema is None:
            return write_data

        # schema 完全一致且无 timestamp 列时跳过
        if write_data.schema == self._arrow_schema and not any(
            isinstance(tp, pa.TimestampType) for tp in write_data.schema.types
        ):
            return write_data

        pa_dec_types = (pa.Decimal128Type,)
        if hasattr(pa, "Decimal256Type"):
            pa_dec_types += (pa.Decimal256Type,)

        lower_to_col = {
            n.lower(): c for n, c in zip(write_data.schema.names, write_data.columns)
        }
        casted_arrays = []
        for name, tp in zip(self._arrow_schema.names, self._arrow_schema.types):
            lower_name = name.lower()
            col = lower_to_col.get(lower_name)
            if col is None:
                raise ValueError(
                    f"Column '{name}' not found in trace data "
                    f"(writer-{self.write_id})"
                )

            # timestamp: 先统一转 timestamp,再做时区本地化
            if isinstance(tp, pa.TimestampType):
                if not isinstance(col.type, pa.TimestampType):
                    col = col.cast(pa.timestamp(tp.unit))
                odps_type = self._odps_col_types.get(lower_name)
                if odps_type == ODPS_TIMESTAMP_NTZ:
                    col = self._localize_timezone(col, "UTC")
                else:
                    col = self._localize_timezone(col)
                col = col.cast(pa.timestamp(tp.unit, col.type.tz))

            # decimal: 字符串列需先转 Decimal 再转 arrow decimal
            elif (
                isinstance(tp, pa_dec_types)
                and isinstance(col, (pa.Array, pa.ChunkedArray))
                and col.type in (pa.binary(), pa.string())
            ):
                col = self._str_to_decimal_array(col, tp)

            # 通用 cast
            if col.type == tp:
                casted_arrays.append(col)
            else:
                try:
                    casted_arrays.append(col.cast(tp, safe=False))
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    raise ValueError(
                        f"Failed to cast column {name} to type {tp}"
                    ) from None
        return pa.RecordBatch.from_arrays(casted_arrays, schema=self._arrow_schema)

    def _flush_buffer(self):
        """V3 的 flush:pa.concat_arrays 合并 + storage API 写入。

        - 用户必须 add_to_trace 所有声明的 fields,缺列直接 raise(fail-fast,不兜底)。
        - storage API 不做自动类型转换,_align_types 手动 cast 到服务端真实 schema。
        - block_number 在 write 之前就 +1,确保 retry 用新编号,避免 "writer duplicated"。
        """
        if not self.columns:
            return

        # 按 fields 顺序拼接(对齐 ODPS schema),缺列直接 raise
        arrays = []
        for name in self.fields:
            chunks = self.columns.get(name)
            if not chunks:
                raise ValueError(
                    f"Missing column '{name}' in trace data "
                    f"(writer-{self.write_id}, block_number={self._block_number})"
                )
            arrays.append(pa.concat_arrays(chunks))
        write_data = pa.RecordBatch.from_arrays(arrays, names=self.fields)

        # 类型对齐:将各列 cast 到 ODPS 表真实类型
        write_data = self._align_types(write_data)

        row_num = write_data.num_rows

        # block_number 在写之前就 take,即使本次 write 失败,下次 retry 用新编号
        # (避免 "ODPS-0420081: The writer is duplicated")
        bn = self._block_number
        self._block_number += 1

        write_req = WriteRowsRequest(
            session_id=self._session_id,
            block_number=bn,
        )
        arrow_writer = self._client.write_rows_arrow(write_req)
        arrow_writer.write(write_data)
        commit_msg, ok = arrow_writer.finish()
        if not ok:
            raise RuntimeError(
                f"storage write failed (writer-{self.write_id}, "
                f"block_number={bn}, row_num={row_num})"
            )

        self._commit_messages.append(commit_msg)
        self.write_count += row_num
        self._flush_count += 1
        self._total_flushed_nbytes += self.buffered_size
        self.columns = {}
        self.buffered_size = 0

        # 自适应调整 size_threshold
        if self._auto_threshold and self._flush_count == self._AUTO_WARMUP_FLUSHES:
            self._adjust_threshold()

        # 攒 2 个成功的 block commit 一次(用 commit_messages 长度判断,
        # 排除掉因 retry 跳过的 block_number)
        if len(self._commit_messages) >= 2:
            self.commit()
        self._last_flush_time = time.time()

    def _adjust_threshold(self):
        """根据前几次 flush 的吞吐率,调整 size_threshold。

        目标:每个 block 攒约 _AUTO_FLUSH_INTERVAL 秒的数据量,
        clamp 到 [_AUTO_MIN_THRESHOLD, _AUTO_MAX_THRESHOLD]。
        """
        elapsed = time.time() - self._first_write_time
        if elapsed <= 0:
            return
        rate = self._total_flushed_nbytes / elapsed  # bytes/s
        target = int(rate * self._AUTO_FLUSH_INTERVAL)
        new_threshold = max(self._AUTO_MIN_THRESHOLD, min(self._max_size_threshold, target))
        if new_threshold != self.size_threshold:
            logger.info(
                f"[writer-{self.write_id}] auto-adjust size_threshold: "
                f"{self.size_threshold / 1024 / 1024:.0f} MiB -> "
                f"{new_threshold / 1024 / 1024:.0f} MiB "
                f"(rate={rate / 1024 / 1024:.1f} MB/s)"
            )
            self.size_threshold = new_threshold

    @retry(retry_count=3, interval=10)
    def commit(self, force: bool = False):
        """V3 的 commit:storage API commit_write_session,然后开新 session。

        多 writer 并发时,第一次 commit 用 lock 串行化避免 partition DDL race;
        partition 建好后所有 writer 并行 commit。
        """
        if force:
            self._flush_buffer()
        if not self._commit_messages:
            return

        if not self._partition_ready.is_set():
            # 第一次 commit:抢锁串行,避免多 writer 同时触发 CREATE PARTITION DDL
            with self._partition_lock:
                # 双重检查:进锁前可能别的 writer 已经把分区建好了
                if not self._partition_ready.is_set():
                    self._do_commit()
                    self._partition_ready.set()
                    return
                # 别人已经建好分区,锁内直接 commit 即可(也不会撞 race)
                self._do_commit()
        else:
            # 分区已存在,服务端 commit 不触发 DDL,可放心并行
            self._do_commit()

    def _do_commit(self):
        """实际执行 commit_write_session + 开新 session。被 commit() 包装。"""
        session_req = SessionRequest(session_id=self._session_id)
        self._client.commit_write_session(session_req, self._commit_messages)
        self._commit_messages = []
        # 开新 session 给后续写入(内部会重置 self._block_number = 0)
        self._open_write_session()

    def run(self):
        """try/finally 保证最后一定 flush + 报告 write_count。"""
        try:
            while True:
                data = self.in_queue.get()
                if data is None:
                    break
                self.write(data)
        except Exception as e:
            # 异常时通过 error_queue 通知主线程,带完整上下文
            self.error_queue.put(
                {
                    "writer_id": self.write_id,
                    "exception": e,
                    "traceback": traceback.format_exc(),
                    "block_number": self._block_number,
                    "write_count": self.write_count,
                }
            )
            raise
        finally:
            try:
                self.commit(force=True)
            except Exception as e:
                logger.error(  # noqa: G201
                    f"[rank-{rank}] [writer-{self.write_id}] "
                    f"final commit failed: {e!r}",
                    exc_info=True,
                )  # noqa: G201
            logger.info(
                f"[rank-{rank}] [writer-{self.write_id}] "
                f"write_count = {self.write_count}"
            )


class TraceToOdpsHookV2(Hook):
    """V2 hook:threading + data convert + storage + error_queue + put timeout。

    用法跟老版一致:
        from recis.hooks import add_to_trace
        from recis.hooks.trace_to_odps_hook_v2 import TraceToOdpsHookV2

        hook = TraceToOdpsHookV2(config, fields, types, worker_num=4)
        trainer.add_hook(hook)
    """

    def __init__(
        self,
        config: Dict,
        fields: List[str],
        types: List[str],
        worker_num: int = 1,
        size_threshold: Optional[int] = None,
        max_size_threshold: int = 4 * 1024 * 1024 * 1024,
    ) -> None:
        super().__init__()

        self.queue: queue.Queue = queue.Queue(maxsize=worker_num)
        # error_queue:writer → 主线程,即时报告错误
        self.error_queue: queue.Queue = queue.Queue()

        # 跨 writer 共享:第一次 commit 串行化,避免 storage API 的 partition DDL race
        # 一旦任意 writer commit 成功(分区建好),partition_ready set,后续全并行
        self._partition_ready = threading.Event()
        self._partition_lock = threading.Lock()

        self.writer_num = worker_num
        self.writers: List[TraceWriterV2] = []
        for i in range(self.writer_num):
            self.writers.append(
                TraceWriterV2(
                    config,
                    fields,
                    types,
                    writer_id=i,
                    in_queue=self.queue,
                    error_queue=self.error_queue,
                    partition_ready=self._partition_ready,
                    partition_lock=self._partition_lock,
                    size_threshold=size_threshold,
                    max_size_threshold=max_size_threshold,
                )
            )
        for w in self.writers:
            w.start()

        # 兜底:进程退出前 best-effort 调一次 end(),即使 trainer 没调
        # (atexit 对 SIGKILL / 段错误无效,但能 cover Python 异常 / KeyboardInterrupt / sys.exit)
        self._closed = False
        self._start_time = time.time()
        atexit.register(self._safe_shutdown)
        MonitorReporter.report(
            "trace_hook_init", 1,
            {"trace_hook_version": "v2"},
            force=True, type="counter",
        )

    def _safe_shutdown(self):
        """atexit 注册的兜底,失败不抛。"""
        if self._closed:
            return
        try:
            self.end()
        except Exception as e:
            logger.warning(
                f"atexit shutdown of TraceToOdpsHookV2 failed: {e!r}"
            )

    def _check_errors(self):
        """主线程立即检查 error_queue,有错误就抛出带完整上下文的异常。"""
        try:
            err = self.error_queue.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(
            f"TraceWriter-{err['writer_id']} failed at "
            f"block_number={err['block_number']}, write_count={err['write_count']}\n"
            f"Original: {err['exception']!r}\n{err['traceback']}"
        ) from err["exception"]

    def _check_alive(self) -> bool:
        """所有 writer 是否仍在运行。"""
        return all(w.is_alive() for w in self.writers)

    def after_step(self, is_train=True, *args, **kwargs):
        """每个 step 调:先查错误队列 → put 数据(带超时循环防死锁)。"""
        self._check_errors()
        data = get_trace_map()
        # 带超时的 put 循环:writer 卡死时不会无限阻塞主训练线程
        while True:
            self._check_errors()
            if not self._check_alive():
                raise RuntimeError(
                    "All TraceWriters died, cannot enqueue more data. "
                    "Check writer logs for original failure."
                )
            try:
                self.queue.put(data, timeout=5)
                break
            except queue.Full:
                continue
        clear_trace_map()

    def end(self, is_train=True, *args, **kwargs):
        """优雅关闭:发哨兵 + join 带超时。幂等(可被 trainer + atexit 各调一次)。

        fail-fast 策略:
        - 进入 end() 时若任一 writer 已死 → pipeline 异常,raise(优先抛 error_queue 的原因)
        - 全活才正常给每个 writer 发一个 None,然后 join
        - put(None) 队列满时持续重试(等 writer 完成 commit / DDL 腾出空间)
        """
        if self._closed:
            return
        self._closed = True

        # 给每个 writer 投一个 None。
        # 判断 writer 是"被我们 None 干掉"还是"因 error 死":
        #   dead_count > n_sent  → 有 writer 因 error 死(不是消费 None),raise
        #   dead_count <= n_sent → 死掉的都是消费 None 的正常死亡
        # 360s 超时:writer 卡死不消费,raise
        n_sent = 0
        target = len(self.writers)
        deadline = time.time() + 360
        while n_sent < target:
            put_ok = False
            while time.time() < deadline:
                dead_count = sum(
                    1 for w in self.writers if not w.is_alive()
                )
                if dead_count > n_sent:
                    # 死亡数超过已发 None 数 → 有 writer 因 error 死,raise
                    self._check_errors()
                    dead_ids = [
                        w.write_id for w in self.writers if not w.is_alive()
                    ]
                    raise RuntimeError(
                        f"writer(s) {dead_ids} died during shutdown "
                        f"(unexpected: only sent {n_sent} sentinels)"
                    )
                try:
                    self.queue.put(None, timeout=5)
                    n_sent += 1
                    put_ok = True
                    break
                except queue.Full:
                    continue
            if not put_ok:
                raise RuntimeError(
                    f"writer not consuming queue for 360s during shutdown "
                    f"(sent {n_sent}/{target}); may be stuck in commit/retry."
                )

        for w in self.writers:
            w.join(timeout=1200)
            if w.is_alive():
                # daemon=True,主进程退出时会强制终止;这里只能 log
                logger.error(
                    f"writer-{w.write_id} did not exit in 1200s; "
                    f"will be force-terminated when main exits."
                )
        # 关闭阶段最后看一眼有没有遗漏的错误
        try:
            self._check_errors()
        except RuntimeError as e:
            logger.error(f"writer error during shutdown: {e}")

        total_write_count = sum(w.write_count for w in self.writers)
        total_flushed_nbytes = sum(w._total_flushed_nbytes for w in self.writers)
        total_flush_count = sum(w._flush_count for w in self.writers)
        elapsed = time.time() - self._start_time
        logger.info(f"[rank-{rank}] TraceToOdpsHookV2 total write_count = {total_write_count}")

        tag = {"trace_hook_version": "v2"}
        MonitorReporter.report("trace_hook_write_count", total_write_count, tag, force=True)
        MonitorReporter.report("trace_hook_flush_count", total_flush_count, tag, force=True)
        MonitorReporter.report("trace_hook_elapsed_s", elapsed, tag, force=True)
        if elapsed > 0:
            throughput = total_flushed_nbytes / elapsed / 1024 / 1024
            MonitorReporter.report("trace_hook_throughput_mbytes_s", throughput, tag, force=True)
