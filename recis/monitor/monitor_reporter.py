import os
import time
from contextlib import contextmanager
from functools import wraps

import torch

# from recis.common.singleton import SingletonMeta
from recis.monitor.monitor import GetFactory, PointType
from recis.utils.logger import Logger


TAG_MAP = {
    "gauge": PointType.kGauge,
    "gauge_sticky": PointType.kGaugeSticky,
    "counter": PointType.kCounter,
    "summary": PointType.kSummary,
}
PREPARE_NAME = "prepare"
MODEL_FWD_NAME = "model_forward"
SPARSE_FWD_NAME = "sparse_forward"
FEA_ENGINE_NAME = "feature_engine"
EMB_ENGINE_NAME = "embedding_engine"
EMB_BYTES_NAME = "embedding_bytes"
REDUCE_EMB_BYTES_NAME = "reduce_embedding_bytes"
ID_SIZE_NAME = "id_size"
UNIQUE_ID_SIZE_NAME = "unique_id_size"
ID_SIZE_A2A_TIME_NAME = "id_size_a2a_time"
QPS_NAME = "qps"
FLOPS_NAME = "flops"
FLOPS_PEAK = "flops_peak"
MFU_NAME = "mfu"
TRAIN_QPS_NAME = "train_qps"
EVAL_QPS_NAME = "eval_qps"
HT_ID_ACT_SIZE = "ht_id_activate_size"
HT_ID_TOTAL_SIZE = "ht_id_total_size"
HT_ALLOCATOR_ID_ACT_SIZE = "ht_allocator_id_activate_size"
HT_ALLOCATOR_ID_TOTAL_SIZE = "ht_allocator_id_total_size"
HT_ID_TOTAL_BYTES = "ht_id_memory"
HT_EMB_BYTES = "ht_emb_memory"
HT_ALL_SLOT_BYTES = "ht_all_slot_memory"
DS_END_LATENCY = "dataset_end.latency"
LOAD_SIZE_NAME = "load_size"
LOAD_TIME_NAME = "load_time"
SAVE_TIME_NAME = "save_time"


# class MonitorReporter(metaclass=SingletonMeta):
class MonitorReporter:
    _reportable = False
    logger = Logger("Metrics Reporter")
    monitor_prefix = "recis.framework"
    if not os.environ.get("BUILD_DOCUMENT", None) == "1":
        monitor_cli = GetFactory().get_client(monitor_prefix)

    # FIXME: SingletonMeta object decorator only works in py39+ which not supported by ruff rule now
    # def __init__(self) -> None:
    #     self.logger = Logger("Metrics Reporter")
    #     self.monitor_prefix = "recis.framework"
    #     self.monitor_cli = GetFactory().get_client(self.monitor_prefix)
    #     self._activate = False
    @classmethod
    def set_reportable(cls, reportable: bool):
        cls._reportable = reportable

    @classmethod
    def reportable(cls, force):
        return cls._reportable or force

    @classmethod
    def report(cls, metric_name, metric_val, tag=None, force=False, type="gauge"):
        if not cls.reportable(force):
            return
        tag = tag if tag is not None else {}
        cls.monitor_cli.report(metric_name, metric_val, tag, TAG_MAP[type])

    @classmethod
    def report_size(cls, metric_name, tensor, tag=None, force=False, type="gauge"):
        if not cls.reportable(force):
            return
        cls.report(metric_name, tensor.numel(), tag=tag, force=force, type=type)

    @classmethod
    def report_bytes(cls, metric_name, tensor, tag=None, force=False, type="gauge"):
        if not cls.reportable(force):
            return
        cls.report(
            metric_name,
            tensor.numel() * tensor.element_size(),
            tag=tag,
            force=force,
            type=type,
        )

    @classmethod
    @contextmanager
    def report_time(cls, metric_name, tag=None, force=False, type="gauge"):
        if cls.reportable(force):
            start = time.time()
            try:
                yield
            finally:
                torch.cuda.synchronize()
                end = time.time()
                elapsed = (end - start) * 1000  # ms
                cls.report(metric_name, elapsed, tag=tag, force=force, type=type)
        else:
            yield

    @classmethod
    def report_time_wrapper(cls, metric_name, tag=None, force=False, type="gauge"):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with cls.report_time(metric_name, tag=tag, force=force, type=type):
                    out = func(*args, **kwargs)
                return out

            return wrapper

        return decorator

    @classmethod
    def report_forward(
        cls, model: torch.nn.Module, metric_name, tag=None, force=False, type="gauge"
    ):
        fwd_fn = model.forward
        fwd_fn = cls.report_time_wrapper(metric_name, tag=tag, force=force, type=type)(
            fwd_fn
        )
        model.forward = fwd_fn
        return model
