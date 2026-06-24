import time
from dataclasses import dataclass
from typing import Optional

import torch

from recis.hooks.hook import Hook
from recis.monitor.gpuinfo_inquirer import Inquirer, Precision
from recis.monitor.monitor_reporter import (
    EVAL_QPS_NAME,
    FLOPS_NAME,
    FLOPS_PEAK,
    HT_ALL_SLOT_BYTES,
    HT_ALLOCATOR_ID_ACT_SIZE,
    HT_ALLOCATOR_ID_TOTAL_SIZE,
    HT_EMB_BYTES,
    HT_ID_ACT_SIZE,
    HT_ID_TOTAL_BYTES,
    HT_ID_TOTAL_SIZE,
    MFU_NAME,
    PREPARE_NAME,
    QPS_NAME,
    TRAIN_QPS_NAME,
    MonitorReporter,
)
from recis.nn.modules.hashtable import filter_out_sparse_param
from recis.utils.logger import Logger


logger = Logger(__name__)


@dataclass
class ReportArguments:
    """Report arguments for monitor

    Args:
        interval_step (int, optional): report interval step. Defaults to 100.
        tflops_peak (float, optional): peak tflops. this will be used to calculate mfu. Defaults to -1 (means auto-detect).
    """

    interval_step: int = 100
    tflops_peak: float = -1

    # TODO: consider if tflops_step_ratio_map is needed
    # e.g. tflops_step_ratio_map: dict[str, float] = {"train": 1.0, "eval": 0.5, "tower_foo": 0.3, "tower_bar": 0.7} }
    #     when report_metrics, use map[step_name] to multiply the original flops

    def __post_init__(self):
        if self.tflops_peak and float(self.tflops_peak) > 0:
            self.tflops_peak = float(self.tflops_peak)
            return

        detected_tflops_peak = Inquirer.get_peak_tflops(
            device_index=0, precision=Precision.fp32
        )
        if detected_tflops_peak is None:
            detected_tflops_peak = 148.0
            logger.warning(
                f"Tflops peak detect none as default: {detected_tflops_peak}"
            )

        self.tflops_peak = float(detected_tflops_peak)


class MetricReportHook(Hook):
    _internal_profs = {
        FLOPS_NAME: 0,
    }

    def _get_model_precision(self, model: torch.nn.Module) -> Precision:
        try:
            dtype_map = {
                torch.float32: Precision.fp32,
                torch.float16: Precision.fp16,
                torch.bfloat16: Precision.bf16,
                torch.int8: Precision.int8,
            }
            dtype = next(
                (x.dtype for x in model.parameters()),
                next((x.dtype for x in model.buffers()), torch.float32),
            )
            return dtype_map.get(dtype, Precision.fp32)
        except Exception:
            return Precision.fp32

    def __init__(
        self,
        model: torch.nn.Module,
        report_args: Optional[ReportArguments] = None,
    ):
        super().__init__()
        self.model = model

        if report_args is not None:
            logger.info(f"Tflops peak set to: {report_args.tflops_peak}")
        else:
            precision = self._get_model_precision(self.model)
            tflops_peak = Inquirer.get_peak_tflops(device_index=0, precision=precision)
            report_args = ReportArguments(tflops_peak=tflops_peak)
            logger.info(f"Tflops peak detect: {tflops_peak} as precision: {precision}")
        self.hashtables = filter_out_sparse_param(model)
        self.args = report_args
        self.steps = 0
        self.train_steps = 0
        self.eval_steps = 0
        self.interval_time = time.time()
        self.step_time = time.time()
        self.activate = False  # indicate whether current step is activate to report

    def _reset(self):
        self.train_steps = 0
        self.eval_steps = 0
        self.interval_time = time.time()

    def _report_metrics(self):
        # qps, train qps, eval qps
        spend_time = time.time() - self.interval_time
        # qps = self.args.interval_step / spend_time # unprecise when window exchange
        qps = (self.train_steps + self.eval_steps) / spend_time
        train_qps = self.train_steps / spend_time
        eval_qps = self.eval_steps / spend_time
        flops_peak = self.args.tflops_peak * 1e12
        flops_total = (
            self.__class__._internal_profs.get(FLOPS_NAME, 0)
            * self.args.interval_step
            / spend_time
        )
        mfu = round(flops_total / flops_peak, 5)
        MonitorReporter.report(QPS_NAME, qps, {"recis_qps_type": QPS_NAME})
        MonitorReporter.report(QPS_NAME, train_qps, {"recis_qps_type": TRAIN_QPS_NAME})
        MonitorReporter.report(QPS_NAME, eval_qps, {"recis_qps_type": EVAL_QPS_NAME})
        MonitorReporter.report(
            FLOPS_NAME, flops_total, {"recis_flops_type": FLOPS_NAME}
        )
        MonitorReporter.report(FLOPS_NAME, flops_peak, {"recis_flops_type": FLOPS_PEAK})
        MonitorReporter.report(MFU_NAME, mfu, {"recis_mfu_type": MFU_NAME})

        # hashtable
        for ht_name, ht in self.hashtables.items():
            act_num, total_num = ht.id_info()
            MonitorReporter.report(
                HT_ID_ACT_SIZE, act_num, {"recis_ht_name": ht_name}, type="gauge_sticky"
            )
            MonitorReporter.report(
                HT_ID_TOTAL_SIZE,
                total_num,
                {"recis_ht_name": ht_name},
                type="gauge_sticky",
            )
            allocator_act_num, allocator_total_num = ht.allocator_id_info()
            MonitorReporter.report(
                HT_ALLOCATOR_ID_ACT_SIZE,
                allocator_act_num,
                {"recis_ht_name": ht_name},
                type="gauge_sticky",
            )
            MonitorReporter.report(
                HT_ALLOCATOR_ID_TOTAL_SIZE,
                allocator_total_num,
                {"recis_ht_name": ht_name},
                type="gauge_sticky",
            )
            total_mem = ht.id_memory_info()
            MonitorReporter.report(
                HT_ID_TOTAL_BYTES,
                total_mem,
                {"recis_ht_name": ht_name},
                type="gauge_sticky",
            )
            emb_mem, total_mem = ht.emb_memory_info()
            MonitorReporter.report(
                HT_EMB_BYTES, emb_mem, {"recis_ht_name": ht_name}, type="gauge_sticky"
            )
            MonitorReporter.report(
                HT_ALL_SLOT_BYTES,
                total_mem,
                {"recis_ht_name": ht_name},
                type="gauge_sticky",
            )

    def before_step(self, is_train=True, *args, **kwargs):
        if self.args.interval_step is None:
            return
        if self.steps % self.args.interval_step != 0:
            return
        self.step_time = time.time()
        self.activate = True
        MonitorReporter.set_reportable(True)

    def after_step(self, is_train=True, *args, **kwargs):
        self.steps += 1
        if is_train:
            self.train_steps += 1
        else:
            self.eval_steps += 1
        if not self.activate:
            return
        self._report_metrics()
        self._reset()
        MonitorReporter.set_reportable(False)
        self.activate = False

    def out_off_data(self, *args, **kwargs):
        self._reset()
        MonitorReporter.set_reportable(False)
        self.activate = False

    def after_data(self, is_train=True, *args, **kwargs):
        if self.activate:
            eclapsed_time = (time.time() - self.step_time) * 1000
            MonitorReporter.report(PREPARE_NAME, eclapsed_time)
