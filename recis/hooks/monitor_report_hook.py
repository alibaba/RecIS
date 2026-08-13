import time
from dataclasses import dataclass
from typing import Optional

import torch

from recis.hooks.hook import Hook
from recis.monitor.flops_estimator import StartupFlopsState
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
        tflops_peak (float, optional): Peak TFLOPS used to calculate MFU.
            Non-positive values enable startup input-dtype mixed peak estimation.
        compute_precision (Precision, optional): Explicit compute precision used
            for a scalar peak lookup and disables input-dtype collection.
        eval_flops_ratio (float, optional): Eval-to-train FLOPS ratio used to
            normalize startup samples and report windows. Defaults to 1/3.
        min_peak_coverage (float, optional): Minimum fraction of profiler-countable
            FLOPS with known input-dtype peaks required to report lower-bound MFU.
            Defaults to 0.99.
    """

    interval_step: Optional[int] = 100
    tflops_peak: Optional[float] = -1
    compute_precision: Optional[Precision] = None
    eval_flops_ratio: float = 1.0 / 3.0
    min_peak_coverage: float = 0.99

    def __post_init__(self):
        if self.interval_step is not None and self.interval_step <= 0:
            raise ValueError("interval_step must be positive or None")
        if self.tflops_peak is not None and float(self.tflops_peak) > 0:
            self.tflops_peak = float(self.tflops_peak)
        else:
            self.tflops_peak = None
        if isinstance(self.compute_precision, str):
            try:
                self.compute_precision = Precision(self.compute_precision.lower())
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported compute precision: {self.compute_precision}"
                ) from exc
        if self.eval_flops_ratio <= 0:
            raise ValueError("eval_flops_ratio must be positive")
        if not 0 < self.min_peak_coverage <= 1:
            raise ValueError("min_peak_coverage must be in (0, 1]")


class MetricReportHook(Hook):
    """Hook that reports runtime throughput, FLOPS, and MFU metrics.

    Args:
        model (torch.nn.Module): Model used for sparse-memory reporting and as
            the fallback source of compute precision.
        report_args (ReportArguments, optional): Monitor interval, peak, and
            train/eval FLOPS conversion settings. Defaults to ReportArguments().
        mixed_precision (str, optional): Trainer precision hint used for peak
            selection and diagnostic logs when compute_precision is not
            explicitly configured. Defaults to None.
        flops_state (StartupFlopsState, optional): State shared with
            _InitialProfilerHook for consuming the startup FLOPS estimate. A
            new state is created when omitted.
    """

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

    def _resolve_precision(
        self,
        explicit_precision: Optional[Precision],
        mixed_precision: Optional[str],
    ):
        if explicit_precision is not None:
            return explicit_precision, "explicit"
        if mixed_precision:
            try:
                return Precision(mixed_precision.lower()), "accelerator"
            except ValueError:
                logger.warning(
                    "Unsupported Trainer mixed precision '%s'; "
                    "falling back to model parameter dtype.",
                    mixed_precision,
                )
        return self._get_model_precision(self.model), "model_parameters"

    def __init__(
        self,
        model: torch.nn.Module,
        report_args: Optional[ReportArguments] = None,
        mixed_precision: Optional[str] = None,
        flops_state: Optional[StartupFlopsState] = None,
    ):
        super().__init__()
        self.model = model
        self.args = report_args if report_args is not None else ReportArguments()
        self.flops_state = flops_state or StartupFlopsState()
        self.precision, self.precision_source = self._resolve_precision(
            self.args.compute_precision,
            mixed_precision,
        )
        if self.args.tflops_peak is not None:
            self.scalar_tflops_peak = self.args.tflops_peak
            self.collect_input_dtypes = False
            self.scalar_precision_basis = "explicit_tflops_peak"
            peak_source = "explicit"
        elif self.args.compute_precision is not None:
            self.scalar_tflops_peak = Inquirer.get_peak_tflops(
                device_index=0,
                precision=self.args.compute_precision,
            )
            self.args.tflops_peak = self.scalar_tflops_peak
            self.collect_input_dtypes = False
            self.scalar_precision_basis = "explicit_compute_precision"
            peak_source = "device_catalog"
        else:
            self.scalar_tflops_peak = None
            self.collect_input_dtypes = True
            self.scalar_precision_basis = "operator_input_dtype"
            peak_source = "startup_input_dtype_mix"

        if self.collect_input_dtypes:
            logger.info(
                "TFLOPS peak will be estimated from startup operator input dtypes "
                "(peak_source=%s, actual_math_mode=not_inferred)",
                peak_source,
            )
        elif self.scalar_tflops_peak is None:
            logger.warning(
                "MFU disabled: no peak TFLOPS for precision=%s "
                "(precision_source=%s). FLOPS and QPS remain enabled.",
                self.precision.value,
                self.precision_source,
            )
        else:
            logger.info(
                "TFLOPS peak=%s, precision=%s, precision_source=%s, peak_source=%s",
                self.scalar_tflops_peak,
                self.precision.value,
                self.precision_source,
                peak_source,
            )
        self.hashtables = filter_out_sparse_param(model)
        self.train_steps = 0
        self.eval_steps = 0
        self.interval_time = None
        self.step_time = None

    def _reset(self):
        self.train_steps = 0
        self.eval_steps = 0
        self.interval_time = None
        self.step_time = None

    @property
    def _window_steps(self):
        return self.train_steps + self.eval_steps

    def _report_metrics(self):
        window_steps = self._window_steps
        if self.interval_time is None or window_steps == 0:
            return
        # qps, train qps, eval qps
        spend_time = max(time.monotonic() - self.interval_time, 1e-12)
        qps = window_steps / spend_time
        train_qps = self.train_steps / spend_time
        eval_qps = self.eval_steps / spend_time

        MonitorReporter.report(QPS_NAME, qps, {"recis_qps_type": QPS_NAME})
        MonitorReporter.report(QPS_NAME, train_qps, {"recis_qps_type": TRAIN_QPS_NAME})
        MonitorReporter.report(QPS_NAME, eval_qps, {"recis_qps_type": EVAL_QPS_NAME})

        estimate = self.flops_state.estimate
        if estimate is None:
            logger.warning(
                "FLOPS/MFU unavailable: %s (window_steps=%s)",
                self.flops_state.invalid_reason or "startup_estimate_pending",
                window_steps,
            )
        else:
            window_scale = (
                self.train_steps + self.args.eval_flops_ratio * self.eval_steps
            )
            window_flops = estimate.train_equiv_flops_per_step * window_scale
            flops_total = window_flops / spend_time
            MonitorReporter.report(
                FLOPS_NAME,
                flops_total,
                {"recis_flops_type": FLOPS_NAME},
            )

            ideal_seconds = estimate.train_equiv_ideal_seconds_per_step
            if ideal_seconds is not None:
                if estimate.mixed_tflops_peak is not None:
                    flops_peak = estimate.mixed_tflops_peak * 1e12
                    MonitorReporter.report(
                        FLOPS_NAME,
                        flops_peak,
                        {"recis_flops_type": FLOPS_PEAK},
                    )
                mfu = round(ideal_seconds * window_scale / spend_time, 5)
                MonitorReporter.report(
                    MFU_NAME,
                    mfu,
                    {"recis_mfu_type": MFU_NAME},
                )

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

    def _flush(self):
        should_report = self.args.interval_step is not None and self._window_steps > 0
        try:
            if should_report:
                MonitorReporter.set_reportable(True)
                self._report_metrics()
        finally:
            self._reset()
            MonitorReporter.set_reportable(False)

    def before_step(self, is_train=True, *args, **kwargs):
        if self.args.interval_step is None:
            return
        if self._window_steps == 0 and self.interval_time is None:
            # monotonic is more robust than time.time().
            self.interval_time = time.monotonic()
        if self._window_steps + 1 >= self.args.interval_step:
            self.step_time = time.monotonic()
            MonitorReporter.set_reportable(True)

    def after_step(self, is_train=True, *args, **kwargs):
        if is_train:
            self.train_steps += 1
        else:
            self.eval_steps += 1
        if (
            self.args.interval_step is not None
            and self._window_steps >= self.args.interval_step
        ):
            self._flush()

    def end(self, *args, **kwargs):
        self._flush()

    def after_data(self, is_train=True, *args, **kwargs):
        if self.step_time is not None:
            elapsed_time = (time.monotonic() - self.step_time) * 1000
            MonitorReporter.report(PREPARE_NAME, elapsed_time)
