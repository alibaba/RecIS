import traceback
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)
from torch import nn
from torch.utils.data import Dataset

from recis.framework.checkpoint_manager import ExtraFields, Saver, SaverOptions
from recis.framework.metrics import add_metric, get_log_metrics
from recis.framework.pipeline_utils import (
    PREFETCH_AFTER_SPARSE_FORWARD,
    PREFETCH_BEFORE_BACKWARD,
    PREFETCH_BEFORE_FORWARD,
    PREFETCH_BEFORE_OPTIM_STEP,
    PrefetchArguments,
    notify_prefetch,
    resolve_prefetch_model,
    setup_pipeline_prefetch,
    wrap_with_prefetch,
)
from recis.framework.request_adapter import RequestAdapter
from recis.framework.server import server
from recis.hooks import Hook, LoggerHook, ProfilerHook
from recis.hooks.auto_profiler_hook import AutoProfilerArguments, _build_auto_profiler
from recis.hooks.checkpoint_hooks import (
    CheckpointLoadArguments,
    CheckpointLoadHook,
    CheckpointSaveArguments,
    CheckpointSaveHook,
)
from recis.hooks.initial_profiler_hook import _InitialProfilerHook
from recis.hooks.monitor_report_hook import MetricReportHook, ReportArguments
from recis.hooks.mos_report_hook import MosReporterEvalHook
from recis.monitor.monitor_reporter import MODEL_FWD_NAME, MonitorReporter
from recis.optim import sparse_optim
from recis.utils.data_utils import copy_data_to_device
from recis.utils.logger import Logger


logger = Logger(__name__)


@dataclass
class TrainingArguments:
    """Configuration class for training parameters.

    This dataclass contains all the configuration parameters needed for training,
    including optimization settings, logging intervals, and checkpoint management.

    Attributes:
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
                                         before performing an optimizer step. Defaults to 1.
        output_dir (str): Directory where checkpoints and logs will be saved.
                         Defaults to "output_dir".
        model_bank (Optional[list]): List of model bank paths for initialization.
                                   Defaults to None.
        log_steps (int): Number of training steps between logging. Defaults to 100.
        train_steps (Optional[int]): Maximum number of training steps. If None,
                                   will train for full epochs. Defaults to None.
        train_epoch (Optional[int]): Number of training epochs. Defaults to 1.
        eval_steps (Optional[int]): Number of evaluation steps. If None, evaluates
                                  on full dataset. Defaults to None.
        save_steps (Optional[int]): Number of steps between checkpoint saves.
                                  Defaults to 1000.
        max_to_keep (int): Maximum number of checkpoints to keep. Defaults to 5.
        save_concurrency_per_rank (int): Number of concurrent save operations per rank.
                                        Defaults to 4.
        save_every_n_windows (Optional[int]): Number of io windows to save checkpoints. Defaults to 1.
        save_every_n_epochs (Optional[int]): Number of epochs to save checkpoints. Defaults to None.
        save_end (bool): Whether to save checkpoints at the end of training. Defaults to True.
        load_update_steps (Optional[int]): Number of steps to load dynamic model bank. Defaults to None.
        load_update_windows (Optional[int]): Number of window to load dynamic model bank. Defaults to 1.
        load_update_epochs (Optional[int]): Number of epochs to load dynamic model bank. Defaults to None.
        params_not_save (Optional[list]): Names of parameters not to save. Defaults to None.
        save_filter_fn ([Callable]): Function to filter checkpoint blocks. Defaults to None.
        saver_option (Optional[SaverOptions]): Options for checkpoint saver. Defaults to None.
        ckpt_save_arg (Optional[CheckpointSaveArguments]): Arguments for checkpoint save. Defaults to None.
        ckpt_load_arg (Optional[CheckpointLoadArguments]): Arguments for checkpoint load. Defaults to None.
        mixed_precision (Optional[str]): Mixed precision training mode. Defaults to None. Only support "bf16" and "fp16".
        window_iter (Optional[int]): Number of windows to iter. Defaults to None.
        eval_mos_report_uri (Optional[str]): URI for MOS report when eval. Defaults to None.
        prefetch (PrefetchArguments): Pipeline prefetch configuration (enable
            switch / buffer size, custom transform fn, side-stream priority,
            notify position). Defaults to a disabled ``PrefetchArguments()``.
    """

    gradient_accumulation_steps: int = 1
    output_dir: str = "output_dir"
    model_bank: Optional[list] = None
    log_steps: int = 100
    train_steps: Optional[int] = None
    train_epoch: Optional[int] = 1
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = 1000
    max_to_keep: int = 5
    save_concurrency_per_rank: int = 4
    save_every_n_windows: int = 1
    save_every_n_epochs: Optional[int] = None
    save_end: Optional[bool] = True
    load_update_steps: Optional[int] = None
    load_update_windows: Optional[int] = 1
    load_update_epochs: Optional[int] = None
    params_not_save: Optional[List[str]] = None
    save_filter_fn: Optional[Callable] = None
    saver_option: Optional[SaverOptions] = None
    ckpt_save_arg: Optional[CheckpointSaveArguments] = None
    ckpt_load_arg: Optional[CheckpointLoadArguments] = None
    mixed_precision: Optional[str] = None
    window_iter: Optional[int] = None
    eval_mos_report_uri: Optional[str] = None
    prefetch: PrefetchArguments = field(default_factory=PrefetchArguments)


class Trainer:
    """Main training orchestrator with distributed training and checkpoint management.

    The Trainer class provides a comprehensive training framework that handles:
    - Distributed training coordination using Accelerate
    - Automatic checkpoint saving and loading
    - Training and evaluation loops with metrics tracking
    - Hook system for extensible training workflows
    - Support for both dense and sparse optimizers

    Attributes:
        args (TrainingArguments): Training configuration parameters.
        hooks (List[Hook]): List of training hooks for extensibility.
        train_dataset (Optional[Dataset]): Training dataset.
        eval_dataset (Optional[Dataset]): Evaluation dataset.
        model (nn.Module): The model to train.
        dense_optimizer (torch.optim.Optimizer): Dense parameter optimizer.
        dense_lr_scheduler: Learning rate scheduler for dense optimizer.
        sparse_optimizer (Optional[sparse_optim.SparseOptimizer]): Sparse parameter optimizer.
        data_to_cuda (bool): Whether to automatically move data to CUDA.
        accelerator (Accelerator): Accelerate instance for distributed training.

    Example:

    .. code-block:: python

        from recis.framework import Trainer, TrainingArguments
        from recis.optim import SparseAdamW
        from torch.optim import AdamW

        # Set training arguments
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            train_steps=10000,
            eval_steps=1000,
            save_steps=2000,
            log_steps=100,
            gradient_accumulation_steps=4,
        )

        # split sparse params
        from recis.nn.modules.hashtable import filter_out_sparse_param

        sparse_params = filter_out_sparse_param(model)

        # create optimizers
        sparse_optimizer = SparseAdamW(sparse_params, lr=0.001)
        dense_optimizer = AdamW(model.parameters(), lr=0.001)

        # create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dense_optimizers=(dense_optimizer, None),
            sparse_optimizer=sparse_optimizer,
            data_to_cuda=True,
        )
        # pipeline prefetch is configured via TrainingArguments, e.g.:
        # training_args.prefetch = PrefetchArguments(
        #     enable_pipeline_prefetch=1,
        #     notify_position="before_backward",
        # )

        # train the model
        trainer.train()
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        hooks: Optional[List[Hook]] = None,
        dense_optimizers: Tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
        ] = (None, None),
        sparse_optimizer: Optional[sparse_optim.SparseOptimizer] = None,
        data_to_cuda: bool = False,
        ddp_find_unused_parameters: bool = True,
        ddp_broadcast_buffers: bool = True,
        saver: Optional[Saver] = None,
        ddp_gradient_as_bucket_view: bool = False,
        ddp_bucket_cap_mb: int = 25,
        ddp_static_graph: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the Trainer with model, datasets, and training configuration.

        Args:
            model (Optional[nn.Module]): The model to train.
            args (TrainingArguments): Training configuration. If None, uses default.
            train_dataset (Optional[Dataset]): Training dataset.
            eval_dataset (Optional[Dataset]): Evaluation dataset.
            hooks (Optional[List[Hook]]): List of training hooks for extensibility.
            dense_optimizers (Tuple): Tuple of (optimizer, lr_scheduler) for dense parameters.
            sparse_optimizer (Optional[sparse_optim.SparseOptimizer]): Optimizer for sparse parameters.
            data_to_cuda (bool): Whether to automatically move data to CUDA. Defaults to False.
            ddp_find_unused_parameters (bool): Passed to DDP. When True, DDP
                traverses the whole autograd graph after every forward to
                detect unused parameters, which is expensive for large sparse
                graphs. Enable only if the dense model has parameters that
                may not receive gradients in some steps. Defaults to True to
                preserve the existing Trainer behavior.
            ddp_broadcast_buffers (bool): Passed to DDP. When True, DDP
                broadcasts all module buffers from rank 0 at every forward,
                which adds one collective per step (sparse-side buffers such
                as hashtable filter steps do not need it). Enable only if the
                dense model relies on synced buffers (e.g. BatchNorm running
                stats). Defaults to True, matching DDP's default behavior.
            ddp_gradient_as_bucket_view (bool): Passed to DDP. When True,
                gradients are views into communication buckets, reducing peak
                memory usage and copies. Defaults to False, preserving the
                existing Trainer/DDP behavior.
            ddp_bucket_cap_mb (int): Passed to DDP. Controls the communication
                bucket size in MiB. Defaults to 25, matching DDP's default.
            ddp_static_graph (bool): Passed to DDP. When True, DDP assumes that
                the used-parameter set and control flow remain stable for the
                entire training loop. Enable only for models that satisfy this
                requirement. Defaults to False, matching DDP's default.
            **kwargs: Additional arguments passed to Accelerator.
                - monitor_report_args (recis.hooks.monitor_report_hook.ReportArguments)
                - auto_profiler_args (recis.hooks.auto_profiler_hook.AutoProfilerArguments)
        """
        if hooks is None:
            hooks = []
        if args is None:
            args = TrainingArguments()
        self.args = args
        self.hooks = hooks
        self._auto_profiler_hook = None
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model = model
        self.dense_optimizer = dense_optimizers[0]
        self.dense_lr_scheduler = dense_optimizers[1]
        self.sparse_optimizer = sparse_optimizer
        self.data_to_cuda = data_to_cuda
        self._active_prefetch_iter = None
        prefetch_args = (
            args.prefetch if args.prefetch is not None else PrefetchArguments()
        )
        self._prefetch_stream_priority = prefetch_args.stream_priority
        self._prefetch_notify_position = prefetch_args.notify_position
        self._prefetch_enable_thread = prefetch_args.enable_thread_prefetch
        self._prefetch_fetch_in_thread = prefetch_args.fetch_data_in_thread
        (
            self._pipeline_prefetch_transform,
            self._prefetch_buffer_size,
        ) = setup_pipeline_prefetch(
            model=model,
            data_to_cuda=data_to_cuda,
            enable_pipeline_prefetch=prefetch_args.enable_pipeline_prefetch,
            pipeline_prefetch_fn=prefetch_args.pipeline_prefetch_fn,
        )
        self._setup_sparse_forward_notify(model)
        self.mixed_precision = args.mixed_precision
        if self.mixed_precision is not None:
            assert self.mixed_precision in ["bf16", "fp16"], (
                "mixed_precision must be 'bf16' or 'fp16'"
            )
        self._monitor_report_args: Optional[ReportArguments] = kwargs.pop(
            "monitor_report_args", None
        )
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=ddp_find_unused_parameters,
            broadcast_buffers=ddp_broadcast_buffers,
            gradient_as_bucket_view=ddp_gradient_as_bucket_view,
            bucket_cap_mb=ddp_bucket_cap_mb,
            static_graph=ddp_static_graph,
        )
        self._auto_profiler_args: Optional[AutoProfilerArguments] = kwargs.pop(
            "auto_profiler_args", None
        )
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            **kwargs,
        )
        named_optimizer = False
        if hasattr(self.dense_optimizer, "named_optimizer"):
            named_optimizer = True
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        (
            self.model,
            self.dense_optimizer,
            self.dense_lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.dense_optimizer, self.dense_lr_scheduler
        )
        if named_optimizer:
            self.dense_optimizer.named_optimizer = True
        MonitorReporter.report_forward(self.model, MODEL_FWD_NAME)
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.set_grad_accum_steps(self.gradient_accumulation_steps)
        self._global_step = torch.scalar_tensor(0, dtype=torch.int64)
        self._epoch = torch.scalar_tensor(0, dtype=torch.int64)
        self.saver = self.init_saver(model, args, saver)
        self.stop_state = torch.scalar_tensor(0, dtype=torch.int64).cuda()
        self.init_hooks()

    def _setup_sparse_forward_notify(self, model):
        """Register a forward hook for the ``after_sparse_forward`` position.

        Finds the sparse submodule (e.g. RecISModel, the same module that
        provides ``prefetch_step``) and notifies the prefetch iterator right
        after its forward completes, so the next-batch transform overlaps
        with the dense forward as well. Registered on the unwrapped module,
        which DDP shares by reference, so the hook survives ``prepare()``.
        """
        if (
            self._pipeline_prefetch_transform is None
            or self._prefetch_notify_position != PREFETCH_AFTER_SPARSE_FORWARD
        ):
            return
        sparse_module = resolve_prefetch_model(model)
        if sparse_module is None:
            raise TypeError(
                "notify_position='after_sparse_forward' requires a submodule "
                "implementing prefetch_step() (e.g. RecISModel) to hook; "
                f"none found under {type(model).__name__}."
            )

        def _notify_hook(module, args, output):
            # Train-only: eval iterators are not lazy_start and the train
            # iterator must not be released by eval-time forwards.
            if module.training:
                notify_prefetch(self._active_prefetch_iter)

        sparse_module.register_forward_hook(_notify_hook)
        logger.info(
            "Prefetch notify hooked after forward of "
            f"{type(sparse_module).__name__} (after_sparse_forward)"
        )

    def init_saver(self, model, args, saver):
        saver = self.build_saver(model, args, saver)
        if self.train_dataset is not None:
            saver.register_io_state(ExtraFields.train_io, self.train_dataset)
            if hasattr(self.train_dataset, "_window_paths"):
                saver.register_for_checkpointing(
                    ExtraFields.train_window_io, self.train_dataset
                )
        if self.eval_dataset is not None and hasattr(
            self.eval_dataset, "_window_paths"
        ):
            saver.register_io_state(ExtraFields.eval_io, self.eval_dataset)
            saver.register_for_checkpointing(
                ExtraFields.eval_window_io, self.eval_dataset
            )
        if self.dense_optimizer is not None:
            saver.register_for_checkpointing(
                ExtraFields.recis_dense_optim, self.dense_optimizer
            )
        if not saver.get_extra_data(ExtraFields.global_step):
            saver.register_for_checkpointing(ExtraFields.global_step, self._global_step)
        if not saver.get_extra_data(ExtraFields.train_epoch):
            saver.register_for_checkpointing(ExtraFields.train_epoch, self._epoch)
        return saver

    def build_saver(self, model, args, saver):
        if saver is None:
            saver_option = args.saver_option
            if saver_option is None:
                saver_option = SaverOptions(
                    model,
                    self.sparse_optimizer,
                    args.output_dir,
                    args.model_bank,
                    args.max_to_keep,
                    args.save_concurrency_per_rank,
                    args.params_not_save,
                    args.save_filter_fn,
                )
            saver = Saver(saver_option)
        return saver

    def init_hooks(self):
        self.hooks.append(LoggerHook(self.args.log_steps))
        if self.args.ckpt_save_arg is not None:
            ckpt_save_arg = self.args.ckpt_save_arg
        else:
            ckpt_save_arg = CheckpointSaveArguments(
                self.args.save_steps,
                self.args.save_every_n_windows,
                self.args.save_every_n_epochs,
                self.args.save_end,
            )
        self.hooks.append(
            CheckpointSaveHook(
                self.saver, self._global_step, self._epoch, ckpt_save_arg
            )
        )
        if self.args.ckpt_load_arg is not None:
            ckpt_load_arg = self.args.ckpt_load_arg
        else:
            ckpt_load_arg = CheckpointLoadArguments(
                self.args.load_update_steps,
                self.args.load_update_windows,
                self.args.load_update_epochs,
            )
        self.hooks.append(
            CheckpointLoadHook(
                self.saver, self._global_step, self._epoch, ckpt_load_arg
            )
        )
        metric_report_hook = MetricReportHook(
            model=self.model,
            report_args=self._monitor_report_args,
            mixed_precision=self.mixed_precision,
        )
        self.hooks.append(
            _InitialProfilerHook(
                flops_state=metric_report_hook.flops_state,
                eval_flops_ratio=metric_report_hook.args.eval_flops_ratio,
                collect_input_dtypes=metric_report_hook.collect_input_dtypes,
                mixed_precision=self.mixed_precision,
                scalar_tflops_peak=metric_report_hook.scalar_tflops_peak,
                scalar_precision_basis=metric_report_hook.scalar_precision_basis,
                min_peak_coverage=metric_report_hook.args.min_peak_coverage,
            )
        )
        # Prevent profiler conflicts; user-configured profiler takes precedence.
        has_manual_profiler = any(isinstance(hook, ProfilerHook) for hook in self.hooks)
        if not has_manual_profiler:
            auto_profiler_arg = self._auto_profiler_args
            if auto_profiler_arg is None:
                auto_profiler_arg = AutoProfilerArguments()
            try:
                self._auto_profiler_hook = _build_auto_profiler(
                    args=auto_profiler_arg,
                    rank=self.accelerator.process_index,
                    world_size=self.accelerator.num_processes,
                )
            except Exception:
                # Auto profiler must never break trainer init.
                self._auto_profiler_hook = None
                logger.error(f"Skip auto profiler:\n{traceback.format_exc()}")
            if self._auto_profiler_hook is not None:
                self.hooks.append(self._auto_profiler_hook)

        self.hooks.append(metric_report_hook)
        if self.args.eval_mos_report_uri:
            self.hooks.append(MosReporterEvalHook(self.args.eval_mos_report_uri))

    def add_hooks(self, hooks: List[Hook]):
        """Add multiple hooks to the trainer.

        Args:
            hooks (List[Hook]): List of hooks to add.
        """
        for hook in hooks:
            self.add_hook(hook)

    def add_hook(self, hook: Hook):
        """Add a single hook to the trainer.

        Args:
            hook (Hook): The hook to add.
        """

        # Prevent profiler conflicts; user-configured profiler takes precedence.
        if isinstance(hook, ProfilerHook) and self._auto_profiler_hook is not None:
            for index, registered_hook in enumerate(self.hooks):
                if registered_hook is self._auto_profiler_hook:
                    self.hooks.pop(index)
                    break
            self._auto_profiler_hook = None
        self.hooks.append(hook)

    def train(self, train_steps=None):
        """Execute the training loop.

        Args:
            train_steps (Optional[int]): Override for number of training steps.
                                       If None, uses args.train_steps.
        """
        for hook in self.hooks:
            hook.start(is_train=True)
        if hasattr(self.train_dataset, "_window_paths"):
            train_loop_fn = self._train_loop_by_window
            for hook in self.hooks:
                hook.window_mode()
        else:
            train_loop_fn = self._train_loop
        while self._epoch < self.args.train_epoch:
            for hook in self.hooks:
                hook.before_epoch(is_train=True)
            train_loop_fn(
                self.args.train_steps if train_steps is None else train_steps,
                epoch=self._epoch,
            )
            self.train_dataset.reset()
            for hook in self.hooks:
                hook.after_epoch(is_train=True)
        for hook in self.hooks:
            hook.end(is_train=True)

    def evaluate(self, eval_steps=None):
        """Execute the evaluation loop.

        Args:
            eval_steps (Optional[int]): Override for number of evaluation steps.
                                      If None, evaluates on full dataset.
        """
        for hook in self.hooks:
            hook.start(is_train=False)
        if hasattr(self.eval_dataset, "_window_paths"):
            eval_loop_fn = self._eval_loop_by_window
            for hook in self.hooks:
                hook.window_mode()
        else:
            eval_loop_fn = self._eval_loop
        for hook in self.hooks:
            hook.before_epoch(is_train=False)
        eval_loop_fn(
            self.args.eval_steps if eval_steps is None else eval_steps,
        )
        for hook in self.hooks:
            hook.after_epoch(is_train=False)
            hook.end(is_train=False)

    def train_and_evaluate(self, train_steps=None, eval_steps=None):
        """Execute alternating training and evaluation loops.

        Args:
            train_steps (Optional[int]): Override for number of training steps per epoch.
            eval_steps (Optional[int]): Override for number of evaluation steps.
        """
        for hook in self.hooks:
            hook.start(is_train=True)
        if hasattr(self.train_dataset, "_window_paths"):
            assert hasattr(self.eval_dataset, "_window_paths"), (
                "train and eval dataset should both be window io"
            )
            loop_fn = self._train_eval_loop_by_window
            for hook in self.hooks:
                hook.window_mode()
        else:
            assert not hasattr(self.eval_dataset, "_window_paths"), (
                "train and eval dataset should both not window io"
            )
            loop_fn = self._train_eval_loop
        while self._epoch < self.args.train_epoch:
            for hook in self.hooks:
                hook.before_epoch(is_train=True)
            loop_fn(
                self.args.train_steps if train_steps is None else train_steps,
                self.args.eval_steps if eval_steps is None else eval_steps,
                epoch=self._epoch,
            )
            self.train_dataset.reset()
            self.eval_dataset.reset()
            for hook in self.hooks:
                hook.after_epoch(is_train=True)
        for hook in self.hooks:
            hook.end(is_train=True)

    def get_new_window_iter(self, dataset):
        if not hasattr(dataset, "_window_paths"):
            raise TypeError("dataset must be window_io")
        while True:
            try:
                need_skip = dataset.next_window()
            except StopIteration:
                logger.info("Window IO Finish")
                return None
            except Exception as e:
                raise e
            read_offset = int(dataset._read_offset[0])
            if need_skip:
                logger.info(f"Skip for window, offset = {read_offset}")
            else:
                logger.info(f"Next window, offset = {read_offset}")
                break
        return iter(dataset)

    def sync_exit_flag(self, flag: bool):
        self.stop_state.fill_(int(flag))
        dist.all_reduce(self.stop_state, op=dist.ReduceOp.MAX)
        return bool(self.stop_state.item())

    def server(
        self,
        orc_path,
        name_list,
        request_adapter: Optional[RequestAdapter] = None,
        need_flatten=False,
    ):
        for hook in self.hooks:
            hook.start(is_train=True)
        server(
            orc_path,
            self.model,
            name_list,
            self.train_dataset,
            request_adapter=request_adapter,
            need_flatten=need_flatten,
        )

    def _train_loop_by_window(self, max_steps=None, epoch=1):
        window_iter = 0
        self.model.train()
        while True:
            if (
                self.args.window_iter is not None
                and window_iter >= self.args.window_iter
            ):
                break
            iterator = self.get_new_window_iter(self.train_dataset)
            need_break = iterator is None
            need_break = self.sync_exit_flag(need_break)
            if need_break:
                break
            iterator = wrap_with_prefetch(
                iterator,
                self._pipeline_prefetch_transform,
                buffer_size=self._prefetch_buffer_size,
                lazy_start=True,
                stream_priority=self._prefetch_stream_priority,
                enable_thread=self._prefetch_enable_thread,
                fetch_in_thread=self._prefetch_fetch_in_thread,
            )
            for hook in self.hooks:
                hook.before_window(is_train=True)
            self._train_loop_internal(iterator, max_steps, epoch)
            for hook in self.hooks:
                hook.after_window(is_train=True)
            window_iter += 1

    def _eval_loop_by_window(self, max_steps=None):
        window_iter = 0
        self.model.eval()
        while True:
            if (
                self.args.window_iter is not None
                and window_iter >= self.args.window_iter
            ):
                break
            iterator = self.get_new_window_iter(self.eval_dataset)
            need_break = iterator is None
            need_break = self.sync_exit_flag(need_break)
            if need_break:
                break
            iterator = wrap_with_prefetch(
                iterator,
                self._pipeline_prefetch_transform,
                buffer_size=self._prefetch_buffer_size,
                stream_priority=self._prefetch_stream_priority,
                enable_thread=self._prefetch_enable_thread,
                fetch_in_thread=self._prefetch_fetch_in_thread,
            )
            for hook in self.hooks:
                hook.before_window(is_train=False)
            self._eval_loop_internal(iterator, max_steps)
            for hook in self.hooks:
                hook.after_window(is_train=False)
            window_iter += 1

    def _train_eval_loop_by_window(self, train_steps=None, eval_steps=None, epoch=1):
        window_iter = 0
        while True:
            if (
                self.args.window_iter is not None
                and window_iter >= self.args.window_iter
            ):
                break
            train_iterator = self.get_new_window_iter(self.train_dataset)
            train_need_break = train_iterator is None
            train_need_break = self.sync_exit_flag(train_need_break)
            if train_need_break:
                logger.info(
                    "train_and_eval window will stop, because train dataset has no window to read."
                )
                break
            train_iterator = wrap_with_prefetch(
                train_iterator,
                self._pipeline_prefetch_transform,
                buffer_size=self._prefetch_buffer_size,
                lazy_start=True,
                stream_priority=self._prefetch_stream_priority,
                enable_thread=self._prefetch_enable_thread,
                fetch_in_thread=self._prefetch_fetch_in_thread,
            )
            for hook in self.hooks:
                hook.before_window(is_train=True)
            self.model.train()
            self._train_loop_internal(train_iterator, train_steps, epoch)
            eval_iterator = self.get_new_window_iter(self.eval_dataset)
            eval_need_break = eval_iterator is None
            eval_need_break = self.sync_exit_flag(eval_need_break)
            if eval_need_break:
                logger.info(
                    "train_and_eval window will stop, because eval dataset has no window to read."
                )
                break
            eval_iterator = wrap_with_prefetch(
                eval_iterator,
                self._pipeline_prefetch_transform,
                buffer_size=self._prefetch_buffer_size,
                stream_priority=self._prefetch_stream_priority,
                enable_thread=self._prefetch_enable_thread,
                fetch_in_thread=self._prefetch_fetch_in_thread,
            )
            for hook in self.hooks:
                hook.after_window(is_train=True)
            self.model.eval()
            self._eval_loop_internal(eval_iterator, eval_steps)
            for hook in self.hooks:
                hook.after_window(is_train=False)
            window_iter += 1

    def _train_loop(self, max_steps=None, epoch=1):
        self.model.train()
        iterator = wrap_with_prefetch(
            iter(self.train_dataset),
            self._pipeline_prefetch_transform,
            buffer_size=self._prefetch_buffer_size,
            lazy_start=True,
            stream_priority=self._prefetch_stream_priority,
            enable_thread=self._prefetch_enable_thread,
            fetch_in_thread=self._prefetch_fetch_in_thread,
        )
        self._train_loop_internal(iterator, max_steps, epoch)

    def _eval_loop(self, max_steps=None):
        self.model.eval()
        iterator = wrap_with_prefetch(
            iter(self.eval_dataset),
            self._pipeline_prefetch_transform,
            buffer_size=self._prefetch_buffer_size,
            stream_priority=self._prefetch_stream_priority,
            enable_thread=self._prefetch_enable_thread,
            fetch_in_thread=self._prefetch_fetch_in_thread,
        )
        self._eval_loop_internal(iterator, max_steps)

    def _train_eval_loop(self, train_steps=None, eval_steps=None, epoch=1):
        self._train_loop(train_steps, epoch)
        self._eval_loop(eval_steps)

    def _eval_loop_internal(self, data_iter, max_steps=None):
        lstep = 0
        while True:
            if max_steps is not None and lstep >= max_steps:
                break
            for hook in self.hooks:
                hook.before_step(is_train=False)
            stop_flag, data = next(data_iter)
            need_break = self.sync_exit_flag(stop_flag)
            if need_break:
                for hook in self.hooks:
                    hook.out_off_data()
                break
            if self._pipeline_prefetch_transform is None and self.data_to_cuda:
                data = copy_data_to_device(data, "cuda", non_blocking=True)
            for hook in self.hooks:
                hook.after_data(is_train=False, data=data)
            with torch.no_grad():
                eval_result = self.model(data)
            for hook in self.hooks:
                hook.after_step(
                    metrics=get_log_metrics(),
                    global_step=self._global_step,
                    is_train=False,
                    eval_result=eval_result,
                )
            lstep += 1
        for hook in self.hooks:
            hook.after_eval()

    def _train_loop_internal(self, data_iter, max_steps=None, epoch=1):
        if self._pipeline_prefetch_transform is not None:
            self._active_prefetch_iter = data_iter
        lstep = 0
        while True:
            if max_steps is not None and lstep >= max_steps:
                break
            for hook in self.hooks:
                hook.before_step(is_train=True)
            stop_flag, data = next(data_iter)
            need_break = self.sync_exit_flag(stop_flag)
            if need_break:
                for hook in self.hooks:
                    hook.out_off_data()
                break
            if self._pipeline_prefetch_transform is None and self.data_to_cuda:
                data = copy_data_to_device(data, "cuda", non_blocking=True)
            for hook in self.hooks:
                hook.after_data(is_train=True, data=data)
            with self.accelerator.accumulate(self.model):
                self._train_step(data, epoch)
            for hook in self.hooks:
                hook.after_step(
                    metrics=get_log_metrics(),
                    global_step=self._global_step,
                    is_train=True,
                )
            lstep += 1

        # Drop the reference so the after_sparse_forward hook cannot notify
        # a finished iterator (e.g. during eval-time forwards), and the old
        # iterator can be reclaimed promptly on window switches.
        self._active_prefetch_iter = None

        for hook in self.hooks:
            hook.after_train()

    @property
    def output_dir(self):
        """转发 Saver.output_dir. openlm_hub 模式下返回 MOS URI 而非文件路径."""
        if self.saver is not None:
            return self.saver.output_dir
        return None

    def _train_step(self, data, epoch):
        # Release next-step prefetch at the configured position so its
        # transform overlaps with the remaining phases of this step.
        if self._prefetch_notify_position == PREFETCH_BEFORE_FORWARD:
            notify_prefetch(self._active_prefetch_iter)
        with self.accelerator.autocast():
            loss = self.model(data)

        # Release prefetch before backward. Loss materialization is deferred
        # until after optimizer work so it does not delay DDP communication.
        if self._prefetch_notify_position == PREFETCH_BEFORE_BACKWARD:
            notify_prefetch(self._active_prefetch_iter)

        add_metric("epoch", epoch, report_to_mos=True)

        self.accelerator.backward(loss)
        for hook in self.hooks:
            hook.after_backward(is_train=True)

        if self._prefetch_notify_position == PREFETCH_BEFORE_OPTIM_STEP:
            notify_prefetch(self._active_prefetch_iter)
        # order must be step before zero grad
        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()
        if self.dense_lr_scheduler is not None:
            self.dense_lr_scheduler.step()
        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        add_metric("loss", loss.item(), report_to_mos=True)
