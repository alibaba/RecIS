import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional

import torch

from recis.framework.checkpoint_compat import (
    collect_filter_global_step_names,
    use_child_filter_global_step_names,
)
from recis.framework.filesystem import get_file_system
from recis.framework.model_bank import (
    MBC,
    ModelBankParser,
    get_update_path,
    load_pt_file,
    pickle_to_torch,
    show_model_bank_format,
)
from recis.nn.modules.hashtable import (
    filter_out_sparse_param,
    split_sparse_dense_state_dict,
)
from recis.optim.sparse_optim import SparseOptimizer
from recis.serialize import Loader as SLoader, Saver as SSaver
from recis.utils.logger import Logger


logger = Logger(__name__)


def get_default_sync_fn(shard_num):
    if shard_num > 1:
        sync_func = torch.distributed.barrier
    else:

        def sync_func():
            return None

    return sync_func


class ExtraFields:
    global_step = "global_step"
    recis_dense_optim = "recis.dense.optim."
    train_io = "train_io"
    eval_io = "eval_io"
    train_window_io = "train_window_io"
    eval_window_io = "eval_window_io"
    io_state = "io_state"
    train_epoch = "train_epoch"
    prev_optim = "dense_optimizer"

    _fields = {
        global_step,
        recis_dense_optim,
        train_io,
        eval_io,
        train_window_io,
        eval_window_io,
        train_epoch,
    }

    @classmethod
    def get_io_fields(cls):
        return {
            cls.train_window_io,
            cls.eval_window_io,
            cls.train_io,
            cls.eval_io,
            cls.train_epoch,
        }

    @classmethod
    def all_fields(cls):
        return cls._fields

    @classmethod
    def __contains__(cls, item):
        return item in cls._fields


def filter_bank(model_bank_conf: dict, internal: dict):
    load_info = {k: {k: []} for k in internal.keys()}
    for k in model_bank_conf.keys():
        if "@" in k:
            name, type = k.split("@")
            assert name in load_info, f"name {name} not found in internal"
            load_info[name][name].append(type)
        else:
            name = k
            assert name in load_info, f"name {name} not found in internal"

    # if not load any sparse model, not load sparse_adamw_beta optimizer
    if len(model_bank_conf) == 0:
        load_info = {k: v for k, v in load_info.items() if len(v[k]) > 0}

    new_load_info = {}
    table_mapping = {}
    for key, conf in model_bank_conf.items():
        if MBC.ONAME in conf:
            src_table = key.split("@")[0]
            tgt_table = conf[MBC.ONAME].split("@")[0]
            if src_table not in table_mapping:
                table_mapping[src_table] = tgt_table
            else:
                assert table_mapping[src_table] == tgt_table, (
                    f"table {src_table} mapping to different table {tgt_table}"
                )

    for top_key, inner_dict in load_info.items():
        inner_key = next(iter(inner_dict.keys()))
        inner_value = inner_dict[inner_key]
        if inner_key in table_mapping:
            target_table = table_mapping[inner_key]
            new_load_info[top_key] = {target_table: inner_value}
        else:
            new_load_info[top_key] = inner_dict

    return new_load_info


@dataclass
class SaverOptions:
    model: torch.nn.Module
    sparse_optim: Optional[SparseOptimizer]
    output_dir: Optional[str] = None
    model_bank: Optional[list] = None
    max_keep: int = 1
    concurrency: int = 4
    params_not_save: Optional[List[str]] = None
    save_filter_fn: Optional[Callable] = None


class Saver:
    """Checkpoint saver for managing model and training state persistence.

    The Saver class handles the saving and loading of model checkpoints including:
    - Dense and sparse model parameters
    - Optimizer states
    - IO states for datasets
    - Checkpoint versioning and cleanup
    - Support for distributed filesystems

    Example:
        >>> saver = Saver(
        ...     model=model,
        ...     sparse_optim=sparse_optimizer,
        ...     output_dir="./checkpoints",
        ...     max_keep=5,
        ... )
        >>> saver.save("checkpoint_001")
    """

    kIndexSuffix = ".index"
    kIndexName = "index"

    def __init__(
        self,
        options: SaverOptions,
    ):
        """Initialize the checkpoint saver.

        Args:
            model (torch.nn.Module): The model to save checkpoints for.
            sparse_optim (Optional): Sparse optimizer instance for sparse parameters.
            output_dir (str): Directory to save checkpoints. Defaults to "./".
            max_keep (int): Maximum number of checkpoints to keep. Defaults to 1.
            concurrency (int): Number of concurrent save operations. Defaults to 4.
        """
        self._shard_id = int(os.environ.get("RANK", 0))
        self._shard_num = int(os.environ.get("WORLD_SIZE", 1))
        self._model = options.model
        self._filter_global_step_names = collect_filter_global_step_names(self._model)
        self._sparse_state_dict, self._dense_state_dict = split_sparse_dense_state_dict(
            self._model.state_dict()
        )
        use_child_filter_global_step_names(
            self._dense_state_dict, self._filter_global_step_names
        )
        self._dense_name_aliases = {
            child_name: group.legacy_names
            for group in self._filter_global_step_names
            for child_name in group.child_names
        }
        self._dense_name_to_runtime = {
            child_name: group.runtime_name
            for group in self._filter_global_step_names
            for child_name in group.child_names
        }
        self._dense_parameter_names = {
            name for name, _ in self._model.named_parameters()
        }
        self._checkpoint_file = "checkpoint"
        self._checkpoint_version_list = []
        self._max_keep = options.max_keep
        self._extra_save_dict = {}

        self._output_dir = options.output_dir

        self._sparse_optim = options.sparse_optim
        self._sparse_optim_state = {}
        if self._sparse_optim is not None:
            self._sparse_optim_state = self._sparse_optim.state_dict()
            self._sparse_state_dict.update(self._sparse_optim_state)
        self._concurrency = options.concurrency
        self._sparse_filter_fn = self.build_sparse_filter_fn(options)
        self._io_state = {}

        self._dense_names = self._get_dense_names()
        self._sparse_names, self._sparse_tables = self._get_sparse_names()

        self._model_names = (
            self._dense_names | self._sparse_names | ExtraFields.all_fields()
        )

        self._model_bank_content = options.model_bank
        self._has_bank = False
        if self._model_bank_content is None or (
            isinstance(self._model_bank_content, list)
            and len(self._model_bank_content) == 0
        ):
            logger.warning("No model bank provided, use default model bank")
            self._model_bank_content = []
        self._init_model_bank(self._model_bank_content)

    def build_sparse_filter_fn(self, args):
        def filter_fn(blocks):
            if args.params_not_save is not None:
                filtered_blocks = set()
                params_not_save = set(args.params_not_save)
                for block in blocks:
                    if block.tensor_name() in params_not_save:
                        filtered_blocks.add(block)
                blocks = list(set(blocks) - filtered_blocks)
            if args.save_filter_fn is not None:
                blocks = args.save_filter_fn(blocks)
            return blocks

        return filter_fn

    def _check_name_conflict(self):
        dense_names = set()
        for name, _ in self._model.named_parameters():
            dense_names.add(name)

        for key in self._sparse_state_dict.keys():
            if key in dense_names:
                raise ValueError(
                    f"model name conflict, sparse and dense names should not have intersection: {key}"
                )

    def _init_model_bank(self, model_bank=None):
        model_bank_content = (
            model_bank if model_bank is not None else self._model_bank_content
        )
        self._check_name_conflict()

        self._model_bank_parser = ModelBankParser(
            self._output_dir,
            model_bank_content,
            self._model_names,
            self._sparse_names,
            self._sparse_tables,
            self._dense_names,
            ExtraFields,
            self._dense_name_aliases,
            self._dense_parameter_names,
            self._dense_name_to_runtime,
        )

        self._has_bank = self._model_bank_parser.has_bank()
        self._all_model_bank = self._model_bank_parser.parse_all_model_bank()
        self._dynamic_model_bank = self._model_bank_parser.parse_dynamic_model_bank()

        if 0 == self._shard_id:
            self._show_model_bank_table()

    def _show_model_bank_table(self):
        show_model_bank_format(
            "all_model_bank",
            self._all_model_bank,
        )

        show_model_bank_format(
            "dynamic_model_bank",
            self._dynamic_model_bank,
        )

    @property
    def output_dir(self):
        """Return the checkpoint output directory."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        """Redirect checkpoint output to a different directory."""
        self._output_dir = value

    def _get_dense_names(self):
        return set(self._dense_state_dict.keys())

    def _get_sparse_names(self):
        model_names = set()
        sparse_state_copy = self._sparse_state_dict.copy()
        sparse_state_dict, dense_state_dict = split_sparse_dense_state_dict(
            sparse_state_copy
        )
        model_names.update(dense_state_dict.keys())
        for hashtable_obj in sparse_state_dict.values():
            slot_group = hashtable_obj.slot_group()
            children_info = hashtable_obj.children_info()
            children_names = children_info.children()
            for child_name in children_names:
                slots = slot_group.slots()
                for slot in slots:
                    model_names.add(f"{child_name}@{slot.name()}")
                model_names.add(f"{child_name}@id")

        sparse_tables = set()
        for tensor in model_names:
            if "@" in tensor:
                sparse_tables.add(tensor.split("@")[0])

        return model_names, sparse_tables

    def register_io_state(self, name, obj: object):
        """Register an object for IO state persistence.

        Args:
            name (str): Name identifier for the IO state.
            obj (object): Object that supports IO state dump/load operations.

        Raises:
            ValueError: If the name is already registered.
        """
        if name not in self._io_state:
            self._io_state[name] = obj
        else:
            raise ValueError(f"name {name} already registered in io state")

    def register_for_checkpointing(self, name, obj: object):
        """Register an object for checkpointing.

        Args:
            name (str): Name identifier for the checkpointed object.
            obj (object): Object to include in checkpoints.

        Raises:
            ValueError: If the name is already registered.
        """
        if name not in self._extra_save_dict:
            self._extra_save_dict[name] = obj
        else:
            raise ValueError(f"name {name} already registered")

    def _resolve_save_context(self, ckpt_id: str):
        """Resolve a checkpoint path and its filesystem."""
        ckpt_path = os.path.join(self._output_dir, ckpt_id)
        return ckpt_path, get_file_system(ckpt_path)

    def save(
        self,
        ckpt_id: str,
        label_key: Optional[str] = None,
        label_value: Optional[str] = None,
        sync_func: Optional[Callable] = None,
    ):
        """Save a complete checkpoint with the given ID.

        Args:
            ckpt_id (str): Unique identifier for this checkpoint.
            label_key (str): Optional checkpoint label key. Defaults to None.
            label_value (str): Optional checkpoint label value. Defaults to None.
            sync_func (Callable): Function to sync files. Defaults to None.
        """
        del label_key, label_value
        if not sync_func:
            sync_func = get_default_sync_fn(self._shard_num)

        ckpt_path, fs = self._resolve_save_context(ckpt_id)

        logger.info(f"Save checkpoint {ckpt_id} to {ckpt_path}")
        if not fs.exists(ckpt_path):
            fs.makedirs(ckpt_path + "/", exist_ok=True)
        if len(self._sparse_state_dict.keys()) > 0:
            self.save_sparse_params(
                self._shard_id,
                self._shard_num,
                ckpt_path,
                self._sparse_state_dict,
                self._concurrency,
                sync_func,
            )

        # save train and eval io states (flush live iterator positions first)
        for io in self._io_state.values():
            flush_func = getattr(io, "_flush_io_state", None)
            if flush_func is not None:
                flush_func()

        io_states = {}
        for io_name, io in self._io_state.items():
            io_states[io_name] = io.dump_io_state()
        if io_states:
            with fs.open(
                os.path.join(ckpt_path, f"io_state_{self._shard_id}.pt"), "wb"
            ) as f:
                torch.save(io_states, f=f)

        if self._shard_id == 0:
            self._save_rank0_states(ckpt_path, fs, io_states)
            self._update_ckpt_index(ckpt_id, ckpt_path, fs)
            if len(self._checkpoint_version_list) > self._max_keep:
                self._evict_old_ckpt(self._checkpoint_version_list[0], ckpt_path, fs)
        # Checkpointing also supports CPU-only jobs. Avoid triggering CUDA lazy
        # initialization when no CUDA work has been performed in this process.
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        sync_func()

    def save_sparse_params(
        self,
        shard_id: int,
        shard_num: int,
        ckpt_path: str,
        sparse_state_dict: OrderedDict,
        concurrent: int = 16,
        sync_func: Optional[Callable] = None,
    ):
        """Save sparse parameters using distributed saving.

        Args:
            shard_id (int): Current shard ID.
            shard_num (int): Total number of shards.
            ckpt_path (str): Path to save checkpoint.
            sparse_state_dict (OrderedDict): Sparse parameters to save.
            concurrent (int): Number of concurrent save operations. Defaults to 16.
            sync_func (Optional[Callable]): Synchronization function for distributed saving.
        """
        if not sync_func:
            sync_func = get_default_sync_fn(shard_num)
        sparse_state_dict_copy = sparse_state_dict.copy()
        sparse_state_dict, dense_state_dict = split_sparse_dense_state_dict(
            sparse_state_dict_copy
        )
        saver = SSaver(
            shard_index=shard_id,
            shard_num=shard_num,
            parallel=concurrent,
            hashtables=sparse_state_dict,
            tensors=dense_state_dict,
            path=ckpt_path,
            filter_func=self._sparse_filter_fn,
        )
        saver.save()
        sync_func()

    def save_sparse_meta(self, dirname: str):
        """Save sparse parameter metadata to index file.

        Args:
            dirname (str): Directory containing sparse parameter files.
        """
        fs = get_file_system(dirname)
        with fs.open(os.path.join(dirname, "index"), "w") as out_f:
            for filename in fs.listdir(dirname, detail=False):
                if filename.endswith(self.kIndexSuffix):
                    with fs.open(filename, "r") as inf:
                        out_f.write(inf.read())
                    fs.delete(filename)

    def _save_generic(self, value):
        return value.state_dict() if hasattr(value, "state_dict") else value

    def save_dense_params(
        self,
        ckpt_path: str,
        dense_state_dict: OrderedDict,
        fs=None,
    ):
        """Save dense model parameters.

        Args:
            ckpt_path (str): Path to save checkpoint.
            dense_state_dict (dict): Dense parameters to save.
            fs: Optional filesystem instance. If None, will be resolved from
                ckpt_path via get_file_system(). Pass an explicit instance when
                the checkpoint path uses a custom filesystem protocol.
        """
        if fs is None:
            fs = get_file_system(ckpt_path)
        pt_file = os.path.join(ckpt_path, "model.pt")
        with fs.open(pt_file, "wb") as f:
            torch.save(dense_state_dict, f=f)

        self._save_dense_meta(fs, ckpt_path, dense_state_dict)

    def _save_dense_meta(
        self,
        fs,
        ckpt_path: str,
        dense_state_dict: OrderedDict,
        meta_file: str = "torch_rank_weights_embs_table_multi_shard.json",
    ):
        meta_file_path = os.path.join(ckpt_path, meta_file)
        data = {}
        for name, tensor in dense_state_dict.items():
            if isinstance(tensor, torch.Tensor):
                shape_list = [int(dim) for dim in tensor.shape]
                value = {}
                value["name"] = name
                value["dense"] = True
                value["dimension"] = 0
                value["is_hashmap"] = False
                value["dtype"] = str(tensor.dtype).replace("torch.", "")
                value["shape"] = shape_list
                data[name] = value
            else:
                logger.warning(
                    f"{name} is not torch.Tensor in dense_state_dict, will not be saved to torch_rank_weights_embs_table_multi_shard.json"
                )

        existing_data = {}
        if not fs.exists(meta_file_path):
            logger.warning(
                f"Meta file {meta_file_path} not found after saving sparse params"
            )
        else:
            with fs.open(meta_file_path, "r") as f:
                existing_data = json.load(f)
        existing_data.update(data)
        json_str = json.dumps(existing_data, indent=4)
        with fs.open(meta_file_path, "w") as out_f:
            out_f.write(json_str)  # 1次 write，全部内容

    def _save_rank0_states(self, ckpt_path: str, fs, io_states: dict):
        """rank-0 专属的状态落盘: 补空索引 + dense 参数 + extra 参数.

        - sparse 为空时补写空的 index / tensorkey.json, 保证后续 load 不报错.
        - 保存 dense 参数到 model.pt.
        - 保存 extra 参数 (optimizer / global_step 等) 到 extra.pt,
          并写 io_state_count 供 load 时校验 shard 数.
        """
        if not fs.exists(os.path.join(ckpt_path, "index")):
            logger.warning("Sparse params is empty!")
            empty_index = {}
            empty_index["file_index"] = {}
            empty_index["block_index"] = {}
            with fs.open(os.path.join(ckpt_path, "index"), "w") as f:
                json.dump(empty_index, f, indent=4)

            tensorkey_json = {}
            with fs.open(os.path.join(ckpt_path, "tensorkey.json"), "w") as f:
                json.dump(tensorkey_json, f, indent=4)

        if len(self._dense_state_dict.keys()) > 0:
            self.save_dense_params(ckpt_path, self._dense_state_dict, fs=fs)
        if len(self._extra_save_dict.keys()) > 0:
            extra_save = {}
            for key, value in self._extra_save_dict.items():
                if key == ExtraFields.recis_dense_optim:
                    extra_save[key] = value.state_dict()
                else:
                    extra_save[key] = self._save_generic(value)
            with fs.open(os.path.join(ckpt_path, "extra.pt"), "wb") as f:
                torch.save(extra_save, f=f)
            if io_states:
                with fs.open(os.path.join(ckpt_path, "io_state_count"), "w") as f:
                    f.write(f"{self._shard_num}")

    def _update_ckpt_index(self, ckpt_id: str, ckpt_path: str, fs):
        """Append a checkpoint ID to the local checkpoint index."""
        del ckpt_path
        checkpoint_data = ckpt_id + "\n"
        index_path = os.path.join(self._output_dir, self._checkpoint_file)
        if fs.exists(index_path):
            with fs.open(index_path, "r") as out_f:
                checkpoint_data = out_f.read() + ckpt_id + "\n"

        with fs.open(index_path, "w") as out_f:
            out_f.write(checkpoint_data)

        self._checkpoint_version_list.append(ckpt_id)

    def _evict_old_ckpt(self, ckpt_id_to_remove: str, ckpt_path: str, fs):
        """Remove an old checkpoint and update the checkpoint index."""
        del ckpt_path
        old_path = os.path.join(self._output_dir, ckpt_id_to_remove + "/")
        logger.info(f"Remove checkpoint {old_path}")
        try:
            fs.rm(old_path, recursive=True)
        except FileNotFoundError:
            logger.warning(f"Checkpoint path already removed: {old_path}")

        index_path = os.path.join(self._output_dir, self._checkpoint_file)
        with fs.open(index_path, "r") as in_f:
            checkpoint_ids = [
                line.strip() for line in in_f.read().splitlines() if line.strip()
            ]
        with fs.open(index_path, "w") as out_f:
            for checkpoint_id in checkpoint_ids:
                if checkpoint_id != ckpt_id_to_remove:
                    out_f.write(checkpoint_id + "\n")
        self._checkpoint_version_list = self._checkpoint_version_list[1:]

    def _load_sparse_model(self, ckpt_dir: str, model_bank_conf: dict):
        """Load sparse parameters from checkpoint.

        Args:
            ckpt_dir (str): Directory containing the checkpoint.
            model_bank_conf (dict): Model bank config.
        """
        sparse_state_copy = self._sparse_state_dict.copy()
        sparse_state_dict, dense_state_dict = split_sparse_dense_state_dict(
            sparse_state_copy
        )

        filter_func = partial(filter_bank, model_bank_conf)

        loader = SLoader(
            ckpt_dir,
            hashtables=sparse_state_dict,
            tensors=dense_state_dict,
            filter_func=filter_func,
        )
        logger.info(f"load sparse model from checkpoint {ckpt_dir}")
        loader.load()

    def _load_dense_model(self, ckpt_dir: str, model_bank_conf: dict) -> set[str]:
        """Load dense model parameters from checkpoint.

        Args:
            ckpt_dir (str): Directory containing the checkpoint.
            strict (bool): Whether to strictly enforce state dict keys match. Defaults to True.
        """
        if len(model_bank_conf) == 0:
            return set()
        state_dict_loaded, from_pickle = load_pt_file(ckpt_dir, "model")
        if from_pickle:
            state_dict_loaded = pickle_to_torch(state_dict_loaded)

        if len(state_dict_loaded) == 0:
            logger.warning(f"No dense model found in {ckpt_dir}")
            return set()

        filter_dict = {}
        for k in model_bank_conf.keys():
            if MBC.ONAME in model_bank_conf[k]:
                source_name = model_bank_conf[k][MBC.ONAME]
            else:
                source_name = next(
                    (
                        name
                        for name in (k, *self._dense_name_aliases.get(k, ()))
                        if name in state_dict_loaded
                    ),
                    None,
                )

            if source_name is None or source_name not in state_dict_loaded:
                logger.warning(f"No dense model found dst, for {source_name or k}")
                continue

            runtime_name = self._dense_name_to_runtime.get(k, k)
            value = state_dict_loaded[source_name]
            if runtime_name in filter_dict and not torch.equal(
                filter_dict[runtime_name], value
            ):
                raise RuntimeError(
                    "Inconsistent filter global steps for coalesced hashtable "
                    f"children: {runtime_name}"
                )
            filter_dict[runtime_name] = value

        if len(filter_dict) != 0:
            logger.info(f"Load dense model from checkpoint {ckpt_dir}")
            missing, unexpected = self._model.load_state_dict(filter_dict, strict=False)
            if len(missing) > 0:
                logger.warning(f"Missing keys in dense model: {missing}")
            if len(unexpected) > 0:
                logger.warning(f"Unexpected keys in dense model: {unexpected}")
            return {
                i
                for i, _ in self._model.named_parameters()
                if i not in set(missing) and i not in set(unexpected)
            }
        else:
            logger.info("No dense model to load")

        return set()

    @property
    def model(self):
        return self._model

    def _load_extra_params(
        self,
        ckpt_dir: str,
        model_bank_conf: dict,
        dense_optim_args: dict,
        shared_id: int = 0,
    ):
        """Load extra parameters and IO states from checkpoint.

        Args:
            ckpt_dir (str): Directory containing the checkpoint.
            model_bank_conf (dict): Model bank config.
            shared_id (int): Shard ID for loading IO states. Defaults to 0.
        """
        fs = get_file_system(os.path.join(ckpt_dir, "index"))

        if (
            ExtraFields.train_io in model_bank_conf
            and ExtraFields.eval_io in model_bank_conf
        ):
            with fs.open(os.path.join(ckpt_dir, "io_state_count"), "r") as f:
                shard_num = int(f.read())
            with fs.open(os.path.join(ckpt_dir, f"io_state_{shared_id}.pt"), "rb") as f:
                io_state = torch.load(f=f, weights_only=False)
            for io_name, io in self._io_state.items():
                assert shard_num == io._worker_num, (
                    f"IO states size not equal to worker num, expect: {io._worker_num}, got: {shard_num}"
                )
                if io_name in io_state:
                    logger.info(f"Load io state for dataset: {io_name}")
                    io.load_io_state(io_state[io_name])
                else:
                    logger.warning(f"No io state found for dataset: {io_name}")
        else:
            logger.info("Skip loading io_state because it is not in model bank config")

        extra_data, from_pickle = load_pt_file(ckpt_dir, "extra")
        if from_pickle:
            extra_data = pickle_to_torch(extra_data)
            if ExtraFields.recis_dense_optim in extra_data:
                extra_data[ExtraFields.recis_dense_optim]["param_groups"] = (
                    self._extra_save_dict[ExtraFields.recis_dense_optim].state_dict()[
                        "param_groups"
                    ]
                )

        if len(extra_data) == 0:
            logger.warning(f"No extra data found in {ckpt_dir}")
            return

        if ExtraFields.prev_optim in extra_data:
            extra_data[ExtraFields.recis_dense_optim] = extra_data.pop(
                ExtraFields.prev_optim
            )

        logger.info(f"load extra params from checkpoint {ckpt_dir}")
        for key, value in self._extra_save_dict.items():
            if key not in model_bank_conf:
                logger.info(
                    f"Skip loading {key} because it is not in model bank config"
                )
                continue

            if key not in extra_data:
                logger.info(f"No {key} found in {ckpt_dir} when load extra params")
                continue

            data = extra_data[key]
            if hasattr(value, "load_state_dict"):
                if hasattr(value, "named_optimizer") and value.named_optimizer:
                    # for accelerate named optimizer
                    if hasattr(value, "optimizer"):
                        value.optimizer.load_state_dict(data, **dense_optim_args)
                    else:
                        value.load_state_dict(data, **dense_optim_args)
                    logger.warning("dense optimizer param group info:")
                    for pg in value.param_groups:
                        logger.warning(
                            json.dumps(
                                {k: v for k, v in pg.items() if k != "params"}, indent=4
                            )
                        )
                else:
                    value.load_state_dict(data)
                    if isinstance(value, torch.optim.Optimizer):
                        logger.warning(
                            f"Load dense optimizer from {ckpt_dir} may cause error, please upgrade to PyTorch>=2.6.0 and use named optimizer"
                        )
            elif isinstance(value, torch.Tensor):
                value.copy_(data)
            else:
                value = data

            logger.info(f"load {key} from ckpt {ckpt_dir}'s extra_data")
            self._extra_save_dict[key] = value

    def load(
        self,
        ckpt_path: Optional[str] = None,
        ckpt_id: Optional[str] = None,
        direct_path=False,
        model_bank_conf: Optional[dict] = None,
    ):
        """Load a checkpoint from an explicit path or the checkpoint index."""
        if model_bank_conf is None:
            model_bank_conf = {}
        if direct_path:
            if not ckpt_path:
                return
            logger.info(f"Load checkpoint from explicit path: {ckpt_path}")
        else:
            ckpt_path = self._output_dir if not ckpt_path else ckpt_path
            fs = get_file_system(ckpt_path)
            if ckpt_id is None:
                index_path = os.path.join(ckpt_path, self._checkpoint_file)
                if not fs.exists(index_path):
                    logger.warning(f"Checkpoint index not found under {ckpt_path}")
                    return
                with fs.open(index_path, "r") as index_file:
                    checkpoint_ids = [
                        line.strip()
                        for line in index_file.read().splitlines()
                        if line.strip()
                    ]
                if not checkpoint_ids:
                    logger.warning(f"Checkpoint index is empty under {ckpt_path}")
                    return
                ckpt_id = checkpoint_ids[-1]
            logger.info(f"Load checkpoint from {ckpt_path} (ckpt_id={ckpt_id})")
            ckpt_path = os.path.join(ckpt_path, ckpt_id)
        self.load_by_config(ckpt_path, self._shard_id, model_bank_conf)

    def _convert_valid_names(self, valid_names, model, optimizer):
        """
        convert valid model names to optimizer param names
        """
        if optimizer is None:
            logger.warning("No dense optimizer registered, return empty set")
            return set()

        model_dict = dict(model.named_parameters())
        optim_dict = {}

        for group in optimizer.param_groups:
            if "param_names" not in group:
                msg = ", ".join(
                    [
                        "No param_names found in optimizer param groups",
                        "this may cause error when load dense optimizer",
                        "please upgrade to PyTorch>=2.6.0 and use wrapped_named_optimizer.",
                    ]
                )
                logger.warning(msg)
                return valid_names
            optim_dict.update(dict(zip(group["params"], group["param_names"])))

        res = set()
        for name in valid_names:
            res.add(optim_dict[model_dict[name]])
        return res

    def load_by_config(
        self,
        ckpt_path: str,
        shared_id: int = 0,
        model_bank_conf: Optional[dict] = None,
    ):
        if model_bank_conf is None:
            model_bank_conf = {}
        assert len(model_bank_conf) > 0, "Model bank config is empty"

        sparse_model_bank = {
            k: v for k, v in model_bank_conf.items() if k in self._sparse_names
        }
        self._load_sparse_model(ckpt_path, sparse_model_bank)

        dense_model_bank = {
            k: v for k, v in model_bank_conf.items() if k in self._dense_names
        }

        valid_dense_names = self._convert_valid_names(
            self._load_dense_model(ckpt_path, dense_model_bank),
            self._model,
            self._extra_save_dict.get(ExtraFields.recis_dense_optim, None),
        )

        load_map = {
            k: v[MBC.ONAME]
            for k, v in model_bank_conf.items()
            if MBC.ONAME in v and k in self._dense_names
        }
        strict = not next(iter(model_bank_conf.values())).get(MBC.IGNORE_ERROR, True)
        dense_optim_args = {
            "valid_names": valid_dense_names,
            "load_map": load_map,
            "strict": strict,
        }

        extra_set = set(self._extra_save_dict.keys())
        extra_set.update(ExtraFields.get_io_fields())
        extra_model_bank = {k: v for k, v in model_bank_conf.items() if k in extra_set}
        self._load_extra_params(
            ckpt_path, extra_model_bank, dense_optim_args, shared_id
        )

    def get_extra_data(self, name: str):
        if name in self._extra_save_dict:
            return self._extra_save_dict[name]
        else:
            return None

    def _clear_hashtables_if_needed(self, var_config_dict: dict):
        """Clear hashtables for variables that require it."""
        cleared = set()
        for var_name, var_config in var_config_dict.items():
            if var_config.get("hashtable_clear", False):
                sparse_params = filter_out_sparse_param(self._model)
                for hashtable_obj in sparse_params.values():
                    for child_name in hashtable_obj.children_info().children():
                        if (
                            var_name.startswith(child_name)
                            or var_name.replace("@*", "") == child_name
                        ) and child_name not in cleared:
                            logger.warning(f"Clearing hashtable {child_name}")
                            hashtable_obj.clear(child_name)
                            cleared.add(child_name)

    def _load_variables(self, model_bank: dict):
        for path, vars in model_bank.items():
            ckpt_path = get_update_path(path)
            if ckpt_path == "":
                raise ValueError(f"No update path found in {path}")

            # Create model_bank_conf for only vars
            var_config_dict = {}
            for var_name in vars:
                var_config_dict[var_name] = vars[var_name]
            # Clear hashtables if needed
            self._clear_hashtables_if_needed(var_config_dict)

            self.load(
                ckpt_path=ckpt_path,
                model_bank_conf=var_config_dict,
                direct_path=True,
            )

    def update_load(self):
        if self._has_bank:
            if len(self._dynamic_model_bank) > 0:
                logger.info("Starting update_load")
                self._load_variables(self._dynamic_model_bank)
                return
        logger.info("No dynamic model bank provided, skip load model")

    def restore(self):
        if self._has_bank:
            if len(self._all_model_bank) > 0:
                logger.info("Starting init_reload")
                self._load_variables(self._all_model_bank)
                return
        logger.info("No model bank provided, skip load model")
