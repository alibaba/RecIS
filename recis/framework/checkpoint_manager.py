import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional

import torch

from recis.framework.filesystem import get_file_system
from recis.framework.model_bank import (
    MBC,
    ModelBankParser,
    get_update_path,
    load_pt_file,
    pickle_to_torch,
    show_model_bank_format,
)
from recis.info import is_internal_enabled
from recis.nn.modules.hashtable import (
    filter_out_sparse_param,
    split_sparse_dense_state_dict,
)
from recis.optim.sparse_optim import SparseOptimizer
from recis.serialize import Loader as SLoader, Saver as SSaver
from recis.utils.logger import Logger
from recis.utils.openlm_hub_helper import (
    MOS_URI_PREFIX,
    OPENLM_HUB_CKPT_AVAILABLE,
    OpenlmHubHelper,
)


if is_internal_enabled() and not os.environ.get("BUILD_DOCUMENT", None) == "1":
    from pangudfs_client.common.exception.exceptions import PanguException

    from recis.utils.mos import Mos
else:
    PanguException = None
    Mos = None

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
        self._sparse_state_dict, self._dense_state_dict = split_sparse_dense_state_dict(
            self._model.state_dict()
        )
        self._checkpoint_file = "checkpoint"
        self._checkpoint_version_list = []
        self._max_keep = options.max_keep
        self._extra_save_dict = {}

        self._mos = None
        self._output_dir = options.output_dir
        # openlm_hub 模式下 save 时由 _resolve_save_context 赋值,
        # 供 _register_ckpt 使用; 非 openlm_hub 时保持 None.
        self._ckpt_file_manager = None
        self.openlm_hub_helper = None
        if self._output_dir.startswith(MOS_URI_PREFIX):
            assert Mos is not None, "Cannot import mos, check internal version."
            self._mos = Mos(self._output_dir)
            self._output_dir = self._mos.real_physical_path
            if OPENLM_HUB_CKPT_AVAILABLE:
                self.openlm_hub_helper = OpenlmHubHelper(
                    self._mos.version_uri, self._mos.user_id
                )

        # output_dir 是 MOS uri → 自动走标准 openlm_hub ckpt 流程
        self._is_openlm_hub_ckpt = self.openlm_hub_helper is not None

        # 标准 openlm_hub 用法: ckpt 写入路径由 MosCkptFileManager 每次 save 决定,
        # Saver.output_dir 返回 MOS uri 当标识用, 不是文件系统路径.
        if self._is_openlm_hub_ckpt:
            logger.info(
                f"标准 openlm_hub ckpt 模式: Saver.output_dir = "
                f"{self.openlm_hub_helper.version_uri} (MOS uri, not a filesystem path)"
            )

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

    def _maybe_inject_mos_resume_entry(self, model_bank_content):
        """openlm_hub 标准用法下做断点续训 —— 帮任务自动找
        上次存的 ckpt 接着练, 给 model_bank 末尾塞一条续训用的条目.

        老模式: save 时往 ``{output_dir}/checkpoint``
        追加一行 ckpt_id 当索引, 启动时 ``ModelBankParser._complete_model_bank``
        读这个文件取最后一行就是 latest ckpt, 直接续训. 不依赖 openlm_hub / MOS.

        openlm_hub 标准用法 = ckpt 注册和路径分配全交给 MOS, recis 这边不再写
        ``{output_dir}/checkpoint`` 索引文件. 老的索引查找读到空, 不补的话训练每次
        启动都是冷启动. 本方法是这种用法下的替身: 调
        ``MosCkptFileManager(version_uri, mode='r')`` 直接问 MOS 当前 version 下
        最新 ckpt 的物理路径, 拼成一条跟 ``_complete_model_bank`` 等价的条目
        追加到列表末尾. 训练启动时 ModelBankParser 照常处理, 效果等于自动续训.

        Returns:
            list: 传入的 model_bank_content. 命中 MOS 最新 ckpt 时末尾追加一条
            续训条目; 没启用 openlm_hub 标准用法 / 没接 MOS / MOS 查不到时原样返回.
        """
        if not self._is_openlm_hub_ckpt:
            return model_bank_content
        result = self.openlm_hub_helper.resolve_latest_resume()
        if result is None:
            logger.info(
                f"No existing ckpt under {self.openlm_hub_helper.version_uri}; skip auto-resume entry"
            )
            return model_bank_content
        resume_path, ckpt_physical_path = result
        current_app = os.environ.get("HIPPO_APP", "")
        tag = " (cross-app)" if current_app and current_app not in ckpt_physical_path else ""
        logger.info(f"Auto-resume entry resolved via openlm_hub{tag}: {resume_path}")
        # entry schema 对齐 _complete_model_bank, parser 视作老路径查找等价物
        entry = {
            MBC.PATH: resume_path,
            MBC.LOAD: {"*"},
            MBC.EXCLUDE: set(),
            MBC.IS_DYNAMIC: False,
            MBC.HASHTABLE_CLEAR: True,
            MBC.IGNORE_ERROR: True,
            MBC.ONAME: [],
        }
        return list(model_bank_content) + [entry]

    def _init_model_bank(self, model_bank=None):
        model_bank_content = (
            model_bank if model_bank is not None else self._model_bank_content
        )
        model_bank_content = self._maybe_inject_mos_resume_entry(model_bank_content)

        self._check_name_conflict()

        self._model_bank_parser = ModelBankParser(
            self._output_dir,
            model_bank_content,
            self._model_names,
            self._sparse_names,
            self._sparse_tables,
            self._dense_names,
            ExtraFields,
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
        """ckpt 写出位置. 两种语义:

        - 老协议: 真实文件系统路径, ckpt 落在 ``{output_dir}/{ckpt_id}/``,
          可 ``ls`` / ``cd``.
        - 标准 openlm_hub 用法: MOS uri (``model.proj.name/version=xxx``),
          **不是路径** -- 实际写入位置每次 save 由 MosCkptFileManager 决定.
          只能当模型标识用 (log / tracker tag / MOS API), 不要 ``os.path.join``.
        """
        if self._is_openlm_hub_ckpt:
            return self.openlm_hub_helper.version_uri
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        """仅用于 testcase, 部分 testcase 用它把 save 重定向到临时目录。仅在非 MOS 模式下生效
        （那时 getter 返回的是 _output_dir)
        """
        self._output_dir = value

    @property
    def mos(self):
        """:class:`recis.utils.mos.Mos` 客户端, 非 MOS 任务返回 ``None``.

        - ``saver.mos.version_uri``: MOS 上的模型标识
        - ``saver.mos.last_ckpt_id``: 最近一次注册的 ckpt id
        """
        return self._mos

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
        """解析 ckpt 写入路径与文件系统。

        - openlm_hub 模式: rank 0 调用 MosCkptFileManager 获取写入路径并存入
          self._ckpt_file_manager（该字段在 __init__ 中初始化为 None），
          其他 rank 通过 broadcast 同步 ckpt_path 后自行创建文件系统对象。
        - 老协议: 直接拼接 output_dir / ckpt_id。

        Returns:
            tuple: (ckpt_path, fs)
        """
        if self._is_openlm_hub_ckpt:
            # 仅 rank 0 调用 MOS，避免多 rank 独立调 MosCkptFileManager 时因
            # 时序差异（如 pangu 切换）导致各 worker 拿到不同的写入路径。
            if self._shard_id == 0:
                self._ckpt_file_manager, ckpt_path, fs = self.openlm_hub_helper.get_save_context(ckpt_id)
            else:
                ckpt_path = ""
            if self._shard_num > 1:
                obj_list = [ckpt_path]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                ckpt_path = obj_list[0]
            if self._shard_id != 0:
                fs = get_file_system(ckpt_path)
            return ckpt_path, fs
        else:
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

        流程::

            save(ckpt_id)
            |
            +-- 1. 解析路径 + 文件系统
            |   +-- [openlm_hub] rank 0: helper.get_save_context()
            |   |               其他 rank: broadcast 同步 ckpt_path
            |   +-- [老协议]     os.path.join(output_dir, ckpt_id)
            |
            +-- 2. makedirs(ckpt_path)
            |
            +-- 3. save_sparse_params()                 <- all rank
            |
            +-- 4. flush & save io_states               <- all rank
            |
            +-- 5. if shard_id == 0:                        <- rank-0 only
            |   +-- a. _save_rank0_states()   补空索引 / dense / extra 落盘
            |   +-- b. _update_ckpt_index()   写索引文件 + version_list
            |   +-- c. _evict_old_ckpt()      淘汰旧 ckpt (len > max_keep 时)
            |   +-- d. _register_ckpt()       注册 ckpt + 上报 metrics
            |
            +-- 6. cuda.synchronize + sync_func

        Args:
            ckpt_id (str): Unique identifier for this checkpoint.
            label_key (str): Key for the label when saving to MOS. Defaults to None.
            label_value (str): Value for the label when saving to MOS. Defaults to None.
            sync_func (Callable): Function to sync files. Defaults to None.
        """
        if not sync_func:
            sync_func = get_default_sync_fn(self._shard_num)

        ckpt_path, fs = self._resolve_save_context(ckpt_id)

        logger.info(f"Save checkpoint {ckpt_id} to {ckpt_path}")
        if not fs.exists(ckpt_path):
            try:
                fs.makedirs(ckpt_path + "/", exist_ok=True)
            except PanguException as e:
                if e.pangu_err_no == 7:
                    pass
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
                self._evict_old_ckpt(
                    self._checkpoint_version_list[0], ckpt_path, fs
                )
            self._register_ckpt(
                self._ckpt_file_manager, ckpt_id, ckpt_path, label_key, label_value
            )
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
                ckpt_path via get_file_system(). Pass explicitly when using
                MosCkptFileManager to avoid protocol resolution issues.
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
        with fs.open(meta_file_path, "w") as out_f:
            json.dump(existing_data, out_f, indent=4)

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
        """更新 ckpt 版本列表, 老协议下同步写 checkpoint 索引文件.

        - openlm_hub 模式跳过索引文件 (由 MOS register_ckpt 接管);
          老协议追加 ckpt_id 到 checkpoint 索引文件.
        - 追加 version_list; openlm_hub 模式额外缓存 WRITE 路径
          (供后续 _evict_old_ckpt 删文件用).
        """
        if not self._is_openlm_hub_ckpt:
            checkpoint_data = ckpt_id + "\n"
            if fs.exists(os.path.join(self._output_dir, self._checkpoint_file)):
                with fs.open(
                    os.path.join(self._output_dir, self._checkpoint_file), "r"
                ) as out_f:
                    checkpoint_data = out_f.read() + ckpt_id + "\n"

            with fs.open(
                os.path.join(self._output_dir, self._checkpoint_file), "w"
            ) as out_f:
                out_f.write(checkpoint_data)

        self._checkpoint_version_list.append(ckpt_id)
        if self._is_openlm_hub_ckpt:
            self.openlm_hub_helper.cache_write_path(ckpt_id, ckpt_path)

    def _evict_old_ckpt(self, ckpt_id_to_remove: str, ckpt_path: str, fs):
        """淘汰旧 ckpt: 删文件 + 注销 MOS 记录."""
        if self._is_openlm_hub_ckpt:
            old_ckpt_path = self.openlm_hub_helper.pop_write_path(ckpt_id_to_remove)
            logger.info(
                f"Remove checkpoint {ckpt_id_to_remove}: {old_ckpt_path}"
            )
            if old_ckpt_path is not None:
                fs.rm(old_ckpt_path + "/", recursive=True)
        else:
            logger.info(
                f"Remove checkpoint {os.path.join(self._output_dir, ckpt_id_to_remove)}"
            )
            fs.rm(
                os.path.join(self._output_dir, ckpt_id_to_remove + "/"),
                recursive=True,
            )
            remains = []
            with fs.open(
                os.path.join(self._output_dir, self._checkpoint_file), "r"
            ) as f:
                lines = [
                    line.strip()
                    for line in f.read().split("\n")
                    if len(line.strip()) != 0
                ]
                for ckpt_id in lines:
                    if ckpt_id != ckpt_id_to_remove:
                        remains.append(ckpt_id)
            with fs.open(
                os.path.join(self._output_dir, self._checkpoint_file), "w"
            ) as f:
                for ckpt_id in remains:
                    f.write(ckpt_id + "\n")
        self._checkpoint_version_list = self._checkpoint_version_list[1:]
        if self._is_openlm_hub_ckpt:
            self.openlm_hub_helper.delete(ckpt_id_to_remove)
        elif self._mos:
            self._mos.ckpt_update(
                ckpt_id=ckpt_id_to_remove, path=ckpt_path, is_delete=True
            )

    def _register_ckpt(
        self,
        ckpt_file_manager,
        ckpt_id: str,
        ckpt_path: str,
        label_key: Optional[str],
        label_value: Optional[str],
    ):
        """注册 ckpt 到 MOS 并上报 metrics.

        Args:
            ckpt_file_manager: MosCkptFileManager 对象 (openlm_hub 模式下
                self._ckpt_file_manager 的值), 老协议时传 None。
            ckpt_id: checkpoint 标识。
            ckpt_path: checkpoint 物理路径。
            label_key: MOS label key。
            label_value: MOS label value。
        """
        if self.openlm_hub_helper and ckpt_file_manager is not None:
            ckpt_labels = []
            if label_key is not None and label_value is not None:
                ckpt_labels.append(f"{label_key}={label_value}")
            self.openlm_hub_helper.register_and_report(
                ckpt_file_manager, ckpt_id, labels=ckpt_labels
            )
            self._mos.last_ckpt_id = ckpt_id
        elif self._mos:
            self._mos.ckpt_update(
                ckpt_id=ckpt_id,
                path=ckpt_path,
                label_key=label_key,
                label_value=label_value,
            )

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
                oname = model_bank_conf[k][MBC.ONAME]
                if oname in state_dict_loaded:
                    filter_dict[k] = state_dict_loaded[oname]
                else:
                    logger.warning(f"[oname] No dense model found dst, for {oname}")
            else:
                filter_dict[k] = state_dict_loaded[k]

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
        """根据传入的入参组合决定从哪里加载 ckpt, 真值表:

        +----------------------------------------+----------+-------------------------------+
        | usage                                  | branch   | resolved ckpt_path            |
        +----------------------------------------+----------+-------------------------------+
        | load()                                 | by mode  | see below                     |
        +----------------------------------------+----------+-------------------------------+
        | load(ckpt_id="ckpt-100")               | by mode  | see below                     |
        +----------------------------------------+----------+-------------------------------+
        | load(ckpt_path="/x/y/")                | literal  | "/x/y/"                       |
        +----------------------------------------+----------+-------------------------------+
        | load(ckpt_path="/x/y/", direct_path=1) | literal  | "/x/y/" (same as above)       |
        +----------------------------------------+----------+-------------------------------+

        "by mode" 进一步分两种：
        - 标准 openlm_hub 用法：走 MOS 查
            - ckpt_id=None    → MosCkptFileManager(version_uri, "r") 拿最新
            - ckpt_id="xxx"   → MosCkptFileManager(version_uri/ckpt_id=xxx, "r")
        - 老协议：读 {output_dir}/checkpoint 索引文件
            - ckpt_id=None    → 取索引最后一行
            - ckpt_id="xxx"   → 直接拼 {output_dir}/{ckpt_id}/

        关键点：**只要 caller 传了非空 ckpt_path, 就一律按字面路径加载**，不会
        被 MOS 查询或索引文件覆盖。这样 caller 的"我要加载这个具体路径"意图
        永远不会被框架静默改写。
        """
        if model_bank_conf is None:
            model_bank_conf = {}
        if direct_path or (self._is_openlm_hub_ckpt and ckpt_path):
            # 传了ckpt_path 直接用, 不走 MOS 查询也不查索引文件.
            # openlm_hub 模式下也一样: 显式路径优先于自动解析.
            if not ckpt_path:
                return
            logger.info(f"Load ckpt from literal path: {ckpt_path}")
        elif self._is_openlm_hub_ckpt:
            ckpt_path = self.openlm_hub_helper.resolve_load_path(ckpt_id)
            if ckpt_path is None:
                return
            logger.info(f"Load ckpt resolved via openlm_hub: {ckpt_path}")
        else:
            ckpt_path = self._output_dir if not ckpt_path else ckpt_path
            fs = get_file_system(ckpt_path)
            if ckpt_id is None:
                if fs.exists(os.path.join(ckpt_path, self._checkpoint_file)):
                    content = fs.open(
                        os.path.join(ckpt_path, self._checkpoint_file), "r"
                    ).read()
                    lines = content.split("\n")[::-1]
                    ckpt_id = None
                    for line in lines:
                        if len(line) == 0:
                            continue
                        ckpt_id = line.strip()
                        break
                else:
                    logger.warning(f"Checkpoint index not found under {ckpt_path}")
                    return
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
