"""GpuShardSampler: GPU 分片负样本采样器。

接口与 recis 的 LocalRpcDataSampler 完全一致，可作为 drop-in 替换。
在 io_utils.py 的 generate_batch 中通过配置开关选择。

核心区别:
    LocalRpcDataSampler: 单进程 sampler server, CPU 哈希表 + Pack gather, RPC 通信
    GpuShardSampler:     每 rank 本地 GPU 分片, fused CUDA kernels + NCCL all_to_all
"""

import logging
import threading
from typing import Optional

import torch
import torch.distributed as dist

from recis.data.gpu_shard_sampler.comm import (
    all_gather_sorted_ids,
    shard_pack_feature_unified,
)
from recis.data.gpu_shard_sampler.shard import GpuShard, build_cpu_shard


logger = logging.getLogger(__name__)


class GpuShardSampler:
    """GPU 分片负样本采样器。

    将负样本表分片到多张 GPU。
    查找时通过 4 轮 NCCL all_to_all 跨 rank 通信，本地用 GPU gather。

    生命周期:
        1. __init__: 读取配置
        2. prepare_reload(batch): CPU 阶段，可异步（排序、分类、CSR）
        3. commit_reload(): GPU 阶段，主线程同步（to_device + all_gather + default_value 校验 + swap）
        4. reload(batch): 同步便捷函数 = prepare_reload + commit_reload
        5. pack_feature(ids): fused_route_ids + 4 轮 all_to_all + 本地 GPU gather
        6. combine(batch, output, sample_cnts): 复用现有 C++ 算子
    """

    def __init__(
        self,
        cfg,
        local_rank: int,
        world_size: int,
        device: str,
        group: Optional[dist.ProcessGroup] = None,
        default_value: int = -1,
    ):
        self.rank = local_rank
        self.world_size = world_size
        self.device = device
        self.group = group if group is not None else dist.group.WORLD
        self.cfg = cfg
        self.default_value = default_value

        self.shard: Optional[GpuShard] = None
        self.all_sorted_ids: list = []
        self.sorted_ptrs = None
        self.sorted_sizes = None
        self._initialized = False
        self._checked_defaults = set()

        self._reload_lock = threading.Lock()
        self._pending_shard: Optional[GpuShard] = None

        logger.info(
            f"GpuShardSampler initialized: rank={local_rank}, "
            f"world_size={world_size}, device={device}, default_value={default_value}"
        )

    def prepare_reload(self, batch):
        """Phase 1 (async safe): ODPS 数据 → CPU shard。

        在后台线程中调用，完成排序、分类、CSR 构建，不涉及 GPU 或 NCCL。
        产物存储在 self._pending_shard，等待 commit_reload 消费。
        """
        feature_table = batch[0] if isinstance(batch, list) else batch
        logger.info(
            f"Rank {self.rank}: preparing CPU shard from {len(feature_table)} features..."
        )
        self._pending_shard = build_cpu_shard(feature_table, self.rank)
        logger.info(f"Rank {self.rank}: CPU shard prepared: {self._pending_shard}")

    def commit_reload(self):
        """Phase 2 (main thread only): CPU shard → GPU + all_gather + default_value 校验 + 原子交换。

        必须在主线程同步调用（含 NCCL all_gather 集体操作）。
        通过 _reload_lock 保证 pack_feature 期间的并发安全。
        """
        if self._pending_shard is None:
            return

        new_shard = self._pending_shard.to_device(self.device)
        logger.info(f"Rank {self.rank}: shard moved to GPU: {new_shard}")

        new_all_sorted_ids = all_gather_sorted_ids(
            new_shard.sorted_ids, self.group, self.device
        )

        # 预计算 sorted_ptrs/sorted_sizes（供 fused_route_ids kernel 使用）
        new_sorted_ptrs = torch.tensor(
            [s.data_ptr() for s in new_all_sorted_ids],
            dtype=torch.int64,
            device=self.device,
        )
        new_sorted_sizes = torch.tensor(
            [s.numel() for s in new_all_sorted_ids],
            dtype=torch.int32,
            device=self.device,
        )

        with self._reload_lock:
            self.shard = new_shard
            self.all_sorted_ids = new_all_sorted_ids
            self.sorted_ptrs = new_sorted_ptrs
            self.sorted_sizes = new_sorted_sizes

        self._pending_shard = None
        self._initialized = True
        logger.info(
            f"Rank {self.rank}: GpuShardSampler ready, all_gathered {len(self.all_sorted_ids)} sorted_ids"
        )

    def reload(self, batch: dict):
        """同步 reload（首次加载用）。等价于 prepare_reload + commit_reload。"""
        self.prepare_reload(batch)
        self.commit_reload()

    def valid_sample_ids(
        self,
        ids: torch.Tensor,
        default_value: int = -1,
        all_sorted_ids: Optional[list] = None,
    ) -> torch.Tensor:
        """验证 sample IDs，未命中的替换为 default_value。"""
        if not self._initialized:
            raise RuntimeError("GpuShardSampler not initialized, call reload() first")

        sorted_ids = (
            all_sorted_ids if all_sorted_ids is not None else self.all_sorted_ids
        )
        ids = ids.to(self.device)

        from recis.lib import gpu_shard_kernels as _kernels

        if not _kernels.HAS_CUDA_KERNEL:
            raise RuntimeError(
                "CUDA kernel not available, fused_valid_sample_ids requires CUDA support"
            )
        return _kernels.fused_valid_sample_ids(ids, sorted_ids, default_value)

    def pack_feature(
        self,
        local_data_sample_ids: torch.Tensor,
        default_value: int = -1,
    ) -> dict:
        """Gather 负样本特征，通过 fused_route_ids + 4 轮 all_to_all 跨 rank 通信。

        先调用 fused_route_ids 将不存在的 ID 替换为 default_value（一次 kernel 完成路由 + 替换），
        然后传预计算的 target_ranks 给 shard_pack_feature_unified。

        Args:
            local_data_sample_ids: [Q] int64, 待查询的 unit_id 列表。
            default_value: int, 不存在的 ID 替换为此值。

        Returns:
            dict[str, Tensor | RaggedTensor]: 特征名 → gather 结果。
        """
        if not self._initialized:
            raise RuntimeError("GpuShardSampler not initialized, call reload() first")

        with self._reload_lock:
            shard = self.shard
            sorted_ptrs = self.sorted_ptrs
            sorted_sizes = self.sorted_sizes

        ids = local_data_sample_ids.to(self.device)

        # One-time validation: check default_value exists in some rank's sorted_ids
        if default_value not in self._checked_defaults:
            with self._reload_lock:
                all_sorted_ids = self.all_sorted_ids
            found_default = False
            for sorted_ids_r in all_sorted_ids:
                if sorted_ids_r.numel() == 0:
                    continue
                pos = torch.searchsorted(sorted_ids_r, default_value)
                pos_clamped = pos.clamp(max=sorted_ids_r.numel() - 1)
                if sorted_ids_r[pos_clamped] == default_value:
                    found_default = True
                    break
            if not found_default:
                raise ValueError(
                    f"default_value={default_value} not found in any rank's sorted_ids. "
                    "Ensure the negative sample table contains the default_value ID."
                )
            self._checked_defaults.add(default_value)

        # 一次 kernel 完成路由 + 替换（fused_route_ids）
        from recis.lib import gpu_shard_kernels as _kernels

        if not _kernels.HAS_CUDA_KERNEL:
            raise RuntimeError(
                "CUDA kernel not available, fused_route_ids requires CUDA support"
            )
        valid_ids, target_ranks, _found_mask = _kernels.fused_route_ids(
            ids,
            sorted_ptrs,
            sorted_sizes,
            default_value,
            self.world_size,
        )

        # 传预计算的目标 rank 给 shard_pack_feature_unified，不再二次跨 rank 路由
        results = shard_pack_feature_unified(
            valid_ids,
            target_ranks,
            shard,
            self.group,
            self.device,
        )
        return results

    def combine(
        self,
        batch: list,
        output_sample_table: dict,
        sample_cnts: torch.Tensor,
    ) -> list:
        """合并正样本和负样本特征。

        直接实现 LocalRpcDataSampler.combine 的逻辑，
        不依赖 LocalRpcDataSampler 实例。
        """
        group_id_t = []
        indicators_t = []
        indicators_name = []
        do_classify = False
        for dic in batch:
            for name in dic:
                data = dic[name]
                if name.startswith("_indicator"):
                    indicators_name.append(name)
                    assert isinstance(data, torch.Tensor)
                    indicators_t.append(data)
                elif name == "_sample_group_id":
                    assert isinstance(data, torch.Tensor)
                    group_id_t.append(data)
                    do_classify = True
        input_group_indicators_t = group_id_t + indicators_t
        # repeat_interleave(input, sample_cnts + 1) = tile_with_sample_counts(input, sample_cnts)
        repeats = sample_cnts + 1
        output_group_indicators_t = [
            torch.repeat_interleave(inp, repeats, dim=0)
            for inp in input_group_indicators_t
        ]
        idx = 0
        if do_classify:
            output_sample_table["_sample_group_id"] = output_group_indicators_t[idx]
            idx += 1
        for dic in batch[1:]:
            indicator_name = indicators_name.pop(0)
            dic[indicator_name] = output_group_indicators_t[idx]
            idx += 1
        output_batch = [output_sample_table]
        output_batch.extend(batch[1:])
        return output_batch

    def combine_vector_with_sample_counts(
        self,
        origin_vector: torch.Tensor,
        sample_counts: torch.Tensor,
        sampled_vector: torch.Tensor,
    ) -> torch.Tensor:
        """组合正负样本 ID（复用 C++ 算子）。"""
        return torch.ops.recis.combine_vector_with_sample_counts(
            origin_vector, sample_counts, sampled_vector
        )
