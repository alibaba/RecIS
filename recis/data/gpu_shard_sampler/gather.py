"""Unified本地 gather 逻辑。"""

import torch

from recis.data.gpu_shard_sampler.shard import GpuShard


__all__ = ["unified_gather"]


def _local_lookup(
    shard: GpuShard,
    query_ids: torch.Tensor,
) -> tuple:
    """在本地分片中查找 query_ids，返回行索引和命中掩码。

    使用 torch.searchsorted 做 binary search：3.68M 项 ≈ 22 步，
    47K query 并行执行，耗时 <0.1ms。比 GPU hash table (cuco) 更简单，
    无需编译额外 CUDA 扩展。

    Args:
        shard: 本 rank 的 GPU 分片。
        query_ids: [Q] int64, 待查找的 unit_id。

    Returns:
        Tuple of:
            - positions: [Q] int64, query_id 在本分片中的行索引
              （未命中的位置值无意义，需配合 found_mask 使用）。
            - found_mask: [Q] bool, 是否在本分片中找到。
    """
    positions = torch.searchsorted(shard.sorted_ids, query_ids)
    positions_clamped = positions.clamp(max=shard.num_items - 1)
    found_mask = shard.sorted_ids[positions_clamped] == query_ids
    return positions, found_mask


def unified_gather(
    shard: GpuShard,
    recv_ids: torch.Tensor,
) -> tuple:
    """Unified local gather: single CUDA kernel for all entries.

    Receiver-side searchsorted to get row indices (no sender-side indices needed).
    Single kernel handles dense + ragged + weights using flat_bytes abstraction.

    Args:
        shard: Local GpuShard with data on GPU.
        recv_ids: [total_recv] int64, query IDs received from other ranks.

    Returns:
        Tuple of:
            - all_lengths: [E, total_recv] int64, length per item per entry.
            - all_values_list: list[Tensor], gathered values per entry (flat).
    """
    positions, _ = _local_lookup(shard, recv_ids)
    safe_indices = positions

    try:
        from recis.lib import gpu_shard_kernels as _kernels

        if _kernels.HAS_CUDA_KERNEL:
            all_lengths, all_values_list = _kernels.unified_batched_gather(
                shard.flat_offsets,
                shard.offset_indices,
                shard.value_ptrs,
                shard.flat_bytes_vec,
                safe_indices,
            )
            return all_lengths, all_values_list
    except Exception:
        pass

    # Generic Python fallback for tests and debugging. The distributed pack path
    # still requires fused serialize/deserialize kernels.
    all_lengths_list = []
    all_values_list = []

    for f in range(shard.num_entries):
        offsets_f = shard.flat_offsets[shard.offset_indices[f]]
        begins = offsets_f[safe_indices]
        ends = offsets_f[safe_indices + 1]
        lengths = ends - begins
        all_lengths_list.append(lengths)

        if f < shard.num_features and not shard.is_ragged[f]:
            all_values_list.append(shard.value_vec[f][safe_indices])
            continue

        out_offsets = torch.zeros(
            len(safe_indices) + 1, dtype=torch.int64, device=safe_indices.device
        )
        out_offsets[1:] = lengths.cumsum(0)
        if lengths.sum() > 0:
            max_len = lengths.max().item()
            inner = torch.arange(max_len, device=safe_indices.device)
            src_idx = begins.unsqueeze(1) + inner.unsqueeze(0)
            mask = inner.unsqueeze(0) < lengths.unsqueeze(1)
            all_values_list.append(shard.value_vec[f][src_idx[mask]])
        else:
            all_values_list.append(
                torch.empty(
                    0, dtype=shard.value_vec[f].dtype, device=safe_indices.device
                )
            )

    all_lengths = torch.stack(all_lengths_list, dim=0)
    return all_lengths, all_values_list
