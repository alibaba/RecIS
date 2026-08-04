"""Unified跨 rank pack_feature 通信路径。"""

import torch
import torch.distributed as dist

from recis.ragged.tensor import RaggedTensor


def all_gather_sorted_ids(local_sorted_ids, group, device):
    """加载时一次性 all_gather 所有 rank 的 sorted_ids。

    各 rank 的 sorted_ids 大小可能不同（58.8M / 16 不整除），
    需要先 padding 到相同大小再 all_gather，然后按实际大小截取。
    """
    world_size = dist.get_world_size(group)
    local_size = local_sorted_ids.numel()

    # 先 all_gather 各 rank 的实际大小
    sizes = [
        torch.empty(1, dtype=torch.int64, device=device) for _ in range(world_size)
    ]
    dist.all_gather(
        sizes, torch.tensor([local_size], dtype=torch.int64, device=device), group=group
    )
    all_sizes = [s.item() for s in sizes]
    max_size = max(all_sizes)

    # padding 到 max_size（用 INT64_MAX 填充，不影响 searchsorted 正确性）
    if local_size < max_size:
        pad = torch.full(
            (max_size - local_size,),
            9223372036854775807,
            dtype=torch.int64,
            device=device,
        )
        padded = torch.cat([local_sorted_ids, pad])
    else:
        padded = local_sorted_ids

    # all_gather（现在所有 rank 大小相同）
    gather_list_padded = [
        torch.empty(max_size, dtype=torch.int64, device=device)
        for _ in range(world_size)
    ]
    dist.all_gather(gather_list_padded, padded, group=group)

    # 按实际大小截取（去掉 padding）
    return [g[: all_sizes[r]] for r, g in enumerate(gather_list_padded)]


def shard_pack_feature_unified(
    query_ids,
    target_ranks,
    shard,
    group,
    device,
):
    """Unified shard pack feature using fused CUDA kernels.

    Args:
        query_ids: [Q] int64, query IDs (valid_sample_ids already replaced not-found).
        target_ranks: [Q] int64, which rank each ID belongs to (from fused_route_ids).
        shard: Local GpuShard with unified representation.
        group: NCCL process group.
        device: CUDA device.

    Returns:
        dict[str, Tensor | RaggedTensor]: feature name → gathered result.
    """
    world_size = dist.get_world_size(group)
    Q = query_ids.numel()
    E = shard.num_entries

    # ── Step 0: Sort by target_rank + all_to_all IDs ──
    sort_order = target_ranks.argsort()
    sorted_target_ranks = target_ranks[sort_order]
    send_ids = query_ids[sort_order]

    target_counts = torch.bincount(sorted_target_ranks, minlength=world_size)
    source_counts = torch.empty(world_size, dtype=torch.int64, device=device)
    dist.all_to_all_single(source_counts, target_counts, group=group)

    total_recv = source_counts.sum().item()
    input_splits = target_counts.cpu().tolist()
    output_splits = source_counts.cpu().tolist()
    recv_ids = torch.empty(total_recv, dtype=torch.int64, device=device)
    dist.all_to_all_single(
        recv_ids,
        send_ids,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )

    scatter_indices = sort_order
    scatter_offsets = torch.zeros(world_size + 1, dtype=torch.int64, device=device)
    scatter_offsets[1:] = target_counts.cumsum(0)

    source_offsets_gpu = torch.zeros(world_size + 1, dtype=torch.int64, device=device)
    source_offsets_gpu[1:] = source_counts.cumsum(0)

    # ── Step 1: Unified local gather ──
    if total_recv > 0:
        from recis.data.gpu_shard_sampler.gather import unified_gather

        all_lengths, all_values_list = unified_gather(shard, recv_ids)
    else:
        all_lengths = torch.empty(E, 0, dtype=torch.int64, device=device)
        all_values_list = [
            torch.empty(0, dtype=torch.uint8, device=device) for _ in range(E)
        ]

    flat_bytes_vec = shard.flat_bytes_vec

    # ── Step 2: Precompute serialization offsets ──
    # val_counts[f, r] = sum of all_lengths[f, items_from_rank_r]
    # Items are already grouped by source rank → use cumsum + segment diff (no atomics)
    val_counts = torch.zeros(E, world_size, dtype=torch.int64, device=device)
    if total_recv > 0:
        all_lengths_cumsum = torch.zeros(
            E, total_recv + 1, dtype=torch.int64, device=device
        )
        all_lengths_cumsum[:, 1:] = all_lengths.cumsum(dim=1)
        val_counts = all_lengths_cumsum.index_select(
            1, source_offsets_gpu[1:]
        ) - all_lengths_cumsum.index_select(1, source_offsets_gpu[:-1])

    val_starts = torch.zeros(E, world_size + 1, dtype=torch.int64, device=device)
    val_starts[:, 1:] = val_counts.cumsum(dim=1)

    lengths_size = source_counts * E * 8
    val_byte_counts = val_counts * flat_bytes_vec.unsqueeze(1)
    values_size = val_byte_counts.sum(dim=0)
    total_sizes = lengths_size + values_size

    send_offsets = torch.zeros(world_size + 1, dtype=torch.int64, device=device)
    send_offsets[1:] = total_sizes.cumsum(0)

    # Per-entry, per-rank byte offset within all_values section (for parallel serialize)
    entry_val_byte_offsets = torch.zeros(
        E + 1, world_size, dtype=torch.int64, device=device
    )
    entry_val_byte_offsets[1:] = val_byte_counts.cumsum(dim=0)

    # ── Step 3: Fused serialize ──
    send_total = send_offsets[-1].item()
    send_flat = torch.empty(send_total, dtype=torch.uint8, device=device)

    # Build value_ptrs from gathered values (not shard's original data)
    value_ptrs_gpu = torch.tensor(
        [v.data_ptr() for v in all_values_list], dtype=torch.int64, device=device
    )

    try:
        from recis.lib import gpu_shard_kernels as _kernels

        use_fused = _kernels.HAS_CUDA_KERNEL
    except Exception:
        use_fused = False

    if not use_fused:
        raise RuntimeError(
            "shard_pack_feature_unified requires the compiled fused CUDA kernels. "
            "Run the project build before calling pack_feature."
        )

    if total_recv > 0:
        _kernels.fused_serialize(
            all_lengths,
            value_ptrs_gpu,
            source_offsets_gpu,
            val_starts,
            flat_bytes_vec,
            send_offsets,
            entry_val_byte_offsets,
            send_flat,
            E,
            world_size,
            total_recv,
        )
    send_sizes = total_sizes

    # ── Step 4: Exchange buffer sizes + data ──
    recv_sizes = torch.empty(world_size, dtype=torch.int64, device=device)
    dist.all_to_all_single(recv_sizes, send_sizes, group=group)
    recv_sizes_cpu = recv_sizes.cpu()

    recv_total = recv_sizes_cpu.sum().item()
    recv_flat = torch.empty(recv_total, dtype=torch.uint8, device=device)
    recv_offsets = torch.zeros(world_size + 1, dtype=torch.int64, device=device)
    recv_offsets[1:] = recv_sizes.cumsum(0)

    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=recv_sizes_cpu.tolist(),
        input_split_sizes=send_sizes.cpu().tolist(),
        group=group,
    )

    # ── Step 5: Fused deserialize Phase 1 (all_lengths) ──
    out_lengths = torch.zeros(E, Q, dtype=torch.int64, device=device)
    recv_lengths = torch.zeros(E, Q, dtype=torch.int64, device=device)

    if recv_total > 0:
        _kernels.fused_deserialize_fixed(
            recv_flat,
            recv_offsets,
            target_counts,
            scatter_indices,
            scatter_offsets,
            out_lengths,
            recv_lengths,
            E,
            world_size,
            Q,
            Q,
        )

    # ── Step 6: Post-Phase-1 offset computation ──
    totals_per_entry = out_lengths.sum(dim=1)

    # Align each entry's byte size to 8 bytes for safe .view() across dtypes
    entry_byte_sizes = totals_per_entry * flat_bytes_vec
    entry_byte_sizes = ((entry_byte_sizes + 7) // 8) * 8

    feat_val_starts = torch.zeros(E + 1, dtype=torch.int64, device=device)
    feat_val_starts[1:] = entry_byte_sizes.cumsum(0)
    total_vals_bytes = feat_val_starts[-1].item()

    all_offsets = torch.zeros(E, Q + 1, dtype=torch.int64, device=device)
    all_offsets[:, 1:] = out_lengths.cumsum(dim=1)

    # Per-entry, per-rank value counts from recv_lengths (organized by target rank)
    # Items already grouped by target rank → cumsum + segment diff (no atomics)
    val_counts_recv = torch.zeros(E, world_size, dtype=torch.int64, device=device)
    if Q > 0:
        recv_lengths_cumsum = torch.zeros(E, Q + 1, dtype=torch.int64, device=device)
        recv_lengths_cumsum[:, 1:] = recv_lengths.cumsum(dim=1)
        val_counts_recv = recv_lengths_cumsum.index_select(
            1, scatter_offsets[1:]
        ) - recv_lengths_cumsum.index_select(1, scatter_offsets[:-1])

    entry_buf_byte_offsets = torch.zeros(
        E + 1, world_size, dtype=torch.int64, device=device
    )
    entry_buf_byte_offsets[1:] = (val_counts_recv * flat_bytes_vec.unsqueeze(1)).cumsum(
        dim=0
    )

    # Precompute src_offsets: exclusive prefix sum of recv_lengths within each rank section
    # Batch CPU transfer (1 sync) then Python loop (CPU access, no GPU sync)
    scatter_offsets_cpu = scatter_offsets.cpu()
    src_offsets = torch.zeros(E, Q, dtype=torch.int64, device=device)
    for r in range(world_size):
        start = scatter_offsets_cpu[r].item()
        end = scatter_offsets_cpu[r + 1].item()
        if end > start + 1:
            src_offsets[:, start + 1 : end] = recv_lengths[:, start : end - 1].cumsum(
                dim=1
            )

    # ── Step 7: Fused deserialize Phase 2 (values, direct scatter) ──
    out_values = torch.empty(total_vals_bytes, dtype=torch.uint8, device=device)

    if recv_total > 0:
        _kernels.fused_deserialize_values(
            recv_flat,
            recv_offsets,
            target_counts,
            scatter_indices,
            scatter_offsets,
            recv_lengths,
            src_offsets,
            all_offsets,
            feat_val_starts,
            flat_bytes_vec,
            entry_buf_byte_offsets,
            out_values,
            E,
            world_size,
            Q,
            Q,
        )

    # ── Step 8: Build final results ──
    final_results = {}

    # Batch CPU transfer to avoid per-feature GPU syncs
    feat_val_starts_cpu = feat_val_starts.cpu()
    actual_bytes_cpu = (totals_per_entry * flat_bytes_vec).cpu()

    for f in range(shard.num_features):
        name = shard.feature_names[f]
        s = feat_val_starts_cpu[f].item()
        actual_bytes = actual_bytes_cpu[f].item()
        vals = out_values[s : s + actual_bytes].view(shard.feature_dtypes[f])

        if shard.is_ragged[f]:
            offsets = all_offsets[f]
            vals = vals.contiguous()
            rt = RaggedTensor(vals, [offsets])

            w_idx = shard.weight_entry_idx[f]
            if w_idx is not None:
                ws = feat_val_starts_cpu[w_idx].item()
                w_actual_bytes = actual_bytes_cpu[w_idx].item()
                w_vals = (
                    out_values[ws : ws + w_actual_bytes]
                    .view(shard.weight_dtypes[f])
                    .contiguous()
                )
                rt.set_weight(w_vals)
            final_results[name] = rt
        else:
            suffix = shard.feature_shapes[f]
            if suffix:
                final_results[name] = vals.reshape(Q, *suffix)
            else:
                final_results[name] = vals.reshape(Q)

    return final_results
