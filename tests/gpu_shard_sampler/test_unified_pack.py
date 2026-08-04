#!/usr/bin/env python3
"""Single-GPU correctness tests for unified pack pipeline.

Tests fused_route_ids, unified_batched_gather, fused_serialize,
fused_deserialize_fixed, fused_deserialize_values with synthetic data.

Usage:
    python tests/test_unified_pack.py
"""

import torch

from recis.data.gpu_shard_sampler.shard import GpuShard
from recis.lib import gpu_shard_kernels as kernels


def create_mock_shard(num_items, num_ragged, device, has_weights=True):
    """Create a mock GpuShard with known data for testing (per-feature entries)."""
    sorted_ids = torch.sort(torch.randint(0, 100000, (num_items,), device=device))[0]

    # Dense features: each is its own entry
    dense_i64_0 = torch.randint(0, 1000, (num_items,), dtype=torch.int64, device=device)
    dense_i64_1 = torch.randint(
        0, 1000, (num_items, 1), dtype=torch.int64, device=device
    )
    dense_float_0 = torch.rand(num_items, dtype=torch.float32, device=device)
    dense_i32_0 = torch.randint(0, 100, (num_items,), dtype=torch.int32, device=device)

    # Ragged features
    ragged_values = []
    ragged_offsets = []
    for _ in range(num_ragged):
        lengths = torch.randint(0, 5, (num_items,), dtype=torch.int64, device=device)
        offsets = torch.zeros(num_items + 1, dtype=torch.int64, device=device)
        offsets[1:] = lengths.cumsum(0)
        total = lengths.sum().item()
        values = torch.randint(0, 10000, (total,), dtype=torch.int64, device=device)
        ragged_values.append(values)
        ragged_offsets.append(offsets)

    ragged_weights = []
    for f in range(num_ragged):
        if has_weights:
            ragged_weights.append(
                torch.rand(ragged_values[f].numel(), dtype=torch.float32, device=device)
            )
        else:
            ragged_weights.append(None)

    # Build per-feature unified representation
    dense_offset = torch.arange(num_items + 1, dtype=torch.int64, device=device)

    value_vec = []
    offsets_vec = []
    flat_bytes_list = []
    feature_names = []
    feature_dtypes = []
    is_ragged = []
    feature_shapes = []
    weight_entry_idx = []
    weight_dtypes = []

    # Dense entries (one per feature)
    dense_features = [
        ("d64_0", dense_i64_0, torch.int64, ()),
        ("d64_1", dense_i64_1, torch.int64, (1,)),
        ("d32f_0", dense_float_0, torch.float32, ()),
        ("d32i_0", dense_i32_0, torch.int32, ()),
    ]
    for name, val, dtype, shape_suffix in dense_features:
        stored = val.unsqueeze(1) if val.dim() == 1 else val
        value_vec.append(stored)
        offsets_vec.append(dense_offset)
        flat_bytes_list.append(stored.shape[1] * val.element_size())
        feature_names.append(name)
        feature_dtypes.append(dtype)
        is_ragged.append(False)
        feature_shapes.append(shape_suffix)
        weight_entry_idx.append(None)
        weight_dtypes.append(None)

    # Ragged value entries
    for f in range(num_ragged):
        value_vec.append(ragged_values[f])
        offsets_vec.append(ragged_offsets[f])
        flat_bytes_list.append(ragged_values[f].element_size())
        feature_names.append(f"rag_{f}")
        feature_dtypes.append(torch.int64)
        is_ragged.append(True)
        feature_shapes.append(())
        weight_entry_idx.append(None)
        weight_dtypes.append(None)

    # Weight entries (appended after all value entries)
    if has_weights:
        for f in range(num_ragged):
            if ragged_weights[f] is not None:
                dense_count = len(dense_features)
                weight_entry = len(value_vec)
                value_vec.append(ragged_weights[f])
                offsets_vec.append(ragged_offsets[f])
                flat_bytes_list.append(ragged_weights[f].element_size())
                weight_entry_idx[dense_count + f] = weight_entry
                weight_dtypes[dense_count + f] = torch.float32

    num_features = len(feature_names)

    shard = GpuShard(
        sorted_ids=sorted_ids,
        value_vec=value_vec,
        offsets_vec=offsets_vec,
        flat_bytes_vec_list=flat_bytes_list,
        feature_names=feature_names,
        feature_dtypes=feature_dtypes,
        is_ragged=is_ragged,
        feature_shapes=feature_shapes,
        weight_entry_idx=weight_entry_idx,
        weight_dtypes=weight_dtypes,
        num_features=num_features,
        num_ragged=num_ragged,
        num_items=num_items,
        shard_id=0,
    )
    shard.to_device(device)
    return shard


def test_unified_batched_gather():
    """Test unified_batched_gather: verify gather results match manual indexing."""
    print("test_unified_batched_gather...", end=" ")
    device = "cuda:0"
    torch.cuda.set_device(0)

    num_items = 100
    num_ragged = 5
    shard = create_mock_shard(num_items, num_ragged, device, has_weights=True)

    # Query all items in order
    indices = torch.arange(num_items, dtype=torch.int64, device=device)

    all_lengths, all_values = kernels.unified_batched_gather(
        shard.flat_offsets,
        shard.offset_indices,
        shard.value_ptrs,
        shard.flat_bytes_vec,
        indices,
    )

    E = shard.num_entries
    assert all_lengths.shape == (E, num_items), (
        f"Expected ({E}, {num_items}), got {all_lengths.shape}"
    )

    # Check each feature's gathered values
    for f in range(shard.num_features):
        flat_bytes = shard.flat_bytes_vec_list[f]
        dtype = shard.feature_dtypes[f]
        elem_size = torch.tensor([], dtype=dtype).element_size()

        if not shard.is_ragged[f]:
            # Dense: total = num_items, each with flat_bytes/elem_size columns
            cols = flat_bytes // elem_size
            vals = all_values[f].view(dtype).reshape(num_items, cols)
            expected = shard.value_vec[f]
            assert torch.equal(vals, expected), (
                f"dense {shard.feature_names[f]} mismatch"
            )

    # Check ragged lengths
    for f in range(shard.num_features):
        if shard.is_ragged[f]:
            off_idx = shard.offset_indices[f].item()
            expected_lengths = (
                shard.flat_offsets[off_idx][1:]
                - shard.flat_offsets[off_idx][:num_items]
            )
            assert torch.equal(all_lengths[f], expected_lengths), (
                f"ragged {shard.feature_names[f]} lengths mismatch"
            )

    print("PASSED")


def test_fused_route_ids():
    """Test fused_route_ids: verify routing and replacement."""
    print("test_fused_route_ids...", end=" ")
    device = "cuda:0"
    torch.cuda.set_device(0)

    num_items = 50
    num_ragged = 3
    shard = create_mock_shard(num_items, num_ragged, device, has_weights=False)

    # Create sorted_ptrs and sorted_sizes (simulating 2 ranks)
    sorted_ids_r0 = shard.sorted_ids[:25]
    sorted_ids_r1 = shard.sorted_ids[25:]
    sorted_ptrs = torch.tensor(
        [sorted_ids_r0.data_ptr(), sorted_ids_r1.data_ptr()],
        dtype=torch.int64,
        device=device,
    )
    sorted_sizes = torch.tensor([25, 25], dtype=torch.int32, device=device)

    # Query: first 10 from rank 0, next 10 from rank 1, 5 non-existent
    query_ids = torch.cat(
        [
            sorted_ids_r0[:10],
            sorted_ids_r1[:10],
            torch.tensor(
                [999999, 999998, 999997, 999996, 999995],
                dtype=torch.int64,
                device=device,
            ),
        ]
    )
    default_value = sorted_ids_r0[0].item()  # Use first ID as default

    out_ids, target_ranks, found_mask = kernels.fused_route_ids(
        query_ids,
        sorted_ptrs,
        sorted_sizes,
        default_value,
        2,
    )

    # First 10 should be found in rank 0
    assert target_ranks[:10].eq(0).all(), "First 10 should be in rank 0"
    # Next 10 should be found in rank 1
    assert target_ranks[10:20].eq(1).all(), "Next 10 should be in rank 1"
    # Last 5: not found originally, but replaced with default_value (which is in rank 0)
    assert found_mask[20:].all(), "Last 5 should be found as default_value"
    assert out_ids[20:].eq(default_value).all(), (
        "Last 5 should be replaced with default"
    )
    assert target_ranks[20:].ge(0).all(), "Last 5 should have valid target_ranks"

    print("PASSED")


def test_serialize_deserialize_roundtrip():
    """Test serialize → deserialize roundtrip: verify data integrity."""
    print("test_serialize_deserialize_roundtrip...", end=" ")
    device = "cuda:0"
    torch.cuda.set_device(0)

    # Use odd num_items to trigger 8-byte alignment edge case
    # (float32 features produce 51*4=204 bytes, 204 % 8 = 4 ≠ 0)
    num_items = 51
    num_ragged = 4
    shard = create_mock_shard(num_items, num_ragged, device, has_weights=True)

    # Simulate single "rank" (identity: send to self, receive from self)
    world_size = 1
    E = shard.num_entries
    Q = num_items

    # Gather all items
    from recis.data.gpu_shard_sampler.gather import unified_gather

    recv_ids = shard.sorted_ids  # Query all items
    all_lengths, all_values_list = unified_gather(shard, recv_ids)

    total_recv = num_items
    source_counts = torch.tensor([num_items], dtype=torch.int64, device=device)
    source_offsets = torch.tensor([0, num_items], dtype=torch.int64, device=device)

    flat_bytes_vec = shard.flat_bytes_vec

    # Compute val_counts and val_starts
    val_counts = all_lengths.sum(dim=1).unsqueeze(1)  # [E, 1]
    val_starts = torch.zeros(E, 2, dtype=torch.int64, device=device)
    val_starts[:, 1] = val_counts[:, 0]

    # Compute send_offsets
    lengths_size = source_counts * E * 8
    val_byte_counts = val_counts * flat_bytes_vec.unsqueeze(1)
    values_size = val_byte_counts.sum(dim=0)
    total_sizes = lengths_size + values_size
    send_offsets = torch.tensor(
        [0, total_sizes[0].item()], dtype=torch.int64, device=device
    )

    # Serialize
    send_total = total_sizes[0].item()
    send_flat = torch.empty(send_total, dtype=torch.uint8, device=device)

    # Per-entry, per-rank byte offset within all_values section (for parallel serialize)
    entry_val_byte_offsets = torch.zeros(
        E + 1, world_size, dtype=torch.int64, device=device
    )
    entry_val_byte_offsets[1:] = val_byte_counts.cumsum(dim=0)

    value_ptrs = shard.value_ptrs
    kernels.fused_serialize(
        all_lengths,
        value_ptrs,
        source_offsets,
        val_starts,
        flat_bytes_vec,
        send_offsets,
        entry_val_byte_offsets,
        send_flat,
        E,
        world_size,
        total_recv,
    )

    # Deserialize Phase 1
    target_counts = source_counts  # Identity
    scatter_indices = torch.arange(Q, dtype=torch.int64, device=device)
    scatter_offsets = torch.tensor([0, Q], dtype=torch.int64, device=device)
    recv_offsets = send_offsets

    out_lengths = torch.zeros(E, Q, dtype=torch.int64, device=device)
    recv_lengths = torch.zeros(E, total_recv, dtype=torch.int64, device=device)

    kernels.fused_deserialize_fixed(
        send_flat,
        recv_offsets,
        target_counts,
        scatter_indices,
        scatter_offsets,
        out_lengths,
        recv_lengths,
        E,
        world_size,
        Q,
        total_recv,
    )

    # Check lengths match
    assert torch.equal(out_lengths, all_lengths), "Lengths mismatch after roundtrip"

    # Compute offsets for Phase 2 (with 8-byte alignment, matching comm.py)
    totals_per_entry = out_lengths.sum(dim=1)
    entry_byte_sizes = totals_per_entry * flat_bytes_vec
    entry_byte_sizes = ((entry_byte_sizes + 7) // 8) * 8  # align to 8 bytes

    feat_val_starts = torch.zeros(E + 1, dtype=torch.int64, device=device)
    feat_val_starts[1:] = entry_byte_sizes.cumsum(0)
    total_vals_bytes = feat_val_starts[-1].item()

    # Verify all offsets are 8-byte aligned (critical for .view(int64))
    assert (feat_val_starts % 8 == 0).all(), (
        f"feat_val_starts not 8-byte aligned: {feat_val_starts}"
    )

    all_offsets = torch.zeros(E, Q + 1, dtype=torch.int64, device=device)
    all_offsets[:, 1:] = out_lengths.cumsum(dim=1)

    rank_indices = torch.zeros(total_recv, dtype=torch.int64, device=device)
    val_counts_recv = torch.zeros(E, world_size, dtype=torch.int64, device=device)
    val_counts_recv.scatter_add_(
        1, rank_indices.unsqueeze(0).expand(E, -1), recv_lengths
    )
    entry_buf_byte_offsets = torch.zeros(
        E + 1, world_size, dtype=torch.int64, device=device
    )
    entry_buf_byte_offsets[1:] = (val_counts_recv * flat_bytes_vec.unsqueeze(1)).cumsum(
        dim=0
    )

    # Deserialize Phase 2
    out_values = torch.empty(total_vals_bytes, dtype=torch.uint8, device=device)
    # Precompute src_offsets (vectorized, matching comm.py)
    src_offsets = torch.zeros(E, total_recv, dtype=torch.int64, device=device)
    if total_recv > 1:
        cumsum = torch.zeros(E, total_recv, dtype=torch.int64, device=device)
        cumsum[:, 1:] = recv_lengths[:, : total_recv - 1].cumsum(dim=1)
        # Single rank: all items belong to rank 0, start = 0
        src_offsets = cumsum  # cumsum_at_start is 0 for rank 0

    kernels.fused_deserialize_values(
        send_flat,
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
        total_recv,
    )

    # Verify all feature values (use actual data size, not aligned size)
    for f in range(shard.num_features):
        s = feat_val_starts[f].item()
        actual_bytes = (totals_per_entry[f] * flat_bytes_vec[f]).item()
        dtype = shard.feature_dtypes[f]
        flat_bytes = shard.flat_bytes_vec_list[f]
        elem_size = torch.tensor([], dtype=dtype).element_size()

        if not shard.is_ragged[f]:
            cols = flat_bytes // elem_size
            vals = out_values[s : s + actual_bytes].view(dtype).reshape(Q, cols)
            expected = shard.value_vec[f]
            assert torch.equal(vals, expected), (
                f"dense {shard.feature_names[f]} mismatch after roundtrip"
            )
        else:
            vals = out_values[s : s + actual_bytes].view(dtype)
            expected = shard.value_vec[f]
            assert vals.numel() == expected.numel(), (
                f"ragged {shard.feature_names[f]} size mismatch: {vals.numel()} vs {expected.numel()}"
            )
            assert torch.equal(vals, expected), (
                f"ragged {shard.feature_names[f]} values mismatch after roundtrip"
            )

    print("PASSED")


def test_edge_cases():
    """Test edge cases: single item, no ragged, no weights."""
    print("test_edge_cases...", end=" ")
    device = "cuda:0"
    torch.cuda.set_device(0)

    # Single item, no ragged, no weights
    shard = create_mock_shard(
        num_items=1, num_ragged=0, device=device, has_weights=False
    )
    indices = torch.tensor([0], dtype=torch.int64, device=device)

    all_lengths, all_values = kernels.unified_batched_gather(
        shard.flat_offsets,
        shard.offset_indices,
        shard.value_ptrs,
        shard.flat_bytes_vec,
        indices,
    )
    assert all_lengths.shape == (4, 1), f"Expected (4, 1), got {all_lengths.shape}"
    assert all_lengths.all(), "All dense lengths should be 1"

    print("PASSED")


def main():
    print("=== Unified Pack Correctness Tests ===")
    test_unified_batched_gather()
    test_fused_route_ids()
    test_serialize_deserialize_roundtrip()
    test_edge_cases()
    print("=== All tests PASSED ===")


if __name__ == "__main__":
    main()
