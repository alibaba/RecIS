#!/usr/bin/env python3
"""8-GPU integration test for unified pack pipeline.

Each rank generates mock shard data, then tests cross-rank pack_feature.

Usage:
    torchrun --nproc_per_node=8 tests/test_unified_pack_8gpu.py
"""

import torch
import torch.distributed as dist

from recis.data.gpu_shard_sampler.comm import (
    all_gather_sorted_ids,
    shard_pack_feature_unified,
)
from recis.data.gpu_shard_sampler.shard import GpuShard
from recis.lib import gpu_shard_kernels as kernels


try:
    import pytest

    # This file is a torchrun multi-process script (see main()); skip under
    # single-process pytest so CI collection does not run these tests.
    pytestmark = pytest.mark.skipif(
        not dist.is_available() or not dist.is_initialized(),
        reason="multi-GPU test; run with torchrun --nproc_per_node=8",
    )
except ImportError:  # torchrun standalone mode without pytest installed
    pass


def create_mock_shard_for_rank(
    num_items, num_ragged, rank, device, default_id, has_weights=True
):
    """Create a mock GpuShard with disjoint ID ranges per rank (per-feature entries)."""
    id_start = rank * 100000 + 1
    sorted_ids = torch.arange(
        id_start, id_start + num_items, dtype=torch.int64, device=device
    )
    if rank == 0 and default_id not in sorted_ids:
        sorted_ids[0] = default_id
        sorted_ids = sorted_ids.sort()[0]

    # Dense features (each is its own entry)
    dense_features = [
        (
            "d64_0",
            torch.randint(0, 1000, (num_items,), dtype=torch.int64, device=device),
            torch.int64,
            (),
        ),
        (
            "d64_1",
            torch.randint(0, 1000, (num_items, 1), dtype=torch.int64, device=device),
            torch.int64,
            (1,),
        ),
        ("d32f_0", torch.rand(num_items, device=device), torch.float32, ()),
        (
            "d32i_0",
            torch.randint(0, 100, (num_items,), dtype=torch.int32, device=device),
            torch.int32,
            (),
        ),
    ]

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
        shard_id=rank,
    )
    shard.to_device(device)
    return shard


def test_cross_rank_pack_feature():
    """Test pack_feature with cross-rank queries."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    num_items = 200
    num_ragged = 5
    default_value = 1  # ID 1 will be in rank 0

    # Create mock shard directly (disjoint ID ranges per rank)
    shard = create_mock_shard_for_rank(
        num_items, num_ragged, rank, device, default_value
    )

    # all_gather sorted_ids
    all_sorted_ids = all_gather_sorted_ids(shard.sorted_ids, dist.group.WORLD, device)

    # Precompute sorted_ptrs/sorted_sizes
    sorted_ptrs = torch.tensor(
        [s.data_ptr() for s in all_sorted_ids], dtype=torch.int64, device=device
    )
    sorted_sizes = torch.tensor(
        [s.numel() for s in all_sorted_ids], dtype=torch.int32, device=device
    )

    # Create query: 50 from own rank, 50 from other ranks, 10 non-existent
    gpu_sorted_ids = shard.sorted_ids
    own_ids = gpu_sorted_ids[torch.randperm(num_items, device=device)[:50]]
    other_rank = (rank + 1) % world_size
    other_ids = all_sorted_ids[other_rank][
        torch.randperm(all_sorted_ids[other_rank].numel(), device=device)[:50]
    ]
    fake_ids = torch.tensor(
        [
            9999999,
            9999998,
            9999997,
            9999996,
            9999995,
            9999994,
            9999993,
            9999992,
            9999991,
            9999990,
        ],
        dtype=torch.int64,
        device=device,
    )
    query_ids = torch.cat([own_ids, other_ids, fake_ids])

    # Fused route + replace
    valid_ids, target_ranks, found_mask = kernels.fused_route_ids(
        query_ids,
        sorted_ptrs,
        sorted_sizes,
        default_value,
        world_size,
    )

    # Pack feature
    results = shard_pack_feature_unified(
        valid_ids,
        target_ranks,
        shard,
        dist.group.WORLD,
        device,
    )

    # Verify: all results should have Q entries
    Q = query_ids.numel()
    from recis.ragged.tensor import RaggedTensor

    for name, val in results.items():
        if isinstance(val, RaggedTensor):
            offsets = val.offsets()[0]
            assert offsets.numel() == Q + 1, (
                f"{name}: offsets should be [{Q + 1}], got {offsets.numel()}"
            )
        else:
            assert val.shape[0] == Q, (
                f"{name}: shape[0] should be {Q}, got {val.shape[0]}"
            )

    # Verify determinism: same input → same output
    results2 = shard_pack_feature_unified(
        valid_ids,
        target_ranks,
        shard,
        dist.group.WORLD,
        device,
    )

    for name in results:
        v1, v2 = results[name], results2[name]
        if isinstance(v1, RaggedTensor):
            assert torch.equal(v1.offsets()[0], v2.offsets()[0]), (
                f"{name}: non-deterministic offsets"
            )
        else:
            assert torch.equal(v1, v2), f"{name}: non-deterministic"

    if rank == 0:
        print(
            f"  Q={Q}, features={len(results)}, all shapes correct, deterministic=True"
        )

    return True


def test_default_value_replacement():
    """Test that default_value positions return default ID's features."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    num_items = 100
    num_ragged = 3
    default_value = 42

    # Create mock shard (rank 0 includes default_value=42)
    shard = create_mock_shard_for_rank(
        num_items, num_ragged, rank, device, default_value
    )
    all_sorted_ids = all_gather_sorted_ids(shard.sorted_ids, dist.group.WORLD, device)

    sorted_ptrs = torch.tensor(
        [s.data_ptr() for s in all_sorted_ids], dtype=torch.int64, device=device
    )
    sorted_sizes = torch.tensor(
        [s.numel() for s in all_sorted_ids], dtype=torch.int32, device=device
    )

    # Query with mix of valid and invalid IDs
    valid_ids_query = (
        all_sorted_ids[0][:10] if len(all_sorted_ids[0]) > 10 else all_sorted_ids[0]
    )
    invalid_ids = torch.tensor([9999999, 9999998], dtype=torch.int64, device=device)
    query_ids = torch.cat([valid_ids_query, invalid_ids])

    valid_ids, target_ranks, found_mask = kernels.fused_route_ids(
        query_ids,
        sorted_ptrs,
        sorted_sizes,
        default_value,
        world_size,
    )

    # Invalid IDs should be replaced with default_value
    assert valid_ids[-2].item() == default_value, (
        f"Expected default_value={default_value}, got {valid_ids[-2].item()}"
    )
    assert valid_ids[-1].item() == default_value, (
        f"Expected default_value={default_value}, got {valid_ids[-1].item()}"
    )

    results = shard_pack_feature_unified(
        valid_ids,
        target_ranks,
        shard,
        dist.group.WORLD,
        device,
    )

    # Verify: all results should have Q entries
    Q = query_ids.numel()
    for name, val in results.items():
        from recis.ragged.tensor import RaggedTensor

        if isinstance(val, RaggedTensor):
            offsets = val.offsets()[0]
            assert offsets.numel() == Q + 1, (
                f"{name}: offsets should be [{Q + 1}], got {offsets.numel()}"
            )
        else:
            assert val.shape[0] == Q, (
                f"{name}: shape[0] should be {Q}, got {val.shape[0]}"
            )

    if rank == 0:
        print(f"  default_value={default_value} replacement verified, Q={Q}")

    return True


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"=== Unified Pack 8-GPU Integration Tests (world_size={world_size}) ===")

    all_pass = True

    if rank == 0:
        print("test_cross_rank_pack_feature...")
    try:
        all_pass &= test_cross_rank_pack_feature()
    except Exception as e:
        import traceback

        print(f"  Rank {rank}: FAILED: {e}")
        traceback.print_exc()
        all_pass = False
    dist.barrier()

    if rank == 0:
        print("test_default_value_replacement...")
    try:
        all_pass &= test_default_value_replacement()
    except Exception as e:
        print(f"  Rank {rank}: FAILED: {e}")
        all_pass = False
    dist.barrier()

    # Aggregate pass/fail
    pass_tensor = torch.tensor(1 if all_pass else 0, device=f"cuda:{rank}")
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)

    if rank == 0:
        if pass_tensor.item() == 1:
            print("=== All tests PASSED ===")
        else:
            print("=== Some tests FAILED ===")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
