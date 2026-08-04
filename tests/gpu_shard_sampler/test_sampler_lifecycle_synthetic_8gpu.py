#!/usr/bin/env python3
"""8-GPU synthetic lifecycle test for GpuShardSampler.

This test creates its own negative table on every rank and exercises the public
sampler lifecycle instead of loading real dumps:

    prepare_reload -> commit_reload -> valid_sample_ids -> pack_feature

Coverage matrix:
  - dense int64/int32/float32, scalar and multi-column shapes
  - ragged features with empty, fixed, variable, long, and tail-empty rows
  - ragged features with and without float32 weights
  - unsorted input rows and per-rank shard sizes
  - cross-rank routing to every rank
  - heavily skewed traffic to rank 0
  - local-only queries
  - duplicate query IDs
  - missing query IDs replaced by default_value
  - empty query batches

Usage:
    NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 \
      torchrun --nproc_per_node=8 tests/test_sampler_lifecycle_synthetic_8gpu.py
"""

import traceback

import torch
import torch.distributed as dist


DEFAULT_ID = 777


def _make_ragged(unit_ids, rank, feature_idx, pattern, with_weight):
    from recis.ragged.tensor import RaggedTensor

    lengths = []
    for i, uid in enumerate(unit_ids.tolist()):
        if pattern == "empty":
            length = 0
        elif pattern == "single":
            length = 1
        elif pattern == "variable":
            length = (i + rank + feature_idx) % 6
        elif pattern == "long":
            length = 16 if i % 7 == 0 else (i + feature_idx) % 4
        elif pattern == "tail_empty":
            length = 0 if i >= unit_ids.numel() * 3 // 4 else 2
        elif pattern == "default_sensitive":
            length = 5 if uid == DEFAULT_ID else (i % 3)
        else:
            raise ValueError(f"unknown ragged pattern: {pattern}")
        lengths.append(length)

    lengths_t = torch.tensor(lengths, dtype=torch.int64)
    offsets = torch.zeros(unit_ids.numel() + 1, dtype=torch.int64)
    offsets[1:] = lengths_t.cumsum(0)
    total = int(offsets[-1].item())

    values = torch.empty(total, dtype=torch.int64)
    cursor = 0
    for i, length in enumerate(lengths):
        if length == 0:
            continue
        uid = int(unit_ids[i].item())
        row_values = (
            uid * 1000 + feature_idx * 100 + torch.arange(length, dtype=torch.int64)
        )
        values[cursor : cursor + length] = row_values
        cursor += length

    rt = RaggedTensor(values, [offsets])
    if with_weight:
        weights = torch.arange(total, dtype=torch.float32) * 0.125 + float(feature_idx)
        rt.set_weight(weights)
    return rt


def make_feature_table(rank, world_size):
    """Create an unsorted CPU feature table for one rank."""
    num_items = 73 + rank * 9
    id_start = rank * 1_000_000 + 10_000
    ids = id_start + torch.arange(num_items, dtype=torch.int64) * 17 + rank
    if rank == 0:
        ids[5] = DEFAULT_ID

    perm = torch.remainder(
        torch.arange(num_items, dtype=torch.int64) * 37 + 11, num_items
    )
    unit_ids = ids[perm].contiguous()

    dense_i64 = unit_ids * 10 + rank
    dense_i64_vec = torch.stack([unit_ids, unit_ids + 1, unit_ids % 97], dim=1)
    dense_i32 = (unit_ids % 1000).to(torch.int32)
    dense_i32_vec = torch.stack([(unit_ids % 127), (unit_ids % 251)], dim=1).to(
        torch.int32
    )
    dense_f32 = unit_ids.to(torch.float32) * 0.25 + float(rank)
    dense_f32_vec = torch.stack([dense_f32, -dense_f32], dim=1)

    table = {
        "unit_id": unit_ids,
        "dense_i64": dense_i64,
        "dense_i64_vec": dense_i64_vec,
        "dense_i32": dense_i32,
        "dense_i32_vec": dense_i32_vec,
        "dense_f32": dense_f32,
        "dense_f32_vec": dense_f32_vec,
        "_bench_ignored": torch.arange(num_items, dtype=torch.int64),
    }

    ragged_specs = [
        ("rag_empty", "empty", False),
        ("rag_single_w", "single", True),
        ("rag_variable", "variable", False),
        ("rag_long_w", "long", True),
        ("rag_tail_empty", "tail_empty", False),
        ("rag_default_w", "default_sensitive", True),
    ]
    for feature_idx, (name, pattern, with_weight) in enumerate(ragged_specs):
        table[name] = _make_ragged(unit_ids, rank, feature_idx, pattern, with_weight)

    return table


def build_cpu_reference(feature_table):
    from recis.ragged.tensor import RaggedTensor

    unit_ids = feature_table["unit_id"].reshape(-1)
    sorted_order = unit_ids.argsort()
    sorted_ids = unit_ids[sorted_order]

    dense = {}
    ragged = {}
    for name, val in feature_table.items():
        if name.startswith("_bench"):
            continue
        if isinstance(val, torch.Tensor):
            dense[name] = (val[sorted_order].contiguous(), tuple(val.shape[1:]))
        elif isinstance(val, RaggedTensor):
            old_offsets = val.offsets()[0]
            begins = old_offsets[sorted_order]
            ends = old_offsets[sorted_order + 1]
            lengths = ends - begins

            new_offsets = torch.zeros(sorted_order.numel() + 1, dtype=torch.int64)
            new_offsets[1:] = lengths.cumsum(0)
            total = int(new_offsets[-1].item())

            if total == 0:
                new_values = val.values().new_empty((0,))
                gather_idx = torch.empty(0, dtype=torch.int64)
            else:
                max_len = int(lengths.max().item())
                inner = torch.arange(max_len, dtype=torch.int64)
                idx = begins.unsqueeze(1) + inner.unsqueeze(0)
                mask = inner.unsqueeze(0) < lengths.unsqueeze(1)
                gather_idx = idx[mask]
                new_values = val.values()[gather_idx].contiguous()

            weight = val.weight()
            if weight is not None:
                new_weight = (
                    weight[gather_idx].contiguous()
                    if total > 0
                    else weight.new_empty((0,))
                )
            else:
                new_weight = None
            ragged[name] = (new_values, new_offsets, new_weight)
        else:
            raise TypeError(f"unsupported feature type {name}: {type(val)}")

    return {"sorted_ids": sorted_ids, "dense": dense, "ragged": ragged}


def route_reference_rows(query_ids_cpu, all_refs, default_value):
    q = query_ids_cpu.numel()
    shard_ranks = torch.full((q,), -1, dtype=torch.int64)
    row_indices = torch.zeros(q, dtype=torch.int64)

    for rank, ref in enumerate(all_refs):
        sorted_ids = ref["sorted_ids"]
        pos = torch.searchsorted(sorted_ids, query_ids_cpu)
        pos_c = pos.clamp(max=sorted_ids.numel() - 1)
        found = sorted_ids[pos_c] == query_ids_cpu
        assign = found & (shard_ranks == -1)
        shard_ranks[assign] = rank
        row_indices[assign] = pos_c[assign]

    default_rank = -1
    default_row = 0
    default_t = torch.tensor(default_value, dtype=torch.int64)
    for rank, ref in enumerate(all_refs):
        sorted_ids = ref["sorted_ids"]
        pos = torch.searchsorted(sorted_ids, default_t)
        pos_c = pos.clamp(max=sorted_ids.numel() - 1)
        if sorted_ids[pos_c].item() == default_value:
            default_rank = rank
            default_row = int(pos_c.item())
            break
    if default_rank < 0:
        raise AssertionError(f"default_value={default_value} not found")

    missing = shard_ranks == -1
    shard_ranks[missing] = default_rank
    row_indices[missing] = default_row
    return shard_ranks, row_indices


def verify_results(name, query_ids_cpu, results, all_refs, default_value, rank):
    from recis.ragged.tensor import RaggedTensor

    q = query_ids_cpu.numel()
    shard_ranks, row_indices = route_reference_rows(
        query_ids_cpu, all_refs, default_value
    )

    for feature_name in all_refs[0]["dense"]:
        out = results[feature_name].detach().cpu()
        expected = torch.empty_like(out)
        for src_rank, ref in enumerate(all_refs):
            mask = shard_ranks == src_rank
            if not mask.any():
                continue
            rows = row_indices[mask]
            gathered = ref["dense"][feature_name][0][rows].to(out.dtype)
            expected[mask] = gathered.reshape(expected[mask].shape)
        if not torch.equal(out, expected):
            diff = (out != expected).nonzero(as_tuple=False)
            idx = tuple(diff[0].tolist())
            raise AssertionError(
                f"[rank {rank}] scenario={name} dense={feature_name} mismatch at {idx}: "
                f"expected={expected[idx].item()} got={out[idx].item()}"
            )

    for feature_name in all_refs[0]["ragged"]:
        out = results[feature_name]
        if not isinstance(out, RaggedTensor):
            raise AssertionError(f"[rank {rank}] {feature_name} should be RaggedTensor")
        out_values = out.values().detach().cpu()
        out_offsets = out.offsets()[0].detach().cpu()
        out_weight = out.weight()
        out_weight = out_weight.detach().cpu() if out_weight is not None else None

        if out_offsets.numel() != q + 1:
            raise AssertionError(
                f"[rank {rank}] scenario={name} ragged={feature_name} offsets size mismatch: "
                f"expected={q + 1} got={out_offsets.numel()}"
            )

        for i in range(q):
            src_rank = int(shard_ranks[i].item())
            row = int(row_indices[i].item())
            ref_values, ref_offsets, ref_weight = all_refs[src_rank]["ragged"][
                feature_name
            ]
            ref_start = int(ref_offsets[row].item())
            ref_end = int(ref_offsets[row + 1].item())
            out_start = int(out_offsets[i].item())
            out_end = int(out_offsets[i + 1].item())

            expected_values = ref_values[ref_start:ref_end].to(out_values.dtype)
            actual_values = out_values[out_start:out_end]
            if not torch.equal(actual_values, expected_values):
                raise AssertionError(
                    f"[rank {rank}] scenario={name} ragged={feature_name} values mismatch "
                    f"at q={i}, src_rank={src_rank}, row={row}"
                )

            if ref_weight is None:
                if out_weight is not None and out_weight.numel() != 0:
                    raise AssertionError(
                        f"[rank {rank}] scenario={name} ragged={feature_name} unexpected weight"
                    )
            else:
                if out_weight is None:
                    raise AssertionError(
                        f"[rank {rank}] scenario={name} ragged={feature_name} missing weight"
                    )
                expected_weight = ref_weight[ref_start:ref_end].to(out_weight.dtype)
                actual_weight = out_weight[out_start:out_end]
                if not torch.equal(actual_weight, expected_weight):
                    raise AssertionError(
                        f"[rank {rank}] scenario={name} ragged={feature_name} weight mismatch "
                        f"at q={i}, src_rank={src_rank}, row={row}"
                    )


def make_query(name, rank, world_size, all_refs):
    ids = []
    if name == "balanced_all_ranks":
        for src_rank, ref in enumerate(all_refs):
            size = ref["sorted_ids"].numel()
            count = 1 + ((rank + src_rank) % 4)
            rows = [(rank * 13 + src_rank * 7 + i * 5) % size for i in range(count)]
            ids.extend(ref["sorted_ids"][rows].tolist())
        ids.extend([all_refs[rank]["sorted_ids"][0].item()] * 3)
        ids.extend([DEFAULT_ID, -1_000_000_000 - rank, 9_000_000_000 + rank])
    elif name == "skew_rank0_and_missing":
        ref0 = all_refs[0]["sorted_ids"]
        rows = [(rank * 11 + i * 3) % ref0.numel() for i in range(24 + rank)]
        ids.extend(ref0[rows].tolist())
        ids.extend([123_456_789_000 + rank * 10 + i for i in range(6)])
    elif name == "local_only_duplicates":
        ref = all_refs[rank]["sorted_ids"]
        rows = [(rank + i * 2) % ref.numel() for i in range(17)]
        local = ref[rows].tolist()
        ids.extend(local)
        ids.extend(local[:5])
    elif name == "empty_query":
        ids = []
    else:
        raise ValueError(f"unknown scenario: {name}")
    return torch.tensor(ids, dtype=torch.int64)


def verify_valid_sample_ids(sampler, all_refs, rank, device):
    query_cpu = torch.tensor(
        [
            all_refs[rank]["sorted_ids"][0].item(),
            all_refs[(rank + 1) % len(all_refs)]["sorted_ids"][1].item(),
            -90_000_000 - rank,
            DEFAULT_ID,
        ],
        dtype=torch.int64,
    )
    expected = query_cpu.clone()
    expected[2] = DEFAULT_ID
    actual = sampler.valid_sample_ids(query_cpu.to(device), DEFAULT_ID).detach().cpu()
    if not torch.equal(actual, expected):
        raise AssertionError(
            f"[rank {rank}] valid_sample_ids mismatch: expected={expected}, got={actual}"
        )


def verify_combine(sampler, device):
    batch = [
        {
            "_sample_group_id": torch.tensor(
                [10, 20, 30], dtype=torch.int64, device=device
            ),
        },
        {
            "_indicator_main": torch.tensor(
                [1, 0, 1], dtype=torch.int32, device=device
            ),
        },
        {
            "_indicator_aux": torch.tensor([3, 4, 5], dtype=torch.int32, device=device),
        },
    ]
    output = {"dense_i64": torch.arange(6, dtype=torch.int64, device=device)}
    sample_cnts = torch.tensor([0, 2, 1], dtype=torch.int64, device=device)
    combined = sampler.combine(batch, output, sample_cnts)

    expected_group = torch.tensor(
        [10, 20, 20, 20, 30, 30], dtype=torch.int64, device=device
    )
    expected_main = torch.tensor([1, 0, 0, 0, 1, 1], dtype=torch.int32, device=device)
    expected_aux = torch.tensor([3, 4, 4, 4, 5, 5], dtype=torch.int32, device=device)
    if not torch.equal(combined[0]["_sample_group_id"], expected_group):
        raise AssertionError("combine _sample_group_id mismatch")
    if not torch.equal(combined[1]["_indicator_main"], expected_main):
        raise AssertionError("combine _indicator_main mismatch")
    if not torch.equal(combined[2]["_indicator_aux"], expected_aux):
        raise AssertionError("combine _indicator_aux mismatch")


def run_rank():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"expected world_size=8, got {world_size}")

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    from recis.data.gpu_shard_sampler.sampler import GpuShardSampler

    feature_table = make_feature_table(rank, world_size)
    cpu_ref = build_cpu_reference(feature_table)
    all_refs = [None for _ in range(world_size)]
    dist.all_gather_object(all_refs, cpu_ref)

    sampler = GpuShardSampler(
        cfg={"sampler_type": "gpu_shard", "test": "synthetic_lifecycle"},
        local_rank=rank,
        world_size=world_size,
        device=device,
        group=dist.group.WORLD,
        default_value=DEFAULT_ID,
    )
    sampler.prepare_reload(feature_table)
    sampler.commit_reload()
    dist.barrier()

    verify_valid_sample_ids(sampler, all_refs, rank, device)
    verify_combine(sampler, device)

    scenarios = [
        "balanced_all_ranks",
        "skew_rank0_and_missing",
        "local_only_duplicates",
        "empty_query",
    ]
    for scenario in scenarios:
        query_cpu = make_query(scenario, rank, world_size, all_refs)
        results = sampler.pack_feature(query_cpu.to(device), DEFAULT_ID)
        verify_results(scenario, query_cpu, results, all_refs, DEFAULT_ID, rank)
        # Determinism on repeated calls also exercises the cached default check.
        results2 = sampler.pack_feature(query_cpu.to(device), DEFAULT_ID)
        verify_results(
            scenario + "_repeat", query_cpu, results2, all_refs, DEFAULT_ID, rank
        )
        dist.barrier()
        if rank == 0:
            print(f"  {scenario}: verified")


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    ok = True
    try:
        if rank == 0:
            print("=== Synthetic GpuShardSampler lifecycle coverage test ===")
        run_rank()
    except Exception as exc:
        ok = False
        print(f"[rank {rank}] FAILED: {exc}")
        traceback.print_exc()

    device = f"cuda:{rank}"
    status = torch.tensor(1 if ok else 0, dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if rank == 0:
        if int(status.item()) == 1:
            print("=== Synthetic lifecycle coverage test PASSED ===")
        else:
            print("=== Synthetic lifecycle coverage test FAILED ===")

    dist.destroy_process_group()
    if int(status.item()) != 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
