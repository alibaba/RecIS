"""High-level NCU profiling API for recis operators and related ops.

Wraps ncu command construction, execution, and CSV analysis into a single
Python call. Designed to work with NvtxProfileHook for low-overhead
profiling of the last training step.

Supports profiling of:
  - All torch.ops.recis operators (RECIS_OPS)
  - torch.unique and HashTable.embedding_lookup (NON_RECIS_OPS)

Usage:
    from recis.utils.profiler.ncu_profiler import run_ncu_profiling

    # Profile all ops (recis + torch.unique + HashTable)
    run_ncu_profiling(
        script="train.py",
        args="--config config.yaml",
        output_dir="./output_dir/",
    )

    # Profile specific ops only
    run_ncu_profiling(
        script="train.py",
        ops=["segment_sum", "fused_multi_hash"],
        output_dir="./output_dir/",
    )
"""

import os
import subprocess
import sys

from recis.utils.logger import Logger
from recis.utils.profiler.ncu_analyzer import (
    aggregate_by_op,
    generate_report,
    parse_ncu_csv,
)


# All recis operator names registered via m.def() in csrc/bind.cc
RECIS_OPS = [
    "block_apply_adamw",
    "block_filter",
    "block_gather",
    "block_insert",
    "block_insert_with_mask",
    "bucketize_op",
    "combine_vector_with_sample_counts",
    "dense_to_ragged",
    "feature_cross_ragged",
    "free_ids",
    "fused_adamw_tf_apply",
    "fused_bucketized",
    "fused_hash",
    "fused_int64_to_string_int8",
    "fused_multi_hash",
    "fused_ragged_cutoff_2D",
    "fused_ragged_cutoff_3D",
    "fused_string_mask",
    "fused_uint64_mod",
    "gather",
    "gather_ragged_by_padded_index",
    "gauc_calc",
    "gen_segment_indices_by_offset",
    "generate_ids",
    "ids_encode",
    "ids_partition",
    "make_hashtable",
    "mask_key_index",
    "merge_offsets",
    "ragged_tile",
    "ragged_tile_back",
    "ragged_to_dense",
    "ragged_to_sparse",
    "scatter_ids_with_mask",
    "segment_mean",
    "segment_reduce_backward",
    "segment_reduce_forward",
    "segment_sum",
    "tile_with_sample_counts",
    "uint64_mod",
]

# Non-recis ops wrapped by nvtx_wrapper.py (torch.unique, HashTable)
NON_RECIS_OPS = [
    "torch.unique",
    "HashTable.embedding_lookup",
]

# All ops that can be profiled (recis + non-recis)
ALL_PROFILEABLE_OPS = RECIS_OPS + NON_RECIS_OPS

logger = Logger("NcuProfiler")


def run_ncu_profiling(
    script,
    args="",
    output_dir="./",
    ops="all",
    metrics="dram__bytes.sum,gpu__time_duration.sum",
    ncu_path="ncu",
    profile_step=None,
):
    """Run ncu profiling on a training script and analyze the results.

    Builds and executes an ncu command with NVTX filtering for the specified
    recis operators, then parses the CSV output and generates a DRAM analysis
    report to a file. All output files are saved to output_dir.

    Always uses --profile-from-start off (requires NvtxProfileHook in the
    training script to call torch.cuda.profiler.start/stop).

    Args:
        script (str): Path to the Python training script to profile.
        args (str): Command-line arguments to pass to the script. Defaults to "".
        output_dir (str): Directory for all output files (CSV, report).
            Defaults to "./".
        ops (str or list): Which operators to profile. "all" for all
            profileable ops (recis + torch.unique + HashTable), or a
            list of op names. Defaults to "all".
        metrics (str): ncu metrics to collect.
            Defaults to "dram__bytes.sum,gpu__time_duration.sum".
        ncu_path (str): Path to ncu binary. Defaults to "ncu".
        profile_step (int, optional): Unused. NvtxProfileHook controls the
            profile step via its constructor argument (default 23).
            Kept for API compatibility.

    Returns:
        dict: Aggregated profiling results keyed by op name.

    Example:
        >>> from recis.utils.profiler import run_ncu_profiling
        >>> run_ncu_profiling(
        ...     script="train.py",
        ...     args="--config config.yaml",
        ...     output_dir="./output_dir/",
        ... )
    """
    python_path = sys.executable

    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "ops.csv")
    output_report = os.path.join(output_dir, "ncu_report.log")

    # Determine which ops to profile
    if ops == "all":
        ops_list = ALL_PROFILEABLE_OPS
    elif isinstance(ops, str):
        ops_list = [ops]
    else:
        ops_list = list(ops)

    # Validate op names
    for op in ops_list:
        if op not in ALL_PROFILEABLE_OPS:
            logger.warning(f"Unknown op '{op}', not in ALL_PROFILEABLE_OPS list")

    # Build ncu command (always --profile-from-start off)
    cmd = [ncu_path, "--csv", "--log-file", output_csv, "--nvtx"]
    for op in ops_list:
        cmd.append("--nvtx-include")
        cmd.append(f"op:{op}/")
    cmd.append("--metrics")
    cmd.append(metrics)
    cmd.append("--profile-from-start")
    cmd.append("off")
    cmd.append(python_path)
    cmd.append(script)
    if args:
        cmd.extend(args.split())

    logger.info(f"Running ncu with {len(ops_list)} ops")
    logger.info(f"CSV output: {output_csv}")

    # Run ncu (output flows to terminal, inherits parent environment)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"ncu failed with return code {result.returncode}")
        return {}

    # Analyze CSV output
    if not os.path.exists(output_csv):
        logger.error(f"CSV file not found: {output_csv}")
        return {}

    records = parse_ncu_csv(output_csv)
    if not records:
        logger.warning("No profiling records found in the CSV file.")
        return {}

    ops_data = aggregate_by_op(records)

    generate_report(ops_data, output_file=output_report)
    logger.info(f"Report saved to {output_report}")
    return ops_data
