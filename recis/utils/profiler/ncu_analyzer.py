"""Analyze ncu CSV profiling output and produce a DRAM memory access report.

This module provides reusable functions for parsing ncu CSV output,
aggregating DRAM bytes and GPU execution time by NVTX op, and generating
analysis reports with bandwidth calculations.

Usage:
    from recis.utils.profiler.ncu_analyzer import parse_ncu_csv, aggregate_by_op, generate_report

    records = parse_ncu_csv("ops.csv")
    ops = aggregate_by_op(records)
    report = generate_report(ops)
    # or save directly to file:
    generate_report(ops, output_file="report.log")
"""

import csv
import re
from collections import OrderedDict


def _extract_op_name(nvtx_field):
    """Extract the innermost 'op:xxx' from the NVTX Push/Pop range field.

    When NVTX ranges are nested (e.g. HashTable.forward wraps a
    torch.unique or gather call), ncu records all ranges in the field.
    We extract the LAST (innermost) match so each kernel is attributed
    to its most specific op, not the outer wrapper.

    Supports op names with dots (e.g. 'torch.unique',
    'HashTable.embedding_lookup') in addition to plain word names.
    """
    matches = re.findall(r"op:[\w.]+", nvtx_field)
    return matches[-1] if matches else "unknown"


def _parse_value(value_str):
    """Parse a metric value string, handling commas and decimals."""
    s = value_str.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _format_bytes(n):
    """Format bytes into human-readable string."""
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n:.0f} B"


def _format_time(ns):
    """Format nanoseconds into human-readable string."""
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.3f} us"
    return f"{ns:.0f} ns"


def _format_bandwidth(bytes_val, ns_val):
    """Format bandwidth as bytes/ns → GiB/s or MiB/s."""
    if ns_val <= 0 or bytes_val <= 0:
        return "N/A"
    bytes_per_sec = bytes_val / (ns_val / 1e9)
    gib = bytes_per_sec / (1024**3)
    if gib >= 1:
        return f"{gib:.2f} GiB/s"
    mib = bytes_per_sec / (1024**2)
    return f"{mib:.2f} MiB/s"


_METRIC_DRAM = "dram__bytes.sum"
_METRIC_TIME = "gpu__time_duration.sum"


def parse_ncu_csv(filepath):
    """Parse ncu CSV and return list of kernel call records.

    Each kernel call produces two CSV rows (one per metric).  This
    function pairs them by kernel ID (if available) or by encounter
    order, so multiple invocations of the same kernel are preserved
    as separate records.

    Returns a list of dicts with keys:
        op_name, kernel_name, dram_bytes, gpu_time_ns, bandwidth_gbs,
        kernel_id
    """
    # ── Phase 1: collect raw rows grouped by (op, kernel, metric) ──
    bytes_rows = []  # list of (kernel_id, op_name, kernel_name, value)
    time_rows = []  # list of (kernel_id, op_name, kernel_name, value)

    with open(filepath, newline="") as f:
        lines = [ln for ln in f if not ln.startswith("==")]
    reader = csv.DictReader(lines)

    nvtx_col = (
        "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
    )

    has_id = reader.fieldnames and "ID" in reader.fieldnames

    for row in reader:
        metric_name = row.get("Metric Name", "")
        if metric_name not in (_METRIC_DRAM, _METRIC_TIME):
            continue
        nvtx_field = row.get(nvtx_col, "")
        op_name = _extract_op_name(nvtx_field)
        kernel_name = row.get("Kernel Name", "")
        value = _parse_value(row.get("Metric Value", "0"))
        kid = int(_parse_value(row.get("ID", "0"))) if has_id else 0

        if metric_name == _METRIC_DRAM:
            bytes_rows.append((kid, op_name, kernel_name, value))
        else:
            time_rows.append((kid, op_name, kernel_name, value))

    # ── Phase 2: pair bytes and time rows ──
    records = []

    if has_id:
        # Pair by kernel_id for robustness
        time_map = {}
        for kid, op, kn, val in time_rows:
            time_map[kid] = (op, kn, val)

        for kid, op_name, kernel_name, dram_bytes in bytes_rows:
            gpu_time_ns = 0.0
            if kid in time_map:
                _, _, gpu_time_ns = time_map[kid]
            bw = dram_bytes / (gpu_time_ns / 1e9) if gpu_time_ns > 0 else 0.0
            records.append(
                {
                    "op_name": op_name,
                    "kernel_name": kernel_name,
                    "dram_bytes": int(dram_bytes),
                    "gpu_time_ns": gpu_time_ns,
                    "bandwidth_gbs": bw,
                    "kernel_id": kid,
                }
            )
    else:
        # Fallback: pair by encounter order
        time_iter = iter(time_rows)

        for _, op_name, kernel_name, dram_bytes in bytes_rows:
            gpu_time_ns = 0.0
            try:
                _, _, _, gpu_time_ns = next(time_iter)
            except StopIteration:
                pass

            bw = dram_bytes / (gpu_time_ns / 1e9) if gpu_time_ns > 0 else 0.0
            records.append(
                {
                    "op_name": op_name,
                    "kernel_name": kernel_name,
                    "dram_bytes": int(dram_bytes),
                    "gpu_time_ns": gpu_time_ns,
                    "bandwidth_gbs": bw,
                    "kernel_id": 0,
                }
            )

    return records


def _normalize_kernel_name(name):
    """Normalize kernel name for pattern matching.

    Strips template parameters (<...>) and common suffixes so that
    variants like 'CatArrayBatchedCopy_aligned16_cont' and
    'CatArrayBatchedCopy<at::native::<unnamed>::...' both reduce to
    'CatArrayBatchedCopy'.
    """
    # Strip template parameters
    idx = name.find("<")
    if idx >= 0:
        name = name[:idx]
    # Strip common suffixes
    for suffix in ("_aligned16_cont", "_aligned", "_cont"):
        sidx = name.find(suffix)
        if sidx >= 0:
            name = name[:sidx]
    return name


def _detect_calls_by_pattern(kernels):
    """Try to split a flat kernel list into calls by finding a repeating sequence.

    Uses normalized kernel names for comparison.  Returns a list of
    sub-lists (one per call), or None if no repeating pattern is found.

    Handles both evenly-divisible counts (n % L == 0) and non-divisible
    counts (n % L != 0), where the last call is a partial repetition.
    For example, 37 kernels with a 5-kernel pattern yields 7 complete
    calls plus 1 partial call with 2 kernels.

    Only triggered when gap-based detection produced a single call with
    >= 4 kernels (i.e. at least 2 complete repetitions of length >= 2).
    """
    n = len(kernels)
    if n < 4:
        return None

    normalized = [_normalize_kernel_name(k["name"]) for k in kernels]

    # Try pattern lengths from 2 to n//2 (min 2 kernels per call)
    for L in range(2, n // 2 + 1):
        pattern = normalized[:L]

        # Check if the pattern repeats throughout the kernel list.
        # Modulo indexing naturally validates partial last chunks too
        # (when n % L != 0, the trailing kernels must match the
        # corresponding prefix of the pattern).
        match = True
        for i in range(L, n):
            if normalized[i] != pattern[i % L]:
                match = False
                break
        if not match:
            continue

        # Require at least 2 complete repetitions to avoid false positives
        if n // L < 2:
            continue

        # Split into calls of length L (last chunk may be shorter)
        calls = []
        for start in range(0, n, L):
            calls.append(kernels[start : start + L])
        return calls

    return None


# Known "first kernel" patterns for ops whose kernel count varies per invocation
# (e.g. torch.unique: large inputs use 8 Onesweep passes, small inputs use
# SingleTile; HashTable.embedding_lookup: skip insert/vectorized when all keys
# are cached → variable kernel count per call, so gap/pattern detection fails).
# When a kernel name contains one of these substrings AND the current call
# already has kernels, a new call boundary is forced.
_CALL_START_KERNELS = {
    "elementwise_kernel_with_index",  # torch.unique first kernel
    "cuco::detail::find_and_mask",  # HashTable.embedding_lookup first kernel
}


def _is_call_start(kernel_name):
    """Check if a kernel name matches a known call-start pattern."""
    for pattern in _CALL_START_KERNELS:
        if pattern in kernel_name:
            return True
    return False


# Known "last kernel" patterns for ops where the final kernel of each
# invocation is distinctive (e.g. ragged_to_dense always ends with
# ragged_elementwise_to_dense_kernel).  After seeing one of these,
# the next kernel of the same op is forced to start a new call.
_CALL_END_KERNELS = {
    "ragged_elementwise_to_dense_kernel",  # ragged_to_dense last kernel
}


def _is_call_end(kernel_name):
    """Check if a kernel name matches a known call-end pattern."""
    for pattern in _CALL_END_KERNELS:
        if pattern in kernel_name:
            return True
    return False


def aggregate_by_op(records):
    """Group records by op_name, then split into calls (invocations).

    Call boundaries are detected in three phases:
    1. Gap-based + kernel-name-based: gaps in kernel_id (ncu "ID" column)
       or record index, known first-kernel patterns (call-start), and
       known last-kernel patterns (call-end) all force new call
       boundaries in Phase 1.
    2. Pattern-based fallback: for each call with >= 4 kernels that may
       contain multiple actual invocations, try to find a repeating
       kernel-name sequence.

    Returns an OrderedDict keyed by op_name, each value has:
        total_bytes, total_time_ns, bandwidth_gbs, calls
    Each call has: total_bytes, total_time_ns, bandwidth_gbs, kernels
    """
    ops = OrderedDict()

    # ── Phase 1: gap + call-start + call-end boundary detection ──
    for idx, r in enumerate(records):
        op = r["op_name"]
        if op not in ops:
            ops[op] = {
                "total_bytes": 0,
                "total_time_ns": 0.0,
                "calls": [],
                "_last_sort_key": -1,
                "_force_new_call": False,
            }

        op_data = ops[op]
        calls = op_data["calls"]

        # Determine sort key: use kernel_id if available (>0), else record index
        sort_key = r["kernel_id"] if r["kernel_id"] > 0 else idx

        # Detect call boundary:
        #   (a) first kernel of this op → new call
        #   (b) previous kernel was a call-end marker → new call
        #   (c) gap in sort key (other ops' kernels in between) → new call
        #   (d) kernel name matches a known call-start pattern → new call
        new_call = False
        if not calls:
            new_call = True
        elif op_data["_force_new_call"]:
            new_call = True
        elif calls[-1]["kernels"] and sort_key > op_data["_last_sort_key"] + 1:
            new_call = True
        elif calls[-1]["kernels"] and _is_call_start(r["kernel_name"]):
            new_call = True

        # Reset force flag — it applied to THIS kernel's boundary decision
        op_data["_force_new_call"] = False

        if new_call:
            calls.append({"total_bytes": 0, "total_time_ns": 0.0, "kernels": []})

        current_call = calls[-1]
        current_call["kernels"].append(
            {
                "name": r["kernel_name"],
                "bytes": r["dram_bytes"],
                "time_ns": r["gpu_time_ns"],
                "bandwidth_gbs": r["bandwidth_gbs"],
            }
        )
        current_call["total_bytes"] += r["dram_bytes"]
        current_call["total_time_ns"] += r["gpu_time_ns"]

        op_data["total_bytes"] += r["dram_bytes"]
        op_data["total_time_ns"] += r["gpu_time_ns"]
        op_data["_last_sort_key"] = sort_key

        # If this kernel is a call-end marker, force next kernel to new call
        if _is_call_end(r["kernel_name"]):
            op_data["_force_new_call"] = True

    # ── Phase 2: pattern-based fallback for each call ──
    for data in ops.values():
        new_calls = []
        for call in data["calls"]:
            kernels = call["kernels"]
            if len(kernels) >= 4:
                split = _detect_calls_by_pattern(kernels)
                if split and len(split) >= 2:
                    new_calls.extend(
                        {
                            "total_bytes": sum(k["bytes"] for k in chunk),
                            "total_time_ns": sum(k["time_ns"] for k in chunk),
                            "kernels": chunk,
                        }
                        for chunk in split
                    )
                    continue
            new_calls.append(call)
        data["calls"] = new_calls

    # ── Phase 3: compute bandwidths ──
    for data in ops.values():
        data.pop("_last_sort_key", None)
        data.pop("_force_new_call", None)
        t = data["total_time_ns"]
        data["bandwidth_gbs"] = data["total_bytes"] / (t / 1e9) if t > 0 else 0.0
        for call in data["calls"]:
            ct = call["total_time_ns"]
            call["bandwidth_gbs"] = call["total_bytes"] / (ct / 1e9) if ct > 0 else 0.0

    return ops


def generate_report(ops, output_file=None):
    """Generate a DRAM access and bandwidth analysis report.

    Args:
        ops (OrderedDict): Aggregated op data from aggregate_by_op().
        output_file (str, optional): If provided, write report to this file.
            If None, return report as a string.

    Returns:
        str: The report text (also written to file if output_file is given).
    """
    total_bytes = sum(op["total_bytes"] for op in ops.values())
    total_time = sum(op["total_time_ns"] for op in ops.values())
    total_kernels = sum(
        sum(len(c["kernels"]) for c in op["calls"]) for op in ops.values()
    )

    W = 110  # report width

    lines = []
    lines.append("=" * W)
    lines.append("  NCU DRAM Memory Access & Bandwidth Analysis Report")
    lines.append("=" * W)
    lines.append("")
    lines.append(
        f"  Total DRAM Access  : {_format_bytes(total_bytes)}  ({total_bytes:,} bytes)"
    )
    lines.append(
        f"  Total GPU Time     : {_format_time(total_time)}  ({total_time:,.0f} ns)"
    )
    lines.append(f"  Avg DRAM Bandwidth : {_format_bandwidth(total_bytes, total_time)}")
    lines.append(f"  Total Kernel Calls : {total_kernels}")
    lines.append(f"  Unique Ops         : {len(ops)}")
    lines.append("")

    # ── Op summary table ──
    lines.append("-" * W)
    hdr = (
        f"  {'Op Name':<40} {'DRAM Access':>14} {'GPU Time':>14}"
        f" {'Bandwidth':>16} {'Ratio':>8}"
    )
    lines.append(hdr)
    lines.append("-" * W)

    for op_name, data in ops.items():
        ratio = data["total_bytes"] / total_bytes * 100 if total_bytes else 0
        lines.append(
            f"  {op_name:<40} {_format_bytes(data['total_bytes']):>14}"
            f" {_format_time(data['total_time_ns']):>14}"
            f" {_format_bandwidth(data['total_bytes'], data['total_time_ns']):>16}"
            f" {ratio:>7.1f}%"
        )
    lines.append("-" * W)

    # ── Per-call breakdown ──
    lines.append("")
    lines.append("  Per-Call Breakdown:")
    lines.append("-" * W)

    for op_name, data in ops.items():
        lines.append(
            f"\n  [{op_name}]"
            f"  DRAM: {_format_bytes(data['total_bytes'])}"
            f"  Time: {_format_time(data['total_time_ns'])}"
            f"  BW: {_format_bandwidth(data['total_bytes'], data['total_time_ns'])}"
        )
        for call_idx, call in enumerate(data["calls"]):
            lines.append("")
            lines.append(
                f"    [call {call_idx}]"
                f"  DRAM: {_format_bytes(call['total_bytes'])}"
                f"  Time: {_format_time(call['total_time_ns'])}"
                f"  BW: {_format_bandwidth(call['total_bytes'], call['total_time_ns'])}"
            )
            for i, kern in enumerate(call["kernels"]):
                short_name = kern["name"]
                if len(short_name) > 65:
                    short_name = short_name[:62] + "..."
                lines.append(
                    f"      [{i + 1:>3}] {short_name:<67}"
                    f" {_format_bytes(kern['bytes']):>12}"
                    f" {_format_time(kern['time_ns']):>12}"
                    f" {_format_bandwidth(kern['bytes'], kern['time_ns']):>14}"
                )

    lines.append("")
    lines.append("=" * W)
    lines.append("  End of Report")
    lines.append("=" * W)

    report = "\n".join(lines)

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(report + "\n")

    return report
