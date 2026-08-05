"""Memory access tracking setup module.

This module provides centralized configuration for memory access tracking,
including formula registration and automatic operator wrapping.

Usage:
    from recis.utils.profiler.memory_access_setup import setup_memory_access_tracking
    setup_memory_access_tracking()  # Call once at program start
"""

import torch

from recis.utils.profiler.memory_access import MemoryAccessTracker


# ==================== Helper Functions ====================


def _calc_tensor_bytes(tensor: torch.Tensor) -> int:
    """Calculate the number of bytes in a tensor."""
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _calc_tensor_list_bytes(tensors) -> int:
    """Calculate total bytes in a list/tuple of tensors."""
    if tensors is None:
        return 0
    if isinstance(tensors, (list, tuple)):
        return sum(
            _calc_tensor_bytes(t) for t in tensors if isinstance(t, torch.Tensor)
        )
    elif isinstance(tensors, torch.Tensor):
        return _calc_tensor_bytes(tensors)
    return 0


def _safe_get_arg(args, idx, default=None):
    """Safely get argument by index."""
    if args is None or len(args) <= idx:
        return default
    return args[idx]


# ==================== Memory Access Formulas ====================


# ----- Data Processing -----
def _element_wise_op_formula(args, kwargs, output):
    """element_wise_op: Read input + Write output."""
    total = _calc_tensor_bytes(_safe_get_arg(args, 0))
    total += _calc_tensor_bytes(output)
    return total


def _list_element_wise_op_formula(args, kwargs, output):
    """list_element_wise_op: Read input + Write output."""
    total = _calc_tensor_list_bytes(_safe_get_arg(args, 0))
    total += _calc_tensor_list_bytes(output)
    return total


def _bucketize_op_formula(args, kwargs, output):
    """bucketize_op: Read input + Write output."""
    return _element_wise_op_formula(args, kwargs, output)


def _uint64_mod_formula(args, kwargs, output):
    """uint64_mod: Read input + Write output."""
    return _element_wise_op_formula(args, kwargs, output)


def _fused_uint64_mod_formula(args, kwargs, output):
    """fused_uint64_mod: Read values_list + Write output."""
    return _list_element_wise_op_formula(args, kwargs, output)


def _fused_bucketized_formula(args, kwargs, output):
    """fused_bucketized: Read values_list + Write output."""
    return _list_element_wise_op_formula(args, kwargs, output)


def _fused_multi_hash_formula(args, kwargs, output):
    """fused_multi_hash:
    Read inputs * multi_hash_num
    Write outputs.
    """
    total = 0
    inputs = _safe_get_arg(args, 0)
    hash_args = _safe_get_arg(args, 1)
    for inp, hash_arg in zip(inputs, hash_args):
        total = total + _calc_tensor_bytes(inp) * hash_arg.numel()
    total += _calc_tensor_list_bytes(output)
    return total


def _fused_hash_formula(args, kwargs, output):
    """fused_hash: Read inputs + Write outputs."""
    inputs = _safe_get_arg(args, 0)
    splits = _safe_get_arg(args, 1)
    total = _calc_tensor_list_bytes(inputs)
    total += _calc_tensor_list_bytes(splits)
    total += _calc_tensor_list_bytes(output)
    return total


def _fused_int64_to_string_int8_formula(args, kwargs, output):
    """fused_int64_to_string_int8 DRAM estimation (fitted from NCU measurements).

    Physical model:
        total_dram = raw_traffic x cache_factor

        raw_traffic = C_THEORY x input_bytes + write_amp(L) x output_bytes

        4 kernel groups per tensor (T tensors, N elements each):
        1. vectorized_elementwise: zeros output offsets → write offset
        2. fused_calculate_offsets: read input + write offsets
        3. DeviceScan (cumsum): read offset + write cumsum_offset
        4. fused_int64_to_string: read input + read offset + write output

        C_THEORY encompasses all base reads/writes (kernels 1-4 read paths):
            vec(1x+RFO) + calc(2x) + scan(2x) + string_read(2x) ≈ 8x

        Write amplification (string kernel only):
            Each thread writes L consecutive int8 bytes with warp stride ≈ L.
            L=1: perfect 32B sector coalescing, no amplification.
            L>1: partial sector writes → RFO → amp grows as saturating
            function write_amp = 1 + Ax(L-1)/(L+B).

        L2 cache factor (single unified mechanism):
            Combined pressure = ALPHA x per_tensor_mb + BETA x total_ws_mb
            - ALPHA x per_tensor_mb: intra-kernel L2 tile pressure
            - BETA x total_ws_mb: inter-kernel global L2 pressure
            cache_factor = sigmoid(pressure) → 0 when data fits L2, → 1 at scale
    """
    import math

    inputs = _safe_get_arg(args, 0)
    input_bytes = _calc_tensor_list_bytes(inputs)
    output_bytes = _calc_tensor_list_bytes(output[0])
    out_offset_bytes = _calc_tensor_list_bytes(output[1])

    # Derive num_tensors and avg string length
    if isinstance(inputs, (list, tuple)):
        num_tensors = len(inputs)
        total_elements = sum(t.numel() for t in inputs if isinstance(t, torch.Tensor))
    else:
        num_tensors = 1
        total_elements = inputs.numel() if isinstance(inputs, torch.Tensor) else 0
    avg_L = output_bytes / max(total_elements, 1)

    # --- Theoretical traffic (no L2 caching) ---
    # C_THEORY: total read/write coefficient for base kernels + string reads
    _C_THEORY = 8.0000
    base_and_read = _C_THEORY * input_bytes

    # L-dependent write amplification (partial sector RFO)
    # L=1: amp=1 (perfect coalescing), L→∞: amp→1+C_WRITE_A≈13.5
    _C_WRITE_A = 12.5300
    _C_WRITE_B = 41.5500
    write_amp = 1.0 + _C_WRITE_A * (avg_L - 1) / (avg_L + _C_WRITE_B)
    str_write = write_amp * output_bytes

    raw_total = base_and_read + str_write

    # --- Single L2 cache factor (combined pressure metric) ---
    _ALPHA = 4.4500  # per-tensor L2 pressure weight
    _BETA = 0.0600  # total working set pressure weight
    _L2_THRESH = 1.0000  # pressure threshold
    _L2_WIDTH = 20.0000  # sigmoid width
    per_tensor_mb = (input_bytes / max(num_tensors, 1)) / (1024 * 1024)
    total_ws_mb = (input_bytes + out_offset_bytes + output_bytes) / (1024 * 1024)
    l2_pressure = _ALPHA * per_tensor_mb + _BETA * total_ws_mb
    cache_factor = 1.0 / (1.0 + math.exp(-(l2_pressure - _L2_THRESH) / _L2_WIDTH))

    total = raw_total * cache_factor
    return max(total, input_bytes)


def _ids_encode_formula(args, kwargs, output):
    """ids_encode: Read ids_list + Write output."""
    return _list_element_wise_op_formula(args, kwargs, output)


def _ids_partition_formula(args, kwargs, output):
    """ids_partition DRAM estimation (fitted from NCU measurements).

    Physical model:
        23-kernel operator with radix sort (8 passes) + unique_by_key.
        C++ returns std::tuple<Tensor, Tensor, Tensor>:
            (unique_ids, segment_size, reverse_indice)
        Total DRAM has two main components:
        1. Input-dependent kernels (radix_onesweep, hash, hist, adj_diff,
           scan, unique_by_key, elem_idx, etc.): scale with input ids size
           and exhibit super-linear amplification at very large scale.
        2. Unique-dependent kernels (transform hash slice partition):
           scale with unique_ids output size.

        L2 cache sigmoid transition dampens traffic when the working set
        (estimated as ~2.5x input to model radix sort key/value pairs +
        auxiliary buffers) is small.
    """
    import math

    input_tensor = _safe_get_arg(args, 0)
    input_bytes = _calc_tensor_bytes(input_tensor)

    # output = (unique_ids, segment_size, reverse_indice)
    unique_ids_tensor = output[0]
    unique_bytes = _calc_tensor_bytes(unique_ids_tensor)

    input_mb = input_bytes / (1024**2)
    unique_mb = unique_bytes / (1024**2)

    # Working set for L2 cache modeling:
    # radix sort operates on (key_int64 + value_int32) pairs -> ~1.5x input
    # plus hash keys, adj_diff buffers, etc.
    working_set_mb = input_mb * 2.5

    # L2 cache effectiveness sigmoid
    _L2_THRESH_MB = 5.5723
    _L2_WIDTH_MB = 7.3971
    cache_factor = 1.0 / (
        1.0 + math.exp(-(working_set_mb - _L2_THRESH_MB) / _L2_WIDTH_MB)
    )

    # Large-scale amplification (L2 miss / write amplification at >100MB)
    _LARGE_AMP = 0.0946
    _LARGE_THRESH_MB = 876.4806
    _LARGE_WIDTH_MB = 279.8466
    large_factor = 1.0 + _LARGE_AMP / (
        1.0 + math.exp(-(working_set_mb - _LARGE_THRESH_MB) / _LARGE_WIDTH_MB)
    )

    # Fitted coefficients (MB per MB of respective tensor)
    _A_INPUT = 47.5052
    _B_UNIQUE = 4.8328
    _C_OVERHEAD_MB = 0.5857

    total_mb = (
        _A_INPUT * input_mb * large_factor + _B_UNIQUE * unique_mb + _C_OVERHEAD_MB
    ) * cache_factor

    return max(total_mb * (1024**2), input_bytes)


def _merge_offsets_formula(args, kwargs, output):
    """merge_offsets: Read offsets_list + Write output."""
    return _list_element_wise_op_formula(args, kwargs, output)


def _gen_segment_indices_by_offset_formula(args, kwargs, output):
    """gen_segment_indices_by_offset DRAM estimation (fitted from NCU measurements).

    Physical model:
        Kernel: 1 thread per segment, loop writes output[offset[i]+j] = i.
        Warp write stride = seg_len * element_size, causing L2 sector thrashing.

        DRAM traffic = offsets_read + output_bytes * amp(seg_len, N) * cache_factor

        - amp(seg_len, N): output amplification from combined RFO reads + write-back,
          measured by NCU and interpolated log-linearly between calibration points.
          Larger N means more concurrent thread blocks → more L2 pressure → higher amp.
        - cache_factor: sigmoid transition around L2 capacity (~6 MB).
          When total working set fits in L2, DRAM traffic is minimal.
    """
    import math

    offsets = _safe_get_arg(args, 0)
    offsets_bytes = _calc_tensor_bytes(offsets)
    output_bytes = _calc_tensor_bytes(output)

    num_segments = offsets.numel() - 1
    seg_len = output.numel() / max(num_segments, 1)
    total_data_bytes = offsets_bytes + output_bytes
    total_data_mb = total_data_bytes / (1024**2)

    # --- Output amplification factor (from NCU calibration) ---
    # Measured at N=2M (full L2 pressure, cache_factor≈1)
    _AMP_N2M = {1: 0.86, 2: 1.68, 4: 2.15, 8: 2.89, 16: 6.77}
    # Measured at N=200k (moderate L2 pressure, corrected for sigmoid)
    _AMP_N200K = {1: 0.0, 2: 0.70, 4: 2.13, 8: 2.87, 16: 5.62}

    amp_high = _interp_amp_log(seg_len, _AMP_N2M)
    amp_low = _interp_amp_log(seg_len, _AMP_N200K)

    # Blend amplification based on num_segments (log-scale)
    if num_segments <= 200_000:
        amp = amp_low
    elif num_segments >= 2_000_000:
        amp = amp_high
    else:
        t = math.log10(num_segments / 200_000) / math.log10(10)
        amp = amp_low + t * (amp_high - amp_low)

    # --- L2 cache transition (sigmoid) ---
    L2_THRESHOLD_MB = 6.0
    TRANSITION_WIDTH_MB = 1.5
    cache_factor = 1.0 / (
        1.0 + math.exp(-(total_data_mb - L2_THRESHOLD_MB) / TRANSITION_WIDTH_MB)
    )

    total = offsets_bytes * 1.05 + output_bytes * amp * cache_factor
    return max(total, offsets_bytes)


def _interp_amp_log(seg_len, amp_table):
    """Log-linear interpolation of amplification factor from calibration table."""
    import math

    keys = sorted(amp_table.keys())
    if seg_len <= keys[0]:
        return amp_table[keys[0]]
    if seg_len >= keys[-1]:
        # Power-law extrapolation from last two points
        k1, k2 = keys[-2], keys[-1]
        a1, a2 = amp_table[k1], amp_table[k2]
        if a1 > 0 and a2 > 0:
            power = math.log(a2 / a1) / math.log(k2 / k1)
            return a2 * (seg_len / k2) ** power
        return a2

    for j in range(len(keys) - 1):
        if keys[j] <= seg_len <= keys[j + 1]:
            k1, k2 = keys[j], keys[j + 1]
            a1, a2 = amp_table[k1], amp_table[k2]
            t = math.log(seg_len / k1) / math.log(k2 / k1)
            if a1 > 0 and a2 > 0:
                return a1 * (a2 / a1) ** t
            return a1 + t * (a2 - a1)
    return amp_table[keys[-1]]


# ----- Ragged Tensor -----


def _fused_ragged_cutoff_2d_formula(args, kwargs, output):
    """fused_ragged_cutoff_2D DRAM estimation (fitted from NCU measurements).

    Physical model:
        Kernel structure per invocation:
        1. post_cutoff_lens: compute keep/drop/pad lengths per row
        2. seg_scan: prefix-sum of cutoff lengths
        3. vectorized_elementwise: initialize output
        4. reduce: compute per-feature aggregates
        5. fused_ragged_cutoff_2D_kernel (truncation) or
           CatArrayBatchedCopy_aligned16_contig (padding)

        Key insights from kernel source analysis:
        - Truncation: kernel reads ONLY kept values (conditional read inside
          keep-check), NOT all input values. Write has minimal amplification.
        - Per-row metadata overhead (offsets->shmem, cutoff_offsets, drop/pad
          arrays) is independent of value dtype, scales with offset_elem_size.
        - Large-scale metadata amplification: when metadata arrays exceed L2,
          re-fetches increase per-row cost.
        - Padding path: CatArrayBatchedCopy copies input values sequentially.
    """
    import math

    in_values_list = _safe_get_arg(args, 0)
    in_offsets_list = _safe_get_arg(args, 1)
    out_values_list = output[0]

    in_values_bytes = _calc_tensor_list_bytes(in_values_list)
    in_offsets_bytes = _calc_tensor_list_bytes(in_offsets_list)
    out_values_bytes = _calc_tensor_list_bytes(out_values_list)

    # Derive total_rows and offset element size
    if isinstance(in_offsets_list, (list, tuple)) and len(in_offsets_list) > 0:
        off_elem_size = in_offsets_list[0].element_size()
        total_rows = sum(
            t.numel() - 1 for t in in_offsets_list if isinstance(t, torch.Tensor)
        )
    else:
        off_elem_size = 4
        total_rows = max((in_offsets_bytes // off_elem_size) - 1, 1)

    # Branch detection: CatArray path produces output with EXACTLY same total
    # bytes as input (at::cat just concatenates). Truncation kernel always
    # produces different size (rows × keep_len ≠ sum of original row lengths).
    is_cat_path = out_values_bytes >= in_values_bytes

    # --- Fitted parameters (V3 capped-log + L2 overflow, 21-point fit) ---
    # MAPE = 1.06% on 21 NCU calibration points
    _A_VAL = 2.1020  # truncation value I/O factor (read ~1.075 + write ~1.04)
    _B1_META = 5.6922  # metadata base factor (per rows × off_elem unit)
    _B2_META = 1.9469  # metadata log amplification coefficient
    _B3_CAP = (
        1.4978  # metadata log saturation cap (saturates at io_ratio ≈ e^1.5 ≈ 4.5)
    )
    _D1_CAT_VAL = 2.1106  # CatArray value I/O factor
    _D2_CAT_META = 9.0033  # CatArray metadata per-row overhead
    _C1_OFF = 1.0000  # aux: proportional to in_offsets
    _C2_ROW = 0.5413  # aux: per-row metadata factor
    _L2_THRESH = 1.6887  # L2 sigmoid threshold (MB)
    _L2_WIDTH = 2.0628  # L2 sigmoid transition width
    _E_OVERHEAD_MB = 0.0586  # small-data fixed overhead (MB)

    # L2 overflow constants (physics-based, not fitted)
    _META_L2_THRESH_MB = 12.0  # ≈ L2_capacity(40MB) / 3 arrays
    _META_L2_WIDTH_MB = 5.0  # sigmoid transition width
    _F_OVERFLOW = 2.14  # amplification factor (A100 memory hierarchy)

    if not is_cat_path:
        # TRUNCATION: fused_ragged_cutoff_2D_kernel
        # Reads only kept values (out_val) + per-row metadata arrays
        val_io = _A_VAL * out_values_bytes

        # Metadata DRAM: grows with ln(in/out ratio) but saturates at B3_CAP
        # Higher ratio → more "wasted" iterations (kernel iterates all input
        # positions, invalid ones still read drop_num/pad_num) → L2 thrashing
        io_ratio = max(in_values_bytes / max(out_values_bytes, 1.0), 1.0)
        meta_log = min(math.log(io_ratio), _B3_CAP)
        kernel_meta = (
            _B1_META * total_rows * off_elem_size * (1.0 + _B2_META * meta_log)
        )

        # L2 overflow correction: when metadata arrays exceed L2 capacity,
        # low-io-ratio cases suffer extra thrashing (arrays can't stay cached).
        # At high io_ratio (saturated), thrashing is already maximal → no extra effect.
        meta_per_array_mb = total_rows * off_elem_size / (1024 * 1024)
        overflow_sigmoid = 1.0 / (
            1.0
            + math.exp(-(meta_per_array_mb - _META_L2_THRESH_MB) / _META_L2_WIDTH_MB)
        )
        unsaturated_fraction = max(0.0, 1.0 - meta_log / _B3_CAP)
        meta_overflow = 1.0 + _F_OVERFLOW * unsaturated_fraction * overflow_sigmoid
        kernel_meta *= meta_overflow
    else:
        # CAT: CatArrayBatchedCopy_aligned16_contig
        # Copies all input values to contiguous output (no truncation)
        val_io = _D1_CAT_VAL * in_values_bytes
        kernel_meta = _D2_CAT_META * total_rows * off_elem_size

    # Auxiliary sub-kernels: post_cutoff_lens, seg_scan, vec_elementwise, reduce
    other = _C1_OFF * in_offsets_bytes + _C2_ROW * total_rows * off_elem_size

    raw_total = val_io + kernel_meta + other

    # --- L2 cache sigmoid: small data mostly served from L2 cache ---
    working_set_mb = (in_values_bytes + total_rows * off_elem_size * 4) / (1024 * 1024)
    cache_factor = 1.0 / (1.0 + math.exp(-(working_set_mb - _L2_THRESH) / _L2_WIDTH))

    # Total: blend between full-DRAM model and small-data overhead
    total = raw_total / (1024 * 1024) * cache_factor + (
        _E_OVERHEAD_MB * (1.0 - cache_factor)
    )
    return max(total * (1024 * 1024), in_offsets_bytes)


def _fused_ragged_cutoff_3d_formula(args, kwargs, output):
    """fused_ragged_cutoff_3D DRAM estimation (fitted from NCU measurements).

    Physical model:
        10-kernel fused operator. DRAM dominated by:
        - CatArrayBatchCopy (#9): gather-read input values + scatter-write output values
        - seg_gen_offsets (#6): generates output inner offsets

        Key effects:
        - Write amplification grows with actual write size (L2 sector thrashing
          from scatter-writes to non-contiguous positions)
        - L2 cache sigmoid transition dampens traffic for small data
        - Truncation: reduces effective read, with adaptive read_amp
          (localized reads → better L2 hit rate)
        - Padding: adds inner offset entries but not value elements (3D structure)
        - Write amp and offset overhead use actual (non-inflated) data sizes

    Accuracy: MAPE ~3.4% on 12 NCU calibration points covering:
        baseline, large-scale, L2-fit, short-seq, long-seq, large-inner,
        truncation, padding, int64, ultra-large, heavy-truncation,
        multi-tensor-padding scenarios.
    """
    import math

    in_values_list = _safe_get_arg(args, 0)
    in_outer_offsets_list = _safe_get_arg(args, 1)
    in_inner_offsets_list = _safe_get_arg(args, 2)
    out_values_list = output[0]
    out_inner_offsets_list = output[2]

    in_values_bytes = _calc_tensor_list_bytes(in_values_list)
    in_inner_offsets_bytes = _calc_tensor_list_bytes(in_inner_offsets_list)
    out_values_bytes = _calc_tensor_list_bytes(out_values_list)
    out_inner_offsets_bytes = _calc_tensor_list_bytes(out_inner_offsets_list)

    # --- Truncation / padding adaptation ---
    # Actual write: padding doesn't add values, truncation reduces them
    actual_write_bytes = min(in_values_bytes, out_values_bytes)
    # Truncation ratio: 1.0 = no truncation, <1 = truncation
    truncation_ratio = actual_write_bytes / max(in_values_bytes, 1)

    # Adaptive read_amp: truncation -> localized reads -> better L2 hit rate
    _READ_AMP = 1.6989
    _TRUNC_RD = 0.6796
    adaptive_read_amp = _READ_AMP * (_TRUNC_RD + (1.0 - _TRUNC_RD) * truncation_ratio)

    # Effective offset read: use min to avoid padding inflation
    effective_inner_offsets = min(in_inner_offsets_bytes, out_inner_offsets_bytes)
    offset_read_overhead = 0.3 * (effective_inner_offsets + in_inner_offsets_bytes)

    # Write amplification based on actual_write (not inflated out_values)
    actual_write_mb = actual_write_bytes / (1024**2)
    _WRITE_AMP_BASE = 0.6229
    _WRITE_AMP_SLOPE = 0.4339
    write_amp = _WRITE_AMP_BASE + _WRITE_AMP_SLOPE / (
        1.0 + math.exp(-(actual_write_mb - 50) / 30)
    )

    # Total data for L2 cache transition
    total_data_mb = (
        in_values_bytes
        + out_values_bytes
        + in_inner_offsets_bytes
        + out_inner_offsets_bytes
    ) / (1024**2)

    # L2 cache sigmoid transition
    _L2_THRESH = 8.4264
    _L2_WIDTH = 4.1860
    cache_factor = 1.0 / (1.0 + math.exp(-(total_data_mb - _L2_THRESH) / _L2_WIDTH))

    # --- CatArrayBatchCopy contribution (kernel #9) ---
    catarray = (
        adaptive_read_amp * actual_write_bytes
        + offset_read_overhead
        + write_amp * actual_write_bytes
    ) * cache_factor

    # --- seg_gen_offsets contribution (kernel #6) ---
    # Padding inflates out_inner_offsets but actual data is limited
    effective_seg_inner = (
        min(out_inner_offsets_bytes, in_inner_offsets_bytes * 2)
        + in_inner_offsets_bytes
    )
    seg_gen_data_mb = effective_seg_inner / (1024**2)
    seg_cache = 1.0 / (
        1.0 + math.exp(-(seg_gen_data_mb - _L2_THRESH * 0.5) / _L2_WIDTH)
    )
    _SEG_AMP = 0.3244
    seg_gen = effective_seg_inner * _SEG_AMP * seg_cache

    # --- Small kernel overhead (kernels #0-5, #7-8) ---
    _OVERHEAD_MB = 2.0
    overhead_bytes = _OVERHEAD_MB * 1024 * 1024 * cache_factor

    total = catarray + seg_gen + overhead_bytes
    return max(total, _calc_tensor_list_bytes(in_outer_offsets_list))


# ----- Embedding/Segment -----


def _ragged_tile_formula(args, kwargs, output):
    """ragged_tile: Read multiple inputs + Write output tuple."""
    inp_index = _safe_get_arg(args, 2)
    inp_offset = _safe_get_arg(args, 3)
    inp_emb = _safe_get_arg(args, 4)
    real_emb_size = inp_emb.shape[1] * inp_emb.element_size() * inp_index.numel()

    total = (
        real_emb_size + _calc_tensor_bytes(inp_index) + _calc_tensor_bytes(inp_offset)
    )
    total += _calc_tensor_bytes(output[0])
    return total


def _segment_sum_formula(args, kwargs, output):
    """segment_sum:
    1. fill zero to output
    2. segment sum:
        input: reverse_ids_size x emb_size(input) + reverse_ids + weight + segment_ids
        output: output x 2(atomic write, need one more read)
    """
    data = _safe_get_arg(args, 0)
    weight = _safe_get_arg(args, 1)
    indices = _safe_get_arg(args, 2)
    segment_ids = _safe_get_arg(args, 3)

    total = data.shape[1] * indices.numel() * data.element_size()
    total += _calc_tensor_bytes(weight)
    total += _calc_tensor_bytes(indices)
    total += _calc_tensor_bytes(segment_ids)
    total = total + _calc_tensor_bytes(output) * 3
    return total


def _segment_mean_formula(args, kwargs, output):
    """segment_mean:
    1. fill zero to output: tmp_weight
    2. segment weight sum: input_weight / type_size x 32 + segment_ids + tmp_weight / type_size x 32 x 2(atomic)
    2. segment weight norm:  input_weight / type_size x 32 + segment_ids + tmp_weight / type_size x 32 + output
    """

    input_weight = _safe_get_arg(args, 1)
    segment_ids = _safe_get_arg(args, 2)
    input_size = input_weight.numel()
    total = input_size * input_weight.element_size() * 2
    total = total + _calc_tensor_bytes(segment_ids) * 2
    total = total + input_size * 32 * 5
    return total


def _segment_reduce_forward_formula(args, kwargs, output):
    """segment_reduce_forward:
    1. zeros for output: output size
    2. reduce: indices_size x emb_type_size x emb_dim + indices + offsets + weight + output_size x 2(atomic)
    """
    unique_emb = _safe_get_arg(args, 0)
    weight = _safe_get_arg(args, 1)
    reverse_indices = _safe_get_arg(args, 2)
    offsets = _safe_get_arg(args, 3)

    total = unique_emb.shape[1] * unique_emb.element_size() * reverse_indices.numel()
    total = (
        total
        + _calc_tensor_bytes(weight)
        + _calc_tensor_bytes(reverse_indices)
        + _calc_tensor_bytes(offsets)
    )
    total = total + 3 * _calc_tensor_bytes(output)
    return total


# ----- Block Operations -----


def _gather_formula(args, kwargs, output):
    """gather: Read index + 2 x Write output."""
    index = _safe_get_arg(args, 0)

    total = _calc_tensor_bytes(index)
    total = total + 2 * _calc_tensor_bytes(output)
    return total


# ----- PyTorch Native Ops -----


def _torch_unique_formula(args, kwargs, output):
    """torch.unique DRAM estimation (fitted from NCU measurements).

    Two execution paths with distinct memory access patterns:

    Basic (return_inverse=False, return_counts=False):
        - radix_hist: read input (~1x input)
        - radix_onesweep: bit_num passes, sort key only (~2x input per pass)
        - select_sweep: read sorted + write unique output
        int64 → 8 radix passes; int32 → 4 radix passes.

    Inverse (return_inverse=True or return_counts=True):
        - elem_idx: write N*8 index array
        - radix_hist: read input
        - radix_onesweep: bit_num passes, sort (key, value) pairs
          kv_pair = N x (element_size + 8)
        - adj_diff: read sorted keys
        - scan: prefix sum on flags (~N*12)
        - scatter: random write of inverse indices (~9-11x N*8)
        - reduce_by_key: reduce by unique key (scales with M)

    L2 cache sigmoid dampens traffic for small working sets.
    """
    import math

    input_tensor = _safe_get_arg(args, 0)
    if input_tensor is None:
        return 0

    N = input_tensor.numel()
    element_size = input_tensor.element_size()
    bit_num = element_size  # 4 for int32, 8 for int64
    input_bytes = N * element_size

    # Detect execution path
    return_inverse = kwargs.get("return_inverse", False)
    return_counts = kwargs.get("return_counts", False)
    is_inverse = return_inverse or return_counts

    # Extract unique tensor from output
    if isinstance(output, torch.Tensor):
        unique_tensor = output
    elif isinstance(output, (tuple, list)) and len(output) > 0:
        unique_tensor = output[0]
    else:
        unique_tensor = None

    unique_bytes = _calc_tensor_bytes(unique_tensor) if unique_tensor is not None else 0

    input_mb = input_bytes / (1024**2)
    unique_mb = unique_bytes / (1024**2)

    # --- Small data path: SingleTile branch (N <= 2048) ---
    # When N is small, CUB uses a single-tile radix sort with very different
    # memory access patterns.  A direct linear model (no L2 sigmoid) is used.
    # Basic small MAPE = 5.37% on 5 NCU points.
    # Inverse small MAPE = 1.58% on 5 NCU points.
    _SMALL_N_THRESHOLD = 2048
    if N <= _SMALL_N_THRESHOLD:
        if is_inverse:
            # Inverse small: sort(kv) + scatter(nidx) + select(unique) + overhead
            kv_bytes = N * (element_size + 8)
            n_idx_bytes = N * 8
            _SMALL_A_SORT = 2.6728
            _SMALL_B_NIDX = 1.9105
            _SMALL_C_UNIQUE = 0.0000
            _SMALL_D_OVERHEAD_KB = 36.2111
            total_bytes = (
                _SMALL_A_SORT * kv_bytes
                + _SMALL_B_NIDX * n_idx_bytes
                + _SMALL_C_UNIQUE * unique_bytes
                + _SMALL_D_OVERHEAD_KB * 1024
            )
        else:
            # Basic small: sort(input) + sqrt(N) + overhead
            _SMALL_A_SORT = 0.7794
            _SMALL_B_SQRTN = 0.6142
            _SMALL_C_OVERHEAD_KB = 10.1695
            total_bytes = (
                _SMALL_A_SORT * input_bytes
                + _SMALL_B_SQRTN * math.sqrt(N) * 1024
                + _SMALL_C_OVERHEAD_KB * 1024
            )
        return max(total_bytes, input_bytes)

    if is_inverse:
        # --- Inverse path: sort (key, value) pairs ---
        kv_mb = N * (element_size + 8) / (1024**2)
        n_idx_mb = N * 8 / (1024**2)
        working_set_mb = kv_mb * 2.0

        # L2 cache sigmoid
        _L2_THRESH_MB = 9.9766
        _L2_WIDTH_MB = 9.8924
        cache_factor = 1.0 / (
            1.0 + math.exp(-(working_set_mb - _L2_THRESH_MB) / _L2_WIDTH_MB)
        )

        # Fitted coefficients
        _A_SORT = 1.9249  # per-pass sort amplification (~2.0 = read+write)
        _B_SCATTER = 18.8688  # elem_idx + scatter + scan + adj_diff per N*8
        _C_UNIQUE = 3.2495  # reduce_by_key + select per unique byte
        _D_OVERHEAD_MB = 9.0495

        total_mb = (
            _A_SORT * kv_mb * bit_num
            + _B_SCATTER * n_idx_mb
            + _C_UNIQUE * unique_mb
            + _D_OVERHEAD_MB
        ) * cache_factor
    else:
        # --- Basic path: sort key only ---
        working_set_mb = input_mb * 2.0

        # L2 cache sigmoid
        _L2_THRESH_MB = 1.5955
        _L2_WIDTH_MB = 2.8499
        cache_factor = 1.0 / (
            1.0 + math.exp(-(working_set_mb - _L2_THRESH_MB) / _L2_WIDTH_MB)
        )

        # Large-scale amplification (write amplification at >40MB working set)
        _LARGE_AMP = 0.2219
        _LARGE_THRESH_MB = 46.1064
        _LARGE_WIDTH_MB = 11.5765
        large_factor = 1.0 + _LARGE_AMP / (
            1.0 + math.exp(-(working_set_mb - _LARGE_THRESH_MB) / _LARGE_WIDTH_MB)
        )

        # Fitted coefficients
        _A_SORT = 2.0019  # per-pass sort amplification (~2.0 = read+write)
        _B_UNIQUE = 1.0376  # select write per unique byte
        _C_OVERHEAD_MB = 2.9489

        total_mb = (
            _A_SORT * input_mb * bit_num * large_factor
            + _B_UNIQUE * unique_mb
            + _C_OVERHEAD_MB
        ) * cache_factor

    return max(total_mb * (1024**2), input_bytes)


# ----- HashTable (Class Method) -----


def _hashtable_embedding_lookup_formula(args, kwargs, output):
    """HashTable.embedding_lookup:
    avg new index ratio: 0.15(M = 0.15N)
    Lookup Index
        1. lookup index and mask:
            input: N x (int64(ids) + (int64 + int64(map)) x log2(emb_size/640000)(table num))
            output: N x (bool + int64(index))
            Total: N x ((2 + log2(emb_size/640000) x 2) x int64 + 1 x bool)
        2. cumsum over mask for number to generate:
            N x (bool(mask) + int64(new index tmp) x 2 + int64(new index))
            Total: N x (3 x int64 + 1 x bool)
        3. generate new index by cumsum:
            M(number to generate) x int64(new index)
            Total: N x (0.15 x int64)
        4. split lookup result, get new ids and index:
            in: M x (int64(ids) + int64(mask_index)) + N x bool(mask)
            out: M x (int64(new_ids) + int64(new_idx_index))
            Total: N x (0.6 x int64 + 1 x bool)
        5. scatter lookup index result:
            in: M x (int64(new index indices) + int64(new index))
            out M x (int64(lookup out index, only change M) x 2(overhead))
            Total: N x (0.6 x int64)
        6. insert new index pair to hashtable
            in: M x (int64(new_ids) + int64(new_index)) x (1 + log2(emb_size/640000)(table num))
            out: M x (int64(new_ids) + int64(new_index)) x 2(atomic)
            Total: N x (0.6 x (1 + log2(emb_size/640000)) + 1.2) x int64
        Total: N x 8.525 x int64 + 3 x log2(emb_size/640000) x int64
               N x ((2 + log2(emb_size/640000) x 2) x int64 + 1 x bool) = log2(emb_size/640000) x 2 + 2.125
               N x (3 x int64 + 1 x bool) = 3.125
               N x (0.15 x int64) = 0.15
               N x (0.6 x int64 + 1 x bool) = 0.725
               N x (0.6 x int64) = 0.6
               N x (0.6 x (1 + log2(emb_size/640000)) + 1.2) x int64 = 1.8 + log2(emb_size/640000)
    Lookup Embedding
        1. increase block and initialize: M x (emb_dtype x emb_dim) = 0.15 x emb_dtype x emb_dim
        2. gather block: N x (int64(lookup index) + emb_dtype x emb_dim x 2(read write))) = 1 x int64 + 2 x emb_dtype x emb_dim
        Total: 2.15 x emb_dtype x emb_dim + int64
    """
    ht = _safe_get_arg(args, 0)
    real_ht_num = (ht.id_info()[1] // 640000).bit_length()
    ids = _safe_get_arg(args, 1)
    input_bytes = _calc_tensor_bytes(ids)
    total = (8.525 + 3 * real_ht_num) * input_bytes
    out_emb = output[1]
    out_emb_bytes = _calc_tensor_bytes(out_emb)
    total = total + input_bytes + 2.15 * out_emb_bytes
    return total


# ==================== Registration Functions ====================


def _register_all_formulas():
    """Register all memory access calculation formulas."""

    formulas = {
        # ----- Data Processing -----
        "bucketize_op": _bucketize_op_formula,
        "uint64_mod": _uint64_mod_formula,
        "fused_uint64_mod": _fused_uint64_mod_formula,
        "fused_bucketized": _fused_bucketized_formula,
        "fused_multi_hash": _fused_multi_hash_formula,
        "fused_hash": _fused_hash_formula,
        "ids_encode": _ids_encode_formula,
        "ids_partition": _ids_partition_formula,
        "merge_offsets": _merge_offsets_formula,
        "gen_segment_indices_by_offset": _gen_segment_indices_by_offset_formula,
        "fused_int64_to_string_int8": _fused_int64_to_string_int8_formula,
        # ----- Ragged Tensor -----
        "fused_ragged_cutoff_2D": _fused_ragged_cutoff_2d_formula,
        "fused_ragged_cutoff_3D": _fused_ragged_cutoff_3d_formula,
        "ragged_tile": _ragged_tile_formula,
        # ----- Embedding/Segment -----
        "segment_sum": _segment_sum_formula,
        "segment_mean": _segment_mean_formula,
        "segment_reduce_forward": _segment_reduce_forward_formula,
        # ----- Block Operations -----
        "gather": _gather_formula,
        # ----- PyTorch Native Ops -----
        "torch.unique": _torch_unique_formula,
        # ----- Class Methods -----
        "HashTable.embedding_lookup": _hashtable_embedding_lookup_formula,
    }

    for op_name, formula in formulas.items():
        MemoryAccessTracker.register_formula_direct(op_name, formula)


def _wrap_torch_ops():
    """Register torch.ops.recis operators for tracking.

    This function only registers operator info for later patching.
    Actual patching happens when MemoryAccessTracker.context() is entered.
    """
    recis_op_names = [
        # Data Processing
        "bucketize_op",
        "uint64_mod",
        "fused_uint64_mod",
        "fused_bucketized",
        "fused_multi_hash",
        "fused_hash",
        "ids_encode",
        "ids_partition",
        "merge_offsets",
        "gen_segment_indices_by_offset",
        "fused_int64_to_string_int8",
        # Ragged Tensor
        "fused_ragged_cutoff_2D",
        "fused_ragged_cutoff_3D",
        "ragged_tile",
        # Embedding/Segment
        "segment_sum",
        "segment_mean",
        "segment_reduce_forward",
        # Block Operations
        "gather",
    ]

    try:
        recis_ops = torch.ops.recis
        for op_name in recis_op_names:
            if hasattr(recis_ops, op_name):
                MemoryAccessTracker.wrap_module_function(recis_ops, op_name, op_name)
    except Exception as e:
        print(f"Warning: Failed to wrap torch.ops.recis: {e}")

    # Wrap torch.unique
    try:
        MemoryAccessTracker.wrap_module_function(torch, "unique", "torch.unique")
    except Exception as e:
        print(f"Warning: Failed to wrap torch.unique: {e}")


def _wrap_custom_classes():
    """Register custom class methods for tracking.

    This function only registers operator info for later patching.
    Actual patching happens when MemoryAccessTracker.context() is entered.
    """
    try:
        from recis.nn.modules.hashtable import HashTable

        MemoryAccessTracker.wrap_class_method(
            HashTable, "_embedding_lookup_internal", "HashTable.embedding_lookup"
        )
    except ImportError as e:
        print(f"Warning: Failed to register HashTable class: {e}")


# ==================== Initialization ====================

_initialized = False


def setup_memory_access_tracking():
    """Initialize memory access tracking.

    This function:
    1. Registers all memory access calculation formulas
    2. Registers operators for later patching (no actual patching here)

    Operators are only patched when entering MemoryAccessTracker.context(),
    and unpatched when exiting. This ensures zero overhead when not tracking.

    Note:
        Call this function once at program start, after all modules are imported.
    """
    global _initialized
    if _initialized:
        return

    # Register all formulas
    _register_all_formulas()

    # Register operators for patching (no actual patching here)
    _wrap_torch_ops()

    # Register custom class methods for patching
    _wrap_custom_classes()

    _initialized = True
    print(
        "Memory access tracking initialized. Registered ops:",
        MemoryAccessTracker.list_registered_ops(),
    )


def is_initialized() -> bool:
    """Check if memory access tracking has been initialized.

    Returns:
        True if initialized, False otherwise.
    """
    return _initialized
