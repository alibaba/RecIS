// Fused pack kernels: route_ids, unified_gather, serialize, deserialize
//
// All kernels use the unified representation (value_vec / offsets_vec /
// flat_bytes_vec) to treat dense and ragged features identically.

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "fused_pack.h"

// 所有 kernel 启动都跟随调用方的 current stream（torch.cuda.stream 上下文），
// 与函数内部的 ATen 操作同流，避免 legacy 默认流与 current stream 不一致。

// ════════════════════════════════════════════════════════════════════════
// 1. fused_route_ids: route + replace in single kernel
// ════════════════════════════════════════════════════════════════════════

__global__ void fused_route_ids_kernel(const int64_t* __restrict__ ids,
                                       int64_t* __restrict__ out_ids,
                                       int64_t* __restrict__ target_ranks,
                                       bool* __restrict__ found_mask,
                                       const int64_t* __restrict__ sorted_ptrs,
                                       const int* __restrict__ sorted_sizes,
                                       int64_t default_value, int world_size,
                                       int Q) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;

  int64_t id = ids[q];
  int64_t found_rank = -1;
  bool found = false;

  // Search for original ID
  for (int r = 0; r < world_size && !found; r++) {
    const int64_t* sorted = reinterpret_cast<const int64_t*>(sorted_ptrs[r]);
    int n = sorted_sizes[r];
    if (n == 0) continue;
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
      int mid = lo + (hi - lo) / 2;
      int64_t val = sorted[mid];
      if (val == id) {
        found = true;
        found_rank = r;
        break;
      }
      if (val < id)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
  }

  // If not found, search for default_value
  if (!found) {
    id = default_value;
    for (int r = 0; r < world_size && !found; r++) {
      const int64_t* sorted = reinterpret_cast<const int64_t*>(sorted_ptrs[r]);
      int n = sorted_sizes[r];
      if (n == 0) continue;
      int lo = 0, hi = n - 1;
      while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int64_t val = sorted[mid];
        if (val == id) {
          found = true;
          found_rank = r;
          break;
        }
        if (val < id)
          lo = mid + 1;
        else
          hi = mid - 1;
      }
    }
  }

  target_ranks[q] = found_rank;
  found_mask[q] = found;
  out_ids[q] = found ? id : default_value;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_route_ids(
    torch::Tensor ids, torch::Tensor sorted_ptrs, torch::Tensor sorted_sizes,
    int64_t default_value, int64_t world_size) {
  int Q = ids.size(0);
  auto opts = ids.options();
  auto out_ids = torch::empty({Q}, opts);
  auto target_ranks = torch::full({Q}, -1, opts);
  auto found_mask =
      torch::zeros({Q}, torch::dtype(torch::kBool).device(ids.device()));

  if (Q == 0 || world_size == 0)
    return std::make_tuple(out_ids, target_ranks, found_mask);

  int block = 256;
  int grid = (Q + block - 1) / block;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_route_ids_kernel<<<grid, block, 0, stream>>>(
      ids.data_ptr<int64_t>(), out_ids.data_ptr<int64_t>(),
      target_ranks.data_ptr<int64_t>(), found_mask.data_ptr<bool>(),
      sorted_ptrs.data_ptr<int64_t>(), sorted_sizes.data_ptr<int>(),
      default_value, (int)world_size, Q);

  return std::make_tuple(out_ids, target_ranks, found_mask);
}

// ════════════════════════════════════════════════════════════════════════
// 2. unified_batched_gather: single kernel for all entries
// ════════════════════════════════════════════════════════════════════════

__global__ void fused_begins_lengths_unified_kernel(
    const int64_t* __restrict__ flat_offsets,    // [num_unique, num_items+1]
    const int64_t* __restrict__ offset_indices,  // [E]
    const int64_t* __restrict__ indices,         // [Q]
    int64_t* __restrict__ begins,                // [E, Q]
    int64_t* __restrict__ lengths,               // [E, Q]
    int E, int Q, int num_items) {
  int f = blockIdx.y;
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= E || q >= Q) return;

  int off_idx = offset_indices[f];
  int64_t idx = indices[q];
  int64_t b = flat_offsets[off_idx * (num_items + 1) + idx];
  int64_t e = flat_offsets[off_idx * (num_items + 1) + idx + 1];
  begins[f * Q + q] = b;
  lengths[f * Q + q] = e - b;
}

__global__ void fused_segment_copy_unified_kernel(
    const int64_t* __restrict__ begins,       // [E, Q]
    const int64_t* __restrict__ out_offsets,  // [E, Q+1]
    const int64_t* __restrict__ ptrs,         // [E * 2] (src, dst as int64)
    const int64_t* __restrict__ flat_bytes,   // [E]
    int E, int Q) {
  int f = blockIdx.y;
  int seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= E || seg >= Q) return;

  const uint8_t* src = reinterpret_cast<const uint8_t*>(ptrs[f * 2]);
  uint8_t* dst = reinterpret_cast<uint8_t*>(ptrs[f * 2 + 1]);

  int64_t bytes = flat_bytes[f];
  int64_t begin = begins[f * Q + seg];
  int64_t out_off = out_offsets[f * (Q + 1) + seg];
  int64_t length = out_offsets[f * (Q + 1) + seg + 1] - out_off;

  int64_t copy_bytes = length * bytes;
  int64_t src_byte_off = begin * bytes;
  int64_t dst_byte_off = out_off * bytes;
  for (int64_t i = 0; i < copy_bytes; i++) {
    dst[dst_byte_off + i] = src[src_byte_off + i];
  }
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> unified_batched_gather(
    torch::Tensor flat_offsets, torch::Tensor offset_indices,
    torch::Tensor value_ptrs, torch::Tensor flat_bytes, torch::Tensor indices) {
  int E = offset_indices.size(0);
  int Q = indices.size(0);
  auto opts = indices.options();

  if (E == 0 || Q == 0) {
    return std::make_tuple(torch::empty({0, Q}, opts),
                           std::vector<torch::Tensor>{});
  }

  int num_items = flat_offsets.size(1) - 1;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Phase 1: fused begins/lengths
  auto begins = torch::empty({E, Q}, opts);
  auto all_lengths = torch::empty({E, Q}, opts);
  dim3 grid1((Q + 255) / 256, E);
  fused_begins_lengths_unified_kernel<<<grid1, 256, 0, stream>>>(
      flat_offsets.data_ptr<int64_t>(), offset_indices.data_ptr<int64_t>(),
      indices.data_ptr<int64_t>(), begins.data_ptr<int64_t>(),
      all_lengths.data_ptr<int64_t>(), E, Q, num_items);

  // Phase 2: output offsets + totals (1 sync)
  auto all_out_offsets = torch::zeros({E, Q + 1}, opts);
  all_out_offsets.narrow(1, 1, Q) = all_lengths.cumsum(1);
  torch::Tensor totals = all_out_offsets.select(1, Q).cpu();

  // Phase 3: allocate + launch segment copy
  auto flat_bytes_cpu = flat_bytes.cpu();
  auto value_ptrs_cpu = value_ptrs.cpu();
  auto ptrs_cpu = torch::empty({E, 2}, torch::dtype(torch::kInt64));
  std::vector<torch::Tensor> all_out_values;
  for (int f = 0; f < E; f++) {
    int64_t total = totals[f].item<int64_t>();
    int64_t bytes = flat_bytes_cpu[f].item<int64_t>();
    int64_t total_bytes = total * bytes;
    auto out_values = torch::empty(
        {total_bytes}, torch::dtype(torch::kUInt8).device(indices.device()));
    ptrs_cpu[f][0] = value_ptrs_cpu[f].item<int64_t>();
    ptrs_cpu[f][1] = reinterpret_cast<int64_t>(out_values.data_ptr());
    all_out_values.push_back(out_values);
  }
  auto ptrs = ptrs_cpu.to(opts);

  constexpr int BLOCK_SIZE = 256;
  dim3 grid2((Q + BLOCK_SIZE - 1) / BLOCK_SIZE, E);
  fused_segment_copy_unified_kernel<<<grid2, BLOCK_SIZE, 0, stream>>>(
      begins.data_ptr<int64_t>(), all_out_offsets.data_ptr<int64_t>(),
      ptrs.data_ptr<int64_t>(), flat_bytes.data_ptr<int64_t>(), E, Q);

  return std::make_tuple(all_lengths, all_out_values);
}

// ════════════════════════════════════════════════════════════════════════
// 3. fused_serialize: all ranks → flat byte buffer
// ════════════════════════════════════════════════════════════════════════

__global__ void fused_serialize_kernel(
    const int64_t* __restrict__ all_lengths,             // [E, total_recv]
    const int64_t* __restrict__ value_ptrs,              // [E] device pointers
    const int64_t* __restrict__ source_offsets,          // [world_size+1]
    const int64_t* __restrict__ val_starts,              // [E, world_size+1]
    const int64_t* __restrict__ flat_bytes,              // [E]
    const int64_t* __restrict__ send_offsets,            // [world_size+1]
    const int64_t* __restrict__ entry_val_byte_offsets,  // [E+1, world_size]
    uint8_t* __restrict__ send_flat,                     // [total_bytes]
    int E, int world_size, int total_recv) {
  // 1D grid: blockIdx.x = r * E + f
  int r = blockIdx.x / E;
  int f = blockIdx.x % E;
  if (r >= world_size) return;

  int start = source_offsets[r];
  int N = source_offsets[r + 1] - start;
  if (N == 0) return;

  uint8_t* dst = send_flat + send_offsets[r];
  int64_t bytes = flat_bytes[f];

  // Section 1: all_lengths entry f for rank r
  // Layout: [E, N] row-major, entry f at offset f * N * 8
  int64_t lengths_off = (int64_t)f * N * 8;
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    ((int64_t*)(dst + lengths_off))[i] =
        all_lengths[f * total_recv + start + i];

  // Section 2: all_values entry f for rank r
  // all_values starts after all_lengths: E * N * 8 bytes
  int64_t vals_section_start = (int64_t)E * N * 8;
  int64_t val_start = val_starts[f * (world_size + 1) + r];
  int64_t val_end = val_starts[f * (world_size + 1) + r + 1];
  int64_t val_count = val_end - val_start;
  int64_t val_byte_off = entry_val_byte_offsets[f * world_size + r];
  const uint8_t* src =
      reinterpret_cast<const uint8_t*>(value_ptrs[f]) + val_start * bytes;
  int64_t total_bytes = val_count * bytes;
  for (int64_t i = threadIdx.x; i < total_bytes; i += blockDim.x)
    dst[vals_section_start + val_byte_off + i] = src[i];
}

void fused_serialize(torch::Tensor all_lengths, torch::Tensor value_ptrs,
                     torch::Tensor source_offsets, torch::Tensor val_starts,
                     torch::Tensor flat_bytes, torch::Tensor send_offsets,
                     torch::Tensor entry_val_byte_offsets,
                     torch::Tensor send_flat, int64_t E, int64_t world_size,
                     int64_t total_recv) {
  if (total_recv == 0) return;
  int grid = (int)(world_size * E);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_serialize_kernel<<<grid, 1024, 0, stream>>>(
      all_lengths.data_ptr<int64_t>(), value_ptrs.data_ptr<int64_t>(),
      source_offsets.data_ptr<int64_t>(), val_starts.data_ptr<int64_t>(),
      flat_bytes.data_ptr<int64_t>(), send_offsets.data_ptr<int64_t>(),
      entry_val_byte_offsets.data_ptr<int64_t>(), send_flat.data_ptr<uint8_t>(),
      (int)E, (int)world_size, (int)total_recv);
}

// ════════════════════════════════════════════════════════════════════════
// 4. fused_deserialize_fixed: parse all_lengths, scatter to query order
// ════════════════════════════════════════════════════════════════════════

__global__ void fused_deserialize_fixed_kernel(
    const uint8_t* __restrict__ recv_flat,
    const int64_t* __restrict__ recv_offsets,     // [world_size+1]
    const int64_t* __restrict__ recv_counts,      // [world_size]
    const int64_t* __restrict__ scatter_indices,  // [total_items]
    const int64_t* __restrict__ scatter_offsets,  // [world_size+1]
    int64_t* __restrict__ out_lengths,            // [E, Q]
    int64_t* __restrict__ recv_lengths,           // [E, total_items]
    int E, int world_size, int Q, int total_items) {
  int r = blockIdx.x;
  if (r >= world_size) return;

  int N = recv_counts[r];
  if (N == 0) return;

  int base = scatter_offsets[r];
  const uint8_t* src = recv_flat + recv_offsets[r];

  // Section 1: all_lengths [E, N] → scatter
  const int64_t* lengths_src = reinterpret_cast<const int64_t*>(src);
  for (int f = 0; f < E; f++) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
      int64_t val = lengths_src[f * N + i];
      int64_t out_pos = scatter_indices[base + i];
      out_lengths[f * Q + out_pos] = val;
      recv_lengths[f * total_items + base + i] = val;
    }
  }
}

void fused_deserialize_fixed(
    torch::Tensor recv_flat, torch::Tensor recv_offsets,
    torch::Tensor recv_counts, torch::Tensor scatter_indices,
    torch::Tensor scatter_offsets, torch::Tensor out_lengths,
    torch::Tensor recv_lengths, int64_t E, int64_t world_size, int64_t Q,
    int64_t total_items) {
  if (total_items == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_deserialize_fixed_kernel<<<(int)world_size, 1024, 0, stream>>>(
      recv_flat.data_ptr<uint8_t>(), recv_offsets.data_ptr<int64_t>(),
      recv_counts.data_ptr<int64_t>(), scatter_indices.data_ptr<int64_t>(),
      scatter_offsets.data_ptr<int64_t>(), out_lengths.data_ptr<int64_t>(),
      recv_lengths.data_ptr<int64_t>(), (int)E, (int)world_size, (int)Q,
      (int)total_items);
}

// ════════════════════════════════════════════════════════════════════════
// 5. fused_deserialize_values: parse values, direct scatter to final positions
// ════════════════════════════════════════════════════════════════════════

__global__ void fused_deserialize_values_kernel(
    const uint8_t* __restrict__ recv_flat,
    const int64_t* __restrict__ recv_offsets,     // [world_size+1]
    const int64_t* __restrict__ recv_counts,      // [world_size]
    const int64_t* __restrict__ scatter_indices,  // [total_items]
    const int64_t* __restrict__ scatter_offsets,  // [world_size+1]
    const int64_t* __restrict__ recv_lengths,     // [E, total_items]
    const int64_t* __restrict__ src_offsets,  // [E, total_items] precomputed
                                              // exclusive prefix sum per rank
                                              // section
    const int64_t* __restrict__ all_offsets,  // [E, Q+1]
    const int64_t* __restrict__ feat_val_starts,         // [E+1] byte offset
    const int64_t* __restrict__ flat_bytes,              // [E]
    const int64_t* __restrict__ entry_buf_byte_offsets,  // [E+1, world_size]
    uint8_t* __restrict__ out_values,                    // [total_vals_bytes]
    int E, int world_size, int Q, int total_items) {
  int r = blockIdx.x;
  int f = blockIdx.y;
  if (r >= world_size || f >= E) return;

  int N = recv_counts[r];
  if (N == 0) return;

  int base = scatter_offsets[r];
  int64_t bytes = flat_bytes[f];

  // all_values section starts after all_lengths section
  int64_t vals_section_start = (int64_t)E * N * 8;

  // This entry's values start in the buffer
  int64_t buf_byte_off = entry_buf_byte_offsets[f * world_size + r];
  const uint8_t* src =
      recv_flat + recv_offsets[r] + vals_section_start + buf_byte_off;

  // Read lengths for this (rank, entry)
  const int64_t* lengths = recv_lengths + f * total_items + base;

  // Direct scatter to final positions using precomputed src_offsets
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int64_t query_pos = scatter_indices[base + i];
    int64_t dst_byte =
        feat_val_starts[f] + all_offsets[f * (Q + 1) + query_pos] * bytes;
    int64_t src_byte = src_offsets[f * total_items + base + i] * bytes;
    int64_t len = lengths[i];
    int64_t copy_bytes = len * bytes;
    for (int64_t k = 0; k < copy_bytes; k++)
      out_values[dst_byte + k] = src[src_byte + k];
  }
}

void fused_deserialize_values(
    torch::Tensor recv_flat, torch::Tensor recv_offsets,
    torch::Tensor recv_counts, torch::Tensor scatter_indices,
    torch::Tensor scatter_offsets, torch::Tensor recv_lengths,
    torch::Tensor src_offsets, torch::Tensor all_offsets,
    torch::Tensor feat_val_starts, torch::Tensor flat_bytes,
    torch::Tensor entry_buf_byte_offsets, torch::Tensor out_values, int64_t E,
    int64_t world_size, int64_t Q, int64_t total_items) {
  if (total_items == 0 || E == 0) return;

  dim3 grid((int)world_size, (int)E);
  int block = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_deserialize_values_kernel<<<grid, block, 0, stream>>>(
      recv_flat.data_ptr<uint8_t>(), recv_offsets.data_ptr<int64_t>(),
      recv_counts.data_ptr<int64_t>(), scatter_indices.data_ptr<int64_t>(),
      scatter_offsets.data_ptr<int64_t>(), recv_lengths.data_ptr<int64_t>(),
      src_offsets.data_ptr<int64_t>(), all_offsets.data_ptr<int64_t>(),
      feat_val_starts.data_ptr<int64_t>(), flat_bytes.data_ptr<int64_t>(),
      entry_buf_byte_offsets.data_ptr<int64_t>(),
      out_values.data_ptr<uint8_t>(), (int)E, (int)world_size, (int)Q,
      (int)total_items);
}
