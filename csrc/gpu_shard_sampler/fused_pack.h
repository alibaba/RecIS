#pragma once
#include <torch/extension.h>

#include <tuple>

// Fused route + default replacement for pack_feature and valid_sample_ids
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_route_ids(
    torch::Tensor ids, torch::Tensor sorted_ptrs, torch::Tensor sorted_sizes,
    int64_t default_value, int64_t world_size);

// Unified batched gather: single kernel for all entries (dense + ragged +
// weights)
std::tuple<torch::Tensor, std::vector<torch::Tensor>> unified_batched_gather(
    torch::Tensor flat_offsets, torch::Tensor offset_indices,
    torch::Tensor value_ptrs, torch::Tensor flat_bytes, torch::Tensor indices);

// Fused serialize: all ranks' gather results → single flat byte buffer
void fused_serialize(torch::Tensor all_lengths, torch::Tensor value_ptrs,
                     torch::Tensor source_offsets, torch::Tensor val_starts,
                     torch::Tensor flat_bytes, torch::Tensor send_offsets,
                     torch::Tensor entry_val_byte_offsets,
                     torch::Tensor send_flat, int64_t E, int64_t world_size,
                     int64_t total_recv);

// Fused deserialize Phase 1: parse all_lengths, scatter to query order
void fused_deserialize_fixed(
    torch::Tensor recv_flat, torch::Tensor recv_offsets,
    torch::Tensor recv_counts, torch::Tensor scatter_indices,
    torch::Tensor scatter_offsets, torch::Tensor out_lengths,
    torch::Tensor recv_lengths, int64_t E, int64_t world_size, int64_t Q,
    int64_t total_items);

// Fused deserialize Phase 2: parse values, direct scatter to final positions
void fused_deserialize_values(
    torch::Tensor recv_flat, torch::Tensor recv_offsets,
    torch::Tensor recv_counts, torch::Tensor scatter_indices,
    torch::Tensor scatter_offsets, torch::Tensor recv_lengths,
    torch::Tensor src_offsets, torch::Tensor all_offsets,
    torch::Tensor feat_val_starts, torch::Tensor flat_bytes,
    torch::Tensor entry_buf_byte_offsets, torch::Tensor out_values, int64_t E,
    int64_t world_size, int64_t Q, int64_t total_items);
