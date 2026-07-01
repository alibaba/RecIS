#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>

#ifdef USE_ROCM
#include <hipcub/agent/single_pass_scan_operators.hpp>
#include <hipcub/block/block_scan.hpp>
#include <hipcub/hipcub.hpp>
#include <hipcub/util_device.hpp>
#include <hipcub/util_temporary_storage.hpp>
#else
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/cub.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/version.cuh>
#endif

#include <tuple>

#include "ATen/Dispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty_like.h"
#include "c10/core/DeviceType.h"
#include "c10/core/Stream.h"
#include "c10/cuda/CUDAException.h"
#include "c10/cuda/CUDAStream.h"
#include "cuda/cuda_param.cuh"
#include "cuda/utils.cuh"
#include "fused_ragged_cutoff.h"
#include "ragged_common.cuh"
#include "torch/extension.h"

namespace recis {
namespace functional {

#ifdef USE_ROCM
namespace cub = hipcub;
#endif

template <typename index_t>
__global__ void post_cutoff_lens_kernel(
    // __grid_constant__ annotation is only allowed for architecture
    // compute_70 or later
    const index_t** __restrict__ offsets,
    const index_t* __restrict__ keep_lengths,
    const index_t* __restrict__ fea_offset, index_t* __restrict__ cutoff_lens,
    index_t* __restrict__ drop_num, index_t* __restrict__ pad_num,
    int fea_num) {
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  int fea = blockIdx.y;
  // never reach
  if (fea >= fea_num) {
    return;
  }
  int row_max = fea_offset[fea + 1] - fea_offset[fea];
  if (row >= row_max) {
    return;
  }
  const index_t* offset = offsets[fea];
  __shared__ index_t keep_length;
  __shared__ index_t base_idx;
  if (threadIdx.x == 0) {
    keep_length = keep_lengths[fea];
    base_idx = fea_offset[fea];
  }
  __syncthreads();

  index_t beg = offset[row];
  index_t end = offset[row + 1];
  index_t src_len = end - beg;

  // record the reserve lens, dropped and padded num each row
  bool dropped = src_len >= keep_length;
  index_t keep = dropped ? keep_length : src_len;
  index_t glb_row = base_idx + row;
  cutoff_lens[glb_row] = keep;

  index_t drop_len = src_len - keep_length;
  drop_num[glb_row] = dropped ? drop_len : 0;
  pad_num[glb_row] = dropped ? 0 : keep_length - src_len;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> post_cutoff_lens_cuda_op(
    std::vector<at::Tensor> offsets, at::Tensor keep_lens,
    at::Tensor fea_offset, int fea_num, int total_rows, int max_row_num,
    cudaStream_t stream) {
  dim3 block((max_row_num + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK,
             fea_num),
      thread(MAX_THREADS_PER_BLOCK);

  auto cutoff_lens = at::empty({total_rows}, offsets.front().options());
  auto drop_nums = at::empty({total_rows}, offsets.front().options());
  auto pad_nums = at::empty({total_rows}, offsets.front().options());
  AT_DISPATCH_INDEX_TYPES(
      offsets.front().scalar_type(), "post_cutoff_lens_cuda_op", [&] {
        cuda::CudaVecParam<index_t*> offsets_ptrs(fea_num, stream);
        for (int i = 0; i < fea_num; ++i) {
          offsets_ptrs[i] = offsets[i].data_ptr<index_t>();
        }
        post_cutoff_lens_kernel<index_t><<<block, thread, 0, stream>>>(
            const_cast<const index_t**>(offsets_ptrs.data()),
            keep_lens.data_ptr<index_t>(), fea_offset.data_ptr<index_t>(),
            cutoff_lens.data_ptr<index_t>(), drop_nums.data_ptr<index_t>(),
            pad_nums.data_ptr<index_t>(), fea_num);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(cutoff_lens, drop_nums, pad_nums);
}

template <typename value_t, typename ScanTileStateT, int BLOCK_THREADS>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
    seg_scan_sum_kernel(const value_t* __restrict__ values,
                        const value_t* __restrict__ fea_offsets,
                        value_t* __restrict__ offsets,
                        value_t* __restrict__ inclusive_sums_per_feature,
                        //__restrict__ ScanTileStateT* tile_states,
                        ScanTileStateT* tile_states, const int fea_num,
                        int max_tiles_per_fea) {
  int fea = blockIdx.y;
  int tile_idx = blockIdx.x;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (fea >= fea_num) {
    return;
  }

  ScanTileStateT tile_state = tile_states[fea];

  int beg = fea_offsets[fea];
  int end = fea_offsets[fea + 1];
  int rows = end - beg;
  int tiles = (rows + BLOCK_THREADS - 1) / BLOCK_THREADS;

  if (tile_idx >= tiles) {
    return;
  }

  value_t val = 0;
  if (tid < rows) {
    val = values[beg + tid];
  }

  typedef cub::BlockScan<value_t, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>
      BlockScan;
  __shared__ typename BlockScan::TempStorage block_scan_temp_storage;

  value_t inclusive_prefix = 0;
  if (tile_idx == 0) {
    value_t block_aggregate;
    BlockScan(block_scan_temp_storage).ExclusiveSum(val, val, block_aggregate);
    if (tid == 0) {
      tile_state.SetInclusive(0, block_aggregate);
    }
    inclusive_prefix = block_aggregate;
  } else {
    __shared__ typename cub::TilePrefixCallbackOp<value_t, ::cuda::std::plus<>,
                                                  ScanTileStateT>::TempStorage
        cub_callback_temp_storage;

    cub::TilePrefixCallbackOp<value_t, ::cuda::std::plus<>, ScanTileStateT>
        prefix_op(tile_state, cub_callback_temp_storage, ::cuda::std::plus<>{},
                  tile_idx);
    BlockScan(block_scan_temp_storage).ExclusiveSum(val, val, prefix_op);
    inclusive_prefix = prefix_op.GetInclusivePrefix();
  }

  if (tid < rows) {
    offsets[beg + fea + tid] = val;
  }

  if (tile_idx == tiles - 1 && threadIdx.x == 0) {
    inclusive_sums_per_feature[fea] = inclusive_prefix;
    offsets[beg + fea + rows] = inclusive_prefix;
  }
}

template <typename ScanTileStateT>
__global__ void SegScanInitKernel(ScanTileStateT* tile_states, int num_tiles) {
  int fea = blockIdx.y;
  ScanTileStateT tile_state =
      tile_states[fea];  // global mem -> local mem / reg
  tile_state.InitializeStatus(num_tiles);
}

// prepare on the host
template <typename ScanTileStateT>
cudaError_t ragged_tile_state_init(
    cuda::CudaVecParam<ScanTileStateT>& tile_states, void*& tile_state_mem,
    int max_tiles, int fea_num) {
  static const int ALIGN_BYTES = 256;
  static const int ALIGN_MASK = ~(ALIGN_BYTES - 1);
  cudaError_t err = cudaSuccess;
  size_t alloc_sz[1] = {};
  ScanTileStateT::AllocationSize(max_tiles, alloc_sz[0]);
  size_t align_sz = 0;
  void* allocations[1] = {(void*)0x1};
  // AliasTemporaries moved from cub:: to cub::detail:: in CUB 3.0.0
#if defined(NV_PLATFORM) || CUB_VERSION >= 300000
  err = cub::detail::AliasTemporaries((void*)NULL, align_sz, allocations,
                                      alloc_sz);
#else
  err = cub::AliasTemporaries((void*)NULL, align_sz, allocations, alloc_sz);
#endif
  TORCH_CHECK(cudaSuccess == err, cudaGetErrorString(err));

  size_t align_sz_pad = (align_sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  size_t total_align_sz = align_sz_pad * fea_num;
  size_t total_align_sz_pad = align_sz_pad * fea_num + ALIGN_BYTES - 1;

  tile_state_mem = cuda::cuda_malloc<void>(total_align_sz_pad);  // global mem
  allocations[0] = (void*)((size_t(tile_state_mem) + ALIGN_BYTES - 1) &
                           ALIGN_MASK);  // mock AliasTemporaries

  for (int i = 0; i < fea_num; ++i) {
    void* tile_state_ptr = (char*)allocations[0] + align_sz_pad * i;
    err = tile_states[i].Init(max_tiles, tile_state_ptr, align_sz_pad);
    TORCH_CHECK(cudaSuccess == err, cudaGetErrorString(err));
  }
  return err;
}

// TODO: support cub RaggedTileState, add as .cuh in recis
std::tuple<at::Tensor, at::Tensor> seg_scan_cuda(at::Tensor fea_offset,
                                                 int fea_num, int total_rows,
                                                 at::Tensor values,
                                                 int max_row_num,
                                                 cudaStream_t stream) {
  at::Tensor cutoff_offsets =
      at::empty({total_rows + fea_num}, values.options());
  at::Tensor max_offset_seg = torch::empty({fea_num}, values.options());
  int max_tiles =
      (max_row_num + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

  AT_DISPATCH_INDEX_TYPES(values.scalar_type(), "segment_scan_cuda_op", [&] {
  // using index_t = int32_t;
#ifdef NV_PLATFORM
    typedef typename cub::ScanTileState<index_t> ScanTileStateT;
#else
    typedef typename cub::ScanTileState<index_t, false> ScanTileStateT;
#endif
    cuda::CudaVecParam<ScanTileStateT> tile_states(fea_num, stream);
    void* tile_state_mem = nullptr;
    cudaError_t err =
        ragged_tile_state_init(tile_states, tile_state_mem, max_tiles, fea_num);
    TORCH_CHECK(cudaSuccess == err, cudaGetErrorString(err));

    auto init_blocks =
        dim3((max_tiles + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK,
             fea_num);
    auto init_threads = dim3(MAX_THREADS_PER_BLOCK);

    SegScanInitKernel<ScanTileStateT><<<init_blocks, init_threads, 0, stream>>>(
        tile_states.data(), max_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto scan_blocks = dim3(max_tiles, fea_num);
    auto scan_threads = dim3(MAX_THREADS_PER_BLOCK);
    seg_scan_sum_kernel<index_t, ScanTileStateT, MAX_THREADS_PER_BLOCK>
        <<<scan_blocks, scan_threads, 0, stream>>>(
            values.data_ptr<index_t>(), fea_offset.data_ptr<index_t>(),
            cutoff_offsets.data_ptr<index_t>(),
            max_offset_seg.data_ptr<index_t>(), tile_states.data(), fea_num,
            max_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    cuda::delete_cuda_ptr(tile_state_mem);
  });

  return std::make_tuple(cutoff_offsets, max_offset_seg);
}

template <typename value_t, typename ScanTileStateT, int BLOCK_THREADS>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
    seg_gen_offsets_kernel(const value_t* __restrict__ fea_seq_offsets,
                           const value_t** __restrict__ outer_offsets,
                           const value_t** __restrict__ inner_offsets,
                           const value_t* __restrict__ output_inner_fea_offset,
                           const value_t* __restrict__ keep_lens,
                           value_t* __restrict__ output_outer_offsets,
                           value_t* __restrict__ output_inner_offsets,
                           value_t* __restrict__ inclusive_sums,
                           ScanTileStateT* tile_states,
                           const value_t* __restrict__ drop_nums,
                           const bool* __restrict__ drop_sides,
                           const value_t* __restrict__ pad_nums,
                           const bool* __restrict__ pad_sides,
                           const int fea_num, int max_tiles) {
  int fea = blockIdx.y;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int tile_idx = blockIdx.x;
  if (fea >= fea_num) {
    return;
  }
  ScanTileStateT tile_state = tile_states[fea];  // global mem -> local or reg

  value_t fea_seq_beg = fea_seq_offsets[fea];
  value_t fea_seq_end = fea_seq_offsets[fea + 1];
  value_t seq_max = fea_seq_end - fea_seq_beg;
  value_t keep_len = keep_lens[fea];
  value_t cutoff_max_rows = keep_len * seq_max;
  const value_t* outer_offset = outer_offsets[fea];
  const value_t* inner_offset = inner_offsets[fea];
  bool drop_side = drop_sides[fea];
  bool pad_side = pad_sides[fea];

  value_t* output_outer_offset = output_outer_offsets + fea_seq_beg + fea;
  value_t* output_inner_offset =
      output_inner_offsets + output_inner_fea_offset[fea] + fea;

  int seq_id = tid / keep_len;
  int seq_row_id = tid % keep_len;

  int outer_stride = blockDim.x * gridDim.x;
#pragma unroll
  for (int oid = tid; oid < seq_max + 1; oid += outer_stride) {
    output_outer_offset[oid] = oid * keep_len;
  }

  int tiles = (cutoff_max_rows + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (tile_idx >= tiles) {
    return;
  }
  // load val from global mem to reg
  value_t val = 0;
  if (tid < cutoff_max_rows) {
    value_t drop = drop_nums[seq_id + fea_seq_beg];
    value_t pad = pad_nums[seq_id + fea_seq_beg];
    value_t row_beg = outer_offset[seq_id];
    value_t row_len = 0;
    value_t row_id = seq_row_id;
    bool paded = false;
    if (drop > 0) {
      if (drop_side) {
        row_id += drop;
      }
    } else if (pad > 0) {
      if (pad_side) {
        paded = seq_row_id < pad;
        row_id -= pad;
      } else {
        paded = keep_len <= (pad + seq_row_id);
      }
    }
    if (paded) {
      val = 0;
    } else {
      value_t fea_gb_row_id = row_beg + row_id;
      val = inner_offset[fea_gb_row_id + 1] - inner_offset[fea_gb_row_id];
    }
  }

  // scan on reg
  typedef cub::BlockScan<value_t, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>
      BlockScan;
  __shared__ typename BlockScan::TempStorage block_scan_temp_storage;

  value_t inclusive_prefix = 0;
  if (tile_idx == 0) {
    value_t block_aggregate;
    BlockScan(block_scan_temp_storage).ExclusiveSum(val, val, block_aggregate);
    if (tid == 0) {
      tile_state.SetInclusive(0, block_aggregate);
    }
    inclusive_prefix = block_aggregate;
  } else {
    __shared__ typename cub::TilePrefixCallbackOp<value_t, ::cuda::std::plus<>,
                                                  ScanTileStateT>::TempStorage
        cub_callback_temp_storage;
    cub::TilePrefixCallbackOp<value_t, ::cuda::std::plus<>, ScanTileStateT>
        prefix_op(tile_state, cub_callback_temp_storage, ::cuda::std::plus<>{},
                  tile_idx);
    BlockScan(block_scan_temp_storage).ExclusiveSum(val, val, prefix_op);
    inclusive_prefix = prefix_op.GetInclusivePrefix();
  }

  // write back to global mem
  if (tid < cutoff_max_rows) {
    output_inner_offset[tid] = val;
  }

  if (tile_idx == tiles - 1 && threadIdx.x == 0) {
    output_inner_offset[cutoff_max_rows] = inclusive_prefix;
    inclusive_sums[fea] = inclusive_prefix;
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> seg_gen_offsets_cuda(
    at::Tensor fea_seq_offset, std::vector<at::Tensor> outer_offsets,
    std::vector<at::Tensor> inner_offsets, at::Tensor output_inner_fea_offset,
    at::Tensor drop_nums, at::Tensor drop_sides, at::Tensor pad_nums,
    at::Tensor pad_sides, at::Tensor keep_lengths, int fea_num, int total_seqs,
    int total_cutoff_rows, int max_cutoff_rows, cudaStream_t stream) {
  at::Tensor output_outer_offsets =
      at::empty({total_seqs + fea_num},
                outer_offsets.front().options().device(torch::kCUDA));
  at::Tensor output_inner_offsets =
      at::empty({total_cutoff_rows + fea_num},
                inner_offsets.front().options().device(torch::kCUDA));
  at::Tensor max_offset_seg = torch::empty(
      {fea_num}, output_outer_offsets.options().device(torch::kCUDA));

  AT_DISPATCH_INDEX_TYPES(
      inner_offsets.front().scalar_type(), "seg_gen_offsets_cuda_op", [&] {
        cuda::CudaVecParam<index_t*> outer_offsets_ptrs(fea_num, stream);
        cuda::CudaVecParam<index_t*> inner_offsets_ptrs(fea_num, stream);
        for (int i = 0; i < fea_num; ++i) {
          outer_offsets_ptrs[i] = outer_offsets[i].data_ptr<index_t>();
          inner_offsets_ptrs[i] = inner_offsets[i].data_ptr<index_t>();
        }
        int max_tiles = (max_cutoff_rows + MAX_THREADS_PER_BLOCK - 1) /
                        MAX_THREADS_PER_BLOCK;

    // scan init host
#ifdef NV_PLATFORM
        typedef typename cub::ScanTileState<index_t> ScanTileStateT;
#else
        typedef typename cub::ScanTileState<index_t, false> ScanTileStateT;
#endif
        cuda::CudaVecParam<ScanTileStateT> tile_states(fea_num, stream);
        void* tile_state_mem = nullptr;
        cudaError_t err = ragged_tile_state_init(tile_states, tile_state_mem,
                                                 max_tiles, fea_num);
        TORCH_CHECK(cudaSuccess == err, cudaGetErrorString(err));
        // CubDebugExit(error);

        // scan init kernel
        auto init_blocks = dim3(
            (max_tiles + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK,
            fea_num);
        auto init_threads = dim3(MAX_THREADS_PER_BLOCK);

        SegScanInitKernel<ScanTileStateT>
            <<<init_blocks, init_threads, 0, stream>>>(tile_states.data(),
                                                       max_tiles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // scan invoke
        auto scan_blocks = dim3(max_tiles, fea_num);
        auto scan_threads = dim3(MAX_THREADS_PER_BLOCK);

        seg_gen_offsets_kernel<index_t, ScanTileStateT, MAX_THREADS_PER_BLOCK>
            <<<scan_blocks, scan_threads, 0, stream>>>(
                fea_seq_offset.data_ptr<index_t>(),
                const_cast<const index_t**>(outer_offsets_ptrs.data()),
                const_cast<const index_t**>(inner_offsets_ptrs.data()),
                output_inner_fea_offset.data_ptr<index_t>(),
                keep_lengths.data_ptr<index_t>(),
                output_outer_offsets.data_ptr<index_t>(),
                output_inner_offsets.data_ptr<index_t>(),
                max_offset_seg.data_ptr<index_t>(),
                (ScanTileStateT*)tile_states.data(),
                drop_nums.data_ptr<index_t>(), drop_sides.data_ptr<bool>(),
                pad_nums.data_ptr<index_t>(), pad_sides.data_ptr<bool>(),
                fea_num, max_tiles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        cuda::delete_cuda_ptr(tile_state_mem);
      });
  return std::make_tuple(output_outer_offsets, output_inner_offsets,
                         max_offset_seg);
}

template <typename index_t, typename value_t, bool use_shared_mem>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
    fused_ragged_cutoff_3D_kernel(
        const index_t** __restrict__ offsets,        // input outer offset
        const index_t** __restrict__ inner_offsets,  // input inner offset
        const value_t** __restrict__ values,
        const index_t* __restrict__ output_inner_offsets,
        value_t* __restrict__ cutoff_values,
        const index_t* __restrict__ fea_offset,
        const index_t* __restrict__ output_inner_fea_offset,
        const index_t* __restrict__ output_val_fea_offset,
        const index_t* __restrict__ drop_num,
        const bool* __restrict__ drop_sides,  // true: drop left; false: drop
                                              // right
        const index_t* __restrict__ pad_num,
        const bool* __restrict__ pad_sides,  // true: pad left; false: pad right
        const index_t* __restrict__ keep_lens, const int fea_num) {
  int tid = threadIdx.x;
  int data_id = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  int fea = blockIdx.y;
  if (fea >= fea_num) {
    return;
  }

  __shared__ index_t fea_beg;
  __shared__ index_t fea_end;
  __shared__ index_t output_fea_beg;
  __shared__ bool drop_side;
  __shared__ bool pad_side;
  if (tid == 0) {
    fea_beg = fea_offset[fea];
    fea_end = fea_offset[fea + 1];
    output_fea_beg = output_val_fea_offset[fea];
    drop_side = drop_sides[fea];
    pad_side = pad_sides[fea];
  }
  __syncthreads();
  const index_t* offset = offsets[fea];
  const index_t* inner_offset = inner_offsets[fea];
  index_t seq_max = fea_end - fea_beg;
  index_t row_max = offset[seq_max] - offset[0];
  // load global to shared (seq_max + 1)
  extern __shared__ __align__(sizeof(index_t)) unsigned char tmp_smem[];
  if constexpr (use_shared_mem) {
    index_t* smem_offsets = reinterpret_cast<index_t*>(tmp_smem);
    for (index_t i = tid; i < seq_max + 1; i += blockDim.x) {
      smem_offsets[i] = offset[i];
    }
    index_t* smem_inner_offsets =
        reinterpret_cast<index_t*>(tmp_smem + (seq_max + 1) * sizeof(index_t));
    for (index_t i = tid; i < row_max + 1; i += blockDim.x) {
      smem_inner_offsets[i] = inner_offset[i];
    }
    __syncthreads();
    offset = smem_offsets;
    inner_offset = smem_inner_offsets;
  }

  index_t data_max = inner_offset[row_max] - inner_offset[0];
  // early return for empty value
  if (data_max == 0) {
    return;
  }
  __shared__ index_t keep_len;
  __shared__ index_t cutoff_val_num;
  const value_t* value = nullptr;
  const index_t* cutoff_offset = nullptr;
  if (data_id < data_max && tid == 0) {
    keep_len = keep_lens[fea];
  }
  __syncthreads();

  if (data_id < data_max) {
    value = values[fea];
    // fea for fix due to cutoff_offset is inclusive sum + exclusive sum
    const index_t* output_inner_offset =
        output_inner_offsets + output_inner_fea_offset[fea] + fea;
    value_t* cutoff_value = cutoff_values + output_fea_beg;
#pragma unroll
    for (index_t did = data_id; did < data_max; did += stride) {
      index_t row = binary_search(did, inner_offset, row_max + 1);
      index_t seq = binary_search(row, offset, seq_max + 1);
      index_t drop = drop_num[seq + fea_beg];
      index_t pad = pad_num[seq + fea_beg];
      index_t row_beg = offset[seq];
      index_t row_end = offset[seq + 1];
      index_t cutoff_row_beg = keep_len * seq;
      index_t cutoff_row_end = keep_len * (seq + 1);
      index_t row_id = row - row_beg;
      index_t cutoff_row_id = row_id;
      if (drop_side) {  // left drop
        cutoff_row_id -= drop;
      }
      if (pad_side) {  // left pad
        cutoff_row_id += pad;
      }
      if (cutoff_row_id >= 0 && cutoff_row_id < keep_len) {
        index_t output_data_offset_beg =
            output_inner_offset[cutoff_row_beg + cutoff_row_id];
        index_t row_data_id = did - inner_offset[row];
        index_t output_data_offset =
            output_data_offset_beg + did - inner_offset[row];
        cutoff_value[output_data_offset] = value[did];
      }
    }
  }
}

template <typename T>
static int get_median(std::vector<T>& data) {
  if (data.empty()) {
    throw std::invalid_argument("Cannot calculate median of an empty vector.");
  }
  size_t n = data.size();
  size_t target_idx;
  target_idx = (n - 1) / 2;
  std::nth_element(data.begin(), data.begin() + target_idx, data.end());
  return std::max(data[target_idx], static_cast<T>(1));
}

void fused_ragged_cutoff_3D_cuda_op(
    std::vector<at::Tensor> values, std::vector<at::Tensor> offsets,
    std::vector<at::Tensor> inner_offsets, at::Tensor cutoff_values,
    at::Tensor output_inner_offsets, at::Tensor fea_offset,
    at::Tensor output_inner_fea_offset, at::Tensor output_val_fea_offset,
    int32_t fea_num, at::Tensor drop_nums, at::Tensor drop_sides,
    at::Tensor pad_nums, at::Tensor pad_sides, at::Tensor keep_lens,
    cudaStream_t stream) {
  AT_DISPATCH_INDEX_TYPES(
      offsets.front().scalar_type(), "fused_ragged_cutoff_3D_cuda_op_0", [&] {
        AT_DISPATCH_ALL_TYPES(
            values.front().scalar_type(), "fused_ragged_cutoff_3D_cuda_op_1",
            [&]() {
              cuda::CudaVecParam<index_t*> offsets_ptrs(fea_num, stream);
              cuda::CudaVecParam<index_t*> inner_offsets_ptrs(fea_num, stream);
              cuda::CudaVecParam<scalar_t*> values_ptrs(fea_num, stream);

              std::vector<index_t> val_nums;
              index_t max_offsets_ele = 0;
              index_t max_inner_offsets_ele = 0;
              for (int i = 0; i < fea_num; ++i) {
                offsets_ptrs[i] = offsets[i].data_ptr<index_t>();
                inner_offsets_ptrs[i] = inner_offsets[i].data_ptr<index_t>();
                max_offsets_ele = std::max(
                    static_cast<index_t>(offsets[i].numel()), max_offsets_ele);
                max_inner_offsets_ele =
                    std::max(static_cast<index_t>(inner_offsets[i].numel()),
                             max_inner_offsets_ele);
                // max_inner_offsets_ele: max_rows
                values_ptrs[i] = values[i].data_ptr<scalar_t>();
                val_nums.emplace_back(static_cast<index_t>(values[i].numel()));
              }

              size_t shared_mem_size =
                  sizeof(index_t) * (max_offsets_ele + max_inner_offsets_ele);
              const auto threads = dim3(MAX_THREADS_PER_BLOCK);
              int grid_x = (get_median(val_nums) + MAX_THREADS_PER_BLOCK - 1) /
                           MAX_THREADS_PER_BLOCK;
              int grid_y = fea_num;
              const auto blocks = dim3(grid_x, grid_y);

              LAUNCH_KERNEL_SHMEM_DISPATCH(
                  fused_ragged_cutoff_3D_kernel, (index_t, scalar_t), blocks,
                  threads, shared_mem_size, stream,
                  const_cast<const index_t**>(offsets_ptrs.data()),
                  const_cast<const index_t**>(inner_offsets_ptrs.data()),
                  const_cast<const scalar_t**>(values_ptrs.data()),
                  const_cast<const index_t*>(
                      output_inner_offsets.data_ptr<index_t>()),
                  cutoff_values.data_ptr<scalar_t>(),
                  fea_offset.data_ptr<index_t>(),
                  output_inner_fea_offset.data_ptr<index_t>(),
                  output_val_fea_offset.data_ptr<index_t>(),
                  drop_nums.data_ptr<index_t>(), drop_sides.data_ptr<bool>(),
                  pad_nums.data_ptr<index_t>(), pad_sides.data_ptr<bool>(),
                  keep_lens.data_ptr<index_t>(), fea_num);
            });
      });
}

template <typename index_t, typename value_t, bool use_shared_mem>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
    fused_ragged_cutoff_2D_kernel(
        const index_t** __restrict__ offsets,
        const value_t** __restrict__ values,
        const index_t* __restrict__ cutoff_offsets,
        value_t* __restrict__ cutoff_values,
        const index_t* __restrict__ fea_offset,
        const index_t* __restrict__ output_val_fea_offset,
        const index_t* __restrict__ drop_num,
        const index_t* __restrict__ pad_num,
        const index_t* __restrict__ keep_lens,
        const index_t* __restrict__ cutoff_val_nums, const int fea_num,
        const bool* __restrict__ sides) {  // true: drop left; false: drop right
  int tid = threadIdx.x;
  int data_id = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  // use blockIdx.y or binary search to locate the feature number
  int fea = blockIdx.y;
  // never reach
  if (fea >= fea_num) {
    return;
  }
  __shared__ index_t fea_beg;
  __shared__ index_t fea_end;
  __shared__ index_t output_fea_beg;
  __shared__ bool side;
  if (tid == 0) {
    fea_beg = fea_offset[fea];
    fea_end = fea_offset[fea + 1];
    output_fea_beg = output_val_fea_offset[fea];
    side = sides[fea];
  }
  __syncthreads();
  index_t row_max = fea_end - fea_beg;
  // index_t *offset = offsets.vals[fea];
  const index_t* offset = offsets[fea];
  // load global to shared (row_max + 1)
  extern __shared__ __align__(sizeof(index_t)) unsigned char tmp_smem[];
  if constexpr (use_shared_mem) {
    index_t* smem_offsets = reinterpret_cast<index_t*>(tmp_smem);
    for (index_t i = tid; i < row_max + 1; i += blockDim.x) {
      smem_offsets[i] = offset[i];
    }
    __syncthreads();
    offset = smem_offsets;
  }
  index_t data_max = offset[row_max] - offset[0];
  // early return for empty value
  if (data_max == 0) {
    return;
  }
  index_t row = 0;
  index_t src_beg = 0;
  index_t drop = 0;
  index_t pad = 0;
  __shared__ index_t keep_len;
  __shared__ index_t cutoff_val_num;
  const value_t* value = nullptr;
  const index_t* cutoff_offset = nullptr;
  value_t* cutoff_value = nullptr;
  if (data_id < data_max && tid == 0) {
    keep_len = keep_lens[fea];
    cutoff_val_num = cutoff_val_nums[fea];
  }
  __syncthreads();

  if (data_id < data_max) {
    // value = values.vals[fea];
    value = values[fea];
    // fea for fix due to cutoff_offset is inclusive sum + exclusive sum
    cutoff_offset = cutoff_offsets + fea + fea_beg;
    cutoff_value = cutoff_values + output_fea_beg;
#pragma unroll
    for (index_t did = data_id; did < data_max; did += stride) {
      row = binary_search(did, offset, row_max + 1);
      drop = drop_num[row + fea_beg];
      pad = pad_num[row + fea_beg];
      src_beg = offset[row];
      if (side) {
        index_t dst_col = did - src_beg - drop;
        if (dst_col >= 0 && dst_col < keep_len) {  // dst_col < keep_len
          // The did element of the row before the cutoff corresponds to the
          // dst_index th element of the global after the cutoff.
          index_t dst_idx = cutoff_offset[row] + dst_col;  // read global mem
          cutoff_value[dst_idx] =
              value[did];  // read write global mem (both adjacent)
        }
      } else {
        index_t row_len =
            offset[row + 1] -
            offset[row];  // row len. (offset is already loaded into shared mem)
        index_t dst_col = did - src_beg;
        if (dst_col + drop < row_len) {
          index_t dst_idx = cutoff_offset[row] + dst_col;  // read global mem
          cutoff_value[dst_idx] =
              value[did];  // read write global mem (both adjacent)
        }
      }
    }
  }
}

void fused_ragged_cutoff_2D_cuda_op(
    std::vector<at::Tensor> values, std::vector<at::Tensor> offsets,
    at::Tensor cutoff_values, at::Tensor cutoff_offsets, at::Tensor drop_nums,
    at::Tensor pad_nums, at::Tensor keep_lens, at::Tensor fea_offset,
    at::Tensor output_val_fea_offset, int32_t max_row_num, int32_t fea_num,
    at::Tensor cutoff_val_nums, at::Tensor sides, cudaStream_t stream) {
  AT_DISPATCH_INDEX_TYPES(
      offsets.front().scalar_type(), "fused_ragged_cutoff_2D_cuda_op_0", [&] {
        AT_DISPATCH_ALL_TYPES(
            values.front().scalar_type(), "fused_ragged_cutoff_2D_cuda_op_1",
            [&] {
              cuda::CudaVecParam<index_t*> offsets_ptrs(fea_num, stream);
              cuda::CudaVecParam<scalar_t*> values_ptrs(fea_num, stream);

              std::vector<int> val_nums;
              int max_offsets_ele = 0;
              for (int i = 0; i < fea_num; ++i) {
                offsets_ptrs[i] = offsets[i].data_ptr<index_t>();
                max_offsets_ele = std::max(static_cast<int>(offsets[i].numel()),
                                           max_offsets_ele);
                values_ptrs[i] = values[i].data_ptr<scalar_t>();
                val_nums.emplace_back(static_cast<int>(values[i].numel()));
              }

              size_t shared_mem_size = sizeof(index_t) * max_offsets_ele;
              const auto threads = dim3(MAX_THREADS_PER_BLOCK);
              const auto blocks =
                  dim3((get_median(val_nums) + MAX_THREADS_PER_BLOCK - 1) /
                           MAX_THREADS_PER_BLOCK,
                       fea_num);
              LAUNCH_KERNEL_SHMEM_DISPATCH(
                  fused_ragged_cutoff_2D_kernel, (index_t, scalar_t), blocks,
                  threads, shared_mem_size, stream,
                  const_cast<const index_t**>(offsets_ptrs.data()),
                  const_cast<const scalar_t**>(values_ptrs.data()),
                  cutoff_offsets.data_ptr<index_t>(),
                  cutoff_values.data_ptr<scalar_t>(),
                  fea_offset.data_ptr<index_t>(),
                  output_val_fea_offset.data_ptr<index_t>(),
                  drop_nums.data_ptr<index_t>(), pad_nums.data_ptr<index_t>(),
                  keep_lens.data_ptr<index_t>(),
                  cutoff_val_nums.data_ptr<index_t>(), fea_num,
                  sides.data_ptr<bool>());
            });
      });
}

}  // namespace functional
}  // namespace recis
