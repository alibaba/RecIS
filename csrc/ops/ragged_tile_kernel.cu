#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#include "c10/core/DeviceType.h"
#include "c10/util/Optional.h"
#include "cuda/atomic_fast.cuh"
#include "ragged_tile.h"
namespace recis {
namespace functional {
using uint = unsigned int;

template <typename T>
__device__ __inline__ void copy_data(const T* src, T* dst) = delete;

template <>
__device__ __inline__ void copy_data(const float* src, float* dst) {
  ((float4*)dst)[0] = ((float4*)src)[0];
}

template <typename VT, typename TT>
__device__ void copy_table(const VT* __restrict__ value,
                           const TT* __restrict__ table, TT* __restrict__ dst,
                           const uint value_len, const uint seq, const uint dim,
                           const bool left_pad) {
  const uint t_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint threads = gridDim.x * blockDim.x;
  const uint pad_num =
      ((left_pad) & (seq >= value_len)) ? (seq - value_len) : 0;
  for (uint idx = t_id; idx < seq * dim; idx += threads) {
    uint val_id = idx / dim;
    uint dim_id = idx % dim;
    uint dst_pos = val_id * dim + dim_id;
    if ((val_id < value_len + pad_num) && (val_id >= pad_num)) {
      uint value_data = static_cast<uint>(value[val_id - pad_num]);
      dst[dst_pos] = table[value_data * dim + dim_id];
    } else {
      dst[dst_pos] = TT(0);
    }
  }
}  // copy_table

template <typename VT, typename TT>
__device__ void copy_dy(const VT* __restrict__ value, const TT* __restrict__ dy,
                        TT* __restrict__ d_table, const uint value_len,
                        const uint seq, const uint dim, const bool left_pad) {
  const uint t_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint threads = gridDim.x * blockDim.x;
  const uint pad_num =
      ((left_pad) & (seq >= value_len)) ? (seq - value_len) : 0;
  for (uint idx = t_id; idx < seq * dim; idx += threads) {
    uint val_id = idx / dim;
    if ((val_id < value_len + pad_num) && (val_id >= pad_num)) {
      uint dim_id = idx % dim;
      uint dy_pos = val_id * dim + dim_id;
      uint value_data = static_cast<uint>(value[val_id - pad_num]);
      uint table_pos = value_data * dim + dim_id;
      atomic_add_custom<TT>(d_table + table_pos, dy[dy_pos]);
    }
  }
}  // copy_dy

template <typename BT, typename VT, typename OT>
__device__ bool load_offset(const BT* batch_seq, BT* batch_seq_sm,
                            OT* offset_sm, const OT* offset, const int64_t dim,
                            OT* value_start, uint* value_len,
                            int64_t* out_start, uint* seq) {
  const uint tensor_id = blockIdx.z;
  const uint off_id = blockIdx.y;
  const uint t_id = threadIdx.x;
  if (t_id < 6) {
    batch_seq_sm[t_id] = batch_seq[tensor_id * 3 + t_id];
  }
  __syncthreads();
  BT off_start = batch_seq_sm[0];
  BT off_end = batch_seq_sm[3];
  BT off_num = off_end - off_start;
  if (off_id >= off_num) return false;

  BT seq_len = batch_seq_sm[5];
  const OT* off_start_ptr = offset + off_start;
  *out_start = batch_seq_sm[1] * dim + off_id * seq_len * dim;
  *seq = static_cast<uint>(seq_len);
  if (t_id < 2) {
    offset_sm[t_id] = off_start_ptr[off_id + t_id];
  }
  __syncthreads();
  *value_start = offset_sm[0];
  *value_len = static_cast<uint>(offset_sm[1] - *value_start);
  return true;
}

template <typename BT, typename VT, typename OT, typename TT>
static __global__ void ragged_tile_kernel(
    const BT* __restrict__ batch_seq, const VT* __restrict__ value,
    const OT* __restrict__ offset, const TT* __restrict__ table,
    TT* __restrict__ output, const int64_t dim, const int64_t batch_num,
    const bool* __restrict__ left_pad) {
  __shared__ BT batch_seq_sm[6];
  __shared__ OT offset_sm[2];
  OT value_start;
  uint value_len, seq;
  int64_t out_start;
  bool valid =
      load_offset<BT, VT, OT>(batch_seq, batch_seq_sm, offset_sm, offset, dim,
                              &value_start, &value_len, &out_start, &seq);
  if (!valid) return;
  const VT* value_start_ptr = value + value_start;
  TT* out_start_ptr = output + out_start;
  const uint dim_uint = static_cast<uint>(dim);
  copy_table<VT, TT>(value_start_ptr, table, out_start_ptr, value_len, seq,
                     dim_uint, left_pad[blockIdx.z]);
}  // ragged_tile_kernel

template <typename BT, typename VT, typename OT, typename TT>
static __global__ void ragged_tile_back_kernel(
    const BT* __restrict__ batch_seq, const VT* __restrict__ value,
    const OT* __restrict__ offset, const TT* __restrict__ dy,
    TT* __restrict__ d_table, const int64_t dim, const int64_t batch_num,
    const bool* __restrict__ left_pad) {
  __shared__ BT batch_seq_sm[6];
  __shared__ OT offset_sm[2];
  OT value_start;
  uint value_len, seq;
  int64_t out_start;
  bool valid =
      load_offset<BT, VT, OT>(batch_seq, batch_seq_sm, offset_sm, offset, dim,
                              &value_start, &value_len, &out_start, &seq);
  if (!valid) return;
  const VT* value_ptr = value + value_start;
  const TT* dy_ptr = dy + out_start;
  const uint dim_uint = static_cast<uint>(dim);
  copy_dy<VT, TT>(value_ptr, dy_ptr, d_table, value_len, seq, dim_uint,
                  left_pad[blockIdx.z]);
}  // ragged_tile_backward_kernel

template <typename BT>
void ragged_tile_gpu_impl(const std::vector<BT>& batch_seq,
                          torch::Tensor batch_dev, const int64_t batch_num,
                          const int64_t batch_max, const int64_t seq_min,
                          torch::Tensor value, torch::Tensor offset,
                          torch::Tensor table, torch::Tensor out,
                          torch::Tensor left_pad) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t dim = table.sizes()[1];
  auto copy_size = batch_seq.size() * sizeof(BT);
  cudaMemcpyAsync(batch_dev.data_ptr<BT>(), batch_seq.data(), copy_size,
                  cudaMemcpyHostToDevice, stream);
  const int64_t threads = 128;
  const int64_t threads_data = threads * 4;
  int64_t blocks = (seq_min * dim + threads_data - 1) / (threads_data);
  dim3 grid(blocks, batch_max, batch_num);
  dim3 block(threads);

  AT_DISPATCH_INDEX_TYPES(
      value.scalar_type(), "ragged_tile_VT", ([&] {
        using VT = index_t;
        AT_DISPATCH_INDEX_TYPES(
            offset.scalar_type(), "ragged_tile_OT", ([&] {
              using OT = index_t;
              AT_DISPATCH_FLOATING_TYPES_AND2(
                  at::ScalarType::Half,
                  at::ScalarType::BFloat16,  // fp64,fp32,fp16,bf16
                  table.scalar_type(), "ragged_tile", ([&] {
                    ragged_tile_kernel<BT, VT, OT, scalar_t>
                        <<<grid, block, 0, stream>>>(
                            batch_dev.data_ptr<BT>(), value.data_ptr<VT>(),
                            offset.data_ptr<OT>(), table.data_ptr<scalar_t>(),
                            out.data_ptr<scalar_t>(), dim, batch_num,
                            left_pad.data_ptr<bool>());
                  }));
            }));
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}  // ragged_tile_gpu_impl

template <typename BT>
void ragged_tile_back_impl(torch::Tensor batch_dev, const int64_t batch_num,
                           const int64_t batch_max, const int64_t seq_min,
                           torch::Tensor value, torch::Tensor offset,
                           torch::Tensor d_table, torch::Tensor dy,
                           torch::Tensor left_pad) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t dim = dy.sizes()[1];
  const int64_t threads = 128;
  const int64_t block_data_num = threads * 4;
  int64_t blocks = (seq_min * dim + block_data_num - 1) / (block_data_num);
  dim3 grid(blocks, batch_max, batch_num);
  dim3 block(threads);
  AT_DISPATCH_INDEX_TYPES(
      value.scalar_type(), "ragged_tile_backward_VT", ([&] {
        using VT = index_t;
        AT_DISPATCH_INDEX_TYPES(
            offset.scalar_type(), "ragged_tile_backward_OT", ([&] {
              using OT = index_t;
#if defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800  // atomic fp16>=SM70, bf16>=SM80
              AT_DISPATCH_FLOATING_TYPES_AND2(
                  at::ScalarType::Half,
                  at::ScalarType::BFloat16,  // fp64,fp32,fp16,bf16
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
              AT_DISPATCH_FLOATING_TYPES_AND(
                  at::ScalarType::Half,  // fp64,fp32,fp16
#else
              AT_DISPATCH_FLOATING_TYPES(  // fp64,fp32
#endif
                  dy.scalar_type(), "ragged_tile_backward", ([&] {
                    using TT = scalar_t;
                    ragged_tile_back_kernel<BT, VT, OT, TT>
                        <<<grid, block, 0, stream>>>(
                            batch_dev.data_ptr<BT>(), value.data_ptr<VT>(),
                            offset.data_ptr<OT>(), dy.data_ptr<TT>(),
                            d_table.data_ptr<TT>(), dim, batch_num,
                            left_pad.data_ptr<bool>());
                  }));
            }));
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}  // ragged_tile_gpu_impl

template <typename T>
static std::vector<T> cumsum(const std::vector<int64_t>& batch,
                             const std::vector<int64_t>& seq,
                             int64_t& batch_max, int64_t& seq_min) {
  std::vector<T> cum;
  cum.reserve(batch.size() * 3 +
              1 * 3);  //[batch_cum, batch_seq_cum, seq] * num
  cum.push_back(T(0));
  cum.push_back(T(0));
  cum.push_back(T(0));
  T batch_acc = T(0);
  T batch_seq_acc = T(0);
  int64_t max_batch = 0;
  int64_t min_seq = seq[0];
  for (size_t idx = 0; idx < batch.size(); ++idx) {
    max_batch = std::max(max_batch, batch[idx]);
    min_seq = std::min(min_seq, seq[idx]);
    T seq_cur = static_cast<T>(seq[idx]);
    T batch_cur = static_cast<T>(batch[idx]);
    batch_acc += batch_cur;
    batch_seq_acc += batch_cur * seq_cur;
    cum.push_back(batch_acc);
    cum.push_back(batch_seq_acc);
    cum.push_back(seq_cur);
  }
  batch_max = max_batch;
  seq_min = min_seq;
  return cum;
}  // cumsum

using BATCH_T = int64_t;  // batch seq compute type
std::vector<torch::Tensor> ragged_tile(const std::vector<int64_t>& batch,
                                       const std::vector<int64_t>& seq,
                                       torch::Tensor value,
                                       torch::Tensor offset,
                                       torch::Tensor table,
                                       c10::optional<torch::Tensor> left_pad) {
  c10::ScalarType Tensor_T = c10::CppTypeToScalarType<BATCH_T>::value;
  int64_t dim = table.sizes()[1];
  int64_t batch_num = batch.size();
  int64_t batch_max = 0;
  int64_t seq_min = 0;
  auto batch_seq = cumsum<BATCH_T>(batch, seq, batch_max, seq_min);
  std::vector<int64_t> out_shape = {batch_seq[batch_seq.size() - 2], dim};
  torch::Tensor out = torch::empty(out_shape, table.options());
  int64_t batch_len = static_cast<int64_t>(batch.size()) * 3 + 3;
  torch::Tensor batch_seq_gpu =
      torch::empty({batch_len}, offset.options().dtype(Tensor_T));
  torch::Tensor left_pad_tensor;
  if (!left_pad.has_value()) {
    left_pad_tensor =
        torch::zeros({(int64_t)batch.size()},
                     offset.options().dtype(torch::kBool).device(torch::kCUDA));
  } else {
    left_pad_tensor = left_pad.value();
  }
  ragged_tile_gpu_impl<BATCH_T>(batch_seq, batch_seq_gpu, batch_num, batch_max,
                                seq_min, value, offset, table, out,
                                left_pad_tensor);
  return {out, batch_seq_gpu};
}  // ragged_tile

torch::Tensor ragged_tile_back(torch::Tensor batch_seq,
                               const std::vector<int64_t>& batch_info,
                               torch::Tensor value, torch::Tensor offset,
                               torch::Tensor dy, torch::Tensor left_pad) {
  int64_t dim = dy.sizes()[1];
  int64_t batch_num =
      batch_info[1];  //[table_rows, batch_len, batch_max, seq_min]
  int64_t batch_max = batch_info[2];
  int64_t seq_min = batch_info[3];
  int64_t table_rows = batch_info[0];
  std::vector<int64_t> out_shape = {table_rows, dim};
  torch::Tensor d_table = torch::zeros(out_shape, dy.options());
  ragged_tile_back_impl<BATCH_T>(batch_seq, batch_num, batch_max, seq_min,
                                 value, offset, d_table, dy, left_pad);
  return d_table;
}  // ragged_tile
}  // namespace functional
}  // namespace recis
