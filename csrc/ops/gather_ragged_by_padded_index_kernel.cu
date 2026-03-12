#include <c10/util/irange.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include "c10/core/DeviceType.h"
namespace recis {
namespace functional {
template <class T_PAD, class T_PI, class T_SRI, class T_OF, class T_V>
__global__ void gather_ragged_by_padded_index_single_kernel(
    const T_PAD *__restrict__ drop_num_v, const bool *__restrict__ drop_side_v,
    const T_PAD *__restrict__ pad_num_v, const bool *__restrict__ pad_side_v,
    const T_PI *__restrict__ padded_indices_v,
    const T_SRI *__restrict__ src_row_indices_v,
    const T_OF *__restrict__ offset_v, const T_V *__restrict__ value_v,
    T_V *__restrict__ out_value_v, bool *__restrict__ out_mask_v,
    int64_t row_num, int64_t col_num) {
  bool drop_side_val = drop_side_v[0];
  bool pad_side_val = pad_side_v[0];
  const int64_t total_elements = row_num * col_num;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= total_elements) {
    return;
  }

  const int64_t row_idx = idx / col_num;
  const int64_t col_idx = idx % col_num;

  int64_t src_row_index = src_row_indices_v[row_idx];
  int64_t src_row_beg = offset_v[src_row_index];
  int64_t src_row_end = offset_v[src_row_index + 1];
  int64_t drop_num = drop_num_v[src_row_index] * drop_side_val;
  int64_t pad_num = pad_num_v[src_row_index] * pad_side_val;

  int64_t dst_val_index = idx;
  int64_t src_col_index = padded_indices_v[dst_val_index] + drop_num - pad_num;
  int64_t src_val_index = src_row_beg + src_col_index;

  out_mask_v[dst_val_index] = false;
  out_value_v[dst_val_index] = static_cast<T_V>(0);

  if (src_val_index >= src_row_beg && src_val_index < src_row_end) {
    out_value_v[dst_val_index] = value_v[src_val_index];
    out_mask_v[dst_val_index] = true;
  }
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
gather_ragged_by_padded_index_cuda(at::Tensor drop_num, at::Tensor drop_side,
                                   at::Tensor pad_num, at::Tensor pad_side,
                                   at::Tensor padded_indices,
                                   at::Tensor src_row_indices,
                                   std::vector<at::Tensor> offsets,
                                   std::vector<at::Tensor> values) {
  int64_t row_num = padded_indices.size(0);
  int64_t col_num = padded_indices.size(1);
  std::vector<at::Tensor> out_values;
  for (auto input_index : c10::irange(offsets.size())) {
    out_values.push_back(
        torch::empty(row_num * col_num, values[input_index].options()));
  }
  auto out_masks = torch::empty(
      {int64_t(offsets.size()), row_num * col_num},
      values[0].options().device(torch::kCUDA).dtype(torch::kBool));
  int64_t input_row_num = offsets[0].numel() - 1;
  const int64_t total_elements = row_num * col_num;
  const int threads = 128;
  const int64_t blocks = (total_elements + threads - 1) / threads;
  AT_DISPATCH_INDEX_TYPES(
      drop_num.scalar_type(), "gather_ragged_by_padded_index_T_DN", [&]() {
        using T_PAD = index_t;
        AT_DISPATCH_INDEX_TYPES(
            padded_indices.scalar_type(), "gather_ragged_by_padded_index_T_PI",
            [&]() {
              using T_PI = index_t;
              AT_DISPATCH_INDEX_TYPES(
                  src_row_indices.scalar_type(),
                  "gather_ragged_by_padded_index_T_SRI", [&]() {
                    using T_SRI = index_t;
                    for (auto input_index : c10::irange(offsets.size())) {
                      TORCH_CHECK(
                          offsets[input_index].numel() == input_row_num + 1,
                          "offsets[input_index].numel() == row_num + 1");
                      AT_DISPATCH_INDEX_TYPES(
                          offsets[input_index].scalar_type(),
                          "gather_ragged_by_padded_index_T_OF", [&]() {
                            using T_OF = index_t;
                            AT_DISPATCH_ALL_TYPES(
                                values[input_index].scalar_type(),
                                "gather_ragged_by_padded_index_T_V", [&]() {
                                  using T_V = scalar_t;
                                  auto &out_value = out_values[input_index];
                                  auto out_mask =
                                      out_masks.narrow(0, input_index, 1);
                                  gather_ragged_by_padded_index_single_kernel<
                                      T_PAD, T_PI, T_SRI, T_OF, T_V>
                                      <<<blocks, threads>>>(
                                          drop_num.data_ptr<T_PAD>(),
                                          drop_side.data_ptr<bool>(),
                                          pad_num.data_ptr<T_PAD>(),
                                          pad_side.data_ptr<bool>(),
                                          padded_indices.data_ptr<T_PI>(),
                                          src_row_indices.data_ptr<T_SRI>(),
                                          offsets[input_index].data_ptr<T_OF>(),
                                          values[input_index].data_ptr<T_V>(),
                                          out_value.data_ptr<T_V>(),
                                          out_mask.data_ptr<bool>(), row_num,
                                          col_num);
                                });
                          });
                    }
                  });
            });
      });
  std::vector<at::Tensor> out_mask_list(offsets.size());
  for (auto input_index : c10::irange(offsets.size())) {
    out_mask_list[input_index] = out_masks.narrow(0, input_index, 1).flatten();
  }
  return std::make_tuple(out_values, out_mask_list);
}
}  // namespace functional
}  // namespace recis