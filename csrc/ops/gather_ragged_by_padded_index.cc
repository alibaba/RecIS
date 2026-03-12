#include "ops/gather_ragged_by_padded_index.h"

#include <tuple>

#include "ATen/Dispatch.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include "c10/util/irange.h"
#include "ops/ragged_common.cuh"
#include "torch/csrc/autograd/generated/variable_factories.h"
namespace recis {
namespace functional {
template <class T_PAD, class T_PI, class T_SRI, class T_OF, class T_V>
void gather_ragged_by_padded_index_single(
    const T_PAD *drop_num_v, const bool drop_side_val, const T_PAD *pad_num_v,
    const bool pad_side_val, const T_PI *padded_indices_v,
    const T_SRI *src_row_indices_v, const T_OF *offset_v, const T_V *value_v,
    T_V *out_value_v, bool *out_mask_v, int64_t row_num, int64_t col_num) {
  for (auto row_idx : c10::irange(row_num)) {
    int64_t src_row_index = src_row_indices_v[row_idx];
    int64_t src_row_beg = offset_v[src_row_index];
    int64_t src_row_end = offset_v[src_row_index + 1];
    int64_t drop_num = drop_num_v[src_row_index] * drop_side_val;
    int64_t pad_num = pad_num_v[src_row_index] * pad_side_val;
    for (int64_t col_idx : c10::irange(col_num)) {
      int64_t dst_val_index = row_idx * col_num + col_idx;
      int64_t src_col_index =
          padded_indices_v[dst_val_index] + drop_num - pad_num;
      int64_t src_val_index = src_row_beg + src_col_index;
      out_mask_v[dst_val_index] = false;
      out_value_v[dst_val_index] = 0;
      if (src_val_index >= src_row_beg && src_val_index < src_row_end) {
        out_value_v[dst_val_index] = value_v[src_val_index];
        out_mask_v[dst_val_index] = true;
      }
    }
  }
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
gather_ragged_by_padded_index_cpu(at::Tensor drop_num, at::Tensor drop_side,
                                  at::Tensor pad_num, at::Tensor pad_side,
                                  at::Tensor padded_indices,
                                  at::Tensor src_row_indices,
                                  std::vector<at::Tensor> offsets,
                                  std::vector<at::Tensor> values) {
  auto pad_side_val = pad_side.cpu().item<bool>();
  auto drop_side_val = drop_side.cpu().item<bool>();
  pad_num = pad_num.cpu();
  drop_num = drop_num.cpu();
  padded_indices = padded_indices.cpu();
  src_row_indices = src_row_indices.cpu();
  int64_t row_num = padded_indices.size(0);
  int64_t col_num = padded_indices.size(1);
  std::vector<at::Tensor> out_values;
  for (auto input_index : c10::irange(offsets.size())) {
    out_values.push_back(torch::empty(
        row_num * col_num,
        values[input_index].options().pinned_memory(true).device(torch::kCPU)));
  }
  auto out_masks = torch::empty({int64_t(offsets.size()), row_num * col_num},
                                values[0]
                                    .options()
                                    .pinned_memory(true)
                                    .device(torch::kCPU)
                                    .dtype(torch::kBool));
  int64_t input_row_num = offsets[0].numel() - 1;
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
                                  gather_ragged_by_padded_index_single<
                                      T_PAD, T_PI, T_SRI, T_OF, T_V>(
                                      drop_num.data_ptr<T_PAD>(), drop_side_val,
                                      pad_num.data_ptr<T_PAD>(), pad_side_val,
                                      padded_indices.data_ptr<T_PI>(),
                                      src_row_indices.data_ptr<T_SRI>(),
                                      offsets[input_index]
                                          .cpu()
                                          .data_ptr<T_OF>(),
                                      values[input_index].cpu().data_ptr<T_V>(),
                                      out_value.data_ptr<T_V>(),
                                      out_mask.data_ptr<bool>(), row_num,
                                      col_num);
                                });
                          });
                    }
                  });
            });
      });
  out_masks = out_masks.to(torch::kCUDA, true);
  std::vector<at::Tensor> out_value_list(offsets.size());
  std::vector<at::Tensor> out_mask_list(offsets.size());
  for (auto input_index : c10::irange(offsets.size())) {
    out_value_list[input_index] =
        out_values[input_index].to(torch::kCUDA, true).flatten();
    out_mask_list[input_index] = out_masks.narrow(0, input_index, 1).flatten();
  }
  return std::make_tuple(out_value_list, out_mask_list);
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
gather_ragged_by_padded_index(at::Tensor drop_num, at::Tensor drop_side,
                              at::Tensor pad_num, at::Tensor pad_side,
                              at::Tensor padded_indices,
                              at::Tensor src_row_indices,
                              std::vector<at::Tensor> offsets,
                              std::vector<at::Tensor> values) {
  if (all_cuda(values)) {
    return gather_ragged_by_padded_index_cuda(drop_num, drop_side, pad_num,
                                              pad_side, padded_indices,
                                              src_row_indices, offsets, values);
  } else {
    return gather_ragged_by_padded_index_cpu(drop_num, drop_side, pad_num,
                                             pad_side, padded_indices,
                                             src_row_indices, offsets, values);
  }
}
}  // namespace functional
}  // namespace recis