#include "ops/gauc.h"

#include <cstddef>
#include <cstring>
#include <tuple>

#include "ATen/Dispatch.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Exception.h"
#include "c10/util/irange.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

namespace {
template <typename T>
T GetNonClick(T *plabels, size_t k, int dim) {
  if (dim == 1) return 1.0 - plabels[k];
  return plabels[2 * k];
}

template <typename T>
T GetClick(T *plabels, size_t k, int dim) {
  if (dim == 1) return plabels[k];
  return plabels[2 * k + 1];
}

template <typename T>
bool ComputeGauc(T *plabels, T *ppreds, T *pfilter, size_t *pidx, size_t l,
                 size_t r, int dim, double *ret, double *click_count) {
  std::sort(pidx + l, pidx + r, [ppreds, dim](size_t a, size_t b) {
    return GetClick<T>(ppreds, a, dim) < GetClick<T>(ppreds, b, dim);
  });
  double fp1, tp1, fp2, tp2, auc;
  fp1 = tp1 = fp2 = tp2 = auc = 0;
  size_t i;
  for (size_t k = l; k < r; ++k) {
    i = pidx[k];
    if (pfilter != nullptr && pfilter[i] == 0) continue;
    fp2 += GetNonClick<T>(plabels, i, dim);
    tp2 += GetClick<T>(plabels, i, dim);
    auc += (fp2 - fp1) * (tp2 + tp1);
    fp1 = fp2;
    tp1 = tp2;
  }
  double threshold = static_cast<double>(r - l) - 1e-3;
  *click_count = tp2;
  if (tp2 > threshold or fp2 > threshold) {
    *ret = -0.5;
    return true;
  }
  if (tp2 * fp2 > 0) {
    *ret = (1.0 - auc / (2.0 * tp2 * fp2));
    return true;
  }
  return false;
}
}  // namespace
namespace recis {
namespace functional {
std::tuple<torch::Tensor, torch::Tensor> GaucCalc(torch::Tensor labels,
                                                  torch::Tensor predictions,
                                                  torch::Tensor indicators) {
  // check shape
  TORCH_CHECK(indicators.dim() == 1);
  TORCH_CHECK((labels.dim() == 1 || labels.dim() == 2))
  if (labels.dim() == 2) TORCH_CHECK(labels.size(1) == 2);
  TORCH_CHECK(labels.dim() == predictions.dim())
  for (auto i : c10::irange(labels.dim())) {
    TORCH_CHECK(labels.size(i) == predictions.size(i));
  }

  size_t ldim = labels.dim();
  size_t n = labels.size(0);
  std::vector<size_t> index(n);
  for (size_t i = 0; i < n; i++) {
    index[i] = i;
  }

  std::vector<double> auc_values;
  std::vector<int64_t> count_values;

  AT_DISPATCH_FLOATING_TYPES(labels.scalar_type(), "CaucCalc", [&]() {
    auto labels_ptr = labels.data_ptr<scalar_t>();
    auto prediction_ptr = predictions.data_ptr<scalar_t>();
    AT_DISPATCH_INDEX_TYPES(indicators.scalar_type(), "CaucCalc", [&]() {
      auto indicator_ptr = indicators.data_ptr<index_t>();
      size_t begin = 0, end = 0;
      while (begin < n) {
        while (end < n && indicator_ptr[end] == indicator_ptr[begin]) ++end;
        if (end - begin > 1) {
          double auc = 0, click_count = 0;
          if (ComputeGauc<scalar_t>(labels_ptr, prediction_ptr, nullptr,
                                    index.data(), begin, end, ldim, &auc,
                                    &click_count)) {
            if (auc >= 0) {
              auc_values.emplace_back(auc);
              count_values.emplace_back(click_count);
            }
          }
        }
        begin = end;
      }
    });
  });
  auto aucs = torch::empty({int64_t(auc_values.size())},
                           torch::TensorOptions()
                               .device(torch::kCPU)
                               .dtype(at::CppTypeToScalarType<double>::value));
  auto counts =
      torch::empty({int64_t(auc_values.size())},
                   torch::TensorOptions()
                       .device(torch::kCPU)
                       .dtype(at::CppTypeToScalarType<int64_t>::value));
  memcpy(aucs.data_ptr(), auc_values.data(),
         auc_values.size() * sizeof(double));
  memcpy(counts.data_ptr(), count_values.data(),
         count_values.size() * sizeof(int64_t));
  return std::make_tuple(aucs, counts);
}
}  // namespace functional
}  // namespace recis
