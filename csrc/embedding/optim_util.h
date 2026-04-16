#pragma once
#include <torch/torch.h>

#include "optim.h"

namespace recis {
namespace optim {
namespace utils {

inline at::SparseTensorImpl* get_sparse_impl(const at::Tensor& self) {
  return static_cast<at::SparseTensorImpl*>(self.unsafeGetTensorImpl());
}

template <typename OptionsType, typename StateType, typename UpdateKernel>
void apply_sparse_step(
    std::vector<SparseOptimizerParamGroup>& param_groups,
    ska::flat_hash_map<void*, std::unique_ptr<SparseOptimizerParamState>>&
        states,
    UpdateKernel&& update_kernel) {
  torch::NoGradGuard no_grad;
  for (auto& group : param_groups) {
    auto& options = static_cast<OptionsType&>(group.options());
    for (auto& it : group.params()) {
      const std::string& name = it.first;
      auto& p = it.second;
      if (!p.defined()) continue;
      if (!p->HasGrad()) continue;
      const auto& grad = p->Grad();
      if (!grad.defined()) continue;
      TORCH_CHECK(grad.is_sparse(),
                  "SparseOptimizer only supports sparse gradients. Got dense "
                  "gradient for param.");
      auto state_it = states.find(p.get());
      TORCH_CHECK(state_it != states.end(),
                  "Optimizer state not found. Please check your sparse states "
                  "passed to the sparse optimizer");
      auto& state = static_cast<StateType&>(*state_it->second);
      update_kernel(name, p, grad, options, state);
    }
  }
}

}  // namespace utils
}  // namespace optim
}  // namespace recis
