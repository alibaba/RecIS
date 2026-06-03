#pragma once
#include <cmath>
#include <string>
#include <unordered_map>

#include "embedding/optim.h"

namespace recis {
namespace optim {

namespace {

struct SparseAdagradOptions
    : public SparseOptimizerCloneableOptions<SparseAdagradOptions> {
  SparseAdagradOptions(double lr = 1e-2, double lr_decay = 0,
                       double initial_accumulator_value = 0, double eps = 1e-10,
                       double weight_decay = 0)
      : lr_(lr),
        lr_decay_(lr_decay),
        initial_accumulator_value_(initial_accumulator_value),
        eps_(eps),
        weight_decay_(weight_decay) {}
  TORCH_ARG(double, lr) = 1e-2;
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, initial_accumulator_value) = 0;
  TORCH_ARG(double, eps) = 1e-10;
  TORCH_ARG(double, weight_decay) = 0;

 public:
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API SparseAdagradParamState
    : public SparseOptimizerCloneableParamState<SparseAdagradParamState> {
 public:
  SparseAdagradParamState() : step_dtype_(torch::kInt64) {}
  using ParamContainer = at::intrusive_ptr<embedding::Slot>;
  void reset_step() { step_.zero_(); }
  TORCH_API friend bool operator==(const SparseAdagradParamState &lhs,
                                   const SparseAdagradParamState &rhs);
  TORCH_ARG(torch::Tensor, step);
  TORCH_ARG(ParamContainer, state_sum);
  TORCH_ARG(ParamContainer, param);
  TORCH_ARG(HashTablePtr, hashtable);
  TORCH_ARG(torch::Dtype, step_dtype);
};

}  // namespace

class SparseAdagrad : public SparseOptimizer {
 public:
  explicit SparseAdagrad(std::vector<SparseOptimizerParamGroup> param_groups,
                         SparseAdagradOptions defaults = {})
      : SparseOptimizer(std::move(param_groups),
                        std::make_unique<SparseAdagradOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    for (const auto &param_group : param_groups_) {
      for (const auto &param : param_group.params()) {
        init_param_state(param.second, param_group.options());
      }
    }
  }
  explicit SparseAdagrad(std::unordered_map<std::string, HashTablePtr> params,
                         SparseAdagradOptions defaults = {})
      : SparseAdagrad({SparseOptimizerParamGroup(std::move(params))},
                      defaults) {}
  const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                   std::unordered_map<std::string, torch::Tensor>>
  state_dict() override;
  void load_state_dict(torch::Dict<std::string, HashTablePtr> hashtables,
                       torch::Dict<std::string, torch::Tensor> steps) override;
  virtual void add_param_group(
      const SparseOptimizerParamGroup &param_group) override;
  virtual void add_parameters(
      const torch::Dict<std::string, HashTablePtr> &parameters) override;
  void init_param_state(HashTablePtr param,
                        const SparseOptimizerOptions &options);
  void step() override;
  static c10::intrusive_ptr<SparseAdagrad> Make(
      const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
      double lr_decay, double initial_accumulator_value, double eps,
      double weight_decay);
  void reset_state_dict();

 private:
  static const char *Prefix() { return "sparse_adagrad_"; }
  static std::string StepName() {
    static const std::string s = torch::str(Prefix(), "step");
    return s;
  }
  static std::string StateSumName() {
    static const std::string s = torch::str(Prefix(), "state_sum");
    return s;
  }
};

}  // namespace optim
}  // namespace recis
