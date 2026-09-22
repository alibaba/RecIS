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

  // Store updated indices and their corresponding state_sum, parameter and
  // gradient values
  torch::Tensor updated_indices_;
  torch::Tensor updated_state_sum_;
  torch::Tensor updated_params_before_;
  torch::Tensor updated_params_after_;
  torch::Tensor updated_grad_;

 public:
  // Public methods for accessing update information

  // Store the step number when update info was last saved
  int64_t last_saved_step_;

  const torch::Tensor &updated_indices() const { return updated_indices_; }
  void set_updated_indices(const torch::Tensor &indices) {
    updated_indices_ = indices;
  }

  const torch::Tensor &updated_state_sum() const { return updated_state_sum_; }
  void set_updated_state_sum(const torch::Tensor &state_sum) {
    updated_state_sum_ = state_sum;
  }

  const torch::Tensor &updated_params_before() const {
    return updated_params_before_;
  }
  void set_updated_params_before(const torch::Tensor &params) {
    updated_params_before_ = params;
  }

  const torch::Tensor &updated_params_after() const {
    return updated_params_after_;
  }
  void set_updated_params_after(const torch::Tensor &params) {
    updated_params_after_ = params;
  }

  const torch::Tensor &updated_grad() const { return updated_grad_; }
  void set_updated_grad(const torch::Tensor &grad) { updated_grad_ = grad; }

  void clear_update_info() {
    updated_indices_ = torch::Tensor();
    updated_state_sum_ = torch::Tensor();
    updated_params_before_ = torch::Tensor();
    updated_params_after_ = torch::Tensor();
    updated_grad_ = torch::Tensor();
    last_saved_step_ = -1;
  }
};

}  // namespace

class SparseAdagrad : public SparseOptimizer {
 public:
  explicit SparseAdagrad(std::vector<SparseOptimizerParamGroup> param_groups,
                         SparseAdagradOptions defaults = {},
                         int64_t save_update_info_interval = 0)
      : SparseOptimizer(std::move(param_groups),
                        std::make_unique<SparseAdagradOptions>(defaults)),
        save_update_info_interval_(save_update_info_interval) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    TORCH_CHECK(save_update_info_interval_ >= 0,
                "save_update_info_interval must be >= 0 (0 means never save)");
    for (const auto &param_group : param_groups_) {
      for (const auto &param : param_group.params()) {
        init_param_state(param.second, param_group.options());
      }
    }
  }
  explicit SparseAdagrad(std::unordered_map<std::string, HashTablePtr> params,
                         SparseAdagradOptions defaults = {},
                         int64_t save_update_info_interval = 0)
      : SparseAdagrad({SparseOptimizerParamGroup(std::move(params))}, defaults,
                      save_update_info_interval) {}
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
      double weight_decay, int64_t save_update_info_interval = 0);
  void reset_state_dict();

  // Get update information from the last saved step.
  // Returns an empty dict when update info was not saved or no sparse rows were
  // updated.
  torch::Dict<std::string, torch::Tensor> get_step_update_info();

  // Clear update information
  void clear_step_update_info();

  // Set the interval for saving update info. 0 means never save; N > 0 means
  // save every N steps.
  void set_save_update_info_interval(int64_t interval);

  // Get the current save interval
  int64_t save_update_info_interval() const {
    return save_update_info_interval_;
  }

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

  // Interval for saving update information. 0 means never save.
  int64_t save_update_info_interval_;
};

// SparseAdagradSum uses sum of gradients and sum of squared gradients instead
// of averaged gradients. It does not support weight_decay.
class SparseAdagradSum : public SparseOptimizer {
 public:
  explicit SparseAdagradSum(std::vector<SparseOptimizerParamGroup> param_groups,
                            SparseAdagradOptions defaults = {},
                            int64_t save_update_info_interval = 0)
      : SparseOptimizer(std::move(param_groups),
                        std::make_unique<SparseAdagradOptions>(defaults)),
        save_update_info_interval_(save_update_info_interval) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    TORCH_CHECK(save_update_info_interval_ >= 0,
                "save_update_info_interval must be >= 0 (0 means never save)");
    for (const auto &param_group : param_groups_) {
      for (const auto &param : param_group.params()) {
        init_param_state(param.second, param_group.options());
      }
    }
  }
  explicit SparseAdagradSum(
      std::unordered_map<std::string, HashTablePtr> params,
      SparseAdagradOptions defaults = {}, int64_t save_update_info_interval = 0)
      : SparseAdagradSum({SparseOptimizerParamGroup(std::move(params))},
                         defaults, save_update_info_interval) {}
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
  static c10::intrusive_ptr<SparseAdagradSum> Make(
      const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
      double lr_decay, double initial_accumulator_value, double eps,
      int64_t save_update_info_interval = 0);
  void reset_state_dict();

  // Get update information from the last saved step.
  // Returns an empty dict when update info was not saved or no sparse rows were
  // updated.
  torch::Dict<std::string, torch::Tensor> get_step_update_info();

  // Clear update information
  void clear_step_update_info();

  // Set the interval for saving update info. 0 means never save; N > 0 means
  // save every N steps.
  void set_save_update_info_interval(int64_t interval);

  // Get the current save interval
  int64_t save_update_info_interval() const {
    return save_update_info_interval_;
  }

 private:
  // Uses the same state slot names as SparseAdagrad so existing checkpoints can
  // be inherited when switching to SparseAdagradSum with sparse gradient
  // groups.
  static const char *Prefix() { return "sparse_adagrad_"; }
  static std::string StepName() {
    static const std::string s = torch::str(Prefix(), "step");
    return s;
  }
  static std::string StateSumName() {
    static const std::string s = torch::str(Prefix(), "state_sum");
    return s;
  }

  // SparseAdagradSum does not support weight_decay. Its Make factory
  // intentionally does not expose weight_decay because the sum-update kernel
  // consumes pre-computed squared gradients and does not apply L2 penalty.

  // Interval for saving update information. 0 means never save.
  int64_t save_update_info_interval_;
};

}  // namespace optim
}  // namespace recis
