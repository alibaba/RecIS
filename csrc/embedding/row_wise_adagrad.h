#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "embedding/optim.h"

namespace recis {
namespace optim {

namespace {

struct SparseRowWiseAdagradOptions
    : public SparseOptimizerCloneableOptions<SparseRowWiseAdagradOptions> {
  SparseRowWiseAdagradOptions(double lr = 1e-3, double lr_decay = 0,
                              double initial_accumulator_value = 0,
                              double eps = 1e-10, double weight_decay = 0,
                              bool maximize = false)
      : lr_(lr),
        lr_decay_(lr_decay),
        initial_accumulator_value_(initial_accumulator_value),
        eps_(eps),
        weight_decay_(weight_decay),
        maximize_(maximize) {}
  TORCH_ARG(double, lr) = 1e-3;
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, initial_accumulator_value) = 0;
  TORCH_ARG(double, eps) = 1e-10;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(bool, maximize) = false;

 public:
  double get_lr() const override;
  void set_lr(double lr) override;
};

struct TORCH_API SparseRowWiseAdagradParamState
    : public SparseOptimizerCloneableParamState<
          SparseRowWiseAdagradParamState> {
 public:
  SparseRowWiseAdagradParamState() : step_dtype_(torch::kInt64) {}
  using ParamContainer = at::intrusive_ptr<embedding::Slot>;
  void reset_step() { step_.zero_(); }
  TORCH_ARG(torch::Tensor, step);
  TORCH_ARG(ParamContainer, state_sum);
  TORCH_ARG(ParamContainer, param);
  TORCH_ARG(HashTablePtr, hashtable);
  TORCH_ARG(torch::Dtype, step_dtype);
};

}  // namespace

class SparseRowWiseAdagrad : public SparseOptimizer {
 public:
  explicit SparseRowWiseAdagrad(
      std::vector<SparseOptimizerParamGroup> param_groups,
      SparseRowWiseAdagradOptions defaults = {})
      : SparseOptimizer(
            std::move(param_groups),
            std::make_unique<SparseRowWiseAdagradOptions>(defaults)) {
    ValidateOptions(defaults);
    std::unordered_set<std::string> param_names;
    std::unordered_set<void *> params;
    for (const auto &param_group : param_groups_) {
      ValidateOptions(static_cast<const SparseRowWiseAdagradOptions &>(
          param_group.options()));
      for (const auto &param : param_group.params()) {
        TORCH_CHECK(
            param.second.defined(),
            "Cannot optimize an undefined HashTable parameter: ", param.first);
        TORCH_CHECK(param_names.insert(param.first).second,
                    "SparseRowWiseAdagrad parameter names must be unique: ",
                    param.first);
        TORCH_CHECK(params.insert(param.second.get()).second,
                    "some parameters appear in more than one parameter group.");
      }
    }
    for (const auto &param_group : param_groups_) {
      for (const auto &param : param_group.params()) {
        init_param_state(param.second, param_group.options());
      }
    }
  }
  explicit SparseRowWiseAdagrad(
      std::unordered_map<std::string, HashTablePtr> params,
      SparseRowWiseAdagradOptions defaults = {})
      : SparseRowWiseAdagrad({SparseOptimizerParamGroup(std::move(params))},
                             defaults) {}

  const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                   std::unordered_map<std::string, torch::Tensor>>
  state_dict() override;
  void load_state_dict(torch::Dict<std::string, HashTablePtr> hashtables,
                       torch::Dict<std::string, torch::Tensor> steps) override;
  void add_param_group(const SparseOptimizerParamGroup &param_group) override;
  void add_parameters(
      const torch::Dict<std::string, HashTablePtr> &parameters) override;
  void init_param_state(HashTablePtr param,
                        const SparseOptimizerOptions &options);
  void step() override;
  static c10::intrusive_ptr<SparseRowWiseAdagrad> Make(
      const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
      double lr_decay, double initial_accumulator_value, double eps,
      double weight_decay, bool maximize);
  void reset_state_dict();

 private:
  static void ValidateOptions(const SparseRowWiseAdagradOptions &options);
  static const char *Prefix() { return "sparse_row_wise_adagrad_"; }
  static std::string StepName() {
    static const std::string s = torch::str(Prefix(), "step");
    return s;
  }
  static std::string StepName(const std::string &param_name) {
    return torch::str(Prefix(), param_name, "_step");
  }
  static std::string StateSumName() {
    static const std::string s = torch::str(Prefix(), "state_sum");
    return s;
  }
};

class SparseRowWiseAdagradSum : public SparseOptimizer {
 public:
  explicit SparseRowWiseAdagradSum(
      std::vector<SparseOptimizerParamGroup> param_groups,
      SparseRowWiseAdagradOptions defaults = {})
      : SparseOptimizer(
            std::move(param_groups),
            std::make_unique<SparseRowWiseAdagradOptions>(defaults)) {
    ValidateOptions(defaults);
    std::unordered_set<std::string> param_names;
    std::unordered_set<void *> params;
    for (const auto &param_group : param_groups_) {
      ValidateOptions(static_cast<const SparseRowWiseAdagradOptions &>(
          param_group.options()));
      for (const auto &param : param_group.params()) {
        TORCH_CHECK(
            param.second.defined(),
            "Cannot optimize an undefined HashTable parameter: ", param.first);
        TORCH_CHECK(param_names.insert(param.first).second,
                    "SparseRowWiseAdagradSum parameter names must be unique: ",
                    param.first);
        TORCH_CHECK(params.insert(param.second.get()).second,
                    "some parameters appear in more than one parameter group.");
      }
    }
    for (const auto &param_group : param_groups_) {
      for (const auto &param : param_group.params()) {
        init_param_state(param.second, param_group.options());
      }
    }
  }
  explicit SparseRowWiseAdagradSum(
      std::unordered_map<std::string, HashTablePtr> params,
      SparseRowWiseAdagradOptions defaults = {})
      : SparseRowWiseAdagradSum({SparseOptimizerParamGroup(std::move(params))},
                                defaults) {}

  const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                   std::unordered_map<std::string, torch::Tensor>>
  state_dict() override;
  void load_state_dict(torch::Dict<std::string, HashTablePtr> hashtables,
                       torch::Dict<std::string, torch::Tensor> steps) override;
  void add_param_group(const SparseOptimizerParamGroup &param_group) override;
  void add_parameters(
      const torch::Dict<std::string, HashTablePtr> &parameters) override;
  void init_param_state(HashTablePtr param,
                        const SparseOptimizerOptions &options);
  void step() override;
  static c10::intrusive_ptr<SparseRowWiseAdagradSum> Make(
      const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
      double lr_decay, double initial_accumulator_value, double eps,
      double weight_decay, bool maximize);
  void reset_state_dict();

 private:
  static void ValidateOptions(const SparseRowWiseAdagradOptions &options);
  static const char *Prefix() { return "sparse_row_wise_adagrad_sum_"; }
  static std::string StepName() {
    static const std::string s = torch::str(Prefix(), "step");
    return s;
  }
  static std::string StepName(const std::string &param_name) {
    return torch::str(Prefix(), param_name, "_step");
  }
  static std::string StateSumName() {
    static const std::string s = torch::str(Prefix(), "state_sum");
    return s;
  }
};

}  // namespace optim
}  // namespace recis
