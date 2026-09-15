#pragma once
#include <memory>
#include <string>
#include <unordered_map>

#include "ATen/core/ivalue.h"
#include "c10/util/flat_hash_map.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/string_view.h"
#include "embedding/hashtable.h"
#include "torch/extension.h"

namespace recis {
namespace optim {
class SparseOptimizerParamState {
 public:
  SparseOptimizerParamState() = default;
  SparseOptimizerParamState(const SparseOptimizerParamState &) = default;
  SparseOptimizerParamState &operator=(const SparseOptimizerParamState &) =
      default;
  SparseOptimizerParamState(SparseOptimizerParamState &&) noexcept = default;
  SparseOptimizerParamState &operator=(SparseOptimizerParamState &&) noexcept =
      default;
  virtual std::unique_ptr<SparseOptimizerParamState> clone() const;
  virtual ~SparseOptimizerParamState() = default;
};

template <typename Derived>
class SparseOptimizerCloneableParamState : public SparseOptimizerParamState {
  std::unique_ptr<SparseOptimizerParamState> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived &>(*this));
  }
};

class SparseOptimizerOptions {
 public:
  SparseOptimizerOptions() = default;
  SparseOptimizerOptions(const SparseOptimizerOptions &) = default;
  SparseOptimizerOptions &operator=(const SparseOptimizerOptions &) = default;
  SparseOptimizerOptions(SparseOptimizerOptions &&) noexcept = default;
  SparseOptimizerOptions &operator=(SparseOptimizerOptions &&) noexcept =
      default;
  virtual std::unique_ptr<SparseOptimizerOptions> clone() const;
  virtual ~SparseOptimizerOptions() = default;
  virtual double get_lr() const;
  virtual void set_lr(const double lr);
};

template <typename Derived>
class SparseOptimizerCloneableOptions : public SparseOptimizerOptions {
 private:
  std::unique_ptr<SparseOptimizerOptions> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived &>(*this));
  }
};

class SparseOptimizerParamGroup {
 public:
  // NOTE: In order to store `SparseOptimizerParamGroup` in a `std::vector`, it
  // has to be copy-constructible.
  SparseOptimizerParamGroup(const SparseOptimizerParamGroup &param_group)
      : params_(param_group.params()),
        options_(param_group.has_options() ? param_group.options().clone()
                                           : nullptr) {}
  SparseOptimizerParamGroup(
      std::unordered_map<std::string, HashTablePtr> params)
      : SparseOptimizerParamGroup(std::move(params), nullptr) {}
  SparseOptimizerParamGroup(
      std::unordered_map<std::string, HashTablePtr> params,
      std::unique_ptr<SparseOptimizerOptions> options)
      : options_(std::move(options)) {
    for (auto &param : params) {
      if (!param.second.defined() || param.second->RequiresOptimizerState()) {
        params_.emplace(param.first, std::move(param.second));
      }
    }
  }

  bool has_options() const;
  SparseOptimizerOptions &options();
  const SparseOptimizerOptions &options() const;
  void set_options(std::unique_ptr<SparseOptimizerOptions> options);
  std::unordered_map<std::string, HashTablePtr> &params();
  const std::unordered_map<std::string, HashTablePtr> &params() const;

 protected:
  std::unordered_map<std::string, HashTablePtr> params_;
  std::unique_ptr<SparseOptimizerOptions> options_;
};

class SparseOptimizer : public torch::CustomClassHolder {
 public:
  // The copy constructor is deleted, because the user should use the
  // `state_dict` / `load_state_dict` API to copy an optimizer instead.
  SparseOptimizer(const SparseOptimizer &optimizer) = delete;
  SparseOptimizer(SparseOptimizer &&optimizer) = default;

  explicit SparseOptimizer(std::vector<SparseOptimizerParamGroup> param_groups,
                           std::unique_ptr<SparseOptimizerOptions> defaults)
      : defaults_(std::move(defaults)) {
    for (const auto &param_group : param_groups) {
      add_param_group(param_group);
    }
  }

  /// Constructs the `Optimizer` from a vector of parameters.
  explicit SparseOptimizer(
      std::unordered_map<std::string, HashTablePtr> parameters,
      std::unique_ptr<SparseOptimizerOptions> defaults)
      : SparseOptimizer({SparseOptimizerParamGroup(std::move(parameters))},
                        std::move(defaults)) {};

  /// Adds the given param_group to the optimizer's param_group list.
  virtual void add_param_group(const SparseOptimizerParamGroup &param_group);

  virtual ~SparseOptimizer() = default;

  using LossClosure = std::function<at::Tensor()>;
  /// A loss function closure, which is expected to return the loss value.
  virtual void step() = 0;
  /// this is used to get all state variable for param
  virtual const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                           std::unordered_map<std::string, torch::Tensor>>
  state_dict() = 0;
  virtual void load_state_dict(
      torch::Dict<std::string, HashTablePtr> hashtables,
      torch::Dict<std::string, torch::Tensor> steps) = 0;
  virtual void add_parameters(
      const torch::Dict<std::string, HashTablePtr> &parameters);

  /// Zeros out the gradients of all parameters.
  void zero_grad(bool set_to_none = true);

  /// Zeros out the gradient squares of all parameters (for SparseAdagradSum).
  void zero_grad_sq();

  /// Provides a const reference to the parameters in the first param_group this
  /// optimizer holds.
  const std::unordered_map<std::string, HashTablePtr> &parameters()
      const noexcept;

  /// Provides a reference to the parameters in the first param_group this
  /// optimizer holds.
  const std::unordered_map<std::string, HashTablePtr> &parameters() noexcept;

  /// Returns the number of parameters referenced by the optimizer.
  size_t size() const noexcept;

  SparseOptimizerOptions &defaults() noexcept;

  const SparseOptimizerOptions &defaults() const noexcept;

  /// Provides a reference to the param_groups this optimizer holds.
  std::vector<SparseOptimizerParamGroup> &param_groups() noexcept;

  /// Provides a const reference to the param_groups this optimizer holds.
  const std::vector<SparseOptimizerParamGroup> &param_groups() const noexcept;

  /// Set gradient accum steps of this optimizer
  void set_grad_accum_steps(const int64_t steps) noexcept;

  /// Set learning rate of this optimizer
  void set_lr(const double lr) noexcept;

  /// Provides a reference to the state this optimizer holds
  ska::flat_hash_map<void *, std::unique_ptr<SparseOptimizerParamState>> &
  state() noexcept;

  /// Provides a const reference to the state this optimizer holds
  const ska::flat_hash_map<void *, std::unique_ptr<SparseOptimizerParamState>> &
  state() const noexcept;

 protected:
  std::vector<SparseOptimizerParamGroup> param_groups_;
  ska::flat_hash_map<void *, std::unique_ptr<SparseOptimizerParamState>> state_;
  std::unique_ptr<SparseOptimizerOptions> defaults_;
  int64_t grad_accum_steps_ = 1;
};
}  // namespace optim
}  // namespace recis
