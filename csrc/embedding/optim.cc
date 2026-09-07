#include "embedding/optim.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "c10/util/flat_hash_map.h"
#include "embedding/hashtable.h"
namespace recis {
namespace optim {

bool SparseOptimizerParamGroup::has_options() const {
  return options_ != nullptr;
}

SparseOptimizerOptions &SparseOptimizerParamGroup::options() {
  return *options_;
}

const SparseOptimizerOptions &SparseOptimizerParamGroup::options() const {
  return *options_;
}

void SparseOptimizerParamGroup::set_options(
    std::unique_ptr<SparseOptimizerOptions> options) {
  options_ = std::move(options);
}

std::unordered_map<std::string, HashTablePtr> &
SparseOptimizerParamGroup::params() {
  return params_;
}

const std::unordered_map<std::string, HashTablePtr> &
SparseOptimizerParamGroup::params() const {
  return params_;
}

std::unique_ptr<SparseOptimizerParamState> SparseOptimizerParamState::clone()
    const {
  TORCH_CHECK(
      false,
      "clone() has not been implemented for "
      "torch::optim::OptimizerParamState. ",
      "Subclass "
      "torch::optim::OptimizerCloneableParamState<YourOptimizerParamState> ",
      "instead of torch::optim::OptimizerParamState to inherit the ability to "
      "clone.");
}

double SparseOptimizerOptions::get_lr() const {
  TORCH_CHECK(
      false,
      "double get_lr() has not been overridden and implemented in subclass of "
      "torch::optim::OptimizerOptions, you must override it in your subclass.");
}

void SparseOptimizerOptions::set_lr(const double lr) {
  TORCH_CHECK(
      false,
      "double set_lr() has not been overridden and implemented in subclass of "
      "torch::optim::OptimizerOptions, you must override it in your subclass.");
}

std::unique_ptr<SparseOptimizerOptions> SparseOptimizerOptions::clone() const {
  TORCH_CHECK(
      false,
      "clone() has not been implemented for torch::optim::OptimizerOptions. ",
      "Subclass torch::optim::OptimizerCloneableOptions<YourOptimizerOptions> ",
      "instead of torch::optim::OptimizerOptions to inherit the ability to "
      "clone.");
}

void SparseOptimizer::add_param_group(
    const SparseOptimizerParamGroup &param_group) {
  SparseOptimizerParamGroup param_group_(param_group.params());
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group.options().clone());
  }
  for (const auto &param : param_group_.params()) {
    TORCH_CHECK(state_.count(param.second.get()) == 0,
                "some parameters appear in more than one parameter group.");
  }
  param_groups_.emplace_back(param_group_);
}

void SparseOptimizer::add_parameters(
    const torch::Dict<std::string, HashTablePtr> &parameters) {
  auto &parameters_ = param_groups_[0].params();
  for (auto it = parameters.begin(); it != parameters.end(); it++) {
    parameters_[it->key()] = it->value();
  }
}

void SparseOptimizer::zero_grad(bool set_to_none) {
  for (auto &group : param_groups_) {
    for (auto &p : group.params()) {
      p.second->ClearGrad();
    }
  }
}

void SparseOptimizer::zero_grad_sq() {
  for (auto &group : param_groups_) {
    for (auto &p : group.params()) {
      p.second->ClearGradSq();
    }
  }
}

const std::unordered_map<std::string, HashTablePtr> &
SparseOptimizer::parameters() const noexcept {
  return param_groups_.at(0).params();
}

const std::unordered_map<std::string, HashTablePtr> &
SparseOptimizer::parameters() noexcept {
  return param_groups_.at(0).params();
}

size_t SparseOptimizer::size() const noexcept {
  size_t count = 0;
  for (const auto &group : param_groups_) {
    count += group.params().size();
  }
  return count;
}

SparseOptimizerOptions &SparseOptimizer::defaults() noexcept {
  return *defaults_.get();
}

const SparseOptimizerOptions &SparseOptimizer::defaults() const noexcept {
  return *defaults_.get();
}
std::vector<SparseOptimizerParamGroup> &
SparseOptimizer::param_groups() noexcept {
  return param_groups_;
}

const std::vector<SparseOptimizerParamGroup> &SparseOptimizer::param_groups()
    const noexcept {
  return param_groups_;
}

void SparseOptimizer::set_grad_accum_steps(const int64_t steps) noexcept {
  grad_accum_steps_ = steps;
}

void SparseOptimizer::set_lr(const double lr) noexcept {
  for (auto &group : param_groups_) {
    group.options().set_lr(lr);
  }
}

ska::flat_hash_map<void *, std::unique_ptr<SparseOptimizerParamState>> &
SparseOptimizer::state() noexcept {
  return state_;
}

const ska::flat_hash_map<void *, std::unique_ptr<SparseOptimizerParamState>> &
SparseOptimizer::state() const noexcept {
  return state_;
}

}  // namespace optim
}  // namespace recis
