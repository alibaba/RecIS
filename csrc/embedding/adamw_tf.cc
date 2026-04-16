#include "embedding/adamw_tf.h"

#include <ATen/ParallelNative.h>
#include <math.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/ParallelFuture.h"
#include "ATen/SparseTensorImpl.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/function.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/ivalue_inl.h"
#include "ATen/core/jit_type.h"
#include "ATen/ops/ones.h"
#include "ATen/ops/unique_consecutive.h"
#include "ATen/ops/zeros.h"
#include "ATen/record_function.h"
#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/irange.h"
#include "embedding/hashtable.h"
#include "embedding/initializer.h"
#include "embedding/optim.h"
#include "embedding/optim_util.h"
#include "embedding/parallel_util.h"
#include "ops/block_apply_adamw_op.h"
#include "torch/types.h"

namespace recis {
namespace optim {
namespace {

void FusedSparseAdamWTF(torch::Tensor grad, SparseAdamWTFOptions &options,
                        SparseAdamWTFParamState &state) {
  auto table = state.hashtable();
  auto param = state.param();
  grad = grad.to(param->TensorOptions().device());
  auto block_size = param->BlockSize();
  auto exp_avg = state.exp_avg();
  auto exp_avg_sq = state.exp_avg_sq();
  auto beta1 = std::get<0>(options.betas());
  auto beta2 = std::get<1>(options.betas());

  auto weight_decay = 1 - options.weight_decay();
  auto eps = options.eps();

  auto index = utils::get_sparse_impl(grad)->indices();
  auto grad_emb = utils::get_sparse_impl(grad)->values();

  recis::functional::block_apply_adamw(
      index, grad_emb, (*param->Values()), state.beta1(), state.beta2(),
      state.step(), (*exp_avg->Values()), (*exp_avg_sq->Values()), options.lr(),
      beta1, beta2, weight_decay, eps, block_size);
}

std::string kParamNameSep{"sparse_adamw_tf_"};
std::string ExpAvgName() {
  static std::string local_var = torch::str(kParamNameSep, "exp_avg");
  return local_var;
}
std::string ExpAvgSqName() {
  static std::string local_var = torch::str(kParamNameSep, "exp_avg_sq");
  return local_var;
}
const std::string StepName(const std::string &tensor_name) {
  return torch::str(kParamNameSep, "step");
}
const std::string BetaOneName(const std::string &tensor_name) {
  return torch::str(kParamNameSep, "beta1");
}
std::string BetaTwoName(const std::string &tensor_name) {
  return torch::str(kParamNameSep, "beta2");
}
}  // namespace

SparseAdamWTFOptions::SparseAdamWTFOptions(double lr) : lr_(lr) {}
double SparseAdamWTFOptions::get_lr() const { return lr_; }

void SparseAdamWTFOptions::set_lr(const double lr) { lr_ = lr; }

void SparseAdamWTF::step() {
  utils::apply_sparse_step<SparseAdamWTFOptions, SparseAdamWTFParamState>(
      param_groups_, state_,
      [&](const std::string &name, HashTablePtr &p, const torch::Tensor &grad,
          SparseAdamWTFOptions &options, SparseAdamWTFParamState &state) {
        int64_t param_size = state.param()->Values()->size();
        TORCH_CHECK(
            param_size == state.exp_avg()->Values()->size() &&
                param_size == state.exp_avg_sq()->Values()->size(),
            "param size and adamw param size mismatch",
            ", param_size: ", param_size,
            ", adamw exp_avg size: ", state.exp_avg()->Values()->size(),
            ", adamw exp_avg_sq size: ", state.exp_avg_sq()->Values()->size());
        RECORD_FUNCTION(
            torch::str("FusedSparseAdamWTF", "/", name, "/", "Update"),
            std::vector<c10::IValue>());
        FusedSparseAdamWTF(grad, options, state);
      });
}

void SparseAdamWTF::add_param_group(
    const SparseOptimizerParamGroup &param_group) {
  SparseOptimizerParamGroup param_group_(param_group.params());
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group.options().clone());
  }
  for (const auto &param : param_group_.params()) {
    InitParamState(param.first, param.second);
  }
  param_groups_.emplace_back(param_group_);
}

void SparseAdamWTF::add_parameters(
    const torch::Dict<std::string, HashTablePtr> &parameters) {
  auto &parameters_ = param_groups_[0].params();
  for (auto it = parameters.begin(); it != parameters.end(); it++) {
    InitParamState(it->key(), it->value());
    parameters_[it->key()] = it->value();
  }
}
void SparseAdamWTF::InitParamState(const std::string &param_name,
                                   HashTablePtr param) {
  TORCH_CHECK(state_.count(param.get()) == 0,
              "some parameters appear in more than one parameter group.");
  auto state = std::make_unique<SparseAdamWTFParamState>();
  auto slot_group = param->SlotGroup();
  auto emb_slot = slot_group->EmbSlot();
  // Exponential moving average of gradient values
  state->beta1(
      torch::ones({}, at::TensorOptions()
                          .dtype(state->beta_dtype())
                          .device(emb_slot->TensorOptions().device())));
  state->beta2(
      torch::ones({}, at::TensorOptions()
                          .dtype(state->beta_dtype())
                          .device(emb_slot->TensorOptions().device())));
  state->step(
      torch::zeros({1}, at::TensorOptions()
                            .dtype(state->step_dtype())
                            .device(emb_slot->TensorOptions().device())));

  state->param(emb_slot);
  state->exp_avg(slot_group->AppendSlot(ExpAvgName(), emb_slot->Dtype(),
                                        emb_slot->FullShape(1)));
  state->exp_avg_sq(slot_group->AppendSlot(ExpAvgSqName(), emb_slot->Dtype(),
                                           emb_slot->FullShape(1)));
  state->hashtable(param);
  state_[param.get()] = std::move(state);
}

const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                 std::unordered_map<std::string, torch::Tensor>>
SparseAdamWTF::state_dict() {
  std::unordered_map<std::string, HashTablePtr> ret;
  std::unordered_map<std::string, torch::Tensor> steps;
  for (auto &group : param_groups_) {
    for (auto &it : group.params()) {
      auto &p = it.second;
      if (!p.defined()) {
        continue;
      }
      auto &param_name = it.first;
      auto &state = static_cast<SparseAdamWTFParamState &>(*state_[p.get()]);
      steps[StepName(param_name)] = state.step();
      steps[BetaOneName(param_name)] = state.beta1();
      steps[BetaTwoName(param_name)] = state.beta2();
    }
  }
  return std::make_tuple(ret, steps);
}

void SparseAdamWTF::load_state_dict(
    torch::Dict<std::string, HashTablePtr> hashtables,
    torch::Dict<std::string, torch::Tensor> steps) {}

c10::intrusive_ptr<SparseAdamWTF> SparseAdamWTF::Make(
    const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
    double beta1, double beta2, double eps, double weight_decay,
    bool use_nesterov) {
  LOG(WARNING) << at::get_parallel_info();
  SparseAdamWTFOptions option;
  option.lr(lr);
  option.betas(std::make_pair(beta1, beta2));
  option.eps(eps);
  option.weight_decay(weight_decay);
  option.use_nesterov(use_nesterov);
  std::unordered_map<std::string, HashTablePtr> input;
  for (auto it = hashtables.begin(); it != hashtables.end(); it++) {
    input[it->key()] = it->value();
  }
  return c10::make_intrusive<SparseAdamWTF>(input, option);
}

void SparseAdamWTF::reset_state_dict() {}

}  // namespace optim
}  // namespace recis
