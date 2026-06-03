#include "embedding/adagrad.h"

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
#include "c10/core/ScalarType.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
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
#include "ops/block_apply_adagrad_op.h"

namespace recis {
namespace optim {

void SparseAdagrad::init_param_state(
    HashTablePtr param, const SparseOptimizerOptions &sparse_options) {
  TORCH_CHECK(state_.count(param.get()) == 0,
              "some parameters appear in more than one parameter group.");
  const auto &options =
      static_cast<const SparseAdagradOptions &>(sparse_options);
  auto state = std::make_unique<SparseAdagradParamState>();
  auto emb_slot = param->SlotGroup()->EmbSlot();
  state->step(
      torch::zeros({1}, at::TensorOptions()
                            .dtype(state->step_dtype())
                            .device(emb_slot->TensorOptions().device())));
  state->param(emb_slot);
  state->state_sum(param->SlotGroup()->AppendSlot(
      StateSumName(), emb_slot->Dtype(), emb_slot->FullShape(1),
      options.initial_accumulator_value()));

  state->hashtable(param);
  state_[param.get()] = std::move(state);
}

void fused_sparse_adagrad(torch::Tensor grad, SparseAdagradOptions &options,
                          SparseAdagradParamState &state) {
  auto table = state.hashtable();
  auto param = state.param();
  grad = grad.to(param->TensorOptions().device());
  auto block_size = param->BlockSize();
  auto state_sum = state.state_sum();
  auto index = utils::get_sparse_impl(grad)->indices();
  auto grad_emb = utils::get_sparse_impl(grad)->values();
  recis::functional::block_apply_adagrad(
      index, grad_emb, (*param->Values()), state.step(), (*state_sum->Values()),
      options.lr(), options.lr_decay(), options.eps(), options.weight_decay(),
      block_size);
}

void SparseAdagrad::add_param_group(
    const SparseOptimizerParamGroup &param_group) {
  SparseOptimizerParamGroup param_group_(param_group.params());
  // set options for group
  if (!param_group_.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group_.options().clone());
  }
  //  init optimizer global state for hashtable name <-> hashtable ptr
  for (const auto &param : param_group_.params()) {
    init_param_state(param.second, param_group_.options());
  }
  // add param group
  param_groups_.emplace_back(param_group);
}

void SparseAdagrad::add_parameters(
    const torch::Dict<std::string, HashTablePtr> &parameters) {
  TORCH_CHECK(param_groups_.size() == 1,
              "add_parameters only support to add paramaters into group 0");
  auto &params = param_groups_[0].params();
  for (auto it = parameters.begin(); it != parameters.end(); it++) {
    init_param_state(it->value(), param_groups_[0].options());
    params[it->key()] = it->value();
  }
}

void SparseAdagrad::step() {
  utils::apply_sparse_step<SparseAdagradOptions, SparseAdagradParamState>(
      param_groups_, state_,
      [&](const std::string &name, HashTablePtr &p, const torch::Tensor &grad,
          SparseAdagradOptions &options, SparseAdagradParamState &state) {
        int64_t state_sum_size = state.state_sum()->Values()->size();
        int64_t param_size = state.param()->Values()->size();
        TORCH_CHECK(param_size == state_sum_size,
                    "param size and state_sum param size mismatch",
                    ", param_size: ", param_size,
                    ", state_sum size: ", state_sum_size);
        RECORD_FUNCTION(
            torch::str("fused_sparse_adagrad", "/", name, "/", "update"),
            std::vector<c10::IValue>());
        fused_sparse_adagrad(grad, options, state);
      });
}

double SparseAdagradOptions::get_lr() const { return lr_; }
void SparseAdagradOptions::set_lr(const double lr) { lr_ = lr; }

void SparseAdagrad::load_state_dict(
    torch::Dict<std::string, HashTablePtr> hashtables,
    torch::Dict<std::string, torch::Tensor> steps) {
  for (const auto &ht : hashtables) {
    auto ht_ptr = ht.value();
    auto &state = static_cast<SparseAdagradParamState &>(*state_[ht_ptr.get()]);
    for (const auto &step : steps) {
      TORCH_CHECK(step.key() == StepName(), "SparseAdagrad only have ",
                  StepName());
      state.step(step.value());
    }
  }
}

const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                 std::unordered_map<std::string, torch::Tensor>>
SparseAdagrad::state_dict() {
  std::unordered_map<std::string, HashTablePtr> ret;
  std::unordered_map<std::string, torch::Tensor> steps;
  for (auto &group : param_groups_) {
    for (auto &it : group.params()) {
      auto &p = it.second;
      if (!p.defined()) {
        continue;
      }
      const auto &state =
          static_cast<SparseAdagradParamState &>(*state_[p.get()]);
      steps[StepName()] = state.step();
    }
  }
  return std::make_tuple(ret, steps);
}

c10::intrusive_ptr<SparseAdagrad> SparseAdagrad::Make(
    const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
    double lr_decay, double initial_accumulator_value, double eps,
    double weight_decay) {
  LOG(WARNING) << "SparseAdagrad Make: " << at::get_parallel_info()
               << "; lr is " << lr << "; lr_decay is " << lr_decay
               << "; option.initial_accumulator_value is "
               << initial_accumulator_value << "; eps is " << eps
               << "; weight_decay is " << weight_decay;
  SparseAdagradOptions option;
  option.lr(lr);
  option.lr_decay(lr_decay);
  option.initial_accumulator_value(initial_accumulator_value);
  option.eps(eps);
  option.weight_decay(weight_decay);
  std::unordered_map<std::string, HashTablePtr> input;
  for (auto it = hashtables.begin(); it != hashtables.end(); it++) {
    input[it->key()] = it->value();
  }
  auto opt = c10::make_intrusive<SparseAdagrad>(input, option);
  TORCH_CHECK(opt->param_groups().size() >= 1,
              "opt->param_groups().size() must >= 1, but get ",
              opt->param_groups().size());
  return opt;
}

// test method for loading dense state of sparse optimizer
void SparseAdagrad::reset_state_dict() {
  for (auto &group : param_groups_) {
    for (auto &it : group.params()) {
      auto &p = it.second;
      if (!p.defined()) {
        continue;
      }
      LOG(WARNING) << "SparseAdagrad::reset step for " << it.first;
      auto &state = static_cast<SparseAdagradParamState &>(*state_[p.get()]);
      state.reset_step();
    }
  }
}

}  // namespace optim
}  // namespace recis
