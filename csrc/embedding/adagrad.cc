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

namespace {

void fused_sparse_adagrad(torch::Tensor grad, SparseAdagradOptions &options,
                          SparseAdagradParamState &state,
                          bool save_update_info) {
  auto table = state.hashtable();
  auto param = state.param();
  grad = grad.to(param->TensorOptions().device());
  auto block_size = param->BlockSize();
  auto state_sum = state.state_sum();
  auto index = utils::get_sparse_impl(grad)->indices();
  auto grad_emb = utils::get_sparse_impl(grad)->values();

  // Store the indices (these are the ones that were updated)
  if (save_update_info) {
    state.set_updated_indices(index.clone());
  }

  // Prepare output tensors for state_sum, parameter and gradient values
  torch::Tensor output_state_sum_values;
  torch::Tensor output_params_before;
  torch::Tensor output_params_after;
  torch::Tensor output_grad;
  if (save_update_info && index.numel() > 0) {
    auto emb_values = param->Values();
    if (emb_values && !emb_values->empty()) {
      auto embedding_dim = emb_values->at(0).size(1);
      auto dtype = emb_values->at(0).scalar_type();
      output_state_sum_values =
          torch::empty({index.numel(), embedding_dim},
                       param->TensorOptions().dtype(
                           state_sum->Values()->at(0).scalar_type()));
      output_params_before = torch::empty({index.numel(), embedding_dim},
                                          param->TensorOptions().dtype(dtype));
      output_params_after = torch::empty({index.numel(), embedding_dim},
                                         param->TensorOptions().dtype(dtype));
      output_grad = torch::empty({index.numel(), embedding_dim},
                                 param->TensorOptions().dtype(dtype));
    }
  }

  // Apply the Adagrad update and capture all values in one pass
  recis::functional::block_apply_adagrad(
      index, grad_emb, (*param->Values()), state.step(), (*state_sum->Values()),
      options.lr(), options.lr_decay(), options.eps(), options.weight_decay(),
      block_size, output_state_sum_values, output_params_before,
      output_params_after, output_grad);

  // Store the captured values
  if (save_update_info) {
    state.set_updated_state_sum(output_state_sum_values);
    state.set_updated_params_before(output_params_before);
    state.set_updated_params_after(output_params_after);
    state.set_updated_grad(output_grad);
  }
}

// Fused sparse adagrad sum using pre-computed gradient sum of squares
void fused_sparse_adagrad_sum(torch::Tensor grad, torch::Tensor grad_sum_sq,
                              SparseAdagradOptions &options,
                              SparseAdagradParamState &state,
                              bool save_update_info) {
  auto table = state.hashtable();
  auto param = state.param();
  grad = grad.to(param->TensorOptions().device());
  grad_sum_sq = grad_sum_sq.to(param->TensorOptions().device());
  auto block_size = param->BlockSize();
  auto state_sum = state.state_sum();
  auto index = utils::get_sparse_impl(grad)->indices();
  auto grad_emb = utils::get_sparse_impl(grad)->values();
  auto grad_sum_sq_emb = utils::get_sparse_impl(grad_sum_sq)->values();

  // Store the indices (these are the ones that were updated)
  if (save_update_info) {
    state.set_updated_indices(index.clone());
  }

  // Prepare output tensors for state_sum, parameter and gradient values
  torch::Tensor output_state_sum_values;
  torch::Tensor output_params_before;
  torch::Tensor output_params_after;
  torch::Tensor output_grad;
  if (save_update_info && index.numel() > 0) {
    auto emb_values = param->Values();
    if (emb_values && !emb_values->empty()) {
      auto embedding_dim = emb_values->at(0).size(1);
      auto dtype = emb_values->at(0).scalar_type();
      output_state_sum_values =
          torch::empty({index.numel(), embedding_dim},
                       param->TensorOptions().dtype(
                           state_sum->Values()->at(0).scalar_type()));
      output_params_before = torch::empty({index.numel(), embedding_dim},
                                          param->TensorOptions().dtype(dtype));
      output_params_after = torch::empty({index.numel(), embedding_dim},
                                         param->TensorOptions().dtype(dtype));
      output_grad = torch::empty({index.numel(), embedding_dim},
                                 param->TensorOptions().dtype(dtype));
    }
  }

  // Apply the Adagrad Sum update using pre-computed gradient sum of squares
  recis::functional::block_apply_adagrad_sum(
      index, grad_emb, grad_sum_sq_emb, (*param->Values()), state.step(),
      (*state_sum->Values()), options.lr(), options.lr_decay(), options.eps(),
      block_size, output_state_sum_values, output_params_before,
      output_params_after, output_grad);

  // Store the captured values
  if (save_update_info) {
    state.set_updated_state_sum(output_state_sum_values);
    state.set_updated_params_before(output_params_before);
    state.set_updated_params_after(output_params_after);
    state.set_updated_grad(output_grad);
  }
}

bool has_complete_step_update_info(const SparseAdagradParamState &state) {
  return state.updated_indices().defined() &&
         state.updated_state_sum().defined() &&
         state.updated_params_before().defined() &&
         state.updated_params_after().defined() &&
         state.updated_grad().defined();
}

void insert_step_update_info(
    torch::Dict<std::string, torch::Tensor> &update_info,
    const std::string &name, const SparseAdagradParamState &state) {
  if (!has_complete_step_update_info(state)) {
    return;
  }
  update_info.insert(name + "_updated_indices", state.updated_indices());
  update_info.insert(name + "_updated_state_sum", state.updated_state_sum());
  update_info.insert(name + "_updated_params_before",
                     state.updated_params_before());
  update_info.insert(name + "_updated_params_after",
                     state.updated_params_after());
  update_info.insert(name + "_updated_grad", state.updated_grad());
  update_info.insert(name + "_last_saved_step",
                     torch::tensor(state.last_saved_step_));
}

}  // namespace

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
  state->last_saved_step_ = -1;  // Initialize last saved step
  state_[param.get()] = std::move(state);
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

        // Get current step
        int64_t current_step = state.step().item<int64_t>();

        // Determine if we should save update info for this step
        // 0 means never save, positive number means save every N steps
        bool save_update_info =
            (save_update_info_interval_ > 0 &&
             current_step % save_update_info_interval_ == 0);

        // If we're saving update info, record the step number
        if (save_update_info) {
          state.last_saved_step_ = current_step;
        }

        RECORD_FUNCTION(
            torch::str("fused_sparse_adagrad", "/", name, "/", "update"),
            std::vector<c10::IValue>());
        fused_sparse_adagrad(grad, options, state, save_update_info);
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
    double weight_decay, int64_t save_update_info_interval) {
  LOG(WARNING) << "SparseAdagrad Make: " << at::get_parallel_info()
               << "; lr is " << lr << "; lr_decay is " << lr_decay
               << "; option.initial_accumulator_value is "
               << initial_accumulator_value << "; eps is " << eps
               << "; weight_decay is " << weight_decay
               << "; save_update_info_interval is " << save_update_info_interval
               << " (0 means never save)";
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
  auto opt = c10::make_intrusive<SparseAdagrad>(input, option,
                                                save_update_info_interval);
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

torch::Dict<std::string, torch::Tensor> SparseAdagrad::get_step_update_info() {
  torch::Dict<std::string, torch::Tensor> update_info;
  for (auto &group : param_groups_) {
    for (auto &it : group.params()) {
      auto &p = it.second;
      if (!p.defined()) {
        continue;
      }
      const auto &state =
          static_cast<SparseAdagradParamState &>(*state_[p.get()]);
      insert_step_update_info(update_info, it.first, state);
    }
  }
  return update_info;
}

void SparseAdagrad::clear_step_update_info() {
  for (const auto &group : param_groups_) {
    for (const auto &it : group.params()) {
      auto &p = it.second;
      if (!p.defined()) {
        continue;
      }
      auto &state = static_cast<SparseAdagradParamState &>(*state_[p.get()]);
      state.clear_update_info();
    }
  }
}

void SparseAdagrad::set_save_update_info_interval(int64_t interval) {
  TORCH_CHECK(interval >= 0,
              "save_update_info_interval must be >= 0 (0 means never save)");
  save_update_info_interval_ = interval;
}

// ==================== SparseAdagradSum Implementation ====================

void SparseAdagradSum::init_param_state(
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
  state->last_saved_step_ = -1;
  state_[param.get()] = std::move(state);
}

void SparseAdagradSum::add_param_group(
    const SparseOptimizerParamGroup &param_group) {
  SparseOptimizerParamGroup param_group_(param_group.params());
  if (!param_group_.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group_.options().clone());
  }
  for (const auto &param : param_group_.params()) {
    init_param_state(param.second, param_group_.options());
  }
  param_groups_.emplace_back(param_group);
}

void SparseAdagradSum::add_parameters(
    const torch::Dict<std::string, HashTablePtr> &parameters) {
  TORCH_CHECK(param_groups_.size() == 1,
              "add_parameters only support to add paramaters into group 0");
  auto &params = param_groups_[0].params();
  for (auto it = parameters.begin(); it != parameters.end(); it++) {
    init_param_state(it->value(), param_groups_[0].options());
    params[it->key()] = it->value();
  }
}

void SparseAdagradSum::step() {
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

        // Get gradient sum of squares from hashtable
        torch::Tensor grad_sum_sq;
        if (p->HasGradSq()) {
          grad_sum_sq = p->GradSq(grad_accum_steps_);
        } else {
          // Fallback to computing grad^2 if grad_sq is not available
          auto sparse_grad = utils::get_sparse_impl(grad);
          auto grad_values = sparse_grad->values();
          grad_sum_sq = torch::sparse_coo_tensor(sparse_grad->indices(),
                                                 grad_values * grad_values,
                                                 grad_values.sizes());
        }

        // Get current step
        int64_t current_step = state.step().item<int64_t>();

        // Determine if we should save update info for this step
        bool save_update_info =
            (save_update_info_interval_ > 0 &&
             current_step % save_update_info_interval_ == 0);

        // If we're saving update info, record the step number
        if (save_update_info) {
          state.last_saved_step_ = current_step;
        }

        RECORD_FUNCTION(
            torch::str("fused_sparse_adagrad_sum", "/", name, "/", "update"),
            std::vector<c10::IValue>());
        fused_sparse_adagrad_sum(grad, grad_sum_sq, options, state,
                                 save_update_info);
      });
}

void SparseAdagradSum::load_state_dict(
    torch::Dict<std::string, HashTablePtr> hashtables,
    torch::Dict<std::string, torch::Tensor> steps) {
  for (const auto &ht : hashtables) {
    auto ht_ptr = ht.value();
    auto &state = static_cast<SparseAdagradParamState &>(*state_[ht_ptr.get()]);
    for (const auto &step : steps) {
      TORCH_CHECK(step.key() == StepName(), "SparseAdagradSum only have ",
                  StepName());
      state.step(step.value());
    }
  }
}

const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                 std::unordered_map<std::string, torch::Tensor>>
SparseAdagradSum::state_dict() {
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

c10::intrusive_ptr<SparseAdagradSum> SparseAdagradSum::Make(
    const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
    double lr_decay, double initial_accumulator_value, double eps,
    int64_t save_update_info_interval) {
  LOG(WARNING) << "SparseAdagradSum Make: " << at::get_parallel_info()
               << "; lr is " << lr << "; lr_decay is " << lr_decay
               << "; option.initial_accumulator_value is "
               << initial_accumulator_value << "; eps is " << eps
               << "; save_update_info_interval is " << save_update_info_interval
               << " (0 means never save)";
  SparseAdagradOptions option;
  option.lr(lr);
  option.lr_decay(lr_decay);
  option.initial_accumulator_value(initial_accumulator_value);
  option.eps(eps);
  std::unordered_map<std::string, HashTablePtr> input;
  for (auto it = hashtables.begin(); it != hashtables.end(); it++) {
    input[it->key()] = it->value();
  }
  auto opt = c10::make_intrusive<SparseAdagradSum>(input, option,
                                                   save_update_info_interval);
  TORCH_CHECK(opt->param_groups().size() >= 1,
              "opt->param_groups().size() must >= 1, but get ",
              opt->param_groups().size());
  return opt;
}

void SparseAdagradSum::reset_state_dict() {
  for (auto &group : param_groups_) {
    for (auto &it : group.params()) {
      auto &p = it.second;
      if (!p.defined()) {
        continue;
      }
      LOG(WARNING) << "SparseAdagradSum::reset step for " << it.first;
      auto &state = static_cast<SparseAdagradParamState &>(*state_[p.get()]);
      state.reset_step();
    }
  }
}

torch::Dict<std::string, torch::Tensor>
SparseAdagradSum::get_step_update_info() {
  torch::Dict<std::string, torch::Tensor> update_info;
  for (auto &group : param_groups_) {
    for (auto &it : group.params()) {
      auto &p = it.second;
      if (!p.defined()) {
        continue;
      }
      const auto &state =
          static_cast<SparseAdagradParamState &>(*state_[p.get()]);
      insert_step_update_info(update_info, it.first, state);
    }
  }
  return update_info;
}

void SparseAdagradSum::clear_step_update_info() {
  for (const auto &group : param_groups_) {
    for (const auto &it : group.params()) {
      auto &p = it.second;
      if (!p.defined()) {
        continue;
      }
      auto &state = static_cast<SparseAdagradParamState &>(*state_[p.get()]);
      state.clear_update_info();
    }
  }
}

void SparseAdagradSum::set_save_update_info_interval(int64_t interval) {
  TORCH_CHECK(interval >= 0,
              "save_update_info_interval must be >= 0 (0 means never save)");
  save_update_info_interval_ = interval;
}

}  // namespace optim
}  // namespace recis
