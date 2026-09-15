#include "embedding/row_wise_adagrad.h"

#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ATen/Parallel.h"
#include "ATen/SparseTensorImpl.h"
#include "ATen/record_function.h"
#include "embedding/hashtable.h"
#include "embedding/initializer.h"
#include "embedding/optim_util.h"
#include "ops/block_apply_row_wise_adagrad_op.h"

namespace recis {
namespace optim {

namespace {

void ResetUnusedStateRows(
    const HashTablePtr &param,
    const SparseRowWiseAdagradParamState::ParamContainer &state_sum) {
  const auto block_size = param->SlotGroup()->BlockSize();
  const auto state_capacity =
      static_cast<int64_t>(state_sum->Values()->size()) * block_size;
  const auto used_rows = param->IdsNum();
  TORCH_CHECK(used_rows <= state_capacity, "HashTable has ", used_rows,
              " allocated rows, but row-wise state has capacity ",
              state_capacity);
  if (used_rows == state_capacity) {
    return;
  }

  const auto indices =
      torch::arange(used_rows, state_capacity,
                    torch::TensorOptions()
                        .dtype(torch::kInt64)
                        .device(state_sum->TensorOptions().device()));
  state_sum->IndexInsert(indices,
                         state_sum->GenVal(state_capacity - used_rows));
}

}  // namespace

void SparseRowWiseAdagrad::init_param_state(
    HashTablePtr param, const SparseOptimizerOptions &sparse_options) {
  TORCH_CHECK(state_.count(param.get()) == 0,
              "some parameters appear in more than one parameter group.");
  const auto &options =
      static_cast<const SparseRowWiseAdagradOptions &>(sparse_options);
  ValidateOptions(options);
  auto state = std::make_unique<SparseRowWiseAdagradParamState>();
  auto slot_group = param->SlotGroup();
  auto emb_slot = slot_group->EmbSlot();
  // The step is host-only scalar state. Keeping it on CPU avoids synchronizing
  // a CUDA scalar before every update when calculating the decayed learning
  // rate.
  state->step(
      torch::zeros({1}, at::TensorOptions().dtype(state->step_dtype())));
  state->param(emb_slot);

  SparseRowWiseAdagradParamState::ParamContainer state_sum;
  for (const auto &slot : slot_group->Slots()) {
    if (slot->Name() == StateSumName()) {
      state_sum = slot;
      break;
    }
  }
  const bool reused_state_sum = state_sum != nullptr;
  if (!state_sum) {
    state_sum =
        slot_group->AppendSlot(StateSumName(), emb_slot->Dtype(), {1, 1},
                               options.initial_accumulator_value());
  }
  TORCH_CHECK(state_sum->Dtype() == emb_slot->Dtype(), StateSumName(),
              " dtype must match the embedding dtype");
  TORCH_CHECK(state_sum->FlatSize() == 1, StateSumName(),
              " must contain exactly one accumulator per row");

  // AppendSlot only participates in future SlotGroup::IncrementBlock calls.
  // Materialize accumulator blocks that predate optimizer construction.
  const auto param_block_count = emb_slot->Values()->size();
  auto state_block_count = state_sum->Values()->size();
  TORCH_CHECK(state_block_count <= param_block_count, StateSumName(), " has ",
              state_block_count, " blocks, but the embedding has only ",
              param_block_count);
  if (reused_state_sum) {
    auto generator = embedding::MakeConstantGenerator(
        state_sum->FullShape(slot_group->BlockSize()), emb_slot->Dtype(),
        options.initial_accumulator_value());
    generator->set_device(emb_slot->TensorOptions().device());
    state_sum->Generator(std::move(generator));
  }
  while (state_block_count < param_block_count) {
    state_sum->IncrementBlock();
    ++state_block_count;
  }
  if (reused_state_sum) {
    // HashTable keeps spare capacity beyond IdsNum(). Refresh only those
    // unassigned rows so future IDs use this optimizer's configured initial
    // accumulator without disturbing live optimizer state.
    ResetUnusedStateRows(param, state_sum);
  }
  state->state_sum(std::move(state_sum));
  state->hashtable(param);
  state_[param.get()] = std::move(state);
}

void fused_sparse_row_wise_adagrad(torch::Tensor grad,
                                   SparseRowWiseAdagradOptions &options,
                                   SparseRowWiseAdagradParamState &state) {
  auto param = state.param();
  grad = grad.to(param->TensorOptions().device());
  auto index = utils::get_sparse_impl(grad)->indices();
  auto grad_emb = utils::get_sparse_impl(grad)->values();
  if (options.maximize()) {
    grad_emb = -grad_emb;
  }
  recis::functional::block_apply_row_wise_adagrad(
      index, grad_emb, *param->Values(), state.step(),
      *state.state_sum()->Values(), options.lr(), options.lr_decay(),
      options.eps(), options.weight_decay(), param->BlockSize());
}

void SparseRowWiseAdagrad::add_param_group(
    const SparseOptimizerParamGroup &param_group) {
  SparseOptimizerParamGroup new_group(param_group.params());
  if (!param_group.has_options()) {
    new_group.set_options(defaults_->clone());
  } else {
    new_group.set_options(param_group.options().clone());
  }
  ValidateOptions(
      static_cast<const SparseRowWiseAdagradOptions &>(new_group.options()));
  std::unordered_set<std::string> param_names;
  std::unordered_set<void *> new_params;
  for (const auto &group : param_groups_) {
    for (const auto &param : group.params()) {
      param_names.insert(param.first);
    }
  }
  for (const auto &param : new_group.params()) {
    TORCH_CHECK(
        param.second.defined(),
        "Cannot optimize an undefined HashTable parameter: ", param.first);
    TORCH_CHECK(
        param_names.insert(param.first).second,
        "SparseRowWiseAdagrad parameter names must be unique: ", param.first);
    TORCH_CHECK(state_.count(param.second.get()) == 0 &&
                    new_params.insert(param.second.get()).second,
                "some parameters appear in more than one parameter group.");
  }
  for (const auto &param : new_group.params()) {
    init_param_state(param.second, new_group.options());
  }
  param_groups_.emplace_back(std::move(new_group));
}

void SparseRowWiseAdagrad::add_parameters(
    const torch::Dict<std::string, HashTablePtr> &parameters) {
  TORCH_CHECK(param_groups_.size() == 1,
              "add_parameters only supports adding parameters to group 0");
  auto &params = param_groups_[0].params();

  std::unordered_set<void *> new_params;
  for (auto it = parameters.begin(); it != parameters.end(); ++it) {
    TORCH_CHECK(params.count(it->key()) == 0,
                "SparseRowWiseAdagrad already has a parameter named '",
                it->key(), "'");
    TORCH_CHECK(
        it->value().defined(),
        "Cannot add an undefined SparseRowWiseAdagrad parameter: ", it->key());
    TORCH_CHECK(state_.count(it->value().get()) == 0 &&
                    new_params.insert(it->value().get()).second,
                "some parameters appear in more than one parameter group.");
  }
  for (auto it = parameters.begin(); it != parameters.end(); ++it) {
    init_param_state(it->value(), param_groups_[0].options());
    params[it->key()] = it->value();
  }
}

void SparseRowWiseAdagrad::step() {
  utils::apply_sparse_step<SparseRowWiseAdagradOptions,
                           SparseRowWiseAdagradParamState>(
      param_groups_, state_,
      [&](const std::string &name, HashTablePtr &, const torch::Tensor &grad,
          SparseRowWiseAdagradOptions &options,
          SparseRowWiseAdagradParamState &state) {
        TORCH_CHECK(
            state.param()->Values()->size() ==
                state.state_sum()->Values()->size(),
            "param block count and row-wise state_sum block count mismatch");
        RECORD_FUNCTION(
            torch::str("fused_sparse_row_wise_adagrad", "/", name, "/update"),
            std::vector<c10::IValue>());
        fused_sparse_row_wise_adagrad(grad, options, state);
      });
}

double SparseRowWiseAdagradOptions::get_lr() const { return lr_; }
void SparseRowWiseAdagradOptions::set_lr(double lr) { lr_ = lr; }

void SparseRowWiseAdagrad::ValidateOptions(
    const SparseRowWiseAdagradOptions &options) {
  TORCH_CHECK(options.lr() >= 0, "Invalid learning rate: ", options.lr());
  TORCH_CHECK(options.lr_decay() >= 0,
              "Invalid learning rate decay: ", options.lr_decay());
  TORCH_CHECK(options.initial_accumulator_value() >= 0,
              "Invalid initial accumulator value: ",
              options.initial_accumulator_value());
  TORCH_CHECK(options.eps() >= 0, "Invalid epsilon value: ", options.eps());
  TORCH_CHECK(options.weight_decay() >= 0,
              "Invalid weight decay: ", options.weight_decay());
}

void SparseRowWiseAdagrad::load_state_dict(
    torch::Dict<std::string, HashTablePtr> hashtables,
    torch::Dict<std::string, torch::Tensor> steps) {
  std::unordered_map<std::string, HashTablePtr> current_params;
  for (const auto &group : param_groups_) {
    for (const auto &item : group.params()) {
      if (!item.second.defined()) {
        continue;
      }
      const auto inserted = current_params.emplace(item.first, item.second);
      TORCH_CHECK(
          inserted.second,
          "SparseRowWiseAdagrad parameter names must be unique: ", item.first);
    }
  }

  // Hashtable checkpoint loading restores the accumulator slot in place. The
  // objects passed here identify those tables by name; they are deliberately
  // never used as keys into state_, because they may belong to another
  // optimizer instance.
  if (!hashtables.empty()) {
    TORCH_CHECK(hashtables.size() == current_params.size(),
                "SparseRowWiseAdagrad state has ", hashtables.size(),
                " HashTables, but the optimizer has ", current_params.size());
    for (const auto &item : hashtables) {
      const auto current_param = current_params.find(item.key());
      TORCH_CHECK(
          current_param != current_params.end(),
          "Unexpected SparseRowWiseAdagrad HashTable state: ", item.key());
      TORCH_CHECK(
          item.value().defined(),
          "Undefined SparseRowWiseAdagrad HashTable state: ", item.key());
      TORCH_CHECK(
          item.value().get() == current_param->second.get(),
          "SparseRowWiseAdagrad.load_state_dict cannot copy row accumulators ",
          "between different HashTable objects for parameter '", item.key(),
          "'; restore HashTable state with RecIS Saver/Loader first, then "
          "load ",
          "the optimizer tensor state without foreign HashTable objects");
    }
  }

  std::unordered_map<std::string, torch::Tensor> loaded_steps;
  for (const auto &item : steps) {
    loaded_steps.emplace(item.key(), item.value());
  }
  const bool single_param = current_params.size() == 1;
  if (!single_param) {
    TORCH_CHECK(loaded_steps.count(StepName()) == 0,
                "The legacy SparseRowWiseAdagrad step is ambiguous for ",
                current_params.size(),
                " parameters; load a checkpoint with per-parameter steps");
  } else {
    const auto &param_name = current_params.begin()->first;
    TORCH_CHECK(
        !(loaded_steps.count(StepName()) != 0 &&
          loaded_steps.count(StepName(param_name)) != 0),
        "SparseRowWiseAdagrad state contains both legacy and per-parameter ",
        "steps for '", param_name, "'");
  }

  for (const auto &item : loaded_steps) {
    bool expected = single_param && item.first == StepName();
    for (const auto &param : current_params) {
      expected = expected || item.first == StepName(param.first);
    }
    TORCH_CHECK(expected,
                "Unexpected SparseRowWiseAdagrad tensor state: ", item.first);
  }

  std::vector<std::pair<torch::Tensor, torch::Tensor>> step_copies;
  step_copies.reserve(current_params.size());
  for (const auto &item : current_params) {
    const auto keyed_step_name = StepName(item.first);
    auto loaded_step = loaded_steps.find(keyed_step_name);
    if (loaded_step == loaded_steps.end() && single_param) {
      loaded_step = loaded_steps.find(StepName());
    }
    TORCH_CHECK(loaded_step != loaded_steps.end(),
                "Missing SparseRowWiseAdagrad step for parameter '", item.first,
                "' (expected '", keyed_step_name,
                single_param ? torch::str("' or legacy '", StepName(), "'")
                             : std::string("'"),
                ")");
    TORCH_CHECK(
        loaded_step->second.defined() && loaded_step->second.numel() == 1,
        "SparseRowWiseAdagrad step for parameter '", item.first,
        "' must be a defined one-element tensor");
    const auto step_type = loaded_step->second.scalar_type();
    TORCH_CHECK(!loaded_step->second.is_sparse() &&
                    (c10::isFloatingType(step_type) ||
                     c10::isIntegralType(step_type, false)),
                "SparseRowWiseAdagrad step for parameter '", item.first,
                "' must be a dense real numeric tensor");

    if (c10::isFloatingType(step_type)) {
      const auto step_value = loaded_step->second.item<double>();
      TORCH_CHECK(std::isfinite(step_value) && step_value >= 0 &&
                      std::floor(step_value) == step_value &&
                      step_value < static_cast<double>(
                                       std::numeric_limits<int64_t>::max()),
                  "SparseRowWiseAdagrad step for parameter '", item.first,
                  "' must be a non-negative integer representable as int64");
    } else {
      TORCH_CHECK(loaded_step->second.item<int64_t>() >= 0,
                  "SparseRowWiseAdagrad step for parameter '", item.first,
                  "' must be non-negative");
    }

    const auto state_it = state_.find(item.second.get());
    TORCH_CHECK(state_it != state_.end(),
                "Missing SparseRowWiseAdagrad state for parameter '",
                item.first, "'");
    auto &state =
        static_cast<SparseRowWiseAdagradParamState &>(*state_it->second);
    auto converted_step = loaded_step->second.to(state.step().options());
    step_copies.emplace_back(state.step(), std::move(converted_step));
  }

  for (const auto &step_copy : step_copies) {
    step_copy.first.copy_(step_copy.second);
  }
}

const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                 std::unordered_map<std::string, torch::Tensor>>
SparseRowWiseAdagrad::state_dict() {
  std::unordered_map<std::string, HashTablePtr> hashtables;
  std::unordered_map<std::string, torch::Tensor> steps;
  for (auto &group : param_groups_) {
    for (auto &item : group.params()) {
      auto &param = item.second;
      if (!param.defined()) {
        continue;
      }
      const auto inserted = hashtables.emplace(item.first, param);
      TORCH_CHECK(
          inserted.second,
          "SparseRowWiseAdagrad parameter names must be unique: ", item.first);
    }
  }

  const bool single_param = hashtables.size() == 1;
  for (const auto &item : hashtables) {
    const auto state_it = state_.find(item.second.get());
    TORCH_CHECK(state_it != state_.end(),
                "Missing SparseRowWiseAdagrad state for parameter '",
                item.first, "'");
    const auto &state =
        static_cast<SparseRowWiseAdagradParamState &>(*state_it->second);
    const auto step_name = single_param ? StepName() : StepName(item.first);
    TORCH_CHECK(hashtables.count(step_name) == 0,
                "SparseRowWiseAdagrad state key conflicts with parameter '",
                step_name, "'");
    const auto inserted = steps.emplace(step_name, state.step());
    TORCH_CHECK(inserted.second,
                "Duplicate SparseRowWiseAdagrad state key: ", step_name);
  }

  // Each returned HashTable owns the row accumulator slot. Step tensors stay
  // live so CheckpointManager can restore them in place after capturing or
  // refreshing this dictionary.
  for (const auto &item : steps) {
    TORCH_CHECK(hashtables.count(item.first) == 0,
                "SparseRowWiseAdagrad state key conflicts with parameter '",
                item.first, "'");
  }
  return std::make_tuple(hashtables, steps);
}

c10::intrusive_ptr<SparseRowWiseAdagrad> SparseRowWiseAdagrad::Make(
    const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
    double lr_decay, double initial_accumulator_value, double eps,
    double weight_decay, bool maximize) {
  LOG(WARNING) << "SparseRowWiseAdagrad Make: " << at::get_parallel_info()
               << "; lr is " << lr << "; lr_decay is " << lr_decay
               << "; initial_accumulator_value is " << initial_accumulator_value
               << "; eps is " << eps << "; weight_decay is " << weight_decay
               << "; maximize is " << maximize;
  SparseRowWiseAdagradOptions options;
  options.lr(lr);
  options.lr_decay(lr_decay);
  options.initial_accumulator_value(initial_accumulator_value);
  options.eps(eps);
  options.weight_decay(weight_decay);
  options.maximize(maximize);
  std::unordered_map<std::string, HashTablePtr> params;
  for (auto it = hashtables.begin(); it != hashtables.end(); ++it) {
    params[it->key()] = it->value();
  }
  auto optimizer = c10::make_intrusive<SparseRowWiseAdagrad>(params, options);
  TORCH_CHECK(optimizer->size() > 0,
              "SparseRowWiseAdagrad requires at least one parameter");
  return optimizer;
}

void SparseRowWiseAdagrad::reset_state_dict() {
  for (auto &group : param_groups_) {
    for (auto &item : group.params()) {
      auto &param = item.second;
      if (!param.defined()) {
        continue;
      }
      auto &state =
          static_cast<SparseRowWiseAdagradParamState &>(*state_[param.get()]);
      state.reset_step();
    }
  }
}

void SparseRowWiseAdagradSum::init_param_state(
    HashTablePtr param, const SparseOptimizerOptions &sparse_options) {
  TORCH_CHECK(state_.count(param.get()) == 0,
              "some parameters appear in more than one parameter group.");
  const auto &options =
      static_cast<const SparseRowWiseAdagradOptions &>(sparse_options);
  ValidateOptions(options);
  auto state = std::make_unique<SparseRowWiseAdagradParamState>();
  auto slot_group = param->SlotGroup();
  auto emb_slot = slot_group->EmbSlot();
  state->step(
      torch::zeros({1}, at::TensorOptions().dtype(state->step_dtype())));
  state->param(emb_slot);

  SparseRowWiseAdagradParamState::ParamContainer state_sum;
  for (const auto &slot : slot_group->Slots()) {
    if (slot->Name() == StateSumName()) {
      state_sum = slot;
      break;
    }
  }
  const bool reused_state_sum = state_sum != nullptr;
  if (!state_sum) {
    state_sum =
        slot_group->AppendSlot(StateSumName(), emb_slot->Dtype(), {1, 1},
                               options.initial_accumulator_value());
  }
  TORCH_CHECK(state_sum->Dtype() == emb_slot->Dtype(), StateSumName(),
              " dtype must match the embedding dtype");
  TORCH_CHECK(state_sum->FlatSize() == 1, StateSumName(),
              " must contain exactly one accumulator per row");

  const auto param_block_count = emb_slot->Values()->size();
  auto state_block_count = state_sum->Values()->size();
  TORCH_CHECK(state_block_count <= param_block_count, StateSumName(), " has ",
              state_block_count, " blocks, but the embedding has only ",
              param_block_count);
  if (reused_state_sum) {
    auto generator = embedding::MakeConstantGenerator(
        state_sum->FullShape(slot_group->BlockSize()), emb_slot->Dtype(),
        options.initial_accumulator_value());
    generator->set_device(emb_slot->TensorOptions().device());
    state_sum->Generator(std::move(generator));
  }
  while (state_block_count < param_block_count) {
    state_sum->IncrementBlock();
    ++state_block_count;
  }
  if (reused_state_sum) {
    ResetUnusedStateRows(param, state_sum);
  }
  state->state_sum(std::move(state_sum));
  state->hashtable(param);
  state_[param.get()] = std::move(state);
}

void fused_sparse_row_wise_adagrad_sum(torch::Tensor grad,
                                       torch::Tensor grad_sum_sq,
                                       SparseRowWiseAdagradOptions &options,
                                       SparseRowWiseAdagradParamState &state) {
  auto param = state.param();
  grad = grad.to(param->TensorOptions().device());
  grad_sum_sq = grad_sum_sq.to(param->TensorOptions().device());
  auto sparse_grad = utils::get_sparse_impl(grad);
  auto sparse_grad_sq = utils::get_sparse_impl(grad_sum_sq);
  auto index = sparse_grad->indices();
  auto grad_sq_index = sparse_grad_sq->indices();
  TORCH_CHECK(index.sizes() == grad_sq_index.sizes(),
              "SparseRowWiseAdagradSum grad and grad_sq indices shape "
              "mismatch");
  TORCH_CHECK(index.equal(grad_sq_index),
              "SparseRowWiseAdagradSum requires grad and grad_sq indices to "
              "match exactly");
  auto grad_emb = sparse_grad->values();
  auto grad_sum_sq_emb = sparse_grad_sq->values();
  TORCH_CHECK(grad_emb.sizes() == grad_sum_sq_emb.sizes(),
              "SparseRowWiseAdagradSum grad and grad_sq values shape "
              "mismatch");
  TORCH_CHECK(grad_sum_sq_emb.scalar_type() == grad_emb.scalar_type(),
              "SparseRowWiseAdagradSum grad_sq dtype must match grad dtype");
  if (options.maximize()) {
    grad_emb = -grad_emb;
  }
  recis::functional::block_apply_row_wise_adagrad_sum(
      index, grad_emb, grad_sum_sq_emb, *param->Values(), state.step(),
      *state.state_sum()->Values(), options.lr(), options.lr_decay(),
      options.eps(), param->BlockSize());
}

void SparseRowWiseAdagradSum::add_param_group(
    const SparseOptimizerParamGroup &param_group) {
  SparseOptimizerParamGroup new_group(param_group.params());
  if (!param_group.has_options()) {
    new_group.set_options(defaults_->clone());
  } else {
    new_group.set_options(param_group.options().clone());
  }
  ValidateOptions(
      static_cast<const SparseRowWiseAdagradOptions &>(new_group.options()));
  std::unordered_set<std::string> param_names;
  std::unordered_set<void *> new_params;
  for (const auto &group : param_groups_) {
    for (const auto &param : group.params()) {
      param_names.insert(param.first);
    }
  }
  for (const auto &param : new_group.params()) {
    TORCH_CHECK(
        param.second.defined(),
        "Cannot optimize an undefined HashTable parameter: ", param.first);
    TORCH_CHECK(param_names.insert(param.first).second,
                "SparseRowWiseAdagradSum parameter names must be unique: ",
                param.first);
    TORCH_CHECK(state_.count(param.second.get()) == 0 &&
                    new_params.insert(param.second.get()).second,
                "some parameters appear in more than one parameter group.");
  }
  for (const auto &param : new_group.params()) {
    init_param_state(param.second, new_group.options());
  }
  param_groups_.emplace_back(std::move(new_group));
}

void SparseRowWiseAdagradSum::add_parameters(
    const torch::Dict<std::string, HashTablePtr> &parameters) {
  TORCH_CHECK(param_groups_.size() == 1,
              "add_parameters only supports adding parameters to group 0");
  auto &params = param_groups_[0].params();

  std::unordered_set<void *> new_params;
  for (auto it = parameters.begin(); it != parameters.end(); ++it) {
    TORCH_CHECK(params.count(it->key()) == 0,
                "SparseRowWiseAdagradSum already has a parameter named '",
                it->key(), "'");
    TORCH_CHECK(it->value().defined(),
                "Cannot add an undefined SparseRowWiseAdagradSum parameter: ",
                it->key());
    TORCH_CHECK(state_.count(it->value().get()) == 0 &&
                    new_params.insert(it->value().get()).second,
                "some parameters appear in more than one parameter group.");
  }
  for (auto it = parameters.begin(); it != parameters.end(); ++it) {
    init_param_state(it->value(), param_groups_[0].options());
    params[it->key()] = it->value();
  }
}

void SparseRowWiseAdagradSum::step() {
  torch::NoGradGuard no_grad;
  for (auto &group : param_groups_) {
    auto &options = static_cast<SparseRowWiseAdagradOptions &>(group.options());
    for (auto &item : group.params()) {
      const auto &name = item.first;
      auto &p = item.second;
      if (!p.defined() || !p->HasGrad()) {
        continue;
      }
      TORCH_CHECK(
          p->HasGradSq(),
          "SparseRowWiseAdagradSum requires grad_sq for parameter '", name,
          "'. Use grad_reduce_by='hdmp_group_sum' so HashTable backward calls "
          "accept_grad_sq.");
      const auto &grad = p->Grad();
      if (!grad.defined()) {
        continue;
      }
      TORCH_CHECK(grad.is_sparse(),
                  "SparseRowWiseAdagradSum only supports sparse gradients. "
                  "Got dense gradient for param '",
                  name, "'.");
      auto grad_sum_sq = p->GradSq(grad_accum_steps_);
      TORCH_CHECK(grad_sum_sq.defined() && grad_sum_sq.is_sparse(),
                  "SparseRowWiseAdagradSum requires sparse grad_sq for param '",
                  name, "'.");
      auto state_it = state_.find(p.get());
      TORCH_CHECK(state_it != state_.end(),
                  "Optimizer state not found for SparseRowWiseAdagradSum "
                  "parameter '",
                  name, "'");
      auto &state =
          static_cast<SparseRowWiseAdagradParamState &>(*state_it->second);
      TORCH_CHECK(
          state.param()->Values()->size() ==
              state.state_sum()->Values()->size(),
          "param block count and row-wise state_sum block count mismatch");
      RECORD_FUNCTION(
          torch::str("fused_sparse_row_wise_adagrad_sum", "/", name, "/update"),
          std::vector<c10::IValue>());
      fused_sparse_row_wise_adagrad_sum(grad, grad_sum_sq, options, state);
    }
  }
}

void SparseRowWiseAdagradSum::ValidateOptions(
    const SparseRowWiseAdagradOptions &options) {
  TORCH_CHECK(options.lr() >= 0, "Invalid learning rate: ", options.lr());
  TORCH_CHECK(options.lr_decay() >= 0,
              "Invalid learning rate decay: ", options.lr_decay());
  TORCH_CHECK(options.initial_accumulator_value() >= 0,
              "Invalid initial accumulator value: ",
              options.initial_accumulator_value());
  TORCH_CHECK(options.eps() >= 0, "Invalid epsilon value: ", options.eps());
  TORCH_CHECK(options.weight_decay() == 0,
              "SparseRowWiseAdagradSum does not support weight_decay; got ",
              options.weight_decay());
}

void SparseRowWiseAdagradSum::load_state_dict(
    torch::Dict<std::string, HashTablePtr> hashtables,
    torch::Dict<std::string, torch::Tensor> steps) {
  std::unordered_map<std::string, HashTablePtr> current_params;
  for (const auto &group : param_groups_) {
    for (const auto &item : group.params()) {
      if (!item.second.defined()) {
        continue;
      }
      const auto inserted = current_params.emplace(item.first, item.second);
      TORCH_CHECK(inserted.second,
                  "SparseRowWiseAdagradSum parameter names must be unique: ",
                  item.first);
    }
  }

  if (!hashtables.empty()) {
    TORCH_CHECK(hashtables.size() == current_params.size(),
                "SparseRowWiseAdagradSum state has ", hashtables.size(),
                " HashTables, but the optimizer has ", current_params.size());
    for (const auto &item : hashtables) {
      const auto current_param = current_params.find(item.key());
      TORCH_CHECK(
          current_param != current_params.end(),
          "Unexpected SparseRowWiseAdagradSum HashTable state: ", item.key());
      TORCH_CHECK(
          item.value().defined(),
          "Undefined SparseRowWiseAdagradSum HashTable state: ", item.key());
      TORCH_CHECK(
          item.value().get() == current_param->second.get(),
          "SparseRowWiseAdagradSum.load_state_dict cannot copy row "
          "accumulators between different HashTable objects for parameter '",
          item.key(),
          "'; restore HashTable state with RecIS Saver/Loader first, then "
          "load the optimizer tensor state without foreign HashTable objects");
    }
  }

  std::unordered_map<std::string, torch::Tensor> loaded_steps;
  for (const auto &item : steps) {
    loaded_steps.emplace(item.key(), item.value());
  }
  const bool single_param = current_params.size() == 1;
  if (!single_param) {
    TORCH_CHECK(loaded_steps.count(StepName()) == 0,
                "The legacy SparseRowWiseAdagradSum step is ambiguous for ",
                current_params.size(),
                " parameters; load a checkpoint with per-parameter steps");
  } else {
    const auto &param_name = current_params.begin()->first;
    TORCH_CHECK(
        !(loaded_steps.count(StepName()) != 0 &&
          loaded_steps.count(StepName(param_name)) != 0),
        "SparseRowWiseAdagradSum state contains both legacy and per-parameter "
        "steps for '",
        param_name, "'");
  }

  for (const auto &item : loaded_steps) {
    bool expected = single_param && item.first == StepName();
    for (const auto &param : current_params) {
      expected = expected || item.first == StepName(param.first);
    }
    TORCH_CHECK(expected, "Unexpected SparseRowWiseAdagradSum tensor state: ",
                item.first);
  }

  std::vector<std::pair<torch::Tensor, torch::Tensor>> step_copies;
  step_copies.reserve(current_params.size());
  for (const auto &item : current_params) {
    const auto keyed_step_name = StepName(item.first);
    auto loaded_step = loaded_steps.find(keyed_step_name);
    if (loaded_step == loaded_steps.end() && single_param) {
      loaded_step = loaded_steps.find(StepName());
    }
    TORCH_CHECK(loaded_step != loaded_steps.end(),
                "Missing SparseRowWiseAdagradSum step for parameter '",
                item.first, "' (expected '", keyed_step_name,
                single_param ? torch::str("' or legacy '", StepName(), "'")
                             : std::string("'"),
                ")");
    TORCH_CHECK(
        loaded_step->second.defined() && loaded_step->second.numel() == 1,
        "SparseRowWiseAdagradSum step for parameter '", item.first,
        "' must be a defined one-element tensor");
    const auto step_type = loaded_step->second.scalar_type();
    TORCH_CHECK(!loaded_step->second.is_sparse() &&
                    (c10::isFloatingType(step_type) ||
                     c10::isIntegralType(step_type, false)),
                "SparseRowWiseAdagradSum step for parameter '", item.first,
                "' must be a dense real numeric tensor");

    if (c10::isFloatingType(step_type)) {
      const auto step_value = loaded_step->second.item<double>();
      TORCH_CHECK(std::isfinite(step_value) && step_value >= 0 &&
                      std::floor(step_value) == step_value &&
                      step_value < static_cast<double>(
                                       std::numeric_limits<int64_t>::max()),
                  "SparseRowWiseAdagradSum step for parameter '", item.first,
                  "' must be a non-negative integer representable as int64");
    } else {
      TORCH_CHECK(loaded_step->second.item<int64_t>() >= 0,
                  "SparseRowWiseAdagradSum step for parameter '", item.first,
                  "' must be non-negative");
    }

    const auto state_it = state_.find(item.second.get());
    TORCH_CHECK(state_it != state_.end(),
                "Missing SparseRowWiseAdagradSum state for parameter '",
                item.first, "'");
    auto &state =
        static_cast<SparseRowWiseAdagradParamState &>(*state_it->second);
    auto converted_step = loaded_step->second.to(state.step().options());
    step_copies.emplace_back(state.step(), std::move(converted_step));
  }

  for (const auto &step_copy : step_copies) {
    step_copy.first.copy_(step_copy.second);
  }
}

const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                 std::unordered_map<std::string, torch::Tensor>>
SparseRowWiseAdagradSum::state_dict() {
  std::unordered_map<std::string, HashTablePtr> hashtables;
  std::unordered_map<std::string, torch::Tensor> steps;
  for (auto &group : param_groups_) {
    for (auto &item : group.params()) {
      auto &param = item.second;
      if (!param.defined()) {
        continue;
      }
      const auto inserted = hashtables.emplace(item.first, param);
      TORCH_CHECK(inserted.second,
                  "SparseRowWiseAdagradSum parameter names must be unique: ",
                  item.first);
    }
  }

  const bool single_param = hashtables.size() == 1;
  for (const auto &item : hashtables) {
    const auto state_it = state_.find(item.second.get());
    TORCH_CHECK(state_it != state_.end(),
                "Missing SparseRowWiseAdagradSum state for parameter '",
                item.first, "'");
    const auto &state =
        static_cast<SparseRowWiseAdagradParamState &>(*state_it->second);
    const auto step_name = single_param ? StepName() : StepName(item.first);
    TORCH_CHECK(hashtables.count(step_name) == 0,
                "SparseRowWiseAdagradSum state key conflicts with parameter '",
                step_name, "'");
    const auto inserted = steps.emplace(step_name, state.step());
    TORCH_CHECK(inserted.second,
                "Duplicate SparseRowWiseAdagradSum state key: ", step_name);
  }

  for (const auto &item : steps) {
    TORCH_CHECK(hashtables.count(item.first) == 0,
                "SparseRowWiseAdagradSum state key conflicts with parameter '",
                item.first, "'");
  }
  return std::make_tuple(hashtables, steps);
}

c10::intrusive_ptr<SparseRowWiseAdagradSum> SparseRowWiseAdagradSum::Make(
    const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
    double lr_decay, double initial_accumulator_value, double eps,
    double weight_decay, bool maximize) {
  LOG(WARNING) << "SparseRowWiseAdagradSum Make: " << at::get_parallel_info()
               << "; lr is " << lr << "; lr_decay is " << lr_decay
               << "; initial_accumulator_value is " << initial_accumulator_value
               << "; eps is " << eps << "; weight_decay is " << weight_decay
               << "; maximize is " << maximize;
  SparseRowWiseAdagradOptions options;
  options.lr(lr);
  options.lr_decay(lr_decay);
  options.initial_accumulator_value(initial_accumulator_value);
  options.eps(eps);
  options.weight_decay(weight_decay);
  options.maximize(maximize);
  std::unordered_map<std::string, HashTablePtr> params;
  for (auto it = hashtables.begin(); it != hashtables.end(); ++it) {
    params[it->key()] = it->value();
  }
  auto optimizer =
      c10::make_intrusive<SparseRowWiseAdagradSum>(params, options);
  TORCH_CHECK(optimizer->size() > 0,
              "SparseRowWiseAdagradSum requires at least one parameter");
  return optimizer;
}

void SparseRowWiseAdagradSum::reset_state_dict() {
  for (auto &group : param_groups_) {
    for (auto &item : group.params()) {
      auto &param = item.second;
      if (!param.defined()) {
        continue;
      }
      auto &state =
          static_cast<SparseRowWiseAdagradParamState &>(*state_[param.get()]);
      state.reset_step();
    }
  }
}

}  // namespace optim
}  // namespace recis
