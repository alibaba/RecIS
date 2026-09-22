"""Compatibility helpers for checkpoint state names."""

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

import torch


_FILTER_GLOBAL_STEP_STATE_SUFFIX = "._hashtable._filter_hook_impl._global_step"
_FILTER_GLOBAL_STEP_CHILD_SUFFIX = "@filter_global_step"
_LEGACY_SPARSE_GRAD_GROUP_FIELDS = {
    "sparse_grad_group_size": "hdmp_group_size",
    "sparse_grad_group_reduce_by": "hdmp_group_reduce_by",
}


@dataclass(frozen=True)
class _FilterGlobalStepNames:
    """Checkpoint names for one coalesced hashtable filter step."""

    runtime_name: str
    child_names: Tuple[str, ...]
    legacy_names: Tuple[str, ...]


def is_filter_global_step_name(name: str) -> bool:
    """Returns whether a dense checkpoint name stores a filter global step."""
    return name.endswith(_FILTER_GLOBAL_STEP_CHILD_SUFFIX)


def _join_module_name(module_name: str, suffix: str) -> str:
    return f"{module_name}{suffix}" if module_name else suffix.lstrip(".")


def _legacy_filter_global_step_names(
    module_name: str, coalesced_info: str
) -> Tuple[str, ...]:
    """Returns hash-based names from earlier sparse-gradient-group schemas."""
    runtime_info = json.loads(coalesced_info)

    current_hash = hashlib.sha256(coalesced_info.encode()).hexdigest()
    current_marker = f"CoalescedHashtable_{current_hash}"
    if current_marker not in module_name:
        return ()

    legacy_infos = []
    renamed_info = {
        _LEGACY_SPARSE_GRAD_GROUP_FIELDS.get(field_name, field_name): value
        for field_name, value in runtime_info.items()
    }
    if renamed_info.get("grad_reduce_by") == "group_sum":
        renamed_info["grad_reduce_by"] = "hdmp_group_sum"
    if renamed_info != runtime_info:
        legacy_infos.append(renamed_info)

    pre_group_info = {
        field_name: value
        for field_name, value in runtime_info.items()
        if field_name not in _LEGACY_SPARSE_GRAD_GROUP_FIELDS
    }
    if pre_group_info != runtime_info:
        legacy_infos.append(pre_group_info)

    legacy_names = []
    for legacy_info in legacy_infos:
        legacy_hash = hashlib.sha256(json.dumps(legacy_info).encode()).hexdigest()
        legacy_module_name = module_name.replace(
            current_marker, f"CoalescedHashtable_{legacy_hash}", 1
        )
        legacy_name = _join_module_name(
            legacy_module_name, _FILTER_GLOBAL_STEP_STATE_SUFFIX
        )
        if legacy_name not in legacy_names:
            legacy_names.append(legacy_name)
    return tuple(legacy_names)


def collect_filter_global_step_names(
    model: torch.nn.Module,
) -> List[_FilterGlobalStepNames]:
    """Collects stable child names and compatible legacy names."""
    groups = []
    for module_name, module in model.named_modules():
        emb_opt = getattr(module, "_emb_opt", None)
        hashtable = getattr(module, "_hashtable", None)
        filter_hook = getattr(hashtable, "_filter_hook_impl", None)
        if emb_opt is None or not hasattr(filter_hook, "_global_step"):
            continue

        runtime_name = _join_module_name(module_name, _FILTER_GLOBAL_STEP_STATE_SUFFIX)
        child_names = tuple(
            f"{child}{_FILTER_GLOBAL_STEP_CHILD_SUFFIX}" for child in emb_opt.children
        )
        legacy_names = [runtime_name]
        coalesced_info = getattr(module, "_checkpoint_coalesced_info", None)
        if coalesced_info is None:
            coalesced_info = emb_opt.coalesced_info()
        legacy_names.extend(
            name
            for name in _legacy_filter_global_step_names(module_name, coalesced_info)
            if name != runtime_name
        )
        groups.append(
            _FilterGlobalStepNames(
                runtime_name=runtime_name,
                child_names=child_names,
                legacy_names=tuple(legacy_names),
            )
        )
    return groups


def use_child_filter_global_step_names(
    dense_state_dict: OrderedDict,
    groups: List[_FilterGlobalStepNames],
) -> None:
    """Replaces runtime hash-based names with stable child names for saving."""
    for group in groups:
        if group.runtime_name not in dense_state_dict:
            continue
        global_step = dense_state_dict.pop(group.runtime_name)
        for child_name in group.child_names:
            if child_name in dense_state_dict:
                raise ValueError(
                    f"Duplicate filter global step checkpoint name: {child_name}"
                )
            dense_state_dict[child_name] = global_step
