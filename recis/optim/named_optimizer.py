from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from torch.optim import SGD, Adagrad, Adam, AdamW

from recis.optim import AdamWTF
from recis.utils.logger import Logger


logger = Logger(__name__)


def wrapped_named_optimizer(optim):
    class NamedOptimizer(optim):
        def __init__(self, named_params, *args, **kwargs):
            self.named_optimizer = True
            self.param_index_map = {}
            self.index_param_map = {}
            self.param_info = {}
            # convert to param groups
            if not isinstance(named_params, dict):
                # multi param groups or named_params
                named_params = list(named_params)
                if not isinstance(named_params[0], dict):
                    # named_params
                    named_params = [{"params": named_params}]
            else:
                # only one param group
                named_params = [named_params]
            super().__init__(named_params, *args, **kwargs)

        def _raise_error_by_strict(self, message: str, strict: bool):
            if strict:
                raise RuntimeError(message)
            else:
                logger.warning(message)

        def _build_param_group(self, param_group):
            params = param_group.pop("params")
            if isinstance(params, torch.Tensor):
                params = [params]
            param_group["params"] = []
            param_group["param_names"] = []
            cur_size = len(self.index_param_map)
            for i, (name, p) in enumerate(params):
                self.index_param_map[i + cur_size] = name
                if name in self.param_index_map:
                    self._raise_error_by_strict(
                        "found same name in param group, please concact recis develop team to solve this issue",
                        True,
                    )
                self.param_index_map[name] = i + cur_size
                param_group["params"].append(p)
                param_group["param_names"].append(name)

            return param_group

        def add_param_group(self, named_param_group):
            param_group = self._build_param_group(named_param_group)
            super().add_param_group(param_group)

        def state_dict(self, origin: bool = False):
            state = super().state_dict()
            if origin:
                return state

            new_state = {"state": {}, "param_groups": []}

            new_state["state"] = {
                self.index_param_map[k]: v for k, v in state["state"].items()
            }

            new_state["param_groups"] = [
                {
                    "params": [self.index_param_map[p] for p in param_group["params"]],
                    **{k: v for k, v in param_group.items() if k != "params"},
                }
                for param_group in state["param_groups"]
            ]
            return new_state

        def _get_ori_index_param_map(self, state_dict: dict):
            ori_index_param_map = {}
            # origin optimizer format
            # use param_names to map param index to model name
            if "param_names" in state_dict["param_groups"][0]:
                # has param name info, convert to name
                for param_group in state_dict["param_groups"]:
                    pns = param_group.pop("param_names")
                    for i in range(len(param_group["params"])):
                        ori_index_param_map[param_group["params"][i]] = pns[i]
            else:
                logger.warning(
                    "Load optimizer from origin torch optimizer without `param_names`, param index will orgnized by optimizer index, this may conduct error!!!"
                )
                ori_index_param_map = self.index_param_map
            return ori_index_param_map

        def _convert_ori_index_to_named_index(
            self, state_dict: dict, ori_index_param_map: dict
        ):
            for param_group in state_dict["param_groups"]:
                params = param_group.pop("params")
                param_group["params"] = []
                param_group["param_names"] = []
                for p_idx in params:
                    if p_idx in ori_index_param_map:
                        param_group["params"].append(ori_index_param_map[p_idx])
                        param_group["param_names"].append(ori_index_param_map[p_idx])
                    else:
                        logger.warning(
                            f"_convert_ori_index_to_named_index, No model found for {p_idx} in ckpt"
                        )

            ori_state = state_dict.pop("state")
            state_dict["state"] = {}
            for idx, s in ori_state.items():
                if idx in ori_index_param_map:
                    state_dict["state"][ori_index_param_map[idx]] = s
                else:
                    logger.warning(
                        f"_convert_ori_index_to_named_index, No model found for {idx} in ckpt"
                    )

        def _process_load_map(
            self, state_dict: dict, load_map: dict, strict: bool = True
        ):
            all_pns = set(self.index_param_map.values())
            ckpt_pns = set(state_dict["state"].keys())

            pname_map = {}
            for i, param_group in enumerate(state_dict["param_groups"]):
                for j, ckpt_name in enumerate(param_group["params"]):
                    pname_map[ckpt_name] = (i, j)

            for p_name, dst_name in load_map.items():
                if p_name not in all_pns:
                    error_msg = f"Load optimizer state error: {p_name} not in model."
                    self._raise_error_by_strict(error_msg, strict)
                if dst_name not in ckpt_pns:
                    error_msg = f"Load optimizer state error: {dst_name} not in ckpt."
                    self._raise_error_by_strict(error_msg, strict)

                ckpt_name = dst_name
                s = state_dict["state"].pop(ckpt_name)
                state_dict["state"][p_name] = s

                # ckpt name must in pname_map, and p_group_idx[0], p_group_idx[1] should in range
                try:
                    p_group_idx = pname_map[ckpt_name]
                    state_dict["param_groups"][p_group_idx[0]]["params"][
                        p_group_idx[1]
                    ] = p_name
                    state_dict["param_groups"][p_group_idx[0]]["param_names"][
                        p_group_idx[1]
                    ] = p_name
                except Exception as e:
                    raise RuntimeError(
                        "Load optimizer state error when process load map"
                    ) from e

        def _update_param_groups(
            self,
            param_groups: list[dict[str, Any]],
            saved_param_groups: list[dict[str, Any]],
        ):
            saved_group_map = {
                name: idx
                for idx, group in enumerate(saved_param_groups)
                for name in group["param_names"]
            }

            new_groups = []
            for group in param_groups:
                idx = -1
                for name in group["param_names"]:
                    if name in saved_group_map:
                        idx = saved_group_map[name]
                        saved_param_groups[idx]["param_names"] = group["param_names"]
                        saved_param_groups[idx]["params"] = group["params"]
                        new_groups.append(saved_param_groups[idx])
                        break

                if idx == -1:
                    new_groups.append(group)

            return new_groups

        def _safe_load_state_dict(
            self,
            state_dict: dict,
            valid_names: set[str],
        ):
            current_state = super().state_dict()

            # when not train model, state_dict["state"] is empty
            if len(state_dict["state"]) > 0:
                for name in valid_names:
                    idx = self.param_index_map[name]
                    # some layer will not be trained but created by model
                    # so state_dict["state"] will not have this layer, ignore this layer
                    if idx in state_dict["state"]:
                        current_state["state"][idx] = state_dict["state"][idx]

            current_state["param_groups"] = self._update_param_groups(
                deepcopy(self.state_dict(origin=True)["param_groups"]),
                deepcopy(state_dict["param_groups"]),
            )
            super().load_state_dict(current_state)

        def _load_origin_state_dict(
            self,
            state_dict: dict,
            valid_names: set[str],
        ):
            state = state_dict.pop("state")
            state_dict["state"] = {}

            for p_name, s in state.items():
                if p_name in self.param_index_map:
                    state_dict["state"][self.param_index_map[p_name]] = s
                else:
                    logger.warning(
                        f"_load_origin_state_dict, No model found for {p_name} provided in model"
                    )

            for param_group in state_dict["param_groups"]:
                params = param_group.pop("params")
                param_group["params"] = []
                for p_name in params:
                    if p_name in self.param_index_map:
                        param_group["params"].append(self.param_index_map[p_name])
                    else:
                        logger.warning(
                            f"_load_origin_state_dict, No model found for {p_name} provided in model"
                        )

            return self._safe_load_state_dict(state_dict, valid_names)

        def _check_param_groups(self, state_dict: dict):
            return len(state_dict["param_groups"]) == len(self.param_groups)

        def load_state_dict(
            self,
            state_dict: dict,
            valid_names: Optional[set[str]] = None,
            load_map: Optional[Dict[str, str]] = None,
            strict=True,
        ):
            if not self._check_param_groups(state_dict):
                logger.warning(
                    "Load optimizer state error: param groups is not match. may cause error when load dense optimizer !!!"
                )

            if valid_names is None:
                logger.warning(
                    "valid_names is None, this may cause error when load dense optimizer !!!"
                )
                valid_names = set(self.param_index_map.keys())

            # convert origin optimizer with param_names to named optimizer state
            # if not isinstance(list(state_dict["state"].keys())[0], str):
            if not isinstance(state_dict["param_groups"][0]["params"][0], str):
                ori_index_param_map = self._get_ori_index_param_map(state_dict)
                # update state_dict state idx to model_name, param_groups params idx to model_name
                self._convert_ori_index_to_named_index(state_dict, ori_index_param_map)

            if valid_names is not None:
                for name in self.param_index_map.keys():
                    if name not in valid_names:
                        logger.warning(
                            f"Missing {name} in ckpt when load dense optimizer"
                        )

            # convert by load_map
            if load_map is not None:
                self._process_load_map(state_dict, load_map, strict)

            # build origin map
            return self._load_origin_state_dict(state_dict, valid_names)

    return NamedOptimizer


NamedAdagrad = wrapped_named_optimizer(Adagrad)
NamedAdam = wrapped_named_optimizer(Adam)
NamedAdamW = wrapped_named_optimizer(AdamW)
NamedSGD = wrapped_named_optimizer(SGD)
NamedAdamWTF = wrapped_named_optimizer(AdamWTF)
