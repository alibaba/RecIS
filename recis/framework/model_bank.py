import fnmatch
import os
import pickle
import re
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Set

import numpy as np
import torch

from recis.framework.filesystem import get_file_system
from recis.info import is_internal_enabled
from recis.serialize.checkpoint_reader import CheckpointReader
from recis.utils.logger import Logger


if is_internal_enabled() and not os.environ.get("BUILD_DOCUMENT", None) == "1":
    from recis.utils.mos import Mos
else:
    Mos = None


logger = Logger(__name__)
tag = "[ModelBank]"
for level in ("info", "warning", "error"):
    old_func = getattr(logger, level)
    setattr(
        logger,
        level,
        lambda msg, *args, _old=old_func, **kwargs: _old(
            f"{tag} {msg}", *args, **kwargs
        ),
    )


@dataclass
class ModelBankEntry:
    path: str = field(default="")
    load: Set[str] = field(default_factory=lambda: {"*"})
    exclude: Set[str] = field(default_factory=set)

    is_dynamic: bool = False
    hashtable_clear: bool = True
    ignore_error: bool = False
    skip: bool = False
    oname: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelBankEntry":
        if "path" not in d:
            raise ValueError("Missing required field: 'path'")

        allowed_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in d.items() if k in allowed_keys}

        return cls(**filtered_data)

    def __post_init__(self):
        if self.skip:
            logger.warning(f"'skip' is True, skip this model bank: {self.path}.")
            return

        if not isinstance(self.path, str):
            raise TypeError(f"'path' must be a string, got {type(self.path).__name__}")
        if not self.path.strip():
            raise RuntimeError("'path' is empty, not load any model.")

        if isinstance(self.load, list):
            object.__setattr__(self, "load", set(self.load))
        if isinstance(self.exclude, list):
            object.__setattr__(self, "exclude", set(self.exclude))

        if not isinstance(self.load, set):
            raise TypeError(f"'load' must be a set, got {type(self.load).__name__}")
        if not isinstance(self.exclude, set):
            raise TypeError(
                f"'exclude' must be a set, got {type(self.exclude).__name__}"
            )

        if not isinstance(self.hashtable_clear, bool):
            raise TypeError(
                f"'hashtable_clear' must be a bool, got {type(self.hashtable_clear).__name__}"
            )

        if not isinstance(self.is_dynamic, bool):
            raise TypeError(
                f"'is_dynamic' must be a bool, got {type(self.is_dynamic).__name__}"
            )

        if not isinstance(self.ignore_error, bool):
            raise TypeError(
                f"'ignore_error' must be a bool, got {type(self.ignore_error).__name__}"
            )

        if not isinstance(self.oname, list):
            raise TypeError(f"'oname' must be a list, got {type(self.oname).__name__}")
        if not all(isinstance(item, dict) for item in self.oname):
            raise TypeError(
                f"'oname' must be a list of dictionaries, got {type(self.oname).__name__}"
            )


class DensePatternMatcher:
    def __init__(self):
        self.regex_cache = {}

    def _get_regex(self, pattern: str):
        if pattern not in self.regex_cache:
            escaped_pattern = pattern.replace(".", r"\.").replace("?", r"\?")
            regex_pattern = "^" + escaped_pattern.replace("*", "(.*)") + "$"
            self.regex_cache[pattern] = re.compile(regex_pattern)
        return self.regex_cache[pattern]

    def apply_mapping(self, key: str, oname_rules: list) -> Optional[str]:
        """
        mapping key from source to target

        Args:
            key: source key
            oname_rules: oname rules list, each rule is a dictionary {pattern: replacement}

        Returns:
            mapped key, if not mapped, return None
        """
        for rule in oname_rules:
            for pattern, replacement in rule.items():
                if fnmatch.fnmatch(key, pattern):
                    if "*" in pattern and "*" in replacement:
                        regex = self._get_regex(pattern)
                        match = regex.match(key)
                        if match:
                            captured_groups = match.groups()
                            result = replacement
                            for group in captured_groups:
                                result = result.replace("*", group, 1)
                            return result
                    elif "*" in pattern:
                        regex = self._get_regex(pattern)
                        match = regex.match(key)
                        if match:
                            captured_groups = match.groups()
                            if "*" in replacement:
                                pattern_prefix = pattern.split("*")[0]
                                replacement_prefix = replacement.split("*")[0]
                                if (
                                    pattern_prefix
                                    and replacement_prefix
                                    and key.startswith(pattern_prefix)
                                ):
                                    suffix = key[len(pattern_prefix) :]
                                    return replacement_prefix + suffix
                            elif pattern.endswith("*") and not replacement.endswith(
                                "*"
                            ):
                                prefix = pattern.replace("*", "")
                                if key.startswith(prefix):
                                    suffix = key[len(prefix) :]
                                    return replacement + suffix
                            return replacement
                    else:
                        if key == pattern:
                            return replacement

        return None


class MBC:
    PATH = "path"
    LOAD = "load"
    EXCLUDE = "exclude"
    IS_DYNAMIC = "is_dynamic"
    HASHTABLE_CLEAR = "hashtable_clear"
    ONAME = "oname"
    SYMBOL_ALL = "*"
    SYMBOL_EMPTY = ""
    SPECIFIC = "specific"
    COMMON = "common"
    FINAL = "final"
    VARIABLE = "variable"
    IGNORE_ERROR = "ignore_error"


def maybe_get_latest_version(path, force_sub_version=False):
    ckpt_id = None
    fs = get_file_system(path)
    if fs.exists(os.path.join(path, "checkpoint")):
        content = fs.open(os.path.join(path, "checkpoint"), "r").read()
        versions = content.split("\n")[::-1]
        for version in versions:
            if len(version) == 0:
                continue
            ckpt_id = version.strip()
            break
        logger.warning(f"Get latest checkpoint version {ckpt_id} from path {path}.")
    if ckpt_id is not None:
        real_path = os.path.join(path, ckpt_id)
    else:
        real_path = path
        if force_sub_version:
            real_path = ""
    logger.warning(f"Get real ckpt path {real_path} from {path}")
    return real_path


def get_update_path(path, is_bank=True) -> str:
    if len(path) == 0:
        logger.warning("get_update_path: path is empty")
        return ""

    if path.startswith("model."):
        mos = Mos(path, is_bank)
        path = mos.real_physical_path
    path = maybe_get_latest_version(path, (not is_bank))
    return path


def show_model_bank_format(name: str, model_bank):
    if len(model_bank) == 0:
        logger.warning(f"No {name} model bank to show")
        return

    res = f"============= {name} =============\n"
    all_names = []
    all_dyn_strs = []
    all_clear_strs = []
    all_oname_strs = []
    for tensors in model_bank.values():
        for name, meta in tensors.items():
            all_names.append(name)
            all_dyn_strs.append(str(meta.get("is_dynamic", "")))
            all_clear_strs.append(str(meta.get("hashtable_clear", "")))
            all_oname_strs.append(str(meta.get("oname", "")))

    name_width = max([len(n) for n in all_names] + [len("Tensor Name")])
    dyn_width = max([len(s) for s in all_dyn_strs] + [len("is_dynamic")])
    clear_width = max([len(s) for s in all_clear_strs] + [len("hashtable_clear")])
    oname_width = max([len(s) for s in all_oname_strs] + [len("oname")])
    header = (
        f"{'Tensor Name'.ljust(name_width)}  "
        f"{'is_dynamic'.ljust(dyn_width)}  "
        f"{'hashtable_clear'.ljust(clear_width)}  "
        f"{'oname'.ljust(oname_width)}"
    )
    sep_line = "-" * len(header)

    for path, tensors in model_bank.items():
        res += f"Checkpoint: {path}\n"
        res += "=" * len(header) + "\n"
        res += header + "\n"
        res += sep_line + "\n"
        for name in sorted(tensors, key=lambda x: ("@" not in x, x)):
            meta = tensors[name]
            dyn = str(meta.get("is_dynamic", ""))
            clear = str(meta.get("hashtable_clear", ""))
            oname = str(meta.get("oname", ""))
            res += f"{name.ljust(name_width)}  {dyn.ljust(dyn_width)}  {clear.ljust(clear_width)}  {oname.ljust(oname_width)}"
            res += "\n"
        res += "=" * len(header) + "\n"
        res += "\n"
    logger.info(res)


def raise_error(core_text: str, message: str, ignore_error: bool):
    if "*" in core_text:
        logger.warning(message)
    else:
        if not ignore_error:
            raise ValueError(message)
        else:
            logger.warning(message)


def get_match_by_pattern(pattern: str, var_list: set[str]):
    """
    pattern:
        * -> all variables
        model.var_* -> variables starting with model.var_
        model.var_1, model.var_2 -> model.var_1 and model.var_2
    """
    if pattern == MBC.SYMBOL_ALL:
        return var_list
    elif MBC.SYMBOL_EMPTY in pattern and len(pattern) > 1:
        return {var for var in var_list if fnmatch.fnmatch(var, pattern)}
    elif pattern in var_list:
        return {pattern}
    raise ValueError(f"Bad pattern: {pattern} couldn't match any variable")


def pickle_to_torch(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, np.generic):
        return torch.tensor(obj)
    if isinstance(obj, dict):
        return {k: pickle_to_torch(v) for k, v in obj.items()}
    return obj


def load_pt_file(ckpt_dir: str, file_name: str):
    pt_path = os.path.join(ckpt_dir, file_name + ".pt")
    pk_path = os.path.join(ckpt_dir, file_name + ".pkl")
    fs = get_file_system(os.path.join(ckpt_dir, "index"))
    data = {}
    from_pickle = False
    if fs.exists(pt_path):
        with fs.open(pt_path, "rb") as f:
            data = torch.load(f=f, weights_only=False)
    elif fs.exists(pk_path):
        from_pickle = True
        f = fs.open(pk_path, "rb")
        buf = f.read()
        data = pickle.loads(buf, encoding="latin1")
        f.close()
    return data, from_pickle


def parse_sparse_oname(
    onames: list,
    src_names: set[str],
    dst_names: set[str],
    ignore_error: bool,
    oname_success: list,
) -> dict:
    sparse_oname = {}
    for idx, oname in enumerate(onames):
        src_table = next(iter(oname.keys()))
        dst_table = next(iter(oname.values()))

        matched_src_names = get_match_by_pattern(src_table, src_names)
        if not matched_src_names:
            raise_error(
                src_table,
                f"[sparse_oname] Bad oname, src table {src_table} not found in src_names",
                True,
            )
            continue

        matched_dst_names = get_match_by_pattern(dst_table, dst_names)
        if not matched_dst_names:
            raise_error(
                dst_table,
                f"[sparse_oname] Bad oname, dst table {dst_table} not found in dst_names",
                True,
            )
            continue

        if len(matched_dst_names) != len(matched_src_names):
            raise_error(
                "",
                f"[sparse_oname] Bad oname, Dst table {matched_dst_names} has different number of variables than src table {matched_src_names}",
                False,
            )
            continue

        src_table_name = src_table.split("@")[0].rsplit("*", 1)[0]
        dst_table_name = dst_table.split("@")[0].rsplit("*", 1)[0]

        oname_success[idx] = 1
        for src_name in matched_src_names:
            dst_name = src_name.replace(src_table_name, dst_table_name)
            if dst_name not in dst_names:
                raise_error(
                    src_table,
                    f"[sparse_oname] Bad oname, Dst name {dst_name} not found in dst_names",
                    ignore_error,
                )
                continue
            sparse_oname[src_name] = dst_name

    return sparse_oname


def apply_oname_mapping(
    pattern_matcher: DensePatternMatcher, key: str, oname_rules: list
) -> Optional[str]:
    """
    mapping key from source to target (use cached PatternMatcher)

    Args:
        key: source key
        oname_rules: oname rules list, each rule is a dictionary {pattern: replacement}

    Returns:
        mapped key, if not mapped, return None
    """
    return pattern_matcher.apply_mapping(key, oname_rules)


def parse_dense_oname(
    pattern_matcher: DensePatternMatcher,
    oname: list,
    src_keys: set[str],
    dst_keys: set[str],
    ignore_error: bool,
    oname_success: list,
) -> dict:
    """
    mapping key from source to target model

    Optimize:
    - convert dst_keys to set, make lookup from O(n) to O(1)
    - use PatternMatcher to cache regex, avoid duplicate compilation

    Args:
        src_keys: source model state dict keys
        dst_keys: target model state dict keys
        oname: oname rules dict, format: {"oname": [{"pattern": "replacement"}, ...]}

    """
    dense_oname = {}
    oname_rules = oname
    dst_keys_set = set(dst_keys)

    for key in src_keys:
        if "@" in key:
            continue
        mapped_key = None
        for idx, rule in enumerate(oname_rules):
            for pattern in rule.keys():
                if fnmatch.fnmatch(key, pattern):
                    candidate = apply_oname_mapping(pattern_matcher, key, [rule])
                    if candidate and candidate in dst_keys_set:  # O(1)
                        mapped_key = candidate
                        break
            if mapped_key:
                oname_success[idx] = 1
                break

        if not mapped_key:
            mapped_key = apply_oname_mapping(pattern_matcher, key, oname_rules)

        if mapped_key:
            if mapped_key in dst_keys_set:
                dense_oname[key] = mapped_key
                logger.warning(f"[dense_oname] T {key} <- {mapped_key} (from dst_sd)")
            else:
                raise_error(
                    key,
                    f"[dense_oname] F {key} -> {mapped_key} (not found in dst_sd)",
                    True,
                )

    return dense_oname


def parse_oname(
    dense_pattern_matcher: DensePatternMatcher,
    oname: list,
    src_sparse_names: set[str],
    dst_sparse_names: set[str],
    src_dense_names: set[str],
    dst_dense_names: set[str],
    ignore_error: bool,
):
    oname_success = [0 for _ in range(len(oname))]
    dense_oname = parse_dense_oname(
        dense_pattern_matcher,
        oname,
        src_dense_names,
        dst_dense_names,
        ignore_error,
        oname_success,
    )
    sparse_oname = parse_sparse_oname(
        oname,
        src_sparse_names,
        dst_sparse_names,
        ignore_error,
        oname_success,
    )

    for idx, success in enumerate(oname_success):
        if success == 0:
            raise_error(
                next(iter(oname[idx].keys())),
                f"Oname {oname[idx]} failed",
                ignore_error,
            )

    return dense_oname, sparse_oname


class ModelBankParser:
    def __init__(
        self,
        output_dir: str,
        model_bank_content: list[Dict[str, Any]],
        model_names: set[str],
        sparse_model_names: set[str],
        sparse_tables: set[str],
        dense_model_names: set[str],
        extra_fields,
    ):
        self._output_dir = output_dir
        self._model_bank_content = model_bank_content
        self._extra_fields = extra_fields
        self._original_model_names = deepcopy(model_names)
        self._original_dense_model_names = deepcopy(dense_model_names)
        self._original_sparse_model_names = deepcopy(sparse_model_names)
        self._original_sparse_tables = deepcopy(sparse_tables)
        self._dense_oname = {}
        self._sparse_oname = {}
        self._dense_pattern_matcher = DensePatternMatcher()
        self._reset_work_state()

        logger.warning("checking model bank...")
        self._is_model_bank_valid()

    def _reset_work_state(self):
        self._model_names = deepcopy(self._original_model_names)
        self._dense_model_names = deepcopy(self._original_dense_model_names)
        self._sparse_model_names = deepcopy(self._original_sparse_model_names)
        self._sparse_tables = deepcopy(self._original_sparse_tables)
        self._dense_oname = {}
        self._sparse_oname = {}

    def _is_load_valid(self):
        for bank in self._model_bank_content:
            for name in bank[MBC.LOAD]:
                if "*" not in name and name not in self._model_names:
                    raise_error(
                        name,
                        f"Variable {name} not found in model names",
                        False,
                    )

    def has_bank(self):
        return len(self._model_bank) > 0

    def _is_model_bank_valid(self):
        self._complete_model_bank()
        self._is_load_valid()
        self._model_bank = [
            ModelBankEntry.from_dict(bank)
            for bank in self._model_bank_content
            if not bank.get("skip", False)
        ]
        self._complete_sparse_name()
        self._replace_io_fields()

    def _replace_io_fields(self):
        for bank in self._model_bank:
            if self._extra_fields.io_state in bank.load:
                bank.load.discard(self._extra_fields.io_state)
                bank.load.update(self._extra_fields.get_io_fields())
            if self._extra_fields.io_state in bank.exclude:
                bank.exclude.discard(self._extra_fields.io_state)
                bank.exclude.update(self._extra_fields.get_io_fields())

    def _get_dst_names(self, path: str, ignore_error: bool):
        """read index file, model file, extra file to get dst vars"""
        sparse_names = set()
        dense_names = set()
        extra_names = set()
        ckpt_path = path
        ckpt_path = get_update_path(path)
        if ckpt_path == "":
            raise RuntimeError(f"No update path found in {path}")

        logger.info(f"get ckpt names from ckpt_path: {ckpt_path}")
        fs = get_file_system(os.path.join(ckpt_path, "index"))

        reader = CheckpointReader(ckpt_path)
        sparse_names.update(reader.tensor_names())

        if fs.exists(os.path.join(ckpt_path, "model.pt")) or fs.exists(
            os.path.join(ckpt_path, "model.pkl")
        ):
            try:
                data, _ = load_pt_file(ckpt_path, "model")
                dense_names.update(data.keys())
            except Exception as e:
                if ignore_error:
                    logger.warning(f"Load dense model file failed: {e}")
                else:
                    raise e
        else:
            logger.warning(f"Dense model file not found in {ckpt_path}")

        if fs.exists(os.path.join(ckpt_path, "extra.pt")) or fs.exists(
            os.path.join(ckpt_path, "extra.pkl")
        ):
            try:
                data, _ = load_pt_file(ckpt_path, "extra")
                extra_names.update(data.keys())
                if self._extra_fields.prev_optim in data:
                    extra_names.discard(self._extra_fields.prev_optim)
                    extra_names.add(self._extra_fields.recis_dense_optim)
            except Exception as e:
                if ignore_error:
                    logger.warning(f"Load extra model file failed: {e}")
                else:
                    raise e
        else:
            logger.warning(f"Extra model file not found in {ckpt_path}")

        if fs.exists(os.path.join(ckpt_path, "io_state_0.pt")):
            extra_names.update(self._extra_fields.get_io_fields())

        return sparse_names, dense_names, extra_names

    def get_sparse_oname(self) -> dict:
        return self._sparse_oname

    def get_dense_oname(self) -> dict:
        return self._dense_oname

    def _check_dst_valid(
        self,
        name: str,
        bank_load: set[str],
        dst_names: set[str],
        sparse_oname: dict,
        dense_oname: dict,
        path: str,
        ignore_error: bool,
    ):
        cond_1 = name in dst_names
        cond_2 = sparse_oname.get(name, name) in dst_names
        cond_3 = dense_oname.get(name, name) in dst_names
        if not (cond_1 or cond_2 or cond_3):
            if name in bank_load:
                raise_error(
                    name,
                    f"No var {name} found in dst_names, ckpt path: {path}",
                    ignore_error,
                )
            else:
                raise_error(
                    name,
                    f"No var {name} found in dst_names, ckpt path: {path}",
                    True,
                )
        return cond_1 or cond_2 or cond_3

    def _get_names_set(self, names: Set[str]) -> set[str]:
        data = set()
        for name in names:
            data.update(get_match_by_pattern(name, self._model_names))
        return data

    def _add_dense_optim_names(self, names: set[str]):
        """
        if add dense modules, add recis.dense.optim to names automatically
        """

        has_dense_module = False
        for name in names:
            if name in self._dense_model_names:
                has_dense_module = True
                break
        if has_dense_module:
            names.add(self._extra_fields.recis_dense_optim)

    def _travel_model_bank_reversely(self, model_bank: list[ModelBankEntry]):
        var_dict = {}
        for bank in reversed(model_bank):
            if len(self._model_names) == 0:
                logger.warning("all variables are loaded, break parse model bank.")
                break
            path = bank.path
            dst_sparse_names, dst_dense_names, extra_names = self._get_dst_names(
                path, bank.ignore_error
            )
            dst_names = dst_sparse_names | dst_dense_names | extra_names
            if len(dst_names) == 0:
                logger.warning(f"No dst vars found in ckpt: {path}")
                continue

            exclude_names_set = self._get_names_set(bank.exclude)
            load_names_set = self._get_names_set(bank.load)
            self._add_dense_optim_names(load_names_set)

            need_load_names = load_names_set - exclude_names_set
            if len(need_load_names) == 0:
                logger.warning(
                    f"No need to load vars in {path} because all vars are excluded"
                )
                continue

            # parse oname
            oname = bank.oname
            dense_oname, sparse_oname = parse_oname(
                self._dense_pattern_matcher,
                oname,
                {k for k in self._sparse_model_names if k in need_load_names},
                dst_sparse_names,
                {k for k in self._dense_model_names if k in need_load_names},
                dst_dense_names,
                bank.ignore_error,
            )

            for name in need_load_names:
                # check if the variable is in the ckpt list
                add_var = self._check_dst_valid(
                    name,
                    bank.load,
                    dst_names,
                    sparse_oname,
                    dense_oname,
                    path,
                    bank.ignore_error,
                )

                if add_var:
                    var_dict.setdefault(name, {}).update(
                        {
                            MBC.LOAD: path,
                            MBC.IS_DYNAMIC: bank.is_dynamic,
                            MBC.HASHTABLE_CLEAR: bank.hashtable_clear,
                            MBC.IGNORE_ERROR: bank.ignore_error,
                        }
                    )
                    self._model_names.discard(name)

            self._dense_oname.setdefault(path, {}).update(dense_oname)
            self._sparse_oname.setdefault(path, {}).update(sparse_oname)

        return var_dict

    def parse_all_model_bank(self):
        logger.info("parse all model bank")
        self._reset_work_state()
        return self._get_parse_result(self._model_bank)

    def parse_dynamic_model_bank(self):
        logger.info("parse dynamic model bank")
        self._reset_work_state()
        dynamic_model_bank = []
        for bank in self._model_bank:
            if bank.is_dynamic is True:
                dynamic_model_bank.append(bank)
        return self._get_parse_result(dynamic_model_bank)

    def _get_parse_result(self, model_bank: list[ModelBankEntry]):
        var_dict = self._travel_model_bank_reversely(model_bank)
        return self._combine_bank_by_path(var_dict)

    def _combine_bank_by_path(self, var_dict: dict):
        path_dict = {}
        for var in var_dict:
            path = var_dict[var][MBC.LOAD]
            if path not in path_dict:
                path_dict[path] = {}
            path_dict[path][var] = {
                MBC.IS_DYNAMIC: var_dict[var][MBC.IS_DYNAMIC],
                MBC.IGNORE_ERROR: var_dict[var][MBC.IGNORE_ERROR],
            }

            if var in self._sparse_model_names:
                path_dict[path][var][MBC.HASHTABLE_CLEAR] = var_dict[var][
                    MBC.HASHTABLE_CLEAR
                ]
            if var in self._dense_oname[path]:
                path_dict[path][var][MBC.ONAME] = self._dense_oname[path][var]

            if var in self._sparse_oname[path]:
                path_dict[path][var][MBC.ONAME] = self._sparse_oname[path][var]

        return path_dict

    def _complete_sparse_name(self):
        for bank in self._model_bank:
            remove_vars = set()
            added_vars = set()
            if bank.load:
                for var in bank.load:
                    if var in self._sparse_tables:
                        remove_vars.add(var)
                        added_vars.add(var + "*")
            for remove_var in remove_vars:
                bank.load.discard(remove_var)
            for add_var in added_vars:
                bank.load.add(add_var)

            onames = []
            for oname in bank.oname:
                src, dst = next(iter(oname.items()))
                if ("*" in src and "*" not in dst) or ("*" not in src and "*" in dst):
                    raise ValueError(
                        f"Bad oname, src {src} and dst {dst} must have the same number of *"
                    )
                if "*" not in src and src in self._sparse_tables:
                    onames.append({src + "@*": dst + "@*"})
                else:
                    onames.append(oname)
            bank.oname = onames

    def _complete_model_bank(self):
        path = get_update_path(self._output_dir, False)
        if path != "":
            self._model_bank_content.append(
                {
                    MBC.PATH: path,
                    MBC.LOAD: {"*"},
                    MBC.EXCLUDE: set(),
                    MBC.IS_DYNAMIC: False,
                    MBC.HASHTABLE_CLEAR: True,
                    MBC.IGNORE_ERROR: False,
                    MBC.ONAME: [],
                }
            )
