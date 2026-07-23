import torch.nn as nn
import torch
import numpy as np
import copy
import json
from column_io.dataset.nest import nest_seq_leaf_num
from column_io.dataset.odps_env_setup import refresh_odps_io_config
from column_io.dataset import dataset as dataset_io
from column_io.aot_compile.aot_compiler import TensorParam
from column_io.aot_compile.aot_compiler import UserModuleCompiler
from typing import Optional, List, Callable

TORCH_TYPE_MAP = {
    "int64": torch.int64,
    "int32": torch.int32,
    "float": torch.float32,
    "double": torch.float64,
    "float32": torch.float32,
    "float64": torch.float64,
    "bool": torch.bool
}

def _pack_nest_sequence_internal(schema, data_vec, comp, pos):
    if isinstance(schema, dict):
        output = {}
        for key in sorted(schema, key=comp):
            output[key], pos = _pack_nest_sequence_internal(
                schema[key], data_vec, comp, pos
            )
        return output, pos
    elif isinstance(schema, (list, tuple)):
        output = []
        for val in schema:
            new_val, pos = _pack_nest_sequence_internal(val, data_vec, comp, pos)
            output.append(new_val)
        return output, pos
    else:
        val = data_vec[pos]
        pos += 1
        return val, pos


def pack_nest_sequence(schema, data_vec: list, key=lambda x: x):
    # non nested data
    if nest_seq_leaf_num(schema) != len(data_vec):
        raise RuntimeError(
            (
                "schema and data_vec size mismatch"
                "schema : {}"
                "data vec: {}".format(schema, data_vec)
            )
        )
    if not isinstance(schema, (list, tuple, dict)):
        return data_vec[0]
    return _pack_nest_sequence_internal(schema, data_vec, key, 0)[0]


def nest_seq_leaf_num(schema, key=lambda x: x):
    ret = 0
    if isinstance(schema, dict):
        for key in sorted(schema, key=key):
            ret += nest_seq_leaf_num(schema[key], key)
    elif isinstance(schema, (list, tuple)):
        for val in schema:
            ret += nest_seq_leaf_num(val, key)
    else:
        ret += 1
    return ret


def is_string_dtype(arr):
    """Checks if a numpy array has string data type.

    Args:
        arr (np.ndarray): Input numpy array to check.

    Returns:
        bool: True if the array has string data type (Unicode, byte string, or object), False otherwise.
    """
    return arr.dtype.kind in {"U", "S", "O"}

class UserDefineModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def fill_compiler(self, input_schema, schema_type, dense_columns, dense_default_value, batch_size): 
        self._selected_columns = []
        self._dense_column = dense_columns
        self._dense_default_value = dense_default_value
        self._input_schema =  copy.deepcopy(input_schema)
        self._schema_dtype = schema_type
        self._batch_size = batch_size
        self._pop_string_field()
        self._compiler = UserModuleCompiler()
        self._init_compiler()
        self._aot_so_path = None
        super().__init__()
        self._compile_user_module()
   
    def get_aot_so_path(self):
        return self._aot_so_path

    def _parse_schema_type(self, schema_entry, need="value"):
        if schema_entry.startswith("{"):
            parts = schema_entry.strip("{}").split(",")
            d = {}
            for p in parts:
                k2, v2 = p.split(":")
                d[k2.strip()] = v2.strip()
            return d["v"] if need == "value" else d["k"]
        else:
            return schema_entry

    def _pop_string_field(self):
        schema = json.loads(self._schema_dtype)
        remove_list = []
        for table_idx, raw_batch in enumerate(self._input_schema):
            for fn, data in raw_batch.items():
                # ------ ① 提取 value/weight type ------
                value_type = self._parse_schema_type(schema[fn], "value")
                weight_type = None
                if len(data) > 1:
                    weight_type = self._parse_schema_type(schema[fn], "weight")
                # ------ ② 检查是否是 string------
                if value_type == "string" or weight_type == "string":
                    remove_list.append((table_idx, fn))
        for table_idx, fn in remove_list:
            print(f"[AOT-SKIP] Skip string feature: {fn}")
            self._input_schema[table_idx].pop(fn, None)
    
    def _init_compiler(self):
        """
        根据 selected_columns 和 _original_dtype 构建 TensorParam 列表，
        用于描述每个字段的 value + offsets 输入结构。
        支持：
        - 无权重字段：[value_dtype, offset_dtype*]
        - 有权重字段：[ [val_dtype, off_dtype*], [wgt_dtype, off_dtype*] ]
        """
        schema = json.loads(self._schema_dtype)
    
        all_params = []
        for table_idx, raw_batch in enumerate(self._input_schema):
            for fn, data in raw_batch.items():
                # -------- Case 1: len(data)==1 → ragged without weights --------
                if len(data) == 1:
                    one = data[0]
                    vtype = self._parse_schema_type(schema[fn], "value")
                    torch_vtype = TORCH_TYPE_MAP[vtype]
                    if fn in self._dense_column:
                        idx = self._dense_column.index(fn)
                        default_value = self._dense_default_value[idx]
                        default_tensor = torch.tensor(default_value)
                        dense_shape = default_tensor.shape
                        add_batch_shape = [None, *dense_shape]
                        all_params.append(TensorParam(
                            name=f"{fn}_value_tensor",
                            dtype=torch_vtype,
                            shape=add_batch_shape
                        ))

                    else:
                        # Value tensor
                        all_params.append(TensorParam(
                            name=f"{fn}_value_tensor",
                            dtype=torch_vtype
                        ))
                        # Offsets
                        for idx, off_dtype in enumerate(one[1:]):
                            all_params.append(TensorParam(
                                name=f"{fn}_value_offset_tensor_{idx}",
                                dtype=torch.int64
                            ))
                    continue
    
                # -------- Case 2: len(data)>1 → ragged with weights --------
                value_data = data[0]
                weight_data = data[1]
    
                vtype = self._parse_schema_type(schema[fn], "value")
                wtype = self._parse_schema_type(schema[fn], "weight")
    
                t_v = TORCH_TYPE_MAP[vtype]
                t_w = TORCH_TYPE_MAP[wtype]

                # Value tensor
                all_params.append(TensorParam(
                    name=f"{fn}_value_tensor",
                    dtype=t_v
                ))
                # Value offsets
                for idx, off_dtype in enumerate(value_data[1:]):
                    all_params.append(TensorParam(
                        name=f"{fn}_value_offset_tensor_{idx}",
                        dtype=torch.int64
                    ))

                # Weight tensor
                all_params.append(TensorParam(
                    name=f"{fn}_weight_tensor",
                    dtype=t_w
                ))
                # Weight offsets
                for idx, off_dtype in enumerate(weight_data[1:]):
                    all_params.append(TensorParam(
                        name=f"{fn}_weight_offset_tensor_{idx}",
                        dtype=torch.int64
                    ))        
        for param in all_params:
            self._compiler.add_tensor_type(param.name, param.dtype)
            if param.shape is not None:
                self._compiler.add_tensor_shape(param.name, param.shape)
            if param.example_generator is not None:
                self._compiler.add_example_generator(param.name, param.example_generator)

    def forward(self, input_arg):
        input_data = None
        if isinstance(input_arg, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in input_arg):
            input_batch = self._transform_to_batch(input_arg)
            input_data = input_batch
        else:
            input_data = input_arg
        return self.forward_impl(input_data)

 

    def get_user_module_columns(self):
        if len(self._selected_columns) == 0:
            schema_dtype = json.loads(self._schema_dtype)
            for table_idx, raw_batch in enumerate(self._input_schema):
                for fn, data in raw_batch.items():
                    self._selected_columns.append(fn)
        return self._selected_columns

    def get_aot_so_path(self):
        return self._aot_so_path 

    def forward_impl(self, input_data):
        raise NotImplementedError("forward_impl must be implemented by subclasses")
    
    def _transform_to_batch(self, data_vec):
        return pack_nest_sequence(self._input_schema, data_vec)

    def _add_tensor_type(self, name: str, dtype: torch.dtype):
        self._compiler.add_tensor_type(name, dtype)

    def _add_tensor_shape(self, name: str, shape: List[Optional[int]]):
        self._compiler.add_tensor_shape(name, shape)

    def _add_example_generator(self, name: str, fn: Callable[[List[int]], torch.Tensor]):
        self._compiler.add_example_generators(name, fn)
    
    def _compile_user_module(self):
        self._aot_so_path = self._compiler.compile(self.eval(), self._batch_size)
        print(self._aot_so_path)
        



