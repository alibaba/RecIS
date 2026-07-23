import torch
import os
from typing import Optional, List, Callable

class TensorParam:
    def __init__(
        self,
        name: str,
        dtype: torch.dtype,
        shape: Optional[List[Optional[int]]] = None,
        example_generator: Optional[Callable[[List[Optional[int]]], torch.Tensor]] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.example_generator = example_generator


DEFAULT_DYNAMIC_MAX = 65536
DYNAMIC_BATCH_MAX = int(os.environ.get("AOT_BATCH_DIM_MAX", DEFAULT_DYNAMIC_MAX))
class UserModuleCompiler:
    def __init__(self):
        self._user_module_columns = []  # feature_column_name
        self._tensor_types = {}         # tensor name -> dtype
        self._tensor_shapes = {}       # tensor name -> shape template (e.g., [None], [None, 128], etc.)
        self._example_generators = {}   # tensor name -> callable to generate tensor

    def add_user_module_column(self, name: str):
        self._user_module_columns.append(name)
    
    def add_tensor_type(self, name: str, dtype: torch.dtype):
        self._tensor_types[name] = dtype

    def add_tensor_shape(self, name: str, shape: List[Optional[int]]):
        self._tensor_shapes[name] = shape

    def add_example_generator(self, name: str, fn: Callable[[List[int]], torch.Tensor]):
        self._example_generators[name] = fn
    
    def compile(self, user_module, batch_size=None):
        tensors = []
        dynamic_shapes_list = []
        def make_identifier(name: str) -> str:
            # 如果首字符不是字母或下划线，则加前缀 "_"
            if not (name[0].isalpha() or name[0] == "_"):
                name = "_" + name

            # 将非法字符替换为 "_"
            name = "".join(
                c if (c.isalnum() or c == "_") else "_"
                for c in name
            )
            return name
    

        for name, dtype in self._tensor_types.items():
            # 构造 concrete shape
            shape_template = self._tensor_shapes.get(name, [None])
            concrete_shape = [
                batch_size if dim is None else dim
                for dim in shape_template
            ]
            # 创建示例 tensor
            if name in self._example_generators:
                tensor = self._example_generators[name](concrete_shape)
            else:
                if dtype.is_floating_point:
                    tensor = torch.randn(concrete_shape, dtype=dtype)
                else:
                    tensor = torch.randint(0, 100, concrete_shape, dtype=dtype)
            tensors.append(tensor)
            safe_name = make_identifier(name)
            # 构造 dynamic_shapes（只对 None 的维度）
            dyn_shape = {
                i: torch.export.Dim(f"{safe_name}_dim{i}", min=1, max=DYNAMIC_BATCH_MAX)
                for i, dim in enumerate(shape_template) if dim is None
            }
            dynamic_shapes_list.append(dyn_shape)

        # ✅关键：args 是单元素 tuple，元素是 tensor 列表
        example_args = (tensors,)  # ← 一个参数：List[Tensor]

        # ✅dynamic_shapes 也是单元素 tuple，元素是 dynamic shape 列表
        final_dynamic_shapes = (dynamic_shapes_list,)  # ← 结构必须对齐！

        rank = int(os.environ.get("RANK", 0))
        from pathlib import Path
        output_so = Path.cwd() / f"user_define_module_rank{rank}.so"
        so_path = torch._export.aot_compile(
            user_module.eval(),
            args=example_args,
            dynamic_shapes=final_dynamic_shapes,
            options={
                "aot_inductor.output_path": str(output_so),
                "max_autotune": True
            }
        )
        return so_path

