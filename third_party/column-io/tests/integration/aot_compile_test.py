import os
import torch
import torch.nn as nn
from typing import List, Dict, Any

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scene_flag_tensor: torch.Tensor, 
            scene_flag_offset_tensor: torch.Tensor, 
            indicator_tensor: torch.Tensor) -> torch.Tensor:

        condition = (scene_flag_tensor >= 0) & (scene_flag_tensor < 4)
        group_id = torch.where(
             condition,
             torch.zeros_like(scene_flag_tensor, dtype=torch.int64),
             -1 * torch.ones_like(scene_flag_tensor, dtype=torch.int64)
         )
        group_id_gathered = group_id.index_select(0, indicator_tensor.long())
        group_id_reshaped = group_id_gathered.reshape(-1)
        return group_id_reshaped

my_model = MyModule().eval()

scene_batch_dim = torch.export.Dim("scene_batch", min=1, max=4096)
indicator_batch_dim = torch.export.Dim("indicator_batch", min=1, max=8192)  # indicator 可能更长

dynamic_shapes = {
    'scene_flag_tensor': {0: scene_batch_dim},
    'scene_flag_offset_tensor': {0: scene_batch_dim},
    'indicator_tensor': {0: indicator_batch_dim},
}

SCENE_SIZE = 1024
INDICATOR_SIZE = 2048

example_scene_flag = torch.randint(-5, 5, (SCENE_SIZE,), dtype=torch.int64)
example_scene_flag_offset = torch.randint(0, 10, (SCENE_SIZE,), dtype=torch.int64)
example_indicator = torch.randint(0, SCENE_SIZE, (INDICATOR_SIZE,), dtype=torch.int64)

args_for_compile = (
    example_scene_flag,
    example_scene_flag_offset,
    example_indicator   
)

example_indicator_for_compile = torch.randint(0, example_scene_flag.shape[0], (example_scene_flag.shape[0],), dtype=torch.int64)

args_for_compile = (
    example_scene_flag,
    example_scene_flag_offset,
    example_indicator_for_compile
)


dynamicLib_path = torch._export.aot_compile(
    my_model, # 模型实例
    args = args_for_compile, # 模型真实输入
    dynamic_shapes = dynamic_shapes,
    options={
            "aot_inductor.output_path": 'my_module_cpu.so', # 动态库路径
            "max_autotune": True    # 开启最大sm优化
            },
)
print(f"Generated dynamic library at: {dynamicLib_path}")

