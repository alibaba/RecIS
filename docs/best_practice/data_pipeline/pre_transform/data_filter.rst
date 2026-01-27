样本过滤

功能简介
--------

RecIS 支持在 IO 模块进行高效的 **样本过滤（Sample Filtering）** 预处理。通过继承 ``UserDefineModule``，用户可以基于原始特征编写自定义的过滤逻辑。该逻辑在数据读取阶段通过 PyTorch 算子执行。

典型场景包括：
* 基于特定特征值（如 ``scene_flag``），在加载数据时实时生成样本分组标签（Group ID），用于过滤
  

使用样例
--------

以下示例展示了如何定义一个过滤模块，根据 ``scene_flag`` 的取值范围对样本进行分组过滤。

.. note::
   在 ``forward_impl`` 中，``input_data[1]['_indicator']`` 是内置的压缩索引，用于恢复压缩特征为原始的非压缩状态。

.. code-block:: python

    import os
    import torch
    import unittest
    from recis.io import window_io
    from column_io.aot_compile.aot_module import UserDefineModule
    from column_io.dataset import dataset as dataset_io
    from column_io.dataset.odps_env_setup import refresh_odps_io_config

    class MyFilterModule(UserDefineModule):
        """
        自定义过滤模块：基于 scene_flag 进行样本分组
        """
        def __init__(self):
            super().__init__()

        def forward_impl(self, input_data) -> torch.Tensor:
            # 提取 scene_flag 和内置的 _indicator
            scene_flag_tensor = input_data[1]['scene_flag'][0][0]
            indicator_tensor = input_data[1]['_indicator'][0][0]

            # 定义过滤条件：保留 scene_flag 在 [0, 4) 范围内的样本
            condition = (scene_flag_tensor >= 0) & (scene_flag_tensor < 4)
            
            # 生成分组 ID：符合条件的设为 0，不符合的设为 -1
            group_id = torch.where(
                condition,
                torch.zeros_like(scene_flag_tensor, dtype=torch.int64),
                -1 * torch.ones_like(scene_flag_tensor, dtype=torch.int64)
            )
            
            # 根据 indicator 重新映射并打平
            group_id_gathered = group_id.index_select(0, indicator_tensor.long())
            return group_id_gathered.reshape(-1)

    class TestSampleFilter(unittest.TestCase):
        def test_read_with_filter(self):
            # 1. 环境配置
            odps_path = "odps://your_project/tables/your_table/ds=20250101"
            os.environ["access_key"] = "your_access_key"
            os.environ["access_id"] = "your_access_id"
            os.environ["project_name"] = "your_project"
            os.environ["end_point"] = "http://service.odps.aliyun-inc.com/api"
            
            refresh_odps_io_config(
                os.environ["project_name"], 
                os.environ["access_id"], 
                os.environ["access_key"], 
                os.environ["end_point"], 
                table_name=odps_path
            )

            # 2. 初始化 IO 模块，并传入自定义过滤模块 user_define_module
            user_module = MyFilterModule()
            io_class = window_io.make_odps_window_io(row_num=500000)
            
            io = io_class(
                batch_size=4096,
                rank=int(os.environ.get("RANK", 0)),
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
                user_define_module=user_module  # 注入过滤逻辑
            )

            # 3. 添加路径并定义特征
            io.add_path(odps_path)
            io.varlen_feature("scene_flag")
            io.varlen_feature("nickname")

            # 4. 迭代数据
            io.next_window()
            batch = next(iter(io))
            print(f"Filtered batch shape: {batch[1][1]['scene_flag'].shape}")

    if __name__ == "__main__":
        unittest.main()

关键要点
--------

1. **继承关系**：必须继承 ``column_io.aot_compile.aot_module.UserDefineModule``。
2. **输入结构**：``forward_impl`` 的输入 ``input_data`` 是一个嵌套字典，包含了当前 io 内所有被声明的原始特征。
3. **性能优势**：该逻辑运行在 C++ 侧的算子中。
