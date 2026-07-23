import unittest
import torch
import os
from column_io.dataset.odps_env_setup import refresh_odps_io_config
from column_io.dataset import dataset as dataset_io
from column_io.dataset.file_sharding import OdpsTableSharding
from column_io.aot_compile.aot_module import UserDefineModule

class MyModule(UserDefineModule):
    def __init__(self):
        super().__init__()

    def forward_impl(self, input_data) -> torch.Tensor:
        scene_flag_tensor = input_data[1]['scene_flag'][0][0]
        indicator_tensor = input_data[1]['_indicator'][0][0]

        condition = (scene_flag_tensor >= 0) & (scene_flag_tensor < 4)
        group_id = torch.where(
             condition,
             torch.zeros_like(scene_flag_tensor, dtype=torch.int64),
             -1 * torch.ones_like(scene_flag_tensor, dtype=torch.int64)
         )
        group_id_gathered = group_id.index_select(0, indicator_tensor.long())
        group_id_reshaped = group_id_gathered.reshape(-1)
        return group_id_reshaped

class PackerDatasetFilterTest(unittest.TestCase):
  def test_read(self):
    odps_path = os.getenv("ODPS_TABLE_PATH", "")
    access_key = os.getenv("ODPS_ACCESS_KEY", "")
    access_id = os.getenv("ODPS_ACCESS_ID", "")
    end_point = os.getenv("ODPS_ENDPOINT", "")
    project_name = os.getenv("ODPS_PROJECT_NAME", "")
    os.environ["access_key"] = access_key
    os.environ["access_id"] = access_id
    os.environ["project_name"] = project_name
    os.environ["end_point"] = end_point
    refresh_odps_io_config(project_name, access_id, access_key, end_point, table_name=odps_path)

    select_column = [
      "res_bid", 
      "scene_flag",
      "1_prov_name"
    ]
    
    dense_columns = []
    dense_defaults = []
    batch_size = 100
    user_module = MyModule()
    dataset = dataset_io.Dataset.from_odps_source(
            [odps_path],
            True, batch_size,
            select_column, [], [], [],
            dense_columns, dense_defaults)
    dataset = dataset.pack(batch_size, True, gpu_result=True,
        #user_define_module = user_module, 
        dense_columns = dense_columns,
        dense_default_value = dense_defaults 
    )
    iterator = iter(dataset)
    batch = next(iterator)
    dltensor_group_0_0 = batch[0]["_sample_group_id"][0][0]
    dltensor_scene_value = batch[1]["scene_flag"][0][0]
    dltensor_scene_offset = batch[1]["scene_flag"][0][1]
    dltensor_prov_name_value = batch[1]["1_prov_name"][0][0]
    tensor_prov_name_value = torch.from_dlpack(dltensor_prov_name_value)
    tensor_group_0_0 = torch.from_dlpack(dltensor_group_0_0)
    tensor_scene_value = torch.from_dlpack(dltensor_scene_value)
    tensor_scene_offset = torch.from_dlpack(dltensor_scene_offset)
    print(len(tensor_group_0_0))
    assert(len(tensor_group_0_0) == batch_size)
    print(tensor_scene_value)
    print(tensor_scene_offset)
    print(tensor_group_0_0)


if __name__ == "__main__":
  unittest.main()
