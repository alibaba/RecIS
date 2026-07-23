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
        scene_flag_tensor = input_data[0]['scene_index'][0][0]
        #scene_flag_tensor2 = input_data[0]['label'][0][0]

        condition = (scene_flag_tensor == 3)
        group_id = torch.where(
             condition,
             torch.zeros_like(scene_flag_tensor, dtype=torch.int64),
             -1 * torch.ones_like(scene_flag_tensor, dtype=torch.int64)
         )
        group_id_reshaped = group_id.reshape(-1)
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
      "scene_index",
      "landing_page_show",
      "user_poiid_order_click_num",
      "scene_index_embedding",
      "poi_id",
      "query_type_feature",
      "realtime_history_yeo_johnson_smooth_poi_price",
      "label",
      "history_length",
      "all_history_action_ds_delta",
      "all_history_poiweight",
      "ragmm_relevance_level",
      "longtime_mix_seq_poi_price",
      "longtime_mix_seq_hour_index",
      "all_history_action_type_idx",
      "longtime_mix_seq_dist",
    ]

    #select_column = [
    #  "scene_index",
    #  "label",
    #]
    
    #dense_columns = ["label"]
    #dense_defaults = [[0.0]]
    dense_columns = ["scene_index"]
    dense_defaults = [[0.0]]
    #dense_columns = []
    #dense_defaults = []
    #dense_columns = ["scene_index", "label"]
    #dense_defaults = [[0.0], [0.0]]
    batch_size = 100
    user_module = MyModule()
    dataset = dataset_io.Dataset.from_odps_source(
            [odps_path],
            False, batch_size,
            select_column, [], [], [],
            dense_columns, dense_defaults)
    dataset = dataset.pack(batch_size, True, pinned_result=False, gpu_result=True,
        user_define_module = user_module, 
        dense_columns = dense_columns,
        dense_default_value = dense_defaults 
    )
    iterator = iter(dataset)
    batch = next(iterator)
    scene_index_dltensor = batch[0]['scene_index'][0][0]
    scene_index_tensor = torch.from_dlpack(scene_index_dltensor)
    print(scene_index_tensor)



if __name__ == "__main__":
  unittest.main()
