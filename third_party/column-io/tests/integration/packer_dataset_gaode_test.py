import unittest
import torch
import os
from column_io.dataset.odps_env_setup import refresh_odps_io_config
from column_io.dataset import dataset as dataset_io

class PackerDatasetTest(unittest.TestCase):
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
      "scene_index"
    ]
    dense_columns = ["scene_index"]
    dense_defaults = [[0.0]]
    batch_size = 1024
    dataset = dataset_io.Dataset.from_odps_source(
            [odps_path],
            False, batch_size,
            select_column, [], [], [],
            dense_columns, dense_defaults)
    dataset = dataset.pack(1024, True, gpu_result=True)
    iterator = iter(dataset)
    batch = next(iterator)
    dltensor = batch[0]['scene_index'][0][0]
    tensor = torch.from_dlpack(dltensor)
    print(tensor)
    condition = (tensor == 5)
    group_id = torch.where(
         condition,
         torch.zeros_like(tensor, dtype=torch.int64),
         -1 * torch.ones_like(tensor, dtype=torch.int64)
    )
    breakpoint()
    #assert(tensor.shape[0] == 10)


if __name__ == "__main__":
  unittest.main()
