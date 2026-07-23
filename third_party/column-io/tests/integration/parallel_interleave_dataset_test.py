import unittest
import os
from column_io.dataset.odps_env_setup import refresh_odps_io_config
from column_io.dataset import dataset as dataset_io

class ParallelDatasetTest(unittest.TestCase):
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
      "label", #dense double
      "1227_30", #struct
      "154_21"
    ]
    dense_columns = ["label"]
    dense_defaults = [[0, 0]]
    batch_size = 1024
    dataset = dataset_io.Dataset.from_array_slice([odps_path])
    dataset = dataset.pack(batch_size, True)
    dataset = dataset.parallel(
      lambda x: dataset_io.Dataset.from_odps_source([x], True, batch_size, 
                                                    select_column, [], dense_columns, dense_defaults), 
                                                    cycle_length=8, block_length=1,sloppy=True, 
                                                    buffer_output_elements=1, 
                                                    prefetch_input_elements=0)
    iterator = iter(dataset)
    print(next(iterator))

if __name__ == "__main__":
  unittest.main()
