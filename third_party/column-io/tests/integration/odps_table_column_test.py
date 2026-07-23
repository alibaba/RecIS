import unittest
import os
from column_io.dataset.odps_env_setup import refresh_odps_io_config
from column_io.dataset import dataset as dataset_io

class OdpsTableColumnTest(unittest.TestCase):
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
    dataset = dataset_io.Dataset.from_odps_source([odps_path], True, batch_size, select_column, [], dense_columns, dense_defaults)
    iterator = iter(dataset)
    print(next(iterator))

  def test_parse_odps_schema(self):
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
    dataset_io.interface._OdpsTableColumnDataset.parse_schema([odps_path], True, set(select_column), [], dense_columns, dense_defaults)
if __name__ == "__main__":
  unittest.main()
