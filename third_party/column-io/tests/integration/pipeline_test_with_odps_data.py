import unittest
import subprocess
import os
import json
import argparse
import gc
from column_io.dataset.dataset import Dataset
from column_io.dataset.odps_env_setup import refresh_odps_io_config
from column_io.dataset import dataset as dataset_io

class Test():
  fea_conf = json.load(open(os.path.join(os.path.dirname(__file__), "fea_conf.json")))
  def test_run(self, parallel):
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

    select_column = self.fea_conf["varlen"] + self.fea_conf["fixed_len"]
    dense_columns = ["label"]
    dense_defaults = [[0, 1]]
    batch_size = 4096
    thread_num = parallel
    dataset = dataset_io.Dataset.from_list_string([odps_path]*thread_num)
    dataset = dataset.parallel(
      lambda x: dataset_io.Dataset.from_odps_source([x], True, batch_size, 
                                                    select_column, [], dense_columns, dense_defaults), 
                                                    cycle_length=thread_num, block_length=1,sloppy=True, 
                                                    buffer_output_elements=1, 
                                                    prefetch_input_elements=0)
    dataset = dataset.pack(batch_size, drop_remainder=False)
    #dataset = dataset.prefetch(1)
    iterator = iter(dataset)
    state = open('state.bin', 'rb').read()
    iterator.deserialize(state)
    print("deserialize done")
    for _ in range(10):
      next(iterator)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int)
    args = parser.parse_args()
    test = Test()
    test.test_run(args.parallel)
