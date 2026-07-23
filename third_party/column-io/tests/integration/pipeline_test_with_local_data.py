import unittest
import os
from column_io.dataset.dataset import Dataset

class Test(unittest.TestCase):
  def test_run(self):
    test_data_dirs = ["/home/zhouzhengxiong.zzx/workspace/arrow_record_batch"]*10

    list_dataset = Dataset.from_list_string(test_data_dirs)

    select_column = [
      "label", #dense double
      "1227_30", #struct
      "154_21"
    ]
    dense_columns = ["label"]
    dense_defaults = [[0, 0]]
    batch_size = 10*1024

    dataset = list_dataset.parallel(
        lambda x: Dataset.from_rb_files(
            [x], True, int(batch_size/10),
            select_column, [],
            dense_columns, dense_defaults),
        cycle_length=8,
        block_length=1,
        sloppy=True,
        buffer_output_elements=1,
        prefetch_input_elements=0
    )
    dataset = dataset.repeat(1, 16)
    dataset = dataset.pack(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(20)
    iterator = iter(dataset)
    import time
    print("batch size is {}".format(batch_size))
    print("dataset schema is {}".format(dataset.schema))
    for i in range(30):
      beg_ts = time.time()
      next(iterator)
      end_ts = time.time()
      print("batch consume [{}] ms".format((end_ts - beg_ts)*1000))
if __name__ == "__main__":
    unittest.main()
