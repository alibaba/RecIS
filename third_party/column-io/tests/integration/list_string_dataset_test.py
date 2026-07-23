# 运行: pytest tests/integration/list_string_dataset_test.py -vs
# 前置: 需先构建并安装 column_io wheel (pip install dist/*.whl)
# 说明: 验证 py::module_local 修改未破坏基础数据集功能
import unittest
import numpy as np
from column_io.dataset.dataset import Dataset
class ListStringDatasetTest(unittest.TestCase):
  def test_dataset(self):
    test_strings =[b"hello", b"world"]
    dataset = Dataset.from_array_slice(test_strings)
    iterator = iter(dataset)
    outputs = []
    for _ in range(len(test_strings)):
      outputs.append(next(iterator))
    self.assertEqual(outputs, test_strings)
    self.assertRaises(StopIteration, lambda : next(iterator))

if __name__ == "__main__":
  unittest.main()
    
