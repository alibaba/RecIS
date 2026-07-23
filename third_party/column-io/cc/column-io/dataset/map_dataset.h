#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_MAP_DATASET_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_MAP_DATASET_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_set>

#include "column-io/dataset/dataset.h"
#include "column-io/framework/status.h"
namespace column {
namespace dataset {
class MapDataSet {
public:
  static std::shared_ptr<DatasetBase> MakeDataSet(
    const std::shared_ptr<DatasetBase> &input, 
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>& new_input_schema,
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>& old_input_schema,
    const std::vector<std::string>& user_module_columns,
    const std::string& module_so_path);
};
} // namespace dataset
} // namespace column
#endif
