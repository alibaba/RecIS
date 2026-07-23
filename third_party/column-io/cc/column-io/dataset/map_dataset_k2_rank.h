#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_MAP_DATASET_K2_RANK_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_MAP_DATASET_K2_RANK_H_

#include <map>
#include <memory>
#include <string>

#include "column-io/dataset/dataset.h"
namespace column {
namespace dataset {
class MapDataSetK2Rank {
public:
  static std::shared_ptr<DatasetBase> MakeDataSet(
    const std::shared_ptr<DatasetBase> &input, 
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>& new_input_schema,
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>& old_input_schema,
    const std::map<std::string, int32_t> &scene_map);
};
} // namespace dataset
} // namespace column
#endif