#ifndef _COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_ODPS_DATASET_H_
#define _COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_ODPS_DATASET_H_
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <unordered_map>

#include "column-io/dataset/dataset.h"
#include "column-io/framework/status.h"
#include "column-io/framework/types.h"
namespace column {
namespace dataset {
class OdpsTableColumnDataset {
public:
  static absl::StatusOr<std::shared_ptr<DatasetBase>>
  MakeDataset(const std::vector<std::string> &paths,
              bool is_compressed,
              int64_t batch_size,
              const std::vector<std::string> &selected_columns,
              const std::vector<std::string> &input_columns,
              const std::vector<std::string> &hash_features,
              const std::vector<std::string> &hash_types,
              const std::vector<int64_t> &hash_buckets,
              const std::vector<std::string> &dense_columns,
              const std::vector<std::vector<float>> &dense_defaults);

  static std::shared_ptr<DatasetBase>
  MakeDatasetWrapper(const std::vector<std::string> &paths,
                     bool is_compressed,
                     int64_t batch_size,
                     const std::vector<std::string> &selected_columns,
                     const std::vector<std::string> &input_columns,
                     const std::vector<std::string> &hash_features,
                     const std::vector<std::string> &hash_types,
                     const std::vector<int64_t> &hash_buckets,
                     const std::vector<std::string> &dense_columns,
                     const std::vector<std::vector<float>> &dense_defaults);

  static std::shared_ptr<DatasetBuilder>
  MakeBuilder(bool is_compressed,
              int64_t batch_size,
              const std::vector<std::string> &selected_columns,
              const std::vector<std::string> &input_columns,
              const std::vector<std::string> &hash_features,
              const std::vector<std::string> &hash_types,
              const std::vector<int64_t> &hash_buckets,
              const std::vector<std::string> &dense_columns,
              const std::vector<std::vector<float>> &dense_defaults);

  static std::tuple<
      std::vector<std::string>,
      std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>,
	  std::string
	  >
  ParseSchema(const std::vector<std::string> &paths,
              bool is_compressed,
              const std::unordered_set<std::string> &selected_columns,
              const std::vector<std::string> &hash_features,
              const std::vector<std::string> &hash_types,
              const std::vector<int64_t> &hash_buckets,
              const std::vector<std::string> &dense_columns,
              const std::vector<std::vector<float>> &dense_defaults);

  static Status GetTableSize(const std::string &path, size_t *ret);

  static std::unordered_map<std::string, std::string> GetOdpsTableFeatures(const char* str_path, bool is_compressed=false);
};
} // namespace dataset
} // namespace column

#endif // _COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_ODPS_DATASET_H_
