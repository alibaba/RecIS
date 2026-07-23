#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_LOCAL_ORC_DATASET_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_LOCAL_ORC_DATASET_H_
#include <map>
#include <memory>
#include <string>
#include <unordered_set>

#include "absl/status/status.h"
#include "column-io/dataset/dataset.h"
namespace column {
namespace dataset {
class LocalOrcDataset {
public:
  static absl::StatusOr<std::shared_ptr<DatasetBase>>
  MakeDataset(const std::vector<std::string> &paths, bool is_compressed,
              int64_t batch_size,
              const std::vector<std::string> &selected_columns,
              const std::vector<std::string> &input_columns,
              const std::vector<std::string> &hash_features,
              const std::vector<std::string> &hash_types,
              const std::vector<int64_t> &hash_buckets,
              const std::vector<std::string> &dense_columns,
              const std::vector<std::vector<float>> &dense_defaults);

  static std::shared_ptr<DatasetBase>
  MakeDatasetWrapper(const std::vector<std::string> &paths, bool is_compressed,
                     int64_t batch_size,
                     const std::vector<std::string> &selected_columns,
                     const std::vector<std::string> &input_columns,
                     const std::vector<std::string> &hash_features,
                     const std::vector<std::string> &hash_types,
                     const std::vector<int64_t> &hash_buckets,
                     const std::vector<std::string> &dense_columns,
                     const std::vector<std::vector<float>> &dense_defaults);

  static std::shared_ptr<DatasetBuilder>
  MakeBuilder(bool is_compressed, int64_t batch_size,
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
  ParseSchema(const std::vector<std::string> &paths, bool is_compressed,
              const std::unordered_set<std::string> &selected_columns,
              const std::vector<std::string> &hash_features,
              const std::vector<std::string> &hash_types,
              const std::vector<int64_t> &hash_buckets,
              const std::vector<std::string> &dense_columns,
              const std::vector<std::vector<float>> &dense_defaults);
};
} // namespace dataset
} // namespace column
#endif
