#ifndef _COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_ODPS_COMBO_DATASET_H_
#define _COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_ODPS_COMBO_DATASET_H_
#include <cstddef>
#include <memory>
#include <string>

#include "column-io/dataset/dataset.h"
#include "column-io/framework/status.h"
#include "column-io/framework/types.h"

namespace column {
namespace dataset {
class OdpsComboDataset {
public:
  static absl::StatusOr<std::shared_ptr<DatasetBase>>
  MakeDataset(const std::vector<std::vector<std::string>> &paths,
              bool is_compressed, int64_t batch_size,
              const std::vector<std::vector<std::string>> &selected_columns,
              const std::vector<std::vector<std::string>> &input_columns,
              const std::vector<std::string> &hash_features,
              const std::vector<std::string> &hash_types,
              const std::vector<int64_t> &hash_buckets,
              const std::vector<std::string> &dense_columns,
              const std::vector<std::vector<float>> &dense_defaults,
              const bool &check_data, const std::string &primary_key,
              bool turn_on_odps_open_storage);

  static std::shared_ptr<DatasetBase> MakeDatasetWrapper(
      const std::vector<std::vector<std::string>> &paths, bool is_compressed,
      int64_t batch_size,
      const std::vector<std::vector<std::string>> &selected_columns,
      const std::vector<std::vector<std::string>> &input_columns,
      const std::vector<std::string> &hash_features,
      const std::vector<std::string> &hash_types,
      const std::vector<int64_t> &hash_buckets,
      const std::vector<std::string> &dense_columns,
      const std::vector<std::vector<float>> &dense_defaults,
      const bool &check_data, const std::string &primary_key,
      bool turn_on_odps_open_storage);

  static std::shared_ptr<DatasetBuilder>
  MakeBuilder(bool is_compressed, int64_t batch_size,
              const std::vector<std::vector<std::string>> &selected_columns,
              const std::vector<std::vector<std::string>> &input_columns,
              const std::vector<std::string> &hash_features,
              const std::vector<std::string> &hash_types,
              const std::vector<int64_t> &hash_buckets,
              const std::vector<std::string> &dense_columns,
              const std::vector<std::vector<float>> &dense_defaults,
              const bool &check_data, const std::string &primary_key,
              bool turn_on_odps_open_storage);
  //TODO: support map dataset schema
    // static std::tuple<
    //   std::vector<std::string>,
    //   std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>,
	//   std::string
    // >ParseSchema(const std::vector<std::vector<std::string>> &paths, bool is_compressed);
};
} // namespace dataset
} // namespace column
#endif
