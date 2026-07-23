#ifndef _COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_LAKE_STREAM_COLUMN_DATASET_H_
#define _COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_LAKE_STREAM_COLUMN_DATASET_H_
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "column-io/dataset/dataset.h"
#include "column-io/framework/status.h"
#include "column-io/framework/types.h"
namespace column {
namespace dataset {
class LakeStreamColumnDatase {
public:
  static Status ParseConfig(const std::string &config,
                            std::string &lake_path,
                            int64_t &begin_time,
                            int64_t &end_time,
                            int64_t &partition_count,
                            int64_t &slice_index,
                            int64_t &slice_count,
                            std::string &table_service_name,
                            bool &use_querier);

  static absl::StatusOr<std::shared_ptr<DatasetBase>>
  MakeDataset(const std::string &path,
              const std::vector<std::string> &selected_columns,
              const std::vector<std::string> &input_columns,
              const std::vector<std::string> &hash_features,
              const std::vector<std::string> &hash_types,
              const std::vector<int64_t> &hash_buckets,
              const std::vector<std::string> &dense_features,
              const std::vector<std::vector<float>> &dense_defaults,
              bool is_compressed,
              int64_t batch_size,
              bool use_prefetch,
              int64_t prefetch_thread_num,
              int64_t prefetch_buffer_size);

  static std::shared_ptr<DatasetBase>
  MakeDatasetWrapper(const std::string &path,
                     const std::vector<std::string> &selected_columns,
                     const std::vector<std::string> &input_columns,
                     const std::vector<std::string> &hash_features,
                     const std::vector<std::string> &hash_types,
                     const std::vector<int64_t> &hash_buckets,
                     const std::vector<std::string> &dense_features,
                     const std::vector<std::vector<float>> &dense_defaults,
                     bool is_compressed,
                     int64_t batch_size,
                     bool use_prefetch,
                     int64_t prefetch_thread_num,
                     int64_t prefetch_buffer_size);

  static std::shared_ptr<DatasetBuilder>
  MakeBuilder(const std::vector<std::string> &selected_columns,
              const std::vector<std::string> &input_columns,
              const std::vector<std::string> &hash_features,
              const std::vector<std::string> &hash_types,
              const std::vector<int64_t> &hash_buckets,
              const std::vector<std::string> &dense_features,
              const std::vector<std::vector<float>> &dense_defaults,
              bool is_compressed,
              int64_t batch_size,
              bool use_prefetch,
              int64_t prefetch_thread_num,
              int64_t prefetch_buffer_size);

  static std::tuple<
      std::vector<std::string>,
      std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>,
	  std::string
	  >
  ParseSchema(const std::string &paths,
              bool is_compressed,
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
