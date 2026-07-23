#ifndef TENSORFLOW_CORE_PLATFORM_SWIFT_COLUMN_DATA_FORMATER_H
#define TENSORFLOW_CORE_PLATFORM_SWIFT_COLUMN_DATA_FORMATER_H

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// #include "absl/container/flat_hash_map.h"
// #include "absl/container/flat_hash_set.h"
#include "arrow/record_batch.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"

namespace column {
namespace dataset {
struct ColumnSchema {
  // output name to output types
  std::vector<std::map<std::string, std::string>> output_schema;
  // The [col_idx_begin, col_idx_end) for col_name in (vector<Tensor>) produced
  // by flatconvert, used by row_mode CastTensor
  mutable std::unordered_map<std::string, std::pair<size_t, size_t>>
      flatconvert_tensor_spliter;
  // output name of hash features
  std::unordered_map<std::string, std::pair<std::string, int32_t>>
      hash_features;
  // column name to dense defaults
  std::unordered_map<std::string, Tensor> dense_defaults;
  // fields for compressed sample
  // column name to output name
  std::unordered_map<std::string, std::string> alias_map;
  // output name to column name
  std::unordered_map<std::string, std::string> alias_map_reversed;
  // column name to group index
  std::unordered_map<std::string, size_t> group_idx_map;
  std::string DebugString() const;
};

class ColumnDataFormater {
public:
  ColumnDataFormater() : with_null_(false) {}
  ColumnDataFormater(bool with_null) : with_null_(with_null) {}
  virtual ~ColumnDataFormater() {}
  static std::unique_ptr<ColumnDataFormater>
  GetColumnDataFormater(bool is_compressed, bool is_large_list);
  static std::unique_ptr<ColumnDataFormater>
  GetColumnDataFormater(bool is_compressed, bool is_large_list, bool with_null);
  virtual Status
  InitSchema(std::shared_ptr<arrow::Schema> arrow_schema,
             const std::vector<std::string> &hash_features,
             const std::vector<std::string> &hash_types,
             const std::vector<int64_t> &hash_buckets,
             const std::vector<std::string> &dense_features,
             const std::vector<Tensor> &dense_defaults,
             const std::unordered_set<std::string> &selected_columns);
  const ColumnSchema &schema() const { return schema_; }
  Status
  GetOutputSchema(std::vector<std::map<std::string, std::string>> *schema);
  virtual Status
  FormatSample(std::shared_ptr<arrow::RecordBatch> &data,
               std::vector<std::shared_ptr<arrow::RecordBatch>> *formated_data,
               int64 offset_end = -1,
               std::vector<int64> *offset_ends = nullptr) const = 0;
  virtual Status
  Convert(std::vector<std::shared_ptr<arrow::RecordBatch>> &formated_data,
          std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>>
              *output,
          std::vector<int64> offset_ends = {},
          bool force_deep_copy = false) const = 0;
  Status
  FlatConvert(std::vector<std::shared_ptr<arrow::RecordBatch>> &formated_data,
              std::vector<Tensor> *output, std::vector<int64> offset_ends = {},
              bool force_deep_copy = false) const;
  Status FlatConvert(
      std::vector<std::shared_ptr<arrow::RecordBatch>> &formated_data,
      std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>>
          *conv_output,
      std::vector<int64> offset_ends = {}, bool force_deep_copy = false) const;
  virtual void GetInputColumns(std::vector<std::string> *input_columns) = 0;
  void LogDebugString(std::shared_ptr<arrow::RecordBatch> record_batch) const;
  std::string
  DebugString(std::shared_ptr<arrow::RecordBatch> record_batch) const;

protected:
  mutable std::mutex init_mutex_;
  bool schema_inited_{false};
  ColumnSchema schema_;
  bool with_null_{false};
};

} // namespace dataset
} // namespace column

#endif // TENSORFLOW_CORE_PLATFORM_SWIFT_COLUMN_DATA_FORMATER_H
