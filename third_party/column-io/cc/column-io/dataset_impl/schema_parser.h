#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_H_SCHEMA_PARSER_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_H_SCHEMA_PARSER_H_
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "column-io/dataset/formater.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include <arrow/api.h>
namespace column {
namespace dataset {
class SchemaParser {
  using RecordBatchReaderFn = std::function<Status(
      const std::string &path,
      const std::unordered_set<std::string> &selected_columns,
      const std::vector<std::string> &dense_features, bool is_compressed,
      std::shared_ptr<arrow::RecordBatch> *data)>;
  // TODO: impl RecordBatchReaderFnByRows if need,   neednt for just now

public:
  std::tuple<
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
              const std::vector<Tensor> &dense_defaults);

  static std::unique_ptr<SchemaParser> Make(RecordBatchReaderFn fn);

  std::pair<
      std::vector<std::string>,
      std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>>
  ParseSchemaByRows(const std::vector<std::string> &paths,
                    bool is_compressed,
                    const std::vector<std::string> &selected_columns,
                    const std::vector<std::string> &hash_features,
                    const std::vector<std::string> &hash_types,
                    const std::vector<int64_t> &hash_buckets,
                    const std::vector<std::string> &dense_columns,
                    const std::vector<Tensor> &dense_defaults);
  static std::unique_ptr<SchemaParser> MakeByRows(RecordBatchReaderFn fn);

private:
  SchemaParser(RecordBatchReaderFn fn) : rb_reader_(fn) {}
  Status ParseSchemaCommon(
      std::shared_ptr<arrow::RecordBatch> &data,
      const std::unordered_set<std::string> &selected_columns,
      const std::vector<std::string> &hash_features,
      const std::vector<std::string> &hash_types,
      const std::vector<int64_t> &hash_buckets,
      const std::vector<std::string> &dense_features,
      const std::vector<Tensor> &dense_defaults,
      bool is_compressed,
      bool is_large_list,
      std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>
          *output_schema,
      std::vector<std::string> *input_columns,
      bool with_null);
  std::string ExtractElementType(const std::shared_ptr<arrow::DataType>& type);
  std::string BuildSchemaJson(const std::shared_ptr<arrow::Schema>& schema, bool is_compressed);
  RecordBatchReaderFn rb_reader_;
};
} // namespace dataset
} // namespace column
#endif
