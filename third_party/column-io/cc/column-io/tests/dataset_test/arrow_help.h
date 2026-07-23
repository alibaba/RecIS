#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "arrow/api.h"
#include "arrow/record_batch.h"

namespace {


template <typename T>
struct ArrowTraits;

template <>
struct ArrowTraits<int32_t> {
  using BuilderType = arrow::Int32Builder;
  static std::shared_ptr<arrow::DataType> type() { return arrow::int32(); }
};

template <>
struct ArrowTraits<int64_t> {
  using BuilderType = arrow::Int64Builder;
  static std::shared_ptr<arrow::DataType> type() { return arrow::int64(); }
};

template <>
struct ArrowTraits<float> {
  using BuilderType = arrow::FloatBuilder;
  static std::shared_ptr<arrow::DataType> type() { return arrow::float32(); }
};

template <typename T>
std::shared_ptr<arrow::RecordBatch> MakeListBatch(const std::string &column_name,
                                                  const std::vector<std::vector<T>> &rows) {
  auto value_builder = std::make_shared<typename ArrowTraits<T>::BuilderType>();
  arrow::ListBuilder list_builder(arrow::default_memory_pool(), value_builder);

  for (const auto &row : rows) {
    EXPECT_TRUE(list_builder.Append().ok());
    EXPECT_TRUE(value_builder->AppendValues(row).ok());
  }

  std::shared_ptr<arrow::Array> array;
  EXPECT_TRUE(list_builder.Finish(&array).ok());

  auto schema = arrow::schema(
      {arrow::field(column_name, arrow::list(ArrowTraits<T>::type()))});
  return arrow::RecordBatch::Make(
      schema, static_cast<int64_t>(rows.size()), {array});
}

template <typename T>
std::shared_ptr<arrow::RecordBatch> MakeRecordBatch(
    const std::vector<std::pair<std::string, std::vector<std::vector<T>>>>& columns) {
  if (columns.empty()) {
    auto schema = arrow::schema({});
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::RecordBatch::Make(schema, 0, empty_arrays);
  }

  const int64_t num_rows = static_cast<int64_t>(columns[0].second.size());

  for (const auto& [name, col] : columns) {
    if (static_cast<int64_t>(col.size()) != num_rows) {
      throw std::invalid_argument("All columns must have the same number of rows");
    }
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;

  fields.reserve(columns.size());
  arrays.reserve(columns.size());

  for (const auto& [name, col_data] : columns) {
    using ValueBuilder = typename ArrowTraits<T>::BuilderType;

    auto value_builder = std::make_shared<ValueBuilder>();
    arrow::ListBuilder list_builder(arrow::default_memory_pool(), value_builder);

    for (const auto& cell : col_data) {
      auto st = list_builder.Append();
      if (!st.ok()) {
        throw std::runtime_error(st.ToString());
      }

      for (const auto& v : cell) {
        st = value_builder->Append(v);
        if (!st.ok()) {
          throw std::runtime_error(st.ToString());
        }
      }
    }

    std::shared_ptr<arrow::Array> array;
    auto st = list_builder.Finish(&array);
    if (!st.ok()) {
      throw std::runtime_error(st.ToString());
    }

    fields.push_back(arrow::field(name, arrow::list(ArrowTraits<T>::type())));
    arrays.push_back(array);
  }

  auto schema = arrow::schema(fields);
  return arrow::RecordBatch::Make(schema, num_rows, arrays);
}

}