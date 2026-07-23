#include "column-io/dataset/formater.h"

#include <cstddef>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <queue>
#include "absl/log/log.h"
#include "absl/strings/str_replace.h"
#include "arrow/array.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_nested.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/type.h"

#include "column-io/dataset/hash_utils.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"

namespace column {
namespace dataset {
// Outside anonymous namespace just to make the friend declaration in
// tensorflow::Tensor apply.
class ArrowTensorBuffer : public Buffer {
public:
  ArrowTensorBuffer() = delete;

  explicit ArrowTensorBuffer(const std::shared_ptr<arrow::Buffer> &buffer)
      : Buffer((Allocator *)nullptr, (void *)buffer->data(),
               (void *)buffer->data(), buffer->size(), false),
        buffer_(buffer) {}

  void *begin() const override { return (void *)buffer_->data(); }
  size_t size() const override { return buffer_->size(); }

  static Tensor
  TensorFromArrowBuffer(const std::string feature,
                        const std::shared_ptr<arrow::Buffer> &buffer,
                        DataType type, const TensorShape &shape, Status *st,
                        bool force_deep_copy = false) {
    if (buffer->size() < shape.NumElements() * SizeOfType(type)) {
      *st = Status::InvalidArgument(
          feature, "'s size mismatch, shape: ", shape.DebugString(),
          ", dtype: ", type, ", buffer size: ", buffer->size());
      return Tensor();
    }
    // check if buffer is aligned
    /*
    if (reinterpret_cast<intptr_t>(buffer->data()) % EIGEN_MAX_ALIGN_BYTES ==
        0) {
      ArrowTensorBuffer *tensor_buffer = new ArrowTensorBuffer(buffer);
      core::ScopedUnref unref(tensor_buffer);
      *st = Status::OK();
      return Tensor(type, shape, tensor_buffer);
    } else {
      Tensor tensor(allocator, type, shape);
      std::memcpy(const_cast<char *>(tensor.tensor_data().data()),
                  buffer->data(), shape.num_elements() * SizeOfType(type));
      *st = Status::OK();
      return tensor;
    }
    */
    if (!force_deep_copy) {
      ArrowTensorBuffer *tensor_buffer = new ArrowTensorBuffer(buffer);
      Tensor ret(shape, type, tensor_buffer);
      tensor_buffer->Unref();
      return ret;
    } else {
      Tensor tensor(type, shape);
      std::memcpy(const_cast<char *>(tensor.data()), buffer->data(),
                  shape.NumElements() * SizeOfType(type));
      *st = Status::OK();
      return tensor;
    }
  }
  ~ArrowTensorBuffer() { buffer_.reset(); }

private:
  std::shared_ptr<arrow::Buffer> buffer_;
};

namespace { // anomymous namespace

const std::string kLargeList = "large_list";
const std::string kList = "list";

const std::string kNoIndicatorIdx = "0";
const std::string kIndicatorPre = "_indicator_";
const std::string kIndicator = "_indicator";
const std::string kIndicatorType = "int64";

const std::string kStructKey = "k";
const std::string kStructValue = "v";

const int VALUE_BUFFER = 1;

// utility to get types from data type
template <typename DATA_TYPE> struct DataTypeToTypes {};

#define DECLARE_DATA_TYPES(DATA_TYPE, ARROW_TYPE, ARROW_ARRAY_TYPE, TF_TYPE,   \
                           ARROW_BUILDER_TYPE)                                 \
  template <> struct DataTypeToTypes<DATA_TYPE> {                              \
    static const arrow::Type::type ArrowType{ARROW_TYPE};                      \
    typedef ARROW_ARRAY_TYPE ArrowArrayType;                                   \
    static const column::DataType TfType{TF_TYPE};                             \
    typedef ARROW_BUILDER_TYPE ArrowBuilderType;                               \
  };

DECLARE_DATA_TYPES(uint8_t, arrow::Type::UINT8, arrow::UInt8Array,
                   column::DataType::kUInt8, arrow::UInt8Builder)
DECLARE_DATA_TYPES(uint16_t, arrow::Type::UINT16, arrow::UInt16Array,
                   column::DataType::kUInt16, arrow::UInt16Builder)
DECLARE_DATA_TYPES(uint32_t, arrow::Type::UINT32, arrow::UInt32Array,
                   column::DataType::kUInt32, arrow::UInt32Builder)
DECLARE_DATA_TYPES(uint64_t, arrow::Type::UINT64, arrow::UInt64Array,
                   column::DataType::kUInt64, arrow::UInt64Builder)
DECLARE_DATA_TYPES(int8_t, arrow::Type::INT8, arrow::Int8Array,
                   column::DataType::kInt8, arrow::Int8Builder)
DECLARE_DATA_TYPES(int16_t, arrow::Type::INT16, arrow::Int16Array,
                   column::DataType::kInt16, arrow::Int16Builder)
DECLARE_DATA_TYPES(int32_t, arrow::Type::INT32, arrow::Int32Array,
                   column::DataType::kInt32, arrow::Int32Builder)
DECLARE_DATA_TYPES(int64_t, arrow::Type::INT64, arrow::Int64Array,
                   column::DataType::kInt64, arrow::Int64Builder)
DECLARE_DATA_TYPES(float, arrow::Type::FLOAT, arrow::FloatArray,
                   column::DataType::kFloat, arrow::FloatBuilder)
DECLARE_DATA_TYPES(bool, arrow::Type::BOOL, arrow::BooleanArray,
                   column::DataType::kBool, arrow::BooleanBuilder)
DECLARE_DATA_TYPES(double, arrow::Type::DOUBLE, arrow::DoubleArray,
                   column::DataType::kDouble, arrow::DoubleBuilder)

#undef DECLARE_DATA_TYPES

#define DECLARE_BY_NUMERIC_TYPES(DECLARE_MACRO)                                \
  DECLARE_MACRO(uint8_t)                                                       \
  DECLARE_MACRO(uint16_t)                                                      \
  DECLARE_MACRO(uint32_t)                                                      \
  DECLARE_MACRO(uint64_t)                                                      \
  DECLARE_MACRO(int8_t)                                                        \
  DECLARE_MACRO(int16_t)                                                       \
  DECLARE_MACRO(int32_t)                                                       \
  DECLARE_MACRO(int64_t)                                                       \
  DECLARE_MACRO(float)                                                         \
  DECLARE_MACRO(double)

template <typename DATA_TYPE> struct ListTypeToTypes {};

#define DECLARE_LIST_TYPES(LIST_TYPE, LIST_CLASS, LIST_BUILDER)                \
  template <> struct ListTypeToTypes<LIST_TYPE> {                              \
    typedef LIST_CLASS ListClass;                                              \
    typedef LIST_BUILDER ListBuilder;                                          \
  };

DECLARE_LIST_TYPES(arrow::LargeListType, arrow::LargeListArray,
                   arrow::LargeListBuilder)
DECLARE_LIST_TYPES(arrow::ListType, arrow::ListArray, arrow::ListBuilder)

#undef DECLARE_LIST_TYPES

void ReplaceLargeList(std::vector<std::map<std::string, std::string>> *schema) {
  for (auto &map : *schema) {
    for (auto &kv : map) {
      kv.second = absl::StrReplaceAll(kv.second, {{kLargeList, kList}});
    }
  }
}

template <typename LIST_TYPE, typename DATA_TYPE>
Status FillDefaultIfEmpty(const std::string &field_name,
                          const std::shared_ptr<arrow::Array> &input,
                          const Tensor &dense_default,
                          std::shared_ptr<arrow::Array> *output,
                          int64 offset_end = -1) {
  auto casted_input =
      std::dynamic_pointer_cast<typename ListTypeToTypes<LIST_TYPE>::ListClass>(
          input);
  // for compressed: offset_end should be unfold_offset_end
  auto row_count = offset_end == -1 ? casted_input->length() : offset_end;

  bool no_default = false;
  auto dense_dim = dense_default.NumElements();
  if (dense_default.dims() == 0) {
    no_default = true;
    dense_dim = dense_default.Scalar<int32_t>();
  }
  auto value_count = casted_input->value_offset(row_count) - casted_input->value_offset(0);

  if (row_count * dense_dim == value_count) { // no empty row
    *output = input;
    return Status::OK();
  }
  // build new column with no empty row
  auto builder = std::make_shared<
      typename ListTypeToTypes<LIST_TYPE>::ListBuilder>(
      arrow::default_memory_pool(),
      std::make_shared<typename DataTypeToTypes<DATA_TYPE>::ArrowBuilderType>(
          arrow::default_memory_pool()));
  auto value_builder =
      static_cast<typename DataTypeToTypes<DATA_TYPE>::ArrowBuilderType *>(
          builder->value_builder());

  auto input_value_data_ptr =
      casted_input->values()->data()->template GetValues<DATA_TYPE>(1);

  auto default_vec = dense_default.Raw<float>();
  // mem copy for row_count times; may need optimize?
  for (size_t i = 0; i < row_count; ++i) {
    arrow::Status st_unused = builder->Append(); // some compiler give -Werr if no ret
    auto value_len =
        casted_input->value_offset(i + 1) - casted_input->value_offset(i);
    if (value_len != 0) {
      if (value_len != dense_dim) {
        return Status::InvalidArgument(
            "dense feature dim not match default dim: ", value_len,
            ", expected: ", dense_dim, ", field_name: ", field_name);
      }
      auto &&st = value_builder->AppendValues(
          input_value_data_ptr + casted_input->value_offset(i), dense_dim);
      if (!st.ok()) {
        return Status::Internal("fail to append value to arrow array: ",
                                st.ToString(), ", field_name: ", field_name);
      }
    } else {
      if (no_default) {
        return Status::InvalidArgument(
            "dense column with no default is empty, field_name: ", field_name);
      }
      // not using mem copy to void type mismatch
      for (auto i = 0; i < dense_default.NumElements(); ++i) {
        auto &&st = value_builder->Append(default_vec[i]);
        if (!st.ok()) {
          return Status::Internal("fail to append value to arrow array: ",
                                  st.ToString(), ", field_name: ", field_name);
        }
      }
    }
  }
  auto &&st = builder->Finish(output);
  if (!st.ok()) {
    return Status::Internal("fail to fill value to arrow array: ",
                            st.ToString(), ", field_name: ", field_name);
  }
  return Status::OK();
}

template <typename LIST_TYPE>
Status
FillDenseDefault(std::shared_ptr<arrow::RecordBatch> &data,
                 const std::unordered_map<std::string, Tensor> &dense_defaults,
                 std::shared_ptr<arrow::RecordBatch> *formated_data,
                 int64 offset_end = -1) {
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  auto arrow_schema = data->schema();
  for (size_t i = 0; i < arrow_schema->num_fields(); ++i) {
    auto field = arrow_schema->field(i);
    auto column = data->column(i); // with slice info
    auto iter = dense_defaults.find(field->name());
    if (iter == dense_defaults.end()) {
      arrays.emplace_back(column);
      continue;
    }

    auto casted_input = std::dynamic_pointer_cast<
        typename ListTypeToTypes<LIST_TYPE>::ListClass>(column);
    if (!casted_input) {
      return Status::InvalidArgument(
          "dense column data type not supported, column: ", field->name(),
          ", data type: ", column->type()->ToString());
    }

#define FILL_DEFAULT_FIX_WIDTH(data_type)                                      \
  case DataTypeToTypes<data_type>::ArrowType: {                                \
    RETURN_IF_ERROR(FillDefaultIfEmpty<LIST_TYPE, data_type>(                  \
        field->name(), column, iter->second, &replaced, offset_end));          \
    break;                                                                     \
  }
    std::shared_ptr<arrow::Array> replaced;
    switch (casted_input->values()->type()->id()) {
      DECLARE_BY_NUMERIC_TYPES(FILL_DEFAULT_FIX_WIDTH)
    default: {
      return Status::InvalidArgument(
          "dense column data type not supported, column: ", field->name(),
          ", data type: ", column->type()->ToString());
    }
    }
#undef FILL_DEFAULT_FIX_WIDTH
    arrays.emplace_back(replaced);
  }
  *formated_data =
      arrow::RecordBatch::Make(arrow_schema, data->num_rows(), arrays);
  return Status::OK();
}

template <typename LIST_TYPE>
Status AccumulateIndicator(const std::shared_ptr<arrow::Array> &input,
                           std::shared_ptr<arrow::Array> *output,
                           int64 offset_end) {
  auto casted_input = std::dynamic_pointer_cast<
      typename ListTypeToTypes<LIST_TYPE>::ListClass>(input);
  if (!casted_input || casted_input->value_type()->id() != arrow::Type::INT64) {
    return Status::InvalidArgument("indicator type illegal: ",
                                   casted_input->type()->ToString());
  }
  auto row_count = offset_end == -1 ? casted_input->length() : offset_end;
  auto builder =
      std::make_shared<typename ListTypeToTypes<LIST_TYPE>::ListBuilder>(
          arrow::default_memory_pool(),
          std::make_shared<arrow::Int64Builder>(arrow::default_memory_pool()));
  auto value_builder =
      static_cast<arrow::Int64Builder *>(builder->value_builder());
  int64_t acc = 0;
  auto input_value_data_ptr =
      casted_input->values()->data()->template GetValues<int64_t>(1);
  for (size_t i = 0; i < row_count; ++i) {
    builder->Append();
    auto idx_count =
        casted_input->value_offset(i + 1) - casted_input->value_offset(i);
    int64_t max_indicator = -1;
    for (size_t j = 0; j < idx_count; ++j) {
      auto indicator =
          *(input_value_data_ptr + casted_input->value_offset(i) + j);
      value_builder->Append(indicator + acc);
      max_indicator = std::max(max_indicator, indicator);
    }
    acc += max_indicator + 1;
  }
  builder->Finish(output);
  return Status::OK();
}

// FlattenCompressedData
//  data, column_schema, flatten_data: as same as it's name shows
//  offset_end: must mean rows of record_batch (before unfold) (may be sliced or not)
//  offset_ends: must mean unfold rows of each fea group
template <typename LIST_TYPE>
Status FlattenCompressedData(
    std::shared_ptr<arrow::RecordBatch> data, const ColumnSchema &column_schema,
    std::vector<std::shared_ptr<arrow::RecordBatch>> *flatten_data,
    int64 offset_end, 
    std::vector<int64> *offset_ends
) {
  auto row_count = offset_end == -1 ? data->num_rows() : offset_end; 
  auto arrow_schema = data->schema();
  std::vector<std::vector<std::shared_ptr<arrow::Field>>> new_schema;
  std::vector<int64_t> new_num_rows;
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> new_arrays;
  std::vector<std::string> references;
  for (size_t i = 0; i < arrow_schema->num_fields(); ++i) {
    auto field = arrow_schema->field(i);
    auto column = data->column(i);
    auto field_name = field->name();
    auto group_idx_iter = column_schema.group_idx_map.find(field_name);
    if (group_idx_iter ==
        column_schema.group_idx_map.end()) { // field not selected
      continue;
    }
    size_t group_idx = group_idx_iter->second;
    // deal with indicator
    if (field_name.find(kIndicatorPre) == 0) {
      group_idx = 0;
      std::shared_ptr<arrow::Array> acc_column;
      RETURN_IF_ERROR(AccumulateIndicator<LIST_TYPE>(column, &acc_column, offset_end));
      column.swap(acc_column);
    }
    if (new_schema.size() <= group_idx) {
      new_schema.resize(group_idx + 1);
      new_num_rows.resize(group_idx + 1, -1);
      if (offset_ends != nullptr){
        offset_ends->resize(group_idx + 1, -1);
      }
      new_arrays.resize(group_idx + 1);
      references.resize(group_idx + 1);
    }
    // new schema
    auto cast_type = std::dynamic_pointer_cast<LIST_TYPE>(field->type());
    if (!cast_type) {
      return Status::InvalidArgument(
          "field type illegal for compressed sample: ",
          field->type()->ToString(), ", ", field_name);
    }
    new_schema[group_idx].emplace_back(
        cast_type->value_field()->WithName(field_name));
    // new num_rows
    auto cast_array = std::dynamic_pointer_cast<
        typename ListTypeToTypes<LIST_TYPE>::ListClass>(column);

    // unfold row of this fea slice
    auto slice_value_array_length = cast_array->value_offset(row_count) - cast_array->value_offset(0); 
    // get the sliced value without outer nesting
    auto slice_value_array = cast_array->values()->Slice(cast_array->value_offset(0), slice_value_array_length); 


    if (new_num_rows[group_idx] < 0) {
      new_num_rows[group_idx] = slice_value_array_length;
      if (offset_ends != nullptr){
        offset_ends->at(group_idx) = slice_value_array_length;
      }
      references[group_idx] = field_name;
    } else {
      if (new_num_rows[group_idx] != slice_value_array_length) {
        return Status::InvalidArgument(
            "column length not compatible with others, ", references[group_idx],
            ": ", new_num_rows[group_idx], ", ", field_name, ": ",
            slice_value_array_length);
      }
    }
    // new array
    new_arrays[group_idx].emplace_back(slice_value_array);
  }
  // assemble result
  for (size_t i = 0; i < new_schema.size(); ++i) {
    auto schema = std::make_shared<arrow::Schema>(new_schema[i]);
    flatten_data->emplace_back(
        arrow::RecordBatch::Make(schema, new_num_rows[i], new_arrays[i]));
  }
  return Status::OK();
}

#define CHECK_ARROW(arrow_status)                                              \
  do {                                                                         \
    arrow::Status _s = (arrow_status);                                         \
    if (!_s.ok()) {                                                            \
      return Status::Internal(_s.ToString());                                  \
    }                                                                          \
  } while (false)

// Util to convert arrow array to tensors
class ArrowConvertTensor : public arrow::ArrayVisitor {
public:
  ArrowConvertTensor() {}
  ArrowConvertTensor(bool with_null) : with_null_(with_null) {}

  Status Convert(std::string feature, std::shared_ptr<arrow::Array> array,
                 std::vector<std::deque<Tensor>> *out_tensors,
                 std::string hash_type, int64_t hash_bucket,
                 int64 offset_end = -1, bool force_deep_copy = false) {
    feature_ = feature;
    hash_bucket_ = hash_bucket;
    hash_type_ = hash_type;
    out_tensors_ = out_tensors;
    idx_ = 0;

    offset_end_ = offset_end;
    force_deep_copy_ = force_deep_copy;
    if (offset_end == -1) {
      offset_end_ = array->length();
    }
    offset_beg_ = array->offset();
    CHECK_ARROW(array->Accept(this));
    return Status::OK();
  }

  std::deque<Tensor> *current_tensors() {
    out_tensors_->resize(idx_ + 1);
    return &(*out_tensors_)[idx_];
  }

  template <typename LIST_TYPE>
  Status ConvertDense(std::string feature, std::shared_ptr<arrow::Array> array,
                      std::vector<std::deque<Tensor>> *out_tensors,
                      int64 offset_end, bool force_deep_copy = false) {
    feature_ = feature;
    offset_end_ = offset_end == -1 ? array->length() : offset_end;
    offset_beg_ = array->offset();
    force_deep_copy_ = force_deep_copy;
    auto casted_input = std::dynamic_pointer_cast<
        typename ListTypeToTypes<LIST_TYPE>::ListClass>(array);
    if (!casted_input) {
      return Status::InvalidArgument("array not list<numeric>: ", feature_);
    }
    if (casted_input->values()->null_count() != 0) {
      return Status::InvalidArgument(
          "Not support column with null elements inside: ", feature_);
    }
    auto row_count = offset_end_;
    if (row_count == 0) {
      return Status::InvalidArgument("array is empty: ", feature_);
    }
    auto dense_dim = casted_input->value_length(0);
    auto value_count =
        casted_input->value_offset(offset_end_) - casted_input->value_offset(0);
    if (dense_dim <= 0 || row_count * dense_dim != value_count) {
      return Status::InvalidArgument(feature_,
                                     "'s array not dense: ", dense_dim, ", ",
                                     row_count, ", ", value_count);
    }
    // copy data to tensor
    Tensor tensor;
    Status st;
#define COPY_FIX_WIDTH(data_type)                                              \
  case DataTypeToTypes<data_type>::ArrowType: {                                \
    offset_beg_ = casted_input->value_offset(0);                               \
    auto values = casted_input->values()->data()->buffers[VALUE_BUFFER];       \
    std::shared_ptr<arrow::Buffer> sliced_values(new arrow::Buffer(            \
        values, offset_beg_ * SizeOfType(DataTypeToTypes<data_type>::TfType),  \
        value_count * SizeOfType(DataTypeToTypes<data_type>::TfType)));        \
    tensor = ArrowTensorBuffer::TensorFromArrowBuffer(                         \
        feature_, sliced_values, DataTypeToTypes<data_type>::TfType,           \
        TensorShape({size_t(row_count), size_t(dense_dim)}), &st,              \
        force_deep_copy_);                                                     \
    if (!st.ok()) {                                                            \
      return st;                                                               \
    }                                                                          \
    break;                                                                     \
  }
    switch (casted_input->values()->type()->id()) {
      DECLARE_BY_NUMERIC_TYPES(COPY_FIX_WIDTH)
    default: {
      return Status::InvalidArgument(
          feature_, "'s dense column data type not supported, column: ",
          ", data type: ", array->type()->ToString());
    }
    }
#undef COPY_FIX_WIDTH

    out_tensors->resize(1);
    (*out_tensors)[0].emplace_front(tensor);
    return Status::OK();
  }

  template <typename LIST_CLASS>
  arrow::Status VisitList(const LIST_CLASS &array) {
    // copy splits
    DataType data_type;
    size_t type_width;
    if (offset_end_ < 0) {
      return arrow::Status::Invalid(
          "invalid offset_end_ in VisitList, please check converter offset: ", offset_end_);
    }

    if (std::is_same<LIST_CLASS, arrow::LargeListArray>::value) {
      data_type = DataType::kInt64;
      type_width = sizeof(int64_t);
    } else {
      data_type = DataType::kInt32;
      type_width = sizeof(int32_t);
    }

    Status st;
    auto array_len = offset_end_;
    std::shared_ptr<arrow::Buffer> value_offset = array.value_offsets();
    const typename LIST_CLASS::offset_type *value_offset_data =
        reinterpret_cast<const typename LIST_CLASS::offset_type *>(
            value_offset->data());

    // copy value_offsets
    std::shared_ptr<arrow::Buffer> sliced_value_offset(
        new arrow::Buffer(value_offset, offset_beg_ * SizeOfType(data_type),
                          (array_len + 1) * SizeOfType(data_type)));
    Tensor tensor = ArrowTensorBuffer::TensorFromArrowBuffer(
        feature_, sliced_value_offset, data_type, TensorShape({static_cast<size_t>(array_len) + 1}),
        &st, force_deep_copy_);

    if (!st.ok()) {
      return arrow::Status::Invalid(st.error_message());
    }

    // update abs_offset_beg_ and relative offset_end_ for the child_data visit
    offset_beg_ = value_offset_data[offset_beg_]; // == array.value_offset(0)
    offset_end_ = offset_end_ == 0
                      ? offset_end_
                      : array.value_offset(offset_end_) - offset_beg_;

    // fix value_offsets (row splits / sequence splits)
    FixSliceValueOffsets(array, tensor, array_len);

    // save the row splits / sequence splits tensor
    current_tensors()->emplace_front(std::move(tensor));

    // call recursively, visit the sliced child data
    auto result = array.values()->Slice(offset_beg_, offset_end_)->Accept(this);

    return result;
  }

#define VISIT_LIST(LIST_TYPE)                                                  \
  virtual arrow::Status Visit(                                                 \
      const typename ListTypeToTypes<LIST_TYPE>::ListClass &array) override {  \
    return VisitList(array);                                                   \
  }

  VISIT_LIST(arrow::LargeListType)
  VISIT_LIST(arrow::ListType)
#undef VISIT_FIXED_WITH

  // output: key_val, key_splits1, key_splits2...value_val, value_splits1,
  // value_splits2
  arrow::Status Visit(const arrow::StructArray &array) override {
    // TODO: support nested null, including array or elements. currently not
    // urgent
    if (array.null_count() != 0) {
      return arrow::Status::Invalid(
          "Not support column with null elements inside: ", feature_);
    }
    auto key_array = array.GetFieldByName(kStructKey);
    auto value_array = array.GetFieldByName(kStructValue);
    std::deque<Tensor> split_tensors = *current_tensors();
    auto saved_offset_beg = offset_beg_;
    auto saved_offset_end = offset_end_;
    // convert key
    auto st = key_array->Accept(this);
    if (!st.ok())
      return st;
    // convert value
    ++idx_;
    // add duplcate splits
    for (auto iter = split_tensors.rbegin(); iter != split_tensors.rend();
         ++iter) {
      current_tensors()->emplace_front(std::move(*iter));
    }
    offset_beg_ = saved_offset_beg;
    offset_end_ = saved_offset_end;
    st = value_array->Accept(this);
    if (!st.ok())
      return st;

    return arrow::Status::OK();
  }

  template <typename ARRAY_TYPE>
  arrow::Status VisitFixedWidth(const ARRAY_TYPE &array, DataType data_type) {
    // Primitive Arrow arrays have validity and value buffers, currently
    // only arrays with null count == 0 are supported, so only need values here
    auto values = array.data()->buffers[VALUE_BUFFER]; // 不带slice信息
    if (values == NULLPTR) {
      return arrow::Status::Invalid(
          "Received an Arrow array with a NULL value buffer: ", feature_);
    }
    std::shared_ptr<arrow::Buffer> sliced_values(new arrow::Buffer(
        values, 
        static_cast<int64_t>(offset_beg_ * SizeOfType(data_type)),
        static_cast<int64_t>(offset_end_ * SizeOfType(data_type))
    ));
    // shallow copy value
    Status st;
    Tensor tensor = ArrowTensorBuffer::TensorFromArrowBuffer(
        feature_, sliced_values, data_type, TensorShape({static_cast<size_t>(offset_end_)}), &st,
        force_deep_copy_);
    tensor.SetNullBitmapFromArray(array);

    // Note: Combo Buffer for the null_bitmap:
    //  技术上来说 combo模式下null_bitmap需要单独记录slice偏移量. 也即需要算个
    //  [offset_beg_, offset_beg_ + offset_end_) 之间的null bitmap:
    //      tensor.SetNullBitmapFromArray(array, offset_beg_, offset_end_);
    //      null_bitmap() const { return data_->buffers.get(offset...); }
    //  但需要Null的行读业务不用combo. 列读Combo默认不为Null. 故这里忽略Combo对Null的额外实现
    if (!st.ok()) {
      return arrow::Status::Invalid(st.error_message());
    }

    current_tensors()->emplace_front(std::move(tensor));
    return arrow::Status::OK();
  }

#define VISIT_FIXED_WIDTH(DATA_TYPE)                                           \
  virtual arrow::Status Visit(                                                 \
      const typename DataTypeToTypes<DATA_TYPE>::ArrowArrayType &array)        \
      override {                                                               \
    if (!with_null_ && array.null_count() != 0) {                              \
      return arrow::Status::Invalid(                                           \
          "Not support column with null elements inside: ", feature_);         \
    }                                                                          \
    return VisitFixedWidth<                                                    \
        typename DataTypeToTypes<DATA_TYPE>::ArrowArrayType>(                  \
        array, DataTypeToTypes<DATA_TYPE>::TfType);                            \
  }

  DECLARE_BY_NUMERIC_TYPES(VISIT_FIXED_WIDTH)

#undef VISIT_FIXED_WITH

  arrow::Status VisitToInt8(const arrow::StringArray &array) {
    assert(offset_end_ >= 0);
    // copy splits
    DataType data_type;
    size_t type_width;
    if (arrow::Type::LARGE_STRING == array.type_id()) {
      data_type = DataType::kInt64;
      type_width = sizeof(int64_t);
    } else {
      data_type = DataType::kInt32;
      type_width = sizeof(int32_t);
    }
    Status st;
    auto array_len = offset_end_;
    std::shared_ptr<arrow::Buffer> value_offset = array.value_offsets();
    std::shared_ptr<arrow::Buffer> sliced_split(
        new arrow::Buffer(value_offset, offset_beg_ * SizeOfType(data_type),
                          static_cast<size_t>(array_len + 1) * SizeOfType(data_type)));

    Tensor split_tensor = ArrowTensorBuffer::TensorFromArrowBuffer(
        feature_, sliced_split, data_type, TensorShape({static_cast<size_t>(array_len) + 1}),
        &st, force_deep_copy_);

    if (!st.ok()) {
      return arrow::Status::Invalid(st.error_message());
    }

    // fix row splits
    FixSliceValueOffsets(array, split_tensor, array_len);

    current_tensors()->emplace_front(std::move(split_tensor));

    // copy slice data
    // update offset_beg, fix size_of_array
    offset_beg_ = array.value_offset(0);
    size_t size_of_array = array.value_offset(array_len) -
                         offset_beg_; // can deal with empty array
    if (size_of_array < 0) {
        std::string msg = "size_of_array < 0. offset_beg_: ";
        msg += offset_beg_, ", array_len: ", array_len + " for feature:" + feature_;
        return arrow::Status::Invalid(msg);
    }
    std::shared_ptr<arrow::Buffer> value_data=array.value_data();
    if (!value_data) {
      if (size_of_array > 0) {
        std::string msg = "value_data is null but size_of_array > 0, cannot convert Int8 for feature:";
        msg += feature_;
        return arrow::Status::Invalid(msg);
      }
      value_data=arrow::Buffer::FromString(""); // 返回空 buffer（size=0）
    }

    std::shared_ptr<arrow::Buffer> sliced_data(new arrow::Buffer(
        value_data, offset_beg_ * SizeOfType(DataType::kInt8),
        size_of_array * SizeOfType(DataType::kInt8)));
    Tensor value_tensor = ArrowTensorBuffer::TensorFromArrowBuffer(
        feature_, sliced_data, DataType::kInt8, TensorShape({size_of_array}),
        &st, force_deep_copy_); // no need to copy data
    if (!st.ok()) {
      return arrow::Status::Invalid(st.error_message());
    }

    current_tensors()->emplace_front(std::move(value_tensor));
    return arrow::Status::OK();
  }

  arrow::Status VisitToHash(const arrow::StringArray &array) {
    if (hash_type_ == "no_hash") {
      return VisitToInt8(array);
    }
    // size_t array_len = array.length();
    size_t array_len = offset_end_;
    Tensor tensor(DataType::kInt64, TensorShape({array_len}));
    auto tensor_vec = tensor.Raw<int64_t>();

    const char *buf;
    int length;
    for (int32_t i = 0; i < array_len; ++i) {
      buf = reinterpret_cast<const char *>(array.GetValue(i, &length));
      tensor_vec[i] = StringToHash(buf, length, hash_type_, hash_bucket_);
    }

    current_tensors()->emplace_front(std::move(tensor));
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::StringArray &array) override {

    if (!with_null_ && array.null_count() != 0) {
      return arrow::Status::Invalid(
          "Not support column with null elements inside: ", feature_);
    }

    if (!hash_type_.empty()) {
      return VisitToHash(array);
    }

    auto array_len = offset_end_;

    Tensor tensor(DataType::kString, TensorShape({static_cast<size_t>(array_len)}));
    // Note: Combo Buffer for the null_bitmap: 不实现, 原因同VisitFixedWidth
    tensor.SetNullBitmapFromArray(array);
    auto tensor_vec = tensor.Raw<std::string>();
    const char *buf;
    int length;
    for (int32_t i = 0; i < array_len; ++i) {
      // with offset
      buf = reinterpret_cast<const char *>(array.GetValue(i, &length));
      tensor_vec[i].assign(buf, length);
    }

    current_tensors()->emplace_front(std::move(tensor));

    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::BooleanArray &array) override {
    if (!with_null_ && array.null_count() != 0) {
      return arrow::Status::Invalid(
          "Not support column with null elements inside: ", feature_);
    }
    size_t array_len = array.length();
    Tensor tensor(DataType::kBool, TensorShape({array_len}));
    // Note: Combo Buffer for the null_bitmap: 不实现, 原因同VisitFixedWidth
    tensor.SetNullBitmapFromArray(array);
    bool *tensor_vec = tensor.Raw<bool>();
    for (int32_t i = 0; i < array_len; ++i) {
      tensor_vec[i] =
          array.Value(i); // bool type no need ref, copy is more simple
    }
    current_tensors()->emplace_front(std::move(tensor));
    return arrow::Status::OK();
  }

private:
  template <typename ARRAY_CLASS>
  arrow::Status FixSliceValueOffsets(ARRAY_CLASS &array, Tensor value_offset,
                                     int64 ele_len) {
    typename ARRAY_CLASS::offset_type *mutable_raw_value_offset = reinterpret_cast<typename ARRAY_CLASS::offset_type*>(value_offset.mutable_data());
    typename ARRAY_CLASS::offset_type base_offset = mutable_raw_value_offset[0];
    for (int i = 0; i < ele_len + 1; ++i) {
      mutable_raw_value_offset[i] -= base_offset;
    }
    return arrow::Status::OK();
  }

private:
  std::vector<std::deque<Tensor>> *out_tensors_;
  size_t idx_{0};
  int64_t hash_bucket_ = 0;
  std::string hash_type_;
  std::string feature_;
  bool with_null_ = true; // if (row_mode) with_null_ is true, visit null will
                          // allow (no-support struct now).
  int64 offset_end_{-1};  // relative length
  int64 offset_beg_{0};   // absolute begin
  bool force_deep_copy_{false};
};

} // end of anonymous namespace

template <typename LIST_TYPE>
class FlatColumnDataFormater : public ColumnDataFormater {
public:
  FlatColumnDataFormater() {}
  FlatColumnDataFormater(bool with_null) : ColumnDataFormater(with_null) {}
  virtual ~FlatColumnDataFormater() override {}

  Status
  InitOutputSchema(std::shared_ptr<arrow::Schema> arrow_schema,
                   const std::unordered_set<std::string> &selected_columns) {

    std::map<std::string, std::string> outputs;
    std::unordered_set<std::string> picked_columns;
    for (auto &&field : arrow_schema->fields()) {
      if (!selected_columns.empty() &&
          selected_columns.find(field->name()) == selected_columns.end()) {
        continue;
      }
      std::string field_type = field->type()->ToString();

      // change dense type
      auto iter = schema_.dense_defaults.find(field->name());
      if (iter != schema_.dense_defaults.end()) {
        if (field->type()->id() != LIST_TYPE::type_id) {
          return Status::InvalidArgument(
              "dense feature ", field->name(),
              "'s type invalid: ", field->type()->ToString());
        }
        auto list_type = std::dynamic_pointer_cast<LIST_TYPE>(field->type());
        field_type = list_type->value_type()->ToString();
      }
      outputs[field->name()] = field_type;
      picked_columns.insert(field->name());
    }
    schema_.output_schema.emplace_back(std::move(outputs));

    // check if all columns in selected_columns exists
    for (auto iter = selected_columns.begin(); iter != selected_columns.end();
         ++iter) {
      if (picked_columns.find(*iter) == picked_columns.end()) {
        return Status::InvalidArgument("selected column not in sample: ",
                                       *iter);
      }
    }
    return Status::OK();
  }

  Status
  InitSchema(std::shared_ptr<arrow::Schema> arrow_schema,
             const std::vector<std::string> &hash_features,
             const std::vector<std::string> &hash_types,
             const std::vector<int64_t> &hash_buckets,
             const std::vector<std::string> &dense_features,
             const std::vector<Tensor> &dense_defaults,
             const std::unordered_set<std::string> &selected_columns) override {
    std::lock_guard<std::mutex> l(init_mutex_);
    if (schema_inited_)
      return Status::OK();
    RETURN_IF_ERROR(ColumnDataFormater::InitSchema(
        arrow_schema, hash_features, hash_types, hash_buckets, dense_features,
        dense_defaults, selected_columns));
    RETURN_IF_ERROR(InitOutputSchema(arrow_schema, selected_columns));
    schema_inited_ = true;
    return Status::OK();
  }

  Status
  FormatSample(std::shared_ptr<arrow::RecordBatch> &data,
               std::vector<std::shared_ptr<arrow::RecordBatch>> *formated_data,
               int64 offset_end = -1,
               std::vector<int64> *offset_ends = nullptr) const override {
    formated_data->clear();
    offset_end = offset_end == -1 ? data->num_rows() : offset_end;
    std::shared_ptr<arrow::RecordBatch> formated;
    RETURN_IF_ERROR(FillDenseDefault<LIST_TYPE>(data, schema_.dense_defaults,
                                                &formated, offset_end));
    formated_data->emplace_back(formated);
    // keep consistency across comboio (either compressed or not) and dataio
    if (offset_ends != nullptr)
      offset_ends->emplace_back(offset_end);
    return Status::OK();
  }

  Status
  Convert(std::vector<std::shared_ptr<arrow::RecordBatch>> &formated_data,
          std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>>
              *output,
          std::vector<int64> offset_ends = {},
          bool force_deep_copy = false) const override {
    if (offset_ends.size() == 0){
      offset_ends.resize(formated_data.size(), -1);
    }
    auto offset_end = offset_ends[0];
    ArrowConvertTensor converter(with_null_);
    for (size_t i = 0; i < schema_.output_schema.size(); ++i) {
      auto &map = schema_.output_schema[i];
      output->resize(i + 1);
      auto &out_map = (*output)[i];
      for (auto &item : map) {
        std::shared_ptr<arrow::Array> data;
        auto &column_name = item.first;
        auto bucket_iter = schema_.hash_features.find(column_name);
        int32_t bucket_size = 0;
        std::string hash_type;
        if (bucket_iter != schema_.hash_features.end()) {
          bucket_size = bucket_iter->second.second;
          hash_type = bucket_iter->second.first;
        }
        for (auto &record_batch : formated_data) {
          data = record_batch->GetColumnByName(column_name);
          if (data)
            break;
        }
        if (!data) {
          return Status::InvalidArgument("column not found in formated data: ",
                                         column_name);
        }
        std::vector<std::deque<Tensor>> tensors;
        auto iter = schema_.dense_defaults.find(column_name);
        if (iter == schema_.dense_defaults.end()) {
          RETURN_IF_ERROR(converter.Convert(column_name, data, &tensors,
                                            hash_type, bucket_size, offset_end,
                                            force_deep_copy));
        } else {
          RETURN_IF_ERROR(converter.ConvertDense<LIST_TYPE>(
              column_name, data, &tensors, offset_end, force_deep_copy));
        }
        if (tensors.size() > 2) { // only struct lead to size = 2
          return Status::InvalidArgument(
              column_name,
              "'s column data type not supported: ", data->type()->ToString());
        }
        for (size_t k = 0; k < tensors.size(); ++k) {
          out_map[column_name].resize(k + 1);
          out_map[column_name][k].insert(out_map[column_name][k].end(),
                                         tensors[k].begin(), tensors[k].end());
        }
      }
    }
    return Status::OK();
  }

  virtual void
  GetInputColumns(std::vector<std::string> *input_columns) override {
    // float column sample only contains one feature group
    // and the name of input columns is the same as features
    for (auto iter = schema_.output_schema[0].begin();
         iter != schema_.output_schema[0].end(); ++iter) {
      input_columns->push_back(iter->first);
    }
  }
};

template <typename LIST_TYPE>
class CompressedColumnDataFormater : public ColumnDataFormater {
public:
  CompressedColumnDataFormater() {}
  CompressedColumnDataFormater(bool with_null): ColumnDataFormater(with_null) {}
  virtual ~CompressedColumnDataFormater() override {}

  Status
  InitOutputSchema(std::shared_ptr<arrow::Schema> arrow_schema,
                   const std::unordered_set<std::string> &selected_columns) {
    std::unordered_set<std::string> picked_columns;
    for (auto &&field : arrow_schema->fields()) {
      // check if field selected
      if (schema_.alias_map.find(field->name()) == schema_.alias_map.end()) {
        continue;
      }
      auto list_type = std::dynamic_pointer_cast<LIST_TYPE>(field->type());
      auto value_field = list_type->value_field();
      std::string field_type = value_field->type()->ToString();
      auto alias_name = schema_.alias_map[field->name()];
      auto iter = schema_.dense_defaults.find(field->name());

      // change dense type
      if (iter != schema_.dense_defaults.end()) {
        if (value_field->type()->id() != LIST_TYPE::type_id) {
          return Status::InvalidArgument(
              "dense feature ", field->name(),
              "'s type invalid: ", field->type()->ToString());
        }
        auto value_list_type =
            std::dynamic_pointer_cast<LIST_TYPE>(value_field->type());
        field_type = value_list_type->value_type()->ToString();
      }
      auto group_idx = schema_.group_idx_map[field->name()];
      if (schema_.output_schema.size() <= group_idx)
        schema_.output_schema.resize(group_idx + 1);
      schema_.output_schema[group_idx][alias_name] = field_type;
      picked_columns.insert(alias_name);
    }

    // modify indicators types
    for (size_t i = 1; i < schema_.output_schema.size(); ++i) {
      schema_.output_schema[i][kIndicator] = kIndicatorType;
    }

    // check if all columns in selected_columns exists
    for (auto iter = selected_columns.begin(); iter != selected_columns.end();
         ++iter) {
      if (picked_columns.find(*iter) == picked_columns.end()) {
        return Status::InvalidArgument("selected column not in sample: ",
                                       *iter);
      }
    }
    return Status::OK();
  }

  Status
  InitSchema(std::shared_ptr<arrow::Schema> arrow_schema,
             const std::vector<std::string> &hash_features,
             const std::vector<std::string> &hash_types,
             const std::vector<int64_t> &hash_buckets,
             const std::vector<std::string> &dense_features,
             const std::vector<Tensor> &dense_defaults,
             const std::unordered_set<std::string> &selected_columns) override {
    std::lock_guard<std::mutex> l(init_mutex_);
    if (schema_inited_)
      return Status::OK();
    RETURN_IF_ERROR(ColumnDataFormater::InitSchema(
        arrow_schema, hash_features, hash_types, hash_buckets, dense_features,
        dense_defaults, selected_columns));
    // get indicators
    std::unordered_set<std::string> indicators;
    for (auto &&field : arrow_schema->fields()) {
      if (field->name().find(kIndicatorPre) == 0) {
        indicators.insert(field->name());
      }
    }
    // init alias_map and group_idx_map
    std::unordered_map<std::string, Tensor> new_dense_defaults;
    for (auto &&field : arrow_schema->fields()) {
      auto &column_name = field->name();
      size_t pos = column_name.find_last_of("_");
      if (pos == std::string::npos || pos == column_name.length() - 1) {
        LOG(INFO) << "column name has no indicator suffix, skip: "
                  << column_name;
        continue;
      }
      std::string alias = column_name.substr(0, pos);
      if (column_name.find(kIndicatorPre) != 0 && !selected_columns.empty() &&
          selected_columns.find(alias) == selected_columns.end()) {
        continue;
      }
      if (kNoIndicatorIdx != column_name.substr(pos + 1)) {
        std::string indicator_name =
            kIndicatorPre + column_name.substr(pos + 1);
        if (indicators.find(indicator_name) == indicators.end()) {
          return Status::InvalidArgument("Indicator referenced by column [",
                                         column_name, "] not exists!");
        }
      }
      char *end_ptr = nullptr;
      errno = 0;
      auto indicator_idx =
          std::strtol(column_name.c_str() + pos + 1, &end_ptr, 10);
      if (errno != 0 || *end_ptr != 0) {
        return Status::InvalidArgument("column indicator suffix illegal: ",
                                       column_name);
      }
      schema_.group_idx_map[column_name] = indicator_idx;
      schema_.alias_map[column_name] = alias;
      schema_.alias_map_reversed[alias] = column_name;
      // rename dense_defaults
      if (schema_.dense_defaults.find(alias) != schema_.dense_defaults.end()) {
        new_dense_defaults[column_name] = schema_.dense_defaults[alias];
      }
    }
    schema_.dense_defaults = new_dense_defaults;
    RETURN_IF_ERROR(InitOutputSchema(arrow_schema, selected_columns));
    schema_inited_ = true;
    return Status::OK();
  }

  Status
  FormatSample(std::shared_ptr<arrow::RecordBatch> &data,
               std::vector<std::shared_ptr<arrow::RecordBatch>> *formated_data,
               int64 offset_end = -1,
               std::vector<int64> *offset_ends = nullptr) const override {
    std::vector<std::shared_ptr<arrow::RecordBatch>> flatten_data;
    RETURN_IF_ERROR(FlattenCompressedData<LIST_TYPE>(
        data, schema_, &flatten_data, offset_end, offset_ends));
    formated_data->clear();
    for (size_t i = 0; i < flatten_data.size(); ++i) {
      std::shared_ptr<arrow::RecordBatch> formated;
      int64 unfold_offset_end = offset_ends ? offset_ends->at(i) : -1;
      RETURN_IF_ERROR(
          FillDenseDefault<LIST_TYPE>(flatten_data[i], schema_.dense_defaults,
                                      &formated, unfold_offset_end));
      formated_data->emplace_back(formated);
    }
    return Status::OK();
  }

  // TODO: add sliced compressed
  Status
  Convert(std::vector<std::shared_ptr<arrow::RecordBatch>> &formated_data,
          std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>>
              *output,
          std::vector<int64> offset_ends = {},
          bool force_deep_copy = false) const override {
    if (offset_ends.size() == 0){
      offset_ends.resize(formated_data.size(), -1);
    }
    ArrowConvertTensor converter(with_null_);
    for (size_t i = 0; i < schema_.output_schema.size(); ++i) {
      auto &map = schema_.output_schema[i];
      output->resize(i + 1);
      auto &out_map = (*output)[i];
      for (auto &item : map) {
        std::shared_ptr<arrow::Array> data;
        std::string output_column_name = item.first;
        int group_idx = i;
        auto bucket_iter = schema_.hash_features.find(output_column_name);
        int32_t bucket_size = 0;
        std::string hash_type;
        if (bucket_iter != schema_.hash_features.end()) {
          bucket_size = bucket_iter->second.second;
          hash_type = bucket_iter->second.first;
        }
        std::string column_name("");
        if (output_column_name.compare(kIndicator) == 0) {
          char buf[128];
          snprintf(buf, 128, "%s_%lu", kIndicator.c_str(), i);
          column_name.assign(buf);
          group_idx = 0; 
        } else {
          column_name = schema_.alias_map_reversed.at(output_column_name);
        }
        int64 unfold_offset_end = offset_ends.at(group_idx);
        for (auto &record_batch : formated_data) {
          data = record_batch->GetColumnByName(column_name);
          if (data){
            break;
          }
        }
        if (!data) {
          return Status::InvalidArgument("column not found in formated data: ",
                                         output_column_name);
        }
        std::vector<std::deque<Tensor>> tensors;
        auto iter = schema_.dense_defaults.find(column_name);
        if (iter == schema_.dense_defaults.end()) {
          RETURN_IF_ERROR(
              (converter.Convert(column_name, data, &tensors, hash_type,
                                bucket_size, unfold_offset_end, force_deep_copy)));
              // (converter.Convert(column_name, data, &tensors, hash_type,
              //                    bucket_size, -1, force_deep_copy)));
        } else {
          RETURN_IF_ERROR(
              (converter.ConvertDense<LIST_TYPE>(column_name, data, &tensors,
                                              unfold_offset_end, force_deep_copy)));
              // (converter.ConvertDense<LIST_TYPE>(column_name, data, &tensors,
              //                                    -1, force_deep_copy)));
        }
        if (tensors.size() > 2) { // only struct lead to size = 2
          return Status::InvalidArgument(
              "column data type not supported: ", data->type()->ToString(),
              ", feature: ", output_column_name);
        }
        for (size_t k = 0; k < tensors.size(); ++k) {
          out_map[output_column_name].resize(k + 1);
          out_map[output_column_name][k].insert(
              out_map[output_column_name][k].end(), tensors[k].begin(),
              tensors[k].end());
        }
      }
    }
    return Status::OK();
  }

  virtual void
  GetInputColumns(std::vector<std::string> *input_columns) override {
    for (auto iter = schema_.alias_map.begin(); iter != schema_.alias_map.end();
         ++iter) {
      input_columns->push_back(iter->first);
    }
    std::sort(input_columns->begin(), input_columns->end());
  }
};

Status ColumnDataFormater::InitSchema(
    std::shared_ptr<arrow::Schema> arrow_schema,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_features,
    const std::vector<Tensor> &dense_defaults,
    const std::unordered_set<std::string> &selected_columns) {
  if (schema_inited_)
    return Status::OK();
  for (auto i = 0; i < hash_features.size(); ++i) {
    schema_.hash_features.insert(
        {hash_features[i], std::make_pair(hash_types[i], hash_buckets[i])});
  }
  if (dense_features.size() != dense_defaults.size()) {
    return Status::InvalidArgument(
        "dense_features size not match dense_defaults size: ",
        dense_features.size(), ", ", dense_defaults.size());
  }
  for (size_t i = 0; i < dense_features.size(); ++i) {
    if (selected_columns.size() != 0 &&
        selected_columns.find(dense_features[i]) == selected_columns.end()) {
      LOG(INFO) << "dense feature not selected, ignore default: "
                << dense_features[i];
      continue;
    }
    if (schema_.dense_defaults.find(dense_features[i]) !=
        schema_.dense_defaults.end()) {
      return Status::InvalidArgument("duplicated dense feature config: ",
                                     dense_features[i]);
    }
    if (dense_defaults[i].Type() != DataType::kFloat &&
        dense_defaults[i].Type() != DataType::kInt32) {
      return Status::InvalidArgument(
          "dense default type not supported: ", dense_defaults[i].Type(),
          ", feature: ", dense_features[i]);
    }
    if (dense_defaults[i].dims() > 1) {
      return Status::InvalidArgument(
          "dense dim not supported: ", dense_defaults[i].dims(),
          ", feature: ", dense_features[i]);
    }
    schema_.dense_defaults[dense_features[i]] = dense_defaults[i];
  }
  return Status::OK();
}

Status ColumnDataFormater::GetOutputSchema(
    std::vector<std::map<std::string, std::string>> *schema) {
  std::lock_guard<std::mutex> l(init_mutex_);
  if (!schema_inited_) {
    return Status::Internal("schema not inited");
  }
  schema->reserve(schema_.output_schema.size());
  for (auto &&it : schema_.output_schema) {
    schema->emplace_back();
    for (auto &&inter_it : it) {
      schema->back().emplace(inter_it.first, inter_it.second);
    }
  }
  ReplaceLargeList(schema);
  return Status::OK();
}

Status ColumnDataFormater::FlatConvert(
    std::vector<std::shared_ptr<arrow::RecordBatch>> &formated_data,
    std::vector<Tensor> *output, std::vector<int64> offset_ends,
    bool force_deep_copy) const {
  // (压缩.列名)vector<map>           : [ _0非压缩表{ "column_name": content },
  // _1压缩表{ "column_name": content }, ... ]
  // (权重.偏移)vector<vector<Tensor>>: { value{ value, offset }, _weight{value,
  // offset } }
  std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>>
      conv_output;
  auto st = Convert(formated_data, &conv_output);
  if (!st.ok()) {
    return st;
  }
  for (auto &map : conv_output) {
    for (auto &item : map) {
      size_t output_prev_end = output->size();
      for (auto &vec : item.second) {
        output->insert(output->end(), vec.begin(), vec.end());
      }
      // TODO: 找个表确认权重/压缩等特性如何共用一个item.first(col_name). commonio先不考虑这些东西
      schema_.flatconvert_tensor_spliter[item.first] = {output_prev_end, output->size()};
    }
  }
  return Status::OK();
}

Status ColumnDataFormater::FlatConvert(
    std::vector<std::shared_ptr<arrow::RecordBatch>> &formated_data,
    std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>>
        *conv_output,
    std::vector<int64> offset_ends, bool force_deep_copy) const {
  auto st = Convert(formated_data, conv_output, offset_ends, force_deep_copy);
  return st;
}

std::unique_ptr<ColumnDataFormater>
ColumnDataFormater::GetColumnDataFormater(bool is_compressed,
                                          bool is_large_list) {
  return GetColumnDataFormater(is_compressed, is_large_list, false);
}

std::unique_ptr<ColumnDataFormater> ColumnDataFormater::GetColumnDataFormater(
    bool is_compressed, bool is_large_list, bool row_mode_with_null) {
  if (is_large_list) {
    if (is_compressed) {
      auto formater = new CompressedColumnDataFormater<arrow::LargeListType>(
          row_mode_with_null);
      return std::unique_ptr<ColumnDataFormater>(formater);
    } else {
      auto formater =
          new FlatColumnDataFormater<arrow::LargeListType>(row_mode_with_null);
      return std::unique_ptr<ColumnDataFormater>(formater);
    }
  } else {
    if (is_compressed) {
      auto formater =
          new CompressedColumnDataFormater<arrow::ListType>(row_mode_with_null);
      return std::unique_ptr<ColumnDataFormater>(formater);
    } else {
      auto formater =
          new FlatColumnDataFormater<arrow::ListType>(row_mode_with_null);
      return std::unique_ptr<ColumnDataFormater>(formater);
    }
  }
}

void ColumnDataFormater::LogDebugString(
    std::shared_ptr<arrow::RecordBatch> record_batch) const {
  if (!record_batch) {
    LOG(INFO) << "RecordBatch is empty";
  } else {
    LOG(INFO) << "colunm schema: " << record_batch->schema()->ToString()
              << std::endl;
    LOG(INFO) << "rows: " << record_batch->num_rows() << std::endl;
    for (size_t i = 0; i < record_batch->schema()->num_fields(); ++i) {
      LOG(INFO) << record_batch->schema()->field(i)->name() << ": "
                << record_batch->column(i)->ToString() << std::endl;
    }
  }
}

std::string ColumnDataFormater::DebugString(
    std::shared_ptr<arrow::RecordBatch> record_batch) const {
  std::stringstream ss;
  if (!record_batch) {
    ss << "RecordBatch is empty";
  } else {
    ss << "colunm schema: " << record_batch->schema()->ToString() << std::endl;
    ss << "rows: " << record_batch->num_rows() << std::endl;
    for (size_t i = 0; i < record_batch->schema()->num_fields(); ++i) {
      ss << record_batch->schema()->field(i)->name() << ": "
         << record_batch->column(i)->ToString() << std::endl;
    }
  }
  return ss.str();
}


std::string ColumnSchema::DebugString() const {
  std::ostringstream oss;
  oss << "ColumnSchema {\n";

  // 1. output_schema: vector<map<string, string>>
  oss << "  output_schema: [\n";
  for (size_t i = 0; i < output_schema.size(); ++i) {
    oss << "    [index=" << i << "] {";
    const auto &m = output_schema[i];
    bool first = true;
    for (const auto &kv : m) {
      if (!first)
        oss << ", ";
      oss << "\"" << kv.first << "\": \"" << kv.second << "\"";
      first = false;
    }
    oss << "}\n";
  }
  oss << "  ],\n";

  // 2. flatconvert_tensor_spliter: unordered_map<string, pair<size_t, size_t>>
  oss << "  flatconvert_tensor_spliter: {\n";
  for (const auto &kv : flatconvert_tensor_spliter) {
    oss << "    \"" << kv.first << "\": (" << kv.second.first << ", "
        << kv.second.second << "),\n";
  }
  oss << "  },\n";

  // 3. hash_features: unordered_map<string, pair<string, int32_t>>
  oss << "  hash_features: {\n";
  for (const auto &kv : hash_features) {
    oss << "    \"" << kv.first << "\": (\"" << kv.second.first << "\", "
        << kv.second.second << "),\n";
  }
  oss << "  },\n";

  // 4. dense_defaults: unordered_map<string, Tensor>
  oss << "  dense_defaults: {\n";
  for (const auto &kv : dense_defaults) {
    oss << "    \"" << kv.first << "\": " << kv.second.DebugString() << ",\n";
  }
  oss << "  },\n";

  // 5. alias_map: unordered_map<string, string>
  oss << "  alias_map: {\n";
  for (const auto &kv : alias_map) {
    oss << "    \"" << kv.first << "\": \"" << kv.second << "\",\n";
  }
  oss << "  },\n";

  // 6. alias_map_reversed: unordered_map<string, string>
  oss << "  alias_map_reversed: {\n";
  for (const auto &kv : alias_map_reversed) {
    oss << "    \"" << kv.first << "\": \"" << kv.second << "\",\n";
  }
  oss << "  },\n";

  // 7. group_idx_map: unordered_map<string, size_t>
  oss << "  group_idx_map: {\n";
  for (const auto &kv : group_idx_map) {
    oss << "    \"" << kv.first << "\": " << kv.second << ",\n";
  }
  oss << "  }\n";

  oss << "}";
  return oss.str();
}

#undef DECLARE_BY_NUMERIC_TYPES

} // namespace dataset
} // namespace column
