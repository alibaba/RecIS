#include "column-io/dataset_impl/odps_open_storage_row_dataset.h"
#include "column-io/open_storage/common-util/status.h"
#include "arrow/array.h"
#include "arrow/array/array_binary.h"
#include "arrow/array/array_nested.h"
#include "arrow/array/array_primitive.h"
#include "arrow/record_batch.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "absl/log/log.h"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace column {
namespace dataset {
namespace {
using OpenStorageStatus = ::apsara::odps::algo::commonio::Status;
using OpenStorageReader =
    ::apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader;
}  // namespace

// Arrow -> py::object conversion (recursive, supports nested types)
py::object OdpsOpenStorageRowDataset::ArrowCellToPyObject(
    const arrow::Array& array, int64_t index) {
  if (array.IsNull(index)) {
    return py::none();
  }

  switch (array.type_id()) {
    // ---- Boolean ----
    case arrow::Type::BOOL: {
      const auto& typed = static_cast<const arrow::BooleanArray&>(array);
      return py::bool_(typed.Value(index));
    }

    // ---- Integer types -> Python int ----
    case arrow::Type::INT8: {
      const auto& typed = static_cast<const arrow::Int8Array&>(array);
      return py::int_(typed.Value(index));
    }
    case arrow::Type::INT16: {
      const auto& typed = static_cast<const arrow::Int16Array&>(array);
      return py::int_(typed.Value(index));
    }
    case arrow::Type::INT32: {
      const auto& typed = static_cast<const arrow::Int32Array&>(array);
      return py::int_(typed.Value(index));
    }
    case arrow::Type::INT64: {
      const auto& typed = static_cast<const arrow::Int64Array&>(array);
      return py::int_(typed.Value(index));
    }
    case arrow::Type::UINT8: {
      const auto& typed = static_cast<const arrow::UInt8Array&>(array);
      return py::int_(typed.Value(index));
    }
    case arrow::Type::UINT16: {
      const auto& typed = static_cast<const arrow::UInt16Array&>(array);
      return py::int_(typed.Value(index));
    }
    case arrow::Type::UINT32: {
      const auto& typed = static_cast<const arrow::UInt32Array&>(array);
      return py::int_(typed.Value(index));
    }
    case arrow::Type::UINT64: {
      const auto& typed = static_cast<const arrow::UInt64Array&>(array);
      return py::int_(typed.Value(index));
    }

    // ---- Float types -> Python float ----
    case arrow::Type::FLOAT: {
      const auto& typed = static_cast<const arrow::FloatArray&>(array);
      return py::float_(static_cast<double>(typed.Value(index)));
    }
    case arrow::Type::DOUBLE: {
      const auto& typed = static_cast<const arrow::DoubleArray&>(array);
      return py::float_(typed.Value(index));
    }

    // ---- Timestamp (ms or ns depending on unit metadata) -> Python int ----
    case arrow::Type::TIMESTAMP: {
      const auto& typed = static_cast<const arrow::TimestampArray&>(array);
      const auto* type = static_cast<const arrow::TimestampType*>(array.type().get());
      int64_t value = typed.Value(index);
      
      // 根据单位转换到目标单位（tunnell中标单位是秒）
      switch (type->unit()) {
        case arrow::TimeUnit::NANO:   return py::int_(value / 1000000000);
        case arrow::TimeUnit::MICRO:  return py::int_(value / 1000000);
        case arrow::TimeUnit::MILLI:  return py::int_(value / 1000);
        case arrow::TimeUnit::SECOND: return py::int_(value);
        default: {
          LOG(WARNING) << "Unknown timestamp unit: " << static_cast<int>(type->unit());
          break;
        }
      }
      break;
    }
    // ---- String -> Python str ----
    case arrow::Type::STRING: {
      const auto& typed = static_cast<const arrow::StringArray&>(array);
      auto view = typed.GetView(index);
      return py::str(view.data(), view.size());
    }
    case arrow::Type::LARGE_STRING: {
      const auto& typed = static_cast<const arrow::LargeStringArray&>(array);
      auto view = typed.GetView(index);
      return py::str(view.data(), view.size());
    }

    // ---- Binary -> Python bytes ----
    case arrow::Type::BINARY: {
      const auto& typed = static_cast<const arrow::BinaryArray&>(array);
      auto view = typed.GetView(index);
      return py::bytes(view.data(), view.size());
    }
    case arrow::Type::LARGE_BINARY: {
      const auto& typed = static_cast<const arrow::LargeBinaryArray&>(array);
      auto view = typed.GetView(index);
      return py::bytes(view.data(), view.size());
    }

    // ---- List / LargeList -> Python list (recursive) ----
    case arrow::Type::LIST: {
      const auto& typed = static_cast<const arrow::ListArray&>(array);
      const int32_t offset = typed.value_offset(index);
      const int32_t length = typed.value_length(index);
      const auto& values = *typed.values();
      py::list out(length);
      for (int32_t i = 0; i < length; ++i) {
        out[i] = ArrowCellToPyObject(values, offset + i);
      }
      return std::move(out);
    }
    case arrow::Type::LARGE_LIST: {
      const auto& typed = static_cast<const arrow::LargeListArray&>(array);
      const int64_t offset = typed.value_offset(index);
      const int64_t length = typed.value_length(index);
      const auto& values = *typed.values();
      py::list out(length);
      for (int64_t i = 0; i < length; ++i) {
        out[i] = ArrowCellToPyObject(values, offset + i);
      }
      return std::move(out);
    }

    // ---- Struct -> Python dict (field_name -> value) ----
    case arrow::Type::STRUCT: {
      const auto& typed = static_cast<const arrow::StructArray&>(array);
      const auto& struct_type = typed.type();
      const int num_fields = struct_type->num_fields();
      py::dict out;
      for (int f = 0; f < num_fields; ++f) {
        const auto& field_name = struct_type->field(f)->name();
        const auto& child_array = *typed.field(f);
        out[py::str(field_name)] = ArrowCellToPyObject(child_array, index);
      }
      return std::move(out);
    }

    // ---- Map -> Python dict (key -> value, recursive) ----
    case arrow::Type::MAP: {
      const auto& typed = static_cast<const arrow::MapArray&>(array);
      const int32_t offset = typed.value_offset(index);
      const int32_t length = typed.value_length(index);
      const auto& keys_array = *typed.keys();
      const auto& items_array = *typed.items();
      py::dict out;
      for (int32_t i = 0; i < length; ++i) {
        py::object key = ArrowCellToPyObject(keys_array, offset + i);
        py::object val = ArrowCellToPyObject(items_array, offset + i);
        out[key] = val;
      }
      return std::move(out);
    }

    // ---- Fixed-size list -> Python list ----
    case arrow::Type::FIXED_SIZE_LIST: {
      const auto& typed =
          static_cast<const arrow::FixedSizeListArray&>(array);
      const int32_t list_size = typed.list_type()->list_size();
      const int64_t offset = typed.value_offset(index);
      const auto& values = *typed.values();
      py::list out(list_size);
      for (int32_t i = 0; i < list_size; ++i) {
        out[i] = ArrowCellToPyObject(values, offset + i);
      }
      return std::move(out);
    }

    // ---- Fallback: return type name as string with warning ----
    default: {
      LOG(WARNING) << "OdpsOpenStorageRowDataset: unsupported arrow type "
                   << array.type()->ToString()
                   << " (id=" << static_cast<int>(array.type_id())
                   << "), returning type-name string as placeholder";
      return py::str(array.type()->ToString());
    }
  }
}

// ConvertBatch: RecordBatch -> py::list[py::list[py::object]]
py::list OdpsOpenStorageRowDataset::ConvertBatch(
    const std::shared_ptr<arrow::RecordBatch>& batch) {
  if (!batch || batch->num_rows() == 0) {
    return py::list();
  }

  const int64_t num_rows = batch->num_rows();
  const auto& schema = batch->schema();

  // Pre-resolve column indices once per batch.
  std::vector<int> col_indices;
  col_indices.reserve(selected_columns_.size());
  for (const auto& name : selected_columns_) {
    col_indices.push_back(schema->GetFieldIndex(name));
  }

  py::list rows(num_rows);
  for (int64_t r = 0; r < num_rows; ++r) {
    py::tuple row(col_indices.size());
    for (size_t c = 0; c < col_indices.size(); ++c) {
      int idx = col_indices[c];
      if (idx < 0) {
        row[c] = py::none();
      } else {
        row[c] = ArrowCellToPyObject(*batch->column(idx), r);
      }
    }
    rows[r] = std::move(row);
  }
  return rows;
}

// FetchBatch: pure C++ I/O (no Python objects created)
std::shared_ptr<arrow::RecordBatch> OdpsOpenStorageRowDataset::FetchBatch() {
  std::lock_guard<std::mutex> lock(mu_);
  for (;;) {
    if (reach_end_) {
      return nullptr;
    }
    if (reader_ == nullptr) {
      if (file_cur_ >= static_cast<int64_t>(paths_.size())) {
        reach_end_ = true;
        return nullptr;
      }
      OpenCurrentReader();
    }

    std::shared_ptr<arrow::RecordBatch> batch;
    OpenStorageStatus status = reader_->ReadBatch(batch);

    if (status.Ok()) {
      if (batch == nullptr || batch->num_rows() == 0) {
        // LOG(WARNING) << reader_name_ << ": path[" << file_cur_
        //           << "] returned empty batch, advancing to next path";
        reader_.reset();
        ++file_cur_;
        continue;
      }
      return batch;
    }

    if (status.GetCode() == OpenStorageStatus::kOutOfRange) {
      // LOG(WARNING) << reader_name_ << ": path[" << file_cur_
      //           << "] exhausted (OutOfRange), advancing to next path";
      reader_.reset();
      ++file_cur_;
      continue;
    }

    reader_.reset();
    throw std::runtime_error(
        "OdpsOpenStorageRowDataset::FetchBatch failed on path [" +
        (file_cur_ < static_cast<int64_t>(paths_.size())
             ? paths_[file_cur_]
             : std::string("<eof>")) +
        "]: " + status.GetMsg());
  }
}

// ReadBatch: convenience combining Fetch + Convert
py::list OdpsOpenStorageRowDataset::ReadBatch() {
  auto batch = FetchBatch();
  if (!batch) {
    return py::list();
  }
  return ConvertBatch(batch);
}

std::shared_ptr<OdpsOpenStorageRowDataset> OdpsOpenStorageRowDataset::Make(
    const std::vector<std::string>& paths,
    const std::vector<std::string>& selected_columns,
    int64_t batch_size,
    const std::string& reader_name) {
  if (paths.empty()) {
    throw std::invalid_argument(
        "OdpsOpenStorageRowDataset::Make: paths must not be empty");
  }
  if (batch_size <= 0) {
    throw std::invalid_argument(
        "OdpsOpenStorageRowDataset::Make: batch_size must be > 0");
  }
  auto self = std::make_shared<OdpsOpenStorageRowDataset>();
  self->paths_ = paths;
  self->selected_columns_ = selected_columns;
  self->batch_size_ = batch_size;
  self->reader_name_ =
      reader_name.empty() ? std::string("OdpsOpenStorageRowDataset")
                          : reader_name;
  // LOG(INFO) << "OdpsOpenStorageRowDataset::Make: reader_name="
  //           << self->reader_name_ << ", paths=" << paths.size()
  //           << ", selected_columns=" << selected_columns.size()
  //           << ", batch_size=" << batch_size;
  return self;
}

int64_t OdpsOpenStorageRowDataset::GetTableSize(const std::string& path) {
  uint64_t table_size = 0;
  auto status = OpenStorageReader::GetTableSize(path, table_size);
  if (!status.Ok()) {
    throw std::runtime_error(
        "OdpsOpenStorageRowDataset::GetTableSize failed for path [" + path +
        "]: " + status.GetMsg());
  }
  return static_cast<int64_t>(table_size);
}

void OdpsOpenStorageRowDataset::OpenCurrentReader() {
  if (file_cur_ < 0 ||
      file_cur_ >= static_cast<int64_t>(paths_.size())) {
    throw std::runtime_error(
        "OdpsOpenStorageRowDataset::OpenCurrentReader: file_cur_ out of range");
  }
  // LOG(INFO) << reader_name_ << ": opening path[" << file_cur_ << "]="
  //           << paths_[file_cur_]
  //           << (begin_cur_ >= 0
  //                   ? ", seek_to=" + std::to_string(begin_cur_)
  //                   : "");
  const int max_batch_rows =
      static_cast<int>(std::min<int64_t>(batch_size_, 1024));
  std::shared_ptr<OpenStorageReader> reader;
  auto status = OpenStorageReader::CreateReader(
      paths_[file_cur_], max_batch_rows, reader_name_, reader,
      selected_columns_);
  if (!status.Ok() || reader == nullptr) {
    throw std::runtime_error(
        "OdpsOpenStorageRowDataset: CreateReader failed for path [" +
        paths_[file_cur_] + "]: " + status.GetMsg());
  }
  if (begin_cur_ >= 0) {
    auto seek_status = reader->Seek(static_cast<size_t>(begin_cur_));
    if (!seek_status.Ok()) {
      throw std::runtime_error(
          "OdpsOpenStorageRowDataset: Seek to " + std::to_string(begin_cur_) +
          " failed for path [" + paths_[file_cur_] +
          "]: " + seek_status.GetMsg());
    }
    begin_cur_ = -1;
  }
  reader_ = std::move(reader);
}

void OdpsOpenStorageRowDataset::Seek(size_t pos) {
  std::lock_guard<std::mutex> lock(mu_);
  if (file_cur_ >= static_cast<int64_t>(paths_.size())) {
    throw std::runtime_error(
        "OdpsOpenStorageRowDataset::Seek: no active path");
  }
  LOG(INFO) << reader_name_ << ": Seek to pos=" << pos
            << " on path[" << file_cur_ << "]";
  reader_.reset();
  begin_cur_ = static_cast<int64_t>(pos);
  reach_end_ = false;
  OpenCurrentReader();
}

size_t OdpsOpenStorageRowDataset::Tell() {
  std::lock_guard<std::mutex> lock(mu_);
  if (reader_ == nullptr) {
    return begin_cur_ >= 0 ? static_cast<size_t>(begin_cur_) : 0;
  }
  return reader_->Tell();
}

std::string OdpsOpenStorageRowDataset::SaveState() {
  std::lock_guard<std::mutex> lock(mu_);
  int64_t snapshot_begin = begin_cur_;
  if (reader_ != nullptr) {
    snapshot_begin = static_cast<int64_t>(reader_->Tell());
  }
  std::string saved = std::to_string(file_cur_) + ":" + std::to_string(snapshot_begin);
  // LOG(INFO) << reader_name_ << ": SaveState -> \"" << saved << "\"";
  return saved;
}

void OdpsOpenStorageRowDataset::RestoreState(const std::string& state) {
  std::lock_guard<std::mutex> lock(mu_);
  const std::size_t sep = state.find(':');
  if (sep == std::string::npos) {
    throw std::invalid_argument(
        "OdpsOpenStorageRowDataset::RestoreState: invalid state string [" +
        state + "]");
  }
  int64_t file_cur = 0;
  int64_t begin_cur = -1;
  try {
    file_cur = std::stoll(state.substr(0, sep));
    begin_cur = std::stoll(state.substr(sep + 1));
  } catch (const std::exception& e) {
    throw std::invalid_argument(
        std::string("OdpsOpenStorageRowDataset::RestoreState: parse error: ") +
        e.what());
  }
  reader_.reset();
  file_cur_ = file_cur;
  begin_cur_ = begin_cur;
  reach_end_ = file_cur_ >= static_cast<int64_t>(paths_.size());
  // LOG(INFO) << reader_name_ << ": RestoreState from \"" << state
  //           << "\" -> file_cur=" << file_cur_
  //           << ", begin_cur=" << begin_cur_
  //           << ", reach_end=" << reach_end_;
}

}  // namespace dataset
}  // namespace column
