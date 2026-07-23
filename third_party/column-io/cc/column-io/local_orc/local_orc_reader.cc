#include "column-io/local_orc/local_orc_reader.h"
#include "arrow/buffer.h"
#include "arrow/io/api.h"
#include "arrow/adapters/orc/adapter.h"
#include "absl/log/log.h"

namespace column {
namespace local_orc {

LocalOrcReader::LocalOrcReader(const string file_path, const std::vector<string> &selected_columns)
                                : file_path_(file_path), selected_columns_(selected_columns), index_(0) {};


arrow::Status LocalOrcReader::ReadBatch(shared_ptr<RecordBatch> *data) {
  if (reader_ == nullptr) {
    return arrow::Status(arrow::StatusCode::Invalid, "orc_reader_ is null, cannot ReadBatch");
  }
  if (data == nullptr) {
    return arrow::Status(arrow::StatusCode::Invalid, "Null RecordBatch Ptr");
  }
  auto st = reader_->ReadNext(data);
  if (!st.ok()) {
    return arrow::Status(arrow::StatusCode::Invalid, st.message());
  }
  if (*data == nullptr) {
    return arrow::Status(arrow::StatusCode::RError, "OutOfRange");
  }
  return arrow::Status::OK();
}

arrow::Status LocalOrcReader::MakeReader(const string file_path,
                          const vector<string> &input_columns,
                          unique_ptr<LocalOrcReader> *out_reader_ptr) {
  out_reader_ptr->reset(new LocalOrcReader(file_path, input_columns));
  arrow::Status st = (*out_reader_ptr)->MakeReaderInternal();
  return st;
}

arrow::Status LocalOrcReader::MakeReaderInternal() {
  auto file = arrow::io::ReadableFile::Open(file_path_);
  if (!file.ok()) {
    return arrow::Status(arrow::StatusCode::Invalid, file.status().message());
  }
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  arrow::Result<unique_ptr<ORCFileReader>> orc_reader =
      ORCFileReader::Open(file.ValueOrDie(), pool);
  if (!orc_reader.ok()) {
    return arrow::Status(arrow::StatusCode::Invalid, orc_reader.status().message());
  }
  orc_reader_ = std::move(orc_reader).ValueOrDie();

  // InitSchema();
  auto schema = orc_reader_->ReadSchema();
  if (!schema.ok()) {
    return arrow::Status(arrow::StatusCode::Invalid, schema.status().message());
  }
  schema_ = schema.ValueOrDie();

  auto reader = orc_reader_->GetRecordBatchReader(1024, selected_columns_);
  if (!reader.ok()) {
    return arrow::Status(arrow::StatusCode::Invalid, reader.status().message());
  }
  reader_ = std::move(reader).ValueOrDie();
  return arrow::Status::OK();
}

arrow::Status LocalOrcReader::Seek(int64_t index) {
  while (index != index_) {
    shared_ptr<RecordBatch> rb;
    auto st = ReadBatch(&rb);
    if (!st.ok()) {
      return st;
    }
    index_++;
  }
  return arrow::Status::OK();
}

shared_ptr<arrow::Schema> LocalOrcReader::ReadSchema() const {
    return schema_;
}

int64_t LocalOrcReader::Tell() const { return index_; }


} // namespace local_orc
} // namespace column
