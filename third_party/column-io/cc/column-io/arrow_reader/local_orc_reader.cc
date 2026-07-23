#include "column-io/arrow_reader/local_orc_reader.h"

#include <sstream>
#include <iostream>
#include <string>
#include <vector>

#include "arrow/record_batch.h"
#include "arrow/type.h"
#include "absl/log/log.h"

namespace column {
namespace BatchReader {

namespace {
// 将 arrow::Status 转换为 column::Status
column::Status ArrowStatusConvert(const arrow::Status &arrow_st) {
  if (arrow_st.ok()) {
    return column::Status::OK();
  }
  if (arrow_st.code() == arrow::StatusCode::RError) {
    return column::Status::OutOfRange();
  }
  return column::Status::Internal(arrow_st.message());
}

// 将 vector<string> 拼接为逗号分隔的字符串
std::string JoinColumns(const std::vector<std::string> &columns) {
  std::ostringstream joined;
  for (size_t i = 0; i < columns.size(); ++i) {
    if (i > 0) {
      joined << ",";
    }
    joined << columns[i];
  }
  return joined.str();
}
} // namespace

LocalOrcReaderImpl::LocalOrcReaderImpl(CAPI_LOCAL_ORC_ReadCtx *reader_ctx,
                                       column::local_orc::LocalOrcLib *lib)
    : reader_ctx_(reader_ctx), lib_(lib) {}

LocalOrcReaderImpl::~LocalOrcReaderImpl() {
  if (reader_ctx_ != nullptr && lib_ != nullptr) {
    lib_->DeleteReaderCtx(reader_ctx_);
    reader_ctx_ = nullptr;
  }
}

column::Status LocalOrcReaderImpl::Create(
    const std::string &file_path,
    const std::vector<std::string> &input_columns,
    AbstractReaderPtr *out_reader) {
  // 获取 dlwrapper 单例
  column::local_orc::LocalOrcLib *lib = nullptr;
  arrow::Status lib_st = column::local_orc::LocalOrcLib::GetLib(lib);
  if (!lib_st.ok()) {
    return column::Status::Internal("Failed to load LocalOrcLib: ", lib_st.message());
  }

  // 将列名拼接为逗号分隔的字符串，供 CAPI 使用
  std::string columns_str = JoinColumns(input_columns);

  // 通过 dlwrapper 的 MakeReader 创建底层 reader 句柄
  CAPI_LOCAL_ORC_ReadCtx *reader_ctx = lib->MakeReader(
      file_path.c_str(), static_cast<int>(file_path.size()),
      columns_str.c_str(), static_cast<int>(columns_str.size()));

  if (reader_ctx == nullptr) {
    return column::Status::Internal(
        "Failed to create LocalOrc reader for file: ", file_path);
  }

  out_reader->reset(new LocalOrcReaderImpl(reader_ctx, lib));
  return column::Status::OK();
}

column::Status LocalOrcReaderImpl::ReadBatch(
    std::shared_ptr<arrow::RecordBatch> *data) {
  if (reader_ctx_ == nullptr) {
    return column::Status::Internal("LocalOrc reader is not initialized");
  }

  arrow::Status arrow_st;
  lib_->ReadBatch(reader_ctx_, reinterpret_cast<void *>(data),
                  reinterpret_cast<void *>(&arrow_st));

  if (!arrow_st.ok()) {
    column::Status st = ArrowStatusConvert(arrow_st);
    return ArrowStatusConvert(arrow_st);
  }
  return column::Status::OK();
}

column::Status LocalOrcReaderImpl::ReadSchema(
    std::shared_ptr<arrow::Schema> *schema) {
  if (reader_ctx_ == nullptr) {
    return column::Status::Internal("LocalOrc reader is not initialized");
  }

  lib_->ReadSchema(reader_ctx_, reinterpret_cast<void *>(schema));
  return column::Status::OK();
}

column::Status LocalOrcReaderImpl::Seek(int64_t offset) {
  if (reader_ctx_ == nullptr) {
    return column::Status::Internal("LocalOrc reader is not initialized");
  }

  arrow::Status arrow_st;
  lib_->Seek(reader_ctx_, offset, reinterpret_cast<void *>(&arrow_st));

  if (!arrow_st.ok()) {
    return ArrowStatusConvert(arrow_st);
  }
  return column::Status::OK();
}

int64_t LocalOrcReaderImpl::Tell() const {
  if (reader_ctx_ == nullptr) {
    return -1;
  }
  lib_->Tell(reader_ctx_);
  return 0;
}

} // namespace BatchReader
} // namespace column
