#include "column-io/local_orc/local_orc_capi.h"

#include <stdlib.h>
#include <string.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "column-io/local_orc/local_orc_reader.h"

struct CAPI_LOCAL_ORC_ReadCtx {
  std::unique_ptr<column::local_orc::LocalOrcReader> reader;
};

// 辅助函数：将逗号分隔的列名list字符串拆分为 vector<string>
static std::vector<std::string> SplitColumns(const char *columns, size_t length) {
  std::vector<std::string> result;
  std::string input(columns, length);
  std::istringstream stream(input);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      result.push_back(token);
    }
  }
  return result;
}

extern "C" {

__attribute__((visibility("default"))) CAPI_LOCAL_ORC_ReadCtx *
CAPI_LOCAL_ORC_Method_MakeReader(const char *file_path, size_t file_path_len,
                                const char *columns, size_t columns_len) {
  std::string path(file_path, file_path_len);
  std::vector<std::string> selected_columns = SplitColumns(columns, columns_len);

  CAPI_LOCAL_ORC_ReadCtx* ctx = new CAPI_LOCAL_ORC_ReadCtx;
  auto status = column::local_orc::LocalOrcReader::MakeReader(
      path, selected_columns, &ctx->reader);
  if (!status.ok()) {
    delete ctx;
    return nullptr;
  }
  return ctx;
}

__attribute__((visibility("default"))) void
CAPI_LOCAL_ORC_Method_ReadBatch(CAPI_LOCAL_ORC_ReadCtx *ctx, void *batch, void *arrow_status) {
  auto *status_ptr = reinterpret_cast<arrow::Status *>(arrow_status);
  if (ctx == nullptr || ctx->reader == nullptr) {
    *status_ptr = arrow::Status::Invalid("null argument ctx");
    return;
  }
  auto record_batch = reinterpret_cast<std::shared_ptr<arrow::RecordBatch> *>(batch);
  *status_ptr = ctx->reader->ReadBatch(record_batch);
}

__attribute__((visibility("default"))) void
CAPI_LOCAL_ORC_Method_Seek(CAPI_LOCAL_ORC_ReadCtx *ctx, int64_t index, void* arrow_status) {
  auto *status_ptr = reinterpret_cast<arrow::Status *>(arrow_status);
  if (ctx == nullptr || ctx->reader == nullptr) {
    *status_ptr = arrow::Status::Invalid("null argument ctx");
    return;
  }
  *status_ptr = ctx->reader->Seek(index);
}

__attribute__((visibility("default"))) void
CAPI_LOCAL_ORC_Method_Tell(CAPI_LOCAL_ORC_ReadCtx *ctx, int64_t *index) {
  if (ctx == nullptr || ctx->reader == nullptr) {
    *index = -1;
    return;
  }
  *index = ctx->reader->Tell();
}

__attribute__((visibility("default"))) void
CAPI_LOCAL_ORC_Method_ReadSchema(CAPI_LOCAL_ORC_ReadCtx *ctx, void *schema) {
  if (ctx == nullptr || ctx->reader == nullptr || schema == nullptr) {
    return;
  }
  std::shared_ptr<arrow::Schema> schema_ptr = ctx->reader->ReadSchema();
  auto *typed_schema = reinterpret_cast<std::shared_ptr<arrow::Schema> *>(schema);
  *typed_schema = schema_ptr;
  return;
}

__attribute__((visibility("default"))) void
CAPI_LOCAL_ORC_Method_DeleteReaderCtx(CAPI_LOCAL_ORC_ReadCtx *ctx) {
  if (ctx != nullptr) {
    ctx->reader.reset();
    delete ctx;
  }
}

} // end of extern "C"
