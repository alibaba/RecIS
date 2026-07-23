#ifndef COLUMNIO_CC_LOCAL_ORC_DL_WRAPPER_LOCAL_ORC_H_
#define COLUMNIO_CC_LOCAL_ORC_DL_WRAPPER_LOCAL_ORC_H_
#pragma once

#include <functional>
#include "arrow/status.h"
// #include "column-io/framework/common-util/status.h"
#include "column-io/local_orc/local_orc_capi.h"

namespace column {
namespace local_orc {

class LocalOrcLib {
 public:
  static arrow::Status GetLib(LocalOrcLib*& lib) {
    static arrow::Status st;
    static LocalOrcLib* lib_ = [&]() -> LocalOrcLib* {
      LocalOrcLib* lib = new LocalOrcLib();
      st = lib->Open();
      return lib;
    }();
    lib = lib_;
    return st;
  }

  static void LoadWrap() {
    LocalOrcLib* lib;
    auto st = GetLib(lib);
  }

#define DECLARE_INTERFACE_LOCAL_ORC(RetType, FuncName, ...) \
    std::function<RetType(__VA_ARGS__)> FuncName

  DECLARE_INTERFACE_LOCAL_ORC(
      CAPI_LOCAL_ORC_ReadCtx*,
      MakeReader,
      const char* file_path,
      size_t file_path_len,
      const char* columns,
      size_t columns_len);

  DECLARE_INTERFACE_LOCAL_ORC(
      void,
      ReadBatch,
      CAPI_LOCAL_ORC_ReadCtx* reader,
      void* batch,
      void* arrow_status);

  DECLARE_INTERFACE_LOCAL_ORC(
      void,
      Seek,
      CAPI_LOCAL_ORC_ReadCtx* reader,
      int64_t index,
      void* arrow_status);

  DECLARE_INTERFACE_LOCAL_ORC(
      void,
      Tell,
      CAPI_LOCAL_ORC_ReadCtx* reader);

  DECLARE_INTERFACE_LOCAL_ORC(
      void,
      ReadSchema,
      CAPI_LOCAL_ORC_ReadCtx* reader,
      void* schema);

  DECLARE_INTERFACE_LOCAL_ORC(
      void,
      DeleteReaderCtx,
      CAPI_LOCAL_ORC_ReadCtx* reader);

#undef DECLARE_INTERFACE_LOCAL_ORC

 private:
  arrow::Status Open();
};

}  // namespace local_orc
}  // namespace column

#endif  // COLUMNIO_CC_LOCAL_ORC_DL_WRAPPER_LOCAL_ORC_H_
