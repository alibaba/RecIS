#include "column-io/local_orc/dlwrapper_local_orc.h"

#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>

namespace column {
namespace local_orc {

namespace {

arrow::Status LoadLibrary(void** handle, const char* libpath) {
  const char* env_ld_preload = getenv("LD_PRELOAD");
  bool has_malloc_preload =
      env_ld_preload &&
      (strstr(env_ld_preload, "tcmalloc") ||
       strstr(env_ld_preload, "jemalloc") ||
       strstr(env_ld_preload, "mimalloc") ||
       strstr(env_ld_preload, "libmalloc"));
  // 如果使用了注入的分配器库，则不要使用 RTLD_DEEPBIND，否则分配库会打架
  if (has_malloc_preload) {
    *handle = dlopen(libpath, RTLD_LOCAL | RTLD_LAZY);
  } else {
    *handle = dlopen(libpath, RTLD_LOCAL | RTLD_LAZY | RTLD_DEEPBIND);
  }
  if (!(*handle)) {
    arrow::Status st = arrow::Status(arrow::StatusCode::Invalid, std::string(dlerror()));
    return st;
  }
  return arrow::Status::OK();
}

template <typename Ret, typename... Args>
arrow::Status BindSym(
    void* handle,
    std::function<Ret(Args...)>& target_func,
    const char* bind_func_name) {
  void* sym = dlsym(handle, bind_func_name);
  auto error = dlerror();
  if (error) {
    return arrow::Status(arrow::StatusCode::Invalid, std::string(error));
  }
  target_func = reinterpret_cast<Ret(*)(Args...)>(sym);
  return arrow::Status::OK();
}

}  // namespace

arrow::Status LocalOrcLib::Open() {
  const char* kLocalOrcDso = getenv("LOCAL_ORC_so");
  if (!kLocalOrcDso) {
    const std::string& err_msg = "env LOCAL_ORC_so not set";
    std::cout << err_msg << std::endl;
    return arrow::Status(arrow::StatusCode::Invalid, err_msg);
  }
  void* handle;
  arrow::Status st = LoadLibrary(&handle, kLocalOrcDso);
  if (!st.ok()) {
    const std::string& err_msg = "Failed to load LOCAL_ORC_so, check ldd, LD_LIBRARY_PATH.";
    std::cout << err_msg << std::endl;
    return arrow::Status(arrow::StatusCode::Invalid, err_msg);
  }

/******** Functions for Local ORC BEGIN *********/

#define BIND_LOCAL_ORC_FUNC(RetType, FuncName, ...) \
  st = BindSym(handle, FuncName, "CAPI_LOCAL_ORC_Method_" #FuncName); \
  if (!st.ok()) {                                                     \
    return st;                                                        \
  }

  BIND_LOCAL_ORC_FUNC(
      CAPI_LOCAL_ORC_ReadCtx*,
      MakeReader,
      const char* file_path,
      size_t file_path_len,
      const char* columns,
      size_t columns_len);

  BIND_LOCAL_ORC_FUNC(
      void,
      ReadBatch,
      CAPI_LOCAL_ORC_ReadCtx* reader,
      void* batch,
      void* arrow_status);

  BIND_LOCAL_ORC_FUNC(
      void,
      Seek,
      CAPI_LOCAL_ORC_ReadCtx* reader,
      int64_t index,
      void* arrow_status);

  BIND_LOCAL_ORC_FUNC(
      void,
      Tell,
      CAPI_LOCAL_ORC_ReadCtx* reader);

  BIND_LOCAL_ORC_FUNC(
      void,
      ReadSchema,
      CAPI_LOCAL_ORC_ReadCtx* reader,
      void* schema);

  BIND_LOCAL_ORC_FUNC(
      void,
      DeleteReaderCtx,
      CAPI_LOCAL_ORC_ReadCtx* reader);

#undef BIND_LOCAL_ORC_FUNC

/******** Functions for Local ORC END *********/

  return st;
}

}  // namespace local_orc
}  // namespace column
