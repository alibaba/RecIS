#include "dl_wrapper_open_storage.h"
#include <iostream>
#include <dlfcn.h>
#include <cstdlib>

namespace apsara
{
namespace odps
{
namespace tunnel
{
namespace algo
{
namespace tf
{

namespace {

 Status LoadLibrary(void** handle, const char* libpath) {
   Status st;
   *handle = dlopen(libpath, RTLD_LOCAL| RTLD_LAZY | RTLD_DEEPBIND );
   if (!(*handle)) {
     st.Assign(Status::kInternal, dlerror());
     return st;
   }
   return st;
 } 

 template<typename Ret, typename... Args>
 Status BindSym(
     void* handle, 
     std::function<Ret(Args...)>& target_func,
     const char* bind_func_name) {
   Status st;
   void* sym = dlsym(handle, bind_func_name);
   auto error = dlerror();
   if (error) {
     st.Assign(Status::kInternal, error);
     return st;
   }
   target_func = reinterpret_cast<Ret(*)(Args...)>(sym);
   return st;
 }

} // namespace anonymous

Status OdpsOpenStorageLib::Open() {
  Status st;
  const char* kOdpsTunnelDso = getenv("OPENSTORAGEso");
  if (!kOdpsTunnelDso) {
    const std::string& err_msg1 = "env OPENSTORAGEso not set";
    std::cout << err_msg1 << std::endl;
    st.Assign(Status::kInternal, err_msg1);
    return st;
  }
  void *handle;
  st = LoadLibrary(&handle, kOdpsTunnelDso);
  if (!st.Ok()) {
    const std::string& err_msg2 = "Failed to load OPENSTORAGEso, check ldd, LD_LIBRARY_PATH.";
    std::cout << err_msg2 << std::endl;
    st.Assign(Status::kInternal, err_msg2);
    return st;
  }

/******** Functions for Odps Open Storage BEGIN *********/

#define BIND_ODPS_OPEN_STORAGE_FUNC(RetType, FuncName, ...) \
  st = BindSym(handle, FuncName,"CAPI_ODPS_OPEN_STORAGE_" #FuncName); \
  if (!st.Ok()) { \
    return st; \
  }

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
    CreateReadSession,
    const char* access_id,
    const char* access_key,
    const char* tunnel_endpoint,
    const char* project,
    const char* table,
    void* required_partitions,
    void* required_data_columns,
    const char* mode,
    const char* default_project,
    int connect_timeout,
    int rw_timeout,
    char* session_id,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
    GetReadSession,
    const char* access_id,
    const char* access_key,
    const char* tunnel_endpoint,
    const char* session_id,
    const char* project,
    const char* table,
    const char* default_project,
    int connect_timeout,
    int rw_timeout,
    char* session_def_str,
    void* session_def,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void ,
    InitOdpsOpenStorageSessions,
    const char* access_id,
    const char* access_key,
    const char* tunnel_endpoint,
    const char* odps_endpoint,
    const char* projects,
    const char* tables,
    const char* partition_specs,
    const char* physical_partitions,
    const char* required_data_columns,
    const char* sep,
    const char* mode,
    const char* default_project,
    int connect_timeout,
    int rw_timeout,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
    RegisterOdpsOpenStorageSession,
    const char* access_id,
    const char* access_key,
    const char* tunnel_endpoint,
    const char* odps_endpoint,
    const char* project,
    const char* table,
    const char* partition,
    const char* required_data_columns,
    const char* sep,
    const char* mode,
    const char* default_project,
    int connect_timeout,
    int rw_timeout,
    bool register_light,
    const char* session_id,
    long expiration_time,
    long record_count,
    const char* session_def_str,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
    ExtractLocalReadSession,
    char* session_def_str,
    const char* access_id,
    const char* access_key,
    const char* project,
    const char* table,
    const char* partition,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
	RefreshReadSessionBatch,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
	GetHaloAgentMetric,
    const char* halo_endpoint,
    int type,
    int64_t* value);

BIND_ODPS_OPEN_STORAGE_FUNC(
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader*,
    CreateReader,
    const char* path_str,
    int max_batch_rows,
    const char* reader_name,
    const char* const* cols, 
    size_t n,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
    GetTableSize,
    const char* path_str,
    uint64_t* table_size,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
    GetSessionExpireTimestamp,
    const char* session_id,
    void* expire_timestamp);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
    GetSchema,
    const char* config,
    void* schema,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
    ReadBatch,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader,
    void* batch,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
    Seek,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader,
    size_t pos,
    void* status);

BIND_ODPS_OPEN_STORAGE_FUNC(
    size_t,
    Tell,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader);

BIND_ODPS_OPEN_STORAGE_FUNC(
    void,
    DeleteReader,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader,
    void* status);

#undef BIND_ODPS_OPEN_STORAGE_FUNC
  return st;
}

} // namespace tf 
} // namespace algo
} // namespace tunnel
} // namespace odps
} // namespace apsara
