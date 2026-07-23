#include <vector>

#include "storage_api/thirdparty/nlohmann/json.hpp"
#include "column-io/open_storage/plugin/c_api.h"
#include "column-io/open_storage/plugin/odps_open_storage_session.h"
#include "column-io/open_storage/plugin/odps_open_storage_arrow_reader.h"

using apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader;
using apsara::odps::tunnel::algo::tf::OdpsOpenStorageSession;
using apsara::odps::algo::commonio::Status;


/******** Functions for Odps Open Storage BEGIN *********/
struct CAPI_ODPS_SDK_OdpsOpenStorageArrowReader {
  std::shared_ptr<OdpsOpenStorageArrowReader>* reader_;
};

#define DECLARE_INTERFACE_OPEN_STORAGE(RetType, FuncName, ...) \
  __attribute__((visibility("default"))) RetType CAPI_ODPS_OPEN_STORAGE_##FuncName(__VA_ARGS__)

DECLARE_INTERFACE_OPEN_STORAGE(
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
    void* status) {
  auto required_partitions_ = reinterpret_cast<std::vector<std::string>*>(required_partitions);
  auto required_data_columns_ = reinterpret_cast<std::vector<std::string>*>(required_data_columns);
  std::string* session_id_ = reinterpret_cast<std::string*>(session_id);
  auto status_ = reinterpret_cast<Status*>(status);
  *status_ =  OdpsOpenStorageSession::CreateReadSession(
                session_id_,
                access_id, access_key, tunnel_endpoint,
                project, table,
                *required_partitions_, *required_data_columns_,
                mode, default_project,
                connect_timeout, rw_timeout);
}

DECLARE_INTERFACE_OPEN_STORAGE(
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
    void* status) {
  std::string* session_def_str_ = reinterpret_cast<std::string*>(session_def_str);
  nlohmann::json* session_def_ = reinterpret_cast<nlohmann::json*>(session_def);
  auto status_ = reinterpret_cast<Status*>(status);
  *status_ = OdpsOpenStorageSession::GetReadSession(
                session_def_str_, session_def_,
                access_id, access_key, 
                tunnel_endpoint, session_id,
                project, table,
                default_project,
                connect_timeout, rw_timeout);
}

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
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
    void* status) {
  auto status_ = reinterpret_cast<Status*>(status);
  *status_ = OdpsOpenStorageArrowReader::InitOdpsOpenStorageSessions(
                access_id, access_key, tunnel_endpoint, odps_endpoint,
                projects, tables, partition_specs, physical_partitions,
                required_data_columns, sep, mode, default_project,
                connect_timeout, rw_timeout);
}

DECLARE_INTERFACE_OPEN_STORAGE(
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
    void* status) {
  auto status_ = reinterpret_cast<Status*>(status);
  *status_ = OdpsOpenStorageArrowReader::RegisterOdpsOpenStorageSession(
                access_id, access_key,
                tunnel_endpoint, odps_endpoint,
                project, table, partition, required_data_columns,
                sep, mode, default_project,
                connect_timeout, rw_timeout,
                register_light, session_id, expiration_time, record_count,
                session_def_str);
}

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    ExtractLocalReadSession,
    char* session_def_str,
    const char* access_id,
    const char* access_key,
    const char* project,
    const char* table,
    const char* partition,
    void* status) {
  std::string* session_def_str_ = reinterpret_cast<std::string*>(session_def_str);
  auto status_ = reinterpret_cast<Status*>(status);
  *status_ = OdpsOpenStorageSession::ExtractLocalReadSession(
               session_def_str_, access_id, access_key,
               project, table, partition);
}

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    RefreshReadSessionBatch,
    void* status) {
  auto status_ = reinterpret_cast<Status*>(status);
  *status_ = OdpsOpenStorageSession::RefreshReadSessionBatch();
}

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    GetHaloAgentMetric,
    const char* halo_endpoint,
    int type,
    int64_t* value) {
  *value = OdpsOpenStorageSession::GetHaloAgentMetric(halo_endpoint, type);
}

DECLARE_INTERFACE_OPEN_STORAGE(
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader*,
    CreateReader,
    const char* path_str,
    int max_batch_rows,
    const char* reader_name,
    const char* const* cols, 
    size_t n,
    void* status) {
  // TODO: fix this small memory leak
  CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader = new CAPI_ODPS_SDK_OdpsOpenStorageArrowReader;
  reader->reader_ = new std::shared_ptr<OdpsOpenStorageArrowReader>();
  Status* status_ = reinterpret_cast<Status*>(status);
  *status_ =  OdpsOpenStorageArrowReader::CreateReader(
                path_str, max_batch_rows, reader_name,
                cols, n, *(reader->reader_));
  return reader;
}

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    GetTableSize,
    const char* path_str,
    uint64_t* table_size,
    void* status) {
  auto status_ = reinterpret_cast<Status*>(status);
   *status_ = OdpsOpenStorageArrowReader::GetTableSize(path_str, *table_size);
}

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    GetSessionExpireTimestamp,
    const char* session_id,
    void* expire_timestamp) {
  auto expire_timestamp_ = reinterpret_cast<int64_t*>(expire_timestamp);
   *expire_timestamp_ = OdpsOpenStorageArrowReader::GetSessionExpireTimestamp(session_id);
}

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    GetSchema,
    const char* config,
    void* schema,
    void* status) {
  auto schema_ = reinterpret_cast<std::unordered_map<std::string, std::string>*>(schema);
  auto status_ = reinterpret_cast<Status*>(status);
  *status_ = OdpsOpenStorageArrowReader::GetSchema(config, *schema_);
}

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    ReadBatch,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader,
    void* batch,
    void* status) {
  auto batch_ = reinterpret_cast<std::shared_ptr<arrow::RecordBatch>*>(batch);
  auto status_ = reinterpret_cast<Status*>(status);
  *status_ = (*reader->reader_)->ReadBatch(*batch_);
}

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    Seek,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader,
    size_t pos,
    void* status) {
  auto status_ = reinterpret_cast<Status*>(status);
  *status_ = (*reader->reader_)->Seek(pos);
}

DECLARE_INTERFACE_OPEN_STORAGE(
    size_t,
    Tell,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader) {
  return (*reader->reader_)->Tell();
}

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    DeleteReader,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader,
    void* status) {
  auto status_ = reinterpret_cast<Status*>(status);
  *status_ = (*reader->reader_)->Close();
  (*reader->reader_).reset();
  delete reader->reader_;
  delete reader;
}
#undef DECLARE_INTERFACE_OPEN_STORAGE
/******** Functions for Odps Open Storage END *********/
