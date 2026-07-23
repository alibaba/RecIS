#include "odps_open_storage_reader_proxy.h"
#include "column-io/open_storage/common-util/logging.h"
#include <stdexcept>

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
// When creating session, must create for every each partition, 
// in case of wrong row_index and row count.
// TODO. python层如何在调用该方法的饿时候拿到正确的columns 和 partition关系
Status OdpsOpenStorageArrowReaderProxy::CreateReadSession(std::string* session_id,
                                                          const std::string& access_id,
                                                          const std::string& access_key,
                                                          const std::string& tunnel_endpoint,
                                                          const std::string& project,
                                                          const std::string& table,
                                                          std::vector<std::string>& required_partitions,
                                                          std::vector<std::string>& required_data_columns,
                                                          const std::string& mode,
                                                          const std::string& default_project,
                                                          int connect_timeout,
                                                          int rw_timeout) {
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    return st;
  }
  lib->CreateReadSession(
         access_id.c_str(), access_key.c_str(),
         tunnel_endpoint.c_str(),
         project.c_str(), table.c_str(),
         reinterpret_cast<void*>(&required_partitions),
         reinterpret_cast<void*>(&required_data_columns),
         mode.c_str(), default_project.c_str(),
         connect_timeout, rw_timeout,
         reinterpret_cast<char*>(session_id),
         reinterpret_cast<void*>(&st));
  return st;
}

// used for GetTableSize(python)/GetSchema(python)
Status OdpsOpenStorageArrowReaderProxy::GetReadSession(std::string* session_def_str,
                                                       nlohmann::json* session_def,
                                                       const std::string& access_id,
                                                       const std::string& access_key,
                                                       const std::string& tunnel_endpoint,
                                                       const std::string& session_id,
                                                       const std::string& project,
                                                       const std::string& table,
                                                       const std::string& default_project,
                                                       int connect_timeout,
                                                       int rw_timeout) {
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    return st;
  }
  lib->GetReadSession(
         access_id.c_str(), access_key.c_str(),
         tunnel_endpoint.c_str(), session_id.c_str(),
         project.c_str(), table.c_str(),
         default_project.c_str(),
         connect_timeout, rw_timeout,
         reinterpret_cast<char*>(session_def_str),
         reinterpret_cast<void*>(session_def),
         reinterpret_cast<void*>(&st));
  return st;
}

Status OdpsOpenStorageArrowReaderProxy::InitOdpsOpenStorageSessions(
                                          const std::string& access_id,
                                          const std::string& access_key,
                                          const std::string& tunnel_endpoint,
                                          const std::string& odps_endpoint,
                                          const std::string& projects,
                                          const std::string& tables,
                                          const std::string& partition_specs,
                                          const std::string& physical_partitions,
                                          const std::string& required_data_columns,
                                          const std::string& sep,
                                          const std::string& mode,
                                          const std::string& default_project,
                                          int connect_timeout,
                                          int rw_timeout) {
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    return st;
  }
  lib->InitOdpsOpenStorageSessions(
      access_id.c_str(), access_key.c_str(),
      tunnel_endpoint.c_str(), odps_endpoint.c_str(),
      projects.c_str(), tables.c_str(),
      partition_specs.c_str(), physical_partitions.c_str(),
      required_data_columns.c_str(), sep.c_str(), mode.c_str(),
      default_project.c_str(), connect_timeout, rw_timeout,
      reinterpret_cast<void*>(&st));
  // if (!st.Ok()) {
  //   return st;
  // }
  // return st wheather it is ok or not
  return st;
}

Status OdpsOpenStorageArrowReaderProxy::RegisterOdpsOpenStorageSession(
                                          const std::string& access_id,
                                          const std::string& access_key,
                                          const std::string& tunnel_endpoint,
                                          const std::string& odps_endpoint,
                                          const std::string& project,
                                          const std::string& table,
                                          const std::string& partition,
                                          const std::string& required_data_columns,
                                          const std::string& sep,
                                          const std::string& mode,
                                          const std::string& default_project,
                                          int connect_timeout,
                                          int rw_timeout,
                                          const bool register_light,
                                          const std::string& session_id,
                                          const long expiration_time,
                                          const long record_count,
                                          const std::string& session_def_str) {
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    LOG(ERROR) << "Failed to GetLib for RegisterOdpsOpenStorageSession.";
    return st;
  }
  lib->RegisterOdpsOpenStorageSession(
         access_id.c_str(), access_key.c_str(),
         tunnel_endpoint.c_str(), odps_endpoint.c_str(),
         project.c_str(), table.c_str(), partition.c_str(),
         required_data_columns.c_str(), sep.c_str(), mode.c_str(),
         default_project.c_str(), connect_timeout, rw_timeout,
         register_light, session_id.c_str(),
         expiration_time, record_count, session_def_str.c_str(),
         reinterpret_cast<void*>(&st));
  // if (!st.Ok()) {
  //   return st;
  // }
  // return st wheather it is ok or not
  return st;
}

Status OdpsOpenStorageArrowReaderProxy::ExtractLocalReadSession(
                                          std::string& session_def_str,
                                          const std::string& access_id,
                                          const std::string& access_key,
                                          const std::string& project,
                                          const std::string& table,
                                          const std::string& partition) {
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    return st;
  }
  std::string tmp_session_def_str;
  lib->ExtractLocalReadSession(
         reinterpret_cast<char*>(&tmp_session_def_str),
         access_id.c_str(), access_key.c_str(),
         project.c_str(), table.c_str(), partition.c_str(),
         reinterpret_cast<void*>(&st));
  if (st.Ok()) {
    session_def_str = tmp_session_def_str;
  }
  return st;
}

Status OdpsOpenStorageArrowReaderProxy::RefreshReadSessionBatch() {
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    return st;
  }
  lib->RefreshReadSessionBatch(reinterpret_cast<void*>(&st));
  return st;
}

int64_t OdpsOpenStorageArrowReaderProxy::GetHaloAgentMetric(const char* halo_endpoint, int type) {
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    return -1;
  }
  int64_t value = -1;
  lib->GetHaloAgentMetric(halo_endpoint, type, reinterpret_cast<int64_t*>(&value));
  return value;
}

Status OdpsOpenStorageArrowReaderProxy::CreateReader(const std::string& path_str,
                                                     const int max_batch_rows,
                                                     const std::string& reader_name,
                                                     const std::vector<std::string> input_columns,
                                                     std::shared_ptr<OdpsOpenStorageArrowReaderProxy>& ret) {
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    return st;
  }
  std::vector<const char*> ccols;
  ccols.reserve(input_columns.size());
  for (const auto& s : input_columns) {
      ccols.push_back(s.c_str()); 
  }
  CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader = lib->CreateReader(
      path_str.c_str(),
      max_batch_rows,
      reader_name.c_str(),
      ccols.data(),
      ccols.size(),
      reinterpret_cast<void*>(&st));
  if (!st.Ok()) {
    return st;
  }
  ret = std::make_shared<OdpsOpenStorageArrowReaderProxy>(reader, lib);
  return st;
}

Status OdpsOpenStorageArrowReaderProxy::GetTableSize(const std::string& path_str, uint64_t& table_size) {
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    return st;
  }
  lib->GetTableSize(
      path_str.c_str(),
      &table_size,
      reinterpret_cast<void*>(&st));
  return st;
}

int64_t OdpsOpenStorageArrowReaderProxy::GetSessionExpireTimestamp(const std::string& session_id) {
  int64_t ts=0;
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    return ts;
  }
  lib->GetSessionExpireTimestamp(session_id.c_str(), reinterpret_cast<void*>(&ts));
  return ts;
}

Status OdpsOpenStorageArrowReaderProxy::GetSchema(const std::string& config,
                                                  std::unordered_map<std::string, std::string>& schema) {
  OdpsOpenStorageLib* lib;
  auto st = OdpsOpenStorageLib::GetLib(lib);
  if (!st.Ok()) {
    return st;
  }
  lib->GetSchema(
      config.c_str(),
      reinterpret_cast<void*>(&schema),
      reinterpret_cast<Status*>(&st));
  return st;
}

OdpsOpenStorageArrowReaderProxy::OdpsOpenStorageArrowReaderProxy(
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader, 
    OdpsOpenStorageLib* lib):reader_(reader),lib_(lib) {
}

OdpsOpenStorageArrowReaderProxy::~OdpsOpenStorageArrowReaderProxy() {
  Status st;
  lib_->DeleteReader(reader_, reinterpret_cast<void*>(&st));
  if (!st.Ok()) {
    LOG(FATAL) << st.GetMsg();
    exit(-1);
  }
}

Status OdpsOpenStorageArrowReaderProxy::ReadBatch(std::shared_ptr<arrow::RecordBatch>& batch) {
  Status st;
  lib_->ReadBatch(
      reader_,
      reinterpret_cast<void*>(&batch),
      reinterpret_cast<void*>(&st));
  return st;
}

Status OdpsOpenStorageArrowReaderProxy::Seek(size_t pos) {
  Status st;
  lib_->Seek(
      reader_,
      pos,
      reinterpret_cast<void*>(&st));
  return st;
}

size_t OdpsOpenStorageArrowReaderProxy::Tell() {
  return lib_->Tell(reader_);
}

} // namespace tf
} // namespace algo
} // namespace tunnel
} // namespace odps
} // namespace apsara
