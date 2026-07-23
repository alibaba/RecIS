#include "odps_open_storage_arrow_reader.h"
#include "column-io/open_storage/common-util/logging.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace arrow {

class RecordBatch;

}

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
Status OdpsOpenStorageArrowReader::InitOdpsOpenStorageSessions(const std::string& access_id,
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
  return OdpsOpenStorageArrowReaderProxy::InitOdpsOpenStorageSessions(
           access_id, access_key, tunnel_endpoint, odps_endpoint,
           projects, tables, partition_specs, physical_partitions,
           required_data_columns, sep, mode, default_project,
           connect_timeout, rw_timeout);
}

Status OdpsOpenStorageArrowReader::RegisterOdpsOpenStorageSession(
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
  return OdpsOpenStorageArrowReaderProxy::RegisterOdpsOpenStorageSession(
           access_id, access_key,
           tunnel_endpoint, odps_endpoint,
           project, table, partition, required_data_columns, sep,
           mode, default_project, connect_timeout, rw_timeout,
           register_light, session_id,
           expiration_time, record_count, session_def_str);
}

std::string OdpsOpenStorageArrowReader::ExtractLocalReadSession(
                                          const std::string& access_id,
                                          const std::string& access_key,
                                          const std::string& project,
                                          const std::string& table,
                                          const std::string& partition) {
  std::string session_def_str;
  Status st = OdpsOpenStorageArrowReaderProxy::ExtractLocalReadSession(
                session_def_str, access_id, access_key,
                project, table, partition);
  std::stringstream common_msg_ss;
  common_msg_ss << "project: " << project
                << " table: " << table
                << " partition: " << partition;
  if (st.Ok()) {
    LOG(INFO) << "Successfully extract session def str for "
              << common_msg_ss.str();
    return session_def_str;
  } else {
    LOG(INFO) << "Failed to extract session def str for "
              << common_msg_ss.str() << " return {}.";
    return "{}";
  }
}

Status OdpsOpenStorageArrowReader::RefreshReadSessionBatch() {
  return OdpsOpenStorageArrowReaderProxy::RefreshReadSessionBatch();
}

int64_t OdpsOpenStorageArrowReader::GetHaloAgentMetric(const char* halo_endpoint, int type) {
  return OdpsOpenStorageArrowReaderProxy::GetHaloAgentMetric(halo_endpoint, type);
}

// When creating session, must create for every each partition, 
// in case of wrong row_index and row count.
// TODO. python层如何在调用该方法的饿时候拿到正确的columns 和 partition关系
Status OdpsOpenStorageArrowReader::CreateReadSession(std::string* session_id,
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
  std::string session_id_;
  Status stat = OdpsOpenStorageArrowReaderProxy::CreateReadSession(&session_id_, access_id, access_key,
                                                                   tunnel_endpoint, project, table,
                                                                   required_partitions, required_data_columns,
                                                                   mode, default_project,
                                                                   connect_timeout, rw_timeout);
  session_id = &session_id_;
  return stat;
}

Status OdpsOpenStorageArrowReader::GetReadSession(std::string* session_def_str,
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
  std::string session_def_str_;
  nlohmann::json session_def_;
  Status stat = OdpsOpenStorageArrowReaderProxy::GetReadSession(&session_def_str_, &session_def_,
                                                                access_id, access_key,
                                                                tunnel_endpoint, session_id,
                                                                project, table, default_project,
                                                                connect_timeout, rw_timeout);
 *session_def_str = session_def_str_;
 *session_def = session_def_;
 return stat;
}

Status OdpsOpenStorageArrowReader::CreateReader(const std::string& path_str,
                                                const int max_batch_rows,
                                                const std::string& reader_name,
                                                std::shared_ptr<OdpsOpenStorageArrowReader>& ret,
                                                const std::vector<std::string> input_columns) {
  std::shared_ptr<OdpsOpenStorageArrowReaderProxy> reader_proxy;
  Status st;

  st = OdpsOpenStorageArrowReaderProxy::CreateReader(
         path_str, max_batch_rows, reader_name,
         input_columns, reader_proxy);
  std::shared_ptr<OdpsOpenStorageArrowReader> reader = std::make_shared<OdpsOpenStorageArrowReader>();
  reader->reader_proxy_ = reader_proxy;
  ret = reader;
  return st;
}

Status OdpsOpenStorageArrowReader::GetTableSize(const std::string& path_str, uint64_t& table_size) {
  Status stat = OdpsOpenStorageArrowReaderProxy::GetTableSize(path_str, table_size);
  return stat;
}

int64_t OdpsOpenStorageArrowReader::GetSessionExpireTimestamp(const std::string& session_id) {
  return OdpsOpenStorageArrowReaderProxy::GetSessionExpireTimestamp(session_id);
}

Status OdpsOpenStorageArrowReader::GetSchema(const std::string& config,
                                             std::unordered_map<std::string, std::string>& schema) {
  Status stat = OdpsOpenStorageArrowReaderProxy::GetSchema(config, schema);
  return stat;
}

OdpsOpenStorageArrowReader::OdpsOpenStorageArrowReader() {}

OdpsOpenStorageArrowReader::~OdpsOpenStorageArrowReader() {
  reader_proxy_.reset();
}

Status OdpsOpenStorageArrowReader::ReadBatch(std::shared_ptr<arrow::RecordBatch>& batch) {
  return reader_proxy_->ReadBatch(batch);
}

Status OdpsOpenStorageArrowReader::Seek(size_t pos) {
  return reader_proxy_->Seek(pos);
}

size_t OdpsOpenStorageArrowReader::Tell() {
  return reader_proxy_->Tell();
}

} // namespace tf
} // namespace algo
} // namespace tunnel
} // namespace odps
} // namespace apsara
