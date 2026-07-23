//#ifdef TF_ENABLE_ODPS_COLUMN

#ifndef PAIIO_THIRD_PARTY_ODPS_OPEN_STORAGE_ARROW_READER_H_H
#define PAIIO_THIRD_PARTY_ODPS_OPEN_STORAGE_ARROW_READER_H_H

#include "column-io/open_storage/common-util/status.h"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <vector>

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

using apsara::odps::algo::commonio::Status;

class OdpsOpenStorageArrowReader {
  public:
    static Status InitOdpsOpenStorageSessions(const std::string& access_id,
                                            const std::string& access_key,
                                            const std::string& tunnel_endpoint,
                                            const std::string& odps_endpoint,
                                            const std::string& projects,
                                            const std::string& tables,
                                            const std::string& partition_specs,
                                            const std::string& physical_partitions,
                                            const std::string& required_data_columns,
                                            const std::string& sep,
                                            const std::string& mode = "row",
                                            const std::string& default_project = "",
                                            int connect_timeout = 300,
                                            int rw_timeout = 300);

    static Status RegisterOdpsOpenStorageSession(const std::string& access_id,
                                                 const std::string& access_key,
                                                 const std::string& tunnel_endpoint,
                                                 const std::string& odps_endpoint,
                                                 const std::string& project,
                                                 const std::string& table,
                                                 const std::string& partition,
                                                 const std::string& required_data_columns,
                                                 const std::string& sep,
                                                 const std::string& mode = "row",
                                                 const std::string& default_project = "",
                                                 int connect_timeout = 300,
                                                 int rw_timeout = 300,
                                                 const bool register_light = false,
                                                 const std::string& session_id = "",
                                                 const long expiration_time = -1,
                                                 const long record_count = -1,
                                                 const std::string& session_def_str = "");

    // path_str: odps://project_name/tables/table_name/ds=ds_name/scene=scene_name?start=0&end=100
    static Status CreateReader(const std::string& path_str,
                               const int max_batch_rows,
                               const std::string& reader_name,
                               const char* const* cols, 
                               size_t n,
                               std::shared_ptr<OdpsOpenStorageArrowReader>& ret);
    static Status GetTableSize(const std::string& path_str, uint64_t& table_size);
    static int64_t GetSessionExpireTimestamp(const std::string& session_id);
    static Status GetSchema(const std::string& config,
                            std::unordered_map<std::string, std::string>& schema);

    OdpsOpenStorageArrowReader();
    ~OdpsOpenStorageArrowReader();
    Status ReadBatch(std::shared_ptr<arrow::RecordBatch>& batch);
    Status Seek(size_t pos);
    Status Close();
    size_t Tell();
    const std::string& GetReaderName();
  
  private:
    class OdpsOpenStorageArrowReaderImpl;
    static std::once_flag no_columns_log_once_flag_;  // 控制 "No columns" 日志打印

    std::shared_ptr<OdpsOpenStorageArrowReaderImpl> impl_;
};

} // namespace tf
} // namespace algo
} // namespace tunnel
} // namespace odps
} // namespace apsara

#endif // PAIIO_THIRD_PARTY_ODPS_OPEN_STORAGE_ARROW_READER_H_H
//#endif // TF_ENABLE_ODPS_COLUMN
