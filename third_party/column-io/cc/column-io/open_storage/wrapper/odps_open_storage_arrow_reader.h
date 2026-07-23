#ifndef PAIIO_THIRD_PARTY_ODPS_OPEN_STORAGE_ARROW_READER_H_
#define PAIIO_THIRD_PARTY_ODPS_OPEN_STORAGE_ARROW_READER_H_


#include "odps_open_storage_reader_proxy.h"
#include "storage_api/thirdparty/nlohmann/json.hpp"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace arrow {
  class RecordBatch;
} // namespace arrow

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

    static std::string ExtractLocalReadSession(const std::string& access_id,
                                               const std::string& access_key,
                                               const std::string& project,
                                               const std::string& table,
                                               const std::string& partition);

	static Status RefreshReadSessionBatch();
    static int64_t GetHaloAgentMetric(const char* halo_endpoint, int type);

    // When creating session, must create for every each partition, 
    // in case of wrong row_index and row count.
    // TODO. python层如何在调用该方法的饿时候拿到正确的columns 和 partition关系
    static Status CreateReadSession(std::string* session_id,
                                    const std::string& access_id,
                                    const std::string& access_key,
                                    const std::string& tunnel_endpoint,
                                    const std::string& project,
                                    const std::string& table,
                                    std::vector<std::string>& required_partitions,
                                    std::vector<std::string>& required_data_columns,
                                    const std::string& mode = "row",
                                    const std::string& default_project = "",
                                    int connect_timeout = 300,
                                    int rw_timeout = 300);
    static Status GetReadSession(std::string* session_def_str,
                                 nlohmann::json* session_def,
                                 const std::string& access_id,
                                 const std::string& access_key,
                                 const std::string& tunnel_endpoint,
                                 const std::string& session_id,
                                 const std::string& project,
                                 const std::string& table,
                                 const std::string& default_project = "",
                                 int connect_timeout = 300,
                                 int rw_timeout = 300);
    static Status CreateReader(const std::string& path_str,
                               const int max_batch_rows,
                               const std::string& reader_name,
                               std::shared_ptr<OdpsOpenStorageArrowReader>& ret,
                               const std::vector<std::string> input_columns = {});
    static Status GetTableSize(const std::string& path_str, uint64_t& table_size);
    static int64_t GetSessionExpireTimestamp(const std::string& session_id);
    static Status GetSchema(const std::string& config,
                            std::unordered_map<std::string, std::string>& schema);

    OdpsOpenStorageArrowReader();
    ~OdpsOpenStorageArrowReader();
    Status ReadBatch(std::shared_ptr<arrow::RecordBatch>& batch);
    Status Seek(size_t pos);
    size_t Tell();

  //private:
    std::shared_ptr<OdpsOpenStorageArrowReaderProxy> reader_proxy_;
};

} // namespace tf
} // namespace algo
} // namespace tunnel
} // namespace odps
} // namespace apsara

#endif // PAIIO_THIRD_PARTY_ODPS_OPEN_STORAGE_ARROW_READER_H_
