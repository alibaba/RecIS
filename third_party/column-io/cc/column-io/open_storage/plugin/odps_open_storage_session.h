//#ifdef TF_ENABLE_ODPS_COLUMN

#ifndef PAIIO_THIRD_PARTY_ODPS_OPEN_STORAGE_SESSION_H_
#define PAIIO_THIRD_PARTY_ODPS_OPEN_STORAGE_SESSION_H_

#include "column-io/open_storage/common-util/status.h"
#include "column-io/open_storage/common-util/openstorage_metric_reporter.h"
#include "storage_api/include/storage_api.hpp"
#include "storage_api/include/storage_api_arrow.hpp"
#include "storage_api/thirdparty/nlohmann/json.hpp"

#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <future>
#include <cstdint>

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

class OdpsOpenStorageSession {
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
                                                 const std::string& mode,
                                                 const std::string& default_project,
                                                 int connect_timeout,
                                                 int rw_timeout,
                                                 const bool register_light,
                                                 const std::string& session_id,
                                                 const long expiration_time = -1,
                                                 const long record_count = 1,
                                                 const std::string& session_def_str = "");

    static Status ExtractLocalReadSession(std::string* session_def_str,
                                          const std::string& access_id,
                                          const std::string& access_key,
                                          const std::string& project,
                                          const std::string& table,
                                          const std::string& partition);

    static Status GetOdpsOpenStorageSession(const std::string& project,
                                            const std::string& table,
                                            const std::string& partition_spec,
                                            OdpsOpenStorageSession** session);
    static int64_t GetSessionExpireTimestamp(const std::string& session_id);
    static apsara::odps::sdk::storage_api::arrow_adapter::ArrowClient*
             GetArrowClient(const std::string& access_id,
                            const std::string& access_key,
                            const std::string& tunnel_endpoint,
                            const std::string& default_project = "",
                            int connect_timeout = 300,
                            int rw_timeout = 300);
    static apsara::odps::sdk::storage_api::arrow_adapter::ArrowClient*
             GetArrowClient(const apsara::odps::sdk::storage_api::Configuration& configuration);
    static Status CreateReadSession(std::string* session_id,
                                    const std::string& access_id,
                                    const std::string& access_key,
                                    const std::string& tunnel_endpoint,
                                    const std::string& project,
                                    const std::string& table,
                                    const std::vector<std::string>& required_partitions,
                                    const std::vector<std::string>& required_data_columns,
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
    static std::string timestampToReadableTime(long timestamp);
    static Status RefreshReadSessionBatch();
    static int64_t GetHaloAgentMetric(const char* halo_endpoint, int type);
    static Status RefreshReadSession(const std::string& access_id,
                                     const std::string& access_key,
                                     const std::string& tunnel_endpoint,
                                     const std::string& session_id,
                                     const std::string& project,
                                     const std::string& table,
                                     const std::string& default_project = "",
                                     int connect_timeout = 300,
                                     int rw_timeout = 300,
									                   OdpsOpenStorageSession* tmp_session = nullptr);
	  static Status CreatAndGetReadSession(bool is_session_id_normal,
                                         std::string* session_id,
                                         std::string* session_def_str,
                                         nlohmann::json* session_def,
                                         const std::string& access_id,
                                         const std::string& access_key,
                                         const std::string& tunnel_endpoint,
                                         const std::string& project,
                                         const std::string& table,
                                         const std::vector<std::string>& required_partitions,
                                         const std::vector<std::string>& required_data_columns,
                                         const std::string& mode = "row",
                                         const std::string& default_project = "",
                                         int connect_timeout = 300,
                                         int rw_timeout = 300,
                                         int try_times = 360);

    bool IsInitialized();
    long GetTableSize();
    void InitCompressionType();
    std::unordered_map<std::string, std::string> GetSchema();
    std::shared_ptr<apsara::odps::sdk::storage_api::arrow_adapter::Reader>
      CreateOpenStorageReader(long start, long end,
                              int max_batch_rows, int max_batch_raw_size,
                              int cache_size = 1,
                              bool data_columns_unordered = true,
                              const std::string& reader_name = "",
                              const std::vector<std::string>& input_columns = {});
    Status ExtractSessionDef(nlohmann::json* session_def);
    Status ExtractSchema(nlohmann::json* session_def);
    OdpsOpenStorageSession();
    OdpsOpenStorageSession(const std::string& session_id,
                           const std::string& project,
                           const std::string& table,
                           const std::string& partition,
                           apsara::odps::sdk::storage_api::Configuration& configuration,
                           std::vector<std::string>& required_data_columns,
                           const std::string& mode,
                           const std::string& default_project);
    ~OdpsOpenStorageSession();

//  private:
    Status InitOpenStorageSession(const std::string& access_id,
                                const std::string& access_key,
                                const std::string& tunnel_endpoint,
                                const std::string& project,
                                const std::string& table,
                                const std::string& partition_spec,
                                const std::vector<std::string>& physical_partitions,
                                const std::vector<std::string>& required_data_columns,
                                const std::string& mode,
                                const std::string& default_project,
                                int connect_timeout,
                                int rw_timeout,
                                bool register_session_id,
                                bool register_light,
                                const std::string& session_id,
                                const long expiration_time = -1,
                                const long record_count = -1,
                                const std::string& session_def_str = "");

    bool initialized_ = false;
    // followings are arguments
    apsara::odps::sdk::storage_api::Configuration configuration_;
    std::string project_;
    std::string table_;
    std::string partition_spec_;
    std::vector<std::string> physical_partitions_;
    std::vector<std::string> required_data_columns_;
    std::string mode_;
    std::string default_project_;
    int compression_type_ = -1;  // -1: auto compression; 0: UNCOMPRESSED; 1: ZSTD; 2: LZ4_FRAME
    // followings are results
    std::string session_id_ = "";
    nlohmann::json session_def_;
    std::pair<std::shared_ptr<recis::monitor::Client>, std::shared_ptr<recis::monitor::PointTag>> metric_reporter_;
    long expiration_time_ = -1;  // 13bit timestamp
    long record_count_ = -1;     // table size
    std::unordered_map<std::string, std::string> schema_;


  public:
    using MapKeyType = std::tuple<std::string, std::string, std::string>;
    struct MapKeyTypeHash: public std::unary_function<MapKeyType, size_t> {
      size_t operator()(const MapKeyType& key) const {
        auto hash = std::hash<std::string>();
        return hash(std::get<0>(key)) ^ hash(std::get<1>(key)) ^ hash(std::get<2>(key));
      }
    };
    inline static MapKeyType MakeMapKey(const std::string& key_one,const std::string& key_two,const std::string& key_three) {
      return std::move(std::make_tuple(key_one, key_two, key_three));
    }
    using ConfigToSessionValueType = std::unique_ptr<OdpsOpenStorageSession>;
    using ConfigToFutureValueType = std::future<ConfigToSessionValueType>;
    using ConfigToSessionMap = std::unordered_map<const MapKeyType,ConfigToSessionValueType, MapKeyTypeHash>;
    using ConfigToFutureMap = std::unordered_map<const MapKeyType,ConfigToFutureValueType, MapKeyTypeHash>;
  private:
    static ConfigToSessionMap config_to_session_;
    static ConfigToFutureMap config_to_futures_;
    static std::mutex config_to_open_storage_session_lock_;
};

} // namespace tf
} // namespace algo
} // namespace tunnel
} // namespace odps
} // namespace apsara

#endif // PAIIO_THIRD_PARTY_ODPS_OPEN_STORAGE_SESSION_H_
//#endif // TF_ENABLE_ODPS_COLUMN
