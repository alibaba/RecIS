#include <algorithm>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <future>
#include <cstdint>

#include "storage_api/include/get_halo_metrics.hpp"

#include "column-io/open_storage/plugin/odps_open_storage_session.h"
#include "column-io/open_storage/common-util/logging.h"
#include "column-io/open_storage/common-util/common_util.h"


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

OdpsOpenStorageSession::ConfigToSessionMap OdpsOpenStorageSession::config_to_session_;
OdpsOpenStorageSession::ConfigToFutureMap OdpsOpenStorageSession::config_to_futures_;
std::mutex OdpsOpenStorageSession::config_to_open_storage_session_lock_;

static std::unique_ptr<recis::monitor::Factory> monitor_factory_;
static std::mutex monitor_factory_mutex_ = std::mutex();

using xdl::paiio::third_party::common_util::StrSplit;
using xdl::paiio::third_party::common_util::SimpleThreadPool;


namespace {

const std::string& kRecordCount = "RecordCount";
const std::string& kExpirationTime = "ExpirationTime";
const std::string& kSessionId = "SessionId";
const std::string& kSessionStatus = "SessionStatus";
const std::string& kDataSchema = "DataSchema";
const std::string& kINIT = "INIT";
const std::string& kNORMAL = "NORMAL";
const int kConfigurationRetryTimes = 300;
const int kSessionCreationLoopWaitSeconds = 10;
const std::string& kEmptySessionId = "";
const int kDefaultReaderCompressType = -1;

int GetReaderCompressionType() {
  const char* compression_type = getenv("OPEN_STORAGE_READER_COMPRESSION_TYPE");
  if (compression_type == nullptr) {
    return kDefaultReaderCompressType;
  }
  int res = atoi(compression_type);
  if (res != 0 && res != 1 && res != 2) {
    LOG(INFO) << "OPEN_STORAGE_READER_COMPRESSION_TYPE value: " << res
                 << " is invalid, return default: " << kDefaultReaderCompressType;
    res = kDefaultReaderCompressType;
  }
  return res;
}
} // anonymous namespace

void OdpsOpenStorageSession::InitCompressionType() {
  int env_type = GetReaderCompressionType();
  if (env_type >= 0) {
    compression_type_ = env_type;
    LOG(INFO) << "env set compression_type_: " << compression_type_;
  } else {
    const auto& tunnel_endpoint = configuration_.GetTunnelEndpoint();
    bool is_halo_endpoint = (tunnel_endpoint.find("ipc://") == 0);
    compression_type_ = is_halo_endpoint ? 0 : 1; // 0 is UNCOMPRESSED, 1 is ZSTD
    LOG(INFO) << "tunnel endpoint: " << tunnel_endpoint << ", " << "auto set compression_type_: " << compression_type_;
  }
}

// required_data_columns: 表示columns的参数, 用于切列, 必须保证列按照表的顺序排序好, 目前可以考虑只为一种schema服务
// 支持从头创建session, 也应该支持接受传入session, 后面优化目标是避免每个worker自建session
Status OdpsOpenStorageSession::InitOdpsOpenStorageSessions(const std::string& access_id,
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
  Status st;
  LOG(INFO) << "InitOdpsOpenStorageSessions tunnel_endpoint: [" << tunnel_endpoint << "], "
            << "odps_endpoint: [" << odps_endpoint << "].";
  std::vector<std::string> project_vec_ = StrSplit(projects, sep, false);
  std::vector<std::string> table_vec_ = StrSplit(tables, sep, false);
  std::vector<std::string> partition_vec_ = StrSplit(partition_specs, sep, false);
  std::vector<std::string> physical_partitions_vec_ = StrSplit(physical_partitions, sep, false);
  std::vector<std::string> select_column_vec_ = StrSplit(required_data_columns, sep, true);
  if (select_column_vec_.size() == 0) {
    LOG(INFO) << "select_column_vec_ size is 0, create session for all columns.";
  }
  if (project_vec_.size() != table_vec_.size() || project_vec_.size() != partition_vec_.size()) {
    LOG(ERROR) << "session specs vec not equal, project.size:" << project_vec_.size() << ", table.size:" << table_vec_.size()
                << ", partition.size:" << partition_vec_.size();
    return Status(Status::kInvalidArgument, "partition size is unequal");
  }
  if (partition_vec_.size() == 0 ){
    LOG(WARNING) << "Existing following session specs vec is 0, not create session!";
  }
  if (physical_partitions_vec_.size() > 0) {
    if (partition_vec_.size() != 1) {  // 如果physical_partition大于0, 则logical_partiton必须为1, 否则我们不知道对应关系
      std::ostringstream oss;
      oss << "We have PhysicalPartitions num: " << std::to_string(physical_partitions_vec_.size())
          << " > 0 , logicalPartiton size must be 1 , but we got : "
          << std::to_string(partition_vec_.size());
      LOG(ERROR) << oss.str();
      return Status(Status::kInvalidArgument, oss.str());
    }
    std::ostringstream pp_oss;
    pp_oss << "  LogicPartition: " << partition_vec_[0] << "\n";
    pp_oss << "  PhysicalPartitions: \n";
    for (const auto& phy_part: physical_partitions_vec_) {
      pp_oss << "        " << phy_part << "\n";
    }
    LOG(INFO) << "We have " << physical_partitions_vec_.size() << " PhysicalPartitions, "
              << "and going to create session with them as one. \n"
              << pp_oss.str();
  } else if (physical_partitions_vec_.size() == 0) {  // If no physical_partitions, get value from partition_vec
    physical_partitions_vec_ = partition_vec_;
  }
  for(int i = 0; i < table_vec_.size(); ++i){
    std::string& project = project_vec_[i];
    std::string& table = table_vec_[i];
    std::string& partition = partition_vec_[i];
    auto key = MakeMapKey(project, table, partition);
    {
        std::lock_guard<std::mutex> lock(config_to_open_storage_session_lock_);
        if (config_to_session_.find(key) != config_to_session_.end()) continue;
        config_to_session_[key] = std::move(ConfigToSessionValueType());
    }
    // TODO: session并发创建后, 这里的调用可以改为同步形式
    auto func = [=] () -> ConfigToSessionValueType {
        ConfigToSessionValueType tmp_session(new OdpsOpenStorageSession());
        LOG(INFO) << "register session builder for ["
                << project << ","
                << table << ","
                << partition << ","
                << "column num:" << std::to_string(select_column_vec_.size()) << "]";
        auto stats = tmp_session->InitOpenStorageSession(
                        access_id, access_key, tunnel_endpoint,
                        project, table, partition, physical_partitions_vec_,
                        select_column_vec_, mode, default_project, connect_timeout, rw_timeout,
                        false, false, kEmptySessionId);
        if (!stats.Ok()) {
            LOG(ERROR) << "fail in InitOpenStorageSession on stats "
                << "code:" << stats.GetCode() << ","
                << "msg:" << stats.GetMsg();
            // throw c++ exception
            // throw stats;
            std::exit(1);  // 失败需及时退出, 防止后面建Reader才出错, 掩盖错误出处
        }
        return std::move(tmp_session);
    };
    ConfigToFutureValueType ret = SimpleThreadPool::GetInstance()->Enqueue(func);
    {
        std::lock_guard<std::mutex> lock(config_to_open_storage_session_lock_);
        config_to_futures_[key] = std::move(ret);
    }
  }
  return st; // future create session will certainly success for now
}

Status OdpsOpenStorageSession::RegisterOdpsOpenStorageSession(
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
  Status st = Status();

  auto key = MakeMapKey(project, table, partition);
  {
    std::lock_guard<std::mutex> lock(config_to_open_storage_session_lock_);
    //LOG(INFO) << "Register session: " << session_id
    //          << " with project: " << project
    //          << " table: " << table
    //          << " partition: " << partition
    //          << " register_light: " << register_light;  // FIXME: log to another file
    if (config_to_session_[key]) {
      // 应该补充逻辑, 如果 expiration_time - now > 12h 则允许替换, 否则略过
      // 考虑从cache拿的时候就带上 expiration_time/row_count等信息
      LOG(DEBUG) << "Already registered, give up !";
	} else {
      std::vector<std::string> required_data_columns_vec = StrSplit(required_data_columns, sep, true);

      ConfigToSessionValueType tmp_session(new OdpsOpenStorageSession());
      std::vector<std::string> physical_partitions{partition};
      if (register_light) {
        st = tmp_session->InitOpenStorageSession(access_id, access_key, tunnel_endpoint,
                                                 project, table, partition,
                                                 physical_partitions,
                                                 required_data_columns_vec,
                                                 mode, default_project,
                                                 connect_timeout, rw_timeout,
                                                 true, true, session_id,
                                                 expiration_time, record_count,
                                                 session_def_str);
      } else {
        st = tmp_session->InitOpenStorageSession(access_id, access_key, tunnel_endpoint,
                                                 project, table, partition,
                                                 physical_partitions,
                                                 required_data_columns_vec,
                                                 mode, default_project,
                                                 connect_timeout, rw_timeout,
                                                 true, false, session_id);
      }
      if (!st.Ok()) {
        LOG(ERROR) << "fail to GetReadSession for sessionId: " << session_id;
        st.Assign(Status::kInternal, "fail to GetReadSession for sessionId: " + session_id);
        return st;
      }
      // TODO: 验证有无必要在此处进行session合法性检查
      //st = tmp_session->ExtractSessionDef(&session_def);
      //if (st.Ok()) {
      //  LOG(INFO) << "Successfully registered session: " << tmp_session->session_id_
      //            << " expiration_time: " << std::to_string(tmp_session->expiration_time_)
      //            << " record_count: " << std::to_string(tmp_session->record_count_);
      //} else {
      //  st.Assign(Status::kInternal, "fail to ExtractSessionDef for sessionId: " + session_id);
      //  return st;
      //}
      config_to_session_[key] = std::move(tmp_session);
    }
  }

  return st;
}

Status OdpsOpenStorageSession::ExtractLocalReadSession(
                                 std::string* session_def_str,
                                 const std::string& access_id,
                                 const std::string& access_key,
                                 const std::string& project,
                                 const std::string& table,
                                 const std::string& partition) {
  Status st = Status();
  OdpsOpenStorageSession* session = nullptr;
  st = GetOdpsOpenStorageSession(project, table, partition, &session);
  if (!st.Ok()) {
    const std::string& error_msg = "Session not exist with project: " + project
                                   + " table: " + table + " partition: " + partition;
    LOG(ERROR) << error_msg;
    st.Assign(Status::kInternal, error_msg);
    return st;
  }
  *session_def_str = session->session_def_.dump();
  LOG(INFO) << "Extract session: " << session->session_id_
            << " with expiration_time: " << std::to_string(session->expiration_time_)
            << " record_count: " << std::to_string(session->record_count_)
            << " of project: " << project
            << " table: " << table
            << " partition: " << partition;
  return st;
}

Status OdpsOpenStorageSession::GetOdpsOpenStorageSession(const std::string& project,
                                                         const std::string& table,
                                                         const std::string& partition_spec,
                                                         OdpsOpenStorageSession** session) {
  Status st = Status();
  auto key = MakeMapKey(project, table, partition_spec);
  {
    std::lock_guard<std::mutex> lock(config_to_open_storage_session_lock_);
    std::stringstream common_msg_ss;
    common_msg_ss << "Retrive session for ["
                  << project << ","
                  << table << ","
                  << partition_spec << "]";
   const std::string& common_msg = common_msg_ss.str();

    if (!config_to_session_[key]) {
      if (config_to_futures_.find(key) != config_to_futures_.end()) {
        config_to_session_[key] =
          std::move(config_to_futures_[key].get());
      } else {
        LOG(ERROR) << "[ERROR] GetOdpsOpenStorageSession " << common_msg << " failed, I`m going to raise exception !";
        st.Assign(Status::kInternal, "GetOdpsOpenStorageSession " + common_msg + " failed, I`m going to raise exception !");
        return st;
      }
    }
    *session = config_to_session_[key].get();
    if (*session == nullptr) {
      LOG(ERROR) << "[ERROR] GetOdpsOpenStorageSession " << common_msg << " unexpectedly nullptr !";
      st.Assign(Status::kInternal, "GetOdpsOpenStorageSession " + common_msg + " session is unexpectedly nullptr !");
      return st;
    }
    //LOG(INFO) << "Successfully " << common_msg
    //          << ", sessionId: " << (*session)->session_id_;  // FIXME: log to another file
  }
  return st;
}

int64_t OdpsOpenStorageSession::GetSessionExpireTimestamp(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(config_to_open_storage_session_lock_);
    for (const auto& [key, session_ptr] : config_to_session_) {
        if (!session_ptr) {
            LOG(WARNING) << "[GetSessionExpireTimestamp] Null session detected for key: "
                         << "project='" << std::get<0>(key) << "', "
                         << "table='"   << std::get<1>(key)  << "', "
                         << "partition='" << std::get<2>(key) << "'";
            continue;
        }
        if (!session_ptr->initialized_) {
            LOG(WARNING) << "[GetSessionExpireTimestamp] Found uninitialized session for key: "
                         << "project='" << std::get<0>(key) << "', "
                         << "table='"   << std::get<1>(key)  << "', "
                         << "partition='" << std::get<2>(key) << "'";
            continue;
        }
        if (session_ptr->session_id_ == session_id) {
            return (int64_t)session_ptr->expiration_time_;
        }
    }
    return 0;
}



apsara::odps::sdk::storage_api::arrow_adapter::ArrowClient* OdpsOpenStorageSession::GetArrowClient(
                                                              const std::string& access_id,
                                                              const std::string& access_key,
                                                              const std::string& tunnel_endpoint,
                                                              const std::string& default_project,
                                                              int connect_timeout,
                                                              int rw_timeout) {
  apsara::odps::sdk::storage_api::AliyunAccount aliyun_account(access_id, access_key);
  apsara::odps::sdk::storage_api::Configuration configuration;
  configuration.SetDefaultProject(default_project);
  configuration.SetSocketConnectTimeout(connect_timeout);
  configuration.SetSocketTimeout(rw_timeout);
  configuration.SetAccount(aliyun_account);
  configuration.SetTunnelEndpoint(tunnel_endpoint);
  configuration.retryTimes = 300;

  //return std::make_shared<ArrowClient>(configuration);
  auto arrow_client = new apsara::odps::sdk::storage_api::arrow_adapter::ArrowClient(configuration);
  return arrow_client;
}

apsara::odps::sdk::storage_api::arrow_adapter::ArrowClient*
OdpsOpenStorageSession::GetArrowClient(const apsara::odps::sdk::storage_api::Configuration& configuration) {
  auto arrow_client = new apsara::odps::sdk::storage_api::arrow_adapter::ArrowClient(configuration);
  return arrow_client;
}

Status OdpsOpenStorageSession::CreateReadSession(std::string* session_id,
                                                 const std::string& access_id,
                                                 const std::string& access_key,
                                                 const std::string& tunnel_endpoint,
                                                 const std::string& project,
                                                 const std::string& table,
                                                 const std::vector<std::string>& required_partitions,
                                                 const std::vector<std::string>& required_data_columns,
                                                 const std::string& mode,
                                                 const std::string& default_project,
                                                 int connect_timeout,
                                                 int rw_timeout) {
  LOG(INFO) << "required_data_columns length: " << std::to_string(required_data_columns.size());
  apsara::odps::sdk::storage_api::arrow_adapter::ArrowClient* arrow_client
    = GetArrowClient(access_id, access_key, tunnel_endpoint,
                     default_project, connect_timeout, rw_timeout);

  std::vector<std::string> partition_spec;
  apsara::odps::sdk::storage_api::TableIdentifier table_identifier;
  table_identifier.project_ = project;
  table_identifier.table_ = table;

  std::string filter_predicate;

  apsara::odps::sdk::storage_api::TableBatchScanReq req;
  req.required_data_columns_ = std::move(required_data_columns);
  req.table_identifier_ = table_identifier;
  auto iter = std::find(required_partitions.begin(), required_partitions.end(), "");
  if (required_partitions.size() == 0 || iter != required_partitions.end()) {
    // nothing to do now, perhaps adding metrics to count none-partition tables.
  } else {
    req.required_partitions_ = std::move(required_partitions);  // here is partition table
  }

  if (mode == "size") {
    req.split_options_ = apsara::odps::sdk::storage_api::SplitOptions::GetDefaultOptions(apsara::odps::sdk::storage_api::SplitOptions::SIZE);
  } else if (mode == "row") {
    req.split_options_ = apsara::odps::sdk::storage_api::SplitOptions::GetDefaultOptions(apsara::odps::sdk::storage_api::SplitOptions::ROW_OFFSET);
  }

  apsara::odps::sdk::storage_api::TableBatchScanResp resp;
  resp.expiration_time_ = 60;
  Status st = Status();
  arrow_client->CreateReadSession(req, resp);
  nlohmann::json tmp_j = resp;
  if(resp.status_ != apsara::odps::sdk::storage_api::Status::OK) {
    if (resp.status_ == apsara::odps::sdk::storage_api::Status::WAIT) {  // CreateReadSession and CommitWriteSession may process the requst asynchronously
      LOG(WARNING) << "CreateReadSession may process the requst asynchronously: " << resp.error_message_;
      st.Assign(Status::kWait, "CreateReadSession may process the requst asynchronously: " + resp.error_message_);
    } else if (resp.status_ == apsara::odps::sdk::storage_api::Status::CANCELED) {
      LOG(ERROR) << "Failed to create read session, CANCELED: " << resp.error_message_;
      st.Assign(Status::kCancelled, "Failed to create read session, CANCELED: " + resp.error_message_);
    } else {
      //std::stringstream colomns;
      //for (auto it = req.required_data_columns_.cbegin(); it != req.required_data_columns_.cend(); ++it) {
      //  colomns << "     " << *it  << "\n";
      //}
      //LOG(INFO) << "Columns: \n" << colomns.str();
      LOG(ERROR) << "Failed to create read session for "
                 << "project: [" << project
                 << "], table: [" << table
                 << "], required_data_columns_: " << std::to_string(req.required_data_columns_.size())
                 << ", error message: " << resp.error_message_;
      st.Assign(Status::kInternal, "Fail to get read session: " + resp.error_message_);
    }
  }
  *session_id = resp.session_id_;
  // e.g.
  //  {
  //  "DataSchema": {
  //    "DataColumns": [],
  //    "PartitionColumns": []
  //  },
  //  "ExpirationTime": 1715067223411,
  //  "Message": "",
  //  "RecordCount": -1,
  //  "SessionId": "202405061533434c8b580b0000000502",
  //  "SessionStatus": "INIT",
  //  "SessionType": "BATCH_READ",
  //  "SplitsCount": -1,
  //  "SupportedDataFormat": [{
  //    "Type": "ARROW",
  //    "Version": "V5"
  //  }]
  //}
  LOG(INFO) << "resp: " << tmp_j.dump();
  delete arrow_client;
  return st;
}

Status OdpsOpenStorageSession::GetReadSession(std::string* session_def_str,
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
  apsara::odps::sdk::storage_api::arrow_adapter::ArrowClient* arrow_client
    = GetArrowClient(access_id, access_key, tunnel_endpoint,
                     default_project, connect_timeout, rw_timeout);

  apsara::odps::sdk::storage_api::TableIdentifier table_identifier;
  table_identifier.project_ = project;
  table_identifier.table_ = table;

  apsara::odps::sdk::storage_api::SessionReq req;
  req.session_id_ = session_id;
  req.table_identifier_ = table_identifier;

  apsara::odps::sdk::storage_api::TableBatchScanResp resp;
  Status st = Status();
  arrow_client->GetReadSession(req, resp);
  if(resp.status_ != apsara::odps::sdk::storage_api::Status::OK) {
    LOG(ERROR) << "Fail to get read session, resp.status_" << resp.status_
               << " error message: " << resp.error_message_;
    st.Assign(Status::kInternal, "Fail to get read session: " + resp.error_message_);
  }
  delete arrow_client;

  nlohmann::json j_resp = resp;
  if (j_resp.contains(kSessionStatus)) {
    const std::string& sessionStatus = j_resp[kSessionStatus];
    const std::string& error_msg = "SessionId: " + session_id + " status: " + sessionStatus;
    LOG(INFO) << error_msg;
    if (sessionStatus == kINIT) {
      st.Assign(Status::kWait, error_msg);
    } else if (sessionStatus == kNORMAL) {
      st.Assign(Status::kOk, error_msg);
    } else {
      st.Assign(Status::kInternal, error_msg);
    }
  } else {
    LOG(ERROR) << "table batch scan resp has no member: " + kSessionStatus
                + ", error message: " + resp.error_message_;
    st.Assign(Status::kInternal, "table batch scan resp has no member: " + kSessionStatus
                               + ", error message: " + resp.error_message_);
  }

  if (session_def_str == nullptr) {
    const std::string& error_msg = "get read session, session_def_str ptr is nullptr !";
    LOG(ERROR) << error_msg;
    st.Assign(Status::kInternal, error_msg);
    return st;
  }
  if (session_def == nullptr) {
    const std::string& error_msg = "get read session, session_def ptr is nullptr !";
    LOG(ERROR) << error_msg;
    st.Assign(Status::kInternal, error_msg);
    return st;
  }

  *session_def_str = j_resp.dump();
  *session_def = j_resp;
  return st;
}

std::string OdpsOpenStorageSession::timestampToReadableTime(long timestamp) {
    //change the data from long to second
    std::chrono::milliseconds ms(timestamp);
    std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(ms);

    std::time_t t = s.count();
    std::tm tm = *std::localtime(&t);

    //Formatting time "yyyy-MM-dd HH:mm:ss"
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");

    return oss.str();
}

Status OdpsOpenStorageSession::RefreshReadSessionBatch() {
  Status st = Status();
  st.Assign(Status::kOk, "Refresh success!");


  std::lock_guard<std::mutex> lock(config_to_open_storage_session_lock_);
  for (const auto&[key, tmp_session] : config_to_session_) {
    // const auto& key = session.first;
    // const auto& tmp_session = session.second;
    if (tmp_session) {
      if (!tmp_session->initialized_) {
        LOG(WARNING) << "[RefreshReadSessionBatch] Found uninitialized session for  key: "
                     << "project='" << std::get<0>(key) << "', "
                     << "table='"   << std::get<1>(key)  << "', "
                     << "partition='" << std::get<2>(key) << "'";
        continue;
      }
      LOG(INFO) << "session message: ("
                << std::get<0>(key) << ", " << std::get<1>(key)
                << ", " << std::get<2>(key) << ")";
      Status st_single = RefreshReadSession(
                           tmp_session->configuration_.account.GetId(),
                           tmp_session->configuration_.account.GetKey(),
                           tmp_session->configuration_.GetTunnelEndpoint(),
                           tmp_session->session_id_,
                           tmp_session->project_, tmp_session->table_,
                           tmp_session->configuration_.GetDefaultProject(),
                           tmp_session->configuration_.GetSocketConnectTimeout(),
                           tmp_session->configuration_.GetSocketTimeout(),
                           tmp_session.get());
      if (st_single.RefreshFailed()) {
        LOG(ERROR) << "RefreshReadSession failed!";
        st = st_single;
      }
    } else {
      LOG(WARNING) << "[RefreshReadSessionBatch] Null session detected for key: "
                   << "project='" << std::get<0>(key) << "', "
                   << "table='"   << std::get<1>(key)  << "', "
                   << "partition='" << std::get<2>(key) << "'";
      st.Assign(Status::kRefreshFailed, "session_instance lost!");
    }
  }

  return st;
}

using apsara::odps::sdk::storage_api::GetHaloMetrics;
int64_t OdpsOpenStorageSession::GetHaloAgentMetric(const char* halo_endpoint, int type){
    static std::unordered_map<std::string, GetHaloMetrics> EndpointMap;
    // EndpointMap 原先无锁, 靠 Python GIL 隐式串行化. 显式加 mutex 增强线程安全,
    // 使多线程/多上下文调用此函数时不依赖外部锁保证正确性.
    // 同时为切换 free-threaded Python (3.14t+) 后 GIL 不再存在的场景提供安全保障.
    static std::mutex endpoint_map_mutex;

    std::string endpoint = halo_endpoint;
    GetHaloMetrics* metric_cli_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(endpoint_map_mutex);
        auto it = EndpointMap.find(endpoint);
        if (it == EndpointMap.end()) {
            auto res = EndpointMap.emplace(endpoint, GetHaloMetrics(endpoint));
            it = res.first;
        }
        metric_cli_ptr = &(it->second);
    }
    switch (type) {
    case 1:{
        uint64_t resp = metric_cli_ptr->GetReadBytes();
        return static_cast<int64_t>(resp);
    }
    case 2:{
        uint64_t resp = metric_cli_ptr->GetAndResetReadBytes();
        return static_cast<int64_t>(resp);
    }
    default:
        return -10;
    }
}

Status OdpsOpenStorageSession::RefreshReadSession(const std::string& access_id,
                                                  const std::string& access_key,
                                                  const std::string& tunnel_endpoint,
                                                  const std::string& session_id,
                                                  const std::string& project,
                                                  const std::string& table,
                                                  const std::string& default_project,
                                                  int connect_timeout,
                                                  int rw_timeout,
                                                  OdpsOpenStorageSession* tmp_session) {
  apsara::odps::sdk::storage_api::arrow_adapter::ArrowClient* arrow_client
    = GetArrowClient(access_id, access_key, tunnel_endpoint,
                     default_project, connect_timeout, rw_timeout);

  apsara::odps::sdk::storage_api::TableIdentifier table_identifier;
  table_identifier.project_ = project;
  table_identifier.table_ = table;

  apsara::odps::sdk::storage_api::SessionReq req;
  req.session_id_ = session_id;
  req.table_identifier_ = table_identifier;

  Status st = Status();
  st.Assign(Status::kRefreshFailed, "Did nothing!");
  bool is_refreshed = false;
  nlohmann::json j_resp;

  // here wait try_times * 10 seconds, for session_id is ready.
  // TODO: 仿照ReadBatch和ReadBatchimpl, 优化refresh逻辑
  int each_loop_wait_seconds = 5;
  for (int i = 10; i >= 0; --i) {
    req.refresh_ = true;
    apsara::odps::sdk::storage_api::TableBatchScanResp resp;
    apsara::odps::sdk::storage_api::TableBatchScanResp resp_refresh;
    arrow_client->GetReadSession(req, resp_refresh);
    if (resp_refresh.status_ != apsara::odps::sdk::storage_api::Status::OK) {
      req.refresh_ = false;
      arrow_client->GetReadSession(req, resp);
      j_resp = resp;
      if (resp.status_ == apsara::odps::sdk::storage_api::Status::WAIT) {
        LOG(INFO) << "  Wait for the session status to change to NORMAL!"
                  << "  Single wait time: [" << std::to_string(each_loop_wait_seconds) << "] seconds.";
        std::this_thread::sleep_for(std::chrono::seconds(each_loop_wait_seconds));
      } else if (resp.status_ != apsara::odps::sdk::storage_api::Status::OK) {
        LOG(ERROR) << "Fail to refresh session, resp.status_: " << resp.status_
                   << " error message: " << resp.error_message_;
        st.Assign(Status::kRefreshFailed, "Fail to refresh session: " + resp.error_message_);
        break;
      } else {
        if (is_refreshed) {
          LOG(INFO) << "this session_id: " << req.session_id_
                    << "  Successful life extension and the status is NORMAL!";
          st.Assign(Status::kOk, "Refresh success!");
        } else {
          LOG(INFO) << "this session_id: " << req.session_id_
                    << " lifecycle is more than 12h and no need to refresh";
          st.Assign(Status::kNoNeedRefresh, "Registration is less than 12 hours old, no need to refresh.");
        }
        break;
      }
    } else {
      is_refreshed = true;
      LOG(INFO) << "this session_id: " << req.session_id_ << " extend lifecycle successfully" ;
    }
  }
  delete arrow_client;

  if (!tmp_session) {
    LOG(WARNING) << "this session_id: " << req.session_id_ << " is unregistered in manager" ;
    return st;
  }

  if (j_resp.contains(kExpirationTime)) {
    tmp_session->expiration_time_ = j_resp[kExpirationTime].get<long>();
    if (is_refreshed) {
      LOG(INFO) << "new expiration_time: "
                << timestampToReadableTime(tmp_session->expiration_time_)
                << "  timestamp: " << tmp_session->expiration_time_;
    } else {
      LOG(INFO) << "expiration_time do not change: "
                << timestampToReadableTime(tmp_session->expiration_time_)
                << "  timestamp: " << tmp_session->expiration_time_;
    }
  } else {
    tmp_session->expiration_time_ = -1;
    LOG(ERROR) << "cannot find expiration_time for this session: " << req.session_id_ ;
  }
  tmp_session->session_def_ = j_resp;
  return st;
}

Status OdpsOpenStorageSession::CreatAndGetReadSession(bool register_session_id,
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
                                                      const std::string& mode,
                                                      const std::string& default_project,
                                                      int connect_timeout,
                                                      int rw_timeout,
                                                      int try_times) {
  Status st = Status();
  if (register_session_id) {
    LOG(WARNING) << "Normal read session id, going to check expiration or refresh...";
    // TODO: 需考虑 检查sessionId 是否expire, 如expire 则需要create read session或者refresh
  } else {
    LOG(WARNING) << "No normal read session id, going to create read session.";
    st = CreateReadSession(session_id, access_id, access_key, tunnel_endpoint,
                           project, table, required_partitions, required_data_columns,
                           mode, default_project, connect_timeout, rw_timeout);
    if (!st.Ok() and !st.Wait()) {
      return st;
    }
  }

  // here wait try_times * kSessionCreationLoopWaitSeconds seconds, for session_id is ready.
  for (int i = try_times; i >= 0; --i) {
    st = GetReadSession(session_def_str, session_def, access_id, access_key, tunnel_endpoint, *session_id,
                        project, table, default_project, connect_timeout, rw_timeout);
	if (st.Ok()) {
      LOG(INFO) << "SessionId: [" << *session_id << "] works.";
      return st;
    } else {
      LOG(WARNING) << "SessionId: [" << *session_id << "] not works, wait [" << std::to_string(i)
                   << "] * [" << std::to_string(kSessionCreationLoopWaitSeconds) << "] seconds.";
      std::this_thread::sleep_for(std::chrono::seconds(kSessionCreationLoopWaitSeconds));
    }
  }
  const std::string& error_msg = "SessionId: [" + *session_id +
                                 "] failed to work after: [" + std::to_string(try_times) +
                                 "] times trying.";
  LOG(ERROR) << error_msg;
  st.Assign(Status::kInternal, error_msg);
  return st;
}

bool OdpsOpenStorageSession::IsInitialized() {
  return initialized_;
}

// 需要GetReadSession 然后解析json
long OdpsOpenStorageSession::GetTableSize() {
  //long ret = 0;
  //{
  //  std::lock_guard<std::mutex> lock(config_to_open_storage_session_lock_);
  //  ret = record_count_;
  //}
  //return ret;
  //LOG(INFO) << "Project: " << project_
  //          << " table: "  << table_
  //          << " partition: "  << partition_spec_
  //          << " table size: " << std::to_string(record_count_);
  return record_count_;
}

std::unordered_map<std::string, std::string> OdpsOpenStorageSession::GetSchema() {
  auto st = ExtractSessionDef(&session_def_);
  return schema_;
}

// 和之前tunnel session明显不同之处:
// 1. tunnel session是创建reader的时候指定columns
// 2. open storage是创建session的时候指定columns
std::shared_ptr<apsara::odps::sdk::storage_api::arrow_adapter::Reader>
OdpsOpenStorageSession::CreateOpenStorageReader(long start, long end,
                                                int max_batch_rows, int max_batch_raw_size,
                                                int cache_size,
                                                bool data_columns_unordered,
                                                const std::string& reader_name,
                                                const std::vector<std::string>& input_columns) {
  openstorage::DeferUpdater updater(metric_reporter_, openstorage::MetricName::ReadRows, 1);
  apsara::odps::sdk::storage_api::TableIdentifier table_identifier;
  table_identifier.project_ = project_;
  table_identifier.table_ = table_;

  apsara::odps::sdk::storage_api::ReadRowsReq req;
  req.table_identifier_ = table_identifier;
  req.session_id_ = session_id_;
  req.max_batch_rows_ = max_batch_rows;
  if (max_batch_raw_size > 0) {
    req.max_batch_raw_size_ = max_batch_raw_size;
  }

  req.compression_ = static_cast<apsara::odps::sdk::storage_api::Compression::type>(compression_type_);

  req.row_index_ = start;
  req.row_count_ = end - start;
  req.data_columns_unordered_ = data_columns_unordered ? "true" : "false";
  req.data_columns_ = input_columns;
  std::shared_ptr<apsara::odps::sdk::storage_api::arrow_adapter::ArrowClient> arrow_client;
  arrow_client.reset(GetArrowClient(configuration_));

  auto reader = arrow_client->ReadRows(req, cache_size);
  // metric_reporter_->UpdateReadRows(1, latency_ms);

  return reader;
}


/**
e.g.
{
    "DataSchema": {
        "DataColumns": [{
                "Comment": "",
                "Name": "sample_id",
                "Nullable": true,
                "Type": "string"},
            {
                "Comment": "",
                "Name": "label",
                "Nullable": true,
                "Type": "array<double>"},
            {
                "Comment": "",
                "Name": "client_type",
                "Nullable": true,
                "Type": "array<bigint>"},
            {
                "Comment": "",
                "Name": "pay_seq_is_p4p",
                "Nullable": true,
                "Type": "array<array<double>>"}],
        "PartitionColumns": [{
                "Comment": "",
                "Name": "ds",
                "Nullable": true,
                "Type": "string"},
            {
                "Comment": "",
                "Name": "tag",
                "Nullable": true,
                "Type": "string"}]},
    "ExpirationTime": 1713465184134,
    "Message": "",
    "RecordCount": 1568854778,
    "SessionId": "202404180233044c8b580b00001ea902",
    "SessionStatus": "NORMAL",
    "SessionType": "BATCH_READ",
    "SplitsCount": -1,
    "SupportedDataFormat": [{
            "Type": "ARROW",
            "Version": "V5"}]}
**/
Status OdpsOpenStorageSession::ExtractSessionDef(nlohmann::json* session_def) {
  auto st = Status();
  if (session_def == nullptr) {
    LOG(ERROR) << "session_def is nullptr";
    st.Assign(Status::kInternal, "session_def is nullptr");
    return st;
  }
  const std::string& session_def_str = session_def->dump(2);
  if (session_def->contains(kRecordCount)) {
    record_count_ = (*session_def)[kRecordCount].get<long>();
    if (record_count_ == -1) {
      LOG(ERROR) << " '" << kRecordCount << "' value is -1, impossible, session_def: "
                 << session_def_str;
      st.Assign(Status::kInternal, " '" + kRecordCount + "' value is -1, impossible");
      return st;
    }
  } else {
    LOG(ERROR) << "no '" << kRecordCount << "' in session_def json: "
               << session_def_str;
    st.Assign(Status::kInternal, "no '" + kRecordCount + "' in session_def json");
    return st;
  }
  if (session_def->contains(kExpirationTime)) {
    expiration_time_ = (*session_def)[kExpirationTime].get<long>();
    if (expiration_time_ == -1) {
      LOG(ERROR) << " '" << kExpirationTime << "' value is -1, impossible, session_def: "
                 << session_def_str;
      st.Assign(Status::kInternal, " '" + kExpirationTime + "' value is -1, impossible");
      return st;
    }
  } else {
    LOG(ERROR) << "no '" << kExpirationTime << "' in session_def json: "
               << session_def_str;
    st.Assign(Status::kInternal, "no '" + kExpirationTime + "' in session_def json");
    return st;
  }
  if (session_def->contains(kSessionId)) {
    session_id_ = (*session_def)[kSessionId].get<std::string>();
    if (session_id_ == "") {
      LOG(ERROR) << " '" << kSessionId << "' value is empty, impossible, session_def: "
                 << session_def_str;
      st.Assign(Status::kInternal, " '" + kSessionId + "' value is empty, impossible");
      return st;
    }
  } else {
    LOG(ERROR) << "no '" << kSessionId << "' in session_def json: "
               << session_def_str;
    st.Assign(Status::kInternal, "no '" + kSessionId + "' in session_def json");
    return st;
  }

  st = ExtractSchema(session_def);
  return st;
}

Status OdpsOpenStorageSession::ExtractSchema(nlohmann::json* session_def){
  auto st = Status();
  const std::string& session_def_str = session_def->dump(2);
  if (session_def->contains(kDataSchema)) {
  } else {
    LOG(ERROR) << "no '" << kDataSchema << "' in session_def json: "
               << session_def_str;
    st.Assign(Status::kInternal, "no '" + kDataSchema + "' in session_def json");
    return st;
  }
  return st;
}


Status OdpsOpenStorageSession::InitOpenStorageSession(const std::string& access_id,
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
                                                    const long expiration_time,
                                                    const long record_count,
                                                    const std::string& session_def_str) {
  Status st;
  if (register_session_id) {
    if (!register_light) {
      LOG(INFO) << "Register session id: [" << session_id << "]";
    }
    session_id_ = session_id;
  }
  project_ = project;
  table_ = table;
  partition_spec_ = partition_spec;
  physical_partitions_ = physical_partitions;
  required_data_columns_ = required_data_columns;
  apsara::odps::sdk::storage_api::AliyunAccount aliyun_account(access_id, access_key);
  configuration_.SetDefaultProject(default_project);
  configuration_.SetSocketConnectTimeout(connect_timeout);
  configuration_.SetSocketTimeout(rw_timeout);
  configuration_.SetAccount(aliyun_account);
  configuration_.SetTunnelEndpoint(tunnel_endpoint);
  configuration_.retryTimes = kConfigurationRetryTimes;

  if (register_light) {
    try {
      session_def_ = nlohmann::json::parse(session_def_str);
    } catch (const nlohmann::json::exception& je) {
      LOG(ERROR) << "Init open storage session light, JSON Error: " << je.what();
    } catch (const std::exception& e) {
      LOG(ERROR) << "Init open storage session light, Standard Error: " << e.what();
    }
    session_id_ = session_id;
    expiration_time_ = expiration_time;
    record_count_ = record_count;
    initialized_ = true;
    InitCompressionType();
    // metric_reporter_.reset(new openstorage::MetricReporter("openstorage_reader",
    //                                                       {   {"session_id", session_id_},
    //                                                           {"odps_project", project_},
    //                                                           {"odps_table", table_},
    //                                                           {"odps_partition", partition_spec_}
    //                                                       } ));
    {
        std::lock_guard<std::mutex> lg(monitor_factory_mutex_);
        if (monitor_factory_ == nullptr) {
            monitor_factory_ = recis::monitor::Factory::MakeInstance("", std::chrono::seconds(10));
        }
    }
    metric_reporter_ = std::make_pair<std::shared_ptr<recis::monitor::Client>, std::shared_ptr<recis::monitor::PointTag>>(
        monitor_factory_->get_client("xdl.metrics.openstorage_reader"),
        std::make_shared<recis::monitor::PointTag>(std::map<std::string, std::string>{
            {"multiprocess_seq", std::to_string(openstorage::get_child_order())},
            {"session_id", session_id_},
            {"ds_project", project_},
            {"ds_table", table_},
            {"ds_name", "odps_openstorage"}
            // {"odps_partition", partition_spec_}        
        })
    );
    return st;
  }

  std::string tmp_session_def_str;
  st = CreatAndGetReadSession(register_session_id,
                              &session_id_, &tmp_session_def_str, &session_def_,
                              access_id, access_key,
                              tunnel_endpoint, project, table,
                              physical_partitions,
                              required_data_columns, mode, default_project,
                              connect_timeout, rw_timeout);

  std::stringstream pp_oss;
  for (int part_size = 0; part_size < physical_partitions.size(); ++part_size) {
	const auto& phy_part = physical_partitions[part_size];
    if (part_size < physical_partitions.size() - 1) {
      pp_oss << phy_part << ",";
    } else {
      pp_oss << phy_part;
    }
  }
  const std::string& phy_part_list = pp_oss.str();
  std::stringstream common_msg_ss;
  common_msg_ss << "Init odps open storage session with: "
                << "tunnel_endpoint=[" << tunnel_endpoint << "] "
                << "project=[" << project << "] "
                << "table=[" << table << "] "
                << "partition_spec=[" << partition_spec << "] "
                << "physical_partitions=[" << phy_part_list << "] "
                << "connect_timeout=[" << std::to_string(connect_timeout) << "]s "
                << "rw_timeout=[" << std::to_string(rw_timeout) << "]s ";

  const std::string& common_msg = common_msg_ss.str();
  if (!st.Ok()) {
    LOG(ERROR) << "Failed to " << common_msg
               << " CreatAndGetReadSession failed, session_id: " << session_id_;
    initialized_ = false;
    // TODO: 这里创建失败不会返回给python 而是直接退出进程. 故只能从cc报fail, py报succ 存在逻辑mismatch
    // 后续可透出status统一到py汇报.
    // std::shared_ptr<openstorage::MetricReporter> temporary_reporter;
    // temporary_reporter.reset(new openstorage::MetricReporter("openstorage_session_create", 
    //                                                     {   
    //                                                     } ));
    std::string app_id = std::getenv("APP_ID") ? std::getenv("APP_ID") : "null";
    std::string sink_address_ip = getenv("KMONITOR_SINK_ADDRESS") ? getenv("KMONITOR_SINK_ADDRESS") : "localhost";
    // std::string sink_address_ip = getenv("KUBERNETES_NODE_IP") ? getenv("KUBERNETES_NODE_IP") : "localhost";
    std::string scheduler_queue = std::getenv("SCHEDULER_QUEUE") ? std::getenv("SCHEDULER_QUEUE") : "null";
    {
        std::lock_guard<std::mutex> lg(monitor_factory_mutex_);
        if (monitor_factory_ == nullptr) {
            monitor_factory_ = recis::monitor::Factory::MakeInstance("", std::chrono::seconds(10));
        }
    }
    auto client_ = monitor_factory_->get_client("xdl.metrics.openstorage_session_create");
    auto pTag_ = std::make_shared<recis::monitor::PointTag>(std::map<std::string, std::string>{
        {"multiprocess_seq", std::to_string(openstorage::get_child_order())},
        {"session_id", "null"},
        {"ds_project", project_},
        {"ds_table", table_},
        {"ds_name", "odps_openstorage"},
        // {"odps_partition", partition_spec_},
        {"code", std::to_string(openstorage::MetricStatus::REQUEST_ERROR)},
        {"status", "fail"}
    });
    client_->report("create_qps", 1, pTag_, recis::monitor::PointType::kGauge);
    // openstorage::MetricReporter::UpdateImmediate("create_qps", 1,
    //                                             {   {"session_id", "null"},
    //                                                 {"app_id", app_id},
    //                                                 {"host_ip", sink_address_ip},
    //                                                 {"scheduler_queue", scheduler_queue},
    //                                                 {"odps_project", project_},
    //                                                 {"odps_table", table_},
    //                                                 {"odps_partition", partition_spec_},
    //                                                 {"code", std::to_string(openstorage::MetricStatus::REQUEST_ERROR)},
    //                                                 {"status", "fail"}
    //                                             },
    //                                             "openstorage_session_create");
    std::this_thread::sleep_for(std::chrono::seconds(10)); // 10=kmonitor::MetricLevel::NORMAL 聚合粒度,避免打点失败; 后续进程会退出
    return st;
  }
  st = ExtractSessionDef(&session_def_);
  if (!st.Ok()) {
    LOG(ERROR) << "Failed to " << common_msg
               << " ExtractSessionDef failed, session_id: " << session_id_;
    initialized_ = false;
    return st;
  }
  initialized_ = true;
  InitCompressionType();
  // metric_reporter_.reset(new openstorage::MetricReporter("openstorage_reader", 
  //                                                       {   {"session_id", session_id_},
  //                                                           {"odps_project", project_},
  //                                                           {"odps_table", table_},
  //                                                           {"odps_partition", partition_spec_}
  //                                                       } ));
    {
        std::lock_guard<std::mutex> lg(monitor_factory_mutex_);
        if (monitor_factory_ == nullptr) {
            monitor_factory_ = recis::monitor::Factory::MakeInstance("", std::chrono::seconds(10));
        }
    }
  metric_reporter_ = std::make_pair<std::shared_ptr<recis::monitor::Client>, std::shared_ptr<recis::monitor::PointTag>>(
    monitor_factory_->get_client("xdl.metrics.openstorage_reader"),
    std::make_shared<recis::monitor::PointTag>(std::map<std::string, std::string>{
        {"multiprocess_seq", std::to_string(openstorage::get_child_order())},
        {"session_id", session_id_},
        {"ds_project", project_},
        {"ds_table", table_},
        {"ds_name", "odps_openstorage"}
        // {"ds_partition", partition_spec_}        
    })
  );
  LOG(INFO) << "Successfully initialized session, "
            << common_msg;
  return st;
}

OdpsOpenStorageSession::OdpsOpenStorageSession() {}

OdpsOpenStorageSession::OdpsOpenStorageSession(
                           const std::string& session_id,
                           const std::string& project,
                           const std::string& table,
                           const std::string& partition,
                           apsara::odps::sdk::storage_api::Configuration& configuration,
                           std::vector<std::string>& required_data_columns,
                           const std::string& mode,
                           const std::string& default_project):
    session_id_(session_id), project_(project), table_(table),
    partition_spec_(partition), configuration_(configuration),
    required_data_columns_(required_data_columns),
    mode_(mode), default_project_(default_project) {

  initialized_ = true;
  InitCompressionType();
  LOG(INFO) << "Initialized session id: " << session_id_
            << " with project: "<< project_
            << " table: " << table_
            << " partition: " << partition_spec_
            << " mode: " << mode_
            << " default_project" << default_project_
            << " required_data_columns size: " << std::to_string(required_data_columns.size());
}

OdpsOpenStorageSession::~OdpsOpenStorageSession() {}

} // namespace tf
} // namespace algo
} // namespace tunnel
} // namespace odps
} // namespace apsara
