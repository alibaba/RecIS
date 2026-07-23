//#ifdef TF_ENABLE_ODPS_COLUMN

#include "column-io/open_storage/plugin/odps_open_storage_arrow_reader.h"
#include "column-io/open_storage/plugin/odps_open_storage_session.h"
#include "column-io/open_storage/common-util/logging.h"
#include "column-io/open_storage/common-util/status.h"
#include "column-io/open_storage/common-util/common_util.h"

//#include "nlohmann/json.hpp"
#include "storage_api/thirdparty/nlohmann/json.hpp"
#include "include/odps_exception.h"

#include <unordered_map>
#include <memory>
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
using xdl::paiio::third_party::common_util::StrSplit;
using xdl::paiio::third_party::common_util::strip;
//using namespace apsara::odps::sdk::storage_api;
//using namespace apsara::odps::sdk::storage_api::arrow_adapter;
namespace {

const char* kProjectNameTag="project_name";
const char* kTableNameTag="talbe_name";
const char* KPartitionSpecTag="partition_spec";
const char* kStartTag="start";
const char* kEndTag="end";
const int kDefaultReaderRetryTimes = 5;
const int kDefaultReaderRetryWaitSeconds = 60;

int GetReaderRetryTimes() {
  const char* reader_retry_times = getenv("OPEN_STORAGE_READER_RETRY_TIMES");
  if (reader_retry_times == nullptr) {
    return kDefaultReaderRetryTimes;
  }
  return atoi(reader_retry_times);
}
int GetReaderRetryWaitSeconds() {
  const char* reader_wait_seconds = getenv("OPEN_STORAGE_READER_RETRY_WAIT_SECONDS");
  if (reader_wait_seconds == nullptr) {
    return kDefaultReaderRetryWaitSeconds;
  }
  return atoi(reader_wait_seconds);
}

int GetReaderMaxBatchRows() {
  const char* max_batch_rows = getenv("OPEN_STORAGE_READER_MAX_BATCH_ROWS");
  if (max_batch_rows == nullptr) {
    return -1;
  }
  int res = atoi(max_batch_rows);
  if (res <= 0) {
    LOG(INFO) << "OPEN_STORAGE_READER_MAX_BATCH_ROWS value <=0 is invalid and useless.";
    res = -1;
  }
  return res;
}
int GetReaderMaxBatchRawSize() {
  const char* max_batch_raw_size = getenv("OPEN_STORAGE_READER_MAX_BATCH_RAW_SIZE");
  if (max_batch_raw_size == nullptr) {
    return -1;
  }
  int res = atoi(max_batch_raw_size);
  if (res <= 0) {
    LOG(INFO) << "OPEN_STORAGE_READER_MAX_BATCH_RAW_SIZE value <=0 is invalid and useless.";
    res = -1;
  }
  return res;
}
bool GetReaderDataColumnsUnordered() {
  const char* data_columns_unordered = getenv("OPEN_STORAGE_READER_DATA_COLUMNS_UNORDERED");
  if (data_columns_unordered == nullptr) {
    return true;
  }
  if (std::string(data_columns_unordered) != "false") {
    return true;
  }
  return false;
}

// 解析 tunnel 传参格式的表名
void ParseTunnelTableFormat(std::unordered_map<std::string, std::string>& ret,
                            const std::string& config) {
    // 兼容tunnel传参格式
    std::vector<std::string> parts;
    for (const std::string& s : StrSplit(ret[kTableNameTag], ".", false)) {
        parts.emplace_back(strip(s));  // 你的 strip(const std::string&)：去掉首尾空白
    }

    if (parts.size() == 1) {
        ret[kTableNameTag] = parts[0];
    }
    else if (parts.size() == 2) {
        if (!parts[0].empty()) {
            ret[kProjectNameTag] = parts[0];
        }
        if (!parts[1].empty()) {
            ret[kTableNameTag]  = parts[1];
        }
    }
    else if (parts.size() == 3) {
        if (!parts[0].empty()) {
            ret[kProjectNameTag] = parts[0];
        }
        if (!parts[2].empty()) {
            ret[kTableNameTag]  = parts[2];
        }
    }
    else {
        throw std::invalid_argument("Invalid table path " + config);
    }
}

void ParseConfig(const std::string& config, std::unordered_map<std::string, std::string>& ret) {
  /*
   * config is a path with format odps://project_name/tables/table_name/partitions?param=vale&
   * */

  std::string::size_type old_pos = 0;
  std::string::size_type new_pos = 0;

  //extract project_name
  new_pos = config.find("//");
  new_pos += 2;
  old_pos = new_pos;
  new_pos = config.find("/", new_pos);
  ret[kProjectNameTag] = config.substr(old_pos, new_pos - old_pos);

  //extract table_name
  new_pos = config.find("tables");
  new_pos += 7;
  old_pos = new_pos;
  new_pos = config.find("/", new_pos);
  auto tmp_pos = config.find("?");
  bool is_partition_set = false;
  if (new_pos == std::string::npos) {  // for none-partition table
    if (tmp_pos == std::string::npos) {  // for none-partition table without ?start=xx&end=xx
      ret[kTableNameTag] = config.substr(old_pos, config.size() - old_pos);
      ret[KPartitionSpecTag] = "";
      is_partition_set = true;
                  
      // 兼容tunnel传参格式
      ParseTunnelTableFormat(ret, config);
      return;
    } else {  // for none-partition table with ?start=xx&end=xx
      auto s = config.substr(old_pos, tmp_pos);
      ret[kTableNameTag] = config.substr(old_pos, tmp_pos - old_pos);
      ret[KPartitionSpecTag] = "";
      is_partition_set = true;
    }
  } else {
    ret[kTableNameTag] = config.substr(old_pos, new_pos - old_pos);
  }

  // 兼容tunnel传参格式
  ParseTunnelTableFormat(ret, config);

  //extract partitions
  new_pos += 1;
  old_pos = new_pos;
  if (!is_partition_set) {
    new_pos = config.find("?", new_pos);
    if (new_pos == std::string::npos) {
      ret[KPartitionSpecTag] = config.substr(old_pos, config.size() - old_pos);
    } else {
      ret[KPartitionSpecTag] = config.substr(old_pos, new_pos - old_pos);
    }
  }
  //extract param
  if (new_pos == std::string::npos) {
    return;
  }
  new_pos = config.find("?");
  new_pos += 1;
  old_pos = new_pos;
  std::string param_str = config.substr(old_pos, config.size() - old_pos);
  auto param_str_vec = StrSplit(param_str, "&", false);
  for (auto& param: param_str_vec) {
    auto single_param_vec = StrSplit(param, "=", false);
    ret[single_param_vec[0]] = single_param_vec[1];
  }
}
} // anonymous namespace

class OdpsOpenStorageArrowReader::OdpsOpenStorageArrowReaderImpl {
  public:
    OdpsOpenStorageArrowReaderImpl(const std::string& name, const std::string& path_str, int max_batch_rows):
                                    name_(name), path_str_(path_str) {
      std::call_once(param_init_flag_, []{ initParamFromEnv(); } );
      local_max_batch_rows_ = (env_max_batch_rows_ > 0) ? env_max_batch_rows_ : max_batch_rows;
    }

    ~OdpsOpenStorageArrowReaderImpl() {
      Close();
      reader_.reset();
    }

    Status InitReader(const std::unordered_map<std::string, std::string>& config, const std::vector<std::string>& input_columns) {
      config_ = config;
      start_ = 0;
      auto st = Status();
      try {
        st = OdpsOpenStorageSession::GetOdpsOpenStorageSession(
                                                          config_[kProjectNameTag],
                                                          config_[kTableNameTag],
                                                          config_[KPartitionSpecTag],
                                                          &open_storage_session_);
        if (!st.Ok()) {
          LOG(ERROR) << "open storage arrow reader impl init reader, initializing session error !";
          st.Assign(Status::kInternal, "initializing session error !");
          return st;
        }
        if (!open_storage_session_->IsInitialized()) {
          LOG(ERROR) << "open storage arrow reader impl init reader, session is not initialized !";
          st.Assign(Status::kInternal, "session is not initialized !");
          return st;
        }
        auto table_size = open_storage_session_->GetTableSize();
        if (config_.find(kEndTag) != config_.end()) {
          size_t pos = 0;
          uint64_t tmp_end = std::stol(config_[kEndTag], &pos);
          if ( tmp_end < 0 || tmp_end > table_size) {
            st.Assign(Status::kInternal, "end is out of range");
            return st;
          } else {
            end_ = tmp_end;
          }
        } else {
          end_ = table_size;
        }

        if (config_.find(kStartTag) != config_.end()) {
          size_t pos = 0;
          uint64_t tmp_start = std::stol(config_[kStartTag], &pos);
          if (tmp_start < 0) {
            st.Assign(Status::kInternal, "start should large than zero");
            return st;
          }
          start_ = tmp_start;
        }

        if (start_ >= end_) {
          st.Assign(Status::kInternal, "start should be less than end");
          return st;
        }
        //WARN: input_columns会导致url过长. 请在极端场景下关闭选列功能(取消该入参), 并等待odps侧修复问题
        reader_ = open_storage_session_->CreateOpenStorageReader(
                                           start_, end_, local_max_batch_rows_,
                                           env_max_batch_raw_size_,
                                           cache_size,
                                           env_data_columns_unordered_, name_,
                                           input_columns);
        metric_reporter_ = open_storage_session_->metric_reporter_; // reader_ will re-use session's metric
      } catch (apsara::odps::sdk::OdpsException& e){
        st.Assign(Status::kInternal, e.what());
        return st;
      }
      cur_ = start_;
      initialized_ = true;
      return st;
    }

    Status ReadBatchImpl(std::shared_ptr<arrow::RecordBatch>& batch) {
      auto st = Status();
      openstorage::DeferUpdater updater(metric_reporter_, openstorage::MetricName::ReadBatchImpl, 1);
      try {
        // auto start_time = std::chrono::steady_clock::now();
        bool success = reader_->Read(batch);
        // auto end_time = std::chrono::steady_clock::now();
        // auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // std::cout << "[DEBUG] ReadBatchImpl elapsed_time milliseconds: " << elapsed_time.count() << std::endl;
        if (!success) {
          auto reader_st = reader_->GetStatus();
          const std::string& errorMsg = reader_->GetErrorMessage();
          const std::string& requestId = reader_->GetRequestID();
          if(reader_st != apsara::odps::sdk::storage_api::Status::OK){
            LOG(ERROR) << "Read rows error: [" << errorMsg
                       << "], reader_->GetStatus(): [" << std::to_string(reader_st)
                       << "], reader_->GetRequestID(): [" << requestId
                       << "], start_: [" << std::to_string(start_)
                       << "], end_: [" << std::to_string(end_)
                       << "], cur_: [" << std::to_string(cur_)
                       << "], time cost(ms): [" << updater.GetElapsedTimeMs()
                       << "].";
            st.Assign(Status::kInternal, "Status: [" + std::to_string(reader_st)
                                       + "], RequestID: [" + requestId + "].");
            updater.SetSuccess(false);
          } else {
            //LOG(INFO) << "[INFO] Normal OutOfRange: " << path_str_;  // FIXME: log to another file
            st.Assign(Status::kOutOfRange, "OutOfRange");
          }
          // metric_reporter_->UpdateReadBatch(1, elapsed_time.count(), reader_st == apsara::odps::sdk::storage_api::Status::OK); // TODO: 实现mutableMetric风格汇报接口, 汇报st详细值
          return st;
        } else {
          if (batch->num_rows() == 0) {
            if (cur_ == end_) {
              LOG(WARNING) << "Got empty batch at end of range.";
              st.Assign(Status::kOutOfRange, "OutOfRange");
            } else {
              LOG(ERROR) << "Read rows invalid, cur_: [" << std::to_string(cur_)
                         << "] not equal end_: [" << std::to_string(end_)
                         << "] but got empty batch!";
              st.Assign(Status::kInternal, "Read rows invalid, cur_: [" + std::to_string(cur_)
                                         + "] not equal end_: [" + std::to_string(end_)
                                         + "] but got empty batch!");
              updater.SetSuccess(false);
            }
            return st;
          }
        }
        // metric_reporter_->UpdateReadBatch(1, elapsed_time.count(), success);// NOTE: 此为供openstorage panel统计用点(成功/失败均汇报). 与基于tf的(DataIOMetrics)metric_->Update(data->num_rows()不混淆
        if (!record_batch_schema_) {
          record_batch_schema_ = batch->schema();
        }
        cur_ += batch->num_rows(); 
      } catch (apsara::odps::sdk::OdpsException& e) {
        st.Assign(Status::kInternal, e.what());
        // updater.Key()->kmetric_tag = &openstorage::MetricTag::Fail;
        updater.SetSuccess(false);
        return st;
      }
      return st;
    }

    Status ReadBatch(std::shared_ptr<arrow::RecordBatch>& batch) {
      openstorage::DeferUpdater updater(metric_reporter_, openstorage::MetricName::ReadBatch, 1);
      Status st;
      int retry_idx= -1, retry_wait_sec= GetReaderRetryWaitSeconds();
      do{
        retry_idx++;
        st = ReadBatchImpl(batch);
        if (st.Ok() || st.GetCode() == Status::kOutOfRange) {
            return st;
        }
        st = Seek(cur_);
        if (!st.Ok()) {
            LOG(ERROR) << "[ERROR] ReadBatch read failed and seek failed. ";
            updater.SetSuccess(false);
            return st;
        }
        // sleep and retry another time, maybe need log something.
        LOG(WARNING) << "[WARNING] ReadBatch read failed at retried times: [" << retry_idx
                    << "], going to wait [" << retry_wait_sec * retry_idx << "] seconds";
        std::this_thread::sleep_for(std::chrono::seconds(retry_wait_sec * retry_idx));
      } while( retry_idx < GetReaderRetryTimes() );
      LOG(ERROR) << "[ERROR] ReadBatch read failed, stop retrying, total times: " << GetReaderRetryTimes();
      updater.SetSuccess(false);
      return st;
    }

    Status Seek(size_t pos) {
      auto st = Status();
      if ( pos < start_||pos > end_) {
        LOG(ERROR) << "Seek pos: [" << pos
                   << "] invalid,  < start: [" << start_
                   << "] or > end_: [" << end_
                   << "].";
        st.Assign(Status::kInternal, "Seek pos: [" + std::to_string(pos)
                                   + "] invalid,  < start: [" + std::to_string(start_)
                                   + "] or > end_: [" + std::to_string(end_) + "].");
        return st;
      }
      st = Close();
      if (!st.Ok()) {
        return st;
      }
      try {
        reader_ = open_storage_session_->CreateOpenStorageReader(
                                           pos, end_, local_max_batch_rows_,
                                           env_max_batch_raw_size_,
                                           cache_size,
                                           env_data_columns_unordered_, name_);
        // metric_reporter_ = open_storage_session_->metric_reporter_; // Seek needn't refresh metric reporter!
      } catch (apsara::odps::sdk::OdpsException& e) {
        LOG(ERROR) << "seek fail to create reader";
        st.Assign(Status::kInternal, e.what());
        return st;
      }
      cur_ = pos;
      return st;
    }

    Status Close() {
      Status st;
      try {
        if (reader_) {
          reader_->Cancel();
        }
      } catch (apsara::odps::sdk::OdpsException& e) {
        LOG(ERROR) << "Reader Close failed !";
        st.Assign(Status::kInternal, e.what());
        return st;
      }
      return st;
    }

    size_t Tell() {
      return cur_;
    }

    const std::string& GetReaderName() {
      return name_;
    }

  private:
    static void initParamFromEnv() {
      env_data_columns_unordered_ = GetReaderDataColumnsUnordered();
      env_max_batch_rows_ = GetReaderMaxBatchRows();
      env_max_batch_raw_size_ = (GetReaderMaxBatchRawSize() > 0) ? GetReaderMaxBatchRawSize() : 0;
      LOG(INFO) << "OdpsOpenStorageArrowReaderImpl env_data_columns_unordered_: " << env_data_columns_unordered_;
      if (env_max_batch_rows_ > 0) {
        LOG(INFO) << "OdpsOpenStorageArrowReaderImpl OPEN_STORAGE_READER_MAX_BATCH_ROWS valid, "
                     << "env_max_batch_rows_: " << env_max_batch_rows_;
      }
      if (env_max_batch_raw_size_ > 0) {
        LOG(INFO) << "OdpsOpenStorageArrowReaderImpl OPEN_STORAGE_READER_MAX_BATCH_RAW_SIZE valid, "
                     << "env_max_batch_raw_size_: " << env_max_batch_raw_size_;
      }
    }
    static int env_max_batch_rows_;     // row num, 优先于`local_max_batch_rows_`(若OPEN_STORAGE_READER_MAX_BATCH_ROWS被定义)
    static int env_max_batch_raw_size_; // bytes, 0 means useless
    static bool env_data_columns_unordered_;    // false: 要求data列保持scheme同样的顺序; true: 允许两者顺序不一致
    static std::once_flag param_init_flag_;

    //apsara::odps::sdk::IArrowReaderPtr reader_;
    //std::vector<std::string> cols_;
    OdpsOpenStorageSession* open_storage_session_ = nullptr;
    bool initialized_ = false;
    int cache_size = 1;
    int local_max_batch_rows_;
    std::string name_;
    std::string path_str_;
    std::unordered_map<std::string, std::string> config_;
    std::shared_ptr<arrow::Schema> record_batch_schema_;
    std::shared_ptr<apsara::odps::sdk::storage_api::arrow_adapter::Reader> reader_;
    std::pair<std::shared_ptr<recis::monitor::Client>, std::shared_ptr<recis::monitor::PointTag>> metric_reporter_;
    size_t start_;
    size_t end_;
    size_t cur_;
};

int OdpsOpenStorageArrowReader::OdpsOpenStorageArrowReaderImpl::env_max_batch_rows_ = 1024;
int OdpsOpenStorageArrowReader::OdpsOpenStorageArrowReaderImpl::env_max_batch_raw_size_ = 0;
bool OdpsOpenStorageArrowReader::OdpsOpenStorageArrowReaderImpl::env_data_columns_unordered_ = true;
std::once_flag OdpsOpenStorageArrowReader::OdpsOpenStorageArrowReaderImpl::param_init_flag_;
std::once_flag OdpsOpenStorageArrowReader::no_columns_log_once_flag_;

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
  return OdpsOpenStorageSession::InitOdpsOpenStorageSessions(
           access_id, access_key, tunnel_endpoint, odps_endpoint, projects, tables,
           partition_specs, physical_partitions, required_data_columns,
           sep, mode, default_project, connect_timeout, rw_timeout);
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
  return OdpsOpenStorageSession::RegisterOdpsOpenStorageSession(
           access_id, access_key, tunnel_endpoint, odps_endpoint,
           project, table, partition, required_data_columns, sep,
           mode, default_project, connect_timeout, rw_timeout,
           register_light, session_id,
           expiration_time, record_count, session_def_str);
}


// path_str: 
//   e.g. odps://project_name/tables/table_name/ds=ds_name/scene=scene_name?start=0&end=100
Status OdpsOpenStorageArrowReader::CreateReader(const std::string& path_str,
                                                const int max_batch_rows,
                                                const std::string& reader_name,
                                                const char* const* cols, 
                                                size_t n,
                                                std::shared_ptr<OdpsOpenStorageArrowReader>& ret) {
  std::unordered_map<std::string, std::string> config_dict;
  ParseConfig(path_str, config_dict);
  auto reader = new OdpsOpenStorageArrowReader();
  reader->impl_.reset(new OdpsOpenStorageArrowReaderImpl(reader_name, path_str, max_batch_rows));
  ret.reset(reader);
  // 将C风格的字符串指针数组再转换回std::vector<std::string>
  std::vector<std::string> input_columns;
  input_columns.reserve(n);
  if (cols) {
    for (size_t consumed = 0; consumed < n; ++consumed) {
      try {
        const char* p = cols[consumed];
        if (!p) {
          LOG(ERROR) << "Get " << consumed << " column wrong! Get Null! Need to check the reason!";
          auto st = Status();
          st.Assign(Status::kInvalidArgument, "Get " + std::to_string(consumed) + " column wrong! Get Null! Need to check the reason!");
          return st;
        }
        input_columns.emplace_back(p); // 拷贝，拥有自己的存储
      }
      catch (const std::exception& e) {
        LOG(ERROR) << "Get " << consumed << " column wrong! The error information is: " << e.what();
        auto st = Status();
        st.Assign(Status::kInvalidArgument, "Get " + std::to_string(consumed) + " column wrong! The error information is: " + std::string(e.what()));
        return st;
      }
      catch (...) {
        LOG(ERROR) << "Get " << consumed << " column wrong! And is not std error!";
        auto st = Status();
        st.Assign(Status::kInvalidArgument, "Get " + std::to_string(consumed) + " column wrong! And is not std error!");
        return st;
      }
    }
  }
  else {
    std::call_once(no_columns_log_once_flag_, []{
      LOG(INFO) << "No columns selected by user, Will read all columns.";
    });
  }
  return reader->impl_->InitReader(config_dict, input_columns);
}

Status OdpsOpenStorageArrowReader::GetTableSize(const std::string& path_str,
                                                uint64_t& table_size) {
  std::unordered_map<std::string, std::string> config_dict;
  Status st;
  ParseConfig(path_str, config_dict);
  try {
    OdpsOpenStorageSession* session = nullptr;
    st = OdpsOpenStorageSession::GetOdpsOpenStorageSession(
                                        config_dict[kProjectNameTag],
                                        config_dict[kTableNameTag],
                                        config_dict[KPartitionSpecTag],
                                        &session);
    if (!st.Ok()) {
      LOG(ERROR) << "open storage arrow reader impl init reader, initializing session error !";
      st.Assign(Status::kInternal, "initializing session error !");
      return st;
    }
    if (!session->IsInitialized()) {
      LOG(ERROR) << "open storage arrow reader get table size, session is not initialized !";
      st.Assign(Status::kInternal, "session is not initialized !");
      return st;
    }
    table_size = static_cast<uint64_t>(session->GetTableSize());
  } catch (apsara::odps::sdk::OdpsException& e) {
    st.Assign(Status::kInternal, e.what());
  }
  return st;
}

int64_t OdpsOpenStorageArrowReader::GetSessionExpireTimestamp(const std::string& session_id) {
    return OdpsOpenStorageSession::GetSessionExpireTimestamp(session_id);
}


Status OdpsOpenStorageArrowReader::GetSchema(const std::string& config,
                                             std::unordered_map<std::string, std::string>& schema) {
  std::unordered_map<std::string, std::string> config_dict;
  ParseConfig(config, config_dict);
  Status st;
  try {
    OdpsOpenStorageSession* session = nullptr;
    st = OdpsOpenStorageSession::GetOdpsOpenStorageSession(
                                        config_dict[kProjectNameTag],
                                        config_dict[kTableNameTag],
                                        config_dict[KPartitionSpecTag],
                                        &session);
    if (!st.Ok()) {
      LOG(ERROR) << "open storage arrow reader impl init reader, initializing session error !";
      st.Assign(Status::kInternal, "initializing session error !");
      return st;
    }
    if (!session->IsInitialized()) {
      LOG(ERROR) << "open storage arrow reader get schema, session is not initialized !";
      st.Assign(Status::kInternal, "session is not initialized !");
      return st;
    }
    schema = session->GetSchema();
  } catch (apsara::odps::sdk::OdpsException& e) {
    st.Assign(Status::kInternal, e.what());
  }
  return st;
}

OdpsOpenStorageArrowReader::OdpsOpenStorageArrowReader() {}

OdpsOpenStorageArrowReader::~OdpsOpenStorageArrowReader() {
  impl_.reset();
}

Status OdpsOpenStorageArrowReader::ReadBatch(std::shared_ptr<arrow::RecordBatch>& batch) {
  return impl_->ReadBatch(batch);
}

Status OdpsOpenStorageArrowReader::Seek(size_t pos) {
  return impl_->Seek(pos);
}

Status OdpsOpenStorageArrowReader::Close() {
  return impl_->Close();
}

size_t OdpsOpenStorageArrowReader::Tell() {
  return impl_->Tell();
}

} // namespace tf
} // namespace algo
} // namespace tunnel
} // namespace odps
} // namespace apsara

//#endif // TF_ENABLE_ODPS_COLUMN
