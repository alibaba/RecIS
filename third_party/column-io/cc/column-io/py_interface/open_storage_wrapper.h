#ifndef COLUMN_OPEN_STORAGE_WRAPPER_H
#define COLUMN_OPEN_STORAGE_WRAPPER_H
#pragma once
#include <sstream>
#include <sys/time.h>
#include "arrow/type.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#include "column-io/open_storage/wrapper/odps_open_storage_arrow_reader.h"

namespace column
{
namespace open_storage
{

using apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader;

#define CHECK_LOG(condition) \
  if (!(condition)) { \
  LOG(ERROR) << "Check failed: " #condition " " \

#define EXIT_WRAPPER(stmt) \
  stmt; \
  _exit(11); \
  }

enum FsType {
  kFsTypeOdps = 0,
  kFsTypeSwift = 1,
  kFsTypeOdpsTunnel = 2,
  kFsTypeLakeStream = 3,
  kFsTypeOdpsOpenStorage = 4,
  kFsTypeLakeBatch = 5,
  kFsTypeLakeMultiCFStream = 6,
};

namespace tensorflow 
{

long long GetOdpsOpenStorageTableSize(const char* path) {
  uint64_t table_size;
  auto st = apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader::GetTableSize(path, table_size);
  if (!st.Ok()) {
    LOG(ERROR) << "GetOdpsOpenStorageTableSize of path [" << std::string(path) << "] failed";
    EXIT_WRAPPER(CHECK_LOG(false) << "get table size failed, " << st.GetMsg());
  }
  return table_size;
}

std::string GetOdpsOpenStorageTableFeatures(const char* str_path, bool is_compressed=false) {
  std::string path = std::string(str_path);
  // read schema
  std::unordered_map<std::string, std::string> schema;
  auto s = OdpsOpenStorageArrowReader::GetSchema(path, schema);
  std::vector<std::string> input_columns;
  for (auto &type_info: schema) {
    std::string column_name = type_info.first;
    if (is_compressed) {
      size_t pos = column_name.find_last_of("_");
      if (pos == std::string::npos) {
        LOG(INFO) << "compressed column name has no indicator suffix, skip: " << column_name;
        continue;
      }
      std::string alias = column_name.substr(0, pos);
      input_columns.push_back(alias);
    } else {
      input_columns.push_back(column_name);
    }
  }
  // assemble output
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  writer.StartArray();
  for (auto& column : input_columns) {
    writer.String(column.c_str(), column.length(), true);
  }
  writer.EndArray();

  LOG(INFO) << "table all features: " << sb.GetString();
  return sb.GetString();
}

} // namespace tensorflow

const long long GetOdpsOpenStorageTableSize(const char* str_path) {
  return tensorflow::GetOdpsOpenStorageTableSize(str_path);
}

const int64_t GetSessionExpireTimestamp(const char* session_id) {
    int64_t ts = OdpsOpenStorageArrowReader::GetSessionExpireTimestamp(session_id);
    return ts;
}

const long long InitOdpsOpenStorageSessions(
    const char* access_id, const char* access_key,
    const char* tunnel_endpoint, const char* odps_endpoint,
    const char* projects, const char* tables,
    const char* partition_specs, const char* physical_partitions,
    const char* required_data_columns, const char* sep,
    const char* mode, const char* default_project,
    int connect_timeout, int rw_timeout) {
  auto st = OdpsOpenStorageArrowReader::InitOdpsOpenStorageSessions(
      access_id, access_key, tunnel_endpoint, odps_endpoint,
      projects, tables, partition_specs, physical_partitions,
      required_data_columns, sep, mode, default_project, connect_timeout, rw_timeout);
  if (!st.Ok()) {
    LOG(ERROR) << "Init odps open storage sessions failed: "
      << st.GetMsg();
    return -1;
  }
  return 0;
}

const long long RegisterOdpsOpenStorageSession(
    const char* access_id, const char* access_key,
    const char* tunnel_endpoint, const char* odps_endpoint,
    const char* project, const char* table,
    const char* partition, const char* required_data_columns,
    const char* sep, const char* mode, const char* default_project,
    int connect_timeout, int rw_timeout,
    bool register_light, const char* session_id,
    long expiration_time, long record_count, const char* session_def_str) {
  auto st = OdpsOpenStorageArrowReader::RegisterOdpsOpenStorageSession(
      access_id, access_key,
      tunnel_endpoint, odps_endpoint,
      project, table, partition, required_data_columns,
      sep, mode, default_project, connect_timeout, rw_timeout,
      register_light, session_id,
      expiration_time, record_count, session_def_str);
  if (!st.Ok()) {
    LOG(ERROR) << "Register odps open storage sessions failed: "
      << st.GetMsg();
    return -1;
  }
  return 0;
}

const char* ExtractLocalReadSession(const char* access_id, const char* access_key,
                                    const char* project, const char* table,
                                    const char* partition) {
  std::string access_id_s = std::string(access_id);
  std::string access_key_s = std::string(access_key);
  std::string project_s = std::string(project);
  std::string table_s = std::string(table);
  std::string partition_s = std::string(partition);
  std::string ret = OdpsOpenStorageArrowReader::ExtractLocalReadSession(access_id_s,
                      access_key_s, project_s, table_s, partition_s);
  char* buffer = reinterpret_cast<char*>(std::malloc(ret.size() + 1));
  buffer[ret.size()] = '\0';
  std::memcpy(buffer, ret.c_str(), ret.size());
  return buffer;
}

int RefreshReadSessionBatch() {
  apsara::odps::algo::commonio::Status st = OdpsOpenStorageArrowReader::RefreshReadSessionBatch();
  if (st.Ok()) {
	return 0;
  } else {
	return 1;
  } 
}

int64_t GetHaloAgentMetric(const char* halo_endpoint, int type) {
  int64_t ret = OdpsOpenStorageArrowReader::GetHaloAgentMetric(halo_endpoint, type);
  return ret;
}

const char* GetOdpsOpenStorageTableFeatures(const char* str_path, bool is_compressed) {
  auto ret = tensorflow::GetOdpsOpenStorageTableFeatures(str_path, is_compressed);
  char* buffer = reinterpret_cast<char*>(std::malloc(ret.size() + 1));
  buffer[ret.size()] = '\0';
  std::memcpy(buffer, ret.c_str(), ret.size());
  return buffer;
}

void FreeBuffer(void* ptr) {
  free(ptr);
}
 

} //namespace open_storage
} //namespace column

#endif // COLUMN_OPEN_STORAGE_WRAPPER_H
