/*
 * Copyright 1999-2018 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "absl/strings/str_split.h"
#include "column-io/framework/status.h"
#ifdef TF_ENABLE_ODPS_COLUMN

#include "column-io/odps/wrapper/file_system.h"

#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arrow/record_batch.h"
#include "arrow/type.h"

#include "absl/log/log.h"
#include "column-io/odps/proxy/table_reader_proxy.h"
#include "column-io/odps/proxy/utils.h"
#include "column-io/odps/wrapper/odps_table_file_system.h"
#include "column-io/odps/wrapper/odps_table_reader.h"
#include "rapidjson/document.h"

namespace column {
namespace odps {
namespace wrapper {
namespace {

const size_t kBlockSize = 4096;
const char *kDelimiter = "";

} // namespace

OdpsTableFileSystem *OdpsTableFileSystem::Instance() {
  static OdpsTableFileSystem file_system;
  return &file_system;
}

OdpsTableFileSystem::OdpsTableFileSystem() {
  LOG(ERROR) << "Odps Configs not ready";
}

bool OdpsTableFileSystem::CheckIOConfig() {
  static bool ready = []() -> bool {
    if (!proxy::CheckOdpsIoConfigFile()) {
      LOG(ERROR) << "Check odps io config file failed!";
      return false;
    }
    return true;
  }();
  return ready;
}

Status OdpsTableFileSystem::Exists(const std::string &path, bool *ret) {
  odps::AlgoReader *reader = nullptr;
  auto s = CreateFileReader(path, &reader);
  if (!s.ok()) {
    return s;
  }
  delete reader;
  return s;
}

Status OdpsTableFileSystem::IsFile(const std::string &path, bool *ret) {
  auto s = IsDirectory(path, ret);
  if (s.ok()) {
    *ret = (!*ret);
  }
  return s;
}

Status OdpsTableFileSystem::IsDirectory(const std::string &path, bool *ret) {
  auto params = GetParams(path);
  *ret = (params.find("start") == params.end()) &&
         (params.find("end") == params.end());
  return Status();
}

Status OdpsTableFileSystem::GetFileSize(const std::string &path, size_t *ret) {
  auto params = GetParams(path);

  if (params.find("start") != params.end() &&
      params.find("end") != params.end()) {
    size_t start = 0, end = 0;
    GetRange(params, &start, &end);
    if (end < start) {
      std::swap(start, end);
    }
    *ret = end - start;
    return Status();
  }

  std::string spath(path);
  auto npos = spath.find("?");
  if (npos != std::string::npos) {
    spath = spath.substr(0, npos);
  }

  std::string config;
  if (!proxy::GetConfigByName(spath, &config)) {
    LOG(ERROR) << "Get odps config failed, path:" << path;
    return Status::InvalidArgument();
  }

  proxy::TableReaderProxy reader;
  if (!reader.Open(config, "", "").ok()) {
    LOG(ERROR) << "Open table reader failed, path:" << path
               << ", config:" << config;
    return Status::InvalidArgument();
  }

  *ret = reader.GetRowCount();
  return Status();
}

Status OdpsTableFileSystem::List(const std::string &path,
                                 std::vector<std::string> *ret) {
  bool is_dir = false;
  auto s = IsDirectory(path, &is_dir);
  if (!s.ok()) {
    return s;
  }

  if (!is_dir) {
    LOG(ERROR) << path << " is not directory!";
    return Status::InvalidArgument();
  }

  size_t row_count = 0;
  s = GetFileSize(path, &row_count);
  if (!s.ok()) {
    return s;
  }

  size_t block_count = row_count / kBlockSize;
  if (row_count % kBlockSize != 0) {
    ++block_count;
  }

  bool with_params = (path.find("?") != std::string::npos);
  for (size_t i = 0; i < block_count; ++i) {
    size_t start = i * kBlockSize;
    size_t end = start + kBlockSize;
    if (end > row_count) {
      end = row_count;
    }
    std::string pn(path);
    if (with_params) {
      pn += "&";
    } else {
      pn += "?";
    }

    pn += "start=";
    pn += std::to_string(start);
    pn += "&end=";
    pn += std::to_string(end);
    ret->push_back(pn);
  }

  return s;
}

Status OdpsTableFileSystem::Move(const std::string &source,
                                 const std::string &target) {
  return Status::Unimplemented();
}

Status OdpsTableFileSystem::Delete(const std::string &path) {
  return Status::Unimplemented();
}

Status OdpsTableFileSystem::CreateFile(const std::string &path) {
  return Status::Unimplemented();
}

Status OdpsTableFileSystem::CreateDirectory(const std::string &path) {
  return Status::Unimplemented();
}

Status OdpsTableFileSystem::WriteToFile(const std::string &path,
                                        const std::string &data) {
  return Status::Unimplemented();
}

Status OdpsTableFileSystem::CreateFileReader(
    const std::string &path, column::odps::AlgoReader **ret,
    size_t batch_size, const std::vector<std::string> &select_columns) {
  if (path.empty()) {
    LOG(ERROR) << "`path` should not be empty!";
    return Status::InvalidArgument();
  }

  auto params = GetParams(path);

  std::string tpath(path);
  auto pos = path.find("?");
  if (pos != std::string::npos) {
    tpath = path.substr(0, pos);
  }

  size_t start = 0, end = 0;
  GetRange(params, &start, &end);
  if (end == 0) {
    auto s = GetFileSize(tpath, &end);
    if (!s.ok()) {
      LOG(ERROR) << "Create file reader failed, path:" << path;
      return s;
    }
  }

  if (start >= end) {
    LOG(ERROR) << "Error"
               << "Invalid path: " << path;
    return Status::InvalidArgument();
  }

  std::string columns;
  GetSelectedColumns(params, &columns);
  if (columns.empty() && !select_columns.empty()) {
    columns = proxy::JoinString(select_columns, ",");
  }

  LOG(INFO) << "Select_columns: " << columns;

  std::string delimiter(kDelimiter);
  GetDelimiter(params, &delimiter);

  std::string config;
  if (!proxy::GetConfigByName(tpath, &config)) {
    LOG(ERROR) << "Get Odps Config failed, path: " << path;
    return Status::InvalidArgument();
  }

  auto reader = new OdpsTableReader(config, start, end);
  if (!reader->Init(delimiter, columns, batch_size).ok()) {
    LOG(ERROR) << "Init OdpsTableReader failed!";
    delete reader;
    return Status::Internal();
  }

  *ret = reader;
  return Status();
}

Status OdpsTableFileSystem::CreateFileWriter(const std::string &path,
                                             odps::AlgoReader **ret) {
  return Status::Unimplemented();
}

std::unordered_map<std::string, std::string>
OdpsTableFileSystem::GetParams(const std::string &path) {
  std::unordered_map<std::string, std::string> params;
  std::string spath(path);
  auto pos = spath.find("?");
  if (pos != std::string::npos) {
    auto suffix = spath.substr(pos + 1);
    std::vector<std::string> vecs = absl::StrSplit(suffix, "&");
    for (auto &vec : vecs) {
      // LOG(INFO) << "substr is :" << vec;
      std::vector<std::string> kv = absl::StrSplit(vec, "=");
      if (kv.size() == 2) {
        params.insert({kv[0], kv[1]});
      }
    }
  }

  return params;
}

void OdpsTableFileSystem::GetRange(
    const std::unordered_map<std::string, std::string> &params, size_t *start,
    size_t *end) {
  auto it = params.find("start");
  if (it != params.end()) {
    *start = std::stoull(it->second);
  }

  it = params.find("end");
  if (it != params.end()) {
    *end = std::stoull(it->second);
  }
}

void OdpsTableFileSystem::GetDelimiter(
    const std::unordered_map<std::string, std::string> &params,
    std::string *delimiter) {
  auto it = params.find("delimiter");
  if (it != params.end()) {
    *delimiter = it->second;
  }
}

void OdpsTableFileSystem::GetSelectedColumns(
    const std::unordered_map<std::string, std::string> &params,
    std::string *columns) {
  auto it = params.find("select");
  if (it != params.end()) {
    *columns = it->second;
  }
}

} // namespace wrapper
} // namespace odps
} // namespace column

#endif // TF_ENABLE_ODPS_COLUMN
