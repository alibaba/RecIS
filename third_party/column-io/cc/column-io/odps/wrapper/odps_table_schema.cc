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

#include "column-io/framework/status.h"
#ifdef TF_ENABLE_ODPS_COLUMN

#include "column-io/odps/wrapper/odps_table_file_system.h"
#include "column-io/odps/wrapper/odps_table_reader.h"
#include "column-io/odps/wrapper/odps_table_schema.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"

namespace column {
namespace odps {
namespace wrapper {

const char *kNoIndicatorIdx = "0";
const char *kIndicatorPre = "_indicator_";

Status OdpsTableSchema::Sync(OdpsTableFileSystem *fs,
                             const std::vector<std::string> &paths,
                             bool compressed, bool allow_missing,
                             const std::vector<std::string> &selected_cols) {
  alias_map_.clear();
  indicator_map_.clear();
  indicators_.clear();
  select_columns_.clear();
  type_infos_.clear();

  for (const auto &path : paths) {
    odps::AlgoReader *raw_reader = nullptr;
    auto s = fs->CreateFileReader(path, &raw_reader);
    if (!s.ok()) {
      LOG(ERROR) << "Create OdpsTableReader failed, path: " << path;
      return s;
    }

    // Guard raw_reader
    std::unique_ptr<OdpsTableReader> reader(
        reinterpret_cast<OdpsTableReader *>(raw_reader));

    std::unordered_map<std::string, std::string> schema;
    s = reader->GetSchema(&schema);
    if (!s.ok()) {
      LOG(ERROR) << "Get OdpsTableSchema failed, path: " << path;
      return s;
    }

    // Merge Table Schema
    for (auto &type_info : schema) {
      auto it = type_infos_.find(type_info.first);
      if (it != type_infos_.end() && it->second != type_info.second) {
        LOG(ERROR) << "Multiple types map to column: " << it->first;
        return Status::FailedPrecondition();
      }

      if (it == type_infos_.end()) {
        type_infos_.insert({type_info.first, type_info.second});
      }
    }
  }

  std::unordered_set<std::string> selected_set(selected_cols.begin(),
                                               selected_cols.end());
  if (selected_set.size() != selected_cols.size()) {
    LOG(ERROR) << "Duplicated columns specified in selected columns";
    return Status::InvalidArgument();
  }

  if (!Reset(compressed, selected_set)) {
    return Status::Internal();
  }

  std::unordered_set<std::string> hit_indicators;
  // Select user specified columns
  for (const auto &column : selected_cols) {
    if (column.find(kIndicatorPre) == 0) { // Indicator column
      select_columns_.emplace_back(column);
    } else {
      auto column_name = GetColumnName(column);
      if (column_name.empty()) {
        LOG(ERROR) << "Column not exists: " + column;
        if (!allow_missing) {
          return Status::FailedPrecondition();
        }
      }

      if (indicator_map_.find(column) != indicator_map_.end()) {
        hit_indicators.insert(indicator_map_[column]);
      }
      select_columns_.emplace_back(column_name);
    }
  }

  // Select hit indicators
  for (auto &hit_indicator : hit_indicators) {
    if (selected_set.find(hit_indicator) == selected_set.end()) {
      select_columns_.emplace_back(hit_indicator);
    }
  }

  return Status();
} // namespace
  // StatusOdpsTableSchema::Sync(OdpsTableFileSystem*fs,conststd::vector<std::string>&paths,boolcompressed,boolallow_missing,conststd::vector<std::string>&selected_cols)

bool OdpsTableSchema::Reset(
    bool compressed, const std::unordered_set<std::string> &selected_cols) {
  for (auto &type_info : type_infos_) {
    auto &column_name = type_info.first;
    if (compressed) {
      if (column_name.find(kIndicatorPre) == 0) { // Indicator, skip
        continue;
      }

      size_t pos = column_name.find_last_of("_");
      std::string alias = column_name.substr(0, pos);
      if (selected_cols.count(alias) == 0) {
        continue;
      }

      alias_map_.insert({alias, column_name});
      if (pos != std::string::npos) {
        std::string indicator_name =
            kIndicatorPre + column_name.substr(pos + 1);
        if (kNoIndicatorIdx != column_name.substr(pos + 1)) {
          if (type_infos_.find(indicator_name) == type_infos_.end()) {
            LOG(ERROR) << "Indicator referenced by column: " << column_name
                       << " not exists";
            return false;
          }
          indicator_map_.insert({alias, indicator_name});
          indicators_.insert(indicator_name);
        } else {
          indicator_map_.insert({alias, ""});
        }
      }
    } else {
      if (selected_cols.count(column_name) == 0) {
        continue;
      }
      alias_map_.insert({column_name, column_name});
    }
  }

  return true;
}

} // namespace wrapper
} // namespace odps
} // namespace column

#endif // TF_ENABLE_ODPS_COLUMN
