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

#ifdef TF_ENABLE_ODPS_COLUMN

#ifndef PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_FILE_SYSTEM_H_
#define PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_FILE_SYSTEM_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "column-io/framework/status.h"

#include "column-io/odps/wrapper/file_system.h"


namespace column {
namespace odps {
namespace wrapper {
class OdpsTableFileSystem : public odps::FileSystem {
public:
  static OdpsTableFileSystem *Instance();

  Status Exists(const std::string &path, bool *ret) override;

  Status IsFile(const std::string &path, bool *ret) override;

  Status IsDirectory(const std::string &path, bool *ret) override;

  Status GetFileSize(const std::string &path, size_t *ret) override;

  Status List(const std::string &path, std::vector<std::string> *ret) override;

  Status Move(const std::string &source, const std::string &target) override;

  Status Delete(const std::string &path) override;

  Status CreateFile(const std::string &path);

  Status CreateDirectory(const std::string &path) override;

  Status WriteToFile(const std::string &path, const std::string &data);

  Status CreateFileReader(const std::string &path,
                          odps::AlgoReader **ret) override {
    std::vector<std::string> empty_vec;
    return CreateFileReader(path, ret, 128, empty_vec);
  }

  Status CreateFileReader(const std::string &path,
                          odps::AlgoReader **ret, size_t batch_size,
                          const std::vector<std::string> &select_columns);

  Status CreateFileWriter(const std::string &path,
                          odps::AlgoReader **ret) override;

  static bool CheckIOConfig();

private:
  std::unordered_map<std::string, std::string>
  GetParams(const std::string &path);

  void GetRange(const std::unordered_map<std::string, std::string> &params,
                size_t *start, size_t *end);

  void GetDelimiter(const std::unordered_map<std::string, std::string> &params,
                    std::string *delimiter);

  void
  GetSelectedColumns(const std::unordered_map<std::string, std::string> &params,
                     std::string *columns);

private:
  OdpsTableFileSystem();
  OdpsTableFileSystem(const OdpsTableFileSystem &other); // No copy
};

} // namespace wrapper
} // namespace odps
} // namespace column

#endif // PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_FILE_SYSTEM_H_

#endif // TF_ENABLE_ODPS_COLUMN
