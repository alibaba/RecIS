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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_READER_H_
#define PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_READER_H_

#include <stdint.h>

#include <string>
#include <unordered_map>

#include "column-io/framework/status.h"

#include "column-io/odps/wrapper/algo_reader.h"
#include "column-io/odps/proxy/table_reader_proxy.h"

namespace arrow {
class RecordBatch;
} // namespace arrow

namespace column {
namespace odps {
namespace wrapper {
class OdpsTableReader : public odps::AlgoReader {
public:
  OdpsTableReader(const std::string &conf, size_t start, size_t end);

  Status Init(const std::string &delimiter, const std::string &columns,
              size_t batch_size);

  Status Read(char *buff, size_t n, size_t *ret);

  Status Seek(size_t offset) override;

  Status CountRecords(size_t *psize);

  Status ReadLine(std::string *line);

  int64_t Tell() override;

  Status ReadBatch(std::shared_ptr<arrow::RecordBatch> *batch) override;

  Status GetSchema(std::unordered_map<std::string, std::string> *schema) const;

  size_t GetReadBytes();

private:
  bool ReachEnd();

private:
  proxy::TableReaderProxy reader_;
  std::string conf_;
  size_t start_;
  size_t end_;
  size_t cur_;
  size_t batch_size_;
};

} // namespace wrapper
} // namespace odps
} // namespace column

#endif // PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_READER_H_

#endif // TF_ENABLE_ODPS_COLUMN
