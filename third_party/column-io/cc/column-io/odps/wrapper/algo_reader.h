#ifndef _COMMON_IO_COLUMN_ODPS_ALGO_READER_H
#define _COMMON_IO_COLUMN_ODPS_ALGO_READER_H
#pragma once
#include <string>
#include <unordered_map>

#include "arrow/record_batch.h"
#include "column-io/framework/status.h"

namespace column {
namespace odps {
class AlgoReader {
public:
  virtual Status Seek(size_t offset) = 0;
  virtual int64_t Tell() = 0;
  virtual Status ReadBatch(std::shared_ptr<arrow::RecordBatch> *batch) = 0;
  virtual ~AlgoReader() = default;
};
} // namespace odps
} // namespace column
#endif // _COMMON_IO_COLUMN_ODPS_ALGO_READER_H