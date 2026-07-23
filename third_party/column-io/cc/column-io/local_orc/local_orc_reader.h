#ifndef _CC_COLUMNIO_LOCAL_ORC_LOCAL_ORC_READER_H_
#define _CC_COLUMNIO_LOCAL_ORC_LOCAL_ORC_READER_H_
#pragma once

#include <string>
#include <vector>
#include "arrow/status.h"
#include "arrow/record_batch.h"
#include "arrow/adapters/orc/adapter.h"

namespace column {
namespace local_orc {

using std::string;
using std::vector;
using std::unique_ptr;
using std::shared_ptr;
using arrow::RecordBatch;
using arrow::adapters::orc::ORCFileReader;

class LocalOrcReader {
public:
  static arrow::Status MakeReader(const string file_path,
                                const vector<string> &input_columns,
                                std::unique_ptr<LocalOrcReader> *out_reader_ptr);

  arrow::Status ReadBatch(shared_ptr<RecordBatch> *data);
  arrow::Status Seek(int64_t index);
  shared_ptr<arrow::Schema> ReadSchema() const;
  int64_t Tell() const;

private:
  LocalOrcReader(const string file_path, const std::vector<string> &selected_columns);

  arrow::Status MakeReaderInternal();

  // arrow::Status InitSchema();

  string file_path_;
  vector<string> selected_columns_;
  shared_ptr<ORCFileReader> orc_reader_;
  shared_ptr<arrow::RecordBatchReader> reader_;
  shared_ptr<arrow::Schema> schema_;
  int64_t index_;
};

}
}

#endif
