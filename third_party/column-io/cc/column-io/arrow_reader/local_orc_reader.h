#ifndef _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_LOCAL_ORC_H_
#define _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_LOCAL_ORC_H_
#pragma once

#include <string>
#include <vector>

#include "column-io/arrow_reader/abstract_reader.h"
#include "column-io/local_orc/dlwrapper_local_orc.h"

namespace column {
namespace BatchReader {

class LocalOrcReaderImpl : public AbstractReader {
public:
  static column::Status Create(const std::string &file_path,
                               const std::vector<std::string> &input_columns,
                               AbstractReaderPtr *out_reader);

  ~LocalOrcReaderImpl() override;

  column::Status ReadBatch(std::shared_ptr<arrow::RecordBatch> *data) override;
  column::Status ReadSchema(std::shared_ptr<arrow::Schema> *schema) override;
  column::Status Seek(int64_t offset) override;
  int64_t Tell() const override;

private:
  LocalOrcReaderImpl(CAPI_LOCAL_ORC_ReadCtx *reader_ctx,
                     column::local_orc::LocalOrcLib *lib);

  CAPI_LOCAL_ORC_ReadCtx *reader_ctx_;
  column::local_orc::LocalOrcLib *lib_;
};

} // namespace BatchReader
} // namespace column

#endif // _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_LOCAL_ORC_H_