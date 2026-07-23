#ifndef _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_OPENSTORAGE_H_
#define _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_OPENSTORAGE_H_
#pragma once
#include <string>
#include <vector>

#include "column-io/arrow_reader/abstract_reader.h"
#include "column-io/open_storage/wrapper/odps_open_storage_arrow_reader.h"

namespace column {
namespace BatchReader{

class OdpsOpenstorageReaderImpl : public AbstractReader {
public:
  static column::Status Create(const std::string &path, int32_t batch_size,
                       const std::vector<std::string> &input_columns,
                       AbstractReaderPtr *out_reader);

  ~OdpsOpenstorageReaderImpl() override;

  column::Status ReadBatch(std::shared_ptr<arrow::RecordBatch> *data) override;
  column::Status Seek(int64_t offset) override;
  column::Status GetTableSize(size_t *size) const override;
  int64_t Tell() const override;

private:
  OdpsOpenstorageReaderImpl(
      std::shared_ptr<
          apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader>
          reader,
      std::string path);

  std::string path_;
  std::shared_ptr<apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader>
      reader_;
};

} // namespace BatchReader
} // namespace column

#endif // _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_OPENSTORAGE_H_
