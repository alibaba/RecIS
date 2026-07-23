#ifndef _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_ODPS_H_
#define _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_ODPS_H_
#pragma once
#include <string>
#include <vector>

#include "column-io/arrow_reader/abstract_reader.h"
#if (_GLIBCXX_USE_CXX11_ABI != 1)
#include "column-io/odps/wrapper/odps_table_file_system.h"
#include "column-io/odps/wrapper/odps_table_reader.h"
#endif

namespace column {
namespace BatchReader {

class OdpsAlgoReaderImpl : public AbstractReader {
public:
  static column::Status Create(const std::string &path, int32_t batch_size, const std::vector<std::string> &input_columns, AbstractReaderPtr *out_reader);

  ~OdpsAlgoReaderImpl() override;

  column::Status ReadBatch(std::shared_ptr<arrow::RecordBatch> *data) override;
  column::Status Seek(int64_t offset) override;
  column::Status GetTableSize(size_t *size) const override;
  int64_t Tell() const override;

private:
  //TODO: 把声明中的OdpsAlgo符号改到实现文件中
#if (_GLIBCXX_USE_CXX11_ABI != 1)
  explicit OdpsAlgoReaderImpl(std::unique_ptr<column::odps::wrapper::OdpsTableReader> reader);
  std::unique_ptr<column::odps::wrapper::OdpsTableReader> reader_;
#else
  explicit OdpsAlgoReaderImpl();
#endif
};

} // namespace BatchReader
} // namespace column

#endif // _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_ODPS_H_
