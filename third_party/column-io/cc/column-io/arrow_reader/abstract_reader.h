#ifndef _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_ABSTRACT_H_
#define _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_ABSTRACT_H_
#pragma once
#include <string>
#include <vector>
#include <memory>
#include "arrow/record_batch.h"

#include "column-io/framework/status.h"

namespace column {
namespace BatchReader{

// AbstractReader: Any reader that provides arrow::RecordBatch read function
class AbstractReader {
protected:
  AbstractReader() = default;
public:
  virtual ~AbstractReader() = default;
  //AbstractReader(AbstractReader&&) = default;
  AbstractReader(const AbstractReader&) = delete;
  AbstractReader& operator=(const AbstractReader&) = delete;
  //AbstractReader& operator=(AbstractReader&&) = default;
  virtual column::Status ReadBatch(std::shared_ptr<arrow::RecordBatch> *data) = 0;
  virtual column::Status ReadSchema(std::shared_ptr<arrow::Schema> *schema) {throw std::runtime_error("AbstractReader.ReadSchema UnImplemented");};
  virtual column::Status Seek(int64_t offset) {throw std::runtime_error("AbstractReader.Seek UnImplemented");};
  virtual column::Status GetTableSize(size_t *size) const {throw std::runtime_error("AbstractReader.GetTableSize UnImplemented");};
  virtual int64_t Tell() const {throw std::runtime_error("AbstractReader.Tell UnImplemented");};
  virtual int64_t TellTimeStamp() const {throw std::runtime_error("AbstractReader.TellTimeStamp UnImplemented");};
};

typedef std::unique_ptr<AbstractReader> AbstractReaderPtr;

} // namespace BatchReader
} // namespace column

#endif // _COLUMN_IO_CC_COLUMN_IO_ARROW_READER_ABSTRACT_H_
