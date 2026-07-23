#include "lake_table_reader.h"
namespace {
template <typename R, typename... Args>
bool BindFunc(void *handle, const char *functionName,
              std::function<R(Args...)> *func) {
  void *symbol_ptr = nullptr;
  symbol_ptr = dlsym(handle, functionName);
  if (!symbol_ptr) {
    std::cout << dlerror() << std::endl;
    return false;
  }
  *func = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
  return true;
}

#define CHECK_FUNC_VALID(funcName)                                             \
  {                                                                            \
    if (!funcName) {                                                           \
      throw std::runtime_error("function is invalid!");                        \
    }                                                                          \
  }

} // namespace

namespace lake {

LakeTableReader::LakeTableReader() {}

LakeTableReader::~LakeTableReader() {
  if (_libHandle) {
    dlclose(_libHandle);
  }
  if (_readerPtr) {
    _funcDestoryLakeTableReader(_readerPtr);
  }
}

Status LakeTableReader::Init() {
  _libHandle = dlopen(getenv("LAKERUNTIMEso"), RTLD_LAZY | RTLD_LOCAL);
  if (!_libHandle) {
    return Status::IOError("lake so not found! so path: " +
                           std::string(getenv("LAKERUNTIMEso")));
  }

#define BIND_LAKE_READER_FUNC(function, functionName)                          \
  {                                                                            \
    if (!BindFunc(_libHandle, #functionName, function)) {                      \
      return Status::InvalidArgument("function not found!");                   \
    }                                                                          \
  }
  BIND_LAKE_READER_FUNC(&_funcOpen, openLakeTableReader);
  BIND_LAKE_READER_FUNC(&_funcSeek, seek);
  BIND_LAKE_READER_FUNC(&_funcCountRecords, countRecords);
  BIND_LAKE_READER_FUNC(&_funcTell, tell);
  BIND_LAKE_READER_FUNC(&_funcReadBatch, readBatch);
  BIND_LAKE_READER_FUNC(&_funcStartPos, startPos);
  BIND_LAKE_READER_FUNC(&_funcEndPos, endPos);
  BIND_LAKE_READER_FUNC(&_funcReachEnd, reachEnd);
  BIND_LAKE_READER_FUNC(&_funcClose, closeReader);
  BIND_LAKE_READER_FUNC(&_funcDestoryLakeTableReader, destoryLakeTableReader);
  BIND_LAKE_READER_FUNC(&_funcOpenSlice, openSliceLakeTableReader);
  BIND_LAKE_READER_FUNC(&_funcGetReadBytes, getReadBytes);
  return Status::OK();
}

Status LakeTableReader::Open(const std::string &table_path, size_t start,
                             size_t end, size_t batch_size,
                             const std::vector<std::string> &select_columns,
                             const std::string &io_config) {
  auto s = Init();
  if (!s.Ok()) {
    return s;
  }
  CHECK_FUNC_VALID(_funcOpen);
  const char *columns[select_columns.size()];
  for (size_t i = 0; i < select_columns.size(); i++) {
    columns[i] = select_columns[i].c_str();
  }
  return Status(Status::Code(
      _funcOpen(&_readerPtr, table_path.c_str(), start, end, batch_size,
                columns, select_columns.size(), io_config.c_str())));
}

Status
LakeTableReader::OpenSlice(const std::string &table_path, size_t slice_id,
                           size_t slice_count, size_t batch_size,
                           const std::vector<std::string> &select_columns,
                           const std::string &io_config) {
  auto s = Init();
  if (!s.Ok()) {
    return s;
  }
  CHECK_FUNC_VALID(_funcOpenSlice);
  const char *columns[select_columns.size()];
  for (size_t i = 0; i < select_columns.size(); i++) {
    columns[i] = select_columns[i].c_str();
  }
  return Status(Status::Code(_funcOpenSlice(
      &_readerPtr, table_path.c_str(), slice_id, slice_count, batch_size,
      columns, select_columns.size(), io_config.c_str())));
}

Status LakeTableReader::Seek(size_t offset) {
  CHECK_FUNC_VALID(_funcSeek);
  return Status(Status::Code(_funcSeek(_readerPtr, offset)));
}

Status LakeTableReader::CountRecords(size_t *psize) {
  CHECK_FUNC_VALID(_funcCountRecords);
  return Status(Status::Code(_funcCountRecords(_readerPtr, psize)));
}

int64_t LakeTableReader::Tell() {
  CHECK_FUNC_VALID(_funcTell);
  return _funcTell(_readerPtr);
}

Status LakeTableReader::ReadBatch(std::shared_ptr<arrow::RecordBatch>* batch) {
  CHECK_FUNC_VALID(_funcReadBatch);
  ArrowArray Cbatch{};
  ArrowSchema Cschema{};
  int code = _funcReadBatch(_readerPtr, &Cbatch, &Cschema, -1);
  if(code!=Status::Code::kOk)
  {
    return Status(Status::Code(code),"ReadBatch error.");
  }

  auto result = arrow::ImportRecordBatch(&Cbatch, &Cschema);
  if(!result.ok())
  {
    return Status(Status::Code::kError, "ImportRecordBatch error with arrow error: " + result.status().ToString());
  }
  *batch = result.MoveValueUnsafe();
  return Status(Status::Code(code));
}

uint64_t LakeTableReader::StartPos() {
  CHECK_FUNC_VALID(_funcStartPos);
  return _funcStartPos(_readerPtr);
}

uint64_t LakeTableReader::EndPos() {
  CHECK_FUNC_VALID(_funcEndPos);
  return _funcEndPos(_readerPtr);
}
bool LakeTableReader::ReachEnd() {
  CHECK_FUNC_VALID(_funcReachEnd);
  return _funcReachEnd(_readerPtr);
}

void LakeTableReader::CloseReader() {
  CHECK_FUNC_VALID(_funcClose);
  _funcClose(_readerPtr);
}

uint64_t LakeTableReader::GetReadBytes() {
  CHECK_FUNC_VALID(_funcGetReadBytes);
  return _funcGetReadBytes(_readerPtr);
}
} // namespace lake
