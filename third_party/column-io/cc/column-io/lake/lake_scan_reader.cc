#include "lake_scan_reader.h"
#include <string.h>
namespace {
template <typename R, typename... Args>
bool BindFunc(void* handle, const char* functionName,
              std::function<R(Args...)>* func) {
  void* symbol_ptr = nullptr;      
  symbol_ptr = dlsym(handle, functionName); 
  if (!symbol_ptr) {         
    std::cout << dlerror() << std::endl;                         
    return false;                                      
  }      
  *func = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
  return true;
}

#define CHECK_FUNC_VALID(funcName) { \
  if (!funcName) {      \
    throw std::runtime_error("function is invalid!");   \
  }   \
}

}

namespace lake {

LakeScanReader::LakeScanReader() : _init(false) {}

LakeScanReader::~LakeScanReader() {
  if (_libHandle) {
    dlclose(_libHandle);
  }
  if (_init) {
    _funcDestoryLakeScanReader(_readerPtr);
  }
}

Status LakeScanReader::Init() {
  _libHandle = dlopen(getenv("LAKERUNTIMEso"), RTLD_LAZY | RTLD_DEEPBIND | RTLD_LOCAL);
  if (!_libHandle) {
    return Status::IOError("lake so not found! so path: " + std::string(getenv("LAKERUNTIMEso")));
  }

  #define BIND_LAKE_READER_FUNC(function, functionName)                      \
  {                                                   \
    if (!BindFunc(_libHandle, #functionName, function)) {  \
      return Status::InvalidArgument("function not found!");           \
    }                   \
  }

  BIND_LAKE_READER_FUNC(&_funcOpenSlice, openLakeScanReader);
  BIND_LAKE_READER_FUNC(&_funcOpenWithTimeRange, openLakeScanReaderWithTimeRange);
  BIND_LAKE_READER_FUNC(&_funcDestoryLakeScanReader, destoryLakeScanReader);
  BIND_LAKE_READER_FUNC(&_funcReadBatch, readBatchWithScanReader);
  BIND_LAKE_READER_FUNC(&_funcGetReadBytes, getReadBytesWithScanReader);
  BIND_LAKE_READER_FUNC(&_funcReachEnd, reachEndWithScanReader);
  BIND_LAKE_READER_FUNC(&_funcClose, closeWithScanReader);
  BIND_LAKE_READER_FUNC(&_funcSeek, seekWithScanReader);
  BIND_LAKE_READER_FUNC(&_funcTell, tellWithScanReader);
  BIND_LAKE_READER_FUNC(&_funcSeekTimeStamp, seekTimeStampWithScanReader);
  BIND_LAKE_READER_FUNC(&_funcSeekTimeStampRange, seekTimeStampRangeWithScanReader);
  BIND_LAKE_READER_FUNC(&_funcTellTimeStamp, tellTimeStampWithScanReader);
  return Status::OK();
}

Status LakeScanReader::Open(
        const std::string& admin_address, const std::string& path, size_t start_time, size_t end_time, size_t slice_id, size_t slice_count, size_t batch_size, 
        const std::vector<std::string>& select_columns, bool use_prefetch, size_t thread_num,
        size_t buffer_size, const std::string& query, const std::vector<std::string>& orc_names) {
  auto s = Init();
  if (!s.Ok()) {
    return s;
  }
  CHECK_FUNC_VALID(_funcOpenWithTimeRange);
  const char* columns[select_columns.size()];
  for (size_t i = 0; i < select_columns.size(); i++) {
    columns[i] = select_columns[i].c_str();
  }
  std::string query_tag = "";
  if (getenv("HIPPO_APP") == nullptr) {
    query_tag = "jobname=" + path;
  } else {
    query_tag = "jobname=" + std::string(getenv("HIPPO_APP"));
  }
  const char* names[orc_names.size()];
  for (size_t i = 0; i < orc_names.size(); i++) {
    names[i] = orc_names[i].c_str();
  }
  char* error_message;
  int code = _funcOpenWithTimeRange(
      &_readerPtr, admin_address.c_str(), path.c_str(), start_time, end_time, slice_id, slice_count, batch_size, columns, select_columns.size(),
      query_tag.c_str(), &error_message, use_prefetch, thread_num, buffer_size, query.c_str(), "", names, orc_names.size());
  _init = code ? false : true;
  return Status(Status::Code(code));
}

Status LakeScanReader::Open(
        const std::string& path, size_t slice_id, size_t slice_count, size_t batch_size, 
        const std::vector<std::string>& select_columns, bool use_prefetch, size_t thread_num,
        size_t buffer_size, const std::string& query)
{
  auto s = Init();
  if (!s.Ok()) {
    return s;
  }
  CHECK_FUNC_VALID(_funcOpenSlice);
  const char* columns[select_columns.size()];
  for (size_t i = 0; i < select_columns.size(); i++) {
    columns[i] = select_columns[i].c_str();
  }
  char* error_message;
  int code = _funcOpenSlice(
      &_readerPtr,  path.c_str(),  slice_id, slice_count, batch_size, columns, select_columns.size(),
      &error_message, use_prefetch, thread_num, buffer_size, query.c_str(), "");
  _init = code ? false : true;
  return Status(Status::Code(code));
}


Status LakeScanReader::SeekTimeStamp(int64_t begin) {
  CHECK_FUNC_VALID(_funcSeekTimeStamp);
  char* error_message;
  int code = _funcSeekTimeStamp(_readerPtr, begin, &error_message);
  return Status(Status::Code(code));
}

Status LakeScanReader::SeekTimeStampRange(int64_t begin, int64_t end) {
  CHECK_FUNC_VALID(_funcSeekTimeStampRange);
  char* error_message;
  int code = _funcSeekTimeStampRange(_readerPtr, begin, end, &error_message);
  return Status(Status::Code(code));
}

Status LakeScanReader::ReadBatch(std::shared_ptr<arrow::RecordBatch>* batch) {
  CHECK_FUNC_VALID(_funcReadBatch);
  char* error_message;
  ArrowArray Cbatch{};
  ArrowSchema Cschema{};
  int code = _funcReadBatch(_readerPtr, &Cbatch, &Cschema, &error_message, -1);
  if(code!=Status::Code::kOk)
  {
    return Status(Status::Code(code),"ReadBatch error with error message: " + std::string(error_message));
  }

  auto result = arrow::ImportRecordBatch(&Cbatch, &Cschema);
  if(!result.ok())
  {
    return Status(Status::Code::kError, "ImportRecordBatch error with arrow error: " + result.status().ToString());
  }
  *batch = result.MoveValueUnsafe();
  return Status(Status::Code(code));
}

int64_t LakeScanReader::TellTimeStamp() {
  CHECK_FUNC_VALID(_funcTellTimeStamp);
  return _funcTellTimeStamp(_readerPtr);
}

void LakeScanReader::CloseReader() {
  CHECK_FUNC_VALID(_funcClose);
  _funcClose(_readerPtr);
}

uint64_t LakeScanReader::GetReadBytes() {
  CHECK_FUNC_VALID(_funcGetReadBytes);
  return _funcGetReadBytes(_readerPtr);
}

bool LakeScanReader::ReachEnd() {
  CHECK_FUNC_VALID(_funcReachEnd);
  return _funcReachEnd(_readerPtr);
}

Status LakeScanReader::Seek(size_t offset) {
  CHECK_FUNC_VALID(_funcSeek);
  char* error_message;
  int code = _funcSeek(_readerPtr, offset, &error_message);
  return Status(Status::Code(code));
}

int64_t LakeScanReader::Tell() {
  CHECK_FUNC_VALID(_funcTell);
  return _funcTell(_readerPtr);
}

}
