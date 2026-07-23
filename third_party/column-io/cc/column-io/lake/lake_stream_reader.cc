#include "lake_stream_reader.h"
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

LakeStreamReader::LakeStreamReader() : _init(false) {}

LakeStreamReader::~LakeStreamReader() {
  if (_libHandle) {
    dlclose(_libHandle);
  }
  if (_init) {
    _funcDestoryLakeStreamReader(_readerPtr);
  }
}

Status LakeStreamReader::Init() {
  _libHandle = dlopen(getenv("LAKERUNTIMEso"), RTLD_LAZY | RTLD_LOCAL);
  if (!_libHandle) {
    return Status::IOError("lake so not found! so path: " + std::string(getenv("LAKERUNTIMEso")));
  }

  #define BIND_LAKE_READER_FUNC(function, functionName)                      \
  {                                                   \
    if (!BindFunc(_libHandle, #functionName, function)) {  \
      return Status::InvalidArgument("function not found!");           \
    }                   \
  }
  BIND_LAKE_READER_FUNC(&_funcOpen, openLakeStreamReader);
  BIND_LAKE_READER_FUNC(&_funcOpenWithQuerier, openLakeStreamReaderWithQuerier);
  BIND_LAKE_READER_FUNC(&_funcDestoryLakeStreamReader, destoryLakeStreamReader);
  BIND_LAKE_READER_FUNC(&_funcSeekTimeStamp, seekTimeStamp);
  BIND_LAKE_READER_FUNC(&_funcSeekTimeStampRange, seekTimeStampRange);
  BIND_LAKE_READER_FUNC(&_funcReadBatch, readBatchWithStream);
  BIND_LAKE_READER_FUNC(&_funcTell, tellTimeStamp);
  BIND_LAKE_READER_FUNC(&_funcClose, closeStreamReader);
  BIND_LAKE_READER_FUNC(&_funcGetReadBytes, getReadBytesWithStream);
  BIND_LAKE_READER_FUNC(&_funcGetSeekLatency, getSeekLatency);
  return Status::OK();
}

Status LakeStreamReader::Open(
    const std::string& path, size_t slice_id, size_t slice_count, size_t batch_size, 
    const std::vector<std::string>& select_columns, bool use_prefetch, size_t thread_num,
    size_t buffer_size, size_t sharding_seed, const std::string& query) {
  auto s = Init();
  if (!s.Ok()) {
    return s;
  }
  CHECK_FUNC_VALID(_funcOpen);
  const char* columns[select_columns.size()];
  for (size_t i = 0; i < select_columns.size(); i++) {
    columns[i] = select_columns[i].c_str();
  }
  int code = _funcOpen(
      &_readerPtr, path.c_str(), slice_id, slice_count, batch_size, columns, select_columns.size(),
      use_prefetch, thread_num, buffer_size, sharding_seed, query.c_str(), "");
  _init = code ? false : true;
  return Status(Status::Code(code));
}

Status LakeStreamReader::Open(
    const std::string& table_service_name, const std::string& path, size_t slice_id, size_t slice_count, size_t batch_size, 
    const std::vector<std::string>& select_columns, bool use_prefetch, size_t thread_num,
    size_t buffer_size, size_t sharding_seed, const std::string& query) {
  auto s = Init();
  if (!s.Ok()) {
    return s;
  }
  CHECK_FUNC_VALID(_funcOpenWithQuerier);
  const char* columns[select_columns.size()];
  for (size_t i = 0; i < select_columns.size(); i++) {
    columns[i] = select_columns[i].c_str();
  }
  std::string queryTag = "";
  if (getenv("HIPPO_APP") == nullptr) {
    queryTag = "jobname=" + path;
  } else {
    queryTag = "jobname=" + std::string(getenv("HIPPO_APP"));
  }
  int code = _funcOpenWithQuerier(
      &_readerPtr, table_service_name.c_str(), path.c_str(), slice_id, slice_count, batch_size, columns, select_columns.size(),
      queryTag.c_str(), use_prefetch, thread_num, buffer_size, sharding_seed, query.c_str(), ("KMONITOR_ADAPTER_USER_TAG_0:" + table_service_name).c_str());
  _init = code ? false : true;
  return Status(Status::Code(code));
}

Status LakeStreamReader::SeekTimeStamp(int64_t begin) {
  CHECK_FUNC_VALID(_funcSeekTimeStamp);
  return Status(Status::Code(_funcSeekTimeStamp(_readerPtr, begin)));
}

Status LakeStreamReader::SeekTimeStampRange(int64_t begin, int64_t end) {
  CHECK_FUNC_VALID(_funcSeekTimeStampRange);
  return Status(Status::Code(_funcSeekTimeStampRange(_readerPtr, begin, end)));
}

Status LakeStreamReader::ReadBatch(std::shared_ptr<arrow::RecordBatch>* batch) {
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

int64_t LakeStreamReader::TellTimeStamp() {
  CHECK_FUNC_VALID(_funcTell);
  return _funcTell(_readerPtr);
}

void LakeStreamReader::CloseReader() {
  CHECK_FUNC_VALID(_funcClose);
  _funcClose(_readerPtr);
}

uint64_t LakeStreamReader::GetReadBytes() {
  CHECK_FUNC_VALID(_funcGetReadBytes);
  return _funcGetReadBytes(_readerPtr);
}

int64_t LakeStreamReader::GetSeekLatency() {
  CHECK_FUNC_VALID(_funcGetSeekLatency);
  return _funcGetSeekLatency(_readerPtr);
}

}
