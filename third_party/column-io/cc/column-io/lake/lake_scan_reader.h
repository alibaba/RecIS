#ifndef PAIIO_CC_IO_ALGO_LAKE_SCAN_READER_H_
#define PAIIO_CC_IO_ALGO_LAKE_SCAN_READER_H_

#include "status.h"
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include "arrow/c/bridge.h"

namespace arrow {
class RecordBatch;
}

namespace lake {

class LakeScanReader {
public:
  LakeScanReader();
  ~LakeScanReader();
  Status Open(
    const std::string& admin_address, const std::string& path, size_t start_time, size_t end_time, size_t slice_id, size_t slice_count, size_t batch_size, 
    const std::vector<std::string>& select_columns = {}, bool use_prefetch = false, size_t thread_num = 3,
    size_t buffer_size = 3, const std::string& query = "", const std::vector<std::string>& orc_names = {});
  Status Open(
      const std::string& path, size_t slice_id, size_t slice_count, size_t batch_size, 
      const std::vector<std::string>& select_columns = {}, bool use_prefetch = false, size_t thread_num = 3,
      size_t buffer_size = 3, const std::string& query = "");
  Status SeekTimeStamp(int64_t begin);
  Status SeekTimeStampRange(int64_t begin, int64_t end);
  Status ReadBatch(std::shared_ptr<arrow::RecordBatch>* batch);
  int64_t TellTimeStamp();
  void CloseReader();
  uint64_t GetReadBytes();
  bool ReachEnd();
  Status Seek(size_t offset);
  int64_t Tell();

private:
  Status Init();
  bool _init;
  void *_libHandle;
  void *_readerPtr;
  std::function<int(void**, const char*,size_t, size_t, size_t,  const char**, 
                    size_t, char**, bool, size_t, size_t, const char*, const char*)> _funcOpenSlice;

  std::function<int(void**, const char*, const char*, size_t, size_t, size_t, size_t, size_t, const char**, 
                    size_t, const char*, char**, bool, size_t, size_t, const char*, const char*, const char**, size_t)> _funcOpenWithTimeRange;
  std::function<int(void*)> _funcDestoryLakeScanReader;
  std::function<int(void*,  struct ArrowArray*, struct ArrowSchema*, char**, int64_t)> _funcReadBatch;
  std::function<uint64_t(void*)> _funcGetReadBytes;
  std::function<bool(void*)> _funcReachEnd;
  std::function<void(void*)> _funcClose;
  std::function<int(void*, size_t, char**)> _funcSeek;
  std::function<int64_t(void*)> _funcTell;
  std::function<int(void*, size_t, char**)> _funcSeekTimeStamp;
  std::function<int(void*, size_t, size_t, char**)> _funcSeekTimeStampRange;
  std::function<int64_t(void*)> _funcTellTimeStamp;
};
} // namespace lake

#endif