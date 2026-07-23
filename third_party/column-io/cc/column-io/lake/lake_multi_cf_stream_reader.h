#ifndef PAIIO_CC_IO_ALGO_LAKE_MULTI_CF_STREAM_READER_H_
#define PAIIO_CC_IO_ALGO_LAKE_MULTI_CF_STREAM_READER_H_

#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <memory>
#include <functional>
#include "status.h"
#include "arrow/c/bridge.h"

namespace arrow {
class RecordBatch;
}

namespace lake {

class LakeMultiCFStreamReader {
  public:
    LakeMultiCFStreamReader();
    ~LakeMultiCFStreamReader();
    Status Open(
        const std::string& path, size_t slice_id, size_t slice_count, size_t batch_size, 
        const std::vector<std::string>& select_columns = {}, bool use_prefetch = false, size_t thread_num = 3,
        size_t buffer_size = 3, size_t sharding_seed = 65535, const std::string& query = "");
    Status Open(
        const std::string& table_service_name, const std::string& path, size_t slice_id, size_t slice_count, size_t batch_size, 
        const std::vector<std::string>& select_columns = {}, bool use_prefetch = false, size_t thread_num = 3,
        size_t buffer_size = 3, size_t sharding_seed = 65535, const std::string& query = "");
    Status SeekTimeStamp(int64_t begin);
    Status SeekTimeStampRange(int64_t begin, int64_t end);
    Status ReadBatch(std::shared_ptr<arrow::RecordBatch>* batch);
    int64_t TellTimeStamp();
    void CloseReader();
    uint64_t GetReadBytes();
    int64_t GetSeekLatency();

  private:
    Status Init();
    bool _init;
    void* _libHandle;
    void* _readerPtr;
    std::function<int(void**, const char*, size_t, size_t, size_t, const char**, 
                      size_t, bool, size_t, size_t, size_t, const char*, const char*)> _funcOpen;
    std::function<int(void**, const char*, const char*, size_t, size_t, size_t, const char**, 
                      size_t, const char*, bool, size_t, size_t, size_t, const char*, const char*)> _funcOpenWithQuerier;
    std::function<int(void*)> _funcDestoryLakeMultiCFStreamReader;
    std::function<int(void*, size_t)> _funcSeekTimeStamp;
    std::function<int(void*, size_t, size_t)> _funcSeekTimeStampRange;
    std::function<int(void*, struct ArrowArray*, struct ArrowSchema*, int64_t)> _funcReadBatch;
    std::function<int64_t(void*)> _funcTell;
    std::function<void(void*)> _funcClose;
    std::function<uint64_t(void*)> _funcGetReadBytes;
    std::function<int64_t(void*)> _funcGetSeekLatency;
};
}

#endif