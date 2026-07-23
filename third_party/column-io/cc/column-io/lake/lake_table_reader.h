#ifndef PAIIO_CC_IO_ALGO_LAKE_TABLE_READER_H_
#define PAIIO_CC_IO_ALGO_LAKE_TABLE_READER_H_

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

class LakeTableReader {
public:
  LakeTableReader();
  ~LakeTableReader();
  Status Open(const std::string &table_path, size_t start, size_t end,
              size_t batch_size,
              const std::vector<std::string> &select_columns = {},
              const std::string &io_config = "");
  Status OpenSlice(const std::string &table_path, size_t slice_id,
                   size_t slice_count, size_t batch_size,
                   const std::vector<std::string> &select_columns = {},
                   const std::string &io_config = "");
  Status Seek(size_t offset);
  Status CountRecords(size_t *psize);
  int64_t Tell();
  Status ReadBatch(std::shared_ptr<arrow::RecordBatch> *batch);
  uint64_t StartPos();
  uint64_t EndPos();
  bool ReachEnd();
  void CloseReader();
  uint64_t GetReadBytes();

private:
  Status Init();
  void *_libHandle;
  void *_readerPtr;
  std::function<int(void **, const char *, size_t, size_t, size_t,
                    const char **, size_t, const char *)>
      _funcOpen;
  std::function<int(void *, size_t)> _funcSeek;
  std::function<int(void *, size_t *)> _funcCountRecords;
  std::function<int64_t(void *)> _funcTell;
  std::function<int(void *, struct ArrowArray*, struct ArrowSchema* , int64_t)> _funcReadBatch;
  std::function<uint64_t(void *)> _funcStartPos;
  std::function<uint64_t(void *)> _funcEndPos;
  std::function<bool(void *)> _funcReachEnd;
  std::function<void(void *)> _funcClose;
  std::function<int(void *)> _funcDestoryLakeTableReader;
  std::function<int(void **, const char *, size_t, size_t, size_t,
                    const char **, size_t, const char *)>
      _funcOpenSlice;
  std::function<uint64_t(void *)> _funcGetReadBytes;
};
} // namespace lake

#endif