// Copyright (c) Alibaba.
// Row-based ODPS Open Storage reader.
//
// This file mirrors the role of ``odps_open_storage_dataset.h`` but produces
// row-major Python objects instead of ``column::Tensor`` batches. Internally
// it owns the same kind of state machine (``file_cur_`` / ``begin_cur_`` /
// per-path ``OdpsOpenStorageArrowReader``) so that multi-path inputs, Seek
// and Save/Restore behave identically to the column-oriented dataset.
//
// Unlike the pure-STL RowCell approach, this class directly constructs
// ``pybind11::object`` values from Arrow arrays for maximum performance
// (one fewer copy for string/binary, one fewer traversal pass). The GIL
// is managed by the caller (interface.cc): release before FetchBatch, then
// reacquire before ConvertBatch.
#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_ODPS_OPEN_STORAGE_ROW_DATASET_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_ODPS_OPEN_STORAGE_ROW_DATASET_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "arrow/record_batch.h"
#include "column-io/open_storage/wrapper/odps_open_storage_arrow_reader.h"

namespace column {
namespace dataset {

// The read pipeline is split into two phases to allow the caller to manage
// the GIL optimally:
//   1. FetchBatch() — pure C++ I/O, no Python objects. GIL can be released.
//   2. ConvertBatch() — iterates the fetched RecordBatch and constructs
//      py::object cells. GIL must be held.
class OdpsOpenStorageRowDataset {
 public:
  static std::shared_ptr<OdpsOpenStorageRowDataset> Make(
      const std::vector<std::string>& paths,
      const std::vector<std::string>& selected_columns,
      int64_t batch_size,
      const std::string& reader_name);

  static int64_t GetTableSize(const std::string& path);

  OdpsOpenStorageRowDataset() = default;
  ~OdpsOpenStorageRowDataset() = default;

  // Phase 1: Fetch the next Arrow RecordBatch from the underlying reader.
  // pure C++ I/O (no Python objects created)
  std::shared_ptr<arrow::RecordBatch> FetchBatch();

  // Phase 2: Convert a fetched RecordBatch into a Python list[list[object]].
  pybind11::list ConvertBatch(
      const std::shared_ptr<arrow::RecordBatch>& batch);

  // Convenience: FetchBatch + ConvertBatch in one call.
  pybind11::list ReadBatch();

  void Seek(size_t pos);

  size_t Tell();

  int64_t FileCursor() const { return file_cur_; }

  std::string SaveState();
  void RestoreState(const std::string& state);

  const std::vector<std::string>& column_names() const {
    return selected_columns_;
  }

  const std::vector<std::string>& paths() const { return paths_; }

  // Arrow cell -> py::object conversion (recursive, handles nested types).
  static pybind11::object ArrowCellToPyObject(
      const arrow::Array& array, int64_t index);

 private:
  void OpenCurrentReader();

  std::mutex mu_;
  std::vector<std::string> paths_;
  std::vector<std::string> selected_columns_;
  int64_t batch_size_{1024};
  std::string reader_name_;

  std::shared_ptr<apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader>
      reader_;
  int64_t file_cur_{0};
  int64_t begin_cur_{-1};
  bool reach_end_{false};
};

}  // namespace dataset
}  // namespace column

#endif  // COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_ODPS_OPEN_STORAGE_ROW_DATASET_H_
