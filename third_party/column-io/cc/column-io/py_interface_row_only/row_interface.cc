#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <chrono>
#include <iostream>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "arrow/record_batch.h"
#include "column-io/dataset_impl/odps_open_storage_row_dataset.h"
#include "column-io/py_interface/open_storage_wrapper.h"

namespace py = pybind11;

// Row-only 构建始终使用 module_local，防止与 GPU 版的 py_interface 共享全局类型注册表
#define PY_LOCAL , py::module_local()

namespace column {
namespace dataset {
void GlobalInit() {
  static bool initialized = false;
  if (initialized) return;
  initialized = true;
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  absl::InitializeLog();
}
}  // namespace dataset
}  // namespace column

PYBIND11_MODULE(py_interface, m) {
#if INTERNAL_VERSION
  //open storage
  m.def("GetOdpsOpenStorageTableSize", column::open_storage::GetOdpsOpenStorageTableSize);
  m.def("InitOdpsOpenStorageSessions", column::open_storage::InitOdpsOpenStorageSessions);
  m.def("RegisterOdpsOpenStorageSession", column::open_storage::RegisterOdpsOpenStorageSession);
  m.def("ExtractLocalReadSession", column::open_storage::ExtractLocalReadSession);
  m.def("RefreshReadSessionBatch", column::open_storage::RefreshReadSessionBatch);
  m.def("GetHaloAgentMetric", column::open_storage::GetHaloAgentMetric);
  m.def("GetOdpsOpenStorageTableFeatures", column::open_storage::GetOdpsOpenStorageTableFeatures);
  m.def("GetSessionExpireTimestamp", column::open_storage::GetSessionExpireTimestamp);
  m.def("FreeBuffer", column::open_storage::FreeBuffer);
#endif

  m.def("_global_init", column::dataset::GlobalInit);

#if INTERNAL_VERSION
  // Row-based ODPS open-storage reader. 
  py::class_<column::dataset::OdpsOpenStorageRowDataset,
             std::shared_ptr<column::dataset::OdpsOpenStorageRowDataset>>(
      m, "_OdpsOpenStorageRowDataset" PY_LOCAL)
      .def_static("make", &column::dataset::OdpsOpenStorageRowDataset::Make,
                  py::arg("paths"), py::arg("selected_columns"),
                  py::arg("batch_size"), py::arg("reader_name"))
      .def_static("get_table_size",
                  &column::dataset::OdpsOpenStorageRowDataset::GetTableSize,
                  py::arg("path"))
      .def("read_batch",
           [](column::dataset::OdpsOpenStorageRowDataset& self) -> py::list {
             // Phase 1: release GIL for pure C++ I/O (network + Arrow decode)
             std::shared_ptr<arrow::RecordBatch> batch;
             {
               py::gil_scoped_release release;
               batch = self.FetchBatch();
             }
             // Phase 2: GIL held -- convert Arrow cells to Python objects
             if (!batch) {
               return py::list();
             }
             return self.ConvertBatch(batch);
           })
      .def("seek", &column::dataset::OdpsOpenStorageRowDataset::Seek,
           py::arg("pos"))
      .def("tell", &column::dataset::OdpsOpenStorageRowDataset::Tell)
      .def("file_cursor",
           &column::dataset::OdpsOpenStorageRowDataset::FileCursor)
      .def("save_state",
           &column::dataset::OdpsOpenStorageRowDataset::SaveState)
      .def("restore_state",
           &column::dataset::OdpsOpenStorageRowDataset::RestoreState,
           py::arg("state"))
      .def_property_readonly(
          "column_names",
          &column::dataset::OdpsOpenStorageRowDataset::column_names)
      .def_property_readonly(
          "paths", &column::dataset::OdpsOpenStorageRowDataset::paths);

#endif
}
