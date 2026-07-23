#include "absl/log/initialize.h"
#include "column-io/dataset/dataset.h"
#include "column-io/dataset/list_dataset.h"
#include "column-io/dataset/list_combo_dataset.h"
#include "column-io/dataset/packer.h"
#include "column-io/dataset/parallel_dataset.h"
#include "column-io/dataset/prefetch_dataset.h"
#include "column-io/dataset/repeat_dataset.h"
#include "column-io/dataset/map_dataset.h"
#include "column-io/dataset_impl/local_rb_stream_dataset.h"
#include "column-io/dataset_impl/local_orc_dataset.h"
#if INTERNAL_VERSION
#include "column-io/dataset/map_dataset_k2_rank.h"
#include "column-io/dataset/map_dataset_k2_rank_canxi.h"
#include "column-io/dataset/map_dataset_sample_filter.h"
#include "column-io/dataset_impl/lake_stream_column_dataset.h"
#include "column-io/dataset_impl/lake_multi_cf_stream_column_dataset.h"
#include "column-io/dataset_impl/lake_batch_column_dataset.h"
#if (_GLIBCXX_USE_CXX11_ABI != 1)
#include "column-io/dataset_impl/odps_table_column_dataset.h"
#endif
#include "column-io/dataset_impl/odps_open_storage_dataset.h"
#include "column-io/dataset_impl/odps_combo_dataset.h"
#include "column-io/lake/lake_fslib_helper.h"
#include "column-io/odps/proxy/lib_odps.h"
//#include "column-io/open_storage/wrapper/dl_wrapper_open_storage.h"
//#include "column-io/open_storage/wrapper/odps_open_storage_arrow_reader.h"
#include "column-io/dataset_impl/odps_open_storage_row_dataset.h"
#include "column-io/py_interface/open_storage_wrapper.h"
#endif
#include "column-io/framework/types.h"
#include "column-io/py_interface/converter.h"
#include "column-io/py_interface/dataset.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <memory>
#include <pybind11/detail/common.h>

#include "arrow/record_batch.h"
namespace py = pybind11;

// CPU 构建使用 module_local 防止与 GPU 版的 py_interface 共享全局类型注册表
#ifdef NEED_CPU_ONLY
#define PY_LOCAL , py::module_local()
#else
#define PY_LOCAL
#endif

PYBIND11_MODULE(py_interface, m) {
#if INTERNAL_VERSION
  //open storage
  // 以下 5 个函数内部涉及网络请求/future.get()/sleep重试, 阻塞秒~分钟级,
  // 且 C++ 实现不访问 py:: 对象, 有内部 mutex 保证线程安全, 故释放 GIL.
  m.def("GetOdpsOpenStorageTableSize", column::open_storage::GetOdpsOpenStorageTableSize,
        py::call_guard<py::gil_scoped_release>());
  m.def("RegisterOdpsOpenStorageSession", column::open_storage::RegisterOdpsOpenStorageSession,
        py::call_guard<py::gil_scoped_release>());
  m.def("ExtractLocalReadSession", column::open_storage::ExtractLocalReadSession,
        py::call_guard<py::gil_scoped_release>());
  m.def("RefreshReadSessionBatch", column::open_storage::RefreshReadSessionBatch,
        py::call_guard<py::gil_scoped_release>());
  m.def("GetOdpsOpenStorageTableFeatures", column::open_storage::GetOdpsOpenStorageTableFeatures,
        py::call_guard<py::gil_scoped_release>());
  // 以下 4 个函数阻塞 ≤毫秒级, 释放 GIL 收益极小, 不改.
  m.def("InitOdpsOpenStorageSessions", column::open_storage::InitOdpsOpenStorageSessions);
  m.def("GetHaloAgentMetric", column::open_storage::GetHaloAgentMetric);
  m.def("GetSessionExpireTimestamp", column::dataset::OdpsOpenStorageDataset::GetSessionExpireTimestamp);
  m.def("FreeBuffer", column::open_storage::FreeBuffer);
#endif

  m.def("_global_init", column::dataset::GlobalInit);
  // iterator
  py::class_<column::dataset::IteratorBase,
             std::shared_ptr<column::dataset::IteratorBase>>(m, "_Iterator" PY_LOCAL);
  m.def("MakeIterator", column::dataset::MakeIterator);
  m.def("GetNextFromIterator", &column::dataset::GetNextFromIterator,
        pybind11::return_value_policy::move);
  m.def("SerializeIteraterStateToString",
        &column::dataset::SerializeIteraterStateToString);
  m.def("DerializeIteraterStateFromString",
        &column::dataset::DeserializeIteratorStateFromString);

  // dataset
  py::class_<column::dataset::DatasetBase,
             std::shared_ptr<column::dataset::DatasetBase>>(m, "_Dataset" PY_LOCAL);
  py::class_<column::dataset::DatasetBuilder,
             std::shared_ptr<column::dataset::DatasetBuilder>>(
      m, "_DatasetBuilder" PY_LOCAL);

  // internal dataset.
  py::class_<column::dataset::ListStringDataset>(m, "_ListStringDataset" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::ListStringDataset::MakeDataset);

  py::class_<column::dataset::Packer>(m, "_PackerDataset" PY_LOCAL)
      .def_static("make_dataset", column::dataset::Packer::MakeDataset)
      .def_static("make_reorder_dataset",
                  column::dataset::Packer::MakeReorderDataset);

  py::class_<column::dataset::ParallelDataset>(m, "_ParallelDataset" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::ParallelDataset::MakeDataset);

  py::class_<column::dataset::RepeatDataset>(m, "_RepeatDataset" PY_LOCAL)
      .def_static("make_dataset", column::dataset::RepeatDataset::MakeDataset);

  py::class_<column::dataset::PrefetchDataset>(m, "_PrefetchDataset" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::PrefetchDataset::MakeDataset);

  py::class_<column::dataset::LocalRBStreamDataset>(m, "_LocalRBStreamDataset" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::LocalRBStreamDataset::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::LocalRBStreamDataset::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::LocalRBStreamDataset::ParseSchema);

  py::class_<column::dataset::MapDataSet>(m, "_MapDataSet" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::MapDataSet::MakeDataSet);

#if INTERNAL_VERSION
  py::class_<column::dataset::MapDataSetK2Rank>(m, "_MapDataSetRank" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::MapDataSetK2Rank::MakeDataSet);

  py::class_<column::dataset::MapDataSetK2RankCanXi>(m, "_MapDataSetRankCanXi" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::MapDataSetK2RankCanXi::MakeDataSet);

  py::class_<column::dataset::MapDataSetSampleFilter>(m,
                                                     "_MapDataSetSampleFilter"
                                                     PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::MapDataSetSampleFilter::MakeDataSet);

#endif

#if INTERNAL_VERSION
  // source dataset.
  #if (_GLIBCXX_USE_CXX11_ABI == 0)
  py::class_<column::dataset::OdpsTableColumnDataset>(m,
                                                      "_OdpsTableColumnDataset"
                                                      PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::OdpsTableColumnDataset::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::OdpsTableColumnDataset::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::OdpsTableColumnDataset::ParseSchema)
      .def_static("get_table_size", column::dataset::GetTableSize)
      .def_static("load_odps_plugin", column::odps::proxy::LibOdps::LoadWrap)
      .def_static("get_table_features", column::dataset::OdpsTableColumnDataset::GetOdpsTableFeatures);
  #endif

  // odps-openstorage dataset
  py::class_<column::dataset::OdpsOpenStorageDataset>(m,
                                                      "_OdpsOpenStorageDataset"
                                                      PY_LOCAL)

      .def_static("make_dataset",
                  column::dataset::OdpsOpenStorageDataset::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::OdpsOpenStorageDataset::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::OdpsOpenStorageDataset::ParseSchema)
      .def_static("get_table_size", column::dataset::GetTableSize)
      .def_static("load_open_storage_plugin", apsara::odps::tunnel::algo::tf::OdpsOpenStorageLib::LoadWrap)
      .def_static("get_table_features", column::dataset::OdpsOpenStorageDataset::GetOdpsTableFeatures);

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

  // combo dataset
  py::class_<column::dataset::ListStringComboDataset>(m, "_ListStringComboDataset" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::ListStringComboDataset::MakeDataset);

  // combo dataset
  py::class_<column::dataset::OdpsComboDataset>(m, "_OdpsComboDataset" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::OdpsComboDataset::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::OdpsComboDataset::MakeBuilder);

  py::class_<column::dataset::LakeStreamColumnDatase>(
      m, "_LakeStreamColumnDataset" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::LakeStreamColumnDatase::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::LakeStreamColumnDatase::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::LakeStreamColumnDatase::ParseSchema)
      .def_static("close_pangu", lake::closePangu);

  py::class_<column::dataset::LakeMultiCFStreamColumnDatase>(
      m, "_LakeMultiCFStreamColumnDataset" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::LakeMultiCFStreamColumnDatase::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::LakeMultiCFStreamColumnDatase::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::LakeMultiCFStreamColumnDatase::ParseSchema)
      .def_static("close_pangu", lake::closePangu);

  py::class_<column::dataset::LakeBatchColumnDatase>(
      m, "_LakeBatchColumnDataset" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::LakeBatchColumnDatase::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::LakeBatchColumnDatase::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::LakeBatchColumnDatase::ParseSchema)
      .def_static("parse_schema_by_rows",
                  column::dataset::LakeBatchColumnDatase::ParseSchemaByRows)
      .def_static("close_pangu", lake::closePangu);
#endif

  py::class_<column::dataset::LocalOrcDataset>(m, "_LocalOrcDataset" PY_LOCAL)
      .def_static("make_dataset",
                  column::dataset::LocalOrcDataset::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::LocalOrcDataset::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::LocalOrcDataset::ParseSchema);
}
