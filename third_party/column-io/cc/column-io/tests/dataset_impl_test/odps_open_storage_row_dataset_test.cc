// Pybind11 test module: exposes test_convert_ipc_file() for Python integration
// tests that verify ArrowCellToPyObject correctness end-to-end.
//
// Built only when BUILD_TESTING=ON && INTERNAL_VERSION=1.
// Python tests import this module directly instead of going through the main
// py_interface module.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "arrow/io/file.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"
#include "arrow/record_batch.h"
#include "arrow/util/config.h"

#include "column-io/dataset/formater.h"
#include "column-io/dataset_impl/odps_open_storage_row_dataset.h"
#include "column-io/py_interface/converter.h"

namespace py = pybind11;

PYBIND11_MODULE(odps_open_storage_row_dataset_test, m) {
  m.doc() = "Test-only pybind module for OdpsOpenStorageRowDataset ArrowCellToPyObject verification";

  m.def(
      "test_convert_ipc_file",
      [](const std::string &ipc_file_path,
         const std::vector<std::string> &selected_columns) -> py::list {
        auto file_result = arrow::io::ReadableFile::Open(ipc_file_path);
        if (!file_result.ok()) {
          throw std::runtime_error(
              "test_convert_ipc_file: failed to open file [" + ipc_file_path +
              "]: " + file_result.status().ToString());
        }
        auto input_file = file_result.ValueOrDie();

        auto reader_result =
            arrow::ipc::RecordBatchFileReader::Open(input_file);
        if (!reader_result.ok()) {
          throw std::runtime_error(
              "test_convert_ipc_file: failed to open IPC reader: " +
              reader_result.status().ToString());
        }
        auto reader = reader_result.ValueOrDie();

        py::list all_rows;
        for (int i = 0; i < reader->num_record_batches(); ++i) {
          auto batch_result = reader->ReadRecordBatch(i);
          if (!batch_result.ok()) {
            throw std::runtime_error(
                "test_convert_ipc_file: failed to read batch " +
                std::to_string(i) + ": " +
                batch_result.status().ToString());
          }
          auto batch = batch_result.ValueOrDie();

          const auto &schema = batch->schema();
          std::vector<int> col_indices;
          if (selected_columns.empty()) {
            for (int c = 0; c < schema->num_fields(); ++c) {
              col_indices.push_back(c);
            }
          } else {
            for (const auto &name : selected_columns) {
              col_indices.push_back(schema->GetFieldIndex(name));
            }
          }

          const int64_t num_rows = batch->num_rows();
          for (int64_t r = 0; r < num_rows; ++r) {
            py::tuple row(col_indices.size());
            for (size_t c = 0; c < col_indices.size(); ++c) {
              int idx = col_indices[c];
              if (idx < 0) {
                row[c] = py::none();
              } else {
                row[c] =
                    column::dataset::OdpsOpenStorageRowDataset
                        ::ArrowCellToPyObject(*batch->column(idx), r);
              }
            }
            all_rows.append(std::move(row));
          }
        }
        return all_rows;
      },
      py::arg("ipc_file_path"),
      py::arg("selected_columns") = std::vector<std::string>{},
      "Read a local Arrow IPC file and convert all batches through "
      "ArrowCellToPyObject, returning list[tuple].");

  // Accepts an OdpsOpenStorageRowDataset instance created by the main
  // py_interface module. pybind11 resolves the C++ type across modules
  // automatically via shared type_info when both modules are loaded.
  m.def(
      "dump_batches_to_ipc",
      [](column::dataset::OdpsOpenStorageRowDataset &self,
         const std::string &output_path, int max_batches) -> int {
             int count = 0;
             py::gil_scoped_release release;

             auto file_result =
                 arrow::io::FileOutputStream::Open(output_path);
             if (!file_result.ok()) {
               throw std::runtime_error(
                   "dump_batches_to_ipc: failed to open [" + output_path +
                   "]: " + file_result.status().ToString());
             }
             auto out_file = file_result.ValueOrDie();
             std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;

             for (int i = 0; i < max_batches; ++i) {
               auto batch = self.FetchBatch();
               if (!batch) break;
               if (!writer) {
#if ARROW_VERSION_MAJOR >= 9
                auto writer_result =
                     arrow::ipc::MakeFileWriter(out_file.get(),
                                                batch->schema());
#else
                auto writer_result =
                     arrow::ipc::NewFileWriter(out_file.get(),
                                               batch->schema());
#endif
                 if (!writer_result.ok()) {
                   throw std::runtime_error(
                       "dump_batches_to_ipc: failed to create IPC writer: " +
                       writer_result.status().ToString());
                 }
                 writer = writer_result.ValueOrDie();
               }
               auto status = writer->WriteRecordBatch(*batch);
               if (!status.ok()) {
                 throw std::runtime_error(
                     "dump_batches_to_ipc: WriteRecordBatch failed: " +
                     status.ToString());
               }
               ++count;
             }
             if (writer) {
               auto s = writer->Close();
               if (!s.ok()) {
                 throw std::runtime_error(
                     "dump_batches_to_ipc: Close writer failed: " +
                     s.ToString());
               }
             }
             auto s = out_file->Close();
             if (!s.ok()) {
               throw std::runtime_error(
                   "dump_batches_to_ipc: Close file failed: " +
                   s.ToString());
             }
             return count;
           },
      py::arg("dataset"), py::arg("output_path"),
      py::arg("max_batches") = 1000,
      "Dump batches from an OdpsOpenStorageRowDataset to a local IPC file.");

  m.def(
      "test_convert_ipc_file_v1",
      [](const std::string &ipc_file_path,
         const std::vector<std::string> &selected_columns) -> py::list {
        auto file_result = arrow::io::ReadableFile::Open(ipc_file_path);
        if (!file_result.ok()) {
          throw std::runtime_error(
              "test_convert_ipc_file_v1: failed to open file [" +
              ipc_file_path + "]: " + file_result.status().ToString());
        }
        auto input_file = file_result.ValueOrDie();

        auto reader_result =
            arrow::ipc::RecordBatchFileReader::Open(input_file);
        if (!reader_result.ok()) {
          throw std::runtime_error(
              "test_convert_ipc_file_v1: failed to open IPC reader: " +
              reader_result.status().ToString());
        }
        auto reader = reader_result.ValueOrDie();

        auto formater = column::dataset::ColumnDataFormater
            ::GetColumnDataFormater(
                /*is_compressed=*/false, /*is_large_list=*/false,
                /*with_null=*/true);

        bool schema_inited = false;
        std::vector<std::string> col_order;

        py::list all_rows;
        for (int i = 0; i < reader->num_record_batches(); ++i) {
          auto batch_result = reader->ReadRecordBatch(i);
          if (!batch_result.ok()) {
            throw std::runtime_error(
                "test_convert_ipc_file_v1: failed to read batch " +
                std::to_string(i) + ": " +
                batch_result.status().ToString());
          }
          auto batch = batch_result.ValueOrDie();
          if (batch->num_rows() == 0) continue;

          if (!schema_inited) {
            std::unordered_set<std::string> selected_set(
                selected_columns.begin(), selected_columns.end());
            auto st = formater->InitSchema(
                batch->schema(), {}, {}, {}, {}, {}, selected_set);
            if (!st.ok()) {
              throw std::runtime_error(
                  "test_convert_ipc_file_v1: InitSchema failed: " +
                  st.DebugString());
            }
            if (selected_columns.empty()) {
              for (const auto &field : batch->schema()->fields()) {
                col_order.push_back(field->name());
              }
            } else {
              col_order = selected_columns;
            }
            schema_inited = true;
          }

          std::vector<std::shared_ptr<arrow::RecordBatch>> formated;
          auto st = formater->FormatSample(batch, &formated);
          if (!st.ok()) {
            throw std::runtime_error(
                "test_convert_ipc_file_v1: FormatSample failed: " +
                st.DebugString());
          }

          std::vector<column::Tensor> out_tensors;
          st = formater->FlatConvert(formated, &out_tensors);
          if (!st.ok()) {
            throw std::runtime_error(
                "test_convert_ipc_file_v1: FlatConvert failed: " +
                st.DebugString());
          }

          const auto &spliter_map =
              formater->schema().flatconvert_tensor_spliter;
          std::vector<column::Tensor> reordered;
          std::vector<size_t> spliter{0};
          for (const auto &name : col_order) {
            auto it = spliter_map.find(name);
            if (it == spliter_map.end()) {
              throw std::runtime_error(
                  "test_convert_ipc_file_v1: column '" + name +
                  "' not found in spliter map");
            }
            size_t b = it->second.first;
            size_t e = it->second.second;
            reordered.insert(reordered.end(),
                             out_tensors.begin() + b,
                             out_tensors.begin() + e);
            spliter.push_back(reordered.size());
          }

          py::list batch_rows =
              column::py_interface::CastTensorsToPythonTuples(
                  reordered, spliter);
          for (auto &item : batch_rows) {
            all_rows.append(item);
          }
        }
        return all_rows;
      },
      py::arg("ipc_file_path"),
      py::arg("selected_columns") = std::vector<std::string>{},
      "Read a local Arrow IPC file and convert all batches through "
      "V1 ColumnDataFormater + CastTensorsToPythonTuples pipeline, "
      "returning list[tuple]. For V1 vs V2 conversion benchmarking.");

}
