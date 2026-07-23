#include "column-io/py_interface/dataset.h"
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <algorithm>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <vector>
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"

#include "column-io/dataset/iterator.h"
#include "column-io/dataset/list_dataset.h"
#include "column-io/dataset/vec_tensor_converter.h"
#include "column-io/dataset_impl/lake_stream_column_dataset.h"
#include "column-io/dataset_impl/lake_multi_cf_stream_column_dataset.h"
#include "column-io/dataset_impl/lake_batch_column_dataset.h"
#include "column-io/dataset_impl/local_rb_stream_dataset.h"
#include "column-io/dataset_impl/local_orc_dataset.h"
#if (_GLIBCXX_USE_CXX11_ABI != 1 )
#include "column-io/dataset_impl/odps_table_column_dataset.h"
#endif
#include "column-io/dataset_impl/odps_open_storage_dataset.h"
#include "column-io/dataset_impl/odps_combo_dataset.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/tensor_shape.h"
#include "column-io/framework/types.h"
#include "column-io/py_interface/converter.h"

namespace column {
namespace dataset {
namespace detail {
struct StatusExcept : std::exception {
  StatusExcept(const std::string_view &err_info) : err_info_(err_info) {}
  const char *what() const noexcept override { return err_info_.c_str(); }
  static StatusExcept FromStatus(const Status &st) {
    std::ostringstream excp_msg;
    excp_msg << "error_code: " << st.code() << " error msg:" << st.error_message();
    return StatusExcept(excp_msg.str());
  }
  static StatusExcept FromStatus(const absl::Status &st) {
    std::ostringstream excp_msg;
    excp_msg << "error_code: " << st.code() << " error msg:" << st.message();
    return StatusExcept(excp_msg.str());
  }

private:
  std::string err_info_;
};

} // namespace detail

absl::LogSeverity AbslGetLogLevelFromEnv() {
  const static char log_level = getenv("NEBULA_IO_LOG_LEVEL") ? *getenv("NEBULA_IO_LOG_LEVEL") : 1 ; // 1 == INFO in NEBULA_IO_LOG_LEVEL
  int absl_loglevel = atoi(&log_level) -1 ; // absl::LogSeverity missing Debug levle. awful abseil
  absl_loglevel = std::max(0, absl_loglevel);
  absl_loglevel = std::min(3, absl_loglevel); //why 3, not LogSeverity::LevelNumber? awful abseil
  absl::LogSeverity log_severity = static_cast<absl::LogSeverity>(absl_loglevel);
  return log_severity;
}

void GlobalInit() {
  // 防止与 column_io (GPU) 同进程加载时重复初始化 absl::InitializeLog 导致 crash
  static bool initialized = false;
  if (initialized) return;
  initialized = true;
  // do initialize for absl log.
  // TODO: merge three log frameworks
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  absl::LogSeverityAtLeast level = static_cast<absl::LogSeverityAtLeast>(AbslGetLogLevelFromEnv());
  absl::SetMinLogLevel(level);
  // NOTE absl NOT suppor format. unless hack into log_format.cc:FormatBoundedFields. awful abseil
  // absl::SetLogFormat(); 
  absl::InitializeLog();
}

pybind11::object GetNextFromIterator(std::shared_ptr<IteratorBase> iterator, bool row_mode) {
  bool end_of_sequence;
  std::vector<Tensor> outputs;
  // 若希望读取行存格式batch, 则需从迭代器中同时得到`outputs`及分割符(否则后续`数据`和`ragged_rank`将混淆在outputs中). 原列存读取模式下本参数不起作用
  std::vector<size_t> outputs_row_spliter; 
  Status st;
  {
    pybind11::gil_scoped_release lock;
    if (!iterator) {
        throw std::runtime_error("Iterator is null or expired");
    }
    st = iterator->GetNext(&outputs, &end_of_sequence, &outputs_row_spliter);
  }
  if (end_of_sequence) {
    throw pybind11::stop_iteration();
  }
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(
        Status::Internal(st.error_message()));
  }

  if( row_mode ){
    // iterator->selected_columns_;
    return py_interface::CastTensorsToPythonTuples(outputs, outputs_row_spliter);
  }

  pybind11::list ret;
  for (auto &&tensor : outputs) {
    ret.append(py_interface::CastTensorToDLPack(std::move(tensor)));
  }
  return ret;
}

pybind11::bytes
SerializeIteraterStateToString(std::shared_ptr<IteratorBase> iterator) {
  std::string msg;
  auto st = column::dataset::SerializeIteraterToString(iterator, &msg);
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(st);
  }
  pybind11::gil_scoped_acquire l;
  return pybind11::bytes(msg);
};

void DeserializeIteratorStateFromString(std::shared_ptr<IteratorBase> iterator,
                                        const std::string &msg) {
  auto st = column::dataset::DeserializeIteratorFromString(iterator, msg);
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(st);
  }
}

std::shared_ptr<IteratorBase>
MakeIterator(std::shared_ptr<DatasetBase> dataset) {
  std::shared_ptr<IteratorBase> iter;
  auto st = dataset->MakeIterator(dataset->name(), &iter);
  if (!st.ok()) {
    auto e = detail::StatusExcept::FromStatus(st);
    throw e;
  }
  return iter;
}

bool IsTurnOnOdpsOpenStorage() {
  const char* turn_on = std::getenv("ODPS_OPEN_STORAGE_MDL");
  if (turn_on != nullptr && std::string(turn_on) == "1") {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<DatasetBase> LocalRBStreamDataset::MakeDatasetWrapper(
    const std::vector<std::string> &paths, bool is_compressed,
    int64_t batch_size, const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto st =
      MakeDataset(paths, is_compressed, batch_size, selected_columns,
                  input_columns, hash_features, hash_types, hash_buckets, dense_columns, dense_defaults);
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(st.status());
  }
  return st.value();
}

#if INTERNAL_VERSION
size_t GetTableSize(const std::string &path) {
  size_t ret;
  Status st;
  if (IsTurnOnOdpsOpenStorage()) {
    st = OdpsOpenStorageDataset::GetTableSize(path, &ret);
  } else {
#if (_GLIBCXX_USE_CXX11_ABI != 1 )
    st = OdpsTableColumnDataset::GetTableSize(path, &ret);
#else
    throw std::runtime_error("ColumnIO Not support GetTableSize in this algosdk version");
#endif
  }
  if (!st.ok()) {
    auto e = detail::StatusExcept::FromStatus(st);
    throw e;
  }
  return ret;
}

#if (_GLIBCXX_USE_CXX11_ABI != 1)
std::shared_ptr<DatasetBase> OdpsTableColumnDataset::MakeDatasetWrapper(
    const std::vector<std::string> &paths,
    bool is_compressed,
    int64_t batch_size,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto st =
      MakeDataset(paths, is_compressed, batch_size, selected_columns,
                  input_columns, hash_features, hash_types, hash_buckets, dense_columns, dense_defaults);
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(st.status());
  }
  return st.value();
}
#endif

std::shared_ptr<DatasetBase> OdpsComboDataset::MakeDatasetWrapper(
    const std::vector<std::vector<std::string>> &paths,
    bool is_compressed,
    int64_t batch_size,
    const std::vector<std::vector<std::string>> &selected_columns,
    const std::vector<std::vector<std::string>> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults,
    const bool &check_data,
    const std::string &primary_key,
    bool turn_on_odps_open_storage) {
  auto st =
      MakeDataset(paths, is_compressed, batch_size, selected_columns,
                  input_columns, hash_features, hash_types, hash_buckets,
                  dense_columns, dense_defaults, check_data, primary_key, turn_on_odps_open_storage);
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(st.status());
  }
  return st.value();
}

std::shared_ptr<DatasetBase> OdpsOpenStorageDataset::MakeDatasetWrapper(
    const std::vector<std::string> &paths,
    bool is_compressed,
    int64_t batch_size,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults,
    bool use_xrec) {
  auto st =
      MakeDataset(paths, is_compressed, batch_size, selected_columns,
                  input_columns, hash_features, hash_types, hash_buckets,
                  dense_columns, dense_defaults, use_xrec);
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(st.status());
  }
  return st.value();
}

std::shared_ptr<DatasetBase> LakeStreamColumnDatase::MakeDatasetWrapper(
    const std::string &path,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_features,
    const std::vector<std::vector<float>> &dense_defaults,
    bool is_compressed,
    int64_t batch_size,
    bool use_prefetch,
    int64_t prefetch_thread_num,
    int64_t prefetch_buffer_size) {
  auto st =
      MakeDataset(path, selected_columns, input_columns,
                  hash_features, hash_types, hash_buckets,
                  dense_features, dense_defaults,
                  is_compressed, batch_size,
                  use_prefetch, prefetch_thread_num, prefetch_buffer_size);
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(st.status());
  }
  return st.value();
}

std::shared_ptr<DatasetBase> LakeMultiCFStreamColumnDatase::MakeDatasetWrapper(
    const std::string &path,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_features,
    const std::vector<std::vector<float>> &dense_defaults,
    bool is_compressed,
    int64_t batch_size,
    bool use_prefetch,
    int64_t prefetch_thread_num,
    int64_t prefetch_buffer_size) {
  auto st =
      MakeDataset(path, selected_columns, input_columns,
                  hash_features, hash_types, hash_buckets,
                  dense_features, dense_defaults,
                  is_compressed, batch_size,
                  use_prefetch, prefetch_thread_num, prefetch_buffer_size);
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(st.status());
  }
  return st.value();
}

std::shared_ptr<DatasetBase> LakeBatchColumnDatase::MakeDatasetWrapper(
    const std::string &path,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_features,
    const std::vector<std::vector<float>> &dense_defaults,
    bool is_compressed,
    int64_t batch_size,
    bool use_prefetch,
    int64_t prefetch_thread_num,
    int64_t prefetch_buffer_size) {
  auto st =
      MakeDataset(path, selected_columns, input_columns,
                  hash_features, hash_types, hash_buckets,
                  dense_features, detail::VecsToTensor(dense_defaults),
                  is_compressed, batch_size,
                  use_prefetch, prefetch_thread_num, prefetch_buffer_size);
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(st.status());
  }
  return st.value();
}
#endif

std::shared_ptr<DatasetBase> LocalOrcDataset::MakeDatasetWrapper(
    const std::vector<std::string> &paths, bool is_compressed,
    int64_t batch_size, const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto st =
      MakeDataset(paths, is_compressed, batch_size, selected_columns,
                  input_columns, hash_features, hash_types, hash_buckets, dense_columns, dense_defaults);
  if (!st.ok()) {
    throw detail::StatusExcept::FromStatus(st.status());
  }
  return st.value();
}

} // namespace dataset
} // namespace column
