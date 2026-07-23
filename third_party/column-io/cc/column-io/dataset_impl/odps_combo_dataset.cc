#include "column-io/dataset_impl/odps_combo_dataset.h"

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "arrow/record_batch.h"
#include "arrow/type.h"
#include "arrow/array.h"

#include "column-io/dataset/formater.h"
#include "column-io/dataset/vec_tensor_converter.h"
#include "column-io/dataset_impl/path_parser.h"
#include "column-io/dataset_impl/schema_parser.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include "column-io/arrow_reader/abstract_reader.h"
#include "column-io/arrow_reader/odps_algo_reader.h"
#include "column-io/arrow_reader/odps_openstorage_reader.h"
// #include "column-io/odps/wrapper/odps_table_file_system.h"
// #include "column-io/odps/wrapper/odps_table_reader.h"
// #include "column-io/open_storage/wrapper/odps_open_storage_arrow_reader.h"


namespace column {
namespace dataset {
namespace {

bool GetEnvBool(const std::string &var_name, bool def_value = false) {
  const char *env_value = std::getenv(var_name.c_str());
  if (env_value == nullptr) {
    return def_value;
  }
  std::string value = env_value;
  for (auto &c : value) {
    c = std::tolower(c);
  }
  return value == "true" || value == "1" || value == "yes" || value == "on";
}

int64_t GetEnvInt64(const std::string &var_name, int64_t default_value = 0) {
  const char *env_value = std::getenv(var_name.c_str());

  if (env_value == nullptr) {
    return default_value;
  }

  try {
    return std::stoll(env_value);
  } catch (const std::exception &e) {
    throw std::invalid_argument("Failed to parse environment variable '" +
                                var_name + "': " + std::string(env_value) +
                                " (" + e.what() + ")");
  }
}

namespace {
  const std::string kDatasetName = "OdpsTableColumnCombo";
  const std::string kIndicator = "_indicator";
}

class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name,
          const std::vector<std::vector<std::string>> &paths,
          bool is_compressed, int64_t batch_size,
          const std::vector<std::vector<std::string>> &selected_columns,
          const std::vector<std::vector<std::string>> &input_columns,
          const std::vector<std::string> &hash_features,
          const std::vector<std::string> &hash_types,
          const std::vector<int64_t> &hash_buckets,
          const std::vector<std::string> &dense_features,
          const std::vector<Tensor> &dense_defaults, bool check_data,
          std::string primary_key, bool turn_on_odps_open_storage)
      : DatasetBase(name), paths_(std::move(paths)), batch_size_(batch_size),
        selected_columns_(std::move(selected_columns)),
        input_columns_(input_columns), hash_features_(hash_features),
        hash_types_(hash_types), hash_buckets_(hash_buckets),
        dense_features_(dense_features), dense_defaults_(dense_defaults),
        is_compressed_(is_compressed), ds_name_(name), check_data_(check_data),
        primary_key_(primary_key),
        turn_on_odps_open_storage_(turn_on_odps_open_storage) {}

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::shared_ptr<IteratorBase>(
        new Iterator(std::dynamic_pointer_cast<Dataset>(shared_from_this()),
                     (prefix + "::OdpsComboDataset"), ds_name_, batch_size_,
                     is_compressed_, turn_on_odps_open_storage_));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  private:
    void ParseCurrentPointTag() {
        std::string project = "";
        std::string table = "";
        if (file_cur_ < dataset()->paths_.size()) {
            const std::string& filepath = dataset()->paths_[file_cur_].front();
            // E.g. "odps://rec_test/tables/mainsearch/ds=20251231?start=1&end=10"
            ParseOdpsUrl(filepath, project, table);
            // 兼容tunnel传参格式
            ParseTunnelTableFormat(filepath, project, table);
        }
        point_tag_ = std::make_shared<recis::monitor::PointTag>(std::map<std::string, std::string>{
            {"ds_project", project.empty() ? "null" : project},
            {"ds_table", table.empty() ? "null" : table},
            {"ds_name", "odps_combo"}, // TODO: add ds_name_ for iterator from constructor
            {"multiprocess_seq", std::to_string(this->get_child_order())}
        });
    }
  public:
    explicit Iterator(const std::shared_ptr<Dataset> datataset,
                      const std::string &prefix, const std::string &ds_name,
                      int32_t batch_size, bool is_compressed,
                      bool turn_on_odps_open_storage)
        : DatasetIterator<Dataset>({datataset, prefix}), reach_end_(false),
          batch_size_(batch_size), is_compressed_(is_compressed),
          turn_on_odps_open_storage_(turn_on_odps_open_storage),
          record_interval_(kRecordInterval), force_seek_(false),
          force_deep_copy_(false) {
      table_num_ = dataset()->paths_[0].size();
      // read_costs_accum_.resize(table_num_);
      // read_bytes_accum_.resize(table_num_);
      datas_cache_.resize(table_num_);
      readers_.resize(table_num_);
      selected_table_columns_.resize(table_num_);
      column_formaters_.resize(table_num_);
      force_seek_ = GetEnvBool("COMBO_FORCE_SEEK", false);
      force_deep_copy_ = GetEnvBool("COMBO_FORCE_DEEP_COPY", false);
      record_interval_ = GetEnvInt64("COMBO_RECORD_INTERVAL", kRecordInterval);
      LOG(INFO) << "Iterator init. "
                << "table_num_ is " << table_num_ 
                << "; is_compressed_ is " << is_compressed_ 
                << "; turn_on_odps_open_storage_ is " << turn_on_odps_open_storage_
                << "; force_seek_ is " << force_seek_ 
                << "; force_deep_copy_ is " << force_deep_copy_
                << "; record_interval_ is " << record_interval_;
      if (is_compressed_) {
        LOG(WARNING) << "OdpsComboDataset is experimentally supporting is_compressed now, be warn of that.";
      }
    }

    Status InitSchema(std::shared_ptr<arrow::RecordBatch> &data,
                      size_t table_idx) {
      if (column_formaters_[table_idx])
        return column::Status::OK();
      // init formater
      column_formaters_[table_idx] = ColumnDataFormater::GetColumnDataFormater(
          dataset()->is_compressed_, false);
      std::unordered_set<std::string> selected_columns;
      const auto &cols = dataset()->selected_columns_[table_idx];
      selected_columns.reserve(cols.size());
      selected_columns.insert(cols.begin(), cols.end());
      auto st = column_formaters_[table_idx]->InitSchema(
          data->schema(), dataset()->hash_features_, dataset()->hash_types_,
          dataset()->hash_buckets_, dataset()->dense_features_,
          dataset()->dense_defaults_, selected_columns);
      if (!st.ok()) {
        column_formaters_[table_idx].reset();
        return st;
      }
      return column::Status::OK();
    }

    Status GetNextInternal(
        std::vector<Tensor> *out_tensors, bool *end_of_sequence,
        std::vector<size_t> *outputs_row_spliter = nullptr) override {
      std::lock_guard<std::mutex> l(mu_);
      // auto now_micros = []() -> uint64_t {
      //   auto now = std::chrono::steady_clock::now();
      //   auto duration = now.time_since_epoch();
      //   return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
      // };
      int loop_cnt = 0;
      do {
        if (reach_end_) {
          *end_of_sequence = true;
          return column::Status::OK();
        }
        if (!(status_.ok())) {
          return status_;
        }
        if (point_tag_ == nullptr) {
            ParseCurrentPointTag();
        }
        column::Status s = column::Status::OK();
        if (readers_[0]) {
          *end_of_sequence = false;
          std::vector<std::shared_ptr<arrow::RecordBatch>> datas;
          datas.resize(table_num_);

          size_t min_batch_rows = 0;
          int32_t min_reader_idx = -1;
          bool all_batch_size_same = true;

          for (size_t i = 0; i < table_num_; ++i) {
            column::Status column_st = column::Status::OK();
            std::shared_ptr<arrow::RecordBatch>& data = datas[i];
            // time_start = now_micros();

            int64_t delay_ms = 1;
            uint64_t rows_read = 0;

            if (datas_cache_[i]) {
              datas[i] = datas_cache_[i];
              // rows_read is not really happen
            } else {
              auto &reader = readers_.at(i);
              if (!reader) {
                column_st = column::Status::InvalidArgument("reader ", i, " terminated unexpectedly.");
                break;
              }
              uint64_t rows_before_read = reader->Tell();
              int64_t time_before_read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
              column_st = reader->ReadBatch(&data);

              int64_t time_after_read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
              delay_ms = std::max(0L, time_after_read_ms - time_before_read_ms);
              rows_read = reader->Tell() - rows_before_read ;
            }
            metric_cli_->report("read_row",  rows_read, point_tag_, recis::monitor::PointType::kCounter);
            metric_cli_->report("read_batch", 1, point_tag_, recis::monitor::PointType::kCounter);
            metric_cli_->report("read_latency_ms", delay_ms, point_tag_, recis::monitor::PointType::kGauge);

            if (column_st.code() == ErrorCode::OUT_OF_RANGE) {
                LOG(INFO) << "read_batch kOutOfRange! current table_idx: " << file_cur_
                          << ", file_path: " << dataset()->paths_[file_cur_][i];
                s = column::Status::OutOfRange(column_st.code() + "; table idx: " + std::to_string(file_cur_));
                continue;
            }
            if (!column_st.ok()) {
                LOG(ERROR) << "read_batch Internal column_st.GetCode is " << column_st.code()
                        << " column_st.GetMsg() is " << column_st.error_message() << ", table_idx: " << file_cur_
                        << ", table_path: " << dataset()->paths_[file_cur_][i];
                s = column::Status::Internal(column_st.error_message());
                break;
            }
            if (data->num_rows() == 0) {
              s = column::Status::DataLoss("Get empty batch for reader ", i, ", please check data.");
              break;
            }
            // column_st is NO-OutRange, and OK, and NOT-empty-Batch, can process data
            if (min_reader_idx == -1) {
              min_batch_rows = data->num_rows();
              min_reader_idx = i;
            } else if (min_batch_rows != data->num_rows()) {
              all_batch_size_same = false;
              if (min_batch_rows > data->num_rows()) {
                min_batch_rows = data->num_rows();
                min_reader_idx = i;
              }
            }
          } // end for, read each table

          if (s.code() == ErrorCode::OUT_OF_RANGE && min_reader_idx != -1) {
            s = column::Status::Internal("Some reader exits early. Please check data. "
                                "min_reader_idx: ", min_reader_idx);
          }

          if (s.ok() && min_batch_rows == 0) {
            for (int i = 0; i < table_num_; ++i) {
              if (datas_cache_[i]) {
                s = column::Status::Internal("Error in reader: ", i, ", the buffer remaining: ",
                  datas_cache_[i]->num_rows(), " has no data to use. reader msg: ", s.error_message());
                break;
              }
            }
          }

          if (force_seek_) {
            // Need to seek back reader
            // force seek back
            if (s.ok() && min_reader_idx != -1) {
              int64_t seek_offset = 0;
              column::Status common_sb_st;
              seek_offset = readers_.at(min_reader_idx)->Tell();
              for (size_t i = 0; i < table_num_; ++i) {
                common_sb_st = readers_.at(i)->Seek(seek_offset);
                if (!common_sb_st.ok()) {
                  s = Status::Internal("Seek back failed for reader ", i, ": ",
                                       common_sb_st.error_message());
                  break;
                }
                datas[i] = datas[i]->Slice(0, min_batch_rows);
              }
            }
          }

          // the reader will never read_batch until its buffer completely
          // consumed each reader has at most one buffer
          if (s.ok()) {
            for (size_t i = 0; i < table_num_; ++i) {
              if (datas_cache_[i]) {
                if (force_seek_) {
                  s = column::Status::Internal("Never use cache in seek mode for reader ", i);
                  break;
                }
                // reader cache num_rows() > other reader read_batch num_rows();
                // slice the cache [min_batch_rows, end_of_this_batch)
                if (min_batch_rows < datas_cache_[i]->num_rows()) {
                  datas_cache_[i] = datas_cache_[i]->Slice(min_batch_rows);
                }
                // reader cache num_rows() == other reader read_batch num_rows();
                // erase the cache
                else if (min_batch_rows == datas_cache_[i]->num_rows()) {
                  datas_cache_[i].reset();
                }
                // reader cache num_rows() < min_batch_rows; never happened
                else {
                  s = column::Status::Unknown(
                      "min_batch_rows: ", min_batch_rows, "; but buffer ", i,
                      " has less: ", datas_cache_[i]->num_rows());
                  break;
                }
              } else {
                // reader read_batch num_rows() > min_batch_rows; cache the extra
                // data
                if (min_batch_rows < datas[i]->num_rows()) {
                  if (force_seek_) {
                    s = column::Status::Internal(
                        "Never make cache in seek mode for reader ", i,
                        "; min_batch_rows :", min_batch_rows, "; data rows ", datas[i]->num_rows());
                    break;
                  }
                  datas_cache_[i] = datas[i]->Slice(min_batch_rows);
                  // reader read_batch num_rows() < min_batch_rows; never
                  // happened
                } else if (min_batch_rows > datas[i]->num_rows()) {
                  s = column::Status::Unknown("min_batch_rows: ", min_batch_rows,
                                      "; but reader ", i,
                                      " get less: ", datas[i]->num_rows());
                  break;
                }
              }
            }
          }

          // Check row by primary key
          if (s.ok() && dataset()->check_data_) {
            auto main_col = datas[0]->GetColumnByName(dataset()->primary_key_);
            if (!main_col) {
              s = column::Status::InvalidArgument(
                  "Primary key column `", dataset()->primary_key_,
                  "` not existed in the first input table.");
            } else {
              for (size_t i = 1; i < datas.size(); ++i) {
                auto col = datas[i]->GetColumnByName(dataset()->primary_key_);
                if (!main_col->RangeEquals(col, 0, min_batch_rows, 0)) {
                  LOG(ERROR) << "we failed check pri key "
                             << dataset()->primary_key_ << " for table " << i;
                  LOG(ERROR) << "main_col is "
                             << main_col->Slice(0, min_batch_rows)->ToString()
                             << " length "
                             << main_col->Slice(0, min_batch_rows)->length();
                  LOG(ERROR)
                      << "col is " << col->Slice(0, min_batch_rows)->ToString()
                      << " length " << col->Slice(0, min_batch_rows)->length();
                  s = column::Status::DataLoss(
                      "Primary key check failed between reader ", i,
                      " and reader 0.");
                  break;
                }
              }
            }
          }

          /*Format data. Conditions:
            1. Some tables are compressed while the others are not. Raise
            Internal error
            2. Format sample from specific table failed (e.g., type mismatched).
            Raise InvalidArgument error
            3. Check unfold rows failed(e.g., unfold rows differ). Raise
            DataLoss error
            4. Init schema failed (e.g., dim invalid). Raise InvalidArgument
            error
            5. Flatten sample failed (e.g., column not found). Raise
            InvalidArgument error
            6. Successfully format data
          */
          
          if (s.ok()) {
            // vector -> flatten / compressed
            // map key -> feature name
            // map value -> value, row split
            int64_t time_before_format_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>> merged_conv_output;
            // s = FormatTableSample(merged_conv_output, datas);
            // vector -> table base / table inc
            //    vector -> unfold rows of each group (either compressed or not,
            //    sliced or not)
            std::vector<std::vector<int64>> offset_ends(table_num_);
            for (size_t i = 0; i < table_num_; ++i) {
              auto &data = datas[i];
              if (!column_formaters_[i]) {
                s = InitSchema(data, i);
              }
              if (!s.ok()) {
                LOG(ERROR) << "InitSchema error for reader " << i;
                break;
              }
              std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
              s = column_formaters_[i]->FormatSample(
                  data, &formated_data, min_batch_rows, &offset_ends[i]);
              // check for unfold rows of each group between tables. mainly for
              // compressed table
              if (i > 0) {
                for (size_t k = 0; k < std::min(offset_ends[i - 1].size(), offset_ends[i].size()); ++k) {
                  if (offset_ends[i - 1][k] != offset_ends[i][k]) {
                    std::ostringstream oss;
                    oss << "group " << k << " of columns in table " << dataset()->paths_[file_cur_][i - 1] << " is " << offset_ends[i - 1][k];
                    oss << "; but in table " << dataset()->paths_[file_cur_][i] << " is " << offset_ends[i][k] << " which mismatch";
                    s = column::Status::DataLoss(oss.str());
                    break;
                  }
                }
                if (!s.ok()) {
                  LOG(ERROR) << "Error when comparing offset between tables ";
                  break;
                }
              }
              // copy on write
              bool copy_this_batch = force_deep_copy_;
              for(int col_i=0; col_i < formated_data[0]->num_columns(); col_i++){
                // _indicator 列经过累加重组, 已丢失offset信息、无法用于判断应否deep_copy
                const auto& col_name = formated_data[0]->column_name(col_i);
                if (col_name.rfind(kIndicator, 0) == 0) {continue;}
                copy_this_batch = copy_this_batch || (formated_data[0]->column(col_i)->offset() != 0);
                break;
              }
              std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>> conv_output;
              s = column_formaters_[i]->FlatConvert(
                  formated_data, &conv_output, offset_ends[i], copy_this_batch);
              if (!s.ok()) {
                LOG(ERROR) << "Error when flattening data batch from reader " << i;
                break;
              }
              if (merged_conv_output.size() == 0) {
                merged_conv_output.resize(conv_output.size());
              } else if (merged_conv_output.size() != conv_output.size()) {
                s = column::Status::InvalidArgument("Sample type is different among tables!");
                break;
              }
              for (size_t j = 0; j < merged_conv_output.size(); ++j) {
                merged_conv_output[j].insert(conv_output[j].begin(), conv_output[j].end());
              }
            } // end for format each table sample

            if (s.ok()) { // Format sample completed, return the sample and status OK
              for (auto &map : merged_conv_output) {
                for (auto &item : map) {
                  for (auto &vec : item.second) {
                    out_tensors->insert(out_tensors->end(), vec.begin(), vec.end());
                  }
                }
              }
              int64_t time_after_format_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
              double format_cost_ms = time_after_format_ms - time_before_format_ms;
              metric_cli_->report("read_postproc_latency_ms", format_cost_ms, point_tag_, recis::monitor::PointType::kGauge);

              status_ = s;
              return status_;
            }
          }
          std::fill(readers_.begin(), readers_.end(), nullptr);
          if (s.code() == ErrorCode::OUT_OF_RANGE) {
            ++file_cur_;
            continue; // Continue to while
          } else {
            status_ = s;
            LOG(ERROR) << "OdpsComboDataset iterator error: " << status_.DebugString();
            return status_;
          }
        } else {
          // open readers
          if (file_cur_ >= dataset()->paths_.size()) {
            reach_end_ = true;
            continue;
          }
          size_t target_table_size_current_group = 0;
          for (size_t i = 0; i < table_num_; ++i) {
            const std::string& launch_file = dataset()->paths_[file_cur_][i];
            LOG(INFO) << "odps-combo launch open file: " << launch_file << "; batch_size: " << batch_size_ << "; openstorage: " << turn_on_odps_open_storage_;
            if (turn_on_odps_open_storage_) {
              RETURN_IF_ERROR(BatchReader::OdpsOpenstorageReaderImpl::Create(
                  launch_file, batch_size_, dataset()->input_columns_[i], &readers_[i]));
            } else {
              RETURN_IF_ERROR(BatchReader::OdpsAlgoReaderImpl::Create(
                  launch_file, batch_size_, dataset()->input_columns_[i], &readers_[i]));
            }
            if (begin_cur_ < 0) {
                continue;
            }
            // begin_cur_ is inited from RestoreInternal
            size_t table_size;
            RETURN_IF_ERROR(readers_.at(i)->GetTableSize(&table_size));
            if (i == 0) {
              target_table_size_current_group = table_size;
            } else if (target_table_size_current_group != table_size) {
              s = column::Status::InvalidArgument(
                "Table ", i, " has different size from ", "Table 0: ", table_size, 
                " vs ", target_table_size_current_group);
              std::fill(readers_.begin(), readers_.end(), nullptr);
              break;
            }
            LOG(INFO) << "seek file_path " << launch_file << " to offset: " << begin_cur_;
            s = readers_.at(i)->Seek(begin_cur_);
            if (!s.ok()) {
            s = column::Status::InvalidArgument("Fail to seek file_path: ", launch_file, ", to offset: ", begin_cur_);
              std::fill(readers_.begin(), readers_.end(), nullptr);
              break;
            }
            if (i == table_num_ - 1) {
              begin_cur_ = -1;
            }
          } // end for (size_t i = 0; i < table_num_; ++i)
          if (!s.ok()) {
            LOG(ERROR) << "some error happened when open reader";
            status_ = s;
            return status_;
          }
        }
      } while (true);
    }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      std::lock_guard<std::mutex> l(mu_);
      RETURN_IF_ERROR(writer->WriteInt(fullname("file_cur_"), file_cur_));
      LOG(INFO) << "save file_cur_: " << file_cur_;
      if (readers_[0]) {
        RETURN_IF_ERROR(
            writer->WriteInt(fullname("begin_cur_"), readers_[0]->Tell()));
        LOG(INFO) << "save begin_cur_: " << readers_[0]->Tell();
      } else {
        RETURN_IF_ERROR(writer->WriteInt(fullname("begin_cur_"), begin_cur_));
        LOG(INFO) << "reader_ is null, save begin_cur_: " << begin_cur_;
      }
      return column::Status::OK();
    }

    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> l(mu_);
      RETURN_IF_ERROR(reader->ReadInt(fullname("file_cur_"), file_cur_));
      LOG(INFO) << "restore file_cur_: " << file_cur_;
      RETURN_IF_ERROR(reader->ReadInt(fullname("begin_cur_"), begin_cur_));
      LOG(INFO) << "restore begin_cur_: " << begin_cur_;
      return column::Status::OK();
    }

  private:
    enum { kRecordInterval = 500 };
    // int64 step_accum_{0};
    int64 record_interval_{0};
    // uint64_t get_next_costs_accum_{0};
    // uint64_t format_costs_accum_{0};
    // std::vector<uint64_t> read_costs_accum_;
    // std::vector<uint64_t> read_bytes_accum_;

    std::mutex mu_;
    int64_t file_cur_{0};
    int64_t begin_cur_{-1};
    std::vector<std::shared_ptr<arrow::RecordBatch>> datas_cache_;
    std::vector<BatchReader::AbstractReaderPtr> readers_;
    std::vector<std::unique_ptr<ColumnDataFormater>> column_formaters_;
    std::vector<std::vector<std::string>> selected_table_columns_;
    int32_t batch_size_;
    size_t table_num_;
    bool reach_end_;
    std::shared_ptr<recis::monitor::PointTag> point_tag_;
    Status status_;
    bool is_compressed_;
    bool turn_on_odps_open_storage_;
    bool force_seek_;
    bool force_deep_copy_;
  };

  const std::vector<std::vector<std::string>> paths_;
  const std::vector<std::vector<std::string>> input_columns_;
  const std::vector<std::vector<std::string>> selected_columns_;
  const std::vector<std::string> hash_features_;
  const std::vector<std::string> hash_types_;
  const std::vector<int64_t> hash_buckets_;
  const std::vector<std::string> dense_features_;
  std::vector<Tensor> dense_defaults_;
  int32_t batch_size_;
  std::string ds_name_;
  bool check_data_;
  std::string primary_key_;
  bool is_compressed_;
  bool turn_on_odps_open_storage_;
};

} // namespace

absl::StatusOr<std::shared_ptr<DatasetBase>> OdpsComboDataset::MakeDataset(
    const std::vector<std::vector<std::string>> &paths, bool is_compressed,
    int64_t batch_size,
    const std::vector<std::vector<std::string>> &selected_columns,
    const std::vector<std::vector<std::string>> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults,
    const bool &check_data, const std::string &primary_key,
    bool turn_on_odps_open_storage) {
  return std::shared_ptr<DatasetBase>(new Dataset(
      kDatasetName, paths, is_compressed, batch_size, selected_columns,
      input_columns, hash_features, hash_types, hash_buckets, dense_columns,
      detail::VecsToTensor<float>(dense_defaults), check_data, primary_key,
      turn_on_odps_open_storage));
}

std::shared_ptr<DatasetBuilder> OdpsComboDataset::MakeBuilder(
    bool is_compressed, int64_t batch_size,
    const std::vector<std::vector<std::string>> &selected_columns,
    const std::vector<std::vector<std::string>> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults,
    const bool &check_data, const std::string &primary_key,
    bool turn_on_odps_open_storage) {
  return DatasetBuilder::Make(
      [=](const std::vector<std::string> &paths)
          -> absl::StatusOr<std::shared_ptr<DatasetBase>> {
        return MakeDataset({paths}, is_compressed, batch_size, selected_columns,
                           input_columns, hash_features, hash_types,
                           hash_buckets, dense_columns, dense_defaults,
                           check_data, primary_key, turn_on_odps_open_storage);
      });
}

} // namespace dataset
} // namespace column
