
#include "column-io/dataset_impl/lake_batch_column_dataset.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "arrow/array.h"
#include "column-io/dataset/dataset.h"
#include "column-io/dataset/formater.h"
#include "column-io/dataset/vec_tensor_converter.h"
#include "column-io/dataset_impl/schema_parser.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/lake/lake_scan_reader.h"
#include "column-io/lake/status.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <chrono>
namespace column {
namespace dataset {
namespace {

const std::string kDatasetName{"LakeBatchColumnDataset"};
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &ds_name,
          const std::string &path,
          int64_t slice_index,
          int64_t slice_count,
          const std::vector<std::string> &selected_columns,
          const std::vector<std::string> &input_columns,
          const std::vector<std::string> &hash_features,
          const std::vector<std::string> &hash_types,
          const std::vector<int64_t> &hash_buckets,
          const std::vector<std::string> &dense_features,
          const std::vector<Tensor> &dense_defaults,
          bool is_compressed,
          int64_t batch_size,
          bool use_prefetch,
          int64_t prefetch_thread_num,
          int64_t prefetch_buffer_size)
      : DatasetBase(ds_name),
        ds_name_(ds_name),
        path_(path),
        slice_index_(slice_index),
        slice_count_(slice_count),
        selected_columns_(selected_columns),
        input_columns_(input_columns),
        hash_features_(hash_features),
        hash_types_(hash_types),
        hash_buckets_(hash_buckets),
        dense_features_(dense_features),
        is_compressed_(is_compressed),
        batch_size_(batch_size),
        use_prefetch_(use_prefetch),
        prefetch_thread_num_(prefetch_thread_num),
        prefetch_buffer_size_(prefetch_buffer_size) {
    dense_defaults_.reserve(dense_defaults.size());
    for (const Tensor &tensor : dense_defaults) {
      dense_defaults_.push_back(tensor);
    }
    // 行存输出格式下out_tensors顺序和selected_columns_需一致. dataset.py:GetNextFromIterator 也有这个参数的逻辑 
    std::string row_mode = std::getenv("ODPS_DATASET_ROW_MODE") ? std::getenv("ODPS_DATASET_ROW_MODE") : "0"; // DOUBT: need ailake_..row_mode?
    row_mode_tensor_split_ = (row_mode == "1") ;
  }

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::shared_ptr<IteratorBase>(new Iterator(
        std::dynamic_pointer_cast<Dataset>(shared_from_this()),
        absl::StrCat(prefix, "::LakeBatchColumnDataset"), ds_name_));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  private:
    void ParseCurrentPointTag() {
        std::string project = "";
        std::string table = "";
        if (!dataset()->path_.empty()) {
            // E.g. path=pangu://na61--cn-beijing/nebula/ai_lake/path1/path2/pathN/alimama_project/table_sampling/default/current/data
            // (cluster)/(project)/(table)/(column_family)/(partition)/data
            std::string filepath = dataset()->path_; 
            filepath = filepath.back() == '/' ? filepath.substr(0, filepath.size() - 1) : filepath;
            std::string segments[5]; // 
            for (size_t i = 0, pos = filepath.length(); i < 5; ++i) { // 从后往前提取5个段 data, partition, column_family, table, project, other prefix
                size_t end = pos;
                if (end > 0 && filepath[end - 1] == '/') { end--; } // 跳过当前位置的内容，找到前一个 '/'
                size_t start = filepath.rfind('/', end - 1);
                start = (start == std::string::npos) ? 0 : start+1;
                segments[i] = filepath.substr(start, end - start);
                pos = start - 1;
                if (pos == std::string::npos || pos == 0) { break; }
            }
            project = segments[4];
            table = segments[3];
        }
        point_tag_ = std::make_shared<recis::monitor::PointTag>(std::map<std::string, std::string>{
            {"ds_project", project.empty() ? "null" : project},
            {"ds_table", table.empty() ? "null" : table.substr(0, table.find("/"))},
            {"ds_name", "lake_batch"} // TODO: add ds_name_ for iterator from constructor
        });
    }
  public:
    explicit Iterator(const std::shared_ptr<Dataset> dataset,
                      const std::string &prefix, const std::string &ds_name)
        : DatasetIterator<Dataset>({dataset, prefix}) {
      reach_end_ = false;
    }

    Status InitSchema(std::shared_ptr<arrow::RecordBatch> &data) {
      if ((column_formater_))
        return Status::OK();
      // init formater
      column_formater_ = ColumnDataFormater::GetColumnDataFormater(
          dataset()->is_compressed_, false, dataset()->row_mode_tensor_split_);
      std::unordered_set<std::string> selected_columns;
      selected_columns.reserve(dataset()->selected_columns_.size());
      selected_columns.insert(dataset()->selected_columns_.begin(),
                              dataset()->selected_columns_.end());
      auto st = column_formater_->InitSchema(
          data->schema(), dataset()->hash_features_, dataset()->hash_types_, dataset()->hash_buckets_,
          dataset()->dense_features_, dataset()->dense_defaults_, selected_columns);
      if (!st.ok())
        column_formater_.reset();
      return st;
    }

    Status GetNextInternal(std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) override {
      std::lock_guard<std::mutex> l(mu_);
      do {
        // Using TableDataConnector to process all files.
        // Multi-thread prefetched could be done in the Connector.
        if (reach_end_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        if (!(status_.ok())) {
          return status_;
        }
        if (point_tag_ == nullptr) {
            ParseCurrentPointTag();
        }
        if (reader_) {
          *end_of_sequence = false;
          std::shared_ptr<arrow::RecordBatch> data;
          Status s;
          int64_t rows_before_read = reader_->Tell();
          int64_t time_before_read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
          auto common_st = reader_->ReadBatch(&data);
          int64_t time_after_read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
          int64_t delay_ms = std::max(0L, time_after_read_ms - time_before_read_ms);
          uint64_t rows_read = reader_->Tell() - rows_before_read ;

          metric_cli_->report("read_row",  rows_read, point_tag_, recis::monitor::PointType::kCounter); //TODO: read bytes from halo
          metric_cli_->report("read_batch", 1, point_tag_, recis::monitor::PointType::kCounter);
          metric_cli_->report("read_latency_ms", delay_ms, point_tag_, recis::monitor::PointType::kGauge);

          if (common_st.Ok()) {
            s = InitSchema(data);
            if ( !s.ok() ) {
              reader_.reset();
              status_ = s;
              return s;
            }
            std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
            RETURN_IF_ERROR(column_formater_->FormatSample(data, &formated_data));
            RETURN_IF_ERROR(column_formater_->FlatConvert(formated_data, out_tensors));

            if (dataset()->row_mode_tensor_split_) {
                if( outputs_row_spliter == nullptr) {
                    throw std::runtime_error("outputs_row_spliter is nullptr when getting row mode batches");
                }else{
                    outputs_row_spliter->emplace_back(0);
                }
                std::vector<Tensor> temp1_tensor_order;
                const auto& spliter_map = column_formater_->schema().flatconvert_tensor_spliter; // std::unordered_map<std::string, std::pair<size_t, size_t>>

                for (const std::string& col_name : dataset()->selected_columns_) {
                    size_t begin = 0, end = 0;
                    std::tie(begin, end) = spliter_map.at(col_name);
                    temp1_tensor_order.insert(temp1_tensor_order.end(), (*out_tensors).begin()+begin, (*out_tensors).begin()+end);
                    outputs_row_spliter->emplace_back(temp1_tensor_order.size());
                }
                out_tensors->swap(temp1_tensor_order);
            }
          } else if (common_st.code() == lake::Status::Code::kOutOfRange) {
            s = Status::OutOfRange(common_st.what());
          } else {
            s = Status::Internal(common_st.what());
          }
          if (s.ok()) {
            return s;
          }
          // deal with errors
          reader_.reset();
          if (s.code() != ErrorCode::OUT_OF_RANGE) {
            status_ = s;
            return s;
          }
          reach_end_ = true;
          continue;
        } else {
          // open reader
          Status s = Status::OK();
          reader_.reset(new lake::LakeScanReader());
          auto common_st = lake::Status::OK();
          common_st = reader_->Open(
              dataset()->path_,
              static_cast<size_t>(dataset()->slice_index_),
              static_cast<size_t>(dataset()->slice_count_),
              static_cast<size_t>(dataset()->batch_size_),
              dataset()->input_columns_, dataset()->use_prefetch_,
              static_cast<size_t>(dataset()->prefetch_thread_num_),
              static_cast<size_t>(dataset()->prefetch_buffer_size_));
          ParseCurrentPointTag();
          if (!common_st.Ok()) {
            if (common_st.code() == lake::Status::Code::kOutOfRange) {
              s = Status::OutOfRange(common_st.what());
              reach_end_ = true;
              continue;
            } else {
              s = Status::Internal("fail to open lake batch reader, error: ",
                                   common_st.code());
            }
          }
          if (!s.ok()) {
            status_ = s;
            reader_.reset();
            return s;
          }
		  if(begin_cur_ >= 0) {
			reader_->Seek(begin_cur_);
		  }
        }
      } while (true);
    }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      // mutex_lock l(mu_);
      std::lock_guard<std::mutex> l(mu_);
      if (reader_) {
        RETURN_IF_ERROR((writer->WriteScalar(fullname("begin_cur_"),
                                             reader_->Tell())));
        LOG(INFO) << "save begin_cur_: " << reader_->Tell();
      } else {
        RETURN_IF_ERROR(
            (writer->WriteScalar(fullname("begin_cur_"), begin_cur_)));
        LOG(INFO) << "reader_ is null, save begin_cur_: " << begin_cur_;
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> l(mu_);
      int64_t tmp;
      RETURN_IF_ERROR((reader->ReadScalar(fullname("begin_cur_"), &tmp)));
      LOG(INFO) << "restore begin_cur_: " << tmp;
      begin_cur_ = tmp;
      return Status::OK();
    }

  private:
    std::mutex mu_;
    int64_t begin_cur_{-1};
    std::unique_ptr<lake::LakeScanReader> reader_;
    std::unique_ptr<ColumnDataFormater> column_formater_;
    bool reach_end_;
    std::shared_ptr<recis::monitor::PointTag> point_tag_;
    Status status_;
  };

  std::string path_;

  int64_t slice_index_;
  int64_t slice_count_;

  const std::vector<std::string> selected_columns_;
  const std::vector<std::string> input_columns_;
  const std::vector<std::string> hash_features_;
  const std::vector<std::string> hash_types_;
  const std::vector<int64_t> hash_buckets_;
  const std::vector<std::string> dense_features_;
  std::vector<Tensor> dense_defaults_;
  bool row_mode_tensor_split_;
  bool is_compressed_;
  std::string ds_name_;
  int64_t batch_size_;
  bool use_prefetch_;
  int64_t prefetch_thread_num_;
  int64_t prefetch_buffer_size_;
};
const std::string kIndicator = "_indicator";

Status ReadLakeBatchRecordBatch(
    const std::string &path,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &dense_features, bool is_compressed,
    std::shared_ptr<arrow::RecordBatch> *data) {
  // parse path

  std::string lake_path;
  int64_t slice_index = 0;
  int64_t slice_count = 1;
  

  RETURN_IF_ERROR(LakeBatchColumnDatase::ParseConfig(
      path, lake_path, slice_index, slice_count));
  LOG(INFO) << "yd, "
            << ",lake_path = " << lake_path 
            << ",slice_index = " << slice_index
            << ",slice_count = " << slice_count;
  // init reader
  lake::LakeScanReader data_reader;
  auto common_st = lake::Status::OK();
  common_st = data_reader.Open(lake_path, slice_index,
                               slice_count, static_cast<size_t>(5), {}, false,
                               3, 3);
  if (common_st.code() == lake::Status::Code::kOutOfRange) {
    return Status::OutOfRange("");
  } else if (!common_st.Ok()) {
    return Status::Internal("fail to init lake batch reader, error: ",
                            common_st.what());
  }

  // read
  std::shared_ptr<arrow::RecordBatch> rb;
  common_st = data_reader.ReadBatch(&rb);
  if (common_st.code() == lake::Status::Code::kOutOfRange) {
    return Status::OutOfRange();
  } else if (!common_st.Ok()) {
    return Status::Internal("fail to read message, error info: ",
                            common_st.what());
  }
  LOG(INFO) << "read data, schema: " << rb->schema()->ToString().c_str();
  (*data) = rb;
  return Status::OK();
}
} // namespace

std::tuple<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>,
	std::string
	>
LakeBatchColumnDatase::ParseSchema(
    const std::string &paths,
    bool is_compressed,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto parser = SchemaParser::Make(ReadLakeBatchRecordBatch);
  return parser->ParseSchema({paths}, is_compressed, selected_columns,
                             hash_features, hash_types, hash_buckets, dense_columns,
                             detail::VecsToTensor(dense_defaults));
}

std::pair<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>>
LakeBatchColumnDatase::ParseSchemaByRows(
    const std::string &paths, bool is_compressed,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto parser = SchemaParser::Make(ReadLakeBatchRecordBatch);
  return parser->ParseSchemaByRows({paths}, is_compressed, selected_columns,
                                   hash_features, hash_types, hash_buckets, dense_columns,
                                   detail::VecsToTensor(dense_defaults));
}

Status LakeBatchColumnDatase::ParseConfig(
    const std::string &path, std::string &lake_path, int64_t &slice_index,
    int64_t &slice_count) {
  std::vector<std::string> configs =
      absl::StrSplit(path, "|", absl::SkipEmpty());

  if (configs.size() != 2) {
    return Status::Internal("`path` format illegal, should be: main dir| "
                            "slice index;slice count, path[", path, "]");
  }

  std::string &temp=configs[0];
  std::vector<std::string> kv = absl::StrSplit(temp, "=");
  lake_path = kv[1];
  LOG(INFO) << "read path: " << lake_path;

  std::vector<std::string> slice_config =
      absl::StrSplit(configs[1], ";", absl::SkipEmpty());
  if (!absl::SimpleAtoi(slice_config[0], &slice_index) || !(slice_index >= 0)) {
    return Status::Internal("`slice index` should not be negative: ",
                            slice_config[0]);
  }
  if (!absl::SimpleAtoi(slice_config[1], &slice_count) || !(slice_count > 0)) {
    return Status::Internal("`slice index` should be positive: ",
                            slice_config[1]);
  }
  return Status::OK();
}

absl::StatusOr<std::shared_ptr<DatasetBase>>
LakeBatchColumnDatase::MakeDataset(const std::string& path,
                                   const std::vector<std::string> &selected_columns,
                                   const std::vector<std::string> &input_columns,
                                   const std::vector<std::string> &hash_features,
                                   const std::vector<std::string> &hash_types,
                                   const std::vector<int64_t> &hash_buckets,
                                   const std::vector<std::string> &dense_features,
                                   const std::vector<Tensor> &dense_defaults,
                                   bool is_compressed, int64_t batch_size,
                                   bool use_prefetch,
                                   int64_t prefetch_thread_num,
                                   int64_t prefetch_buffer_size) {

  std::string lake_path{""};
  int64_t slice_index = 0;
  int64_t slice_count = -1;
  auto st =
      ParseConfig(path, lake_path, slice_index, slice_count);
  if (!st.ok()) {
    return absl::InternalError(st.error_message());
  }
  std::shared_ptr<DatasetBase> ret;
  ret.reset(new Dataset(
      kDatasetName, lake_path,
      slice_index, slice_count, selected_columns, input_columns,
      hash_features, hash_types, hash_buckets, dense_features,
      dense_defaults, is_compressed, batch_size, use_prefetch,
      prefetch_thread_num, prefetch_buffer_size));
  return ret;
}

std::shared_ptr<DatasetBuilder> LakeBatchColumnDatase::MakeBuilder(
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_features,
    const std::vector<std::vector<float>>& dense_defaults,
    bool is_compressed,
    int64_t batch_size,
    bool use_prefetch,
    int64_t prefetch_thread_num,
    int64_t prefetch_buffer_size) {
  return DatasetBuilder::Make(
      [=](const std::string &path)
          -> absl::StatusOr<std::shared_ptr<DatasetBase>> {
        return MakeDataset(path, selected_columns, input_columns,
                           hash_features, hash_types, hash_buckets, dense_features,
                           detail::VecsToTensor(dense_defaults), is_compressed,
                           batch_size, use_prefetch, prefetch_thread_num,
                           prefetch_buffer_size);
      });
}

} // namespace dataset
} // namespace column
