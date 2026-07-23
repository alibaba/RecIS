
#include "column-io/dataset_impl/lake_stream_column_dataset.h"
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
#include "column-io/lake/lake_stream_reader.h"
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

const std::string kDatasetName{"LakeStreamColumnDataset"};
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &ds_name,
          const std::string &path,
          int64_t begin_time,
          int64_t end_time,
          int64_t partition_count,
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
          int64_t prefetch_buffer_size,
          const std::string &table_service_name)
      : DatasetBase(ds_name), ds_name_(ds_name), path_(path),
        begin_time_(begin_time), end_time_(end_time),
        partition_count_(partition_count),
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
        prefetch_buffer_size_(prefetch_buffer_size),
        table_service_name_(table_service_name) {
    dense_defaults_.reserve(dense_defaults.size());
    for (const auto &tensor : dense_defaults) {
      dense_defaults_.push_back(tensor);
    }
  }

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::shared_ptr<IteratorBase>(new Iterator(
        std::dynamic_pointer_cast<Dataset>(shared_from_this()),
        absl::StrCat(prefix, "::LakeStreamColumnDataset"), ds_name_));
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
            {"ds_name", "lake_stream"} // TODO: add ds_name_ for iterator from constructor
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
          dataset()->is_compressed_, false);
      std::unordered_set<std::string> selected_columns;
      selected_columns.reserve(dataset()->selected_columns_.size());
      selected_columns.insert(dataset()->selected_columns_.begin(),
                              dataset()->selected_columns_.end());
      auto st = column_formater_->InitSchema(
          data->schema(), dataset()->hash_features_,dataset()->hash_types_, dataset()->hash_buckets_,
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
          auto common_st = reader_->ReadBatch(&data);

          int64_t curr_sec = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
          int64_t delay_sec = std::max(0L, curr_sec - reader_->TellTimeStamp() / 1000000);
          metric_cli_->report("read_batch", 1, point_tag_, recis::monitor::PointType::kCounter);
          metric_cli_->report("sample_delay_sec", delay_sec, point_tag_, recis::monitor::PointType::kGauge);

          if (common_st.Ok()) {
            s = InitSchema(data);
            if ((s.ok())) {
              std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
              RETURN_IF_ERROR(
                  column_formater_->FormatSample(data, &formated_data));
              RETURN_IF_ERROR(
                  column_formater_->FlatConvert(formated_data, out_tensors));
            }
          } else {
            if (common_st.code() == lake::Status::Code::kOutOfRange) {
              s = Status::OutOfRange(common_st.what());
            } else {
              s = Status::Internal(common_st.what());
            }
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
          reader_.reset(new lake::LakeStreamReader());
          auto common_st = lake::Status::OK();
          common_st = reader_->Open(
              dataset()->table_service_name_, dataset()->path_,
              static_cast<size_t>(dataset()->slice_index_),
              static_cast<size_t>(dataset()->slice_count_),
              static_cast<size_t>(dataset()->batch_size_),
              dataset()->input_columns_, dataset()->use_prefetch_,
              static_cast<size_t>(dataset()->prefetch_thread_num_),
              static_cast<size_t>(dataset()->prefetch_buffer_size_),
              static_cast<size_t>(dataset()->partition_count_));
          ParseCurrentPointTag();
          if (!common_st.Ok()) {
            if (common_st.code() == lake::Status::Code::kOutOfRange) {
              s = Status::OutOfRange(common_st.what());
              reach_end_ = true;
              continue;
            } else {
              s = Status::Internal("fail to open lake stream reader, error: ",
                                   common_st.code());
            }
          } else {
            int64_t begin = dataset()->begin_time_;
            if (begin_cur_ >= 0) { // begin_cur_ is inited from RestoreInternal
              begin = begin_cur_;
            }
            if (dataset()->end_time_ <= 0) {
              common_st = reader_->SeekTimeStamp(begin);
            } else {
              common_st =
                  reader_->SeekTimeStampRange(begin, dataset()->end_time_);
            }
            if (!common_st.Ok()) {
              if (common_st.code() == lake::Status::Code::kOutOfRange) {
                // s = errors::OutOfRange(common_st.code());
                s = Status::OutOfRange();
                reach_end_ = true;
                continue;
              } else {
                s = Status::Internal(
                    "fail to seek lake stream reader: ", dataset()->begin_time_,
                    ", ", dataset()->end_time_, ", error: ", common_st.code());
              }
            }
          }
          if (!s.ok()) {
            status_ = s;
            reader_.reset();
            return s;
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
                                             reader_->TellTimeStamp())));
        LOG(INFO) << "save begin_cur_: " << reader_->TellTimeStamp();
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
    std::unique_ptr<lake::LakeStreamReader> reader_;
    std::unique_ptr<ColumnDataFormater> column_formater_;
    bool reach_end_;
    std::shared_ptr<recis::monitor::PointTag> point_tag_;
    Status status_;
  };

  std::string path_;
  int64_t begin_time_;
  int64_t end_time_;
  int64_t partition_count_;
  int64_t slice_index_;
  int64_t slice_count_;

  const std::vector<std::string> selected_columns_;
  const std::vector<std::string> input_columns_;
  const std::vector<std::string> hash_features_;
  const std::vector<std::string> hash_types_;
  const std::vector<int64_t> hash_buckets_;
  const std::vector<std::string> dense_features_;
  std::vector<Tensor> dense_defaults_;
  bool is_compressed_;
  std::string ds_name_;
  int64_t batch_size_;
  bool use_prefetch_;
  int64_t prefetch_thread_num_;
  int64_t prefetch_buffer_size_;
  std::string table_service_name_;
};
const std::string kIndicator = "_indicator";

Status ReadLakeStreamRecordBatch(
    const std::string &path,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &dense_features, bool is_compressed,
    std::shared_ptr<arrow::RecordBatch> *data) {
  // parse path

  std::string lake_path;
  int64_t begin_time = -1;
  int64_t end_time = -1;
  int64_t partition_count = -1;
  int64_t slice_index = 0;
  int64_t slice_count = 1;
  std::string table_service_name = "";
  bool use_querier = false;

  RETURN_IF_ERROR(LakeStreamColumnDatase::ParseConfig(
      path, lake_path, begin_time, end_time, partition_count, slice_index,
      slice_count, table_service_name, use_querier));
  LOG(INFO) << "zzx, "
            << ",lake_path = " << lake_path << ",begin_time = " << begin_time
            << ",end_time = " << end_time
            << ",partition_count = " << partition_count
            << ",slice_index = " << slice_index
            << ",slice_count = " << slice_count
            << ",table_service_name = " << table_service_name
            << ",use_querier = " << use_querier;
  // init reader
  lake::LakeStreamReader data_reader;
  auto common_st = lake::Status::OK();
  common_st = data_reader.Open(table_service_name, lake_path, slice_index,
                               slice_count, static_cast<size_t>(5), {}, false,
                               3, 3, static_cast<size_t>(partition_count));
  if (common_st.code() == lake::Status::Code::kOutOfRange) {
    return Status::OutOfRange("");
  } else if (!common_st.Ok()) {
    return Status::Internal("fail to init lake stream reader, error: ",
                            common_st.what());
  }
  struct timeval tval;
  gettimeofday(&tval, NULL);
  int64 current_time = tval.tv_sec * 1000000 + tval.tv_usec;
  begin_time = std::min<int64_t>(begin_time, current_time);
  common_st = data_reader.SeekTimeStamp(begin_time);
  if (common_st.code() == lake::Status::Code::kOutOfRange) {
    return Status::OutOfRange();
  } else if (!common_st.Ok()) {
    return Status::Internal("fail to seek reader to: ", begin_time,
                            ", error: ", common_st.what());
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
LakeStreamColumnDatase::ParseSchema(
    const std::string &paths, bool is_compressed,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto parser = SchemaParser::Make(ReadLakeStreamRecordBatch);
  return parser->ParseSchema({paths}, is_compressed, selected_columns,
                             hash_features, hash_types, hash_buckets, dense_columns,
                             detail::VecsToTensor(dense_defaults));
}

Status LakeStreamColumnDatase::ParseConfig(
    const std::string &path, std::string &lake_path, int64_t &begin_time,
    int64_t &end_time, int64_t &partition_count, int64_t &slice_index,
    int64_t &slice_count, std::string &table_service_name, bool &use_querier) {
  std::vector<std::string> configs =
      absl::StrSplit(path, "|", absl::SkipEmpty());

  if (configs.size() != 4) {
    return Status::Internal("`path` format illegal, should be: main dir|start "
                            "time;end time|hash|slice index;slice count, path[",
                            path, "]");
  }

  std::vector<std::string> path_and_service =
      absl::StrSplit(configs[0], ";", absl::SkipEmpty());
  for (auto &temp : path_and_service) {
    std::vector<std::string> kv = absl::StrSplit(temp, "=");
    if (kv[0] == "serviceName") {
      table_service_name = kv[1];
      use_querier = true;
    } else if (kv[0] == "path") {
      lake_path = kv[1];
    }
  }
  LOG(INFO) << "read path: " << lake_path
            << ", service name: " << table_service_name;

  std::vector<std::string> times =
      absl::StrSplit(configs[1], ";", absl::SkipEmpty());
  for (auto &time : times) {
    std::vector<std::string> kv = absl::StrSplit(time, "=");
    if (kv[0] == "begin") {
      if (!absl::SimpleAtoi(kv[1], &begin_time) || !(begin_time > 0)) {
        return Status::Internal("`begin` in `paths` should be positive: ",
                                kv[1]);
      }
    } else if (kv[0] == "end") {
      if (!absl::SimpleAtoi(kv[1], &end_time) || !(end_time > 0)) {
        return Status::Internal("`end` in `paths` should be positive: ", kv[1]);
      }
    }
  }
  if (!absl::SimpleAtoi(configs[2], &partition_count) ||
      (partition_count < 0)) {
    return Status::Internal("`partition count` should be positive: ",
                            configs[2]);
  }

  std::vector<std::string> slice_config =
      absl::StrSplit(configs[3], ";", absl::SkipEmpty());
  if (!absl::SimpleAtoi(slice_config[0], &slice_index) || !(slice_index >= 0)) {
    return Status::Internal("`slice index` should not beg negative: ",
                            slice_config[0]);
  }
  if (!absl::SimpleAtoi(slice_config[1], &slice_count) || !(slice_count > 0)) {
    return Status::Internal("`slice index` should be positive: ",
                            slice_config[1]);
  }
  return Status::OK();
}

absl::StatusOr<std::shared_ptr<DatasetBase>>
LakeStreamColumnDatase::MakeDataset(const std::string &path,
                                    const std::vector<std::string> &selected_columns,
                                    const std::vector<std::string> &input_columns,
                                    const std::vector<std::string> &hash_features,
                                    const std::vector<std::string> &hash_types,
                                    const std::vector<int64_t> &hash_buckets,
                                    const std::vector<std::string> &dense_features,
                                    const std::vector<std::vector<float>> &dense_defaults,
                                    bool is_compressed, int64_t batch_size,
                                    bool use_prefetch,
                                    int64_t prefetch_thread_num,
                                    int64_t prefetch_buffer_size) {
  bool use_querier{false};
  std::string table_service_name{""};
  std::string lake_path{""};
  int64_t begin_time = -1;
  int64_t end_time = -1;
  int64_t partition_count = -1;
  int64_t slice_index = -1;
  int64_t slice_count = -1;
  auto st =
      ParseConfig(path, lake_path, begin_time, end_time, partition_count,
                  slice_index, slice_count, table_service_name, use_querier);
  if (!st.ok()) {
    return absl::InternalError(st.error_message());
  }
  std::shared_ptr<DatasetBase> ret;
  ret.reset(new Dataset(
      kDatasetName, lake_path, begin_time, end_time, partition_count,
      slice_index, slice_count, selected_columns, input_columns,
      hash_features, hash_types, hash_buckets, dense_features,
      detail::VecsToTensor(dense_defaults), is_compressed,
      batch_size, use_prefetch, prefetch_thread_num, prefetch_buffer_size,
      table_service_name));
  return ret;
}

std::shared_ptr<DatasetBuilder> LakeStreamColumnDatase::MakeBuilder(
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_features,
    const std::vector<std::vector<float>> &dense_defaults, bool is_compressed,
    int64_t batch_size, bool use_prefetch, int64_t prefetch_thread_num,
    int64_t prefetch_buffer_size) {
  return DatasetBuilder::Make(
      [=](const std::string &path)
          -> absl::StatusOr<std::shared_ptr<DatasetBase>> {
        return MakeDataset(path, selected_columns, input_columns,
                           hash_features, hash_types, hash_buckets,
                           dense_features, dense_defaults, is_compressed,
                           batch_size, use_prefetch, prefetch_thread_num,
                           prefetch_buffer_size);
      });
}

} // namespace dataset
} // namespace column
