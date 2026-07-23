#if (_GLIBCXX_USE_CXX11_ABI != 1)

#include "column-io/dataset_impl/odps_table_column_dataset.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "arrow/record_batch.h"
#include "arrow/type.h"
#include "column-io/dataset/formater.h"
#include "column-io/dataset/vec_tensor_converter.h"
#include "column-io/dataset_impl/path_parser.h"
#include "column-io/dataset_impl/schema_parser.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include "column-io/odps/wrapper/odps_table_file_system.h"
#include "column-io/odps/wrapper/odps_table_reader.h"
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
namespace column {
namespace dataset {
namespace {
const std::string kDatasetName = "OdpsTableColumn";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name,
          const std::vector<std::string> &paths,
          bool is_compressed,
          int64_t batch_size,
          const std::vector<std::string> &selected_columns,
          const std::vector<std::string> &input_columns,
          const std::vector<std::string> &hash_features,
          const std::vector<std::string> &hash_types,
          const std::vector<int64_t> &hash_buckets,
          const std::vector<std::string> &dense_features,
          const std::vector<Tensor> &dense_defaults)
      : DatasetBase(name), paths_(std::move(paths)),
        input_columns_(input_columns), batch_size_(batch_size),
        selected_columns_(std::move(selected_columns)),
        hash_features_(hash_features),
        hash_types_(hash_types),
        hash_buckets_(hash_buckets), 
		dense_features_(dense_features),
        dense_defaults_(dense_defaults), is_compressed_(is_compressed),
        ds_name_(name) {
    fs_ = odps::wrapper::OdpsTableFileSystem::Instance();
  }

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::shared_ptr<IteratorBase>(
        new Iterator(std::dynamic_pointer_cast<Dataset>(shared_from_this()),
                     absl::StrCat(prefix, "::TableColumnDataset"), ds_name_));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  private:
    void ParseCurrentPointTag() {
        std::string project = "";
        std::string table = "";
        if (file_cur_ < dataset()->paths_.size()) {
            const std::string& filepath = dataset()->paths_[file_cur_];
            // E.g. "odps://rec_test/tables/mainsearch/ds=20251231?start=1&end=10"
            ParseOdpsUrl(filepath, project, table);
            // 兼容tunnel传参格式
            ParseTunnelTableFormat(filepath, project, table);
        }
        point_tag_ = std::make_shared<recis::monitor::PointTag>(std::map<std::string, std::string>{
            {"ds_project", project.empty() ? "null" : project},
            {"ds_table", table.empty() ? "null" : table},
            {"ds_name", "odps_algo"}, // TODO: add ds_name_ for iterator from constructor
            {"multiprocess_seq", std::to_string(this->get_child_order())}
        });
    }
  public:
    explicit Iterator(const std::shared_ptr<Dataset> datataset,
                      const std::string &prefix, const std::string &ds_name)
        : DatasetIterator<Dataset>({datataset, prefix}) {
      reach_end_ = false;
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
          uint64_t size_before_read = reader_->GetReadBytes();
          int64_t time_before_read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
          
          auto s = reader_->ReadBatch(&data);
          uint64_t read_size = reader_->GetReadBytes() - size_before_read;
          int64_t time_after_read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
          int64_t delay_ms = std::max(0L, time_after_read_ms - time_before_read_ms);

          metric_cli_->report("read_byte",  read_size, point_tag_, recis::monitor::PointType::kCounter);
          metric_cli_->report("read_batch", 1, point_tag_, recis::monitor::PointType::kCounter);
          metric_cli_->report("read_latency_ms", delay_ms, point_tag_, recis::monitor::PointType::kGauge);

          if (s.ok()) {
            RETURN_IF_ERROR(InitSchema(data));
          }
          if (s.ok()) {
            std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
            RETURN_IF_ERROR(formater_->FormatSample(data, &formated_data));
            RETURN_IF_ERROR(formater_->FlatConvert(formated_data, out_tensors));
            return s;
          }
          // deal with errors
          reader_.reset();
          if (s.code() != ErrorCode::OUT_OF_RANGE) {
            return s;
          }
          ++file_cur_;
          ParseCurrentPointTag();
        } else {
          if (file_cur_ >= dataset()->paths_.size()) {
            reach_end_ = true;
            continue;
          }
          // open reader
          auto *fs = dataset()->fs_;
          column::odps::AlgoReader *raw_reader = nullptr;
          int32_t batch_size = dataset()->batch_size_;
          if (dataset()->is_compressed_)
            batch_size = std::max(1, dataset()->batch_size_ / 8);
          LOG(INFO) << "launch open file: " << dataset()->paths_[file_cur_];
          auto algo_st =
              fs->CreateFileReader(dataset()->paths_[file_cur_], &raw_reader,
                                   batch_size, dataset()->input_columns_);
          LOG(INFO) << "open file: " << dataset()->paths_[file_cur_];
          if (!algo_st.ok()) {
            status_ =
                Status::InvalidArgument("Create odps file reader failed: ",
                                        dataset()->paths_[file_cur_]);
            return status_;
          }
          auto odps_reader =
              dynamic_cast<column::odps::wrapper::OdpsTableReader *>(
                  raw_reader);
          if (odps_reader == nullptr) {
            status_ = Status::InvalidArgument("Cast odps file reader failed: ",
                                              dataset()->paths_[file_cur_]);
            return status_;
          }
          reader_.reset(odps_reader);
          if (begin_cur_ >= 0) { // begin_cur_ is inited from RestoreInternal
            // validate begin_cur_
            size_t table_size;
            reader_->CountRecords(&table_size);
            if (begin_cur_ == table_size) {
              LOG(INFO) << "file: " << dataset()->paths_[file_cur_]
                        << " reached end, skip. begin_cur_:  " << begin_cur_
                        << ", table_size: " << table_size;
              reader_.reset();
              ++file_cur_;
              begin_cur_ = -1;
              continue;
            }
            // seek
            LOG(INFO) << "seek file " << dataset()->paths_[file_cur_] << " to "
                      << begin_cur_;
            if (!reader_->Seek(begin_cur_).ok()) {
              status_ = Status::InvalidArgument(
                  "Fail to seek path: ", dataset()->paths_[file_cur_],
                  ", to offset: ", begin_cur_);
              reader_.reset();
              return status_;
            }
            begin_cur_ = -1;
          }
        }
      } while (true);
    }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      std::lock_guard<std::mutex> l(mu_);
      RETURN_IF_ERROR(writer->WriteInt(fullname("file_cur_"), file_cur_));
      LOG(INFO) << "save file_cur_: " << file_cur_;
      if (reader_) {
        RETURN_IF_ERROR(
            writer->WriteInt(fullname("begin_cur_"), reader_->Tell()));
        LOG(INFO) << "save begin_cur_: " << reader_->Tell();
      } else {
        RETURN_IF_ERROR(writer->WriteInt(fullname("begin_cur_"), begin_cur_));
        LOG(INFO) << "reader_ is null, save begin_cur_: " << begin_cur_;
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> l(mu_);
      RETURN_IF_ERROR(reader->ReadInt(fullname("file_cur_"), file_cur_));
      LOG(INFO) << "restore file_cur_: " << file_cur_;
      RETURN_IF_ERROR(reader->ReadInt(fullname("begin_cur_"), begin_cur_));
      LOG(INFO) << "restore begin_cur_: " << begin_cur_;
      return Status::OK();
    }

  private:
    Status InitSchema(std::shared_ptr<arrow::RecordBatch> &data) {
      if (formater_)
        return Status::OK();
      // init formater
      formater_ = ColumnDataFormater::GetColumnDataFormater(
          dataset()->is_compressed_, false);
      std::unordered_set<std::string> selected_columns;
      selected_columns.reserve(dataset()->selected_columns_.size());
      selected_columns.insert(dataset()->selected_columns_.begin(),
                              dataset()->selected_columns_.end());
      auto st = formater_->InitSchema(
          data->schema(), dataset()->hash_features_, dataset()->hash_types_, dataset()->hash_buckets_, dataset()->dense_features_,
          dataset()->dense_defaults_, selected_columns);
      if (!st.ok()) {
        formater_.reset();
        return st;
      }
      return Status::OK();
    }
    std::mutex mu_;
    int64_t file_cur_{0};
    int64_t begin_cur_{-1};
    std::unique_ptr<column::odps::wrapper::OdpsTableReader> reader_;
    std::unique_ptr<ColumnDataFormater> formater_;
    bool reach_end_;
    std::shared_ptr<recis::monitor::PointTag> point_tag_;
    Status status_;
  };

  const std::vector<std::string> paths_;
  const std::vector<std::string> input_columns_;
  const std::vector<std::string> selected_columns_;
  const std::vector<std::string> hash_features_;
  const std::vector<std::string> hash_types_;
  const std::vector<int64_t> hash_buckets_;
  const std::vector<std::string> dense_features_;
  std::vector<Tensor> dense_defaults_;
  bool is_compressed_;
  int32_t batch_size_;
  column::odps::wrapper::OdpsTableFileSystem *fs_;
  std::string ds_name_;
};

Status ReadOdpsTableColumnBatch(const std::string &path,
                                std::shared_ptr<arrow::RecordBatch> &output) {
  column::odps::wrapper::OdpsTableFileSystem *fs =
      odps::wrapper::OdpsTableFileSystem::Instance();
  column::odps::AlgoReader *raw_reader = nullptr;
  auto algo_st = fs->CreateFileReader(path, &raw_reader, 1, {});
  if (!algo_st.ok()) {
    return algo_st;
  }
  auto odps_reader =
      dynamic_cast<column::odps::wrapper::OdpsTableReader *>(raw_reader);

  return Status::OK();
}

const std::string kIndicator = "_indicator";
Status GetInputColumnsFromOdpsSchema(
    odps::wrapper::OdpsTableFileSystem *fs, const std::string &path,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &dense_features, bool is_compressed,
    std::vector<std::string> *input_columns_from_schema) {
  // init reader
  odps::AlgoReader *raw_reader = nullptr;
  std::vector<std::string> empty_vec;
  auto s = fs->CreateFileReader(path, &raw_reader, 1, empty_vec);
  CHECK(s.ok() && raw_reader != nullptr)
      << "Create table reader for path [" << path << "] failed";
  auto odps_reader = dynamic_cast<odps::wrapper::OdpsTableReader *>(raw_reader);
  CHECK(odps_reader != nullptr) << "fail to cast FileReader to OdpsTableReader";
  std::unique_ptr<odps::wrapper::OdpsTableReader> reader(odps_reader);
  // read schema
  std::unordered_map<std::string, std::string> schema;
  s = reader->GetSchema(&schema);
  CHECK(s.ok()) << "fail to GetSchema from path: " << path
                << ", error: " << s.error_message();

  std::unordered_set<std::string> useful_names;
  for (auto &feature : selected_columns) {
    useful_names.insert(feature);
  }
  for (auto &feature : dense_features) {
    useful_names.insert(feature);
  }
  if (is_compressed)
    useful_names.insert(kIndicator);

  for (auto &type_info : schema) {
    std::string column_name = type_info.first;
    if (is_compressed) {
      size_t pos = column_name.find_last_of("_");
      if (pos == std::string::npos) {
        LOG(INFO) << "compressed column name has no indicator suffix, skip: "
                  << column_name;
        continue;
      }
      std::string alias = column_name.substr(0, pos);
      if (useful_names.count(alias) == 0) {
        LOG(INFO) << "compressed column not use, skip: " << column_name;
        continue;
      }
    } else {
      if (useful_names.count(column_name) == 0) {
        LOG(INFO) << "column not use, skip: " << column_name;
        continue;
      }
    }
    input_columns_from_schema->push_back(column_name);
  }

  return Status::OK();
}
Status
ReadOdpsRecordBatch(const std::string &path,
                    const std::unordered_set<std::string> &selected_columns,
                    const std::vector<std::string> &dense_features,
                    bool is_compressed,
                    std::shared_ptr<arrow::RecordBatch> *data) {
  // init fs
  auto fs = odps::wrapper::OdpsTableFileSystem::Instance();
  CHECK(fs) << "Init odps filesystem failed";
  // init reader
  std::vector<std::string> input_columns_from_schema;
  GetInputColumnsFromOdpsSchema(fs, path, selected_columns, dense_features,
                                is_compressed, &input_columns_from_schema);
  odps::AlgoReader *raw_reader = nullptr;
  auto s =
      fs->CreateFileReader(path, &raw_reader, 8, input_columns_from_schema);
  CHECK(s.ok() && raw_reader != nullptr)
      << "Create table reader for path [" << path << "] failed";
  auto odps_reader = dynamic_cast<odps::wrapper::OdpsTableReader *>(raw_reader);
  CHECK(odps_reader != nullptr) << "fail to cast FileReader to OdpsTableReader";
  std::unique_ptr<odps::wrapper::OdpsTableReader> reader(odps_reader);
  // read data
  std::shared_ptr<arrow::RecordBatch> rb;
  s = reader->ReadBatch(&rb);
  CHECK(s.ok()) << "fail to read RecordBatch from path: " << path
                << ", error: " << s.error_message();
  LOG(INFO) << "read data, schema: " << rb->schema()->ToString().c_str();
  (*data) = rb;
  return Status::OK();
}
} // namespace

absl::StatusOr<std::shared_ptr<DatasetBase>>
OdpsTableColumnDataset::MakeDataset(
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
  return std::shared_ptr<DatasetBase>(
      new Dataset(kDatasetName, paths, is_compressed, batch_size,
                  selected_columns, input_columns, hash_features, hash_types, hash_buckets, dense_columns,
                  detail::VecsToTensor<float>(dense_defaults)));
}

std::shared_ptr<DatasetBuilder> OdpsTableColumnDataset::MakeBuilder(
    bool is_compressed,
    int64_t batch_size,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  return DatasetBuilder::Make(
      [=](const std::string &path)
          -> absl::StatusOr<std::shared_ptr<DatasetBase>> {
        return MakeDataset({path}, is_compressed, batch_size, selected_columns,
                           input_columns, hash_features, hash_types, hash_buckets, dense_columns,
                           dense_defaults);
      });
}

std::tuple<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>,
	std::string
	>
OdpsTableColumnDataset::ParseSchema(
    const std::vector<std::string> &paths,
    bool is_compressed,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types, 
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto parser = SchemaParser::Make(ReadOdpsRecordBatch);
  return parser->ParseSchema(paths, is_compressed, selected_columns,
                             hash_features, hash_types, hash_buckets, dense_columns,
                             detail::VecsToTensor(dense_defaults));
}

Status OdpsTableColumnDataset::GetTableSize(const std::string &path,
                                            size_t *ret) {
  auto fs = odps::wrapper::OdpsTableFileSystem::Instance();
  return fs->GetFileSize(path, ret);
}

std::unordered_map<std::string, std::string> OdpsTableColumnDataset::GetOdpsTableFeatures(const char* str_path, bool is_compressed) {
  std::string path = std::string(str_path);
  LOG(INFO) << "OdpsTableColumnDataset::GetOdpsTableFeatures: " << path << "; is_compressed " << is_compressed;
  auto fs = odps::wrapper::OdpsTableFileSystem::Instance();
  std::unordered_map<std::string, std::string> tmp;
  if (!fs) {
    LOG(ERROR) << "Get OdpsTableFileSystem of path [" << path << "] failed";
    return tmp; 
  }

  column::odps::AlgoReader *raw_reader = nullptr;
  auto s = fs->CreateFileReader(path, &raw_reader);
  if (!s.ok()) {
    LOG(ERROR) << "Create OdpsTableReader failed, path: " << path;
    return tmp;
  }

  // create reader
  std::unique_ptr<column::odps::wrapper::OdpsTableReader> reader(
      reinterpret_cast<column::odps::wrapper::OdpsTableReader *>(raw_reader));

  std::unordered_map<std::string, std::string> schema;
  s = reader->GetSchema(&schema);
  if (!s.ok()) {
    LOG(ERROR) << "Get OdpsTableSchema failed, path: " << path;
    return tmp;
  }
  return schema;
}

} // namespace dataset
} // namespace column

#endif // (_GLIBCXX_USE_CXX11_ABI != 0)
