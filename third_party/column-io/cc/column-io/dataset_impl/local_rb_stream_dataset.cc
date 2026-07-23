#include "column-io/dataset_impl/local_rb_stream_dataset.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "arrow/buffer.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"
#include "arrow/ipc/reader.h"
#include "arrow/record_batch.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "column-io/dataset/formater.h"
#include "column-io/dataset/vec_tensor_converter.h"
#include "column-io/dataset_impl/schema_parser.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include <cstddef>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
namespace column {
namespace dataset {
namespace {
class ArrowInputStream : public arrow::io::InputStream {
public:
  arrow::Status Close() override { return arrow::Status::OK(); }

  bool closed() const override { return buffer_.get() == nullptr; }

  arrow::Status Abort() override { return arrow::Status::OK(); }

  arrow::Result<int64_t> Tell() const override {
    return arrow::Result<int64_t>(position_);
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void *out) override {
    int64_t bytes_read;
    if (buffer_) {
      if (position_ >= buffer_->size()) {
        bytes_read = 0;
        return arrow::Result<int64_t>(bytes_read);
      }
      int64_t left = buffer_->size() - position_;
      bytes_read = nbytes < left ? nbytes : left;
      std::memcpy(out, buffer_->data() + position_, bytes_read);
      position_ += bytes_read;
      return arrow::Result<int64_t>(bytes_read);
    } else {
      return arrow::Result<int64_t>(arrow::Status::RError("buffer is null"));
    }
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    std::shared_ptr<arrow::Buffer> out;
    if (buffer_) {
      int64_t bytes_read;
      int64_t left = buffer_->size() - position_;
      if (left <= 0) {
        bytes_read = 0;
      } else {
        bytes_read = nbytes < left ? nbytes : left;
      }
      out.reset(new arrow::Buffer(buffer_, position_, bytes_read));
      position_ += bytes_read;
      return arrow::Result<std::shared_ptr<arrow::Buffer>>(out);
    } else {
      return arrow::Result<std::shared_ptr<arrow::Buffer>>(
          arrow::Status::RError("buffer is null"));
    }
  }

  void set_buffer(const std::shared_ptr<arrow::Buffer> &buffer) {
    position_ = 0;
    buffer_ = buffer;
  }

private:
  std::shared_ptr<arrow::Buffer> buffer_;
  int64_t position_ = 0;
};

class LocalFileReader {
public:
  static Status OpenFile(const std::string dir_name,
                         std::unique_ptr<LocalFileReader> *out) {
    (*out).reset(new LocalFileReader(dir_name));
    return (*out)->Init();
  }

  Status ReadRecordBatch(std::shared_ptr<arrow::RecordBatch> *out) {
    if (index_ >= files_.size()) {
      return Status::OutOfRange();
    }
    const auto &current_file = files_[index_++];
    std::shared_ptr<arrow::Buffer> tmp_buf;
    RETURN_IF_ERROR(FileToBuffer(current_file, &tmp_buf));
    stream_->set_buffer(tmp_buf);
    auto st = reader_->ReadNext(out);
    if (!st.ok()) {
      return Status::Internal(st.message());
    }
    return Status::OK();
  }

  Status Seek(int64_t index) {
    index_ = index;
    return Status::OK();
  }

  int64_t Tell() { return index_; }

private:
  LocalFileReader(const std::string &dir_name)
      : dir_name_(dir_name), index_(0) {}

  Status Init() {
    RETURN_IF_ERROR(ListFiles(dir_name_, schema_file_, files_));
    stream_.reset(new ArrowInputStream);
    std::shared_ptr<arrow::Buffer> buf;
    RETURN_IF_ERROR(FileToBuffer(schema_file_, &buf));
    stream_->set_buffer(buf);
    auto result = arrow::ipc::RecordBatchStreamReader::Open(stream_.get());
    if (!result.ok()) {
      return Status::Internal(result.status().message());
    }
    reader_ = std::move(result).ValueOrDie();
    schema_ = reader_->schema();
    return Status::OK();
  }
  Status ListFiles(const std::string &dir, std::string &schema_file,
                   std::vector<std::string> &files) {
    DIR *d;
    struct dirent *node;
    schema_file.clear();
    files.clear();
    d = opendir(dir.c_str());
    if (!d) {
      return Status::InvalidArgument("Open dir [", dir, "] failed!");
    }
    while ((node = readdir(d)) != NULL) {
      if (absl::EndsWith(node->d_name, file_suffix_)) {
        if (absl::StrContains(node->d_name, "schema")) {
          schema_file = absl::StrCat(dir_name_, "/", node->d_name);
        } else {
          files.emplace_back(absl::StrCat(dir_name_, "/", node->d_name));
        }
      }
    }
    closedir(d);
    if (schema_file.size() == 0) {
      return Status::InvalidArgument("not find schame file in dir [", dir, "]");
    }
    return Status::OK();
  }

  Status FileToString(const std::string &file, std::string &data) {
    std::ifstream fstream(file);
    if (!fstream) {
      return Status::Internal("open file [", file, "] failed");
    }
    std::stringstream ss;
    ss << fstream.rdbuf();
    data = ss.str();
    return Status::OK();
  }

  Status StringToBuffer(const std::string &data,
                        std::shared_ptr<arrow::Buffer> *out) {
    auto result = arrow::AllocateBuffer(data.size());
    if (!result.ok()) {
      return Status::Internal(result.status().message());
    }
    (*out) = std::move(result).ValueOrDie();
    memcpy((*out)->mutable_data(), data.data(), data.size());
    return Status::OK();
  }

  Status FileToBuffer(const std::string &file,
                      std::shared_ptr<arrow::Buffer> *out) {
    std::string tmp_str;
    RETURN_IF_ERROR(FileToString(file, tmp_str));
    RETURN_IF_ERROR(StringToBuffer(tmp_str, out));
    return Status::OK();
  }

  std::string dir_name_;
  std::string schema_file_;
  std::vector<std::string> files_;
  std::unique_ptr<ArrowInputStream> stream_;
  std::shared_ptr<arrow::RecordBatchReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  int64_t index_;
  static const char *file_suffix_;
};
const char *LocalFileReader::file_suffix_ = ".rb_data";

const std::string kDatasetName = "LocalRBStreamDataset";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, const std::vector<std::string> &paths,
          bool is_compressed, int64_t batch_size,
          const std::vector<std::string> &selected_columns,
          const std::vector<std::string> &input_columns,
          const std::vector<std::string> &hash_features,
          std::vector<std::string> hash_types,
          std::vector<int64_t> hash_buckets,
          const std::vector<std::string> &dense_columns,
          const std::vector<Tensor> &dense_defaults)
      : DatasetBase(name), paths_(paths), is_compressed_(is_compressed),
        batch_size_(batch_size), selected_columns_(selected_columns),
        input_columns_(input_columns), hash_features_(hash_features), hash_types_(hash_types), hash_buckets_(hash_buckets),
        dense_columns_(dense_columns), dense_defaults_(dense_defaults) {}

protected:
  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::make_shared<Iterator>(
        std::dynamic_pointer_cast<Dataset>(shared_from_this()), prefix);
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    Iterator(const std::shared_ptr<Dataset> &dataset, const std::string &prefix)
        : DatasetIterator<Dataset>({dataset, prefix}), index_(0) {}

  protected:
    Status GetNextInternal(std::vector<Tensor> *outputs,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) {
      std::lock_guard<std::mutex> l(mu_);
      if (index_ >= dataset()->paths_.size()) {
        *end_of_sequence = true;
        return Status::OK();
      }
      while (true) {
        if (!reader_) {
          RETURN_IF_ERROR(
              LocalFileReader::OpenFile(dataset()->paths_[index_], &reader_));
        }
        std::shared_ptr<arrow::RecordBatch> rb;
        auto st = reader_->ReadRecordBatch(&rb);
        if (st.code() == ErrorCode::OUT_OF_RANGE) {
          if (index_ == dataset()->paths_.size() - 1) {
            *end_of_sequence = true;
            return Status::OutOfRange();
          } else {
            reader_.reset();
            index_++;
            continue;
          }
        } else if (!st.ok()) {
          return st;
        }
        st = (InitSchema(rb));
        outputs->clear();
        std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
        st = (formater_->FormatSample(rb, &formated_data));
        st = (formater_->FlatConvert(formated_data, outputs));
        break;
      }
      return Status::OK();
    }

    Status SaveInternal(IteratorStateWriter *writer) {
      std::lock_guard<std::mutex> l(mu_);
      RETURN_IF_ERROR(writer->WriteInt(fullname("index"), index_));
      return writer->WriteInt(fullname("file_cur"), reader_->Tell());
    }

    Status RestoreInternal(IteratorStateReader *reader) {
      std::lock_guard<std::mutex> l(mu_);
      int64_t index;
      RETURN_IF_ERROR(reader->ReadInt(fullname("index"), index));
      int64_t file_cur;
      RETURN_IF_ERROR(reader->ReadInt(fullname("file_cur"), file_cur));
      index_ = index;
      RETURN_IF_ERROR(
          LocalFileReader::OpenFile(dataset()->paths_[index_], &reader_));
      RETURN_IF_ERROR(reader_->Seek(file_cur));
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
          data->schema(), dataset()->hash_features_,dataset()->hash_types_, dataset()->hash_buckets_, dataset()->dense_columns_,
          dataset()->dense_defaults_, selected_columns);
      if (!st.ok()) {
        formater_.reset();
        return st;
      }
      return Status::OK();
    }
    std::unique_ptr<LocalFileReader> reader_;
    std::unique_ptr<ColumnDataFormater> formater_;
    std::mutex mu_;
    int64 index_;
  };
  const std::vector<std::string> paths_;
  const bool is_compressed_;
  const int64_t batch_size_;
  const std::vector<std::string> selected_columns_;
  const std::vector<std::string> input_columns_;
  const std::vector<std::string> hash_features_;
  const std::vector<std::string> hash_types_;
  const std::vector<int64_t> hash_buckets_;
  const std::vector<std::string> dense_columns_;
  const std::vector<Tensor> dense_defaults_;
};
const std::string kIndicator = "_indicator";

Status ReadRecordBatch(const std::string &path,
                       const std::unordered_set<std::string> &selected_columns,
                       const std::vector<std::string> &dense_features,
                       bool is_compressed,
                       std::shared_ptr<arrow::RecordBatch> *data) {
  // init reader
  std::unique_ptr<LocalFileReader> reader;
  RETURN_IF_ERROR(LocalFileReader::OpenFile(path, &reader));
  auto st = reader->ReadRecordBatch(data);
  CHECK(st.ok()) << "Read batch failed at path [" << path << "]";
  LOG(INFO) << "read data, schema: " << (*(data))->schema()->ToString().c_str();
  return Status::OK();
}
} // namespace

absl::StatusOr<std::shared_ptr<DatasetBase>> LocalRBStreamDataset::MakeDataset(
    const std::vector<std::string> &paths, bool is_compressed,
    int64_t batch_size, const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  return std::shared_ptr<DatasetBase>(
      new Dataset(kDatasetName, paths, is_compressed, batch_size,
                  selected_columns, input_columns, hash_features,hash_types, hash_buckets, dense_columns,
                  detail::VecsToTensor(dense_defaults)));
}

std::shared_ptr<DatasetBuilder> LocalRBStreamDataset::MakeBuilder(
    bool is_compressed, int64_t batch_size,
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
LocalRBStreamDataset::ParseSchema(
    const std::vector<std::string> &paths, bool is_compressed,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto parser = SchemaParser::Make(ReadRecordBatch);
  return parser->ParseSchema(paths, is_compressed, selected_columns,
                             hash_features, hash_types, hash_buckets, dense_columns,
                             detail::VecsToTensor(dense_defaults));
}
} // namespace dataset
} // namespace column
