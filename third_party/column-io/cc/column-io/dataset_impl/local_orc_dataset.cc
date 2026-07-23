#include "column-io/dataset_impl/local_orc_dataset.h"
#include <random>
#include <cstddef>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
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
#include "column-io/arrow_reader/abstract_reader.h"
#include "column-io/arrow_reader/local_orc_reader.h"

namespace column {
namespace dataset {
namespace {


const std::string kDatasetName = "LocalOrcDataset";
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
          RETURN_IF_ERROR(column::BatchReader::LocalOrcReaderImpl::Create(
                              dataset()->paths_[index_], dataset()->input_columns_, &reader_));
        }
        std::shared_ptr<arrow::RecordBatch> rb;
        auto st = reader_->ReadBatch(&rb);
        //TODO: add byte, latency if need in local mode
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
      RETURN_IF_ERROR(column::BatchReader::LocalOrcReaderImpl::Create(
                          dataset()->paths_[index_], dataset()->input_columns_, &reader_));
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
    // std::unique_ptr<LocalOrcReader> reader_;
    std::unique_ptr<column::BatchReader::AbstractReader> reader_;
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
Status GetInputColumnsFromOrcSchema(
    const std::string &path,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &dense_features, bool is_compressed,
    std::vector<std::string> *input_columns_from_schema) {
  std::unique_ptr<column::BatchReader::AbstractReader> tmp_reader;
  RETURN_IF_ERROR(column::BatchReader::LocalOrcReaderImpl::Create(path, *input_columns_from_schema, &tmp_reader));
  std::unordered_set<std::string> useful_names;
  for (auto &feature : selected_columns) {
    useful_names.insert(feature);
  }
  for (auto &feature : dense_features) {
    useful_names.insert(feature);
  }
  if (is_compressed) {
    useful_names.insert(kIndicator);
  }
  std::shared_ptr<arrow::Schema> schema;
  RETURN_IF_ERROR(tmp_reader->ReadSchema(&schema));
  std::set<std::string> column_names;
  for (int i = 0; i < schema->num_fields(); ++i) {
    column_names.insert(schema->field(i)->name());
  }
  for (auto &column_name : column_names) {
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

Status ReadBatch(const std::string &path,
                       const std::unordered_set<std::string> &selected_columns,
                       const std::vector<std::string> &dense_features,
                       bool is_compressed,
                       std::shared_ptr<arrow::RecordBatch> *data) {
  // init reader
  std::vector<std::string> input_columns_from_schema;//有后缀的列column
  GetInputColumnsFromOrcSchema(path, selected_columns, dense_features,
                                is_compressed, &input_columns_from_schema);
  std::unique_ptr<column::BatchReader::AbstractReader> reader;
  RETURN_IF_ERROR(column::BatchReader::LocalOrcReaderImpl::Create(path, input_columns_from_schema, &reader));
  auto st = reader->ReadBatch(data);
  CHECK(st.ok()) << "Read batch failed at path [" << path << "]";
  LOG(INFO) << "read data, schema: " << (*(data))->schema()->ToString().c_str();
  return Status::OK();
}
} // namespace

absl::StatusOr<std::shared_ptr<DatasetBase>> LocalOrcDataset::MakeDataset(
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

std::shared_ptr<DatasetBuilder> LocalOrcDataset::MakeBuilder(
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
LocalOrcDataset::ParseSchema(
    const std::vector<std::string> &paths, bool is_compressed,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto parser = SchemaParser::Make(ReadBatch);
  return parser->ParseSchema(paths, is_compressed, selected_columns,
                             hash_features, hash_types, hash_buckets, dense_columns,
                             detail::VecsToTensor(dense_defaults));
}
} // namespace dataset
} // namespace column
