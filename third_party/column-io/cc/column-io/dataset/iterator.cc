#include "column-io/dataset/iterator.h"

#include <memory>
#include <string>
// #include "absl/container/flat_hash_map.h"

#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/tensor.pb.h"
#include "column-io/framework/tensor_util.h"
#include "column-io/framework/types.h"

namespace column {
namespace dataset {
namespace {
class IteratorStateWriterImpl : public IteratorStateWriter {
public:
  IteratorStateWriterImpl(std::unordered_map<std::string, Tensor> *states_map)
      : states_map_(states_map) {}
  Status WriteString(const std::string &key, const std::string &val) {
    COLUMN_RETURN_NOT_OK(CheckKey(key));
    return Write<std::string>(key, val);
  }
  Status WriteScalar(const std::string &key, int64_t val) {
    COLUMN_RETURN_NOT_OK(CheckKey(key));
    return Write<int64_t>(key, val);
  }
  Status WriteScalar(const std::string &key, const std::string &val) {
    COLUMN_RETURN_NOT_OK(CheckKey(key));
    return Write<std::string>(key, val);
  }
  Status WriteInt(const std::string &key, int64_t val) {
    COLUMN_RETURN_NOT_OK(CheckKey(key));
    return Write<int64_t>(key, val);
  }
  Status WriteFloat(const std::string &key, double val) {
    COLUMN_RETURN_NOT_OK(CheckKey(key));
    return Write<double>(key, val);
  }
  Status WriteTensor(const std::string &key, const Tensor tensor) {
    COLUMN_RETURN_NOT_OK(CheckKey(key));
    (*states_map_)[key] = tensor;
    return Status::OK();
  }

  IteratorStateWriterImpl(IteratorStateWriterImpl &) = delete;
  IteratorStateWriterImpl &operator=(IteratorStateWriterImpl &) = delete;

private:
  template <typename T> Status Write(const std::string &key, const T &value) {
    Tensor tensor(DataTypeToEnum<T>::value);
    tensor.Raw<T>()[0] = value;
    (*states_map_)[key] = tensor;
    return Status::OK();
  }

  Status CheckKey(const std::string &key) {
    if (states_map_->count(key)) {
      return Status::Internal("duplicated key [", key, "]");
    }
    return Status::OK();
  }

  std::unordered_map<std::string, Tensor> *states_map_;
};

class IteratorStateReaderImpl : public IteratorStateReader {
public:
  IteratorStateReaderImpl(
      const std::unordered_map<std::string, Tensor> *states_map)
      : states_map_(states_map) {}
  bool Contain(const std::string &key) { return states_map_->count(key); }
  bool Contains(const std::string &key) { return states_map_->count(key); }
  Status ReadString(const std::string &key, std::string &val) {
    Tensor tmp_val;
    COLUMN_RETURN_NOT_OK(Read<std::string>(key, tmp_val));
    val = tmp_val.Raw<std::string>()[0];
    return Status::OK();
  }
  Status ReadInt(const std::string &key, int64_t &val) {
    Tensor tmp_val;
    COLUMN_RETURN_NOT_OK(Read<int64_t>(key, tmp_val));
    val = tmp_val.Raw<int64_t>()[0];
    return Status::OK();
  }
  Status ReadFloat(const std::string &key, double &val) {
    Tensor tmp_val;
    COLUMN_RETURN_NOT_OK(Read<double>(key, tmp_val));
    val = tmp_val.Raw<double>()[0];
    return Status::OK();
  }
  Status ReadTensor(const std::string &key, Tensor &tensor) {
    if (states_map_->count(key) == 0) {
      return Status::Internal("not find key [", key, "]");
    }
    tensor = (*states_map_).at(key);
    return Status::OK();
  }
  Status ReadScalar(const std::string &key, int64_t *val) {
    Tensor tmp_val;
    COLUMN_RETURN_NOT_OK(Read<int64_t>(key, tmp_val));
    *val = tmp_val.Raw<int64_t>()[0];
    return Status::OK();
  }
  Status ReadScalar(const std::string &key, std::string *val) {
    Tensor tmp_val;
    COLUMN_RETURN_NOT_OK(Read<std::string>(key, tmp_val));
    *val = tmp_val.Raw<std::string>()[0];
    return Status::OK();
  }
  IteratorStateReaderImpl(IteratorStateReaderImpl &) = delete;
  IteratorStateReaderImpl &operator=(IteratorStateReaderImpl &) = delete;

private:
  template <typename T> Status Read(const std::string &key, Tensor &tensor) {
    if (states_map_->count(key) == 0) {
      return Status::Internal("not find key [", key, "]");
    }
    tensor = (*states_map_).at(key);
    return Status::OK();
  }
  const std::unordered_map<std::string, Tensor> *states_map_;
};

Status EncodeStates(const std::unordered_map<std::string, Tensor> states,
                    std::string *msg) {
  TensorDataProto proto;
  for (auto &&it : states) {
    auto tensor_data = proto.add_tensor_data();
    tensor_data->mutable_name()->assign(it.first);
    COLUMN_RETURN_NOT_OK(Encode(it.second, tensor_data->mutable_proto()));
  }
  if (!proto.SerializeToString(msg)) {
    return Status::Internal("Encode iterator state failed");
  }
  return Status::OK();
}

Status DecodeStates(const std::string &msg,
                    std::unordered_map<std::string, Tensor> &states) {
  TensorDataProto proto;
  if (!proto.ParseFromString(msg)) {
    return Status::Internal("Decode iterator state failed!");
  }
  for (auto it = proto.tensor_data().begin(); it != proto.tensor_data().end();
       it++) {
    if (states.count(it->name()) != 0) {
      return Status::InvalidArgument("Duplicate key [", it->name(), "]");
    }
    Tensor tensor;
    COLUMN_RETURN_NOT_OK(DecodeAndMake(it->proto(), &tensor));
    states[it->name()] = tensor;
  }
  return Status::OK();
}

} // namespace

Status TensorDataset::Make(const std::string &states,
                           std::unique_ptr<TensorDataset> *tensor_dataset) {
  tensor_dataset->reset(new TensorDataset);
  return DecodeStates(states, (*tensor_dataset)->states_map_);
}

std::unique_ptr<TensorDataset> TensorDataset::Make() {
  return std::unique_ptr<TensorDataset>(new TensorDataset);
}

IteratorStateWriter *TensorDataset::GetWriter() { return writer_.get(); }

IteratorStateReader *TensorDataset::GetReader() { return reader_.get(); }

Status TensorDataset::GetStates(std::string *states) {
  return EncodeStates(states_map_, states);
}

TensorDataset::TensorDataset() {
  reader_.reset(new IteratorStateReaderImpl(&states_map_));
  writer_.reset(new IteratorStateWriterImpl(&states_map_));
}

Status SerializeIteraterToString(std::shared_ptr<IteratorBase> iterator,
                                 std::string *out) {
  auto dataset = TensorDataset::Make();
  COLUMN_RETURN_NOT_OK(iterator->Save(dataset->GetWriter()));
  return dataset->GetStates(out);
}

Status DeserializeIteratorFromString(std::shared_ptr<IteratorBase> iterator,
                                     const std::string &msg) {
  std::unique_ptr<TensorDataset> dataset;
  COLUMN_RETURN_NOT_OK(TensorDataset::Make(msg, &dataset));
  return iterator->Restore(dataset->GetReader());
}
} // namespace dataset
} // namespace column
