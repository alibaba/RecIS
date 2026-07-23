// #if (_GLIBCXX_USE_CXX11_ABI != 1)

#include "column-io/dataset/list_combo_dataset.h"
#include "absl/strings/str_cat.h"
#include "column-io/dataset/vec_tensor_converter.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include <cstddef>
#include <memory>
#include <mutex>
#include <stddef.h>
#include <string>
#include <vector>
namespace column {
namespace dataset {
namespace {
static const std::string kDatasetName = "ListComboDataset";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, std::vector<Tensor> inputs)
      : DatasetBase(name), inputs_(inputs) {}

protected:
  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::shared_ptr<IteratorBase>(
        new Iterator(std::dynamic_pointer_cast<Dataset>(shared_from_this()),
                     absl::StrCat(prefix, name())));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    Iterator(const std::shared_ptr<Dataset> dataset, const std::string &prefix)
        : DatasetIterator<Dataset>({dataset, prefix}), index_(0) {}
    Status Initialize() override { return Status::OK(); }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      std::lock_guard<std::mutex> lock(mu_);
      RETURN_IF_ERROR(writer->WriteInt(fullname("index"), index_));
      return Status::OK();
    }
    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> lock(mu_);
      RETURN_IF_ERROR(reader->ReadInt(fullname("index"), index_));
      return Status::OK();
    }
    Status GetNextInternal(std::vector<Tensor> *outputs,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (dataset()->inputs_.size() <= 0){
            return Status::InvalidArgument("list_combo_dataset inputs buffer is empty !");
        }
        if (index_ >= dataset()->inputs_.size()) {
            *end_of_sequence = true;
            return Status::OutOfRange();
        }
        *end_of_sequence = false;
        outputs->clear();
        // outputs->emplace_back(Tensor(dataset()->inputs_[index_].Type()));
        switch (dataset()->inputs_[index_].Type()) {
        case kString: {
            const auto& current_tensor = dataset()->inputs_[index_];
            outputs->emplace_back(current_tensor);
            break;
        }
        default:
            return Status::InvalidArgument("type [", dataset()->inputs_[index_].Type(),
                                        "] is not supported in plain converter");
        }

        index_++;
        return Status::OK();
    }

  private:
    int64_t index_;
    std::mutex mu_;
  };
  std::vector<Tensor> inputs_;
};

} // namespace
std::shared_ptr<DatasetBase>
ListStringComboDataset::MakeDataset(const std::vector<std::vector<std::string>> &inputs) {
  return std::make_shared<Dataset>(kDatasetName,
                                   detail::VecsToTensor<std::string>(inputs));
}
} // namespace dataset
} // namespace column

// #endif
