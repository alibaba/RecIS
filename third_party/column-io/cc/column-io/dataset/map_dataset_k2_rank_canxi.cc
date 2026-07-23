#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "column-io/dataset/map_dataset_k2_rank_canxi.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include <map>
#include <string>
#include <vector>

namespace column {
namespace dataset {
namespace {
const std::string kDatasetName = "MapDataSetRankCanxi";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, const std::shared_ptr<DatasetBase> &input,
          const std::vector<
              std::map<std::string, std::vector<std::vector<int64_t>>>>
              &new_input_schema,
          const std::vector<
              std::map<std::string, std::vector<std::vector<int64_t>>>>
              &old_input_schema,
          const bool &odl_mode)
      : DatasetBase(name), input_(input), new_input_schema_(new_input_schema),
        old_input_schema_(old_input_schema), odl_mode_(odl_mode){};
  std::shared_ptr<IteratorBase> MakeIteratorInternal(const std::string &prefix) override {
    return std::make_shared<Iterator>(
        std::dynamic_pointer_cast<Dataset>(shared_from_this()),
        absl::StrCat(prefix, "::", name()));
  }	
private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    Iterator(const std::shared_ptr<Dataset> dataset, const std::string &prefix)
        : DatasetIterator<Dataset>({dataset, prefix}) {}

    Status Initialize() override {
      return this->dataset()->input_->MakeIterator(prefix(), &input_impl_);
    }

    Status GetNextInternal(std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) override {
      std::lock_guard<std::mutex> l(mu_);
      if (!input_impl_) {
        *end_of_sequence = true;
        return Status::OK();
      }
      std::vector<Tensor> tmp_output;
      auto st = input_impl_->GetNext(&tmp_output, end_of_sequence);
      if (*end_of_sequence) {
        return Status::OK();
      }
      if (!st.ok()) {
        return st;
      }

      int64_t sample_id_pos =
          this->dataset()->old_input_schema_[0].at("sample_id")[0][0];
      const Tensor sample_id = tmp_output[sample_id_pos];
      Tensor sample_group_id = Tensor(kInt64, sample_id.Shape());

      MapInternal(sample_id, sample_group_id, this->dataset()->odl_mode_);

      int32_t group_id_t_pos = this->dataset()->new_input_schema_[0].at("_sample_group_id")[0][0];

      out_tensors->clear();
      out_tensors->resize(tmp_output.size());
      for (size_t i = 0; i < tmp_output.size(); ++i) {
        (*out_tensors)[i] = tmp_output[i];
      }
      
      out_tensors->insert(out_tensors->begin() + group_id_t_pos, sample_group_id);
      return Status::OK();
    }
  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      std::lock_guard<std::mutex> l(mu_);
      if (input_impl_) {
        RETURN_IF_ERROR(writer->WriteString(fullname("input"), ""));
        RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> l(mu_);
      if (reader->Contain(fullname("input"))) {
        RETURN_IF_ERROR(RestoreInput(reader, input_impl_));
      }
      return Status::OK();
    } 

  private:
    /*
    def test_filter_fn(*args):
    structure = tuple(args)
    sample_id = structure[0]["sample_id"][0][0]
    group_idx = to_stag_filter_nitem(sample_id, mode='test')
    structure[0]['_sample_group_id'] = ((group_idx,),)
    return structure
    */
    void MapInternal(Tensor sample_id, Tensor sample_group_id, bool odl_mode) {
      int64_t item_num = sample_id.NumElements();
      std::string *sample_id_vec = sample_id.Raw<std::string>();
      int64_t *sample_group_id_vec = sample_group_id.Raw<int64_t>();
      for (int i = 0; i < item_num; ++i) {
        int64_t group_idx = IsEntityItem(sample_id_vec[i], odl_mode);
        sample_group_id_vec[i] = group_idx;
      }
    }

    /*
    def is_entity_item(string):
        subType = string.split('subProductType:' if xdl.get_config("task_mode")=='odl' else 'sub_product_type:')[1].split(',')[0].strip()
        if subType.endswith("1^11") or subType.endswith("2^21") or subType.endswith("1^11^1"):
            return True
        else:
            return False
    */
    int64_t IsEntityItem(const std::string &sample_id, bool odl_mode) {
      std::string delimiter =
          odl_mode ? "subProductType:" : "sub_product_type:";

      std::vector<std::string> tokens =
          absl::StrSplit(sample_id, absl::ByString(delimiter));
      if (tokens.size() < 2) {
        return -1;
      }
      std::pair<absl::string_view, absl::string_view> sub_parts =
          absl::StrSplit(tokens[1], absl::MaxSplits(',', 1));
      absl::string_view sub_type = absl::StripAsciiWhitespace(sub_parts.first);
      if (absl::EndsWith(sub_type, "1^11") ||
          absl::EndsWith(sub_type, "2^21") ||
          absl::EndsWith(sub_type, "1^11^1")) {
        return 0;
      } else {
        return -1;
      }
    }

    std::mutex mu_;
    std::shared_ptr<IteratorBase> input_impl_;
  };

  std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>> new_input_schema_;
  std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>> old_input_schema_;
  std::shared_ptr<DatasetBase> input_; 
  const bool odl_mode_;
};
}

std::shared_ptr<dataset::DatasetBase> MapDataSetK2RankCanXi::MakeDataSet(
    const std::shared_ptr<DatasetBase> &input,
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>
        &new_input_schema,
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>
        &old_input_schema,
    const bool odl_mode) {
  return std::make_shared<Dataset>(kDatasetName, input, new_input_schema,
                                   old_input_schema, odl_mode);
}

} // namespace dataset
} // namespace column