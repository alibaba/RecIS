#include "column-io/dataset/map_dataset_k2_rank.h"

#include <exception>
#include <map>
#include <string>
#include <vector>
// #include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"

#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"

namespace column {
namespace dataset {
namespace {
const std::string kDatasetName = "MapDataSetRank";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, const std::shared_ptr<DatasetBase> &input,
          const std::vector<
              std::map<std::string, std::vector<std::vector<int64_t>>>>
              &new_input_schema,
          const std::vector<
              std::map<std::string, std::vector<std::vector<int64_t>>>>
              &old_input_schema,
          const std::map<std::string, int32_t> &scene_map)
      : DatasetBase(name), input_(input), new_input_schema_(new_input_schema),
        old_input_schema_(old_input_schema) {
    for (auto &it : scene_map) {
      scene_map_[it.first] = it.second;
    }
  };
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
      Tensor scene_flag = Tensor(kInt32, sample_id.Shape());
      MapInternal(sample_id, sample_group_id, scene_flag);

      int32_t group_id_t_pos = this->dataset()->new_input_schema_[0].at("_sample_group_id")[0][0];
      int32_t scene_flag_pos = this->dataset()->new_input_schema_[0].at("scene_flag")[0][0];

      out_tensors->clear();
      out_tensors->resize(tmp_output.size());
      for (size_t i = 0; i < tmp_output.size(); ++i) {
        (*out_tensors)[i] = tmp_output[i];
      }
      if (group_id_t_pos < scene_flag_pos) {
        out_tensors->insert(out_tensors->begin() + group_id_t_pos, sample_group_id);
        out_tensors->insert(out_tensors->begin() + scene_flag_pos, scene_flag);
      } else {
        out_tensors->insert(out_tensors->begin() + scene_flag_pos, scene_flag);
        out_tensors->insert(out_tensors->begin() + group_id_t_pos, sample_group_id);
      }
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
    def filter_fn(*args):
    structure = tuple(args)
    sample_id = structure[0]["sample_id"][0][0]
    scene_flag = to_scene_flag_filter(sample_id)

    group_id = tf.clip_by_value(scene_flag, -1, 0)
    structure[0]["_sample_group_id"] = ((group_id,),)
    structure[0]["scene_flag"] = ((scene_flag,),)

    return structure 
    */
	void
    MapInternal(Tensor sample_id, 
		Tensor sample_group_id,
                Tensor scene_flag) {
      int64_t item_num = sample_id.NumElements();
      std::string *sample_id_vec = sample_id.Raw<std::string>();
      int32_t *scene_flag_vec = scene_flag.Raw<int32_t>();
      int64_t *sample_group_id_vec = sample_group_id.Raw<int64_t>();
      for (int i = 0; i < item_num; ++i) {
        int32_t scene_flag =
            SceneFlagParse(sample_id_vec[i], this->dataset()->scene_map_);
        scene_flag_vec[i] = scene_flag;
        scene_flag = scene_flag < -1 ? -1 : scene_flag;
        scene_flag = scene_flag > 0 ? 0 : scene_flag;
        sample_group_id_vec[i] = scene_flag;
      }
    }

    /*
    def scene_flag_filter(string):
    try:
        output = string.strip().split("\001")[1].split(",")
        config = {x.split(":")[0]: x.split(":")[1] for x in output}
        pid = config.get("pid", "0")
        # ocpx_price = config.get("mid_select_ocpx_price", "0")
        scene_flag = scene_map.get(pid, -1)
    except:
        scene_flag = -1
    return scene_flag
    */
    int32_t
    SceneFlagParse(const std::string &sample_id,
                   const std::unordered_map<std::string, int32_t> &scene_map) {
      int32_t scene_flag = -1;
      try {
        auto strip_sample_id = absl::StripAsciiWhitespace(sample_id);
        std::vector<std::string> tokens =
            absl::StrSplit(strip_sample_id, '\001');
        if (tokens.size() < 2) {
          return scene_flag;
        }
        std::vector<std::string> config_tokens = absl::StrSplit(tokens[1], ',');
        for (auto &token : config_tokens) {
          std::vector<std::string> keys = absl::StrSplit(token, ':');
          if (keys.size() != 2) {
            continue;
          }
          if (keys[0] == "pid") {
            scene_flag = scene_map.at(keys[1]);
            break;
          }
        }
      } catch (std::exception &e) {
        ;
      }
      return scene_flag;
    }

    std::mutex mu_;
    std::shared_ptr<IteratorBase> input_impl_;
  };

  std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>> new_input_schema_;
  std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>> old_input_schema_;
  std::shared_ptr<DatasetBase> input_; 
  std::unordered_map<std::string, int32_t> scene_map_;
};
}

std::shared_ptr<dataset::DatasetBase> 
MapDataSetK2Rank::MakeDataSet(
    const std::shared_ptr<DatasetBase>& input,
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>& new_input_schema,
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>& old_input_schema,
    const std::map<std::string, int32_t> &scene_map) {
  return std::make_shared<Dataset>(kDatasetName, input, new_input_schema, old_input_schema, scene_map);
}

} // namespace dataset
} // namespace column
