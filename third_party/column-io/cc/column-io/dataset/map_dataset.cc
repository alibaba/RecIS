#include "column-io/dataset/map_dataset.h"
#include "column-io/plugin/aoti_loader.h"

namespace column {
namespace dataset {
const std::string kDatasetName = "MapDataSet";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, 
          const std::shared_ptr<DatasetBase> &input,
          const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>& new_input_schema,
          const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>& old_input_schema,
          const std::vector<std::string>& user_module_columns,
	  const std::string& module_so_path)
      : DatasetBase(name), input_(input), new_input_schema_(new_input_schema), old_input_schema_(old_input_schema), user_module_columns_(user_module_columns), module_so_path_(module_so_path) {
    const char* klib_plugin_so = getenv("LIB_PLUGIN_SO");    
    if (klib_plugin_so != nullptr) {
      plugin_so_path_ = klib_plugin_so;
    } else {
      plugin_so_path_ = "";
    }
  };
  std::shared_ptr<IteratorBase> MakeIteratorInternal(const std::string &prefix) override {
    return std::make_shared<Iterator>(
        std::dynamic_pointer_cast<Dataset>(shared_from_this()),
        absl::StrCat(prefix, "::", name()));
  }	

  const std::string& GetModuleSoPath() {
    return module_so_path_; 
  }
  const std::string GetPluginSoPath() {
    return plugin_so_path_; 
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

      //fill output tensors
      out_tensors->clear();
      out_tensors->resize(tmp_output.size());
      for (size_t i = 0; i < tmp_output.size(); ++i) {
        (*out_tensors)[i] = tmp_output[i];
      }
      std::vector<Tensor> selected_user_module_tensors;
      bool ret = SelectUserModuleColumns(out_tensors, selected_user_module_tensors);
      if (!ret) {
        return Status(INVALID_ARGUMENT, "input selected user module columns is not correct");
      }
      
      //execute aot runner
      if (!this->dataset()->GetModuleSoPath().empty()) {
        if (!column::plugin::AOTILoader::Load(this->dataset()->GetPluginSoPath())) {
          return Status(INVALID_ARGUMENT, "Failed to load AOTI plugin");
        }
        AOTIExecutorImpl* executor = column::plugin::AOTILoader::CreateExecutor(this->dataset()->GetModuleSoPath().c_str());
        TensorArrayImpl* inputs = column::plugin::AOTILoader::CreateTensorArray();
        for (size_t i = 0; i < selected_user_module_tensors.size(); ++i) {
          column::plugin::AOTILoader::TensorArrayPushBack(inputs, &selected_user_module_tensors[i]);
        }
        TensorArrayImpl* outputs = nullptr;
        int aot_run_ret = column::plugin::AOTILoader::ExecutorRun(executor, inputs, &outputs);
        if (aot_run_ret != 0) {
          return Status(INVALID_ARGUMENT, "aoti_executor_run failed");
        }
        if (column::plugin::AOTILoader::TensorArraySize(outputs) != size_t(1)) {
          return Status(UNAVAILABLE, "aot run output vector tensor size is not 1");
        } 
        int32_t group_id_t_pos = this->dataset()->new_input_schema_[0]["_sample_group_id"][0][0];
        const column::Tensor* group_id_t = column::plugin::AOTILoader::TensorArrayGet(outputs, 0);
        out_tensors->insert(out_tensors->begin() + group_id_t_pos, *group_id_t);
        column::plugin::AOTILoader::DestroyTensorArray(inputs);
        column::plugin::AOTILoader::DestroyTensorArray(outputs);
        column::plugin::AOTILoader::DestroyExecutor(executor);
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
    bool SelectUserModuleColumns(const std::vector<Tensor>* output_tensors, std::vector<Tensor>& selected_user_module_tensors) {
      auto& old_schema = this->dataset()->old_input_schema_;
      auto& new_schema = this->dataset()->new_input_schema_;
      for (const auto& column_name : this->dataset()->user_module_columns_) {
        if (old_schema[0].find(column_name) != old_schema[0].end()) {
          for (const auto& pos_vec : old_schema[0][column_name]) {
            for (int64_t pos : pos_vec) {
              selected_user_module_tensors.emplace_back((*output_tensors)[pos]);
            }
          }
        } else if (old_schema[1].find(column_name) != old_schema[1].end()) {
          for (const auto& pos_vec : old_schema[1][column_name]) {
            for (int64_t pos : pos_vec) {
              selected_user_module_tensors.emplace_back((*output_tensors)[pos]);
            }
          }
        } else {
          LOG(ERROR) << "user module column name [" << column_name << "] is not found in schema map";
          return false;
        }
      }
      return true;
    }

    std::mutex mu_;
    std::shared_ptr<IteratorBase> input_impl_;
  };

  std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>> new_input_schema_;
  std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>> old_input_schema_;
  std::vector<std::string> user_module_columns_;
  std::string module_so_path_;
  std::string plugin_so_path_;
  std::shared_ptr<DatasetBase> input_; 
};

std::shared_ptr<dataset::DatasetBase> 
MapDataSet::MakeDataSet(
    const std::shared_ptr<DatasetBase>& input,
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>& new_input_schema,
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>& old_input_schema,
    const std::vector<std::string>& user_module_columns,
    const std::string& module_so_path) {
  return std::make_shared<Dataset>(kDatasetName, input, new_input_schema, old_input_schema, user_module_columns, module_so_path);
}

} // namespace dataset
} // namespace column

