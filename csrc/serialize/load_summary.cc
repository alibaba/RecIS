#include "serialize/load_summary.h"

#include <string>

#include "ATen/core/Dict.h"
#include "ATen/core/List.h"
#include "serialize/name.h"
namespace recis {
namespace serialize {
void LoadSummary::InitFromLoadInfo(const LoadInfo &load_info) {
  for (auto &load_entry : load_info.Infos()) {
    auto &dst_variable_name = load_entry.first;
    // for dense variable
    if (load_entry.second.begin()->second.empty()) {
      AddVariableToLoad(dst_variable_name);
    } else {
      // for sparse variable
      for (auto &load_entry_it : load_entry.second) {
        for (auto &slot_name : load_entry_it.second) {
          AddVariableToLoad(HTSlotNameEncode(dst_variable_name, slot_name));
        }
      }
    }
  }
}
void LoadSummary::AddVariableToLoad(const std::string &variable_name) {
  variables_to_load_.emplace(variable_name);
}
void LoadSummary::MarkVariableLoaded(const std::string &dst_variable_name,
                                     const std::string &src_variable_name) {
  variables_to_load_.erase(dst_variable_name);
  variable_load_map_[dst_variable_name] = src_variable_name;
}
torch::List<std::string> LoadSummary::VariablesToLoad() const {
  torch::List<std::string> ret;
  for (const auto &variable_name : variables_to_load_) {
    ret.push_back(variable_name);
  }
  return ret;
}
torch::Dict<std::string, std::string> LoadSummary::VariablesLoadMap() const {
  torch::Dict<std::string, std::string> ret;
  for (const auto &item : variable_load_map_) {
    ret.insert(item.first, item.second);
  }
  return ret;
}
}  // namespace serialize
}  // namespace recis