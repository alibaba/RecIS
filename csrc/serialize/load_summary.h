#pragma once
#include <serialize/load_info.h>
#include <torch/extension.h>

#include <map>
#include <set>
#include <string>

#include "ATen/core/Dict.h"
#include "ATen/core/List.h"
#include "ATen/core/ivalue.h"
namespace recis {
namespace serialize {
/**
 *@brief LoadSummary
 *  This class has the following three functionalities:
 *  1. record variables which is not loaded
 *  2. record variables which is loaded, and the corresponding source variable
 *name
 *  3. return the above information to python for more friendly notification.
 */
class LoadSummary : public torch::CustomClassHolder {
 public:
  void InitFromLoadInfo(const LoadInfo &load_info);
  /**
   * @brief Adds a variable to the set of variables pending loading.
   *
   * The format of @p variable_name depends on the variable type:
   *
   * - For **dense variables**: a simple variable name (e.g., @c
   * "learning_rate", @c "global_step").
   * - For **sparse variables**: a composite identifier in the format
   *   <tt>shared_name&lt;SPLIT_SYMBOL&gt;slot_name</tt>, where:
   *   - <b>shared_name</b> is the common name shared across all slots of the
   * sparse variable (e.g., "embedding_table");
   *   - <b>slot_name</b> identifies a specific slot or partition (e.g., "id",
   * "embedding");
   *   - <b>SPLIT_SYMBOL</b> is a fixed delimiter (currently TensorSymbolAt(),
   * TensorSymbolDot()).
   *
   *
   * @param variable_name The identifier of the variable to be loaded.
   *        Must follow the appropriate format based on variable type.
   *
   * @note
   * - This function is idempotent: adding the same @p variable_name multiple
   * times has no additional effect (the underlying container is a set).
   */
  void AddVariableToLoad(const std::string &variable_name);
  /**
   * @brief Marks the destination variable as successfully loaded from the
   * source variable.
   *
   * This function signals that the variable identified by @p dst_variable_name
   * has been loaded using data from the source identified by @p
   * src_variable_name. It is typically used during checkpoint restoration when
   * variable names differ between the current model and the saved checkpoint
   * (e.g., due to renaming, versioning, or aliasing).
   *
   * Both @p dst_variable_name and @p src_variable_name must adhere to the
   * following format, depending on the variable type.(see AddVariableToLoad())
   *
   * @param dst_variable_name The target variable name in the current model
   * context. Must follow the dense or sparse naming convention as appropriate.
   *
   * @param src_variable_name The source variable name as stored in the
   * checkpoint or data source. Must also follow the same naming convention.
   *
   * @note
   * - After this call, @p dst_variable_name is removed from the pending load
   * set (i.e., it will no longer appear in @c VariablesToLoad()).
   * - The function assumes @p dst_variable_name was previously added via @c
   * AddVariableToLoad(). Calling this on an unregistered variable results in
   * undefined behavior.
   */
  void MarkVariableLoaded(const std::string &dst_variable_name,
                          const std::string &src_variable_name);
  torch::List<std::string> VariablesToLoad() const;
  torch::Dict<std::string, std::string> VariablesLoadMap() const;

 private:
  std::set<std::string> variables_to_load_;
  std::map<std::string, std::string> variable_load_map_;
};
}  // namespace serialize
}  // namespace recis