#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_MAP_DATASET_SAMPLE_FILTER_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_MAP_DATASET_SAMPLE_FILTER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "column-io/dataset/dataset.h"

namespace column {
namespace dataset {

// MapDataSetSampleFilter
//
// Row-level denylist filter keyed on the upstream `sample_id` string column.
//
// `sample_id` is expected to have the form
//   "<prefix>\x01<k1>:<v1>,<k2>:<v2>,...,<kN>:<vN>"
// (see dingtalk doc YMyQA2dXW7gYo6Mzc5dYqM0oWzlwrZgb for the producer
// contract). For each row we parse sample_id into a kv-map; if ANY key in
// `filter_dict` has its parsed value listed in `filter_dict[key]`, the row is
// dropped (OR-across-keys denylist semantics).
//
// This class does not physically drop rows by itself. It only injects a new
// `_sample_group_id` int64 column where -1 marks "drop" and 0 marks "keep".
// The actual row drop + ragged/indicator rewrite happens in the downstream
// Packer's `GetNextForGroup` path (`packer.cc:459-470`), which is auto-
// enabled when the schema contains `_sample_group_id`
// (see `column_io/dataset/dataset.py:670-672`). Therefore the user MUST chain
// `.pack(...)` after `.map(name="sample_filter", ...)`; otherwise the filter
// is a no-op and `_sample_group_id` simply passes through downstream.
class MapDataSetSampleFilter {
public:
  // Args:
  //   input             — upstream dataset; its schema must contain the
  //                       "sample_id" column.
  //   new_input_schema  — output schema (input + injected _sample_group_id).
  //   old_input_schema  — original input schema (used to locate sample_id
  //                       column position via .at("sample_id")).
  //   filter_dict       — { key: [denylist values] }. Empty map = no-op.
  //
  // Returns: a DatasetBase whose iterator emits the input tensors plus an
  // additional _sample_group_id int64 tensor at the position dictated by
  // new_input_schema.
  static std::shared_ptr<DatasetBase> MakeDataSet(
      const std::shared_ptr<DatasetBase> &input,
      const std::vector<
          std::map<std::string, std::vector<std::vector<int64_t>>>>
          &new_input_schema,
      const std::vector<
          std::map<std::string, std::vector<std::vector<int64_t>>>>
          &old_input_schema,
      const std::map<std::string, std::vector<std::string>> &filter_dict);
};

} // namespace dataset
} // namespace column

#endif // COLUMN_IO_CC_COLUMN_IO_DATASET_MAP_DATASET_SAMPLE_FILTER_H_
