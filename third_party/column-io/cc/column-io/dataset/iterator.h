#ifndef _COLUMN_IO_CC_COLUMN_IO_DATASET_ITERATOR_H
#define _COLUMN_IO_CC_COLUMN_IO_DATASET_ITERATOR_H

#include <memory>
#include <string>

// #include "absl/container/flat_hash_map.h"
#include "column-io/dataset/dataset.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"

namespace column {
namespace dataset {
class TensorDataset {
public:
  // for restore
  static Status Make(const std::string &states,
                     std::unique_ptr<TensorDataset> *tensor_dataset);
  // for write
  static std::unique_ptr<TensorDataset> Make();

  IteratorStateWriter *GetWriter();
  IteratorStateReader *GetReader();
  Status GetStates(std::string *states);

private:
  TensorDataset();
  std::unique_ptr<IteratorStateReader> reader_;
  std::unique_ptr<IteratorStateWriter> writer_;
  std::unordered_map<std::string, Tensor> states_map_;
};

Status SerializeIteraterToString(std::shared_ptr<IteratorBase> iterator,
                                 std::string *out);
Status DeserializeIteratorFromString(std::shared_ptr<IteratorBase> iterator,
                                     const std::string &out);

} // namespace dataset
} // namespace column
#endif