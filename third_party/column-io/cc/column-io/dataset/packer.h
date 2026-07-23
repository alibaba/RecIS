#ifndef _COMMON_IO_COLUMN_DATASET_PACKER_H_
#define _COMMON_IO_COLUMN_DATASET_PACKER_H_
#include <cstddef>
#include <functional>
#include <memory>

#include "arrow/record_batch.h"
#include "column-io/dataset/dataset.h"
#include "column-io/framework/status.h"
#include "column-io/framework/thread_pool.h"
#include "column-io/framework/types.h"
namespace column {
namespace dataset {
class Packer {
public:

  Status Run(std::vector<std::vector<Tensor>> &output_elements,
             std::vector<Tensor> batch_element, 
             std::vector<int> the_group_indexes,
             bool is_end); 

  Packer(const std::vector<int> &pack_tables, const std::vector<int> &ragged_ranks, 
          const std::vector<int> &splits_sub_idx, int num_tables, framework::StdThreadPool *pool, int batch_size,
          int group_id, Allocator* allocator);

  void SetBatchSize(int batch_size) {
    batch_size_ = batch_size;
  }

  static std::shared_ptr<DatasetBase>
  MakeReorderDataset(const std::shared_ptr<DatasetBase> input,
                     const std::vector<int64> &new_order);
  static std::shared_ptr<DatasetBase>
  MakeDataset(const std::shared_ptr<DatasetBase> &input, size_t batch_size,
              bool drop_remainder, const std::vector<int> &pack_tables,
              int num_tables, const std::vector<int> &ragged_ranks,
              int64 parallel, bool pinned_result, bool gpu_result, bool do_classify = false);

private:
  Status Run(std::vector<Tensor> &out_tensors,
             std::vector<std::vector<Tensor>> &batch_element_vec, 
             std::vector<std::vector<std::vector<int>>> &all_table_ranges,
             std::vector<int> &batch_sample_cnt, int sample_cnt);

protected:
  const int num_tables_;
  const int group_id_;
  int batch_size_;
  const std::vector<int> pack_tables_;
  const std::vector<int> ragged_ranks_;
  const std::vector<int> splits_sub_idx_;
  std::deque<std::vector<Tensor>> input_batch_deque_; //queue<batchs>
  std::deque<std::vector<std::vector<int>>> input_batch_table_ranges_;  //queue<batchs<tables<ids>>>
  std::deque<int> input_batch_sample_count_; //queue<sample cnt belong this group in input>
  int total_size_;
  int batch_size_idx_, sample_group_size_idx_, sample_group_id_idx_;
  int indicators_idx_, values_idx_, splits_idx_;
  framework::StdThreadPool *pool_;
  Allocator* allocator_;
  int num_tensors_;
  int num_splits_;
};
} // namespace dataset
} // namespace column
#endif
