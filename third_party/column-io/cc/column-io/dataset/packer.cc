#include "column-io/dataset/packer.h"
#include "absl/log/log.h"
#include "absl/synchronization/blocking_counter.h"
#include "column-io/dataset/dataset.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include "column-io/framework/cuda_utils.h"
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <map>
#include <unordered_set>
#include <type_traits>
#include <unistd.h>

namespace column {
namespace dataset {
// The macro CASES() expands to a switch statement conditioned on
// TYPE_ENUM. Each case expands the STMTS after a typedef for T.
#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)                                                      \
  case DataTypeToEnum<TYPE>::value: {                                          \
    typedef TYPE T;                                                            \
    STMTS;                                                                     \
    break;                                                                     \
  }
#define CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, INVALID, DEFAULT)                 \
  switch (TYPE_ENUM) {                                                         \
    CASE(float, SINGLE_ARG(STMTS))                                             \
    CASE(double, SINGLE_ARG(STMTS))                                            \
    CASE(int32_t, SINGLE_ARG(STMTS))                                           \
    CASE(uint8_t, SINGLE_ARG(STMTS))                                           \
    CASE(uint16_t, SINGLE_ARG(STMTS))                                          \
    CASE(uint32_t, SINGLE_ARG(STMTS))                                          \
    CASE(uint64_t, SINGLE_ARG(STMTS))                                          \
    CASE(int16_t, SINGLE_ARG(STMTS))                                           \
    CASE(int8_t, SINGLE_ARG(STMTS))                                            \
    CASE(std::string, SINGLE_ARG(STMTS))                                       \
    CASE(int64_t, SINGLE_ARG(STMTS))                                           \
    CASE(bool, SINGLE_ARG(STMTS))                                              \
  default:                                                                     \
    DEFAULT;                                                                   \
    break;                                                                     \
  }

#define CASES(TYPE_ENUM, STMTS)                                                \
  CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, LOG(FATAL) << "Type not set";           \
                  , LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)


namespace {
const std::string &kDatasetName("PackerDataset");
template <typename T> struct is_simple_type {
  static constexpr bool value = std::is_trivial<T>::value;
};
const int kPackTensorBlock = 16;
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, size_t batch_size, bool drop_remainder,
          const std::vector<int> &pack_tables, int num_tables,
          const std::vector<int> &ragged_ranks,
          const std::shared_ptr<DatasetBase> input, int64 parallel,
          bool pinned_result, bool gpu_result, bool do_classify = false)
      : DatasetBase(name), batch_size_(batch_size),
        drop_remainder_(drop_remainder), pack_tables_(pack_tables),
        ragged_ranks_(ragged_ranks), input_(input), num_tables_(num_tables),
        splits_sub_idx_(ragged_ranks.size()),
        pool_(new framework::StdThreadPool("PackerDatasetPool", parallel)),
        pinned_result_(pinned_result), gpu_result_(gpu_result), do_classify_(do_classify) {
    for (int i = 1; i < ragged_ranks.size(); ++i) {
      splits_sub_idx_[i] = splits_sub_idx_[i - 1] + ragged_ranks[i - 1];
    }
#ifndef CPU_ONLY
    if (gpu_result_) {
      device_id_ = GetCudaDeviceId();
      GPU_CK(cudaSetDevice(device_id_));
      GPU_CK(cudaStreamCreate(&stream_));
    }
#endif
  }

  int32_t get_batch_size(std::unique_ptr<std::vector<Tensor>> &batch) {
    return get_batch_size(*(batch.get()));
  }
  int32_t get_batch_size(const std::vector<Tensor> &batch, bool do_classify = false) {
    // take indicator size as batch size.
    if (num_tables_ > 1) {
      return batch[0].NumElements();
    }
    // take first indice tensor size as batch size.
    if (ragged_ranks_[0] == 0) {
      return batch[0].Shape()[0];
    } else {
      int split_index = 0;
      if (do_classify) {
        split_index = ragged_ranks_.size() + ragged_ranks_[0];
      } else {
        split_index = ragged_ranks_.size() + ragged_ranks_[0] - 1;
      }
      return batch[split_index].Shape()[0] - 1;
    }
  }
  ~Dataset() override {
#ifndef CPU_ONLY
    if (gpu_result_) {
      GPU_CK(cudaStreamSynchronize(stream_));
      GPU_CK(cudaStreamDestroy(stream_));
    }
#endif
  }

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::unique_ptr<IteratorBase>(
        new Iterator(std::dynamic_pointer_cast<Dataset>(shared_from_this()),
                     absl::StrCat(prefix, "::PackV2"), batch_size_));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    explicit Iterator(const std::shared_ptr<Dataset> dataset,
                      const std::string &prefix, size_t batch_size)
        : DatasetIterator<Dataset>({dataset, prefix}), batch_size_(batch_size) {
          allocator_ = GetAllocator(this->dataset()->pinned_result_ || this->dataset()->gpu_result_);
          if (this->dataset()->gpu_result_) {
            cuda_allocator_ = 
                CreateCudaAllocator(this->dataset()->stream_, this->dataset()->device_id_);
          }
    }

    ~Iterator() {
      if (cuda_allocator_) {
        cuda_allocator_->Unref();
      }
    }

    Status Initialize() override {
      return this->dataset()->input_->MakeIterator(this->prefix(),
                                                   &input_impl_);
    }

    size_t get_batch_size() { return batch_size_; }

    Status GetNextInternal(std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) override {
      if (this->dataset()->do_classify_) return GetNextForGroup(out_tensors, end_of_sequence);
      int num_tensors = this->dataset()->ragged_ranks_.size();
      int num_splits = this->dataset()->splits_sub_idx_.back() +
                       this->dataset()->ragged_ranks_.back();
      int indicators_idx = 0;
      int values_idx = indicators_idx + (this->dataset()->num_tables_ - 1);
      int splits_idx = values_idx + num_tensors;

      std::vector<std::vector<Tensor>> batch_elements;
      int total_size = 0, head_offset = 0;

      std::lock_guard<std::mutex> l(
          mu_); // Consider reduce the granularity mutex ?
      *end_of_sequence = false;

      auto cur_batch_size = get_batch_size();

      if (remain_) {
        // maybe modify cur_batch_size according to data in "remain_"
        head_offset = remain_offset_;
        total_size += dataset()->get_batch_size(remain_) - remain_offset_;
        batch_elements.emplace_back(std::move(*remain_));
        remain_.reset();
      }

      while (total_size < cur_batch_size && !*end_of_sequence && input_impl_) {
        std::vector<Tensor> batch_element;
        RETURN_IF_ERROR(input_impl_->GetNext(&batch_element, end_of_sequence));
        if (!*end_of_sequence) {
          total_size += dataset()->get_batch_size(batch_element);
          batch_elements.emplace_back(std::move(batch_element));
        } else {
          input_impl_.reset();
        }
      }

      if (total_size == 0 ||
          (this->dataset()->drop_remainder_ && total_size < cur_batch_size)) {
        *end_of_sequence = true;
        return Status::OK();
      }

      Tensor batch_size_t(kInt32, {});
      std::vector<Tensor> indicators_t(this->dataset()->num_tables_ - 1);
      std::vector<Tensor> values_t(num_tensors);
      std::vector<Tensor> splits_t(num_splits);

      std::vector<std::vector<std::pair<int, int>>> all_table_ranges;
      all_table_ranges.reserve(batch_elements.size());

      int batch_size = std::min<int64_t>(total_size, cur_batch_size);
      for (int i = 0; i < batch_elements.size(); ++i) {
        std::vector<Tensor> &batch_element = batch_elements[i];

        // std::vector<std::vector<int>> table_ranges;
        std::vector<std::pair<int, int>> table_ranges;
        int begin = 0, end = dataset()->get_batch_size(batch_element);
        if (i == 0) {
          begin = head_offset;
        }
        if (i == batch_elements.size() - 1) {
          end -= total_size - batch_size;
        }
        table_ranges.emplace_back(begin, end);
        for (int j = 0; j < this->dataset()->num_tables_ - 1; ++j) {
          Tensor &indicators_t = batch_element[indicators_idx + j];
          auto indicators = indicators_t.Raw<int64_t>();
          int common_begin = std::numeric_limits<int>::max(),
              common_end = std::numeric_limits<int>::min();
          for (int k = begin; k < end; ++k) {
            int refer = static_cast<int>(indicators[k]);
            common_begin = std::min(common_begin, refer);
            common_end = std::max(common_end, refer + 1);
          }
          table_ranges.emplace_back(common_begin, common_end);
        }
        all_table_ranges.emplace_back(table_ranges);
      }

      auto pack_tensor = [&](int j) {
        int table = this->dataset()->pack_tables_[j];
        int sub_idx = this->dataset()->splits_sub_idx_[j];


        // Do statistic.

        std::vector<int> ragged_size(this->dataset()->ragged_ranks_[j] + 1, 0);
        for (int i = 0; i < batch_elements.size(); ++i) {
          std::vector<Tensor> &batch_element = batch_elements[i];

          int begin, end;
          std::tie(begin, end) = all_table_ranges[i][table];
          for (int k = this->dataset()->ragged_ranks_[j] - 1; k >= 0; --k) {
#define DECLARE_HANDLE_FOR_TYPE(Type)                                          \
  auto handle_##Type = [&]() {                                                 \
    ragged_size[k + 1] += end - begin;                                         \
    auto current_splits = batch_element[splits_idx + sub_idx + k].Raw<Type>(); \
    begin = current_splits[begin];                                             \
    end = current_splits[end];                                                 \
  }
            DECLARE_HANDLE_FOR_TYPE(int32_t);
            DECLARE_HANDLE_FOR_TYPE(int64_t);
#undef DECLARE_HANDLE_FOR_TYPE
            if (batch_element[splits_idx + sub_idx + k].Type() == kInt64) {
              handle_int64_t();
            } else {
              handle_int32_t();
            }
          }
          ragged_size[0] += end - begin;
        }

        // Allocate tensors.
        for (int k = 0; k < this->dataset()->ragged_ranks_[j]; ++k) {
#define DECLARE_HANDLE_FOR_TYPE(Type, TfType)                                  \
  auto handle_##Type = [&]() {                                                 \
    splits_t[sub_idx + k] = Tensor(allocator_, {size_t(ragged_size[k + 1] + 1)}, TfType);  \
    splits_t[sub_idx + k].Raw<Type>()[0] = 0;                                  \
  }
          DECLARE_HANDLE_FOR_TYPE(int32_t, kInt32);
          DECLARE_HANDLE_FOR_TYPE(int64_t, kInt64);
#undef DECLARE_HANDLE_FOR_TYPE
          if (batch_elements[0][splits_idx + sub_idx + k].Type() == kInt64) {
            handle_int64_t();
          } else {
            handle_int32_t();
          }
        }

        DataType dtype = batch_elements[0][values_idx + j].Type();
        TensorShape shape = batch_elements[0][values_idx + j].Shape();
        TensorShape old = shape;
        shape.Set(0, ragged_size[0]);
        values_t[j] = Tensor(allocator_, shape, dtype);
        int dense_dim = 1;
        for (int i = 1; i < shape.Size(); ++i) {
          dense_dim *= shape.Dims()[i];
        }

        // Copy results.

        int next_offset = 0;
        for (int i = 0; i < batch_elements.size(); ++i) {
          std::vector<Tensor> &batch_element = batch_elements[i];

          int begin, end;
          std::tie(begin, end) = all_table_ranges[i][table];
          int offset = next_offset;
          next_offset += end - begin;

          for (int k = this->dataset()->ragged_ranks_[j] - 1; k >= 0; --k) {
#define DECLARE_HANDLE_FOR_TYPE(Type)                                          \
  auto handle_##Type = [&]() {                                                 \
    auto splits = splits_t[sub_idx + k].Raw<Type>();                           \
    auto current_splits = batch_element[splits_idx + sub_idx + k].Raw<Type>(); \
    int splits_offset = splits[offset];                                        \
    int splits_begin = current_splits[begin];                                  \
    int splits_end = current_splits[end];                                      \
    for (int l = 1; l <= end - begin; ++l) {                                   \
      splits[offset + l] =                                                     \
          current_splits[begin + l] + splits_offset - splits_begin;            \
    }                                                                          \
    begin = splits_begin;                                                      \
    end = splits_end;                                                          \
    offset = splits_offset;                                                    \
  }
            DECLARE_HANDLE_FOR_TYPE(int32_t);
            DECLARE_HANDLE_FOR_TYPE(int64_t);
#undef DECLARE_HANDLE_FOR_TYPE
            if (batch_element[splits_idx + sub_idx + k].Type() == kInt64) {
              handle_int64_t();
            } else {
              handle_int32_t();
            }
          }

          CASES(
              dtype, do {
                auto values = values_t[j].Raw<T>();
                auto current_values = batch_element[values_idx + j].Raw<T>();
                if (is_simple_type<T>::value) {
                  std::memcpy(&values[offset * dense_dim],
                              &current_values[begin * dense_dim],
                              sizeof(T) * (end - begin) * dense_dim);
                } else {
                  for (int l = 0; l < (end - begin) * dense_dim; ++l) {
                    values[offset + l] = current_values[begin + l];
                  }
                }
              } while (0));
        }
        return Status::OK();
      };

      absl::BlockingCounter counter((num_tensors - 1) / kPackTensorBlock + 1);
      Status status;
      std::mutex status_mu;
      for (int job_begin = 0; job_begin < num_tensors;
           job_begin += kPackTensorBlock) {
        int job_end = std::min(job_begin + kPackTensorBlock, num_tensors);
        dataset()->pool_->Schedule([job_begin, job_end, &status, &status_mu,
                                    &counter, &pack_tensor]() {
          for (int j = job_begin; j < job_end; ++j) {
            Status s = pack_tensor(j);
            {
              std::lock_guard<std::mutex> l(status_mu);
              status = s;
            }
          }
          counter.DecrementCount();
        });
      }

      batch_size_t.Raw<int32_t>()[0] = 0;
      for (int i = 0; i < batch_elements.size(); ++i) {
        int begin, end;
        std::tie(begin, end) = all_table_ranges[i][0];
        batch_size_t.Raw<int32_t>()[0] += end - begin;
      }

      for (int j = 0; j < this->dataset()->num_tables_ - 1; ++j) {
        indicators_t[j] =
            Tensor(allocator_, {(size_t)batch_size_t.Scalar<int32_t>()}, kInt64);

        int offset = 0, indicators_offset = 0;
        for (int i = 0; i < batch_elements.size(); ++i) {
          std::vector<Tensor> &batch_element = batch_elements[i];

          int begin, end, indicators_begin, indicators_end;
          std::tie(begin, end) = all_table_ranges[i][0];
          std::tie(indicators_begin, indicators_end) =
              all_table_ranges[i][j + 1];

          auto indicators = indicators_t[j].Raw<int64_t>();
          auto current_indicators =
              batch_element[indicators_idx + j].Raw<int64_t>();

          for (int l = 0; l < end - begin; ++l) {
            indicators[offset + l] = current_indicators[begin + l];
            if (indicators[offset + l] >= 0) { // for no refer
              indicators[offset + l] += indicators_offset - indicators_begin;
            }
          }

          offset += end - begin;
          indicators_offset += indicators_end - indicators_begin;
        }
      }

      counter.Wait();
      RETURN_IF_ERROR(status);

      remain_offset_ = all_table_ranges.back()[0].second;
      if (remain_offset_ < dataset()->get_batch_size(batch_elements.back())) {
        remain_.reset(
            new std::vector<Tensor>(std::move(batch_elements.back())));
      }

      if (this->dataset()->gpu_result_) {
        out_tensors->reserve(
            indicators_t.size() + values_t.size() + splits_t.size());
        auto copy_and_push = [&](std::vector<Tensor>& tensors) {
          for (auto& tensor : tensors) {
            if (tensor.Type() == kString) {
              out_tensors->emplace_back(std::move(tensor));
            } else {
              auto tmp_tensor = Tensor(
                  cuda_allocator_,
                  tensor.Shape(), tensor.Type(),
#ifdef USE_ROCM
                  {kDLROCM, this->dataset()->device_id_});
#else
                  {kDLCUDA, this->dataset()->device_id_});
#endif
              GPU_CK(cudaMemcpyAsync(
                    tmp_tensor.mutable_data(),
                    tensor.data(), tensor.TotalBytes(),
                    cudaMemcpyDefault, this->dataset()->stream_));
              out_tensors->emplace_back(std::move(tmp_tensor));
            }
          }
        };
        copy_and_push(indicators_t);
        copy_and_push(values_t);
        copy_and_push(splits_t);
        GPU_CK(cudaStreamSynchronize(this->dataset()->stream_));
      } else {
        std::move(indicators_t.begin(), indicators_t.end(),
                  std::back_inserter(*out_tensors));
        std::move(values_t.begin(), values_t.end(),
                  std::back_inserter(*out_tensors));
        std::move(splits_t.begin(), splits_t.end(),
                  std::back_inserter(*out_tensors));
      }
      *end_of_sequence = false;
      return Status::OK();
    }

    Status GetNextForGroup(std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) {
      std::lock_guard<std::mutex> l(mu_);
      *end_of_sequence = false;
      int sample_group_idx = 0;
      auto cur_batch_size = get_batch_size();
      while (output_elements_.empty() && input_impl_ && !*end_of_sequence) {
        std::vector<Tensor> batch_element;
        RETURN_IF_ERROR(input_impl_->GetNext(&batch_element, end_of_sequence));
        std::map<int, std::vector<int>> map_group_indexes;
        if (!batch_element.empty()) {
          int begin = 0, end = dataset()->get_batch_size(batch_element, true);
          Tensor group_id_t = batch_element[sample_group_idx];
          auto sample_group_ids = group_id_t.Raw<int64_t>();
          auto indicators_t = batch_element[1].Raw<int64_t>();
          for (int i = begin; i < end; i++) {
            int32_t group_id = sample_group_ids[i];
            if (group_id < 0) {
              continue;
            }
            if (max_group_id_ < group_id) max_group_id_ = group_id;
            if (map_group_indexes.find(group_id) == map_group_indexes.end()) {
              map_group_indexes[group_id] = std::vector<int>({i});
            } else {
              map_group_indexes[group_id].push_back(i);
            }
          }
        }
        while (!batch_element.empty() && packers_.size() <= max_group_id_) {
            auto new_packer = new Packer(this->dataset()->pack_tables_, 
              this->dataset()->ragged_ranks_,
              this->dataset()->splits_sub_idx_, 
              this->dataset()->num_tables_, 
              this->dataset()->pool_.get(), 
              cur_batch_size, 
              packers_.size(),
              allocator_);
            packers_.emplace_back(new_packer);
        }
        for (int i = 0; i < packers_.size(); i++) {
          packers_[i]->SetBatchSize(cur_batch_size);
          std::vector<std::vector<Tensor>> tmp_output;
          std::vector<int> the_group_indexes;
          if (map_group_indexes.find(i) != map_group_indexes.end()) {
            the_group_indexes = map_group_indexes[i];
          }
          RETURN_IF_ERROR(packers_[i]->Run(tmp_output, batch_element,
            the_group_indexes, *end_of_sequence));
          for (auto &tmp_elem: tmp_output) {
            if (this->dataset()->drop_remainder_ 
                && this->dataset()->get_batch_size(tmp_elem, true) < cur_batch_size) {
              continue;
            }
            output_elements_.push(std::move(tmp_elem));
          }
          the_group_indexes.clear();
          tmp_output.clear();
        }
        map_group_indexes.clear();
        batch_element.clear();
      }
      if (*end_of_sequence) input_impl_.reset();
      if (output_elements_.empty()) {
        *end_of_sequence = true;
        return Status::OK();
      } 
      *out_tensors = std::move(output_elements_.front());
      output_elements_.pop();
      // 2. [新增] 如果配置了 GPU 结果，执行拷贝
      if (this->dataset()->gpu_result_) {
        std::vector<Tensor> gpu_tensors;
        gpu_tensors.reserve(out_tensors->size());
        
        auto copy_and_push = [&](Tensor& tensor) {
          if (tensor.Type() == kString) {
            // 字符串通常保留在 CPU
            gpu_tensors.emplace_back(std::move(tensor));
          } else {
            // 分配 GPU 内存
            auto tmp_tensor = Tensor(
                cuda_allocator_,
                tensor.Shape(), 
                tensor.Type(),
             #ifdef USE_ROCM
                {kDLROCM, this->dataset()->device_id_});
             #else
                {kDLCUDA, this->dataset()->device_id_});
             #endif
            
            // 异步拷贝 CPU -> GPU
            GPU_CK(cudaMemcpyAsync(
                  tmp_tensor.mutable_data(),
                  tensor.data(), 
                  tensor.TotalBytes(),
                  cudaMemcpyDefault, 
                  this->dataset()->stream_));
            
            gpu_tensors.emplace_back(std::move(tmp_tensor));
          }
        };

        // 遍历所有输出的 Tensor 进行拷贝
        for (auto& t : *out_tensors) {
          copy_and_push(t);
        }
        
        // 同步流，确保数据拷贝完成后再返回给 Python 端
        GPU_CK(cudaStreamSynchronize(this->dataset()->stream_));
        
        // 替换原向量
        *out_tensors = std::move(gpu_tensors);
      }
      *end_of_sequence = false;
      return Status::OK();
    }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      std::lock_guard<std::mutex> l(mu_);
      if (!input_impl_) {
        RETURN_IF_ERROR(writer->WriteScalar(fullname("input_impl_empty"), ""));
      } else {
        RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> l(mu_);
      if (!reader->Contains(fullname("input_impl_empty"))) {
        RETURN_IF_ERROR(RestoreInput(reader, input_impl_));
      } else {
        input_impl_.reset();
      }
      return Status::OK();
    }

  private:
    std::mutex mu_;
    std::shared_ptr<IteratorBase> input_impl_;
    std::unique_ptr<std::vector<Tensor>> remain_;
    int remain_offset_;
    size_t batch_size_;
    std::queue<std::vector<Tensor>> output_elements_;
    std::vector<std::unique_ptr<Packer>> packers_;
    int max_group_id_ = 0;
    Allocator* allocator_ = nullptr;
    Allocator* cuda_allocator_ = nullptr;
  };

  size_t batch_size_;
  const std::vector<int> pack_tables_;
  const std::vector<int> ragged_ranks_;

  const std::shared_ptr<DatasetBase> input_;

  std::unique_ptr<framework::StdThreadPool> pool_;
  cudaStream_t stream_;
  int num_tables_;
  bool do_classify_;
  std::vector<int> splits_sub_idx_;
  bool drop_remainder_;
  bool pinned_result_;
  bool gpu_result_;
  int device_id_;
};
//#undef CASES
//#undef CASE

} // namespace

std::shared_ptr<DatasetBase>
Packer::MakeDataset(const std::shared_ptr<DatasetBase> &input,
                    size_t batch_size, bool drop_remainder,
                    const std::vector<int> &pack_tables, int num_tables,
                    const std::vector<int> &ragged_ranks, int64 parallel,
                    bool pinned_result, bool gpu_result, bool do_classify) {
  return std::make_shared<Dataset>(kDatasetName, batch_size, drop_remainder,
                                   pack_tables, num_tables, ragged_ranks, input,
                                   parallel, pinned_result, gpu_result, do_classify);
}

Status Packer::Run(std::vector<std::vector<Tensor>> &output_elements,
                   std::vector<Tensor> batch_element, 
                   std::vector<int> the_group_indexes,
                   bool is_end) {
  if (batch_element.empty()) {
    if (is_end && total_size_ > 0) {
      int sample_cnt = 0;
      std::vector<int> batch_sample_cnt; 
      std::vector<std::vector<Tensor>> batch_element_vec;
      std::vector<std::vector<std::vector<int>>> all_table_ranges;
      //reserve memory
      batch_sample_cnt.reserve(input_batch_sample_count_.size());
      batch_element_vec.reserve(input_batch_deque_.size());
      all_table_ranges.reserve(input_batch_table_ranges_.size());
      while (!input_batch_deque_.empty()) {
        batch_sample_cnt.push_back(input_batch_sample_count_.front());
        batch_element_vec.push_back(input_batch_deque_.front());
        all_table_ranges.push_back(input_batch_table_ranges_.front());
        sample_cnt += input_batch_sample_count_.front();
        input_batch_sample_count_.pop_front();
        input_batch_deque_.pop_front();
        input_batch_table_ranges_.pop_front();
      }
      std::vector<Tensor> out_tensors;
      if (sample_cnt == 0) return Status::OK();
      RETURN_IF_ERROR(Run(out_tensors, batch_element_vec, all_table_ranges, batch_sample_cnt, sample_cnt));
      output_elements.emplace_back(std::move(out_tensors));
      total_size_ = 0;
    }
    return Status::OK();
  }

  // return if there are not sample belong to this group 
  if (the_group_indexes.size() == 0) return Status::OK(); 

  std::vector<std::vector<int>> batch_table_ranges;
  batch_table_ranges.emplace_back(the_group_indexes);
  input_batch_table_ranges_.push_back(batch_table_ranges);
  input_batch_deque_.push_back(batch_element);
  input_batch_sample_count_.push_back(the_group_indexes.size());
  total_size_ += the_group_indexes.size();//积攒一些数据

  // total_size_很大，batch_size_t很小
  while (total_size_ >= batch_size_) {
    int sample_cnt = 0;
    std::vector<int> batch_sample_cnt; 
    std::vector<std::vector<Tensor>> batch_element_vec;
    std::vector<std::vector<std::vector<int>>> all_table_ranges;//[batch_idx, table_idx, row_idx]
    //packer积攒了很多batch，逐个弹出恢复拼接数据
    while (sample_cnt < batch_size_) {
      batch_sample_cnt.push_back(input_batch_sample_count_.front());
      batch_element_vec.push_back(input_batch_deque_.front());
      all_table_ranges.push_back(input_batch_table_ranges_.front());
      sample_cnt += input_batch_sample_count_.front();
      input_batch_sample_count_.pop_front();
      input_batch_deque_.pop_front();
      input_batch_table_ranges_.pop_front();
    }
    if (sample_cnt > batch_size_) {
      int last_batch_sample_cnt = batch_sample_cnt.back();
      input_batch_sample_count_.push_front(sample_cnt - batch_size_);
      batch_sample_cnt.pop_back();
      last_batch_sample_cnt -= input_batch_sample_count_.front();
      batch_sample_cnt.push_back(last_batch_sample_cnt); 
      input_batch_deque_.push_back(batch_element_vec.back());
      std::vector<int> last_table_ranges;
      for (int i = 0; i < last_batch_sample_cnt; i++) {
        last_table_ranges.push_back(all_table_ranges.back()[0][i]);
      }
      std::vector<int> left_batch_table_ranges;
      //最后一个batch攒了原始样本中的行的标号
      for (int i = last_batch_sample_cnt; i < all_table_ranges.back()[0].size(); i++) {
        left_batch_table_ranges.push_back(all_table_ranges.back()[0][i]);
      }

      all_table_ranges.pop_back();
      std::vector<std::vector<int>> tmp_table_ranges;
      tmp_table_ranges.push_back(last_table_ranges);
      all_table_ranges.push_back(tmp_table_ranges);
      std::vector<std::vector<int>> tmp_batch_table_ranges;
      tmp_batch_table_ranges.push_back(left_batch_table_ranges);
      input_batch_table_ranges_.push_front(tmp_batch_table_ranges);
      sample_cnt -= left_batch_table_ranges.size();
      if (batch_size_ != sample_cnt) {
        return Status(INVALID_ARGUMENT, "batch_size!=sample_cnt.");
      } 
    }
    std::vector<Tensor> out_tensors;
    // produce one new batch
    RETURN_IF_ERROR(Run(out_tensors, batch_element_vec, all_table_ranges, batch_sample_cnt, sample_cnt));
    // put the above new batch into output_elements
    output_elements.emplace_back(std::move(out_tensors));
    total_size_ -= batch_size_;
  }
  return Status::OK();
}

Status Packer::Run(std::vector<Tensor> &out_tensors,
                   std::vector<std::vector<Tensor>> &batch_element_vec, 
                   std::vector<std::vector<std::vector<int>>> &all_table_ranges,
                   std::vector<int> &batch_sample_cnt, int sample_cnt) {
  std::vector<std::vector<int>> batch_table_ranges;
  Tensor batch_size_t(kInt32, {});
  Tensor sample_group_id_t = Tensor(allocator_, 
      {static_cast<size_t>(sample_cnt)}, 
      kInt64);
  std::vector<Tensor> indicators_t(num_tables_ - 1);
  std::vector<Tensor> values_t(num_tensors_);
  std::vector<Tensor> splits_t(num_splits_);

  // get the indicator of each table of each batch
  for (int i = 0; i < batch_element_vec.size(); ++i) {
    std::vector<Tensor> &batch_element = batch_element_vec[i];
    for (int j = 0; j < num_tables_ - 1; ++j) {
      Tensor& batch_indicators_t = batch_element[indicators_idx_ + j];
      auto indicators = batch_indicators_t.Raw<int64>();
      std::unordered_set<int> sample_ids_set;
      std::vector<int> sample_ids;
      for (int k = 0; k < all_table_ranges[i][0].size(); k++) {
        int refer = static_cast<int>(indicators[all_table_ranges[i][0][k]]);
        if (sample_ids_set.find(refer) == sample_ids_set.end()) {
          sample_ids_set.emplace(refer);
          sample_ids.emplace_back(refer);
          //refer id压缩索引 all_table_ranges[i][0][k]原始样本的索引
        }
      }
      all_table_ranges[i].emplace_back(sample_ids);
    }
  }
  

  auto pack_tensor = [&](int j) {
    int table = pack_tables_[j];
    int sub_idx = splits_sub_idx_[j];
    std::vector<int> ragged_size(ragged_ranks_[j] + 1, 0);
    for (int i = 0; i < batch_element_vec.size(); ++i) {
      std::vector<Tensor>& batch_element = batch_element_vec[i];
      // 
      std::vector<int>& sample_idx_vec = all_table_ranges[i][table];
      for (int spl_idx: sample_idx_vec) {
        //由于spl_idx是indicators_t中的结果，[begin, begin + 1)这个区间内
        int begin = spl_idx, end = begin + 1;
        for (int k = ragged_ranks_[j] - 1; k >= 0; --k) {
#define DECLARE_HANDLE_FOR_TYPE(Type) \
            auto handle_##Type = [&]() { \
              ragged_size[k + 1] += end - begin; \
              auto current_splits = \
                batch_element[splits_idx_ + sub_idx + k].Raw<Type>(); \
              begin = current_splits[begin]; \
              end = current_splits[end]; \
            }
            DECLARE_HANDLE_FOR_TYPE(int32_t);
            DECLARE_HANDLE_FOR_TYPE(int64_t);
#undef DECLARE_HANDLE_FOR_TYPE
            if (batch_element[splits_idx_ + sub_idx + k].Type() == kInt64) {
              handle_int64_t();
            } else {
              handle_int32_t();
            }
        }
        ragged_size[0] += end - begin;//value size
      }
    }
    //Allocate tensors
    for (int k = 0; k < ragged_ranks_[j]; ++k) {
#define DECLARE_HANDLE_FOR_TYPE(Allocator, Type, TfType) \
      auto handle_##Type = [&]() { \
        splits_t[sub_idx + k] = \
          Tensor(Allocator, {static_cast<size_t>(ragged_size[k + 1] + 1)}, TfType); \
        splits_t[sub_idx + k].Raw<Type>()[0] = 0; \
      }
      DECLARE_HANDLE_FOR_TYPE(allocator_, int32_t, kInt32);
      DECLARE_HANDLE_FOR_TYPE(allocator_, int64_t, kInt64);
#undef DECLARE_HANDLE_FOR_TYPE
      if (batch_element_vec[0][splits_idx_ + sub_idx + k].Type() == kInt64) {
        handle_int64_t();
      } else {
        handle_int32_t();
      }
    }

    DataType dtype = batch_element_vec[0][values_idx_ + j].Type();
    TensorShape shape = batch_element_vec[0][values_idx_ + j].Shape();
    TensorShape old = shape;
    shape.Set(0, ragged_size[0]);
    values_t[j] = Tensor(allocator_, shape, dtype);
    int dense_dim = 1;
    for (int i = 1; i < shape.Size(); ++i) {
      dense_dim *= shape.Dims()[i];
    }

    // Copy results.
    int next_offset = 0;
    for (int i = 0; i < batch_element_vec.size(); ++i) {
      std::vector<Tensor>& batch_element = batch_element_vec[i];
      std::vector<int> sample_idx_vec = all_table_ranges[i][table];
      for (int spl_idx: sample_idx_vec) {
        int begin = spl_idx, end = begin + 1;
        int offset = next_offset;
        next_offset += end - begin;
        for (int k = ragged_ranks_[j] - 1; k >= 0; --k) {
#define DECLARE_HANDLE_FOR_TYPE(Type) \
          auto handle_##Type = [&]() { \
            auto splits = splits_t[sub_idx + k].Raw<Type>(); \
            auto current_splits = \
              batch_element[splits_idx_ + sub_idx + k].Raw<Type>(); \
            int splits_offset = splits[offset]; \
            int splits_begin = current_splits[begin]; \
            int splits_end = current_splits[end]; \
            for (int l = 1; l <= end - begin; ++l) { \
              splits[offset + l] = \
                current_splits[begin + l] + splits_offset - splits_begin; \
            } \
            begin = splits_begin; \
            end = splits_end; \
            offset = splits_offset; \
          }
          DECLARE_HANDLE_FOR_TYPE(int32_t);
          DECLARE_HANDLE_FOR_TYPE(int64_t);
#undef DECLARE_HANDLE_FOR_TYPE
          if (batch_element[splits_idx_ + sub_idx + k].Type() == kInt64) {
            handle_int64_t();
          } else {
            handle_int32_t();
          }
        }

        CASES(
          dtype, do {
            auto values = values_t[j].Raw<T>();
            auto current_values = batch_element[values_idx_ + j].Raw<T>();
            if (is_simple_type<T>::value) {
              std::memcpy(&values[offset * dense_dim],
                          &current_values[begin * dense_dim],
                          sizeof(T) * (end - begin) * dense_dim);
            } else {
              for (int l = 0; l < (end - begin) * dense_dim; ++l) {
                values[offset + l] = current_values[begin + l];
              }
            }
        } while (0));
      }
    }
    return Status::OK();
  };


  absl::BlockingCounter counter((num_tensors_ - 1) / kPackTensorBlock + 1);
  Status status;
  std::mutex status_mu;
  for (int job_begin = 0; job_begin < num_tensors_;
    job_begin += kPackTensorBlock) {
    int job_end = std::min(job_begin + kPackTensorBlock, num_tensors_);
    pool_->Schedule([job_begin, job_end, &status, &status_mu, &counter, &pack_tensor]() {
      for (int j = job_begin; j < job_end; ++j) {
        Status s = pack_tensor(j);
        {
          std::lock_guard<std::mutex> l(status_mu);
          status = s;
        }
      }
      counter.DecrementCount();
    });
  }



  batch_size_t.Raw<int32_t>()[0] = 0;
  for (int i = 0; i < batch_element_vec.size(); ++i) {
    batch_size_t.Raw<int32_t>()[0] += all_table_ranges[i][0].size();
  }

  if (batch_size_t.Raw<int32_t>()[0] != sample_cnt) {
    LOG(ERROR) << "batch_size:"<< batch_size_t.Raw<int32_t>()[0] << " != sample_cnt:" << sample_cnt; 
    return Status(INVALID_ARGUMENT, "batch_size!=sample_cnt.");
  }

  auto sample_group_ids = sample_group_id_t.Raw<int64_t>();
  for (int i = 0; i < sample_cnt; i++) {
    sample_group_ids[i] = group_id_;
  }

  for (int j = 0; j < num_tables_ - 1; ++j) {
    indicators_t[j] = Tensor(allocator_, {static_cast<size_t>(batch_size_t.Raw<int32_t>()[0])}, kInt64);
    int offset = 0, indicators_offset = 0;
    for (int i = 0; i < batch_element_vec.size(); ++i) {
      std::vector<Tensor>& batch_element = batch_element_vec[i];
      std::vector<int> sample_ids_vec = all_table_ranges[i][0];
      std::vector<int> next_sample_ids_vec = all_table_ranges[i][j + 1];
      int offset_begin = offset;
      int indicators_begin = 0;
      auto indicators = indicators_t[j].Raw<int64_t>();
      auto current_indicators = batch_element[indicators_idx_ + j].Raw<int64_t>();
      for (int sample_id : sample_ids_vec) {
        if(static_cast<int>(current_indicators[sample_id]) == next_sample_ids_vec[indicators_begin]) {
          indicators[offset_begin++] = indicators_offset + indicators_begin;
        } else {
          ++indicators_begin;
          if (static_cast<int>(current_indicators[sample_id]) != next_sample_ids_vec[indicators_begin]) {
            LOG(ERROR) << "current_indicators(sample_id):" << current_indicators[sample_id] << " != " << \
                "next_sample_ids_vec[indicators_begin]:" << next_sample_ids_vec[indicators_begin];
            return Status(INVALID_ARGUMENT, "current_indicators(sample_id) != next_sample_ids_vec[indicators_begin].");
          }
          indicators[offset_begin++] = indicators_offset + indicators_begin;
        } 
      }
      offset += sample_ids_vec.size();
      indicators_offset += next_sample_ids_vec.size();
    }
  }

  counter.Wait();
  RETURN_IF_ERROR(status);
  out_tensors.emplace_back(std::move(sample_group_id_t));
  std::move(indicators_t.begin(), indicators_t.end(),
            std::back_inserter(out_tensors));
  std::move(values_t.begin(), values_t.end(),
            std::back_inserter(out_tensors));
  std::move(splits_t.begin(), splits_t.end(),
            std::back_inserter(out_tensors));
  return Status::OK();
}

Packer::Packer(const std::vector<int> &pack_tables, const std::vector<int> &ragged_ranks, 
        const std::vector<int> &splits_sub_idx, int num_tables, 
    framework::StdThreadPool *pool, int batch_size, int group_id, Allocator* allocator):
  pack_tables_(pack_tables), ragged_ranks_(ragged_ranks), splits_sub_idx_(splits_sub_idx),
  num_tables_(num_tables), pool_(pool), batch_size_(batch_size), group_id_(group_id), total_size_(0), allocator_(allocator) {
  num_tensors_ = ragged_ranks_.size();
  num_splits_ = splits_sub_idx_.back() + ragged_ranks_.back();
  sample_group_id_idx_ = 0;
  indicators_idx_ = 1;
  values_idx_ = indicators_idx_ + (num_tables_ - 1);
  splits_idx_ = values_idx_ + num_tensors_;
}

#undef CASES
#undef CASE
} // namespace dataset
} // namespace column
