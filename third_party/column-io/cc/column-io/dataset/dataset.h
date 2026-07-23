#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_DATASET_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_DATASET_H_
#pragma once
#include <stddef.h>

#include <functional>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "column-io/monitor/metric_client.h"
#include "column-io/framework/refcount.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/thread_pool.h"

namespace column {
namespace dataset {
class IteratorStateWriter {
public:
  virtual Status WriteString(const std::string &key,
                             const std::string &val) = 0;
  virtual Status WriteScalar(const std::string &key, int64_t val) = 0;
  virtual Status WriteScalar(const std::string &key,
                             const std::string &val) = 0;
  virtual Status WriteInt(const std::string &key, int64_t val) = 0;
  virtual Status WriteFloat(const std::string &key, double val) = 0;
  virtual Status WriteTensor(const std::string &key, const Tensor tensor) = 0;
  virtual ~IteratorStateWriter() = default;
};

class IteratorStateReader {
public:
  virtual bool Contain(const std::string &key) = 0;
  virtual bool Contains(const std::string &key) = 0;
  virtual Status ReadString(const std::string &key, std::string &val) = 0;
  virtual Status ReadInt(const std::string &key, int64_t &val) = 0;
  virtual Status ReadFloat(const std::string &key, double &val) = 0;
  virtual Status ReadTensor(const std::string &key, Tensor &tensor) = 0;
  virtual Status ReadScalar(const std::string &key, int64_t *val) = 0;
  virtual Status ReadScalar(const std::string &key, std::string *val) = 0;
  virtual ~IteratorStateReader() = default;
};

class IteratorBase {
public:
  virtual ~IteratorBase() {}
  IteratorBase() {}
  virtual Status Initialize() { return Status::OK(); }
  virtual Status GetNext(std::vector<Tensor> *outputs,
                         bool *end_of_sequence,
                         std::vector<size_t> *outputs_row_spliter = nullptr) = 0;
  virtual Status Save(IteratorStateWriter *writer) = 0;
  virtual Status Restore(IteratorStateReader *reader) = 0;

protected:
  Status SaveInput(IteratorStateWriter *writer,
                   std::shared_ptr<IteratorBase> &input) {
    return input->Save(writer);
  }
  Status RestoreInput(IteratorStateReader *reader,
                      std::shared_ptr<IteratorBase> &input) {
    return input->Restore(reader);
  }
  virtual Status SaveInternal(IteratorStateWriter *writer) = 0;
  virtual Status RestoreInternal(IteratorStateReader *reader) = 0;
  virtual Status GetNextInternal(std::vector<Tensor> *outputs,
                                 bool *end_of_sequence,
                                 std::vector<size_t> *outputs_row_spliter = nullptr) {
    /* @outputs: next遍历得到的原始tensor数据
     * @end_of_sequence: EOF标志位
     * @outputs_row_spliter: 行存输出模式下 原始tensors数据中列间分割点. (列存输出无需本参数: python schema生成ragged_ranks_用于packer.cc分割)
    */
    return Status::Unimplemented(__FUNCTION__, "Not Implemented");
  }
};

class DatasetBase : public std::enable_shared_from_this<DatasetBase> {
public:
  explicit DatasetBase(const std::string &name);
  const std::string &name() const { return name_; }
  Status MakeIterator(const std::string &prefix,
                      std::shared_ptr<IteratorBase> *iterator) {
    *iterator = MakeIteratorInternal(prefix);
    return (*iterator)->Initialize();
  }
  virtual ~DatasetBase() {}

protected:
  virtual std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) = 0;

private:
  const std::string name_;
};

class DatasetIteratorBase : public IteratorBase {
public:
  struct BaseParam {
    const std::shared_ptr<DatasetBase> dataset;
    const std::string prefix;
  };
  virtual Status GetNext(std::vector<Tensor> *outputs,
                         bool *end_of_sequence,
                         std::vector<size_t> *outputs_row_spliter = nullptr) final {
    /* @outputs: next遍历得到的原始tensor数据
     * @end_of_sequence: EOF标志位
     * @ragged_ranks: 行存输出模式下 原始tensors数据中列间分割点. 列存输出模式无需本参数, 由python schema生成ragged_ranks_交予packer分割
    */
    return GetNextInternal(outputs, end_of_sequence, outputs_row_spliter);
  }
  virtual Status Save(IteratorStateWriter *writer) final {
    return SaveInternal(writer);
  }
  virtual Status Restore(IteratorStateReader *reader) final {
    return RestoreInternal(reader);
  }
  virtual const std::string prefix() const { return params_.prefix; }
  DatasetIteratorBase(const BaseParam &param) : params_(param) {}
  const std::string fullname(const std::string &name) {
    std::string fullname = prefix() + ":" + name;
    return fullname;
  }
  ~DatasetIteratorBase() {}

private:
  BaseParam params_;
};

template <typename DatasetType>
class DatasetIterator : public DatasetIteratorBase {
public:
  struct Param {
    const std::shared_ptr<DatasetType> dataset;
    const std::string prefix;
  };
  const std::shared_ptr<DatasetType> dataset() { return typed_dataset_; }
  DatasetIterator(const Param &param)
      : DatasetIteratorBase({param.dataset, param.prefix}) {
    typed_dataset_ = param.dataset;
    metric_cli_ = recis::monitor::Factory::StaticInstance()->get_client("columnio.dataset");
  };
protected:
  std::shared_ptr<recis::monitor::Client> metric_cli_;
  /**
   * @brief 获取自身被父进程启动的顺序
   * @return int 启动顺序
   * 多进程模式下通过不同slice id实现多读, 但metric汇聚时 通过slice id或进程号等方式区分数据源 其开销过大.
   * 不区分进程又会导致进程级别数据聚合紊乱. torch.dataloader未直接赋予columnio层子进程序号, 这里主动实现.
  */
  static int get_child_order() {
    static int order = 0;
    static std::once_flag once_flag;
    std::call_once(once_flag, []() {
      order = []() {
        pid_t my_pid = getpid();
        pid_t parent_pid = getppid();
        char path[256];
        snprintf(path, sizeof(path), "/proc/%d/task/%d/children", parent_pid, parent_pid);
        
        FILE* f = fopen(path, "r");
        if (!f) return -1; // 打开错误, 返回默认值
        int count = 0;
        pid_t child_pid;
        while (fscanf(f, "%d", &child_pid) == 1) {
            count++;
            if (child_pid == my_pid || count > 1024 ) { // 防止死循环. 虽然理论不会发生
                fclose(f);
                return count;
            }
        }
        fclose(f);
        return 0;
      }();
    });
    return order;
  }
private:
  std::shared_ptr<DatasetType> typed_dataset_;
};

class DatasetBuilder {
  using CallFunc = std::function<absl::StatusOr<std::shared_ptr<DatasetBase>>(
      const std::string &)>;
  using CallVectorFunc = std::function<absl::StatusOr<std::shared_ptr<DatasetBase>>(
      const std::vector<std::string> &)>;

public:
  static std::shared_ptr<DatasetBuilder> Make(CallFunc func) {
    return std::shared_ptr<DatasetBuilder>(new DatasetBuilder(func));
  }
  static std::shared_ptr<DatasetBuilder> Make(CallVectorFunc vec_func) {
    return std::shared_ptr<DatasetBuilder>(new DatasetBuilder(vec_func));
  }
  DatasetBuilder() {}
  DatasetBuilder(CallFunc func) : func_(func) {}
  DatasetBuilder(CallVectorFunc vec_func) : vec_func_(vec_func) {}
  ~DatasetBuilder() {}
  absl::StatusOr<std::shared_ptr<DatasetBase>>
  MakeDataset(const std::string &path) {
    return func_(path);
  }
  absl::StatusOr<std::shared_ptr<DatasetBase>>
  MakeDataset(const std::vector<std::string> &path) {
    return vec_func_(path);
  }

private:
  CallFunc func_;
  CallVectorFunc vec_func_;
};

} // namespace dataset
} // namespace column

#endif
