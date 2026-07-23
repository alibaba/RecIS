#include "column-io/plugin/aot_run_c_api.h"
#include "column-io/plugin/cast_to_torch_tensor.h"
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

#include <vector>
#include <memory>
#include <cstring>


#include <iostream>
using namespace torch::inductor;

// 实际实现（仅本模块可见）
struct AOTIExecutorImpl {
  std::shared_ptr<AOTIModelContainerRunner> runner;
};

struct TensorArrayImpl {
  std::vector<column::Tensor> tensors;
};

extern "C" {

// ==================== Executor API ====================

AOTIExecutorImpl* aoti_executor_create(const char* module_so_path) {
  if (!module_so_path) return nullptr;
  auto impl = new AOTIExecutorImpl();
  impl->runner = std::make_shared<AOTIModelContainerRunnerCpu>(module_so_path);
  return impl;
}

void aoti_executor_destroy(AOTIExecutorImpl* executor) {
  delete executor;
}

int aoti_executor_run(
  AOTIExecutorImpl* executor,
  const TensorArrayImpl* inputs,
  TensorArrayImpl** outputs) {

  if (!executor || !inputs || !outputs) {
      LOG(ERROR) << "AOTIExecutorImpl* executor | TensorArrayImpl* inputs | TensorArrayImpl** outputs should not be null pointer";
      return -1;
  }

  *outputs = nullptr; // 初始化输出
  // Step 1: 转换 inputs: column::Tensor -> torch::Tensor
  std::vector<at::Tensor> torch_inputs;
  for (int i = 0; i < inputs->tensors.size(); ++i) {
      const column::Tensor& col_tensor = inputs->tensors[i];
      auto torch_tensor = column::plugin::CastTensorToTorchTensor(col_tensor, column::plugin::ToDLDataType(col_tensor.Type()));
      torch_inputs.push_back(std::move(torch_tensor));
  }

  // Step 2: 调用模型运行
  //auto torch_outputs = executor->runner->run(torch_inputs);
  std::vector<at::Tensor> torch_outputs;
  try {
      torch_outputs = executor->runner->run(torch_inputs);
  } catch (const c10::Error& e) {
      // PyTorch 自己的异常类型，最常见
      std::cerr << "PyTorch c10::Error: " << e.msg() << std::endl;
      // 可以记录日志、设置错误状态等
      return -1; // 或其他错误码
  } catch (const std::exception& e) {
      // 捕获标准异常
      std::cerr << "Standard exception: " << e.what() << std::endl;
      return -1;
  } catch (...) {
      // 捕获未知异常（必须有，否则会 terminate）
      std::cerr << "Unknown exception during model execution." << std::endl;
      return -1;
  }

  // Step 3: 创建输出 TensorArray
  TensorArrayImpl* out_array = aoti_tensor_array_create();
  if (!out_array) {
      LOG(ERROR) << "Alloc output array failed";
      return -1;
  }

  // step4: 处理输出
  if (torch_outputs.size() != 1) {
    LOG(ERROR) << "user module output vector<torch::Tensor> size should be one in current version";
    return -1;
  }
  c10::IntArrayRef sizes = torch_outputs[0].sizes();
  if (sizes.size() != 1) {
    LOG(ERROR) << "user module output torch tensor dim should be one";
    return -1;
  }
  if (!at::isIntegralType(torch_outputs[0].scalar_type(), false)) {
    LOG(ERROR) << "user module output torch tensor type should be integral";
    return -1;
  }
  column::Tensor group_id_t(column::kInt64, {static_cast<size_t>(sizes[0])});
  std::cout << "torch_outputs[0].scalar_type() is [" << torch_outputs[0].scalar_type() << "]" << std::endl;
  AT_DISPATCH_INTEGRAL_TYPES(torch_outputs[0].scalar_type(), "construct output tensor", [&]() {
      auto* data = torch_outputs[0].data_ptr<scalar_t>();
      for (int64_t i = 0; i < sizes[0]; ++i) {
          group_id_t.Raw<int64_t>()[i] = data[i];
      }
  });
  out_array->tensors.emplace_back(std::move(group_id_t));
  *outputs =  out_array;
  return 0;
}

// ==================== TensorArrayImpl API ====================

TensorArrayImpl* aoti_tensor_array_create() {
  return new TensorArrayImpl();
}


int aoti_tensor_array_size(const TensorArrayImpl* arr) {
  if (!arr) return -1;
  return static_cast<int>(arr->tensors.size());
}

int tensor_array_push_back(TensorArrayImpl* arr, void* tensor) {
  if (!arr || !tensor) {
      return -1;
  }

  column::Tensor* t = static_cast<column::Tensor*>(tensor);
  arr->tensors.push_back(*t);
  return 0;  // success
}

const column::Tensor* aoti_tensor_array_get(const TensorArrayImpl* arr, int index) {
  if (!arr || index < 0 || index >= static_cast<int>(arr->tensors.size())) {
      return nullptr;
  }
  return &(arr->tensors[index]);
}

void aoti_tensor_array_destroy(TensorArrayImpl* arr) {
    delete arr;
}

} // extern "C"


