#pragma once
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"


#ifdef __cplusplus
extern "C" {
#endif

// 不透明类型：执行器句柄
typedef struct AOTIExecutorImpl AOTIExecutorImpl;

// 不透明类型：Tensor 数组（用于表示输入/输出）
typedef struct TensorArrayImpl TensorArrayImpl;

/**
 * 创建 AOT 执行器
 * @param module_so_path 模型 .so 文件路径（C 字符串）
 * @return 成功返回句柄，失败返回 NULL
 */
AOTIExecutorImpl* aoti_executor_create(const char* module_so_path);

/**
 * 销毁执行器
 * @param executor 执行器句柄
 */
void aoti_executor_destroy(AOTIExecutorImpl* executor);

/**
 * 运行模型
 * @param executor 执行器
 * @param inputs 输入数组（不透明指针）
 * @param outputs 输出数组（传出，新创建的对象，需调用 aoti_tensor_array_destroy 释放）
 * @return 0 成功，非 0 失败
 */
int aoti_executor_run(
    AOTIExecutorImpl* executor,
    const TensorArrayImpl* inputs,
    TensorArrayImpl** outputs
);

// =====================================================================================
// TensorArrayImpl 的构造与访问 API（必须由同一模块提供）
// =====================================================================================

/**
 * 创建一个空的 TensorArray
 * @return 新建的 TensorArrayImpl*，需调用 destroy 释放
 */
TensorArrayImpl* aoti_tensor_array_create();

/**
 * 获取 TensorArray 中 Tensor 的数量
 * @param arr 数组对象
 * @return Tensor 个数
 */
int aoti_tensor_array_size(const TensorArrayImpl* arr);

/**
 * 获取指定索引处的 column::Tensor 指针（只读）
 * @param arr 数组对象
 * @param index 索引
 * @return 指向内部 Tensor 的 const 指针，越界返回 nullptr
 */
const column::Tensor* aoti_tensor_array_get(const TensorArrayImpl* arr, int index);

/**
 * 销毁 TensorArray 及其内部所有 Tensor
 * @param arr 要销毁的对象
 */
void aoti_tensor_array_destroy(TensorArrayImpl* arr);


/** 
 * C++层std::vector<column::Tensor>浅拷贝至TensorArrayImpl*, 隔离主程序和.so的内存结构
 */
int tensor_array_push_back(TensorArrayImpl* arr, void* tensor);

#ifdef __cplusplus
}
#endif
