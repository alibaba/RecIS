#pragma once

#include "column-io/plugin/aot_run_c_api.h"  // 只依赖函数声明，不依赖实现

#include <string>

namespace column {
namespace plugin {

/**
 * AOTI 插件加载器
 *
 * 功能：
 * - 运行时加载 libaoti_runner_plugin.so
 * - 绑定所有 C ABI 函数指针
 * - 提供类型安全的封装调用
 *
 * 特点：
 * - 不依赖任何插件内部实现
 * - 主模块无需链接 libtorch
 */
class AOTILoader {
public:
    /**
     * 加载插件 so 文件（只成功一次）
     * @param plugin_so_path 插件路径，如 "./plugins/libaoti_runner_plugin.so"
     * @return 是否加载成功
     */
    static bool Load(const std::string& plugin_so_path);

    /// 检查是否已加载成功
    static bool IsLoaded();

    /// 卸载插件（可选）
    static void Unload();

    // === 封装的 Executor API === //
    static AOTIExecutorImpl* CreateExecutor(const char* module_so_path);
    static void DestroyExecutor(AOTIExecutorImpl* executor);
    static int ExecutorRun(AOTIExecutorImpl* executor,
                           const TensorArrayImpl* inputs,
                           TensorArrayImpl** outputs);

    // === 封装的 TensorArray API === //
    static TensorArrayImpl* CreateTensorArray();
    static int TensorArraySize(const TensorArrayImpl* arr);
    static const column::Tensor* TensorArrayGet(TensorArrayImpl* arr, int index);
    static int TensorArrayPushBack(TensorArrayImpl* arr, const column::Tensor* tensor);
    static void DestroyTensorArray(TensorArrayImpl* arr);

private:
    AOTILoader() = delete;
    ~AOTILoader() = delete;

    static void* g_handle;
    static bool g_loaded;

    // 函数指针（绑定到 dlsym 结果）
    static AOTIExecutorImpl* (*create_executor)(const char*);
    static void (*destroy_executor)(AOTIExecutorImpl*);
    static int (*executor_run)(AOTIExecutorImpl*, const TensorArrayImpl*, TensorArrayImpl**);

    static TensorArrayImpl* (*create_tensor_array)();
    static int (*tensor_array_size)(const TensorArrayImpl*);
    static const column::Tensor* (*tensor_array_get)(const TensorArrayImpl*, int);
    static int (*tensor_array_push_back)(TensorArrayImpl*, void*);  // void* = column::Tensor*
    static void (*destroy_tensor_array)(TensorArrayImpl*);
};

}  // namespace plugin
}  // namespace column

