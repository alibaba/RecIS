#include "column-io/plugin/aoti_loader.h"

#include <dlfcn.h>
#include <mutex>
#include <absl/log/log.h>  // 或替换为 printf / logging framework

namespace column {
namespace plugin {

// 静态成员定义
void* AOTILoader::g_handle = nullptr;
bool AOTILoader::g_loaded = false;

// 函数指针初始化为 nullptr
AOTIExecutorImpl* (*AOTILoader::create_executor)(const char*) = nullptr;
void (*AOTILoader::destroy_executor)(AOTIExecutorImpl*) = nullptr;
int (*AOTILoader::executor_run)(AOTIExecutorImpl*, const TensorArrayImpl*, TensorArrayImpl**) = nullptr;

TensorArrayImpl* (*AOTILoader::create_tensor_array)() = nullptr;
int (*AOTILoader::tensor_array_size)(const TensorArrayImpl*) = nullptr;
const column::Tensor* (*AOTILoader::tensor_array_get)(const TensorArrayImpl*, int) = nullptr;
int (*AOTILoader::tensor_array_push_back)(TensorArrayImpl*, void*) = nullptr;
void (*AOTILoader::destroy_tensor_array)(TensorArrayImpl*) = nullptr;

// 全局锁
static std::mutex& GetLoaderMutex() {
    static std::mutex m;
    return m;
}

// 全局注册标致
static std::once_flag register_atexit_flag;

bool AOTILoader::Load(const std::string& plugin_so_path) {
    std::lock_guard<std::mutex> lock(GetLoaderMutex());

    if (g_loaded) {
        return g_handle != nullptr;
    }
    g_loaded = true;

    dlerror();  // 清除旧错误
    g_handle = dlopen(plugin_so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!g_handle) {
        const char* err = dlerror();
        LOG(ERROR) << "Cannot load AOTI plugin: " << (err ? err : "unknown error");
        return false;
    }

#define LOAD_SYM(var, name) do { \
    dlerror(); \
    var = (decltype(var))dlsym(g_handle, name); \
    const char* err = dlerror(); \
    if (err || !var) { \
        LOG(ERROR) << "Cannot load symbol '" << name << "': " << (err ? err : "null pointer"); \
        dlclose(g_handle); \
        g_handle = nullptr; \
        return false; \
    } \
} while(0)

    LOAD_SYM(create_executor, "aoti_executor_create");
    LOAD_SYM(destroy_executor, "aoti_executor_destroy");
    LOAD_SYM(executor_run, "aoti_executor_run");

    LOAD_SYM(create_tensor_array, "aoti_tensor_array_create");
    LOAD_SYM(tensor_array_size, "aoti_tensor_array_size");
    LOAD_SYM(tensor_array_get, "aoti_tensor_array_get");
    LOAD_SYM(tensor_array_push_back, "tensor_array_push_back");
    LOAD_SYM(destroy_tensor_array, "aoti_tensor_array_destroy");

#undef LOAD_SYM

    std::call_once(register_atexit_flag, []() {
        std::atexit([]() {
            Unload();
        });
    });
    LOG(INFO) << "Successfully loaded AOTI plugin from " << plugin_so_path;
    return true;
}

bool AOTILoader::IsLoaded() {
    return g_handle != nullptr;
}

void AOTILoader::Unload() {
    std::lock_guard<std::mutex> lock(GetLoaderMutex());
    if (g_handle) {
        dlclose(g_handle);
        g_handle = nullptr;
        LOG(INFO) << "libplugin.so closed";
        // 函数指针变为无效
    }
}

// === 转发实现 === //

AOTIExecutorImpl* AOTILoader::CreateExecutor(const char* path) {
    return IsLoaded() && create_executor ? create_executor(path) : nullptr;
}

void AOTILoader::DestroyExecutor(AOTIExecutorImpl* e) {
    if (IsLoaded() && destroy_executor && e) {
        destroy_executor(e);
    }
}

int AOTILoader::ExecutorRun(AOTIExecutorImpl* e, const TensorArrayImpl* in, TensorArrayImpl** out) {
    return IsLoaded() && executor_run ? executor_run(e, in, out) : -1;
}

TensorArrayImpl* AOTILoader::CreateTensorArray() {
    return IsLoaded() && create_tensor_array ? create_tensor_array() : nullptr;
}

int AOTILoader::TensorArraySize(const TensorArrayImpl* arr) {
    return IsLoaded() && tensor_array_size ? tensor_array_size(arr) : -1;
}

const column::Tensor* AOTILoader::TensorArrayGet(TensorArrayImpl* arr, int index) {
    return IsLoaded() && tensor_array_get ? tensor_array_get(arr, index) : nullptr;
}

int AOTILoader::TensorArrayPushBack(TensorArrayImpl* arr, const column::Tensor* tensor) {
    if (!IsLoaded() || !tensor_array_push_back || !arr) return -1;
    return tensor_array_push_back(arr, const_cast<column::Tensor*>(tensor));
}

void AOTILoader::DestroyTensorArray(TensorArrayImpl* arr) {
    if (IsLoaded() && destroy_tensor_array && arr) {
        destroy_tensor_array(arr);
    }
}

}  // namespace plugin
}  // namespace column

