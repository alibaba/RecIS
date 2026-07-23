# column_io 与 column_io_cpu 同进程 import 冲突修复

## 1. 问题现象

同一 Python 进程中先 `import column_io`，再 `import column_io_cpu`（或反过来），后者必然失败：

```
ImportError: generic_type: type "_OdpsOpenStorageRowDataset" is already registered!
```

更严重的情况下直接 core dump：

```
absl::log_internal::SetTimeZone() has already been called
Aborted (core dumped)
```

无论哪个包先 import，第二个都会失败。

## 2. 根因分析

两个 `.so` 扩展模块在同一进程中被 `dlopen` 加载后，共享进程级全局状态。有两个独立问题叠加：

### 2.1 pybind11 全局类型注册表冲突

pybind11 内部维护一个进程级单例 `internals`，其中 `registered_types_py` 以 Python 类型名字符串为 key 存储类型信息。

GPU 版的 `py_interface.so` 和 CPU 版的 `py_interface.so` 都通过 `PYBIND11_MODULE(py_interface, m)` 注册了同名类型，例如：

```cpp
// interface.cc (GPU 构建) 和 row_interface.cc (Row 构建) 都有:
py::class_<column::dataset::OdpsOpenStorageRowDataset,
           std::shared_ptr<column::dataset::OdpsOpenStorageRowDataset>>(
    m, "_OdpsOpenStorageRowDataset")
```

第一个模块加载时，`"_OdpsOpenStorageRowDataset"` 被注册到全局表。第二个模块加载时，同名 key 已存在，pybind11 抛出 `generic_type: type "..." is already registered!`。

### 2.2 absl::InitializeLog() 重复初始化

两个模块的 `GlobalInit()` 都调用了 `absl::InitializeLog()`，该函数不可重入：

```cpp
// dataset.cc (GPU/CPU-column 构建):
void GlobalInit() {
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  absl::InitializeLog();  // ← 第二次调用会 crash
}

// row_interface.cc (Row 构建):
void GlobalInit() {
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  absl::InitializeLog();  // ← 同上
}
```

`absl::InitializeLog()` 内部调用 `SetTimeZone()`，第二次调用时检测到已设置，直接 `abort()`。

### 2.3 __init__.py 的 except: pass 掩盖了真实错误

`column_io/__init__.py` 中：

```python
try:
    from .dataset.open_storage_row_reader_v2 import OpenStorageRowReaderV2
    __all__ = ['OpenStorageRowReader']
except:
    pass  # ← 吞掉所有异常
```

第二个包 import 时 pybind11 抛出异常，被 `except: pass` 静默吞掉。表面上 `import` 成功了，但 `OpenStorageRowReader` 不存在，后续使用才报 `AttributeError`，误导排查方向。

## 3. 诊断方法

使用 `tools/diag_import_conflict.py` 脚本，绕过 `__init__.py` 的异常吞噬，暴露真实错误：

```bash
python3 tools/diag_import_conflict.py
```

该脚本测试两种 import 顺序，并手动重跑 import 链路以暴露被 `except: pass` 隐藏的异常堆栈。

## 4. 修复方案

### 4.1 pybind11 冲突 → py::module_local()

项目自带 pybind11 2.11.1，原生支持 `py::module_local()` 属性。将该属性传给 `py::class_`，类型注册从**全局注册表**变为**模块局部注册表**，两个模块可以各自注册同名类型而互不冲突。

**实现方式**：通过 C 预处理宏条件性添加 `py::module_local()`：

```cpp
// interface.cc (GPU/CPU-column 共用)
#ifdef NEED_CPU_ONLY
#define PY_LOCAL , py::module_local()
#else
#define PY_LOCAL
#endif

// 使用时: 逗号在宏里，不在调用处
py::class_<T>(m, "_TypeName" PY_LOCAL)
```

- **GPU 构建**（`NEED_CPU_ONLY` 未定义）：`PY_LOCAL` 展开为空，`py::class_<T>(m, "Name")`，类型注册到全局表（与原来完全一致）
- **CPU 构建**（`NEED_CPU_ONLY` 已定义）：`PY_LOCAL` 展开为 `, py::module_local()`，`py::class_<T>(m, "Name", py::module_local())`，类型注册到模块局部表

Row 构建始终使用 `module_local`（它只服务于 CPU 包）：

```cpp
// row_interface.cc
#define PY_LOCAL , py::module_local()  // 无条件
```

**关键细节**：逗号必须放在宏定义里（`, py::module_local()`），不能放在调用处（`, PY_LOCAL`）。否则 GPU 构建时宏展开为空，会产生尾逗号编译错误：`foo(a, b, )`。

**Python 侧零改动**：`module_local` 类型仍通过 `py_interface._TypeName` 正常访问，不影响任何 Python API。

### 4.2 absl 崩溃 → static guard

在两个 `GlobalInit()` 实现中各加 `static bool initialized` 守卫：

```cpp
void GlobalInit() {
  static bool initialized = false;
  if (initialized) return;
  initialized = true;
  // ... absl::InitializeLog() ...
}
```

每个 `.so` 有独立的 `static` 变量副本，但 `absl::InitializeLog()` 只在第一次调用时执行，第二次直接 return，不会 crash。

### 4.3 涉及文件

| 文件 | 改动 |
|---|---|
| `cc/column-io/py_interface/interface.cc` | 添加 `PY_LOCAL` 宏 + 22 处 `py::class_` 添加参数 |
| `cc/column-io/py_interface_row_only/row_interface.cc` | 添加 `PY_LOCAL` 宏 + 1 处 `py::class_` + `GlobalInit` 加守卫 |
| `cc/column-io/py_interface/dataset.cc` | `GlobalInit` 加守卫 |
| Python 侧 | **零改动** |

## 5. 验证

| 测试 | 命令 | 验证点 |
|---|---|---|
| 基础功能回归 | `pytest tests/integration/list_string_dataset_test.py -vs` | `module_local` 未破坏单包正常使用 |
| 双包共存 | `python3 tools/diag_import_conflict.py` | 两个包同进程 import 不报错 |
| pybind11 官方测试 | `cd third-party/pybind11 && pytest tests/test_local_bindings.py -vs` | `module_local` 特性本身可靠 |
