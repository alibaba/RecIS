#!/usr/bin/env python3
"""
诊断 column_io / column_io_cpu 双包 import 冲突

运行: python3 tools/diag_import_conflict.py
前置: 需同时安装 column_io 和 column_io_cpu 两个包
      pip install column_io-*.whl column_io_cpu-*.whl
说明: 验证 module_local 修复后两个包可以在同一进程中共存
      测试两种 import 顺序: column_io→column_io_cpu 和 column_io_cpu→column_io
"""
import sys
import traceback

def try_import(pkg_name):
    print(f"\n{'='*60}")
    print(f">>> import {pkg_name}")
    print(f"{'='*60}")
    try:
        mod = __import__(pkg_name)
        has_reader = hasattr(mod, 'OpenStorageRowReader')
        print(f"✅ import 成功, OpenStorageRowReader 存在: {has_reader}")
        if not has_reader:
            print("❌ OpenStorageRowReader 不存在，说明 __init__.py 的 except 吞了异常")
            print(">>> 手动重跑 import 链路，暴露真实异常:")
            try:
                if pkg_name == 'column_io':
                    from column_io.dataset.open_storage_row_reader_v2 import OpenStorageRowReaderV2
                else:
                    from column_io_cpu.dataset.open_storage_row_reader_v2 import OpenStorageRowReaderV2
                print(f"  手动 import 成功? OpenStorageRowReaderV2: {OpenStorageRowReaderV2}")
            except Exception as e:
                print(f"  ❌ 真实异常: {type(e).__name__}: {e}")
                traceback.print_exc()
        return has_reader
    except Exception as e:
        print(f"❌ import 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def show_sys_modules(prefix):
    mods = sorted(k for k in sys.modules if k.startswith(prefix))
    print(f"\nsys.modules 中 {prefix}.* 的模块 ({len(mods)} 个):")
    for m in mods:
        mod = sys.modules[m]
        fname = getattr(mod, '__file__', None) or '<builtin>'
        print(f"  {m:<55} {fname}")

def show_pybind11_internals():
    # pybind11 把 internals 存在 sys.modules 的一个 capsule 里
    keys = [k for k in sys.modules if 'internals' in k.lower() or 'pybind' in k.lower()]
    if keys:
        print(f"\npybind11 internals 已加载: {keys}")
    else:
        print("\npybind11 internals 未找到 (可能还没加载)")

# ---- 测试1: column_io 先, column_io_cpu 后 ----
print("\n" + "█"*60)
print("█  测试1: column_io 先, column_io_cpu 后")
print("█"*60)

try_import('column_io')
show_sys_modules('column_io')
show_pybind11_internals()

try_import('column_io_cpu')
show_sys_modules('column_io_cpu')

# 清理，准备测试2
for k in list(sys.modules):
    if k.startswith('column_io'):
        del sys.modules[k]

# ---- 测试2: column_io_cpu 先, column_io 后 ----
print("\n\n" + "█"*60)
print("█  测试2: column_io_cpu 先, column_io 后")
print("█"*60)

try_import('column_io_cpu')
show_sys_modules('column_io_cpu')
show_pybind11_internals()

try_import('column_io')
show_sys_modules('column_io')
