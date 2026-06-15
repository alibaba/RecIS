import os

import torch


__version__ = "1.1.67"

pkg_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(pkg_path, "lib")
try:
    import torch.classes.recis.Hashtable
except Exception:
    if not os.environ.get("BUILD_DOCUMENT", None) == "1":
        lib_path = os.path.join(pkg_path, "lib", "recis.so")
        print(f"[INFO] RecIS load lib {lib_path}")
        torch.classes.load_library(lib_path)

try:
    from . import version_info

    __build_info__ = version_info.get_version_info()
except ImportError:
    __build_info__ = {
        "version": __version__,
        "git": {
            "branch": "unknown",
            "commit_hash": "unknown",
            "commit_hash_full": "unknown",
            "commit_time": "unknown",
            "commit_author": "unknown",
            "commit_message": "unknown",
            "tag": "unknown",
        },
        "build": {
            "build_time": "unknown",
            "build_timestamp": 0,
            "python_version": "unknown",
            "platform": "unknown",
            "hostname": "unknown",
            "build_user": "unknown",
            "internal_version": "0",
            "torch_cuda_arch_list": "",
            "nv_platform": "0",
        },
    }


def get_build_info():
    return __build_info__


def append_fslib_library_path():
    import os

    old_library_path = os.getenv("LD_LIBRARY_PATH", "")

    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.path.join(pkg_path, 'lib')}:{old_library_path}".rstrip(":")
    )
    print("[INFO] recis reloaded LD_LIBRARY_PATH as: ", os.environ["LD_LIBRARY_PATH"])


if get_build_info().get("build", {}).get("internal_version", "0") == "1":
    """
    FSLIB使用LD_LIBRARY_PATH发现各种协议头plugin插件so. RecIS的lib目录不属于默认的系统或python注入路径, 故需要手动更新
    """
    try:
        append_fslib_library_path()
    except Exception:
        pass
