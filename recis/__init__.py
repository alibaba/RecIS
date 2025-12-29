import os

import torch


__version__ = "1.1.0"

pkg_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(pkg_path, "lib")
try:
    import torch.classes.recis.Hashtable
except Exception:
    if not os.environ.get("BUILD_DOCUMENT", None) == "1":
        lib_path = os.path.join(pkg_path, "lib", "recis.so")
        print(f"RecIS load lib {lib_path}")
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
