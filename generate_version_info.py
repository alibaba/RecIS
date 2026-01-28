#!/usr/bin/env python3
"""
生成包含构建信息的 version_info.py 文件
"""

import datetime
import os
import subprocess
import sys


def run_command(cmd, default="unknown"):
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Warning: Command '{cmd}' failed: {result.stderr.strip()}")
            return default
    except Exception as e:
        print(f"Warning: Command '{cmd}' failed with exception: {e}")
        return default


def get_git_info():
    run_command(f"git config --global --add safe.directory {os.getcwd()}")
    git_info = {}
    git_info["branch"] = run_command("git rev-parse --abbrev-ref HEAD")
    git_info["commit_hash"] = run_command("git rev-parse --short HEAD")
    git_info["commit_hash_full"] = run_command("git rev-parse HEAD")
    git_info["commit_time"] = run_command("git log -1 --format=%ci")
    git_info["commit_author"] = run_command("git log -1 --format=%an")
    git_info["commit_message"] = run_command("git log -1 --format=%s")
    git_info["tag"] = run_command("git describe --tags --exact-match", "")
    if not git_info["tag"]:
        git_info["tag"] = run_command("git describe --tags --abbrev=0", "")

    return git_info


def get_build_info():
    build_info = {}

    build_info["build_time"] = datetime.datetime.now().isoformat()
    build_info["build_timestamp"] = int(datetime.datetime.now().timestamp())

    build_info["python_version"] = sys.version
    build_info["platform"] = run_command("uname -a")
    build_info["hostname"] = run_command("hostname")
    build_info["user"] = run_command("whoami")

    build_info["internal_version"] = os.environ.get("INTERNAL_VERSION", "0")
    build_info["torch_cuda_arch_list"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    build_info["nv_platform"] = os.environ.get("NV_PLATFORM", "0")

    return build_info


def generate_version_info_file(output_path="recis/version_info.py"):
    git_info = get_git_info()
    build_info = get_build_info()

    try:
        from version import get_package_version

        main_version, minor_version, patch_version = get_package_version()
        version = f"{main_version}.{minor_version}.{patch_version}"
    except Exception as e:
        print(f"Warning: Could not get package version: {e}")
        version = "unknown"

    content = f'''

VERSION = "{version}"

GIT_BRANCH = "{git_info["branch"]}"
GIT_COMMIT_HASH = "{git_info["commit_hash"]}"
GIT_COMMIT_HASH_FULL = "{git_info["commit_hash_full"]}"
GIT_COMMIT_TIME = "{git_info["commit_time"]}"
GIT_COMMIT_AUTHOR = "{git_info["commit_author"]}"
GIT_COMMIT_MESSAGE = """{git_info["commit_message"]}"""
GIT_TAG = "{git_info["tag"]}"

BUILD_TIME = "{build_info["build_time"]}"
BUILD_TIMESTAMP = {build_info["build_timestamp"]}
PYTHON_VERSION = """{build_info["python_version"]}"""
PLATFORM = "{build_info["platform"]}"
HOSTNAME = "{build_info["hostname"]}"
BUILD_USER = "{build_info["user"]}"

INTERNAL_VERSION = "{build_info["internal_version"]}"
TORCH_CUDA_ARCH_LIST = "{build_info["torch_cuda_arch_list"]}"
NV_PLATFORM = "{build_info["nv_platform"]}"


def get_version_info():
    """返回完整的版本信息字典"""
    return {{
        'version': VERSION,
        'git': {{
            'branch': GIT_BRANCH,
            'commit_hash': GIT_COMMIT_HASH,
            'commit_hash_full': GIT_COMMIT_HASH_FULL,
            'commit_time': GIT_COMMIT_TIME,
            'commit_author': GIT_COMMIT_AUTHOR,
            'commit_message': GIT_COMMIT_MESSAGE,
            'tag': GIT_TAG,
        }},
        'build': {{
            'build_time': BUILD_TIME,
            'build_timestamp': BUILD_TIMESTAMP,
            'python_version': PYTHON_VERSION,
            'platform': PLATFORM,
            'hostname': HOSTNAME,
            'build_user': BUILD_USER,
            'internal_version': INTERNAL_VERSION,
            'torch_cuda_arch_list': TORCH_CUDA_ARCH_LIST,
            'nv_platform': NV_PLATFORM,
        }}
    }}


def print_version_info():
    """打印版本信息"""
    info = get_version_info()
    print(f"RecIS Version: {{info['version']}}")
    print(f"Git Branch: {{info['git']['branch']}}")
    print(f"Git Commit: {{info['git']['commit_hash']}}")
    print(f"Build Time: {{info['build']['build_time']}}")
    print(f"Build User: {{info['build']['build_user']}}")


if __name__ == "__main__":
    print_version_info()
'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated version info file: {output_path}")

    print("\n=== Build Information Summary ===")
    print(f"Version: {version}")
    print(f"Git Branch: {git_info['branch']}")
    print(f"Git Commit: {git_info['commit_hash']}")
    print(f"Build Time: {build_info['build_time']}")
    print(f"Build User: {build_info['user']}")
    print("=" * 35)


if __name__ == "__main__":
    generate_version_info_file()
