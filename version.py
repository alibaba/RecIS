import os
import re
import subprocess

import torch


def get_package_version():
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, "recis", "__init__.py")) as f:
        content = f.read()
        match = re.search(
            r'__version__\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+(?:\.post[0-9]+)?)"', content
        )
        version = match.group(1)
        print(f"RecIS version {version}")
        return version


################ REVISION VERSION ################
def get_revision_git_commit(shortcut=8):  # type: (int) -> str
    try:
        commit_id = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )
        if shortcut > 0:
            commit_id = commit_id[:shortcut]
        return commit_id
    except subprocess.CalledProcessError as e:
        print("get_git_commit_id error:", e.output.decode("utf-8"))
        return ""


def get_revision_git_branch():  # type: () -> str
    try:
        repo = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.STDOUT,
            )
            .strip()
            .decode("utf-8")
        )
        return repo
    except subprocess.CalledProcessError as e:
        print("get_git_branch error:", e.output.decode("utf-8"))
        return ""


################ WHEEL VERSION ################
def get_wheel_main_version():
    import re

    init_path = os.path.join(os.path.dirname(__file__), "recis", "__init__.py")
    with open(init_path) as f:
        groups = re.findall(r"__version__.*([0-9]+)\.([0-9]+)\.([0-9]+)", f.read())
        major, minor, patch = groups[0]
        return f"{major}.{minor}.{patch}"


def get_wheel_tag_device_type():
    import glob
    import re

    # Check ROCm: /opt/rocm-6.2.0 -> rocm620
    rocm_dirs = sorted(glob.glob("/opt/rocm-*"), key=len, reverse=True)
    if rocm_dirs:
        match = re.search(r"/opt/rocm-(\d+)(?:\.(\d+))?(?:\.(\d+))?", rocm_dirs[0])
        if match:
            version = "".join(filter(None, match.groups()))
            return f"rocm{version}"

    # Check PPU: version: 1.4.2-83b025 -> ppu142
    cuppu_dirs = "/usr/local/PPU_SDK/release.yaml"
    if os.path.exists(cuppu_dirs):
        with open(cuppu_dirs) as f:
            content = f.read()
            match = re.search(r"version:\s*(\d+)\.(\d+)(?:\.(\d+))?", content)
            if match:
                version = "".join(filter(None, match.groups()))
                return f"ppu{version}"

    # Check CUDA: /usr/local/cuda-12.8 -> cu128
    cuda_dirs = sorted(glob.glob("/usr/local/cuda-*"), key=len, reverse=True)
    if cuda_dirs:
        match = re.search(
            r"/usr/local/cuda-(\d+)(?:\.(\d+))?(?:\.(\d+))?", cuda_dirs[0]
        )
        if match:
            version = "".join(filter(None, match.groups()))
            return f"cu{version}"

    return "cpu"


def get_wheel_tag_torch():
    def get_torch_version_with_cuda_sdk():
        torch_raw_version = str(torch.__version__).split(".git")[0]
        torch_main_version = "0"
        torch_cuda_version = "cpu"

        if torch.version.cuda is not None:
            torch_main_version = torch_raw_version.replace(".", "").replace("+", "")
            torch_cuda_version = f"cu{torch.version.cuda.replace('.', '')}"
        elif torch.version.hip is not None:
            torch_main_version, hip_version = torch_raw_version.split("rocm")
            torch_main_version = torch_main_version.replace(".", "").replace("+", "")
            torch_cuda_version = f"rocm{hip_version.replace('.', '')}"
        else:
            raise RuntimeError("Neither CUDA nor ROCm/HIP version found in PyTorch")
        return torch_main_version, torch_cuda_version

    try:
        torch_main_version, torch_cuda_version = get_torch_version_with_cuda_sdk()
    except Exception as e:
        print("get_torch_version error:", e)
        torch_main_version, torch_cuda_version = "0", "cpu"
    return f"torch{torch_main_version}{torch_cuda_version}"


def get_wheel_tag_cxx11abi():
    cxx11abi = str(os.getenv("NEED_FSLIB_ABI", "1"))
    if cxx11abi == "0":
        return "fslibabi0"
    return ""  # cxx11abi == "1" as default, no label added


def get_wheel_commit():
    formal_branch_list = ["master", "main"]
    try:
        tag = (
            subprocess.check_output(
                ["git", "tag", "--points-at", "HEAD"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )
        if tag:
            return ""

        branch = get_revision_git_branch()
        if not branch:
            return ""

        if branch in formal_branch_list:
            return ""

        commit = get_revision_git_commit()
        if not commit:
            return ""

        return f"git{commit}"
    except subprocess.CalledProcessError as e:
        print("get_revision_branch_tag error:", e.output.decode("utf-8"))
        return ""


def get_wheel_version():
    main_version = get_package_version()
    label_tags = []

    # required label
    tag_device_type = get_wheel_tag_device_type()
    label_tags.append(tag_device_type)
    torch_version = get_wheel_tag_torch()
    label_tags.append(torch_version)

    # optional label
    tag_abi_type = get_wheel_tag_cxx11abi()
    label_tags.append(tag_abi_type)
    tag_commit_id = get_wheel_commit()
    label_tags.append(tag_commit_id)

    # trim empty labels. compact label into full wheel version
    label_tags = [v for v in label_tags if v]
    version = f"{main_version}+{'.'.join(label_tags)}"
    return version


""" 版本规范. 由 版本号+标签 两部分组成
    1. 版本号: 由 3+1 个部分组成: major.minor.patch.postN
        1.1. major.minor.patch: 常规语义化版本号, 从 recis/__init__.py 中读取
        1.2. postN: 功能&问题临时修复版本号
        1.3. devN/rcN等预发布类版本号, 视情况发布.
    2. 标签: 标签由 固定标签 + 候选标签 组成. 候选标签不必然存在:
        2.1. 固定标签: 固定标签由 GPU设备类型 + torch版本 组成
        2.2. 候选标签: 候选标签由 提交ID 组成

    - 正式版本发布时, 应在代码仓库标记当前的`版本号`作为tag, 用于归档和回溯版本;
    - 正式版本安装时, 由`版本号+固定标签`组合必须能定位到真实唯一的whl, 实现版本发布控制;
    - 非正式版本安装时没有唯一性要求, 可通过候选标签多版本共存;

    Ref Doc: PEP 440 Wheel Version Identification and Dependency Specification
            https://peps.python.org/pep-0440/
"""

if __name__ == "__main__":
    print(get_wheel_version())
