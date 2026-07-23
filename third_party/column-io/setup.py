from setuptools import setup, find_packages
import subprocess
import os,time,json,sys,shutil


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

def get_revision_git_repository():   # type: () -> str
    try:
        repo = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"], stderr=subprocess.STDOUT,
            ).strip().decode("utf-8") )
        return repo
    except subprocess.CalledProcessError as e:
        print("Error:", e.output.decode("utf-8"))
        return ""

def get_revision_git_branch():   # type: () -> str
    try:
        repo = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.STDOUT,
            ).strip().decode("utf-8") )
        return repo
    except subprocess.CalledProcessError as e:
        print("get_git_branch error:", e.output.decode("utf-8"))
        return ""

def get_compile_cmd():  # type: () -> str
    try:
        cmd = subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "cmd="], stderr=subprocess.STDOUT,
        ).strip().decode("utf-8")
        return cmd
    except subprocess.CalledProcessError as e:
        print("get_compile_cmd error:", e.output.decode("utf-8"))
        return ""

def get_compile_env():  # type: () -> dict[str,str]
    envset = set({"PATH", "LD", "FLAG", "NEED", "TF", "TORCH"}) 
    env_dict = {}
    for k, v in os.environ.items():
        # if  k not like %envset[i]%  continue
        if not any(x in k for x in envset):
            continue
        env_dict[k] = v
    return env_dict

def write_current_revision():
    # type: () -> None
    CURRENT_REVISION = {
        "repository": {
            "repo": get_revision_git_repository(),
            "branch": get_revision_git_branch(),
            "commit": get_revision_git_commit(shortcut=-1),
        },
        "compile": {
            "cmd": get_compile_cmd(),
            "time": time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
            "env": get_compile_env(),
        },
    }
    with open("column_io/CURRENT_REVISION.json", "w") as f:
        json.dump(CURRENT_REVISION, f, indent=4)
    return
write_current_revision()


################ PYTHON VERSION ################
cmdclass, python_tag = {}, "py3"
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    python_tag = "cp{}{}".format(sys.version_info.major, sys.version_info.minor)
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False
            self.python_tag = python_tag
        def get_tag(self):
            _, abi_tag, plat = super().get_tag()
            return python_tag, abi_tag, plat
    cmdclass = {"bdist_wheel": bdist_wheel}
except ImportError:
    pass


################ WHEEL VERSION ################
def get_wheel_main_version():
    version_path = os.path.join(os.path.dirname(__file__), "VERSION.txt")
    with open(version_path, "r") as f:
        return f.read().strip()

def get_wheel_tag_device_type():
    if need_cpu_only():
        return "cpu"
    if need_row_only():
        return "row"

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
        with open(cuppu_dirs, "r") as f:
            content = f.read()
            match = re.search(r"version:\s*(\d+)\.(\d+)(?:\.(\d+))?", content)
            if match:
                version = "".join(filter(None, match.groups()))
                return f"cuppu{version}"
    
    # Check CUDA: /usr/local/cuda-12.8 -> cuda128
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
    #TODO: 开发纯cpu模式`NEED_CPU_ONLY`以用于非显卡环境运行的IO
    #TODO: 开发纯行读模式`NEED_ROW_ONLY`以用于移除列读sdk的精简瘦身版
    if need_cpu_only():
        return "torch0cpu"
    if need_row_only():
        return "torch0row"

    def get_torch_version_with_cuda_sdk():
        import torch
        """ torch.__version__:  '2.6.0',    2.8.0+rocm7.0.2.git7a520360
            torch.version.cuda: '12.6',     None
            torch.version.hip:  None,       '7.0.51831-7c9236b16'
        """
        torch_raw_version = str(torch.__version__).split(".git")[0]
        torch_main_version = "0"
        torch_cuda_version = "cpu"

        if torch.version.cuda is not None:
            torch_main_version = torch_raw_version.replace('.', '').replace('+', '')
            torch_cuda_version = f"cu{torch.version.cuda.replace('.', '')}"
        elif torch.version.hip is not None:
            # hip version is messy
            torch_main_version, hip_version = torch_raw_version.split("rocm")
            torch_main_version = torch_main_version.replace('.', '').replace('+', '')
            torch_cuda_version = f"rocm{hip_version.replace('.', '')}"
        else:
            raise RuntimeError(
                "Neither CUDA nor ROCm/HIP version found in PyTorch"
            )
        return torch_main_version, torch_cuda_version
    try:
        torch_main_version, torch_cuda_version = get_torch_version_with_cuda_sdk()
    except Exception as e:
        print("get_torch_version error:", e)
        torch_main_version, torch_cuda_version = "0", "cpu"
    # e.g. torch280cu128, torch240rocm702, torch0cpu
    return f"torch{torch_main_version}{torch_cuda_version}"

def get_wheel_tag_cxx11abi():
    cxx11abi = str(os.getenv("NEED_ODPS_COLUMN", "1"))
    if cxx11abi == "0":
        return "abi0"
    return "" # cxx11abi == "1" as default, no label added

def get_wheel_commit():
    formal_branch_list = ["master", "main"]
    try:
        # 携带正式version标签的构建 不单独生成提交id
        tag = subprocess.check_output(
            ["git", "tag", "--points-at", "HEAD"], stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        if tag:
            return ""

        # 某些无法获取git信息的 actions 构建环境 不单独生成提交id
        branch = get_revision_git_branch()
        if not branch:
            return ""

        # 主干分支提交版本唯一 不单独生成提交id
        if branch in formal_branch_list:
            return ""

        # 某些无法获取git信息的 actions 构建环境 不单独生成提交id
        commit = get_revision_git_commit()
        if not commit:
            return ""

        # 本地/CI环境的非主干分支 生成提交id
        return f"git{commit}"
    except subprocess.CalledProcessError as e:
        print("get_revision_branch_tag error:", e.output.decode("utf-8"))
        return ""


################ FLAGS & COMPILE ################
def is_internal_enabled():
    return int(os.environ.get("INTERNAL_VERSION", 0))


def need_cpu_only():
    return os.environ.get("NEED_CPU_ONLY", "0") == "1"


def need_row_only():
    return os.environ.get("NEED_ROW_ONLY", "0") == "1"


def cmake():
    is_internal = "ON" if is_internal_enabled() else "OFF"
    build_dir = "build"
    os.makedirs(build_dir, exist_ok=True)

    # Row 版不编译 plugin，不需要 torch；GPU/CPU 版需要
    if not need_row_only():
        try:
            subprocess.check_call(
                [sys.executable, "-c", "import torch"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(
                f"torch not found in {sys.executable}. "
                f"GPU/CPU 构建需要 torch，请确认使用了正确的 Python 环境。"
            )

    cmake_bin = os.getenv("CMAKE_BIN_PATH", "cmake")
    cmake_args = [
        cmake_bin, "..",
        "-DINTERNAL_VERSION={}".format(is_internal),
        "-DPYTHON_EXECUTABLE={}".format(sys.executable),
    ]
    # Auto-detect compiler path for environments where /usr/bin/cc is missing
    cc = os.environ.get("CC") or shutil.which("cc")
    cxx = os.environ.get("CXX") or shutil.which("c++")
    if cc:
        cmake_args.append("-DCMAKE_C_COMPILER={}".format(cc))
    if cxx:
        cmake_args.append("-DCMAKE_CXX_COMPILER={}".format(cxx))
    if need_cpu_only():
        cmake_args.append("-DNEED_CPU_ONLY=ON")
    elif need_row_only():
        cmake_args.append("-DNEED_ROW_ONLY=ON")
    if subprocess.check_call(cmake_args, cwd=build_dir) != 0:
        raise RuntimeError("run cmake failed")
    if subprocess.check_call(["make", "-j"], cwd=build_dir) != 0:
        raise RuntimeError("run make failed")

def get_wheel_version():
    main_version = get_wheel_main_version()
    label_tags = []
    cpu_or_row = need_cpu_only() or need_row_only()

    if not cpu_or_row:
        # GPU 版本需要 device type + torch version 标签
        tag_device_type = get_wheel_tag_device_type()
        label_tags.append(tag_device_type)
        torch_version = get_wheel_tag_torch()
        label_tags.append(torch_version)
    elif need_cpu_only():
        # CPU 版本标注 column 能力，以区分 Row 精简版
        label_tags.append("column")
    # Row 版本无额外固定标签，保持高频发布简洁

    # optional label: abi0（abi1 默认省略，三版本统一）
    tag_abi_type = get_wheel_tag_cxx11abi()
    label_tags.append(tag_abi_type)

    # optional label: git commit（三版本统一，仅 dev 分支出现）
    tag_commit_id = get_wheel_commit()
    label_tags.append(tag_commit_id)

    # trim empty labels, compact into wheel version.
    # Row 正式版 abi1 场景下所有标签为空，直接拼接 "+" 会得到 "0.2.66+" 非法 PEP 440 版本号，
    # 因此必须先判空：无标签时返回纯版本号，有标签时才拼接。
    label_tags = [v for v in label_tags if v]
    if label_tags:
        version = f'{main_version}+{".".join(label_tags)}'
    else:
        version = main_version
    return version

""" 版本规范. 由 版本号+标签 两部分组成，遵循 PEP 440
    1. 版本号 major.minor.patch(.postN):
        VERSION.txt 读取, e.g. 0.2.66
        1.1. major.minor.patch: 常规语义化版本号
        1.2. postN: 功能&问题临时修复版本号
        1.3. devN/rcN等预发布类版本号, 视情况发布.

    2. 标签 = 固定标签 + 候选标签，通过 + 连接, e.g. +column.abi0.gitc34de9e

       固定标签(deterministic):
         GPU:  device_type.torch_version    e.g. cu128.torch280cu128
         CPU:  column                       e.g. column
         Row:  无（高频发布，保持简洁）

       候选标签(optional, 为空时跳过):
         abi0: 仅 NEED_ODPS_COLUMN=0 时出现(abi1 默认省略)
         gitXXXXXXXX: 仅非主干分支且无 tag 时出现

    - 正式版本唯一确定: 版本号 + 固定标签（+ abi0 若启用）
    - 非正式版本通过 git commit 标签共存多个构建

    典型产物示例：
      GPU-NV  正式 abi1 : column_io-0.2.66+cu128.torch280cu128-cp310-cp310-linux_x86_64.whl
      GPU-NV  Dev   abi0 : column_io-0.2.66+cu128.torch280cu128.abi0.gitc34de9e-cp310-cp310-linux_x86_64.whl
      GPU-PPU 正式 abi1 : column_io-0.2.66+cuppu153.torch260cuppu153-cp310-cp310-linux_x86_64.whl
      GPU-AMD 正式 abi1 : column_io-0.2.66+rocm702.torch280rocm702-cp310-cp310-linux_x86_64.whl
      CPU     正式 abi1 : column_io_cpu-0.2.66+column-cp310-cp310-linux_x86_64.whl
      CPU     Dev   abi0 : column_io_cpu-0.2.66+column.abi0.gitc34de9e-cp310-cp310-linux_x86_64.whl
      Row     正式 abi1 : column_io_cpu-0.2.66-cp310-cp310-linux_x86_64.whl
      Row     正式 abi0 : column_io_cpu-0.2.66+abi0-cp310-cp310-linux_x86_64.whl
      Row     Dev   abi1 : column_io_cpu-0.2.66+gitc34de9e-cp310-cp310-linux_x86_64.whl

    Ref Doc: PEP 440 Wheel Version Identification and Dependency Specification 
            https://peps.python.org/pep-0440/
"""
version = get_wheel_version()
print(f"[INFO] version: {version}-{python_tag}")
cmake()


################ PACKAGE ################
# column-io 和 column-io-cpu 是两个独立包，必须装到 site-packages/ 下各自命名空间，
# 否则同时安装会互相覆盖。GPU 构建直接用 column_io/ 源码；CPU/Row 构建时先把
# column_io/ copy 为 column_io_cpu/ 并改写内部 import，让 wheel 装到 column_io_cpu/。
_PKG_SRC = "column_io"
_PKG_DST = "column_io_cpu"

def _prepare_cpu_package():
    if os.path.exists(_PKG_DST):
        shutil.rmtree(_PKG_DST)
    shutil.copytree(_PKG_SRC, _PKG_DST,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for root, _, files in os.walk(_PKG_DST):
        for fname in files:
            if not fname.endswith('.py'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as fh:
                content = fh.read()
            if 'column_io.' not in content:
                continue
            content = content.replace('from column_io.', 'from column_io_cpu.')
            content = content.replace('import column_io.', 'import column_io_cpu.')
            with open(fpath, 'w') as fh:
                fh.write(content)

def _cleanup_cpu_package():
    if os.path.exists(_PKG_DST):
        shutil.rmtree(_PKG_DST)

_cpu_or_row = need_cpu_only() or need_row_only()
if _cpu_or_row:
    _prepare_cpu_package()

try:
    setup(
        name="column_io_cpu" if _cpu_or_row else "column_io",
        version=version,
        cmdclass=cmdclass,
        packages=(find_packages(include=['column_io_cpu', 'column_io_cpu.*'])
                  if _cpu_or_row else find_packages()),
        package_data=({'column_io_cpu': ['CURRENT_REVISION.json', 'ODPS_SDK_VERSION']}
                      if _cpu_or_row else {}),
        include_package_data=True,
        install_requires=[],
    )
finally:
    if _cpu_or_row:
        _cleanup_cpu_package()
