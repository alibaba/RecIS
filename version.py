import os
import re
import subprocess

import torch


def get_package_version():
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, "recis", "__init__.py")) as f:
        groups = re.findall(r"__version__.*([0-9]+)\.([0-9]+)\.([0-9]+)", f.read())
        main_version, minor_version, patch_version = groups[0]
        print(f"RecIS version {main_version}.{minor_version}.{patch_version}")
        return main_version, minor_version, patch_version


def get_cuda_version():
    return torch.version.cuda


def get_device_type():
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
    if os.path.exists("/usr/local/PPU_SDK/release.yaml"):
        with open("/usr/local/PPU_SDK/release.yaml") as f:
            content = f.read()
            match = re.search(r"version:\s*(\d+)\.(\d+)(?:\.(\d+))?", content)
            if match:
                version = "".join(filter(None, match.groups()))
                return f"ppu{version}"

    # Check CUDA: /usr/local/cuda-12.8 -> cuda128
    cuda_dirs = sorted(glob.glob("/usr/local/cuda-*"), key=len, reverse=True)
    if cuda_dirs:
        match = re.search(
            r"/usr/local/cuda-(\d+)(?:\.(\d+))?(?:\.(\d+))?", cuda_dirs[0]
        )
        if match:
            version = "".join(filter(None, match.groups()))
            return f"cuda{version}"
    return "cpu"


def get_git_commit(cwd=None):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def get_version():
    version = get_package_version()
    torch_version_clean = torch.__version__.split(".git")[0]
    torch_version = f"torch{torch_version_clean.replace('.', '').replace('+', '')}"

    if torch.version.cuda is not None:
        cuda_version = f"cuda{torch.version.cuda.replace('.', '')}"
    elif torch.version.hip is not None:
        # hip version is messy
        cuda_version = ""
    else:
        raise RuntimeError(
            "Neither CUDA nor ROCm/HIP version found in PyTorch installation"
        )

    version = f"{'.'.join(version)}+{cuda_version}.{torch_version}.git{get_git_commit()}.{get_device_type()}"
    return version


if __name__ == "__main__":
    print(get_version())
