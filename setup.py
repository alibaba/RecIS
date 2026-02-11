import os
import subprocess

from setuptools import find_packages, setup
from torch.cuda import get_device_name
from torch.utils import cpp_extension
from version import get_version


BASEDIR = os.path.dirname(os.path.realpath(__file__))


def get_source_file(source_dir, delete_set, with_path):
    ret = []
    for dirname, dirs, filenames in os.walk(source_dir):
        ret.extend(
            [
                os.path.join(dirname, filename) if with_path else filename
                for filename in filenames
                if (filename.endswith((".cc", ".cu", ".c")))
                and (filename not in delete_set)
            ]
        )
    return ret


def get_source_files(source_dirs, prefix, delete_set, with_path):
    source_dirs = [os.path.join(prefix, dirname) for dirname in source_dirs]
    source_files = []
    for source_dir in source_dirs:
        source_files.extend(get_source_file(source_dir, delete_set, with_path))
    return source_files


def nv_device():
    nv_platform = os.environ.get("NV_PLATFORM", None)
    if nv_platform is not None:
        return nv_platform == "1"
    return "NVIDIA" in get_device_name() or "Tesla" in get_device_name()


def amd_device():
    amd_platform = os.environ.get("AMD_PLATFORM", None)
    if amd_platform is not None:
        return amd_platform == "1"
    return "AMD" in get_device_name()


def get_main_extension():
    include_dirs = [
        "csrc",
        "third_party",
        "third_party/cuCollections/include",
    ]
    if nv_device():
        include_dirs.extend(
            [
                "third_party/cccl/cub",
                "third_party/cccl/thrust",
                "third_party/cccl/libcudacxx/include",
            ]
        )
    elif amd_device():
        pass
    else:
        include_dirs.extend(
            [
                "third_party/cccl/cub",
                "third_party/cccl/thrust",
                "third_party/cccl/libcudacxx/include",
            ]
        )
    library_dirs = ["/usr/local/lib64/"]
    extra_link_args = ["-lomp", "-Wl,-rpath,$ORIGIN"]
    include_dirs = [os.path.join(BASEDIR, include_dir) for include_dir in include_dirs]
    delete_dirs = []
    if not is_internal_enabled():
        delete_dirs.extend(["platform/fslib"])
    else:
        library_dirs.append("recis/lib")
        include_dirs.append(os.path.join(BASEDIR, "build/fs_deps"))
        extra_link_args.append("-lfslib_c_api")
    delete_files = get_source_files(delete_dirs, "csrc", set(), False)

    source_dirs = [
        "embedding",
        "monitor",
        "platform",
        "distributed",
        "ops",
        "serialize",
        "utils",
        "data",
    ]
    source_files = get_source_files(source_dirs, "csrc", set(delete_files), True)

    gcc_args = ["-g", "-fopenmp"]
    nvcc_args = [
        "-O2",
    ]
    # nvcc_args = ["-O0", "--expt-extended-lambda", "--expt-relaxed-constexpr", "-lineinfo", "-G"]

    if nv_device():
        nvcc_args.extend(
            [
                "-lineinfo",
                "--expt-extended-lambda",
                "--expt-relaxed-constexpr",
                "-DNV_PLATFORM=1",
            ]
        )
    elif amd_device():
        nvcc_args.extend(["-DAMD_PLATFORM=1"])
    else:
        nvcc_args.extend(
            [
                "-lineinfo",
                "--expt-extended-lambda",
                "--expt-relaxed-constexpr",
                "-DCCCL_DISABLE_FP16_SUPPORT=1",
            ]
        )

    ext = cpp_extension.CUDAExtension(
        name="recis.lib.recis",
        sources=[
            "csrc/bind.cc",
        ]
        + source_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_link_args=extra_link_args,
        extra_compile_args={
            "cxx": gcc_args,
            "nvcc": nvcc_args,
        },
    )
    print(f"ext is {ext}")
    return ext


def is_internal_enabled():
    return int(os.environ.get("INTERNAL_VERSION", 0))


def generate_version_info():
    try:
        print("Generating version_info.py...")
        subprocess.check_call(
            ["python", os.path.join(BASEDIR, "generate_version_info.py")]
        )
        print("version_info.py generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate version_info.py: {e}")
    except Exception as e:
        print(f"Warning: Error generating version_info.py: {e}")


def prepare_build():
    # 生成版本信息文件
    generate_version_info()

    subprocess.check_call(
        [
            "cp",
            "-f",
            os.path.join(BASEDIR, "recis/__info__.py"),
            os.path.join(BASEDIR, "recis/info.py"),
        ]
    )
    subprocess.check_call(
        [
            "sed",
            "-i",
            f"s/__RECIS_INTERNAL_VERSION__/{is_internal_enabled()}/g",
            os.path.join(BASEDIR, "recis/info.py"),
        ]
    )
    if is_internal_enabled():
        build_fslib()


def build_fslib():
    os.makedirs("build", exist_ok=True)
    install_lib = os.path.join(BASEDIR, "recis", "lib")
    os.makedirs("recis/lib", exist_ok=True)
    cmake_bin_path = os.getenv("CMAKE_BIN_PATH", "cmake")
    # if sys level cmake unavailable, use pip level cmake
    subprocess.check_call(
        [cmake_bin_path, "..", f"-DRECIS_LIB_INSTALL_DIR={install_lib}"], cwd="build"
    )
    subprocess.check_call(["make", "-j64"], cwd="build")
    subprocess.check_call(["make", "install"], cwd="build")


prepare_build()

setup(
    version=get_version(),
    ext_modules=[get_main_extension()],
    data_files=[],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        "build_ext": cpp_extension.BuildExtension.with_options(
            no_python_abi_suffix=True
        )
    },
)
