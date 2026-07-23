#!/bin/bash

# Usage:
#   ./tools/utest_on_cc.sh [VERBOSE]
#
# Arguments:
#   VERBOSE       Whether to show test program's stdout (default: -s)

set -euo pipefail

VERBOSE=${VERBOSE:-" -V "}  # 是否展示测试程序的std输出. 如果不展示则注释本行即可

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-${PROJECT_ROOT}/build}"

CMAKE_BIN_PATH="${CMAKE_BIN_PATH:-cmake}"
PYTHON_BIN_PATH=${PYTHON_BIN_PATH:-/opt/conda/envs/python3.10/bin/python}

echo "============================================"
echo " column-io unit tests on c++ level"
echo " project root : ${PROJECT_ROOT}"
echo " build dir    : ${BUILD_DIR}"
echo "============================================"

# ---- 1. Configure (skip if already configured) ----
if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "[1/3] Configuring CMake ..."
    $CMAKE_BIN_PATH -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -DBUILD_TESTING=ON -DPYTHON_EXECUTABLE="${PYTHON_BIN_PATH}"
else
    echo "[1/3] CMake already configured, skipping configure step."
    echo "      (delete ${BUILD_DIR}/CMakeCache.txt to force re-configure)"
fi

# 某些不规范镜像缺少cuda环境变量:
# export PATH=${CUDA_HOME}/bin:$PATH
# export CUDACXX=${CUDA_HOME}/bin/nvcc

# ---- 2. Build ----
echo "[2/3] Building test targets ..."
$CMAKE_BIN_PATH --build "${BUILD_DIR}" --parallel "$(nproc 2>/dev/null || echo 4)"

# ---- 3. Run all tests via CTest ----
echo "[3/3] Running tests ..."
if ctest --test-dir "${BUILD_DIR}" ${VERBOSE} --output-on-failure --no-tests=error; then
    echo ""
    echo "============================================"
    echo " 全部测试通过 "
    echo "============================================"
    exit 0
else
    echo ""
    echo "============================================"
    echo " 存在测试失败,请检查结果输出 "
    echo "============================================"
    exit -1
fi
