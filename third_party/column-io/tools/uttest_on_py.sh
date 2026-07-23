#!/bin/bash

# Usage:
#   ./tools/utest_on_py.sh [VERBOSE]
#
# Arguments:
#   VERBOSE       Whether to show test program's stdout (default: -s)
#
# Simplified Version: 
#   python -m pytest -s tests/unit/

set -euo pipefail

VERBOSE=${VERBOSE:-" -s "}  # 是否展示测试程序的std输出. 如果不展示则注释本行即可

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-${PROJECT_ROOT}/build}"

PYTHON_BIN_PATH=${PYTHON_BIN_PATH:-python3.10}
CMAKE_BIN_PATH="${CMAKE_BIN_PATH:-cmake}"

echo "============================================"
echo " column-io unit tests on python level"
echo " project root : ${PROJECT_ROOT}"
echo " build dir    : ${BUILD_DIR}"
echo "============================================"

# ---- 1. Configure (skip if already configured) ----
## 1.1. git config is different between container user and base system user
git config --global --add safe.directory "${PROJECT_ROOT}" 2>/dev/null || true
## 1.2. path of `.aoneci` cannot be recognized by python importlib, must copy it
rm -rf ${BUILD_DIR}/*; mkdir -p ${BUILD_DIR};
cp -r ${PROJECT_ROOT}/.aoneci/scripts ${BUILD_DIR}/

# ---- 2. Build ----
${PYTHON_BIN_PATH} -m pip install -r tests/requirements.txt

# ---- 3. Run all tests via CTest ----
${PYTHON_BIN_PATH} -m pytest ${VERBOSE} \
    -p build.scripts.utest_coverage_py \
    --cov=column_io --cov-report=xml \
    --ignore=tests/integration \
    --ignore=third-party \
    tests/