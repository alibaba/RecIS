#!/bin/bash

[ ! -x "$(command -v cmake)" ] && echo "cmake not found, installing..." && pip install -q cmake==3.15.3

[ -z ${CUDA_HOME} ] && echo "[ERROR] env CUDA_HOME not found" && exit 1
[ -z ${CUDACXX} ]   && echo "[WARN] CUDACXX not set, use default nvcc" && export CUDACXX=${CUDA_HOME}/bin/nvcc

INTERNAL_VERSION=${1}
[ -z ${INTERNAL_VERSION} ] && INTERNAL_VERSION=1
echo "INTERNAL_VERSION=$INTERNAL_VERSION"
INTERNAL_VERSION=${INTERNAL_VERSION} python -u setup.py bdist_wheel
#install
python -m pip install -I --no-deps $(find . -name "*.whl")
