#!/bin/bash

INTERNAL_VERSION=${1}
if [ -z ${INTERNAL_VERSION} ];then
    INTERNAL_VERSION=1
fi
echo "INTERNAL_VERSION=$INTERNAL_VERSION"
INTERNAL_VERSION=${INTERNAL_VERSION} python -u setup.py bdist_wheel
#install
python -m pip install -I --no-deps $(find . -name "*.whl")
