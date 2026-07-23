#ifndef COLUMN_PYINTERFACE_CONVERTER_H
#define COLUMN_PYINTERFACE_CONVERTER_H
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include "pybind11/pybind11.h"

namespace column {
namespace py_interface {
pybind11::object CastTensor(Tensor tensor);
pybind11::list CastTensorsToPythonTuples(std::vector<Tensor> tensors, std::vector<size_t>& outputs_row_spliter);
pybind11::object CastTensorToDLPack(Tensor tensor);
} // namespace py_interface
} // namespace column

#endif // COLUMN_PYINTERFACE_CONVERTER_H
