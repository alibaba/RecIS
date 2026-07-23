#ifndef COLUMN_PYINTERFACE_DATASET_H
#define COLUMN_PYINTERFACE_DATASET_H
#pragma once
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include "column-io/dataset/dataset.h"
#include "column-io/dataset/parallel_dataset.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include "pybind11/pybind11.h"
namespace column {
namespace dataset {

void GlobalInit();

pybind11::object GetNextFromIterator(std::shared_ptr<IteratorBase> iterator, bool row_mode = false);

pybind11::bytes
SerializeIteraterStateToString(std::shared_ptr<IteratorBase> iterator);

void DeserializeIteratorStateFromString(std::shared_ptr<IteratorBase> iterator,
                                        const std::string &msg);

std::shared_ptr<IteratorBase>
MakeIterator(std::shared_ptr<DatasetBase> dataset);

std::shared_ptr<DatasetBase>
MakeListStringDataset(const std::vector<std::string> &inputs);

std::shared_ptr<DatasetBase>
MakeListStringComboDataset(const std::vector<std::vector<std::string>> &inputs);

std::shared_ptr<DatasetBase> MakeOdpsTableColumnDataset(
    const std::vector<std::string> &paths, bool is_compressed,
    int64_t batch_size, const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults);

std::shared_ptr<DatasetBuilder> MakeOdpsTableColumnDatasetBuilder(
    bool is_compressed, int64_t batch_size,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults);

std::pair<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>>
ParseOdpsSchema(const std::vector<std::string> &paths, bool is_compressed,
                const std::unordered_set<std::string> &selected_columns,
                const std::vector<std::string> &hash_features,
                const std::vector<std::string> &dense_columns,
                const std::vector<std::vector<float>> &dense_defaults);

size_t GetTableSize(const std::string &path);
} // namespace dataset
} // namespace column

#endif // COLUMN_PYINTERFACE_DATASET_H