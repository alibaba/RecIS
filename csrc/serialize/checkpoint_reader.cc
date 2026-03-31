#include "checkpoint_reader.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ATen/PTThreadPool.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue_inl.h"
#include "ATen/core/jit_type.h"
#include "ATen/ops/empty.h"
#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/logging_is_not_google_glog.h"
#include "serialize/block_info.h"
#include "serialize/load_bundle.h"
#include "serialize/name.h"
#include "serialize/read_block.h"

namespace recis {
namespace serialize {

CheckpointReader::CheckpointReader(const std::string &path) : path_(path) {}

void CheckpointReader::Init() {
  load_bundle_ = LoadBundle::Make(path_);
  // load_bundle_->Build();
}

std::vector<std::string> CheckpointReader::ListTensors() {
  return load_bundle_->ListTensor();
}

at::Tensor CheckpointReader::LoadTensor(const std::string &tensor_name) {
  TORCH_CHECK(load_bundle_->HasTensor(tensor_name), tensor_name, " not found");
  auto slice_infos = load_bundle_->SliceInfos(tensor_name);
  std::sort(slice_infos.begin(), slice_infos.end());
  const size_t n = slice_infos.size();
  TORCH_CHECK(n > 0, tensor_name, " has no slices");

  if (n == 1) {
    const auto &slice_info = slice_infos[0];
    auto block_info =
        load_bundle_->GetBlockInfo(BlockNameEncode(tensor_name, slice_info));
    auto tensor = at::empty(
        block_info->Shape(),
        at::TensorOptions().device(torch::kCPU).dtype(block_info->Dtype()));
    auto tensor_read_block = TensorReadBlock::Make(
        tensor, block_info,
        load_bundle_->BlockReadFile(BlockNameEncode(tensor_name, slice_info)));
    tensor_read_block->Read();
    return tensor;
  }

  std::vector<int64_t> row_starts(n);
  int64_t row_sum = 0;
  at::TensorOptions out_opts = at::TensorOptions().device(torch::kCPU);
  for (size_t i = 0; i < n; ++i) {
    row_starts[i] = row_sum;
    auto block_info = load_bundle_->GetBlockInfo(
        BlockNameEncode(tensor_name, slice_infos[i]));
    if (i == 0) {
      out_opts = out_opts.dtype(block_info->Dtype());
    }
    row_sum += block_info->Shape()[0];
  }
  auto full_shape = load_bundle_->TensorShape(tensor_name);
  TORCH_CHECK(row_sum == full_shape[0], tensor_name,
              ": slice row sum from blocks (", row_sum,
              ") != TensorShape dim0 (", full_shape[0], ")");
  auto out = at::empty(full_shape, out_opts);

  int64_t parallel = static_cast<int64_t>(std::thread::hardware_concurrency());
  if (parallel < 1) {
    parallel = 1;
  }
  parallel = std::min(parallel, static_cast<int64_t>(n));
  at::PTThreadPool pool(parallel);

  auto tensor_name_holder = std::make_shared<std::string>(tensor_name);
  c10::List<at::intrusive_ptr<at::ivalue::Future>> futures(
      at::FutureType::create(at::NoneType::get()));
  for (size_t i = 0; i < n; ++i) {
    auto future = at::make_intrusive<at::ivalue::Future>(at::NoneType::get());
    pool.run([this, tensor_name_holder, i, out, &slice_infos, &row_starts,
              future]() {
      try {
        const auto &slice_info = slice_infos[i];
        auto block_name = BlockNameEncode(*tensor_name_holder, slice_info);
        auto block_info = load_bundle_->GetBlockInfo(block_name);
        auto dst = out.narrow(0, row_starts[i], block_info->Shape()[0]);
        auto tensor_read_block = TensorReadBlock::Make(
            dst, block_info, load_bundle_->BlockReadFile(block_name));
        tensor_read_block->Read();
        future->markCompleted();
      } catch (std::exception &e) {
        LOG(ERROR) << e.what();
        future->setError(std::current_exception());
      } catch (...) {
        LOG(ERROR) << "unknown exception";
        future->setError(std::current_exception());
      }
    });
    futures.push_back(future);
  }
  c10::collectAll(futures)->waitAndThrow();
  pool.waitWorkComplete();

  return out;
}

std::vector<int64_t> CheckpointReader::TensorShape(
    const std::string &tensor_name) {
  TORCH_CHECK(load_bundle_->HasTensor(tensor_name));
  return load_bundle_->TensorShape(tensor_name);
}

at::Tensor CheckpointReader::TensorType(const std::string &tensor_name) {
  TORCH_CHECK(load_bundle_->HasTensor(tensor_name));
  return torch::empty({}, at::TensorOptions()
                              .dtype(load_bundle_->TensorType(tensor_name))
                              .device(torch::kCPU));
}
}  // namespace serialize
}  // namespace recis
