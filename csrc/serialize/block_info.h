#pragma once
#include <torch/extension.h>

#include <array>
#include <string>
#include <vector>

#include "ATen/Utils.h"
#include "ATen/core/ivalue.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/hashtable.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "torch/arg.h"
#include "torch/types.h"
namespace recis {
namespace serialize {
struct BlockInfo : public torch::intrusive_ptr_target {
 public:
  BlockInfo() = default;
  static at::intrusive_ptr<BlockInfo> Make();
  void DecodeFromJson(nlohmann::json::object_t &json);
  TORCH_ARG(torch::Dtype, Dtype);
  TORCH_ARG(std::vector<int64_t>, Shape);
  TORCH_ARG(int64_t, OffsetBeg);
  TORCH_ARG(int64_t, OffsetEnd);
  TORCH_ARG(std::string, DebugInfo);

 public:
  int64_t Size();
  AT_DISALLOW_COPY_AND_ASSIGN(BlockInfo);
};
}  // namespace serialize
}  // namespace recis
