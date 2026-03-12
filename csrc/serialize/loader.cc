#include "serialize/loader.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#include "ATen/PTThreadPool.h"
#include "ATen/Parallel.h"
#include "ATen/core/Dict.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/function.h"
#include "c10/core/thread_pool.h"
#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "c10/util/flat_hash_map.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/string_view.h"
#include "embedding/hashtable.h"
#include "platform/env.h"
#include "platform/file_statistics.h"
#include "platform/filesystem.h"
#include "platform/status.h"
#include "serialize/ht_read_collection.h"
#include "serialize/load_bundle.h"
#include "serialize/load_info.h"
#include "serialize/load_internal.h"
#include "serialize/load_summary.h"
#include "serialize/name.h"
#include "serialize/read_block.h"
#include "utils/str_util.h"
namespace recis {
namespace serialize {

Loader::Loader(const std::string &path, int64_t parallel,
               torch::Dict<std::string, HashTablePtr> hts_to_load,
               torch::Dict<std::string, at::Tensor> tensors_to_load)
    : parallel_(parallel), path_(path) {
  for (auto &kv : hts_to_load) {
    hts_to_load_[kv.key()] = kv.value();
  }
  for (auto &kv : tensors_to_load) {
    tensors_to_load_[kv.key()] = kv.value();
  }
}

std::string Loader::DefaultLoadInfo() {
  LoadInfo load_info;
  for (auto &kv : hts_to_load_) {
    load_info.Append(kv.second);
  }
  for (auto &kv : tensors_to_load_) {
    load_info.Append(kv.first, kv.second);
  }
  return load_info.Serialize();
}

std::tuple<at::intrusive_ptr<LoadSummary>, int64_t> Loader::Load(
    const std::string &load_info) {
  LoadInfo load_info_obj;
  load_info_obj.Deserialize(load_info);

  auto load_bundle = LoadBundle::Make(path_);
  auto loader_internal = LoaderInternal::Make(
      load_info_obj, load_bundle, hts_to_load_, tensors_to_load_, parallel_);

  int64_t load_size = 0;
  loader_internal->Load(load_size);
  return std::make_tuple(loader_internal->GetLoadSummary(), load_size);
}

Loader::~Loader() {}
}  // namespace serialize

}  // namespace recis
