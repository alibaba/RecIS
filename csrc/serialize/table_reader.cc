#include "serialize/table_reader.h"

#include <memory>
#include <string>

#include "c10/util/StringUtil.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/string_view.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "platform/env.h"
#include "platform/filesystem.h"
#include "platform/path.h"
#include "platform/status.h"
#include "serialize/block_info.h"
namespace recis {
namespace serialize {
at::intrusive_ptr<TableReader> TableReader::Make(
    const std::string &dir_name, const std::string &base_name,
    const std::unordered_map<std::string, std::string> &tensor_name_map_) {
  auto obj = at::make_intrusive<TableReader>();
  obj->offset_ = sizeof(int64_t);
  obj->path_ = io::JoinPath(dir_name, base_name);
  obj->SetTensorNameMap(tensor_name_map_);
  obj->LoadMeta();
  return obj;
}

void TableReader::SetTensorNameMap(
    const std::unordered_map<std::string, std::string> &tensor_name_map) {
  tensor_name_map_ = &tensor_name_map;
}

void TableReader::LoadMeta() {
  uint64_t meta_size;
  torch::string_view result;
  std::unique_ptr<RandomAccessFile> file;
  RECIS_STATUS_COND(Env::Default()->NewRandomAccessFile(path_, &file));
  RECIS_STATUS_COND(
      file->Read(0, sizeof(uint64_t), &result, (char *)(&meta_size)));
  std::string meta_str;
  meta_str.resize(meta_size);
  RECIS_STATUS_COND(file->Read(offset_, meta_size, &result, meta_str.data()));
  offset_ += meta_size;
  auto json = nlohmann::json::parse(meta_str);
  for (const auto &kv : json.items()) {
    auto block_name = kv.key();
    if ((nullptr != tensor_name_map_) &&
        (tensor_name_map_->count(block_name))) {
      block_name = tensor_name_map_->at(block_name);
    }
    auto block_info = BlockInfo::Make();
    nlohmann::json::object_t obj_json = kv.value();
    block_info->DecodeFromJson(obj_json);
    block_info->OffsetBeg(block_info->OffsetBeg() + offset_);
    block_info->OffsetEnd(block_info->OffsetEnd() + offset_);
    block_info->DebugInfo(block_name);
    block_infos_[block_name] = block_info;
  }
}

at::intrusive_ptr<BlockInfo> TableReader::BlockInfoOfBlock(
    const std::string &block_info) {
  return block_infos_.at(block_info);
}

std::unique_ptr<RandomAccessFile> TableReader::File() {
  std::unique_ptr<RandomAccessFile> file;
  RECIS_STATUS_COND(Env::Default()->NewRandomAccessFile(path_, &file));
  return file;
}

}  // namespace serialize
}  // namespace recis
