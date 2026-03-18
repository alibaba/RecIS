#include "serialize/name.h"

#include <cstdlib>
#include <string>

#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "platform/path.h"
#include "serialize/string_sep.h"
#include "torch/extension.h"
#include "utils/str_util.h"
namespace recis {
namespace serialize {

std::string FullIndexFileName(const std::string &path) {
  return io::JoinPath(path, "index");
}

std::string FullIndexFileNameTmp(const std::string &path) {
  return io::JoinPath(path, "index.tmp");
}

std::string FullJsonFileName(const std::string &path) {
  return io::JoinPath(path, "torch_rank_weights_embs_table_multi_shard.json");
}

std::string FullTorchRankJsonShardName(const int64_t shard_index,
                                       const std::string &path) {
  return io::JoinPath(path,
                      std::to_string(shard_index) +
                          "torch_rank_weights_embs_table_multi_shard.json");
}

std::string FullTorchRankJsonShardNameTmp(const int64_t shard_index,
                                          const std::string &path) {
  return io::JoinPath(path,
                      std::to_string(shard_index) +
                          "torch_rank_weights_embs_table_multi_shard.json.tmp");
}

std::string IndexTorchRankJsonNameTmp(const int64_t shard_index,
                                      const std::string &path) {
  return io::JoinPath(path,
                      std::to_string(shard_index) +
                          "torch_rank_weights_embs_table_multi_shard.json");
}

std::string FullTensorKeyJsonFileShardName(const std::string &path,
                                           int64_t shard_idx) {
  return io::JoinPath(path, std::to_string(shard_idx) + "tensorkey.json");
}

std::string FullTensorKeyJsonFileShardNameTmp(const std::string &path,
                                              int64_t shard_idx) {
  return io::JoinPath(path, std::to_string(shard_idx) + "tensorkey.json.tmp");
}

std::string FullTensorKeyJsonFileName(const std::string &path) {
  return io::JoinPath(path, "tensorkey.json");
}

std::string IndexFileName(const std::string &path, int64_t shard_index) {
  std::string buffer(100, ' ');
  const char *fmt = "ckpt-%07lld.index";
  auto ret = std::snprintf(&buffer[0], buffer.size(), fmt, shard_index);
  while (static_cast<size_t>(ret) >= buffer.size()) {
    buffer.resize(buffer.size() * 2);
    ret = std::snprintf(&buffer[0], buffer.size(), fmt, shard_index);
  }
  return torch::str(path, '/', buffer.c_str());
}

std::string IndexFileNameTmp(const std::string &path, int64_t shard_index) {
  std::string buffer(100, ' ');
  const char *fmt = "ckpt-%07lld.index.tmp";
  auto ret = std::snprintf(&buffer[0], buffer.size(), fmt, shard_index);
  while (static_cast<size_t>(ret) >= buffer.size()) {
    buffer.resize(buffer.size() * 2);
    ret = std::snprintf(&buffer[0], buffer.size(), fmt, shard_index);
  }
  return io::JoinPath(path, buffer.c_str());
}

std::string DataFileName(int64_t thread_idx, int64_t parallel) {
  std::string buffer(100, ' ');
  const char *fmt = "model-%06lld-of-%06lld.safetensors";
  auto ret =
      std::snprintf(&buffer[0], buffer.size(), fmt, thread_idx, parallel);
  while (static_cast<size_t>(ret) >= buffer.size()) {
    buffer.resize(buffer.size() * 2);
    ret = std::snprintf(&buffer[0], buffer.size(), fmt, thread_idx, parallel);
  }
  return buffer.c_str();
}

std::string DataFileNameTmp(int64_t shard_index, int64_t thread_idx) {
  std::string buffer(100, ' ');
  const char *fmt = "ckpt-%07lld-%07lld.data.tmp";
  auto ret =
      std::snprintf(&buffer[0], buffer.size(), fmt, shard_index, thread_idx);
  while (static_cast<size_t>(ret) >= buffer.size()) {
    buffer.resize(buffer.size() * 2);
    ret =
        std::snprintf(&buffer[0], buffer.size(), fmt, shard_index, thread_idx);
  }
  return buffer.c_str();
}

std::string DataJsonNameTmp(int64_t shard_index, int64_t thread_idx) {
  std::string buffer(100, ' ');
  const char *fmt = "ckpt-%07lld-%07lld.json.tmp";
  auto ret =
      std::snprintf(&buffer[0], buffer.size(), fmt, shard_index, thread_idx);
  while (static_cast<size_t>(ret) >= buffer.size()) {
    buffer.resize(buffer.size() * 2);
    ret =
        std::snprintf(&buffer[0], buffer.size(), fmt, shard_index, thread_idx);
  }
  return buffer.c_str();
}

std::string DataJsonName(int64_t shard_index, int64_t thread_idx) {
  std::string buffer(100, ' ');
  const char *fmt = "ckpt-%07lld-%07lld.json";
  auto ret =
      std::snprintf(&buffer[0], buffer.size(), fmt, shard_index, thread_idx);
  while (static_cast<size_t>(ret) >= buffer.size()) {
    buffer.resize(buffer.size() * 2);
    ret =
        std::snprintf(&buffer[0], buffer.size(), fmt, shard_index, thread_idx);
  }
  return buffer.c_str();
}

std::string IndexTorchRankJsonName(const std::string &path, int64_t shard_index,
                                   int64_t thread_idx) {
  std::string buffer(100, ' ');
  const char *fmt = "ckpt-%07lld-%07lld.json";
  auto ret =
      std::snprintf(&buffer[0], buffer.size(), fmt, shard_index, thread_idx);
  while (static_cast<size_t>(ret) >= buffer.size()) {
    buffer.resize(buffer.size() * 2);
    ret =
        std::snprintf(&buffer[0], buffer.size(), fmt, shard_index, thread_idx);
  }
  return io::JoinPath(path, buffer.c_str());
}

std::string HTIdSlotName() { return "id"; }

std::string HTSlotNameEncode(const std::string &shared_name,
                             const std::string &slot_name) {
  return torch::str(shared_name, serialize::StrSep::kIntraNameSep, slot_name);
}

std::string BlockNameEncode(const std::string &tensor_name,
                            const std::string &slice_info,
                            const std::string &sep_symbol) {
  return torch::str(tensor_name, sep_symbol, slice_info);
}

std::string TensorNameFromBlockName(const std::string &block_name) {
  auto tokens = util::string::StrSplit(block_name, StrSep::kInterNameSep);
  TORCH_CHECK(tokens.size() == 2, "block_name: ", block_name, ", type error");
  return tokens[0];
}

std::string SliceInfoFromBlockName(const std::string &block_name) {
  auto tokens = util::string::StrSplit(block_name, StrSep::kInterNameSep);
  TORCH_CHECK(tokens.size() == 2, "block_name: ", block_name, ", type error");
  return tokens[1];
}
bool IsIdName(const std::string &name) { return name == HTIdSlotName(); }

int64_t BlockWriteSize() {
  const char *env_char = std::getenv("RECIS_BLOCK_WRITE_SIZE");
  if (env_char != nullptr) {
    return std::stoll(env_char);
  } else {
    return 1 << 20;
  }
}

int64_t BlockReadSize() {
  const char *env_char = std::getenv("RECIS_BLOCK_READ_SIZE");
  if (env_char != nullptr) {
    return std::stoll(env_char);
  } else {
    return 1 << 20;
  }
}

}  // namespace serialize
}  // namespace recis
