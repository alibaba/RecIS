#pragma once
#include <stdint.h>

#include <string>
namespace recis {
namespace serialize {

constexpr const char *NewTensorKeyId() { return ".id"; }
constexpr const char *NewTensorKeyEmbedding() { return ".embedding"; }
constexpr const char *TensorKeyPart() { return "/part_"; }

constexpr const char *TensorSymbolSuperScript() { return "^"; }
constexpr const char *TensorSymbolAt() { return "@"; }
constexpr const char *TensorSymbolDot() { return "."; }
constexpr const char *TensorSymbolULine() { return "_"; }

std::string FullTorchRankJsonShardName(const int64_t shard_index,
                                       const std::string &path);
std::string FullTorchRankJsonShardNameTmp(const int64_t shard_index,
                                          const std::string &path);

std::string IndexTorchRankJsonNameTmp(const int64_t shard_index,
                                      const std::string &path);

std::string FullIndexFileName(const std::string &path);
std::string FullIndexFileNameTmp(const std::string &path);
std::string FullJsonFileName(const std::string &path);
std::string FullTensorKeyJsonFileName(const std::string &path);
std::string FullTensorKeyJsonFileShardName(const std::string &path,
                                           int64_t shard_idx);
std::string FullTensorKeyJsonFileShardNameTmp(const std::string &path,
                                              int64_t shard_idx);
std::string IndexFileName(const std::string &path, int64_t shard_index);
std::string IndexFileNameTmp(const std::string &path, int64_t shard_index);
std::string IndexTorchRankJsonName(const std::string &path, int64_t shard_index,
                                   int64_t thread_idx);
std::string DataFileName(int64_t shard_index, int64_t thread_idx);
std::string DataFileNameTmp(int64_t shard_index, int64_t thread_idx);
std::string DataJsonNameTmp(int64_t shard_index, int64_t thread_idx);
std::string DataJsonName(int64_t shard_index, int64_t thread_idx);

std::string HTIdSlotName();
bool IsIdName(const std::string &);
std::string HTSlotNameEncode(const std::string &shared_name,
                             const std::string &slot_name);
std::string BlockNameEncode(
    const std::string &tensor_name, const std::string &slice_info,
    const std::string &sep_symbol = TensorSymbolSuperScript());
std::string TensorNameFromBlockName(const std::string &block_name);
std::string SliceInfoFromBlockName(const std::string &block_name);
constexpr const char *EmptySliceInfo() { return " "; }

int64_t BlockWriteSize();
int64_t BlockReadSize();

constexpr const char *BlockKeyDtype() { return "dtype"; }
constexpr const char *BlockKeyShape() { return "shape"; }
constexpr const char *BlockKeyOffsets() { return "data_offsets"; }
constexpr const char *BlockKeyMeta() { return "__metadata__"; }
}  // namespace serialize
}  // namespace recis
