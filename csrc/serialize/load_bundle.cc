#include "serialize/load_bundle.h"

#include <memory>
#include <string>

#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/string_view.h"
#include "platform/env.h"
#include "platform/filesystem.h"
#include "platform/status.h"
#include "serialize/index_info.h"
#include "serialize/name.h"
#include "serialize/table_reader.h"
namespace recis {
namespace serialize {

at::intrusive_ptr<LoadBundle> LoadBundle::Make(const std::string &path) {
  auto obj = at::make_intrusive<LoadBundle>();
  obj->path_ = path;
  obj->Build();
  return obj;
}

void LoadBundle::Build() {
  // part_name -> model tensor name
  GetTensorNameMap();
  // part_name index info
  BuildIndexInfo();
  readers_.resize(index_info_->FileNum());
}

bool LoadBundle::HasTensor(const std::string &tensor_name) {
  return index_info_->HashTensor(tensor_name);
}

std::vector<std::string> LoadBundle::ListTensor() {
  return index_info_->ListTensor();
}

std::vector<int64_t> LoadBundle::TensorShape(const std::string &tensor_name) {
  std::vector<int64_t> shape;
  for (const auto &slice_info : SliceInfos(tensor_name)) {
    auto block_info = GetBlockInfo(BlockNameEncode(tensor_name, slice_info));
    if (shape.empty()) {
      shape = block_info->Shape();
    } else {
      shape[0] += block_info->Shape()[0];
    }
  }
  return shape;
}

at::ScalarType LoadBundle::TensorType(const std::string &tensor_name) {
  auto slice_info = SliceInfos(tensor_name)[0];
  return GetBlockInfo(BlockNameEncode(tensor_name, slice_info))->Dtype();
}

const std::vector<std::string> LoadBundle::SliceInfos(
    const std::string &tensor_name) {
  return index_info_->SliceInfoOfTensor(tensor_name);
}

at::intrusive_ptr<BlockInfo> LoadBundle::GetBlockInfo(
    const std::string &block_name) {
  int64_t file_index = index_info_->FileIndexOfBlock(block_name);
  TryInitReader(file_index);
  auto file = readers_[file_index];
  return file->BlockInfoOfBlock(block_name);
}

at::intrusive_ptr<TableReader> LoadBundle::BlockReadFile(
    const std::string &block_name) {
  int64_t file_index = index_info_->FileIndexOfBlock(block_name);
  TryInitReader(file_index);
  return readers_[file_index];
}

void LoadBundle::GetTensorNameMap() {
  auto middle_json_file = FullTensorKeyJsonFileName(path_);
  TORCH_CHECK(Env::Default()->FileExists(middle_json_file).ok(), "[",
              middle_json_file, "] not found");
  uint64_t file_size = 0;
  RECIS_STATUS_COND(Env::Default()->GetFileSize(middle_json_file, &file_size));
  std::unique_ptr<RandomAccessFile> file;
  RECIS_STATUS_COND(
      Env::Default()->NewRandomAccessFile(middle_json_file, &file));
  std::string info_buf;
  info_buf.resize(file_size);
  torch::string_view ret_view;
  RECIS_STATUS_COND(file->Read(0, file_size, &ret_view, info_buf.data()));
  tensor_name_map_ = nlohmann::json::parse(info_buf);
}

void LoadBundle::BuildIndexInfo() {
  auto index_file = FullIndexFileName(path_);
  TORCH_CHECK(Env::Default()->FileExists(index_file).ok(), "[", index_file,
              "] not found");
  uint64_t file_size;
  Env::Default()->GetFileSize(index_file, &file_size);
  std::unique_ptr<RandomAccessFile> file;
  RECIS_STATUS_COND(Env::Default()->NewRandomAccessFile(index_file, &file));
  std::string info_buf;
  info_buf.resize(file_size);
  torch::string_view ret_view;
  RECIS_STATUS_COND(file->Read(0, file_size, &ret_view, info_buf.data()));

  index_info_ = IndexInfo::Make();
  index_info_->SetTensorNameMap(tensor_name_map_);
  index_info_->Deserialize(info_buf);
}

void LoadBundle::TryInitReader(int64_t file_index) {
  if (readers_[file_index] == nullptr) {
    readers_[file_index] = TableReader::Make(
        path_, index_info_->FileNameByIndex(file_index), tensor_name_map_);
  }
}
}  // namespace serialize
}  // namespace recis
