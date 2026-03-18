#include "serialize/save_bundle.h"

#include <chrono>
#include <exception>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>

#include "ATen/PTThreadPool.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/jit_type.h"
#include "c10/core/DeviceGuard.h"
#include "c10/cuda/CUDAFunctions.h"
#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/irange.h"
#include "c10/util/logging_is_not_google_glog.h"
#include "c10/util/string_view.h"
#include "platform/env.h"
#include "platform/filesystem.h"
#include "platform/path.h"
#include "platform/status.h"
#include "serialize/index_info.h"
#include "serialize/name.h"
#include "serialize/table_writer.h"
#include "utils/str_util.h"
namespace recis {
namespace serialize {

void SaveBundle::InitTypeMap() {
  type_map_["F64"] = "float64";
  type_map_["F32"] = "float32";
  type_map_["F16"] = "float16";
  type_map_["BF16"] = "bfloat16";
  type_map_["I64"] = "int64";
  type_map_["U64"] = "uint64";
  type_map_["I32"] = "int32";
  type_map_["U32"] = "uint32";
  type_map_["I16"] = "int16";
  type_map_["U16"] = "uint16";
  type_map_["I8"] = "int8";
  type_map_["U8"] = "uint8";
  type_map_["BOOL"] = "bool";
}

at::intrusive_ptr<SaveBundle> SaveBundle::Make(const std::string &path,
                                               int64_t parallel,
                                               int64_t shard_index,
                                               int64_t shard_num) {
  auto ret = at::make_intrusive<SaveBundle>();
  TORCH_CHECK(parallel > 0, "parallel must > 0");
  TORCH_CHECK(shard_index >= 0, "shard_index must >= 0")
  TORCH_CHECK(shard_num >= 0, "shard_num must >= 0")
  TORCH_CHECK(shard_num > shard_index, "shard_num must > shard_index");
  ret->shard_index_ = shard_index;
  ret->shard_num_ = shard_num;
  ret->parallel_ = parallel;
  ret->path_ = path;
  ret->table_index_ = 0;
  ret->table_writers_.resize(parallel);
  ret->table_file_names_.resize(parallel);
  ret->table_json_names_.resize(parallel);
  ret->index_info_ = IndexInfo::Make();
  for (int i : c10::irange(parallel)) {
    ret->table_writers_[i] = TableWriter::Make(
        path, DataFileNameTmp(shard_index, i), DataJsonNameTmp(shard_index, i));
    ret->table_file_names_[i] =
        DataFileName(shard_index * parallel + i, parallel * shard_num);
    ret->table_json_names_[i] = DataJsonName(shard_index, i);
  }
  ret->InitTypeMap();
  return ret;
}

void SaveBundle::AddBlock(at::intrusive_ptr<WriteBlock> write_block) {
  // change @ -> .
  std::string tensor_name = write_block->TensorName();
  auto new_tensor_name =
      util::string::Replace(tensor_name, TensorSymbolAt(), TensorSymbolDot());

  int64_t table_index = IncTableIndex();
  table_writers_[table_index]->AppendWriteBlock(write_block);

  // rename it by part
  if (!write_block->IsDense()) {
    new_tensor_name += (TensorKeyPart() + std::to_string(shard_index_) +
                        TensorSymbolULine() + std::to_string(shard_num_));

    if (0 == tensor_name_map_.count(new_tensor_name)) {
      tensor_name_map_[new_tensor_name] =
          BlockNameEncode(tensor_name, write_block->SliceInfo());
    } else {
      LOG(WARNING) << "duplicate key ";
    }
    write_block->SetTensorName(new_tensor_name);
    index_info_->Append(write_block->TensorName(), "", "",
                        table_file_names_[table_index]);
  } else {
    index_info_->Append(write_block->TensorName(), write_block->SliceInfo(),
                        TensorSymbolSuperScript(),
                        table_file_names_[table_index]);
  }
}

void SaveBundle::Save() {
  LOG(WARNING) << "Save Meta";
  SaveMeta();
  LOG(WARNING) << "Save Table";
  SaveTables();
  MergeJsonInfo();
}

void SaveBundle::MergeMetaInfo() {
  if (shard_index_ != 0) return;
  LOG(WARNING) << "merge meta info";
  std::set<std::string> meta_files;
  for (int i : c10::irange(shard_num_)) {
    auto meta_file = IndexFileName(path_, i);
    meta_files.insert(meta_file);
  }
  LOG(WARNING) << "meta files: " << meta_files;
  auto index_info = IndexInfo::Make();
  while (!meta_files.empty()) {
    std::set<std::string> load_files;
    for (auto meta_file : meta_files) {
      if (Env::Default()->FileExists(meta_file).ok()) {
        uint64_t file_size = 0;
        RECIS_STATUS_COND(Env::Default()->GetFileSize(meta_file, &file_size));
        torch::string_view result;
        std::string content;
        content.resize(file_size);
        {
          std::unique_ptr<RandomAccessFile> file;
          RECIS_STATUS_COND(
              Env::Default()->NewRandomAccessFile(meta_file, &file));
          RECIS_STATUS_COND(file->Read(0, file_size, &result, &content[0]));
        }
        RECIS_STATUS_COND(Env::Default()->DeleteFile(meta_file));
        auto tmp_index_info = IndexInfo::Make();
        tmp_index_info->Deserialize(content);
        index_info->MergeFrom(*tmp_index_info);
        load_files.insert(meta_file);
      }
    }
    for (auto file_name : load_files) {
      meta_files.erase(file_name);
    }
    if (!meta_files.empty()) {
      LOG(WARNING) << "waiting for: " << meta_files;
      std::this_thread::sleep_for(std::chrono::milliseconds(100000));
    }
  }
  auto msg = index_info->Serialize();
  auto filename = FullIndexFileNameTmp(path_);
  std::unique_ptr<WritableFile> file;
  RECIS_STATUS_COND(Env::Default()->NewWritableFile(filename, &file));
  file->Append(msg);
  file->Flush();
  file->Close();
  RECIS_STATUS_COND(Env::Default()->TransactionRenameFile(
      filename, FullIndexFileName(path_)));
}

void SaveBundle::SaveMeta() {
  auto save_file = [](const std::string &file_name, const std::string &msg) {
    std::unique_ptr<WritableFile> file;
    RECIS_STATUS_COND(Env::Default()->NewWritableFile(file_name, &file));
    file->Append(msg);
    file->Flush();
    file->Close();
  };

  {
    auto filename = IndexFileNameTmp(path_, shard_index_);
    auto msg = index_info_->Serialize();
    save_file(filename, msg);
    RECIS_STATUS_COND(Env::Default()->TransactionRenameFile(
        filename, IndexFileName(path_, shard_index_)));
  }

  {
    nlohmann::json tensorkey_json = tensor_name_map_;
    auto msg = tensorkey_json.dump(4);
    auto writing_name = FullTensorKeyJsonFileShardNameTmp(path_, shard_index_);
    save_file(writing_name, msg);
    RECIS_STATUS_COND(Env::Default()->TransactionRenameFile(
        writing_name, FullTensorKeyJsonFileShardName(path_, shard_index_)));
  }
}

void SaveBundle::SaveTables() {
  at::PTThreadPool pool(parallel_);
  c10::List<at::intrusive_ptr<at::ivalue::Future>> futures(
      at::FutureType::create(at::NoneType::get()));

  c10::Device device = torch::cuda::is_available()
                           ? c10::Device(c10::kCUDA, at::cuda::current_device())
                           : c10::Device(c10::kCPU);
  for (auto i : c10::irange(parallel_)) {
    auto future = at::make_intrusive<at::ivalue::Future>(at::NoneType::get());
    pool.run([i, device, future, this]() {
      try {
        c10::DeviceGuard device_guard(device);
        table_writers_[i]->SequentialWrite();
        future->markCompleted();
      } catch (std::exception &e) {
        LOG(ERROR) << e.what();
        future->setError(std::current_exception());
      } catch (...) {
        LOG(ERROR) << "unknown exception";
      }
    });
    futures.push_back(future);
  }
  c10::collectAll(futures)->waitAndThrow();
  pool.waitWorkComplete();
  int valid_num = 0;
  for (auto i : c10::irange(parallel_)) {
    if (table_writers_[i]->Empty()) continue;
    valid_num++;
    RECIS_STATUS_COND(Env::Default()->TransactionRenameFile(
        io::JoinPath(path_, table_writers_[i]->FileName()),
        io::JoinPath(path_, table_file_names_[i])));
    RECIS_STATUS_COND(Env::Default()->TransactionRenameFile(
        io::JoinPath(path_, table_writers_[i]->JsonName()),
        io::JoinPath(path_, table_json_names_[i])));
  }

  MergeParallelTorchRankJson(valid_num);
}

void SaveBundle::MergeParallelTorchRankJson(const int valid_num) {
  std::vector<std::string> sub_json_files;
  for (int i = 0; i < valid_num; i++) {
    auto tmp_json_file = IndexTorchRankJsonName(path_, shard_index_, i);
    sub_json_files.push_back(tmp_json_file);
  }
  nlohmann::json combine_data;
  GetSubFileData(sub_json_files, combine_data);
  auto writing_name = FullTorchRankJsonShardNameTmp(shard_index_, path_);
  DumpJsonFile(combine_data, writing_name);
  RECIS_STATUS_COND(Env::Default()->TransactionRenameFile(
      writing_name, FullTorchRankJsonShardName(shard_index_, path_)));
  return;
}

int64_t SaveBundle::IncTableIndex() {
  auto ret = table_index_;
  table_index_++;
  table_index_ = table_index_ % parallel_;
  return ret;
}

int SaveBundle::GenJsonInfo(const std::string &json_file,
                            nlohmann::json &combine_data) {
  if (Env::Default()->FileExists(json_file).ok()) {
    uint64_t file_size = 0;
    while (file_size == 0) {
      RECIS_STATUS_COND(Env::Default()->GetFileSize(json_file, &file_size));
      if (file_size == 0) {
        LOG(ERROR) << "Sub Json File size error, file name:  " << json_file
                   << ", file size: " << file_size;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
    torch::string_view result;
    std::string content;
    content.resize(file_size);
    {
      std::unique_ptr<RandomAccessFile> file;
      RECIS_STATUS_COND(Env::Default()->NewRandomAccessFile(json_file, &file));
      RECIS_STATUS_COND(file->Read(0, file_size, &result, &content[0]));
    }

    auto dict = nlohmann::json::parse(content);
    RECIS_STATUS_COND(Env::Default()->DeleteFile(json_file));
    combine_data.merge_patch(dict);
  } else {
    LOG(ERROR) << "Not Found Sub Json File" << json_file;
    return -1;
  }
  return 0;
}

void SaveBundle::GetSubFileData(const std::vector<std::string> &json_files,
                                nlohmann::json &json_data) {
  std::unordered_map<std::string, int> record_files;
  while (record_files.size() != json_files.size()) {
    for (const auto &json_file : json_files) {
      if (0 == record_files.count(json_file)) {
        if (0 == GenJsonInfo(json_file, json_data)) {
          record_files[json_file] = 1;
        }
      }
    }
    if (record_files.size() != json_files.size()) {
      LOG(ERROR) << "Not all worker save done, wait 10s ...";
      std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    }
  }
}

void SaveBundle::DumpJsonFile(const nlohmann::json &json_data,
                              const std::string &filename) {
  auto msg = json_data.dump(4);
  std::unique_ptr<WritableFile> file;
  RECIS_STATUS_COND(Env::Default()->NewWritableFile(filename, &file));
  file->Append(msg);
  file->Flush();
  file->Close();
}

void SaveBundle::MergeTorchRankJson() {
  std::vector<std::string> sub_json_files;

  for (int i = 0; i < shard_num_; i++) {
    auto tmp_json_file = IndexTorchRankJsonNameTmp(i, path_);
    sub_json_files.push_back(tmp_json_file);
  }

  nlohmann::json combine_data;
  nlohmann::json sub_data;

  GetSubFileData(sub_json_files, sub_data);

  for (auto &[k, v] : sub_data.items()) {
    if (combine_data.find(k) == combine_data.end()) {
      if (v["dense"]) {
        v.erase("data_offsets");
        v["name"] = k;
        v["dtype"] = type_map_[v["dtype"].get<std::string>()];
        v["dimension"] = 0;
        v["is_hashmap"] = false;
        combine_data[k] = v;
      } else if (k.find(NewTensorKeyEmbedding()) != std::string::npos) {
        v.erase("data_offsets");
        v["name"] = k;
        v["dtype"] = type_map_[v["dtype"].get<std::string>()];
        v["dimension"] = v["shape"].back().get<int>();
        std::string hashmap_key =
            util::string::Replace(k, NewTensorKeyEmbedding(), NewTensorKeyId());
        v["hashmap_key"] = hashmap_key;
        v["hashmap_key_dtype"] = "int64";
        v["hashmap_value"] = k;
        v["is_hashmap"] = true;
        combine_data[k] = v;
      }
      // else v will be `id` or `opt`, which are not needed in the meta file
    } else {
      LOG(WARNING) << "Duplicate key found in data: " << k;
    }
  }

  DumpJsonFile(combine_data, FullJsonFileName(path_));

  return;
}

void SaveBundle::MergeTensorKeyJson() {
  std::vector<std::string> sub_json_files;
  for (int i = 0; i < shard_num_; i++) {
    sub_json_files.push_back(FullTensorKeyJsonFileShardName(path_, i));
  }
  nlohmann::json combine_data;

  GetSubFileData(sub_json_files, combine_data);
  DumpJsonFile(combine_data, FullTensorKeyJsonFileName(path_));
}

void SaveBundle::MergeJsonInfo() {
  if (0 != shard_index_) {
    return;
  }
  LOG(WARNING) << "Merge Json Info";
  MergeTorchRankJson();
  MergeTensorKeyJson();
  return;
}

}  // namespace serialize
}  // namespace recis
