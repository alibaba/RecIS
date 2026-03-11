#include "platform/fslib/fslib_file_system.h"

#include <errno.h>

#include <cstring>
#include <deque>
#include <mutex>

#include "c10/util/string_view.h"
#include "fslib/common/common_type.h"
#include "platform/env.h"
#include "platform/fslib/wrap/fslib_wrapper.h"
#include "platform/path.h"
#include "platform/status.h"

namespace recis {

namespace {
std::once_flag once_flag;
std::once_flag atexit_flag;

void ClosePangu() {
  sleep(2);
  atexit(fslib_wrapper::Wrapper::fslib_fs_FileSystem_close);
}
}  // namespace

FslibTfFileSystem::FslibTfFileSystem() {
  recis::fslib_wrapper::Wrapper::init_alog();
}

#define CHECK_FS_ERROR_CODE(x, err_msg) \
  if (x == fslib::ErrorCode::EC_OK) {   \
    return Status::OK();                \
  }                                     \
  return Status(Code::INTERNAL, torch::str("error_code is: ", x));

class FslibRandomAccessFile : public RandomAccessFile {
 public:
  FslibRandomAccessFile(const std::string &file_name)
      : file_name_(file_name),
        file_(nullptr),
        chunck_(nullptr),
        chunck_offset_(0xFF),
        chunck_size_(0) {
    uint64_t chunk_size = config_chunck_size_;
    chunck_.reset(new char[chunk_size]);

#ifdef TF_ENABLE_TRACING
    DEFINE_READ_FILE_NETWORK_METRIC(dfs);
    DEFINE_READ_FILE_NETWORK_METRIC_GROUP(dfs, file_name);
#endif
  }

  ~FslibRandomAccessFile() override {
    if (file_) {
      delete file_;
      file_ = nullptr;
    }
  }

  Status Read(uint64_t offset, size_t n, torch::string_view *result,
              char *scratch) const override {
    std::lock_guard<std::mutex> lock(mu_);
    char *dst = scratch;
    Status s;
    if (file_ == nullptr) {
      OpenFile();
    }
    if (file_ == nullptr) {
      return Status(Code::INTERNAL,
                    torch::str(file_name_, "[Fslib] open failed"));
    }

    while (n > 0 && s.ok()) {
      s = GetChunck(offset);
      if (!s.ok()) {
        break;
      }
      int64_t r = std::min((int64_t)(chunck_size_ - (offset - chunck_offset_)),
                           (int64_t)n);
      if (r > 0) {
        std::memcpy(dst, chunck_.get() + (offset - chunck_offset_), r);
        dst += r;
        n -= r;
        offset += r;
      } else if (r <= 0) {
        s = Status(Code::OUT_OF_RANGE,
                   "[Fslib] Read less bytes than requested ");
      }
    }
    *result = torch::string_view(scratch, dst - scratch);
    return s;
  }

  Status GetChunck(size_t offset) const {
    Status s;
    char *dst = chunck_.get();
    size_t chunck_offset = offset / config_chunck_size_ * config_chunck_size_;
    if (chunck_offset_ == chunck_offset &&
        chunck_size_ == config_chunck_size_) {
      return Status::OK();
    }
    chunck_offset_ = chunck_offset;
    chunck_size_ = 0;
    size_t n = config_chunck_size_;
    bool eof_retried = false;
    while (n > 0 && s.ok()) {
      // We lock inside the loop rather than outside so we don't block other
      // concurrent readers.
      // mutex_lock lock(mu_);

#ifdef TF_ENABLE_TRACING
      uint64_t start = GetMicroTimeStamp();
#endif

      ssize_t r = file_->pread(dst, n, chunck_offset);

#ifdef TF_ENABLE_TRACING
      uint64_t end = GetMicroTimeStamp();
      read_qps_metric_->update(1);
      read_latency_metric_->update(end - start);
      read_bytes_metric_->update(r);
      read_metric_group_->Update(r, end - start);
#endif

      if (r > 0) {
        dst += r;
        n -= r;
        chunck_offset += r;
      } else if (!eof_retried && r <= 0) {
        if (file_ != nullptr) {
          fslib::ErrorCode ec = file_->close();
          if (ec != fslib::ErrorCode::EC_OK) {
            s = Status(Code::INTERNAL,
                       torch::str(file_name_, "[Fslib] close file failed"));
            LOG(ERROR) << "[Fslib] close file failed: " << file_name_;
          }
        }
        // file_ = fslib::fs::FileSystem::openFile(file_name_, fslib::READ);
        file_ =
            recis::fslib_wrapper::Wrapper::OpenFile(file_name_, fslib::READ);
        if (file_ == nullptr) {
          s = Status(Code::INTERNAL,
                     torch::str(file_name_, "[Fslib] open file failed"));
          break;
        }
        eof_retried = true;
      } else if (eof_retried && r == 0) {
        break;
      } else if (errno == EINTR || errno == EAGAIN) {
        // hdfsPread may return EINTR too. Just retry.
      } else {
        s = Status(Code::INTERNAL,
                   torch::str(file_name_,
                              "[Fslib] read file failed, already retried"));
        break;
      }
    }
    chunck_size_ = chunck_offset - chunck_offset_;
    return s;
  }

  Status OpenFile() const {
    // file_ = fslib::fs::FileSystem::openFile(file_name_, fslib::READ);
    file_ = recis::fslib_wrapper::Wrapper::OpenFile(file_name_, fslib::READ);
    fslib::ErrorCode ec = file_->getLastError();
    if (ec == fslib::ErrorCode::EC_NOENT || ec == fslib::ErrorCode::EC_ISDIR) {
      return Status(Code::INTERNAL,
                    torch::str("[Fslib] open file for read failed, file:",
                               file_name_, ", error:", ec));
    }
    std::call_once(atexit_flag, ClosePangu);
    return Status::OK();
  }

 private:
  mutable std::mutex mu_;
  std::string file_name_;
  mutable recis::fslib_wrapper::FslibFile *file_;

  mutable std::unique_ptr<char[]> chunck_;
  mutable size_t chunck_offset_;
  mutable size_t chunck_size_;

  size_t config_chunck_size_ = 1 << 22;

#ifdef TF_ENABLE_TRACING
  DECLARE_READ_FILE_NETWORK_METRIC();
  DECLARE_READ_FILE_NETWORK_METRIC_GROUP();
#endif
};

class FslibWritableFile : public WritableFile {
 public:
  explicit FslibWritableFile(const std::string &fname, bool append)
      : file_name_(fname), append_(append) {
#ifdef TF_ENABLE_TRACING
    DEFINE_WRITE_FILE_NETWORK_METRIC(dfs);
    DEFINE_WRITE_FILE_NETWORK_METRIC_GROUP(dfs, fname);
#endif
  }

  ~FslibWritableFile() {
    delete file_;
    file_ = nullptr;
  }

  Status Append(torch::string_view data) override {
    if (file_ == nullptr) {
      std::lock_guard<std::mutex> lock(mu_);
      if (file_ == nullptr) {
        OpenFile();
      }
    }
    if (!file_) {
      return Status(Code::INTERNAL, torch::str("[Fslib] open ", file_name_,
                                               " for write failed"));
    }
    const size_t size = data.size();

#ifdef TF_ENABLE_TRACING
    uint64_t start = GetMicroTimeStamp();
#endif

    ssize_t writeLen = file_->write(data.data(), size);

#ifdef TF_ENABLE_TRACING
    uint64_t end = GetMicroTimeStamp();
    write_qps_metric_->update(1);
    write_latency_metric_->update(end - start);
    write_bytes_metric_->update(writeLen);
    write_metric_group_->Update(writeLen, end - start);
#endif

    if (size_t(writeLen) != size) {
      return Status(Code::DATA_LOSS,
                    torch::str("[Fslib] write ", writeLen, " bytes, expected ",
                               size, "bytes"));
    }
    return Status::OK();
  }

  Status Close() override {
    auto s = Flush();
    if (!s.ok()) {
      return s;
    }
    if (!file_) {
      return Status::OK();
    }
    fslib::ErrorCode ec = file_->close();
    if (ec != fslib::EC_OK) {
      auto error_str =
          recis::fslib_wrapper::Wrapper::GetErrorString(file_->getLastError());
      return Status(Code::INTERNAL,
                    torch::str("[Fslib] close failed, error: ", error_str));
    }
    return Status::OK();
  }

  Status Flush() override {
    if (!file_) {
      return Status::OK();
    }
    fslib::ErrorCode ec = file_->flush();
    if (ec != fslib::EC_OK) {
      auto error_str =
          recis::fslib_wrapper::Wrapper::GetErrorString(file_->getLastError());
      return Status(Code::INTERNAL,
                    torch::str("[Fslib] flush failed, error: ", error_str));
    }
    return Status::OK();
  }

  Status Sync() override { return Flush(); }

  Status OpenFile() const {
    fslib::Flag mode = append_ ? fslib::APPEND : fslib::WRITE;
    // file_ = fslib::fs::FileSystem::openFile(file_name_, mode);
    file_ = recis::fslib_wrapper::Wrapper::OpenFile(file_name_, mode);
    std::call_once(atexit_flag, ClosePangu);
    return Status::OK();
  }

 private:
  bool append_ = false;
  std::mutex mu_;
  std::string file_name_;
  // mutable fslib::fs::File *file_ = nullptr;
  mutable recis::fslib_wrapper::FslibFile *file_ = nullptr;

#ifdef TF_ENABLE_TRACING
  DECLARE_WRITE_FILE_NETWORK_METRIC();
  DECLARE_WRITE_FILE_NETWORK_METRIC_GROUP();
#endif
};

Status FslibTfFileSystem::NewRandomAccessFile(
    const std::string &fname, std::unique_ptr<RandomAccessFile> *result) {
  auto file = new FslibRandomAccessFile(TranslateName(fname));
  result->reset(file);
  auto s = file->OpenFile();
  return s;
}

Status FslibTfFileSystem::NewWritableFile(
    const std::string &fname, std::unique_ptr<WritableFile> *result) {
  auto file = new FslibWritableFile(TranslateName(fname), false);
  result->reset(file);
  return file->OpenFile();
}

Status FslibTfFileSystem::NewAppendableFile(
    const std::string &fname, std::unique_ptr<WritableFile> *result) {
  auto file = new FslibWritableFile(TranslateName(fname), true);
  result->reset(file);
  return file->OpenFile();
}

Status FslibTfFileSystem::NewReadOnlyMemoryRegionFromFile(
    const std::string &fname, std::unique_ptr<ReadOnlyMemoryRegion> *result) {
  return Status(
      Code::INTERNAL,
      torch::str("HDFS/zfs/... does not support ReadOnlyMemoryRegion"));
}

Status FslibTfFileSystem::FileExists(const std::string &fname) {
  // fslib::ErrorCode ec = fslib::fs::FileSystem::isExist(fname);
  fslib::ErrorCode ec = recis::fslib_wrapper::Wrapper::isExist(fname);
  std::call_once(atexit_flag, ClosePangu);
  if (ec == fslib::ErrorCode::EC_TRUE) {
    return Status::OK();
  }
  if (ec == fslib::ErrorCode::EC_FALSE) {
    return Status(Code::NOT_FOUND, torch::str(fname, "not found."));
  }
  return Status(Code::INTERNAL,
                torch::str("[Fslib] check file: ", fname,
                           " exists failed error code is ", ec));
}

Status FslibTfFileSystem::GetChildren(const std::string &dir,
                                      std::vector<std::string> *result) {
  // fslib::ErrorCode ec = fslib::fs::FileSystem::listDir(dir, *result);
  fslib::ErrorCode ec = recis::fslib_wrapper::Wrapper::ListDir(dir, *result);
  std::call_once(atexit_flag, ClosePangu);
  CHECK_FS_ERROR_CODE(ec, "[Fslib] get dir children failed")
}

Status FslibTfFileSystem::DeleteFile(const std::string &fname) {
  // fslib::ErrorCode ec = fslib::fs::FileSystem::remove(fname);
  fslib::ErrorCode ec = recis::fslib_wrapper::Wrapper::remove(fname);
  std::call_once(atexit_flag, ClosePangu);
  CHECK_FS_ERROR_CODE(ec, "[Fslib] delete file failed, file: " + fname)
}

Status FslibTfFileSystem::CreateDir(const std::string &name) {
  // fslib::ErrorCode ec = fslib::fs::FileSystem::mkDir(name, true);
  fslib::ErrorCode ec = recis::fslib_wrapper::Wrapper::mkDir(name, true);
  std::call_once(atexit_flag, ClosePangu);
  if (ec == fslib::ErrorCode::EC_EXIST) {
    return Status::OK();
  }
  CHECK_FS_ERROR_CODE(ec, "[Fslib] create dir failed, dir: " + name)
}

Status FslibTfFileSystem::DeleteDir(const std::string &name) {
  // fslib::ErrorCode ec = fslib::fs::FileSystem::remove(name);
  fslib::ErrorCode ec = recis::fslib_wrapper::Wrapper::remove(name);
  std::call_once(atexit_flag, ClosePangu);
  CHECK_FS_ERROR_CODE(ec, "[Fslib] delete dir failed, dir: " + name)
}

Status FslibTfFileSystem::GetFileSize(const std::string &fname,
                                      uint64_t *size) {
  fslib::PathMeta path_meta;
  // fslib::ErrorCode ec = fslib::fs::FileSystem::getPathMeta(fname, path_meta);
  fslib::ErrorCode ec =
      recis::fslib_wrapper::Wrapper::getPathMeta(fname, path_meta);
  std::call_once(atexit_flag, ClosePangu);
  if (ec != fslib::ErrorCode::EC_OK) {
    return Status(Code::INTERNAL,
                  torch::str("[Fslib] get file meta failed, file: ", fname,
                             "error code is: ", ec));
  }
  *size = static_cast<uint64_t>(path_meta.length);
  return Status::OK();
}

Status FslibTfFileSystem::RenameFile(const std::string &src,
                                     const std::string &target) {
  if (FileExists(target) == Status::OK()) {
    // AUTIL_LOG(INFO, "[Fslib] target path exist, delete it before rename");
    // fslib::ErrorCode ec = fslib::fs::FileSystem::remove(target);
    fslib::ErrorCode ec = recis::fslib_wrapper::Wrapper::remove(target);
    std::call_once(atexit_flag, ClosePangu);
    if (ec != fslib::ErrorCode::EC_OK) {
      return Status(
          Code::INTERNAL,
          torch::str("[Fslib] delete target path failed, target: " + target));
    }
  }
  // fslib::ErrorCode ec = fslib::fs::FileSystem::rename(src, target);
  fslib::ErrorCode ec = recis::fslib_wrapper::Wrapper::rename(src, target);
  std::call_once(atexit_flag, ClosePangu);
  CHECK_FS_ERROR_CODE(ec, "[Fslib] rename file failed")
}

Status FslibTfFileSystem::Stat(const std::string &fname, FileStatistics *stat) {
  fslib::PathMeta path_meta;
  // fslib::ErrorCode ec = fslib::fs::FileSystem::getPathMeta(fname, path_meta);
  fslib::ErrorCode ec =
      recis::fslib_wrapper::Wrapper::getPathMeta(fname, path_meta);
  std::call_once(atexit_flag, ClosePangu);
  if (ec != fslib::ErrorCode::EC_OK) {
    return Status(Code::INTERNAL,
                  torch::str("[Fslib] fs get file meta failed, file: ", fname,
                             "error code is: ", ec));
  }
  stat->is_directory = !path_meta.isFile;
  stat->length = path_meta.length;
  stat->mtime_nsec = path_meta.lastModifyTime;

  return Status::OK();
}

Status FslibTfFileSystem::GetMatchingPaths(const std::string &pattern,
                                           std::vector<std::string> *results) {
  results->clear();
  // Find the fixed prefix by looking for the first wildcard.
  const std::string &fixed_prefix =
      pattern.substr(0, pattern.find_first_of("*?[\\"));
  std::vector<std::string> all_files;
  torch::string_view dirsp = io::Dirname(fixed_prefix);
  std::string dir(dirsp.begin(), dirsp.end());

  // Setup a BFS to explore everything under dir.
  std::deque<std::string> dir_q;
  dir_q.push_back(dir);
  Status ret;  // Status to return.
  while (!dir_q.empty()) {
    std::string current_dir = dir_q.front();
    dir_q.pop_front();
    std::vector<std::string> children;
    Status s = GetChildren(current_dir, &children);
    ret.Update(s);
    if (children.empty()) {
      continue;
    }
    for (const auto &child : children) {
      const std::string &child_path = io::JoinPath(current_dir, child);
      // In case the child_path doesn't start with the fixed_prefix then
      // we don't need to explore this path.
      if (child_path.compare(0, fixed_prefix.size(), fixed_prefix) != 0) {
        continue;
      } else {
        // if (fslib::fs::FileSystem::isDirectory(child_path) ==
        //     fslib::ErrorCode::EC_TRUE) {
        //   dir_q.push_back(child_path);
        // }
        if (recis::fslib_wrapper::Wrapper::isExist(child_path) ==
            fslib::ErrorCode::EC_TRUE) {
          dir_q.push_back(child_path);
        }
        all_files.push_back(child_path);
      }
    }
  }

  // Match all obtained files to the input pattern.
  for (const auto &f : all_files) {
    if (Env::Default()->MatchPath(f, pattern)) {
      results->push_back(f);
    }
  }
  return ret;
}

std::string FslibTfFileSystem::TranslateName(const std::string &name) const {
  return name;
}

REGISTER_FILE_SYSTEM("dfs", FslibTfFileSystem);
}  // namespace recis
