#pragma once

#include <cstddef>
#include <string>

#include "ATen/Utils.h"
#include "c_api/fslib/c_api.h"
#include "torch/extension.h"
namespace recis {
namespace fslib_wrapper {
class FslibFile {
 public:
  FslibFile(TypeFslibFileHandle_C handle);
  size_t pread(void *buf, size_t count, size_t offset);
  size_t write(const char *buf, size_t count);
  fslib::ErrorCode close();
  fslib::ErrorCode flush();
  fslib::ErrorCode getLastError();
  AT_DISALLOW_COPY_AND_ASSIGN(FslibFile);

  ~FslibFile();

 private:
  TypeFslibFileHandle_C handle_;
};
class Wrapper {
 public:
  static void init_alog();
  static void fslib_fs_FileSystem_close();
  static std::string GetErrorString(fslib::ErrorCode ec);
  static FslibFile *OpenFile(const std::string &file, fslib::Flag mode);
  static fslib::ErrorCode isExist(const std::string &file);
  static fslib::ErrorCode ListDir(const std::string &dir,
                                  std::vector<std::string> &files);
  static fslib::ErrorCode remove(const std::string &file);
  static fslib::ErrorCode mkDir(const std::string &dir, bool recursive);
  static fslib::ErrorCode getPathMeta(const std::string &file,
                                      fslib::PathMeta &meta);
  static fslib::ErrorCode rename(const std::string &old_file,
                                 const std::string &new_file);
  static fslib::ErrorCode isDirectory(const std::string &file);
};
}  // namespace fslib_wrapper
}  // namespace recis
