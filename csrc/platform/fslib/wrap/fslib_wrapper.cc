#include "platform/fslib/wrap/fslib_wrapper.h"

#include <cstddef>
#include <string>

#include "c_api/fslib/c_api.h"
namespace recis {
namespace fslib_wrapper {
namespace {
std::vector<std::string> StringArrayToVector(char **str_str) {
  std::vector<std::string> result;
  for (int i = 0; str_str[i] != nullptr; i++) {
    result.push_back(str_str[i]);
  }
  return result;
}

void FreeStringArray(char **str_str) {
  for (int i = 0; str_str[i] != nullptr; i++) {
    free(str_str[i]);
  }
  free(str_str);
}
}  // namespace
size_t FslibFile::pread(void *buf, size_t count, size_t offset) {
  return FslibFileHandle_C_Read(handle_, buf, count, offset);
}

size_t FslibFile::write(const char *buf, size_t count) {
  return FslibFileHandle_C_Write(handle_, buf, count);
}

fslib::ErrorCode FslibFile::close() { return FslibFileHandle_C_Close(handle_); }

fslib::ErrorCode FslibFile::flush() { return FslibFileHandle_C_Flush(handle_); }

fslib::ErrorCode FslibFile::getLastError() {
  return FslibFileHandle_C_GetLastError(handle_);
}

FslibFile::~FslibFile() { FslibFileHandle_C_Destory(handle_); }

FslibFile::FslibFile(TypeFslibFileHandle_C handle) : handle_(handle) {}

void Wrapper::init_alog() { init_alog_C(); }
void Wrapper::fslib_fs_FileSystem_close() { Fslib_Fs_FileSystem_Close_C(); }

std::string Wrapper::GetErrorString(fslib::ErrorCode ec) {
  std::string msg;
  char *str_ptr;
  Fslib_Fs_FileSystem_GetErrorString((void **)&str_ptr, ec);
  std::string ret(str_ptr);
  free(str_ptr);
  return ret;
}

FslibFile *Wrapper::OpenFile(const std::string &file, fslib::Flag mode) {
  TypeFslibFileHandle_C handle;
  Fslib_Fs_FileSystem_OpenFile(file.c_str(), mode, &handle);
  if (handle == nullptr) {
    return nullptr;
  }
  return new FslibFile(handle);
}

fslib::ErrorCode Wrapper::isExist(const std::string &file) {
  return Fslib_Fs_FileSystem_IsExist(file.c_str());
}
fslib::ErrorCode Wrapper::ListDir(const std::string &dir,
                                  std::vector<std::string> &files) {
  char **str_str;
  auto ec = Fslib_Fs_FileSystem_ListDir(dir.c_str(), (void ***)(&str_str));
  files = StringArrayToVector(str_str);
  FreeStringArray(str_str);
  return ec;
}

fslib::ErrorCode Wrapper::remove(const std::string &file) {
  return Fslib_Fs_FileSystem_Remove(file.c_str());
}

fslib::ErrorCode Wrapper::mkDir(const std::string &dir, bool recursive) {
  return Fslib_Fs_FileSystem_MkDir(dir.c_str(), recursive);
}

fslib::ErrorCode Wrapper::getPathMeta(const std::string &file,
                                      fslib::PathMeta &meta) {
  return Fslib_Fs_FileSystem_GetPathMeta(file.c_str(), meta);
}

fslib::ErrorCode Wrapper::rename(const std::string &old_file,
                                 const std::string &new_file) {
  return Fslib_Fs_FileSystem_Rename(old_file.c_str(), new_file.c_str());
}
fslib::ErrorCode Wrapper::isDirectory(const std::string &file) {
  return Fslib_Fs_FileSystem_IsDirectory(file.c_str());
}

}  // namespace fslib_wrapper
}  // namespace recis
