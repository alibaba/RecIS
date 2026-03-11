#include "c_api/fslib/c_api.h"

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "autil/Log.h"
#include "fslib/fs/File.h"
#include "fslib/fs/FileSystem.h"
#include "fslib/fslib.h"

namespace {
char **vectorToStringArray(const std::vector<std::string> &vec) {
  char **result = new char *[vec.size() + 1];
  for (int i = 0; i < vec.size(); ++i) {
    result[i] = strdup(
        vec[i].c_str());  // strdup duplicates the string and allocates memory
  }

  result[vec.size()] = nullptr;  // Null terminate the array

  return result;
}

bool IsFslLogEnabled() {
  const char *env = std::getenv("RECIS_OPEN_FSLIB_LOG");
  return (env != nullptr) && (std::strcmp(env, "1") == 0);
}

void InitAlog() {
  if (IsFslLogEnabled()) {
    const char *default_log_conf = R"magic(alog.rootLogger=INFO, rootAppender
alog.max_msg_len=2048
alog.appender.rootAppender=ConsoleAppender
alog.appender.rootAppender.layout=PatternLayout
alog.appender.rootAppender.layout.LogPattern=[%%d] [%%l] [%%t,%%F -- %%f():%%n] %%m
)magic";
    alog::Configurator::configureLoggerFromString(default_log_conf);
    AUTIL_ROOT_LOG_SETLEVEL(DEBUG);
  }
}
std::once_flag once_flag;
}  // namespace

extern "C" {
void init_alog_C() { std::call_once(once_flag, InitAlog); }
void Fslib_Fs_FileSystem_Close_C() { fslib::fs::FileSystem::close(); }
size_t FslibFileHandle_C_Read(void *handle, void *buf, size_t count,
                              size_t offset) {
  fslib::fs::File *obj = static_cast<fslib::fs::File *>(handle);
  return obj->pread(buf, count, offset);
}
size_t FslibFileHandle_C_Write(void *handle, const void *buf, size_t count) {
  fslib::fs::File *obj = static_cast<fslib::fs::File *>(handle);
  return obj->write(buf, count);
}

fslib::ErrorCode FslibFileHandle_C_Close(void *handle) {
  fslib::fs::File *obj = static_cast<fslib::fs::File *>(handle);
  return obj->close();
}

fslib::ErrorCode FslibFileHandle_C_Flush(void *handle) {
  fslib::fs::File *obj = static_cast<fslib::fs::File *>(handle);
  return obj->flush();
}
void FslibFileHandle_C_Destory(void *handle) {
  fslib::fs::File *obj = static_cast<fslib::fs::File *>(handle);
  delete obj;
}

fslib::ErrorCode FslibFileHandle_C_GetLastError(void *handle) {
  fslib::fs::File *obj = static_cast<fslib::fs::File *>(handle);
  return obj->getLastError();
}

void Fslib_Fs_FileSystem_GetErrorString(void **str, fslib::ErrorCode ec) {
  auto msg = fslib::fs::FileSystem::getErrorString(ec);
  (*str) = strdup(msg.c_str());
}

void Fslib_Fs_FileSystem_OpenFile(const char *file_name, fslib::Flag mode,
                                  void **handle) {
  auto file = fslib::fs::FileSystem::openFile(file_name, mode);
  (*handle) = file;
}

fslib::ErrorCode Fslib_Fs_FileSystem_IsExist(const char *file_name) {
  return fslib::fs::FileSystem::isExist(file_name);
}

fslib::ErrorCode Fslib_Fs_FileSystem_ListDir(const char *dir_name,
                                             void ***files) {
  std::vector<std::string> ret;
  fslib::ErrorCode ec = fslib::fs::FileSystem::listDir(dir_name, ret);
  (*files) = (void **)vectorToStringArray(ret);
  return ec;
}

fslib::ErrorCode Fslib_Fs_FileSystem_Remove(const char *file_name) {
  return fslib::fs::FileSystem::remove(file_name);
}
fslib::ErrorCode Fslib_Fs_FileSystem_MkDir(const char *dir_name,
                                           bool recursive) {
  return fslib::fs::FileSystem::mkDir(dir_name, recursive);
}
fslib::ErrorCode Fslib_Fs_FileSystem_Rename(const char *old_file_name,
                                            const char *new_file_name) {
  return fslib::fs::FileSystem::rename(old_file_name, new_file_name);
}

fslib::ErrorCode Fslib_Fs_FileSystem_IsDirectory(const char *file_name) {
  return fslib::fs::FileSystem::isDirectory(file_name);
}
fslib::ErrorCode Fslib_Fs_FileSystem_GetPathMeta(const char *file_name,
                                                 fslib::PathMeta &meta) {
  return fslib::fs::FileSystem::getPathMeta(file_name, meta);
}
}
