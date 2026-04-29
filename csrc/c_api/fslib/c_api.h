#pragma once
#include <stdint.h>

#include <array>
#include <cstddef>

#include "fslib/common/common_type.h"

extern "C" {
using TypeFslibFileHandle_C = void *;

void init_alog_C();
size_t FslibFileHandle_C_Read(void *handle, void *buf, size_t count,
                              size_t offset);
size_t FslibFileHandle_C_Write(void *handle, const void *buf, size_t count);
fslib::ErrorCode FslibFileHandle_C_Close(void *handle);
fslib::ErrorCode FslibFileHandle_C_Flush(void *handle);
fslib::ErrorCode FslibFileHandle_C_GetLastError(void *handle);
void FslibFileHandle_C_Destory(void *handle);

void Fslib_Fs_FileSystem_Close_C();
void Fslib_Fs_FileSystem_GetErrorString(void **str, fslib::ErrorCode ec);
void Fslib_Fs_FileSystem_OpenFile(const char *file_name, fslib::Flag mode,
                                  void **handle);
fslib::ErrorCode Fslib_Fs_FileSystem_IsExist(const char *file_name);
fslib::ErrorCode Fslib_Fs_FileSystem_ListDir(const char *dir_name,
                                             void ***files);
fslib::ErrorCode Fslib_Fs_FileSystem_Remove(const char *file_name);
fslib::ErrorCode Fslib_Fs_FileSystem_MkDir(const char *dir_name,
                                           bool recursive);
fslib::ErrorCode Fslib_Fs_FileSystem_Rename(const char *old_file_name,
                                            const char *new_file_name);
fslib::ErrorCode Fslib_Fs_FileSystem_IsDirectory(const char *file_name);
fslib::ErrorCode Fslib_Fs_FileSystem_GetPathMeta(const char *file_name,
                                                 fslib::PathMeta &meta);
}
