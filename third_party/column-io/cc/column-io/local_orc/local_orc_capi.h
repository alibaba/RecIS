#ifndef COLUMNIO_CC_LOCAL_ORC_C_API_H_
#define COLUMNIO_CC_LOCAL_ORC_C_API_H_
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CAPI_LOCAL_ORC_ReadCtx CAPI_LOCAL_ORC_ReadCtx;

// 打开本地 ORC 文件，返回 reader 句柄。
// file_path: 文件路径字符串
// file_path_len: 文件路径长度
// columns: 以逗号分隔的列名字符串
// columns_len: 列名字符串长度
// 成功返回非空指针，失败返回 nullptr
extern CAPI_LOCAL_ORC_ReadCtx *CAPI_LOCAL_ORC_Method_MakeReader(const char *file_path,
                                                                    size_t file_path_len,
                                                                    const char *columns,
                                                                    size_t columns_len);

extern void CAPI_LOCAL_ORC_Method_ReadBatch(CAPI_LOCAL_ORC_ReadCtx *reader, void *batch, void* arrow_status);

extern void CAPI_LOCAL_ORC_Method_Seek(CAPI_LOCAL_ORC_ReadCtx *reader, int64_t index, void* arrow_status);

extern void CAPI_LOCAL_ORC_Method_Tell(CAPI_LOCAL_ORC_ReadCtx *reader, int64_t *index);

extern void CAPI_LOCAL_ORC_Method_ReadSchema(CAPI_LOCAL_ORC_ReadCtx *reader, void *schema);

extern void CAPI_LOCAL_ORC_Method_DeleteReaderCtx(CAPI_LOCAL_ORC_ReadCtx *reader);

#ifdef __cplusplus
} // end of extern "C"

#endif

#endif // COLUMNIO_CC_LOCAL_ORC_C_API_H_