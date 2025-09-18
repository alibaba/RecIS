#ifndef PAIIO_THIRD_PARTY_ODPS_TUNNEL_PLUGIN_C_API_H_
#define PAIIO_THIRD_PARTY_ODPS_TUNNEL_PLUGIN_C_API_H_

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif


/******** Functions for Odps Open Storage BEGIN *********/
typedef struct CAPI_ODPS_SDK_OdpsOpenStorageArrowReader CAPI_ODPS_SDK_OdpsOpenStorageArrowReader;

#define DECLARE_INTERFACE_OPEN_STORAGE(RetType, FuncName, ...) \
  extern RetType CAPI_ODPS_OPEN_STORAGE_##FuncName(__VA_ARGS__)

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    CreateReadSession,
    const char* access_id,
    const char* access_key,
    const char* tunnel_endpoint,
    const char* project,
    const char* table,
    void* required_partitions,
    void* required_data_columns,
    const char* mode,
    const char* default_project,
    int connect_timeout,
    int rw_timeout,
    char* session_id,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    GetReadSession,
    const char* access_id,
    const char* access_key,
    const char* tunnel_endpoint,
    const char* session_id,
    const char* project,
    const char* table,
    const char* default_project,
    int connect_timeout,
    int rw_timeout,
    char* session_def_str,
    void* session_def,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    InitOdpsOpenStorageSessions,
    const char* access_id,
    const char* access_key,
    const char* tunnel_endpoint,
    const char* odps_endpoint,
    const char* projects,
    const char* tables,
    const char* partition_specs,
    const char* physical_partitions,
    const char* required_data_columns,
    const char* sep,
    const char* mode,
    const char* default_project,
    int connect_timeout,
    int rw_timeout,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    RegisterOdpsOpenStorageSession,
    const char* access_id,
    const char* access_key,
    const char* tunnel_endpoint,
    const char* odps_endpoint,
    const char* project,
    const char* table,
    const char* partition,
    const char* required_data_columns,
    const char* sep,
    const char* mode,
    const char* default_project,
    int connect_timeout,
    int rw_timeout,
    bool register_light,
    const char* session_id,
    long expiration_time,
    long record_count,
    const char* session_def_str,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    ExtractLocalReadSession,
    char* session_def_str,
    const char* access_id,
    const char* access_key,
    const char* project,
    const char* table,
    const char* partition,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    RefreshReadSessionBatch,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader*,
    CreateReader,
    const char* path_str,
    int max_batch_rows,
    const char* reader_name,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    GetTableSize,
    const char* path_str,
    uint64_t* table_size,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    GetSessionExpireTimestamp,
    const char* session_id,
    void* expire_timestamp);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    GetSchema,
    const char* config,
    void* schema,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    ReadBatch,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader,
    void* batch,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    Seek,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader,
    size_t pos,
    void* status);

DECLARE_INTERFACE_OPEN_STORAGE(
    size_t,
    Tell,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader);

DECLARE_INTERFACE_OPEN_STORAGE(
    void,
    DeleteReader,
    CAPI_ODPS_SDK_OdpsOpenStorageArrowReader* reader,
    void* status);
#undef DECLARE_INTERFACE_OPEN_STORAGE
/******** Functions for Odps Open Storage END *********/


#ifdef __cplusplus
}
#endif
#endif // PAIIO_THIRD_PARTY_ODPS_TUNNEL_PLUGIN_C_API_H_
