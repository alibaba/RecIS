/* Recis Module: recis monitor Log Writer
 * this module is ONLY for metric, special customized for multi-process,
 * rotation and abi avoiding. Be warn of that this module CANNOT be used for
 * general logging
 */
#ifndef RECIS_MONITOR_LOG_WRITER_H
#define RECIS_MONITOR_LOG_WRITER_H
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define RECIS_MONITOR_LOG_WRITER_BUF_MAX 32768  // 缓冲区最大字节数
#define RECIS_MONITOR_LOG_KEEP_NUM 10  // 最多保留多少个轮转文件

typedef enum {
  RECIS_MONITOR_LOG_ROTATE_INVALID = 0LL,
  RECIS_MONITOR_LOG_ROTATE_10SECOND = 10LL,
  RECIS_MONITOR_LOG_ROTATE_MINUTE = 60LL,
  RECIS_MONITOR_LOG_ROTATE_10MIN = 600LL,
  RECIS_MONITOR_LOG_ROTATE_HOUR = 3600LL,
  RECIS_MONITOR_LOG_ROTATE_DAY = 86400LL,
  RECIS_MONITOR_LOG_ROTATE_DEFAULT = RECIS_MONITOR_LOG_ROTATE_HOUR
} RECIS_MONITOR_LOG_ROTATE_T;

/**
 * soft_path: 基准文件名，如 "/var/log/xxx.log"
 * interval: 轮转间隔
 * fp:       打开的文件指针
 * cur_slot: 当前轮转时间
 * buf:      缓冲区
 * buf_len:  缓冲区长度
 * mutex:    线程安全互斥锁
 */
typedef struct {
  char soft_path[256];
  FILE *fp;
  time_t cur_slot;
  char buf[RECIS_MONITOR_LOG_WRITER_BUF_MAX];
  size_t buf_len;
  time_t last_write_time;
  RECIS_MONITOR_LOG_ROTATE_T interval;
  pthread_mutex_t mutex;
} RECIS_MONITOR_LOG_WRITER_T;

RECIS_MONITOR_LOG_WRITER_T *recis_monitor_create_logger(const char *base_path);
int recis_monitor_init_logger(RECIS_MONITOR_LOG_WRITER_T *lw,
                              const char *base_path,
                              RECIS_MONITOR_LOG_ROTATE_T interval);
int recis_monitor_close_logger(RECIS_MONITOR_LOG_WRITER_T *lw);
int recis_monitor_logger_write(RECIS_MONITOR_LOG_WRITER_T *lw, const char *msg,
                               bool flush);
int recis_monitor_safe_flush(RECIS_MONITOR_LOG_WRITER_T *lw);
// TODO [Untested and Unused] recis_monitor_log_atfork_setup must be called
// before any fork
void recis_monitor_log_atfork_setup(RECIS_MONITOR_LOG_WRITER_T *lw);

#ifdef __cplusplus
}
#endif

/* C++ OOP interface for pybind calling, if you want to use it in C++ */
#ifdef __cplusplus
#include <string>
namespace recis {
namespace monitor {
class LogWriter {
 private:
  RECIS_MONITOR_LOG_WRITER_T *logger_;

 public:
  LogWriter(const std::string &base_path, const int interval) {
    logger_ = (RECIS_MONITOR_LOG_WRITER_T *)malloc(
        sizeof(RECIS_MONITOR_LOG_WRITER_T));
    recis_monitor_init_logger(
        logger_, base_path.c_str(),
        static_cast<RECIS_MONITOR_LOG_ROTATE_T>(interval));
  }
  ~LogWriter() {
    if (logger_) {
      recis_monitor_close_logger(logger_);
      logger_ = nullptr;
    }
  }
  int inited() { return logger_ != nullptr; }
  int write(const std::string &msg, bool flush = false) {
    if (!logger_) {
      return -1;
    }
    return recis_monitor_logger_write(logger_, msg.c_str(), flush);
  }
  void atfork_setup() = delete;
};
}  // namespace monitor
}  // namespace recis
#endif

#endif  // RECIS_MONITOR_LOG_WRITER_H
