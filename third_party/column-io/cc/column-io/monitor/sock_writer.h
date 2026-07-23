/* Recis Module: recis monitor Sock Writer
 */
#ifndef RECIS_MONITOR_SOCK_WRITER_H
#define RECIS_MONITOR_SOCK_WRITER_H
#pragma once

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

class RECIS_MONITOR_SOCK_WRITER {
 public:
  explicit RECIS_MONITOR_SOCK_WRITER(const char* base_path,
                                     int timeout_ms = 3000)
      : sock_path_(base_path), timeout_ms_(timeout_ms) {};
  int write(const std::string& body) { return write(body.c_str()); };
  int write(const char* body);

 private:
  std::string sock_path_;
  int timeout_ms_;
};

#ifdef __cplusplus
}
#endif

#endif  // __RECIS_MONITOR_SOCK_WRITER_H
