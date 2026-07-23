/*
 * Copyright 1999-2018 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "column-io/odps/plugins/logging.h"

#include <stdlib.h>
#include <time.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <stdint.h>

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

namespace {

class TimeUtils {
public:
  static uint64_t NowMicros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  }
};

LogLevel GetLogLevelFromEnv() {
  const char *log_level = getenv("NEBULA_IO_LOG_LEVEL");
  if (log_level == nullptr) {
    return INFO;
  }
  return static_cast<LogLevel>(atoi(log_level));
}

} // namespace

LogMessage::LogMessage(const char *fname, int line, LogLevel severity)
    : fname_(fname), line_(line), severity_(severity) {}

LogMessage::~LogMessage() {
  static LogLevel min_log_level = GetLogLevelFromEnv();
  if (severity_ >= min_log_level) {
    GenerateLogMessage();
  }
}

void LogMessage::GenerateLogMessage() {
  uint64_t now_micros = TimeUtils::NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  const size_t kBufferSize = 30;
  char time_buffer[kBufferSize];
  strftime(time_buffer, kBufferSize, "%Y-%m-%d %H:%M:%S",
           localtime(&now_seconds));
  fprintf(stderr, "%s.%06d: %c %s:%d] %s\n", time_buffer, micros_remainder,
          "DIWEF"[severity_], fname_, line_, str().c_str());
}

LogMessageFatal::LogMessageFatal(const char *file, int line)
    : LogMessage(file, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  GenerateLogMessage();
  _exit(11);
}

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara
