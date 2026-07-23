//#include "column-io/open_storage/common-util/logging.h"
#include "logging.h"

#include <time.h>

#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <stdint.h>

namespace xdl
{
namespace paiio
{
namespace third_party
{
namespace common_util
{

namespace {

class TimeUtils {
 public:
  static uint64_t NowMicros() {
    struct timeval  tv;
    gettimeofday(&tv, NULL);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  }
};

LogLevel GetLogLevelFromEnv() {
  const char* log_level = getenv("NEBULA_IO_LOG_LEVEL");
  if (log_level == nullptr) {
    return INFO;
  }
  return static_cast<LogLevel>(atoi(log_level));
}

}  // namespace anonymous


LogMessage::LogMessage(const char* fname, int line, LogLevel severity)
  : fname_(fname), line_(line), severity_(severity) {}

LogMessage::~LogMessage() {
  static LogLevel min_log_level = GetLogLevelFromEnv();
  if (severity_ >= min_log_level) {
    GenerateLogMessage();
  }
}

void LogMessage::GenerateLogMessage() {
  const char* LOG_NAMES[] = { "DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
  uint64_t now_micros = TimeUtils::NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  const size_t kBufferSize = 30;
  char time_buffer[kBufferSize];
  strftime(time_buffer, kBufferSize, "%Y-%m-%d %H:%M:%S",
           localtime(&now_seconds));
  fprintf(stderr, "[%s.%06d] [%s] [%s:%d] %s\n", time_buffer, micros_remainder,
          LOG_NAMES[severity_], fname_, line_, str().c_str());
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  GenerateLogMessage();
  _exit(11);
}

} // common_util
} // third_party
} // paiio
} // xdl
