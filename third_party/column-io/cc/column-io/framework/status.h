/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef EULER_COMMON_STATUS_H_
#define EULER_COMMON_STATUS_H_

#include <memory>
#include <sstream>
#include <string>

#include "column-io/framework/error_code.h"
#include "column-io/framework/macros.h"

namespace column {
template <typename T> std::string ToString(T arg) {
  std::stringstream ss;
  ss << arg;
  return ss.str();
}
template <typename T, typename... Args>
std::string ToString(T arg, Args... args) {
  std::stringstream ss;
  ss << arg;
  return ss.str() + ToString(args...);
}
class Status {
public:
  Status();

  Status(ErrorCode code, const std::string_view &msg);

  static Status OK() { return Status(); }

  static Status ArgumentError(const std::string &msg) {
    return Status(INVALID_ARGUMENT, msg);
  }

  bool ok() const { return (code_ == ErrorCode::OK); }

  ErrorCode code() const { return code_; }

  const std::string &error_message() const { return message_; }

  bool operator==(const Status &x) const;
  bool operator!=(const Status &x) const;

  std::string DebugString() const;

#define ADD_METHOD(FUNC, CONS)                                                 \
  template <typename... Args>                                                  \
  static Status __##FUNC##_INTERNAL__(Args... args) {                          \
    return ::column::Status(ErrorCode::CONS, ToString(args...));               \
  }                                                                            \
  template <typename... Args> static Status FUNC(Args... args) {               \
    return __##FUNC##_INTERNAL__("", std::forward<Args>(args)...);             \
  }

  ADD_METHOD(Cancelled, CANCELLED)
  ADD_METHOD(InvalidArgument, INVALID_ARGUMENT)
  ADD_METHOD(NotFound, NOT_FOUND)
  ADD_METHOD(AlreadyExists, ALREADY_EXISTS)
  ADD_METHOD(ResourceExhausted, RESOURCE_EXHAUSTED)
  ADD_METHOD(Unavailable, UNAVAILABLE)
  ADD_METHOD(FailedPrecondition, FAILED_PRECONDITION)
  ADD_METHOD(OutOfRange, OUT_OF_RANGE)
  ADD_METHOD(Unimplemented, UNIMPLEMENTED)
  ADD_METHOD(Internal, INTERNAL)
  ADD_METHOD(Aborted, ABORTED)
  ADD_METHOD(DeadlineExceeded, DEADLINE_EXCEEDED)
  ADD_METHOD(DataLoss, DATA_LOSS)
  ADD_METHOD(Unknown, UNKNOWN)
  ADD_METHOD(PermissionDenied, PERMISSION_DENIED)
  ADD_METHOD(Unauthenticated, UNAUTHENTICATED)

#undef ADD_METHOD // ADD_METHOD

private:
  ErrorCode code_;
  std::string message_;
};

inline bool Status::operator==(const Status &x) const {
  return code_ == x.code_ && message_ == x.message_;
}

inline bool Status::operator!=(const Status &x) const { return !(*this == x); }

std::ostream &operator<<(std::ostream &os, const Status &x);

#define RETURN_IF_ERROR(...)                                                   \
  do {                                                                         \
    const ::column::Status _status = (__VA_ARGS__);                            \
    if (PREDICT_FALSE(!_status.ok()))                                          \
      return _status;                                                          \
  } while (0)

#define EULER_SINGLE_ARG(...) __VA_ARGS__

#define EULER_CHECK_STATUS(STATUS)                                             \
  do {                                                                         \
    ::euler::Status __st__ = STATUS;                                           \
    if (!__st__.ok()) {                                                        \
      std::string msg = std::string("\nCheck Status [") + #STATUS +            \
                        "] at [" __FILE__ + "]" + __func__ + "@" +             \
                        std::to_string(__LINE__) + ". ";                       \
      return ::euler::Status(__st__.code(), msg + __st__.error_message());     \
    }                                                                          \
  } while (0)

#define EULER_CHECK_STATUS_Q(STATUS)                                           \
  do {                                                                         \
    euler::Status __st__ = STATUS;                                             \
    if (!__st__.ok()) {                                                        \
      return __st__;                                                           \
    }                                                                          \
  } while (0)

#define EULER_CHECK_COND(COND, STATUS)                                         \
  do {                                                                         \
    if (!(COND)) {                                                             \
      ::euler::Status __st__ = STATUS;                                         \
      std::string msg = std::string("\nCheck Condition [") + #COND +           \
                        "] at [" __FILE__ + "]" + __func__ + "@" +             \
                        std::to_string(__LINE__) + ". ";                       \
      return ::euler::Status(__st__.code(), msg + __st__.error_message());     \
    }                                                                          \
  } while (0)

#define EULER_CHECK_COND_Q(STATUS)                                             \
  do {                                                                         \
    if (!(COND)) {                                                             \
      return STATUS;                                                           \
    }                                                                          \
  } while (0)

#define COLUMN_RETURN_NOT_OK(STATUS)                                           \
  do {                                                                         \
    ::column::Status __status__ = (STATUS);                                    \
    if (!__status__.ok()) {                                                    \
      return __status__;                                                       \
    }                                                                          \
  } while (0);

} // namespace column

#endif // EULER_COMMON_STATUS_H_
