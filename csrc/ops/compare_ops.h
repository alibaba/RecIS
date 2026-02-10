#pragma once

namespace recis {
namespace functional {

enum CompareOp : int
{
    LT,
    LE,
    GT,
    GE,
    EQ,
};

#define BF_DISPATCH_CASE(enum_type, ...)             \
  case enum_type: {                                  \
    constexpr auto compare_t = enum_type;            \
    __VA_ARGS__();                                   \
    break;                                           \
  }
    
#define BF_DISPATCH_CASE_COMPARE_TYPES(...)          \
  BF_DISPATCH_CASE(CompareOp::LT, __VA_ARGS__)       \
  BF_DISPATCH_CASE(CompareOp::LE, __VA_ARGS__)       \
  BF_DISPATCH_CASE(CompareOp::GT, __VA_ARGS__)       \
  BF_DISPATCH_CASE(CompareOp::GE, __VA_ARGS__)       \
  BF_DISPATCH_CASE(CompareOp::EQ, __VA_ARGS__)

#define BF_DISPATCH_SWITCH(TYPE, ...)                \
  switch (TYPE) {                                    \
    __VA_ARGS__                                      \
    default:                                         \
      TORCH_CHECK_NOT_IMPLEMENTED(                   \
          false,                                     \
          "Invalid TYPE: ",                          \
          std::to_string(TYPE));                     \
  }

#define BF_DISPATCH_COMPARE_TYPES(TYPE, ...)  \
  BF_DISPATCH_SWITCH(TYPE, BF_DISPATCH_CASE_COMPARE_TYPES(__VA_ARGS__))

} // namespace functional
} // namespace recis
