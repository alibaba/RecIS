RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR} tensor.proto tensor_shape.proto types.proto)
file(GLOB framework_CC *.cc)
add_library(framework ${framework_CC} ${PROTO_SRCS})
set(BUILT ${PROJECT_BINARY_DIR})
if (INTERNAL_VERSION)
    if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
        target_compile_definitions(framework PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
    else()
        target_compile_definitions(framework PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    endif()
endif()

# Set GPU-specific include and library paths
if(NEED_CPU_ONLY)
    set(GPU_INCLUDE_DIRS "")
    set(GPU_LIBRARIES "")
elseif(USE_ROCM)
    set(GPU_INCLUDE_DIRS ${HIP_INCLUDE_DIRS})
    set(GPU_LIBRARIES ${HIP_LIBRARIES})
else()
    set(GPU_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    set(GPU_LIBRARIES ${CUDA_LIBRARIES})
endif()

target_include_directories(framework
  PUBLIC ${GPU_INCLUDE_DIRS}
  PUBLIC ${DLPACK_INCLUDE_DIR})

find_package(Threads REQUIRED)
target_link_libraries(
    framework
    PUBLIC ${ARROW_LIBRARY}
    PUBLIC absl::strings absl::check
    PUBLIC ${BUILT}/protobuf/src/protobuf/cmake/libprotobuf.a
    ${GPU_LIBRARIES}
)
add_dependencies(framework make_pyproto_dir)
#message(STATUS arrow include dir  ${CMAKE_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/include)
target_include_directories(
    framework
    PUBLIC
    ${CC_BASE_INCLUDE_DIR}
    ${CC_BUILD_BASE_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/column-io/cc
    ${ARROW_INCLUDE}
)
