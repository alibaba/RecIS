cmake_minimum_required(VERSION 3.21)

include(cmake/gpu_platform.cmake)
detect_gpu_platform()

if(NEED_CPU_ONLY OR NEED_ROW_ONLY)
    project(column-io VERSION 1.0 LANGUAGES CXX)
elseif(USE_ROCM)
    project(column-io VERSION 1.0 LANGUAGES CXX HIP)
else()
    project(column-io VERSION 1.0 LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTING "Build with unit tests" OFF)
message(STATUS "Build with unit tests status: " ${BUILD_TESTING})

option(INTERNAL_VERSION "Option for internal version" ON)
message(STATUS "internal version: " ${INTERNAL_VERSION})
if (INTERNAL_VERSION)
  add_definitions(-DINTERNAL_VERSION=1)
else()
  add_definitions(-DINTERNAL_VERSION=0)
endif(INTERNAL_VERSION)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Configure GPU platform (sets -DCPU_ONLY define in CPU_ONLY mode, or configures CUDA/ROCm)
configure_gpu_platform()

add_compile_options(-ggdb -O3 "-fvisibility=hidden")

#FLAG: NEED_ODPS_COLUMN 表示启用ODPS storage cxx11abi=1接口. 副作用为禁用直读接口
if (INTERNAL_VERSION)
    set(NEED_ODPS_COLUMN $ENV{NEED_ODPS_COLUMN})
    if (DEFINED NEED_ODPS_COLUMN AND NEED_ODPS_COLUMN STREQUAL "0")
        set(_GLIBCXX_USE_CXX11_ABI 0 CACHE INTERNAL "Use C++11ABI=0")
    else()
        set(_GLIBCXX_USE_CXX11_ABI 1 CACHE INTERNAL "Use C++11ABI=1")
    endif()
    message(STATUS "Now ColumnIO Using _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI} -DNDEBUG)
    add_definitions(-DINTERNAL_VERSION=1)
else()
    set(_GLIBCXX_USE_CXX11_ABI 1 CACHE INTERNAL "Use CXX11ABI=1")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1 -DNDEBUG)
    add_definitions(-DINTERNAL_VERSION=0)
endif(INTERNAL_VERSION)

if(${_GLIBCXX_USE_CXX11_ABI} EQUAL 0)
    option(NEED_ARROW_17 "Option for internal arrow version 17.0.0" OFF)
else()
    option(NEED_ARROW_17 "Option for internal arrow version 17.0.0" ON)
endif()
message(STATUS "Option for arrow version 17.0.0: " ${NEED_ARROW_17})

set(THIRD_PARTY_DIR "${PROJECT_BINARY_DIR}/third_party_install"
        CACHE PATH "Where to install third party headers and libs")
set(of_proto_python_dir "${PROJECT_BINARY_DIR}/of_proto_python")
add_custom_target(make_pyproto_dir ALL COMMAND ${CMAKE_COMMAND} -E make_directory
                                               ${of_proto_python_dir})
					       
add_subdirectory(third-party/abseil-cpp-20230802.1)
include(cmake/rapidjson.cmake)
include(cmake/zlib.cmake)
include(cmake/protobuf.cmake)
include(cmake/proto2cpp.cmake)
if (INTERNAL_VERSION)
    if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
        include(cmake/odps.cmake)
        set(ARROW_LIBRARY
            PUBLIC arrow
        )
        set(ARROW_INCLUDE
            PUBLIC ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/include
        )
        #TODO: ARROW17_ORC* is used for local_orc reader and dataset, which have not arrow1 version
        set(ARROW17_ORC_INCLUDE
            ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-4.9.2/arrow-17.0.0-orc/include
        )
        set(ARROW17_ORC_LIBRARY
            ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-4.9.2/arrow-17.0.0-orc/lib64/libarrow.a
            ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-4.9.2/arrow-17.0.0-orc/lib64/libarrow_bundled_dependencies.a
        )
    else()
        if(NEED_ARROW_17)
            set(ARROW_LIBRARY
                PRIVATE -Wl,--start-group
                PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0/lib64/libarrow.a
                PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0/lib64/libarrow_bundled_dependencies.a
                PRIVATE -Wl,--end-group
            )
            set(ARROW_INCLUDE
                PUBLIC ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0/include
            )
        else()
            set(ARROW_LIBRARY
                PRIVATE -Wl,--start-group
                PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libarrow.a
                PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libjemalloc_pic.a
                PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/lz4/lib/liblz4.a
                PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/zstd/lib/libzstd.a
                PRIVATE -Wl,--end-group
            )
            set(ARROW_INCLUDE
                PUBLIC ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/include
            )
        endif()
        #TODO: ARROW17_ORC* is used for local_orc reader and dataset, which have not arrow1 version
        set(ARROW17_ORC_INCLUDE
            ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0-orc/include
        )
        set(ARROW17_ORC_LIBRARY
            ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0-orc/lib64/libarrow.a
            ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0-orc/lib64/libarrow_bundled_dependencies.a
        )
    endif()
    include(cmake/ailake.cmake)
else()
    set(ARROW_LIBRARY arrow)
    set(ARROW_INCLUDE "")
    set(ARROW17_ORC_INCLUDE "")
    set(ARROW17_ORC_LIBRARY arrow)
endif(INTERNAL_VERSION)
if(BUILD_TESTING)
    enable_testing()
    message((STATUS "Adding googletest cmake for Unittests"))
    include(cmake/googletest.cmake)
endif()

set(DLPACK_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third-party/dlpack/include)
message(STATUS "dlpack: " ${DLPACK_INCLUDE_DIR})
set(PYBIND11_PYTHON_VERSION $ENV{COLUMN_PYTHON_VERSION})
add_subdirectory(third-party/pybind11)
add_subdirectory(cc)
