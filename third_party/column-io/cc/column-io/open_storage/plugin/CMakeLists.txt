set(BUILT ${PROJECT_BINARY_DIR})

message(STATUS "BUILT directory is: ${BUILT}")

set(STORAGE_ODPS_SDK_CPP_BASE
  ""
)
set(ODPS_SDK_CPP_PKG
  # witharrow17 相比原版做了什么改动
  "odps_sdk_cpp_20260423_173233.tar.gz"
  #"odps_sdk_cpp_20250819_202903-noprint_failpushlog.tar.gz"
)
set(ODPS_SDK_CPP_PKG_URL "${STORAGE_ODPS_SDK_CPP_BASE}/${ODPS_SDK_CPP_PKG}")
set(ODPS_SDK_CPP_PATH "${BUILT}/${ODPS_SDK_CPP_PKG}")

if(NOT EXISTS ${ODPS_SDK_CPP_PATH})
    message(STATUS "Downloading ${ODPS_SDK_CPP_PKG} from ${ODPS_SDK_CPP_PKG_URL}...")
    file(DOWNLOAD ${ODPS_SDK_CPP_PKG_URL} ${ODPS_SDK_CPP_PATH} TIMEOUT 90)
else()
    message(STATUS "${ODPS_SDK_CPP_PKG} already exists. Skipping download.")
endif()

execute_process(
	COMMAND tar -xzf "${ODPS_SDK_CPP_PATH}" -C "${BUILT}"
	WORKING_DIRECTORY "${BUILT}"
  RESULT_VARIABLE TARGZ_RESULT
)

message(STATUS "Current Binary Directory: ${CMAKE_CURRENT_BINARY_DIR}")
if(NOT TARGZ_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to extract ${ODPS_SDK_CPP_PATH}")
endif()

################ 安装编译前依赖 ################
message(STATUS "installing rapidjson header for openstorage/plugin/ in kmonitor deps of autil deps")
# patch for rapidjson-v1.1.0, see https://github.com/Tencent/rapidjson/commit/3b2441b87f99ab65f37b141a7b548ebadb607b96
set(RAPIDJSON_HEADER_PKG_NAME "rapidjson-v1.1.0-patch.tar.gz")
set(RAPIDJSON_HEADER_PKG_URL "https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz")
set(RAPIDJSON_HEADER_INSTALL_PATH "${BUILT}/odps_sdk_cpp/storage_api/thirdparty")
set(KMONITOR_ADAPTER_RPM_NAME "kmonitor_adapter-0.0.2-3.x86_64.rpm")

if(NOT EXISTS ${BUILT}/${RAPIDJSON_HEADER_PKG_NAME})
    message(STATUS "Downloading ${RAPIDJSON_HEADER_PKG_NAME} from ${RAPIDJSON_HEADER_PKG_URL}...")
    file(DOWNLOAD ${RAPIDJSON_HEADER_PKG_URL} ${BUILT}/${RAPIDJSON_HEADER_PKG_NAME} TIMEOUT 30)
else()
    message(STATUS "${BUILT}/${RAPIDJSON_HEADER_PKG_NAME} already exists. Skipping download.")
endif()
execute_process(
	COMMAND tar -xzf "${BUILT}/${RAPIDJSON_HEADER_PKG_NAME}" -C "${RAPIDJSON_HEADER_INSTALL_PATH}"
	WORKING_DIRECTORY "${BUILT}"
  RESULT_VARIABLE TARGZ_RESULT
)
if(NOT TARGZ_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to extract ${RAPIDJSON_HEADER_PKG_NAME}")
else()
    message(STATUS "finish installing rapidjson header")
endif()
message(STATUS "Current Binary Directory: ${CMAKE_CURRENT_BINARY_DIR}")

# # 安装 kmonitor adapter
# message(STATUS "installing ${KMONITOR_ADAPTER_RPM_NAME} for openstorage/plugin/ ")
# if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
# execute_process(
#     COMMAND sudo yum install -y ""
# )
# else()
# execute_process(
#     COMMAND sudo yum install -y ""
# )
# endif()
# message(STATUS "installed ${KMONITOR_ADAPTER_RPM_NAME} for openstorage/plugin/ ")

# 安装libstdc++-static
execute_process(
	COMMAND sudo yum install -y libstdc++-static
)

# # libidn.a, liblz4.a
# execute_process(
#   # liblz4.a(by kimwang. yum liblz4.a is not fPic)
# 	COMMAND wget -qO /usr/local/lib64/liblz4.a  ""
#   # libidn.a(by liumingjie. yum libidn static is not available in 8u yum)
# 	COMMAND wget -qO /usr/local/lib64/libidn.a  ""
# )

file(GLOB OPEN_STORAGE_PLUGIN_CC_FILTERED *.cc)
set(OPEN_STORAGE_PLUGIN_CC "")

foreach(file ${OPEN_STORAGE_PLUGIN_CC_FILTERED})
    if(NOT file MATCHES "test")
        list(APPEND OPEN_STORAGE_PLUGIN_CC ${file})
    endif()
endforeach()

message(STATUS "Filtered files: ${OPEN_STORAGE_PLUGIN_CC}")

################ 创建编译目标 ################
add_library(open_storage_plugin SHARED ${OPEN_STORAGE_PLUGIN_CC})

################ 添加编译头文件 ################
target_include_directories(open_storage_plugin
  PRIVATE
    ${CC_BASE_INCLUDE_DIR}
    ${BUILT}/odps_sdk_cpp
    ${BUILT}/odps_sdk_cpp/storage_api/thirdparty
    ${BUILT}/odps_sdk_cpp/storage_api/thirdparty/cpp-httplib
    ${ARROW_INCLUDE}
)

################ 添加编译选项 ################
target_compile_options(open_storage_plugin PRIVATE -fvisibility=hidden)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fvisibility.ver "{
global:
  # 隐藏interface.py所需的 CAPI_ODPS_OPEN_STORAGE_* 以外的所有符号
  CAPI_ODPS_OPEN_STORAGE_*;

local:
  # 注意包含recis::monitor::*, 防止和py_interface单例冲突
  *;
};
")

################ 添加链接选项 ################
if(_GLIBCXX_USE_CXX11_ABI EQUAL 0)
    set(ODPS_THIRD_PARTY_SUFFIX "gcc-4.9.2")
    set(BZ2_STATIC_LIB /opt/conda/lib/libbz2.a)
else()
    set(ODPS_THIRD_PARTY_SUFFIX "gcc-8")
endif()
# 搜索ABI相关链接库. brotl snappy为abi=0直读的依赖(即abi=1不依赖), arrow为abi0/1不同版本的依赖
# find all static lib of autil: $(ls /usr/local/lib64/autil/lib*.a)
file(GLOB AUTIL_LIBS "/usr/local/lib64/autil/lib*.a")
# find all static lib of brotl/snappy, this is only for odps_algo, so CXX11_ABI=1 will give it null value
file(GLOB BROTL_LIBS "${BUILT}/odps_sdk_cpp/third_party/${ODPS_THIRD_PARTY_SUFFIX}/brotli-1.0.7/lib/libbrotl*.a")
file(GLOB SNAPY_LIBS "${BUILT}/odps_sdk_cpp/third_party/${ODPS_THIRD_PARTY_SUFFIX}/snappy-1.1.1/lib/libsnappy.a")
target_link_libraries(open_storage_plugin 
  PRIVATE -fPIC
  PRIVATE ${ARROW_LIBRARY}
  -Wl,--start-group
  -lpthread  -ldl
  open_storage_common-util
  monitor
  ################ ODSP 相关 ################
  PRIVATE ${BUILT}/odps_sdk_cpp/third_party/gcc-8/openssl-1.1.1g/lib/libssl.a
  PRIVATE ${BUILT}/odps_sdk_cpp/third_party/gcc-8/openssl-1.1.1g/lib/libcrypto.a
  PRIVATE ${BUILT}/odps_sdk_cpp/third_party/gcc-8/lz4/lib/liblz4.a  # from xdl-util/storage_api_puller
  PRIVATE ${BUILT}/odps_sdk_cpp/third_party/gcc-8/idn/lib/libidn.a  # from xdl-util/storage_api_puller
  
  PRIVATE ${BROTL_LIBS}
  PRIVATE ${SNAPY_LIBS}
  
  # 仅 _GLIBCXX_USE_CXX11_ABI=0 时链接额外的/opt/conda/lib/libbz2.a, ABI=1时ARROW提供
  $<$<BOOL:$<NOT:$<STREQUAL:${_GLIBCXX_USE_CXX11_ABI},0>>>:${BZ2_STATIC_LIB}>

  # NOTE: 改成cxx11abi0后最好不链接std而是直接用cxx11abi0运行环境中的.  暂不确定是否两者cxx11abi0的符号在其他abi上是否兼容
  # PRIVATE -static-libgcc  # 静态链接 libc
  # PRIVATE -static-libstdc++  # 静态链接 libstdc++
  $<$<BOOL:$<NOT:$<STREQUAL:${_GLIBCXX_USE_CXX11_ABI},0>>>:-static-libgcc>
  $<$<BOOL:$<NOT:$<STREQUAL:${_GLIBCXX_USE_CXX11_ABI},0>>>:-static-libstdc++>
  -Wl,--end-group

  PRIVATE "-Wl,-rpath,'$ORIGIN/'" # 优先加载本动态库统计目录的符号[本条仅用作兼容, 全自依赖后理论无需加载旁路 只需glibc]
  -Wl,--exclude-libs,ALL    # 自动隐藏依赖符号
  -Wl,--as-needed   # 移除未依赖符号
  -Wl,-z,defs   # 确保无未定义符号
  -Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/fvisibility.ver  # 隐藏所有内部类, 暴露CAPI_*符号
  
)
