include(ExternalProject)
#FLAG: NEED_ODPS_COLUMN 表示启用ODPS storage接口. 副作用为使用Cxx11abi=1, 禁用直读algo接口这类不兼容新abi的功能
if (DEFINED NEED_ODPS_COLUMN AND NEED_ODPS_COLUMN STREQUAL "0")
    message(INFO "在 CXX11ABI=0 条件下编译Ailake模块, 注意本模块需要配合正确的libfslib-framework ABI0 版本")
    set(_GLIBCXX_USE_CXX11_ABI 0 CACHE INTERNAL "Use C++11ABI=0 for ailake")
    set(lake_sdk_URL "")
	  set(lake_sdk_MD5 "66a0ba35ac5ad841510d90ca18dbcd8e")
else()
    message(INFO "在 CXX11ABI=1 条件下编译Ailake模块, 注意本模块需要配合正确的libfslib-framework ABI1 版本")
    set(_GLIBCXX_USE_CXX11_ABI 1 CACHE INTERNAL "Use C++11ABI=1 for ailake")
    set(lake_sdk_URL "")
    set(lake_sdk_MD5 "b827c94d8aafa4fcd5e47a63b866e0f9")
endif()
ExternalProject_Add(
  lake_sdk
  PREFIX lake-sdk-${lake_sdk_MD5}
  URL ${lake_sdk_URL}
  URL_MD5 ${lake_sdk_MD5}
  CONFIGURE_COMMAND bash -c "echo skipping configuration step" 
  DOWNLOAD_EXTRACT_TIMESTAMP true 
  BUILD_COMMAND bash -c "echo skipping build step" 
  BUILD_IN_SOURCE true 
  INSTALL_COMMAND bash -c "echo skipping install step")

ExternalProject_Get_Property(lake_sdk SOURCE_DIR)
set(lake_sdk_LIBRARY_BASE ${SOURCE_DIR})
add_library(lake INTERFACE)
message("lake source dir: ${SOURCE_DIR}")
target_link_directories(lake INTERFACE ${SOURCE_DIR})
target_link_libraries(lake INTERFACE -l:lib_lake_IO.so -l:liblz4.so -l:libzstd.so)
add_dependencies(lake lake_sdk)
