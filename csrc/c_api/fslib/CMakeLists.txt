file(GLOB fslib_c_api_CC "*.cc")

set(DEPS_DIR "${CMAKE_BINARY_DIR}/fs_deps")
#set(ZIP_URL "https://xrec.oss-cn-hangzhou-zmf.aliyuncs.com/deps/xrec_build_deps.zip")
#set(ZIP_URL "https://maxmit-public-read.oss-cn-hangzhou-zmf.aliyuncs.com/xrec_build_deps_2026-01-22-10-23-57.zip")
set(ZIP_URL "https://maxmit-public-read.oss-cn-hangzhou-zmf.aliyuncs.com/xrec_build_deps_2026-01-22-12-34-51.zip")
set(ZIP_FILE "${CMAKE_BINARY_DIR}/xrec_build_deps.zip")

file(DOWNLOAD ${ZIP_URL} ${ZIP_FILE} SHOW_PROGRESS)

execute_process(
    COMMAND unzip -o -q ${ZIP_FILE} -d ${DEPS_DIR}
    RESULT_VARIABLE result
)

if(result)
    message(FATAL_ERROR "Error during unzip process.")
endif()


add_library(fslib_c_api SHARED ${fslib_c_api_CC})
target_include_directories(fslib_c_api PUBLIC ${CC_BASE_INCLUDE_DIR})
target_include_directories(fslib_c_api SYSTEM PRIVATE ${DEPS_DIR})
target_link_options(fslib_c_api PRIVATE "-Wl,-rpath,$ORIGIN")

target_link_libraries(fslib_c_api PRIVATE 
	${DEPS_DIR}/fslib/lib64/libpangudfs_pangu_client.so.1
	${DEPS_DIR}/fslib/lib64/libfslib-framework.so
	${DEPS_DIR}/fslib/lib64/libfslib_plugin_pangu.so)

target_compile_definitions(fslib_c_api PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(fslib_c_api PRIVATE -std=c++17)
target_compile_options(fslib_c_api PRIVATE -g)
install(TARGETS fslib_c_api DESTINATION ${RECIS_LIB_INSTALL_DIR})
install(FILES
	${DEPS_DIR}/fslib/lib64/libpangudfs_pangu_client.so.1
	${DEPS_DIR}/fslib/lib64/libfslib-framework.so
	${DEPS_DIR}/fslib/lib64/libfslib_plugin_pangu.so
	${DEPS_DIR}/fslib/lib64/libfslib_plugin_dfs.so
  DESTINATION ${RECIS_LIB_INSTALL_DIR})
