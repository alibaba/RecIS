file(GLOB fslib_c_api_CC "*.cc")

set(ZIP_URL "http://sample-lake.oss-cn-beijing.aliyuncs.com/deploy/xdl_fslib_dependency-1.0.2.tar.gz")
set(ZIP_FILE "${CMAKE_BINARY_DIR}/xdl_fslib_dependency.tar.gz")
set(DEPS_DIR "${CMAKE_BINARY_DIR}/fs_deps")

file(DOWNLOAD ${ZIP_URL} ${ZIP_FILE} SHOW_PROGRESS)

execute_process(
    COMMAND sh -c "\
    rm -rf ${DEPS_DIR} && mkdir -p ${DEPS_DIR} && \
    tar xzf ${ZIP_FILE} -C ${DEPS_DIR} && \
    mv ${DEPS_DIR}/usr/local/include/* ${DEPS_DIR}/ && \
    mv ${DEPS_DIR}/usr/local/lib64 ${DEPS_DIR} && \
    true"
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
    # relative lib
    ${DEPS_DIR}/lib64/libpangudfs_pangu_client.so.1
    ${DEPS_DIR}/lib64/libkmonitor_client_cpp.so
    # direct lib
    ${DEPS_DIR}/lib64/libfslib-framework.so
)

target_compile_definitions(fslib_c_api PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(fslib_c_api PRIVATE -std=c++17)
target_compile_options(fslib_c_api PRIVATE -g)
install(TARGETS fslib_c_api DESTINATION ${RECIS_LIB_INSTALL_DIR})
install(FILES
    # relative lib, not certainly depended
    ${DEPS_DIR}/lib64/libpangudfs_pangu_client.so.1
    ${DEPS_DIR}/lib64/libfslib_plugin_pangu.so
    ${DEPS_DIR}/lib64/libkmonitor_client_cpp.so
    ${DEPS_DIR}/lib64/libfslib_plugin_dfs.so
    # direct lib, certainly depended
    ${DEPS_DIR}/lib64/libfslib-framework.so
  DESTINATION ${RECIS_LIB_INSTALL_DIR})
