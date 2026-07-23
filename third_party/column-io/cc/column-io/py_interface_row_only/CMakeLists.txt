pybind11_add_module(py_interface NO_EXTRAS
    row_interface.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../dataset_impl/odps_open_storage_row_dataset.cc
)

target_include_directories(py_interface PRIVATE
    ${CC_BASE_INCLUDE_DIR}
    ${PROJECT_BINARY_DIR}/odps_sdk_cpp
    ${PROJECT_BINARY_DIR}/odps_sdk_cpp/storage_api/thirdparty
    ${PROJECT_BINARY_DIR}/odps_sdk_cpp/storage_api/thirdparty/cpp-httplib
    ${PROJECT_BINARY_DIR}/rapidjson_package-9df48824d6fe9e2cf449ced356c9fe1e/src/rapidjson_package/include
    ${ARROW_INCLUDE}
)

target_link_libraries(py_interface
    PRIVATE open_storage_wrapper
    PRIVATE open_storage_common-util
    PRIVATE monitor
    PRIVATE absl::log
    PRIVATE absl::log_initialize
    PRIVATE absl::strings
    PRIVATE absl::status
    PRIVATE pybind11::pybind11
    PRIVATE pybind11::module
    PRIVATE "-Wl,-rpath,'$ORIGIN/'"
)

if(_GLIBCXX_USE_CXX11_ABI EQUAL 0)
    target_compile_definitions(py_interface PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(py_interface
        PRIVATE -Wl,--start-group
        PRIVATE ${PROJECT_BINARY_DIR}/protobuf/src/protobuf/cmake/libprotobuf.a
        PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-4.9.2/arrow-1.0.0/lib64/libarrow.a
        PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libjemalloc_pic.a
        PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-4.9.2/brotli-1.0.7/lib/libbrotlienc-static.a
        PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-4.9.2/brotli-1.0.7/lib/libbrotlidec-static.a
        PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-4.9.2/brotli-1.0.7/lib/libbrotlicommon-static.a
        PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-4.9.2/zstd-1.5.7/lib/libzstd.a
        PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/lz4/lib/liblz4.a
        PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/zstd/lib/libzstd.a
        PRIVATE /opt/conda/lib/libbz2.a
        PRIVATE -Wl,--end-group
    )
else()
    target_compile_definitions(py_interface PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    if(NEED_ARROW_17)
        target_link_libraries(py_interface
            PRIVATE -Wl,--start-group
            PRIVATE ${PROJECT_BINARY_DIR}/protobuf/src/protobuf/cmake/libprotobuf.a
            PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0/lib64/libarrow.a
            PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0/lib64/libarrow_bundled_dependencies.a
            PRIVATE -Wl,--end-group
        )
    else()
        target_link_libraries(py_interface
            PRIVATE -Wl,--start-group
            PRIVATE ${PROJECT_BINARY_DIR}/protobuf/src/protobuf/cmake/libprotobuf.a
            PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libarrow.a
            PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libjemalloc_pic.a
            PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/lz4/lib/liblz4.a
            PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/zstd/lib/libzstd.a
            PRIVATE -Wl,--end-group
        )
    endif()
endif()

add_custom_target(copy_cxx_lib ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE_DIR:py_interface>/$<TARGET_FILE_NAME:py_interface>
        $<TARGET_FILE_DIR:open_storage_plugin>/$<TARGET_FILE_NAME:open_storage_plugin>
        ${CMAKE_SOURCE_DIR}/column_io/lib
)
add_dependencies(copy_cxx_lib py_interface)
