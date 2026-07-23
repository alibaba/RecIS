file(GLOB pyinterface_CC *.cc)
pybind11_add_module(py_interface NO_EXTRAS ${pyinterface_CC})

set(INTERFACE_INCLUDE_DIR
    ${CC_BASE_INCLUDE_DIR}
    ${PROJECT_BINARY_DIR}/rapidjson_package-${rapidjson_MD5}/src/rapidjson_package/include
    ${DLPACK_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/odps_sdk_cpp)

set(COPY_SOURCE $<TARGET_FILE_DIR:py_interface>/$<TARGET_FILE_NAME:py_interface>)
if (INTERNAL_VERSION)
    if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
        set(INTERFACE_LIBRARY
            PRIVATE "-Wl,--whole-archive"
            PRIVATE dataset
            PRIVATE dataset_impl
            PRIVATE lake_plugin
            PRIVATE open_storage_wrapper
            PRIVATE "-Wl,--no-whole-archive")

        target_compile_definitions(py_interface PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
        set(COPY_SOURCE
            ${COPY_SOURCE}
            # $<TARGET_FILE_DIR:local_orc_plugin>/$<TARGET_FILE_NAME:local_orc_plugin> # abi=0时强制arrow1, 此插件无意义
            $<TARGET_FILE_DIR:open_storage_plugin>/$<TARGET_FILE_NAME:open_storage_plugin>
            $<TARGET_FILE_DIR:odps_plugin>/$<TARGET_FILE_NAME:odps_plugin>
            $<TARGET_FILE_DIR:plugin>/$<TARGET_FILE_NAME:plugin>
            )
        
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
            PRIVATE "-Wl,-rpath,'$ORIGIN/'")
        
        set(lake_sdk_LIBPAIMON_NAME "libpaimon.so")
    else()
        set(INTERFACE_LIBRARY
            PRIVATE "-Wl,--whole-archive"
            PRIVATE dataset
            PRIVATE dataset_impl
            PRIVATE open_storage_wrapper
            PRIVATE "-Wl,--no-whole-archive")
        #FLAG: _GLIBCXX_USE_CXX11_ABI默认不传给g++ -o py_interface.o, 需要通过target_compile_definitions传递编译参数
        target_compile_definitions(py_interface PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
        set(COPY_SOURCE
            ${COPY_SOURCE}
            $<TARGET_FILE_DIR:local_orc_plugin>/$<TARGET_FILE_NAME:local_orc_plugin>
            $<TARGET_FILE_DIR:open_storage_plugin>/$<TARGET_FILE_NAME:open_storage_plugin>
            $<TARGET_FILE_DIR:plugin>/$<TARGET_FILE_NAME:plugin>
            )
        if(NEED_ARROW_17)
            target_link_libraries(py_interface
                PRIVATE -Wl,--start-group
                PRIVATE ${PROJECT_BINARY_DIR}/protobuf/src/protobuf/cmake/libprotobuf.a
                PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0/lib64/libarrow.a
                PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0/lib64/libarrow_bundled_dependencies.a
                PRIVATE -Wl,--end-group
                PRIVATE "-Wl,-rpath,'$ORIGIN/'"
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
                PRIVATE "-Wl,-rpath,'$ORIGIN/'"
            )
        endif()


        set(lake_sdk_LIBPAIMON_NAME "libpaimon.so")
    endif()
    set(INTERFACE_LIBRARY
            ${INTERFACE_LIBRARY}
            PRIVATE absl::statusor 
            PRIVATE absl::strings 
            PRIVATE absl::status 
            PRIVATE absl::log_initialize
            # absl::flat_hash_map pulls in absl::raw_hash_set for out-of-line
            # symbols (e.g. container_internal::kEmptyGroup). Required by
            # dataset/map_dataset_sample_filter.cc which instantiates a
            # flat_hash_map<string, flat_hash_set<string>> filter_dict_.
            PRIVATE absl::flat_hash_map
            PRIVATE pybind11::pybind11
            PRIVATE pybind11::module
            PRIVATE pybind11::embed
            PRIVATE "-rdynamic")
    # lake sdk 在编译时才会会被构建，在 configure 时用 GLOB 找不到构建产物，所以这里改成手动指定
    set(lake_sdk_files
		"${lake_sdk_LIBRARY_BASE}/lib_lake_IO.so"
		"${lake_sdk_LIBRARY_BASE}/liblz4.so"
		"${lake_sdk_LIBRARY_BASE}/liblzma.so"
		"${lake_sdk_LIBRARY_BASE}/${lake_sdk_LIBPAIMON_NAME}"
		"${lake_sdk_LIBRARY_BASE}/libzstd.so")
	message(STATUS "lake_sdk_files are ${lake_sdk_files}")
    set(COPY_SOURCE
        ${COPY_SOURCE}
        ${lake_sdk_files})
	add_dependencies(py_interface lake)
else()
    set(INTERFACE_LIBRARY
        PRIVATE "-Wl,--whole-archive"
        PRIVATE dataset
        PRIVATE dataset_impl
        PRIVATE "-Wl,--no-whole-archive")
    set(INTERFACE_LIBRARY
        ${INTERFACE_LIBRARY}
        PRIVATE absl::statusor
        PRIVATE absl::strings
        PRIVATE absl::status
        PRIVATE absl::log_initialize
        PRIVATE absl::flat_hash_map
        PRIVATE pybind11::pybind11
        PRIVATE pybind11::module
        PRIVATE pybind11::embed
        PRIVATE "-rdynamic")
    set(COPY_SOURCE
        ${COPY_SOURCE}
        $<TARGET_FILE_DIR:local_orc_plugin>/$<TARGET_FILE_NAME:local_orc_plugin>
    )
endif()
target_link_libraries(py_interface ${INTERFACE_LIBRARY})
target_link_libraries(py_interface PRIVATE absl::status absl::strings absl::log_initialize absl::flat_hash_map pybind11::pybind11 pybind11::module pybind11::embed)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fvisibility.ver "{
    # global:
    
    local:
      recis::monitor::*;
      std*recis::monitor::*;
      *recis_monitor_*;
      *RECIS_MONITOR_*;
    };
")
target_link_libraries(py_interface PRIVATE monitor
  # Hide recis::monitor symbols for avoiding conflicts with main recis process symbols
  -Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/fvisibility.ver
)
target_include_directories(py_interface PRIVATE ${INTERFACE_INCLUDE_DIR})

add_custom_target(copy_cxx_lib ALL 
    COMMAND ${CMAKE_COMMAND} -E copy 
        ${COPY_SOURCE}
        ${CMAKE_SOURCE_DIR}/column_io/lib
    )
add_dependencies(copy_cxx_lib py_interface)
add_dependencies(copy_cxx_lib local_orc_plugin)
