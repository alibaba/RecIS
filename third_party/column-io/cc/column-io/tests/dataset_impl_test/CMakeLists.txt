if (INTERNAL_VERSION)
    pybind11_add_module(odps_open_storage_row_dataset_test NO_EXTRAS
        odps_open_storage_row_dataset_test.cc
        ${CMAKE_SOURCE_DIR}/cc/column-io/py_interface/converter.cc)
    set(BUILT ${PROJECT_BINARY_DIR})
    target_include_directories(odps_open_storage_row_dataset_test
        PRIVATE ${CC_BASE_INCLUDE_DIR}
        PRIVATE ${BUILT}/odps_sdk_cpp
        PRIVATE ${BUILT}/odps_sdk_cpp/storage_api/thirdparty
        PRIVATE ${BUILT}/odps_sdk_cpp/storage_api/thirdparty/cpp-httplib
    )
    target_compile_definitions(odps_open_storage_row_dataset_test
        PRIVATE _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
    target_link_libraries(odps_open_storage_row_dataset_test
        PRIVATE dataset
        PRIVATE dataset_impl
        PRIVATE framework
        PRIVATE ${ARROW_LIBRARY}
        PRIVATE pybind11::pybind11
        PRIVATE pybind11::module
    )

    # Copy the built .so to column_io/lib/ alongside py_interface so Python
    # tests can import it directly.
    add_custom_command(TARGET odps_open_storage_row_dataset_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:odps_open_storage_row_dataset_test>
            ${CMAKE_SOURCE_DIR}/column_io/lib/$<TARGET_FILE_NAME:odps_open_storage_row_dataset_test>
    )
endif()
