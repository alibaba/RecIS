set(IMPL_LIBRARY
    PUBLIC absl::log
    PUBLIC absl::status
    PUBLIC framework
	PUBLIC rapidjson
    ${ARROW_LIBRARY}
    )
if (INTERNAL_VERSION)
    file(GLOB dataset_impl_CC *.cc)
	#list(FILTER dataset_impl_CC EXCLUDE REGEX ".*local_orc_dataset\\.cc")
else()
    set(dataset_impl_CC
        local_rb_stream_dataset.cc
        local_orc_dataset.cc
        schema_parser.cc)
endif(INTERNAL_VERSION)
add_library(dataset_impl ${dataset_impl_CC})

if (INTERNAL_VERSION)
    if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
        set(IMPL_LIBRARY
            ${IMPL_LIBRARY}
            PUBLIC odps_wrapper)
        target_compile_definitions(dataset_impl PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    else()
        target_compile_definitions(dataset_impl PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    endif()
    set(IMPL_LIBRARY
        ${IMPL_LIBRARY}
        PUBLIC open_storage_common-util
        PUBLIC open_storage_wrapper
        PUBLIC lake_plugin
        PUBLIC arrow_reader)
else()
    target_compile_definitions(dataset_impl PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    set(IMPL_LIBRARY
        ${IMPL_LIBRARY}
        PUBLIC arrow_reader)
endif(INTERNAL_VERSION)

target_link_libraries(dataset_impl ${IMPL_LIBRARY})
target_include_directories(dataset_impl
  PRIVATE
  ${CC_BASE_INCLUDE_DIR}
  ${PROJECT_BINARY_DIR}/odps_sdk_cpp
)

if (INTERNAL_VERSION)
  target_link_libraries(dataset_impl PUBLIC pybind11::pybind11)
endif()
