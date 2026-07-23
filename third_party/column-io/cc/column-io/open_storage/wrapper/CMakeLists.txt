set(BUILT ${PROJECT_BINARY_DIR})
file(GLOB OPEN_STORAGE_WRAPPER_CC_FILTERED *.cc)
set(OPEN_STORAGE_WRAPPER_CC "")

foreach(file ${OPEN_STORAGE_WRAPPER_CC_FILTERED})
    if(NOT file MATCHES "test")
        list(APPEND OPEN_STORAGE_WRAPPER_CC ${file})
    endif()
endforeach()

message(STATUS "Filtered files: ${OPEN_STORAGE_WRAPPER_CC}")

add_library(open_storage_wrapper ${OPEN_STORAGE_WRAPPER_CC})
target_compile_definitions(open_storage_wrapper
  PUBLIC TF_ENABLE_OPEN_STORAGE_COLUMN
)
if(NEED_ARROW_17)
    set(arrow_include_dir ${BUILT}/odps_sdk_cpp/third_party/gcc-8/arrow-17.0.0/include)
else()
    set(arrow_include_dir ${BUILT}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/include)
endif()
target_include_directories(open_storage_wrapper
  PRIVATE
  ${CC_BASE_INCLUDE_DIR}
  ${BUILT}/odps_sdk_cpp
  ${BUILT}/odps_sdk_cpp/storage_api/thirdparty
  ${BUILT}/odps_sdk_cpp/storage_api/thirdparty/cpp-httplib
  ${ARROW_INCLUDE}
)
target_link_libraries(open_storage_wrapper
  PRIVATE
  open_storage_common-util
  open_storage_plugin
)
