if (NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
    set(_GLIBCXX_USE_CXX11_ABI 1)
endif()

set(IMPL_LIBRARY
    PRIVATE absl::log
    PRIVATE absl::status
    PRIVATE framework
    PRIVATE ${ARROW_LIBRARY}
    PRIVATE local_orc_dlwrapper #---- TODO(xianghouyuan) 非 NEED_ARROW_17, 不要dlopen本模块
    PUBLIC open_storage_common-util
    PUBLIC open_storage_wrapper
    PUBLIC lake_plugin
)

if (INTERNAL_VERSION)
    if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
        set(IMPL_LIBRARY ${IMPL_LIBRARY}
            PUBLIC odps_wrapper
        )
    endif()
    file(GLOB arrow_reader_CC *.cc)
    set(arrow_reader_INCLUDE ${PROJECT_BINARY_DIR}/odps_sdk_cpp)
else()
    message(STATUS "INTERNAL_VERSION OFF, compiling local_orc_reader only")
    set(IMPL_LIBRARY
        PRIVATE absl::log
        PRIVATE absl::status
        PRIVATE framework
        PRIVATE ${ARROW_LIBRARY}
        PRIVATE local_orc_dlwrapper
    )
    set(arrow_reader_CC local_orc_reader.cc)
    set(arrow_reader_INCLUDE "")
endif()

add_library(arrow_reader ${arrow_reader_CC})
target_compile_definitions(arrow_reader PRIVATE _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
target_link_libraries(arrow_reader ${IMPL_LIBRARY})
target_include_directories(arrow_reader
  PRIVATE
  ${CC_BASE_INCLUDE_DIR}
  ${arrow_reader_INCLUDE}
)
