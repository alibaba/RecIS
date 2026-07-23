add_library(open_storage_common-util
    common_util.cc
    logging.cc
)
if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
    set(CPACK_RPM_PACKAGE_REQUIRES 
        "kmonitor_adapter = 0.0.2"
    )
else()
    set(CPACK_RPM_PACKAGE_REQUIRES 
        "kmonitor_adapter = 0.0.2"
    )
endif()
target_include_directories(open_storage_common-util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
