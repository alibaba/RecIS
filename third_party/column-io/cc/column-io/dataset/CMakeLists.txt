file(GLOB dataset_CC *.cc)
add_library(dataset ${dataset_CC})

set(DATASET_LIBRARY
    PUBLIC framework
    PRIVATE absl::strings
    PRIVATE absl::log
    PRIVATE absl::log_globals
    ${ARROW_LIBRARY}
    PRIVATE plugin_aoti_load
    )

if (INTERNAL_VERSION)
    if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
        target_compile_definitions(dataset PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
    else()
        target_compile_definitions(dataset PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    endif()
endif()
target_link_libraries(dataset ${DATASET_LIBRARY})
target_include_directories(dataset PRIVATE ${CC_BASE_INCLUDE_DIR})
target_compile_options(dataset PUBLIC "-fvisibility=hidden")
