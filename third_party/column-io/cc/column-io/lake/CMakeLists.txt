file(GLOB lake_CC *.cc)
add_library(lake_plugin ${lake_CC})
if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
    target_compile_definitions(lake_plugin PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
else()
    target_compile_definitions(lake_plugin PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
endif()
target_link_libraries(dataset_impl PUBLIC ${ARROW_LIBRARY})

if(_GLIBCXX_USE_CXX11_ABI EQUAL 0)
    set(ODPS_THIRD_PARTY_SUFFIX "gcc-4.9.2")
else()
    set(ODPS_THIRD_PARTY_SUFFIX "gcc-8")
endif()

target_include_directories(lake_plugin 
  PUBLIC 
    ${CC_BASE_INCLUDE_DIR}
    ${ARROW_INCLUDE}
)
