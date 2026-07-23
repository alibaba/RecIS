
file(GLOB MONITOR_CC *.cc)
add_library(monitor STATIC ${MONITOR_CC} )
target_include_directories(monitor PRIVATE ${CC_BASE_INCLUDE_DIR})
target_link_libraries(monitor PRIVATE -fPIC pthread)
