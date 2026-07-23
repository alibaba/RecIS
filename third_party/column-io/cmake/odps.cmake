include(ExternalProject)
set(algo_sdk_URL "")
set(algo_sdk_MD5 "f46ece486e9d1285281dac01454cea7f")
ExternalProject_Add(algo_sdk
	                PREFIX algo-sdk-${algo_sdk_MD5}
                    URL ${algo_sdk_URL}
                    URL_MD5 ${algo_sdk_MD5}
                    CONFIGURE_COMMAND bash -c "echo skipping configuration step"
					DOWNLOAD_EXTRACT_TIMESTAMP true
                    BUILD_COMMAND bash -c "echo skipping build step"
                    BUILD_IN_SOURCE true
                    INSTALL_COMMAND bash -c "echo skipping install step")

# Record the ODPS SDK version
file(WRITE ${CMAKE_SOURCE_DIR}/column_io/ODPS_SDK_VERSION "${algo_sdk_URL}\n")

# algo_sdk bundles a stale GTest (headers + static libs) that conflicts with
# the FetchContent GTest 1.17 used by unit tests. Removes it!
ExternalProject_Add_Step(algo_sdk remove_stale_gtest
    COMMENT "Removing stale GTest headers/libs bundled in algo_sdk"
    COMMAND ${CMAKE_COMMAND} -E rm -rf <SOURCE_DIR>/sdk/include/gtest
    COMMAND ${CMAKE_COMMAND} -E rm -f  <SOURCE_DIR>/sdk/lib/libgtest.a
    COMMAND ${CMAKE_COMMAND} -E rm -f  <SOURCE_DIR>/sdk/lib/libgtest_main.a
    DEPENDEES install
    ALWAYS FALSE
)
ExternalProject_Get_Property(algo_sdk SOURCE_DIR)
set(algo_sdk_INCLUDE_BASE ${SOURCE_DIR}/sdk/include)
set(algo_sdk_LIBRARY_BASE ${SOURCE_DIR}/sdk/lib)


add_library(arrow INTERFACE)
target_include_directories(arrow INTERFACE ${algo_sdk_INCLUDE_BASE})
message("algo_sdk_LIBRARY_BASE is ${algo_sdk_LIBRARY_BASE}")
target_link_directories(arrow INTERFACE /worker)
target_link_libraries(arrow INTERFACE -l:libarrow.so.100)
add_dependencies(arrow algo_sdk)

#algo_package
add_library(algo_package INTERFACE)
target_include_directories(algo_package INTERFACE ${algo_sdk_INCLUDE_BASE} ${algo_sdk_INCLUDE_BASE}/apsara)
target_link_directories(algo_package INTERFACE ${algo_sdk_LIBRARY_BASE})
add_dependencies(algo_package algo_sdk)
