message(STATUS "Use C++11ABI=${_GLIBCXX_USE_CXX11_ABI} in unit tests module")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}) # add_subdirectory 之前设置

# When BUILD_TESTING is on, remove all stale GTest headers and libraries bundled
# in third-party SDKs (odps_sdk_cpp, algo-sdk, etc.) to prevent them from
# shadowing the FetchContent GTest 1.17 via include-path priority conflicts.
if(BUILD_TESTING)
    message(STATUS "BUILD_TESTING is on, removing stale GTest from third-party SDKs (CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR})")

    # Find all gtest.h files anywhere under the build directory, then filter out
    # the FetchContent copy (_deps/) which is the one we actually want to keep.
    # For each stale gtest.h, remove its containing gtest/ directory tree.
    file(GLOB_RECURSE _all_gtest_headers "${CMAKE_BINARY_DIR}/gtest.h")
    set(_stale_gtest_dirs_to_remove "")
    foreach(_header IN LISTS _all_gtest_headers)
        # Skip the FetchContent GTest (the one we want to keep)
        string(FIND "${_header}" "_deps/" _is_fetchcontent)
        if(_is_fetchcontent GREATER -1)
            continue()
        endif()
        # Walk up from gtest.h to the containing gtest/ directory
        get_filename_component(_gtest_dir "${_header}" DIRECTORY)
        list(APPEND _stale_gtest_dirs_to_remove "${_gtest_dir}")
    endforeach()
    list(REMOVE_DUPLICATES _stale_gtest_dirs_to_remove)

    foreach(_dir IN LISTS _stale_gtest_dirs_to_remove)
        message(STATUS "  Removing stale GTest dir: ${_dir}")
        file(REMOVE_RECURSE "${_dir}")
    endforeach()

    # Also remove stale GTest static libraries from algo-sdk
    file(GLOB_RECURSE _all_gtest_libs "${CMAKE_BINARY_DIR}/libgtest*.a")
    foreach(_lib IN LISTS _all_gtest_libs)
        string(FIND "${_lib}" "_deps/" _is_fetchcontent)
        string(FIND "${_lib}" "lib/libgtest" _is_project_lib)
        # Skip FetchContent GTest and the project's own build output in lib/
        if(_is_fetchcontent GREATER -1)
            continue()
        endif()
        # Check if this is the project's own lib/libgtest.a (built from FetchContent)
        get_filename_component(_lib_dir "${_lib}" DIRECTORY)
        if("${_lib_dir}" STREQUAL "${CMAKE_BINARY_DIR}/lib")
            continue()
        endif()
        message(STATUS "  Removing stale GTest lib: ${_lib}")
        file(REMOVE "${_lib}")
    endforeach()
endif()

add_subdirectory(arrow_reader_test)
add_subdirectory(dataset_test)
add_subdirectory(dataset_impl_test)
