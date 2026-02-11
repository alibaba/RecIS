cmake_minimum_required(VERSION 3.28)
project(recis_c_api VERSION 1.0)
set(RECIS_LIB_INSTALL_DIR "." CACHE STRING "default dir to install capi")
add_subdirectory(csrc)
