# GPU Platform Auto-Detection and Configuration
# This module provides a function to automatically detect and configure GPU platform

option(NEED_CPU_ONLY "Build without any GPU support (no CUDA/ROCm)" OFF)
option(NEED_ROW_ONLY "Minimal row-reader build (implies CPU_ONLY)" OFF)

if(NEED_ROW_ONLY)
    set(NEED_CPU_ONLY ON CACHE BOOL "Forced ON by ROW_ONLY" FORCE)
endif()

# Function: detect_gpu_platform
#
# Automatically detects whether to use ROCm/HIP or CUDA
# Sets USE_ROCM variable accordingly
#
function(detect_gpu_platform)
    if(NEED_CPU_ONLY)
        message(STATUS "==> CPU_ONLY mode enabled, skipping GPU detection")
        set(USE_ROCM OFF PARENT_SCOPE)
        return()
    endif()
    if(NOT DEFINED USE_ROCM)
        # Try to detect ROCm/HIP environment
        set(ROCM_DETECTED FALSE)
        
        # Check 1: ROCM_PATH environment variable
        if(DEFINED ENV{ROCM_PATH} AND EXISTS "$ENV{ROCM_PATH}")
            message(STATUS "Detected ROCM_PATH: $ENV{ROCM_PATH}")
            set(ROCM_DETECTED TRUE)
        endif()
        
        # Check 2: Look for hipcc compiler
        if(NOT ROCM_DETECTED)
            find_program(HIPCC_FOUND hipcc PATHS /opt/rocm/bin /usr/local/rocm/bin)
            if(HIPCC_FOUND)
                message(STATUS "Detected hipcc at: ${HIPCC_FOUND}")
                set(ROCM_DETECTED TRUE)
            endif()
        endif()
        
        if(ROCM_DETECTED)
            message(STATUS "==> Auto-detected ROCm, will use HIP")
            set(USE_ROCM ON PARENT_SCOPE)

            if(DEFINED ENV{ROCM_PATH})
                set(CMAKE_CXX_COMPILER "$ENV{ROCM_PATH}/bin/hipcc" PARENT_SCOPE)
            else()
                set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc" PARENT_SCOPE)
            endif()
        else()
            message(STATUS "==> No ROCm detected, will use CUDA")
            set(USE_ROCM OFF PARENT_SCOPE)
        endif()
    endif()
endfunction()

# Function: configure_gpu_platform
#
# Configures the GPU platform (CUDA or ROCm) based on USE_ROCM variable
# Must be called after project() declaration
#
function(configure_gpu_platform)
    if(NEED_CPU_ONLY)
        message(STATUS "CPU_ONLY mode: skipping GPU configuration")
        add_definitions(-DCPU_ONLY)
        return()
    endif()

    if(USE_ROCM)
        # Configure ROCm/HIP
        if(NOT DEFINED ROCM_PATH)
            if(DEFINED ENV{ROCM_PATH})
                set(ROCM_PATH $ENV{ROCM_PATH})
            else()
                set(ROCM_PATH "/opt/rocm")
            endif()
        endif()
        
        list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
        find_package(hip REQUIRED CONFIG PATHS ${ROCM_PATH})
        
        # GPU targets
        if(NOT DEFINED GPU_TARGETS)
            set(GPU_TARGETS "gfx942" CACHE STRING "ROCm GPU architectures")
        endif()
        
        # Set HIP architectures
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS} PARENT_SCOPE)
        
        set(HIP_INCLUDE_DIRS ${HIP_INCLUDE_DIRS} PARENT_SCOPE)
        set(HIP_LIBRARIES ${HIP_LIBRARIES} PARENT_SCOPE)
        set(ROCM_PATH ${ROCM_PATH} PARENT_SCOPE)

        set(HIP_INCLUDE_DIRS ${HIP_INCLUDE_DIRS} CACHE PATH "HIP include directories")
        set(HIP_LIBRARIES ${HIP_LIBRARIES} CACHE STRING "HIP libraries")

        add_definitions(-DUSE_ROCM -D__HIP_PLATFORM_AMD__)

        message(STATUS "ROCm configuration:")
        message(STATUS "  Path: ${ROCM_PATH}")
        message(STATUS "  GPU Targets: ${GPU_TARGETS}")
        message(STATUS "  HIP Include: ${HIP_INCLUDE_DIRS}")
        message(STATUS "  HIP Libraries: ${HIP_LIBRARIES}")
        message(STATUS "  CXX Compiler: ${CMAKE_CXX_COMPILER}")
        message(STATUS "  Note: hipcc is required for .cc files to support HIP intrinsics")
    else()
        find_package(CUDA REQUIRED)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "70;80;86;90" PARENT_SCOPE)
        endif()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3" PARENT_SCOPE)

        # CUDA 12.6+ moved CUB/Thrust headers into cccl/ subdirectory.
        # Add the cccl include path so that `#include <cub/...>` reference still works across multiple CUDA versions.
        foreach(_cuda_inc_dir ${CUDA_INCLUDE_DIRS})
            if(EXISTS "${_cuda_inc_dir}/cccl")
                list(APPEND CUDA_INCLUDE_DIRS "${_cuda_inc_dir}/cccl")
                message(STATUS "  Added CCCL include path: ${_cuda_inc_dir}/cccl")
            endif()
        endforeach()

        set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)
        set(CUDA_LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)
        set(CUDA_CUBLAS_LIBRARIES ${CUDA_CUBLAS_LIBRARIES} PARENT_SCOPE)
        set(CUDA_VERSION ${CUDA_VERSION} PARENT_SCOPE)

        set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} CACHE PATH "CUDA include directories")
        set(CUDA_LIBRARIES ${CUDA_LIBRARIES} CACHE STRING "CUDA libraries")

        message(STATUS "CUDA configuration:")
        message(STATUS "  CUDA Version: ${CUDA_VERSION}")
        message(STATUS "  CUDA Include: ${CUDA_INCLUDE_DIRS}")
        message(STATUS "  CUDA Libraries: ${CUDA_LIBRARIES}")
        message(STATUS "  Static Runtime: ${CUDA_USE_STATIC_CUDA_RUNTIME}")
    endif()
endfunction()
