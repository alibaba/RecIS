file(GLOB plugin_CC cast_to_torch_tensor.cc aot_run_impl.cc)
file(GLOB plugin_aoti_load_CC aoti_loader.cc)
add_library(plugin SHARED ${plugin_CC})
add_library(plugin_aoti_load STATIC ${plugin_aoti_load_CC})

set(PYTHON_EXECUTABLE "" CACHE FILEPATH "Python interpreter to use for build")
if(NOT PYTHON_EXECUTABLE)
  message(FATAL_ERROR "PYTHON_EXECUTABLE is not set. Pass -DPYTHON_EXECUTABLE=... to cmake")
endif()

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
          "import torch, os; print(os.path.join(torch.__path__[0], 'share', 'cmake'))"
  OUTPUT_VARIABLE TORCH_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _result
)

if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Failed to query torch cmake dir using ${PYTHON_EXECUTABLE}")
endif()



if(NEED_CPU_ONLY)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import torch, os; print(os.path.join(torch.__path__[0], 'include'))"
    OUTPUT_VARIABLE _TORCH_INC1
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import torch, os; print(os.path.join(torch.__path__[0], 'include', 'torch', 'csrc', 'api', 'include'))"
    OUTPUT_VARIABLE _TORCH_INC2
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import torch, os; print(os.path.join(torch.__path__[0], 'lib'))"
    OUTPUT_VARIABLE _TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(TORCH_INCLUDE_DIRS "${_TORCH_INC1};${_TORCH_INC2}")
  set(TORCH_LIBRARIES "${_TORCH_LIB_DIR}/libtorch.so;${_TORCH_LIB_DIR}/libtorch_cpu.so;${_TORCH_LIB_DIR}/libc10.so;${_TORCH_LIB_DIR}/libtorch_python.so")
else()
  find_package(Torch REQUIRED PATHS ${TORCH_CMAKE_DIR} NO_DEFAULT_PATH)
endif()
target_compile_options(plugin PRIVATE -fvisibility=default)
if(USE_ROCM)
  target_compile_options(plugin PRIVATE -D_LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE)
endif()

set(PLUGIN_INCLUDE_DIR
    ${CC_BASE_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${DLPACK_INCLUDE_DIR}
    )

set(PLUGIN_AOTI_LOAD_INCLUDE_DIR
    ${CC_BASE_INCLUDE_DIR}
    ${DLPACK_INCLUDE_DIR}
)

target_include_directories(plugin 
    PUBLIC 
    ${PLUGIN_INCLUDE_DIR}
)

target_include_directories(plugin_aoti_load
    PUBLIC 
    ${PLUGIN_AOTI_LOAD_INCLUDE_DIR}
)

target_link_libraries(plugin 
    PUBLIC
    ${TORCH_LIBRARIES}
    framework
    absl::strings
    absl::status
    pybind11::pybind11
    pybind11::module
    pybind11::embed
)

target_link_libraries(plugin_aoti_load
    PUBLIC
    framework
    absl::strings
    absl::status
)
