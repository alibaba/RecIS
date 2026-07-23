#TODO: 区分 INTERNAL_VERSION
add_library(local_orc_plugin SHARED
    local_orc_reader.cc
    local_orc_capi.cc
)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fvisibility.ver "{
global:
  # 隐藏py_interface.so所需的 CAPI_LOCAL_ORC_* 以外的所有符号, 尤其注意避免PROTOBUF冲突
  CAPI_LOCAL_ORC_*;

local:
  *;
};
")

target_compile_definitions(local_orc_plugin PUBLIC _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
target_compile_options(local_orc_plugin
    PRIVATE
        -fvisibility=hidden
)
target_include_directories(local_orc_plugin
  PUBLIC
    ${CC_BASE_INCLUDE_DIR}
    ${ARROW17_ORC_INCLUDE}
)
target_link_libraries(local_orc_plugin 
  PRIVATE -fPIC

  -lpthread  -ldl
  -Wl,--start-group
  ${ARROW17_ORC_LIBRARY}
  # monitor
  $<$<BOOL:$<NOT:$<STREQUAL:${_GLIBCXX_USE_CXX11_ABI},0>>>:-static-libgcc>
  $<$<BOOL:$<NOT:$<STREQUAL:${_GLIBCXX_USE_CXX11_ABI},0>>>:-static-libstdc++>
  -Wl,--end-group
  PRIVATE "-Wl,-rpath,'$ORIGIN/'"
  -Wl,--exclude-libs,ALL
  -Wl,--as-needed
  -Wl,-z,defs
  -Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/fvisibility.ver
  PRIVATE absl::log
)


#-------- local_orc_dlwrapper --------#
add_library(local_orc_dlwrapper STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/dlwrapper_local_orc.cc
)
target_include_directories(local_orc_dlwrapper PUBLIC
    ${CC_BASE_INCLUDE_DIR}
    ${ARROW_INCLUDE}
)
target_compile_definitions(local_orc_dlwrapper PRIVATE _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
target_link_libraries(local_orc_dlwrapper 
	PRIVATE ${ARROW_LIBRARY}
	)
