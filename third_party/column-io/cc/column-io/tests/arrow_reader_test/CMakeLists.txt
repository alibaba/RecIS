add_executable(abstract_reader_test abstract_reader_test.cc)
# 避免odps_sdk_cpp和algo-sdk的GTest冲突
target_include_directories(abstract_reader_test BEFORE  PRIVATE
    $<TARGET_PROPERTY:gtest,INTERFACE_INCLUDE_DIRECTORIES>
)
target_include_directories(abstract_reader_test PRIVATE ${CC_BASE_INCLUDE_DIR})
target_link_libraries(abstract_reader_test
  PRIVATE gtest
  PRIVATE gtest_main
  PRIVATE framework
  PRIVATE ${ARROW_LIBRARY}
)
gtest_discover_tests(abstract_reader_test)
