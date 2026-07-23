add_executable(formater_test formater_test.cc)
# 避免odps_sdk_cpp和algo-sdk的GTest冲突
target_include_directories(formater_test BEFORE PRIVATE
    $<TARGET_PROPERTY:gtest,INTERFACE_INCLUDE_DIRECTORIES>
)
target_include_directories(formater_test PRIVATE ${CC_BASE_INCLUDE_DIR})
target_link_libraries(formater_test
  PRIVATE gtest
  PRIVATE gtest_main
  PRIVATE dataset
  PRIVATE framework
  PRIVATE ${ARROW_LIBRARY}
)
gtest_discover_tests(formater_test)
