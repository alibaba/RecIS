if (INTERNAL_VERSION)
  set(GOOGLETEST_URL "https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz")
  set(GOOGLETEST_MD5 b6f100bc2a5853a48046aa168ececf84)
else()
  set(GOOGLETEST_URL "https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz")
  set(GOOGLETEST_MD5 b6f100bc2a5853a48046aa168ececf84)
endif(INTERNAL_VERSION)


include(FetchContent)

# 方式1：在线拉 gtest（最通用）
FetchContent_Declare(
  googletest
  URL ${GOOGLETEST_URL}
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Ensure gtest and all consumers are compiled with the same ABI as the project
target_compile_definitions(gtest      PUBLIC _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
target_compile_definitions(gtest_main PUBLIC _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})

include(GoogleTest)
