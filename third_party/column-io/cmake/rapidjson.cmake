include(ExternalProject)
# patch for rapidjson-v1.1.0, see https://github.com/Tencent/rapidjson/commit/3b2441b87f99ab65f37b141a7b548ebadb607b96
set(rapidjson_URL "https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz")
set(rapidjson_MD5 "badd12c511e081fec6c89c43a7027bce")
ExternalProject_Add(rapidjson_package
	                PREFIX  rapidjson_package-${rapidjson_MD5}
					URL ${rapidjson_URL}
					URL_MD5 ${rapidjson_MD5}
                    CONFIGURE_COMMAND bash -c "echo skipping configuration step"
					          DOWNLOAD_EXTRACT_TIMESTAMP true
                    BUILD_COMMAND bash -c "echo skipping build step"
                    BUILD_IN_SOURCE true
                    INSTALL_COMMAND bash -c "echo skipping install step")
ExternalProject_Get_Property(rapidjson_package SOURCE_DIR)
add_library(rapidjson INTERFACE)
message("SOURCE_DIR of rapidjson_package is ${SOURCE_DIR}")
target_include_directories(rapidjson INTERFACE ${SOURCE_DIR}/include)
add_dependencies(rapidjson rapidjson_package)
