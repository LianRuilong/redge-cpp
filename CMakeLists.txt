cmake_minimum_required(VERSION 3.16)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置安装后的运行时搜索路径
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

message("CMake source dir is: ${CMAKE_SOURCE_DIR}")

# 安装路径设置（可通过 -DCMAKE_INSTALL_PREFIX=xxx 覆盖）
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix" FORCE)
endif()

message("Install path is: ${CMAKE_INSTALL_PREFIX}")

# 第三方库安装路径
set(THIRD_PARTY_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party_install)
message("Third-party install dir is: ${THIRD_PARTY_INSTALL_DIR}")

# 构建第三方依赖
message(STATUS "Building third-party dependencies...")
message("Build third_party command is: ${CMAKE_SOURCE_DIR}/third_party/build_third_party.sh ${THIRD_PARTY_INSTALL_DIR}")
execute_process(
    COMMAND ${CMAKE_SOURCE_DIR}/third_party/build_third_party.sh ${THIRD_PARTY_INSTALL_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party
    RESULT_VARIABLE BUILD_THIRD_PARTY_RESULT
)

if (BUILD_THIRD_PARTY_RESULT)
    message(FATAL_ERROR "Failed to build third-party dependencies!")
endif()

# 添加第三方 include 和 lib 路径
include_directories(${THIRD_PARTY_INSTALL_DIR}/include)
link_directories(${THIRD_PARTY_INSTALL_DIR}/lib)

# 添加源码子目录
add_subdirectory(src/base/logger)
add_subdirectory(src/components/text_embedding)

# 添加测试
enable_testing()
add_subdirectory(testing)

# 安装工程级别元信息（可选）
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MyProjectConfigVersion.cmake"
  VERSION 1.0.0
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
  DESTINATION .
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/MyProjectConfigVersion.cmake"
  DESTINATION lib/cmake/MyProject
)

# 安装 third_party_install 下的库和头文件
install(DIRECTORY ${THIRD_PARTY_INSTALL_DIR}/include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${THIRD_PARTY_INSTALL_DIR}/lib/
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*" PATTERN "*.a" PATTERN "*.dylib"
)
