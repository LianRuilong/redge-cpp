cmake_minimum_required(VERSION 3.16)
project(testing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Building unified testing suite")

# === 依赖路径 ===
set(GTEST_ROOT "${THIRD_PARTY_INSTALL_DIR}/googletest")
set(LOGGER_ROOT "${CMAKE_SOURCE_DIR}/src/base/logger")

# === 找 GTest 和 glog ===
find_package(GTest REQUIRED PATHS ${GTEST_ROOT} NO_DEFAULT_PATH)
find_package(glog REQUIRED PATHS "${THIRD_PARTY_INSTALL_DIR}/glog" NO_DEFAULT_PATH)

# === 头文件路径 ===
include_directories(
    ${GTEST_INCLUDE_DIRS}
    ${LOGGER_ROOT}
)

# === 公共 main.cpp ===
add_library(test_main OBJECT main.cpp)

# === 添加子模块测试 ===
add_subdirectory(text_embedding)

# === 启用测试 ===
enable_testing()
