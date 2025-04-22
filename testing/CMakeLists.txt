cmake_minimum_required(VERSION 3.16)
project(test_embedding)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 打印用于调试的信息
message(STATUS "Building test_embedding")
message(STATUS "THIRD_PARTY_INSTALL_DIR: ${THIRD_PARTY_INSTALL_DIR}")

# === GoogleTest 依赖 ===
set(GTEST_ROOT "${THIRD_PARTY_INSTALL_DIR}/googletest")
find_package(GTest REQUIRED PATHS ${GTEST_ROOT} NO_DEFAULT_PATH)
include_directories(${GTEST_INCLUDE_DIRS})

# === Python 可执行文件路径 (用于调用脚本) ===
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# === 头文件路径 ===
include_directories(
    ${CMAKE_SOURCE_DIR}/src/components/text_embedding
)

# === 源文件 ===
set(TEST_EMBEDDING_SRC test_onnx_embedding.cpp)

# === 可执行文件 ===
add_executable(test_embedding ${TEST_EMBEDDING_SRC})

# === 链接库 ===
target_link_libraries(test_embedding
    text_embedding
    gtest
    gtest_main
)

# === 设置 RPATH 以便运行时能找到共享库 ===
set_target_properties(test_embedding PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

# === 安装测试可执行文件 ===
install(TARGETS test_embedding DESTINATION bin)

# === 添加测试，支持 `ctest` 调用 ===
enable_testing()
add_test(NAME test_embedding_run COMMAND test_embedding)

# === 拷贝 Python 测试脚本 (如果需要安装脚本也可以加) ===
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_onnx_embedding.py
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/scripts)
