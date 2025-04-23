cmake_minimum_required(VERSION 3.16)
project(test_embedding)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# === 测试可执行文件 1: 准确率测试 ===
add_executable(text_embedding_accuracy text_embedding_onnx_accuracy.cpp)
target_link_libraries(text_embedding_accuracy
    text_embedding
    gtest
    gtest_main
)
set_target_properties(text_embedding_accuracy PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)
install(TARGETS text_embedding_accuracy DESTINATION bin)
add_test(NAME text_embedding_accuracy_run COMMAND text_embedding_accuracy)

# === 测试可执行文件 2: QPS 基准测试 ===
add_executable(text_embedding_benchmark text_embedding_onnx_benchmark.cpp)
target_link_libraries(text_embedding_benchmark
    text_embedding
    gtest
    gtest_main
)
set_target_properties(text_embedding_benchmark PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)
install(TARGETS text_embedding_benchmark DESTINATION bin)
add_test(NAME text_embedding_benchmark_run COMMAND text_embedding_benchmark)

# === 安装 Python 脚本（可选）===
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_onnx_embedding.py
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/scripts)

# === 启用测试 ===
enable_testing()
