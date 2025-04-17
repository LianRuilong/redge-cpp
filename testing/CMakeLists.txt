cmake_minimum_required(VERSION 3.16)
project(test_embedding)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 打印用于调试的信息
message(STATUS "Building test_embedding")
message(STATUS "THIRD_PARTY_INSTALL_DIR: ${THIRD_PARTY_INSTALL_DIR}")

# 头文件路径
include_directories(
    ${THIRD_PARTY_INSTALL_DIR}/onnxruntime/include
    ${CMAKE_SOURCE_DIR}/src/components/text_embedding
)

# 测试源文件
set(TEST_EMBEDDING_SRC test_embedding.cpp)

# 生成可执行文件
add_executable(test_embedding ${TEST_EMBEDDING_SRC})

# 链接 text_embedding 动态库
target_link_libraries(test_embedding text_embedding)

# 设置 RPATH 以便运行时能找到共享库
set_target_properties(test_embedding PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

# 安装测试可执行文件
install(TARGETS test_embedding DESTINATION bin)