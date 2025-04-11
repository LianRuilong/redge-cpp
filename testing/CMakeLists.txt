cmake_minimum_required(VERSION 3.16)
project(test_embedding)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 头文件路径
include_directories(
    ${THIRD_PARTY_INSTALL_DIR}/onnxruntime/include
    ${CMAKE_SOURCE_DIR}/src/components/text_embedding
)

# 测试源文件
set(TEST_EMBEDDING_SRC test_embedding.cpp)

# 生成可执行文件
add_executable(test_embedding ${TEST_EMBEDDING_SRC})

# 链接 text_embedding 静态库
target_link_libraries(test_embedding text_embedding)

# 安装测试可执行文件
install(TARGETS test_embedding DESTINATION bin)
