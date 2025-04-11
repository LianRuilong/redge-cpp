cmake_minimum_required(VERSION 3.16)
project(text_embedding)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "THIRD_PARTY_INSTALL_DIR: ${THIRD_PARTY_INSTALL_DIR}")

# 头文件路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${THIRD_PARTY_INSTALL_DIR}/onnxruntime/include
)

# 源文件
file(GLOB TEXT_EMBEDDING_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# 生成静态库
add_library(text_embedding STATIC ${TEXT_EMBEDDING_SRC})

# 链接 ONNXRuntime 库
target_link_libraries(text_embedding
    ${THIRD_PARTY_INSTALL_DIR}/onnxruntime/lib/libonnxruntime.so
)

# 安装目标
install(TARGETS text_embedding DESTINATION lib)
install(FILES text_embedding.h text_embedding_factory.h onnx_embedding.h DESTINATION include)