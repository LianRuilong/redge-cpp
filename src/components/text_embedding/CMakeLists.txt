cmake_minimum_required(VERSION 3.16)
project(text_embedding)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "THIRD_PARTY_INSTALL_DIR: ${THIRD_PARTY_INSTALL_DIR}")

# 源文件
file(GLOB TEXT_EMBEDDING_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# 生成动态库
add_library(text_embedding SHARED ${TEXT_EMBEDDING_SRC})

# 添加头文件路径，仅对当前 target 生效
target_include_directories(text_embedding
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/base/logger
        $<BUILD_INTERFACE:${THIRD_PARTY_INSTALL_DIR}/onnxruntime/include>
        $<BUILD_INTERFACE:${THIRD_PARTY_INSTALL_DIR}/tokenizers-cpp/include>
        $<INSTALL_INTERFACE:include>
)

# 链接依赖库
target_link_libraries(text_embedding
    logger
    ${THIRD_PARTY_INSTALL_DIR}/tokenizers-cpp/lib/libtokenizers_c.a
    ${THIRD_PARTY_INSTALL_DIR}/tokenizers-cpp/lib/libtokenizers_cpp.a
    ${THIRD_PARTY_INSTALL_DIR}/onnxruntime/lib/libonnxruntime.so
)

# 设置库安装路径和头文件安装路径
install(TARGETS text_embedding
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES text_embedding.h text_embedding_factory.h onnx_embedding.h DESTINATION include)