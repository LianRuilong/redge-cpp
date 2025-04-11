cmake_minimum_required(VERSION 3.16)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message("cmake source dir is: ${CMAKE_SOURCE_DIR}")

# 指定第三方库的安装路径
set(THIRD_PARTY_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party_install)
message("third-party install dir is: ${THIRD_PARTY_INSTALL_DIR}")

# 调用 build_third_party.sh 来构建第三方依赖
message(STATUS "Building third-party dependencies...")
message("build third_party command is: ${CMAKE_SOURCE_DIR}/third_party/build_third_party.sh ${THIRD_PARTY_INSTALL_DIR}")
execute_process(
    COMMAND ${CMAKE_SOURCE_DIR}/third_party/build_third_party.sh ${THIRD_PARTY_INSTALL_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party
    RESULT_VARIABLE BUILD_THIRD_PARTY_RESULT
)

if (BUILD_THIRD_PARTY_RESULT)
    message(FATAL_ERROR "Failed to build third-party dependencies!")
endif()

# 添加第三方库的 include 目录
include_directories(${THIRD_PARTY_INSTALL_DIR}/include)
link_directories(${THIRD_PARTY_INSTALL_DIR}/lib)

# 添加 src 目录
add_subdirectory(src/components/text_embedding)

# 添加测试
enable_testing()
add_subdirectory(testing)
