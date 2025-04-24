cmake_minimum_required(VERSION 3.16)
project(logger)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Building logger")

# === 查找 GLOG ===
set(GLOG_ROOT "${THIRD_PARTY_INSTALL_DIR}/glog")
find_package(glog REQUIRED PATHS ${GLOG_ROOT} NO_DEFAULT_PATH)

# === 添加 include 目录（当前目录）===
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# === 构建 logger 共享库 ===
add_library(logger SHARED
    logger.cpp
)

# === 链接 glog ===
target_link_libraries(logger
    glog::glog
)

# === 安装 so 库和头文件 ===
install(TARGETS logger DESTINATION lib)
install(FILES logger.h DESTINATION include/logger)
