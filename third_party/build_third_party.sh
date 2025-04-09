#!/bin/bash

set -e  # 发生错误时立即退出

# 检查是否有 CMake 指定的 BUILD 目录
if [ -z "$1" ]; then
    echo "Usage: build_third_party.sh <install_dir>"
    exit 1
fi

INSTALL_DIR=$(realpath "$1")  # 依赖安装到 CMake 指定的 build 目录
mkdir -p "$INSTALL_DIR"

echo "Installing third-party dependencies into: $INSTALL_DIR"

# 编译 llama.cpp
if [ ! -d "$INSTALL_DIR/llama.cpp" ]; then
    echo "Building llama.cpp..."
    pushd "$(dirname "$0")/llama.cpp"
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/llama.cpp"
    cmake --build . --config Release -j$(nproc)
    # make -j$(nproc)
    make install
    popd
fi

# 编译 ONNX Runtime
if [ ! -d "$INSTALL_DIR/onnxruntime" ]; then
    echo "Building ONNX Runtime..."
    ONNXRUNTIME_SRC_DIR="$(dirname "$0")/onnxruntime"
    
    pushd "$ONNXRUNTIME_SRC_DIR"
    
    ./build.sh --config Release --parallel --build_shared_lib --skip_tests \
               --cmake_extra_defines CMAKE_INSTALL_PREFIX="$INSTALL_DIR/onnxruntime"
    BUILD_STATUS=$?
    popd

    if [ $BUILD_STATUS -ne 0 ]; then
        echo "ONNX Runtime build failed. Exiting."
        exit 1
    fi

    echo "Installing ONNX Runtime..."
    # 查找正确的 make install 目录
    INSTALL_DIR_FOUND=$(find "$ONNXRUNTIME_SRC_DIR/build" -type d -path "*/Linux/Release" | head -n 1)
    
    if [ -z "$INSTALL_DIR_FOUND" ]; then
        echo "Error: Could not find the ONNX Runtime build output directory."
        exit 1
    fi
    
    pushd "$INSTALL_DIR_FOUND"
    make install
    popd
fi

# 编译 SQLite
if [ ! -d "$INSTALL_DIR/sqlite" ]; then
    echo "Building SQLite..."
    SQLITE_SRC_DIR="$(dirname "$0")/sqlite"
    
    pushd "$SQLITE_SRC_DIR"
    
    mkdir -p build && cd build
    ../configure --prefix="$INSTALL_DIR/sqlite" \
             --disable-tcl \
             CFLAGS="-DSQLITE_ENABLE_VECTOR"
    # ../configure --prefix="$INSTALL_DIR/sqlite-install" --with-tcl="$INSTALL_DIR/sqlite-install"
    make -j$(nproc)
    make install

    popd
fi

# 编译 cppjieba
if [ ! -d "$INSTALL_DIR/cppjieba" ]; then
    echo "Building cppjieba..."
    CPPJIEBA_SRC_DIR="$(dirname "$0")/cppjieba"
    
    # if [ ! -d "$CPPJIEBA_SRC_DIR" ]; then
    #     git clone https://github.com/yanyiwu/cppjieba.git "$CPPJIEBA_SRC_DIR"
    # fi

    pushd "$CPPJIEBA_SRC_DIR"

    git submodule init
    git submodule update
    
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/cppjieba"
    make -j$(nproc)
    make install

    popd
fi

echo "All third_party dependencies built successfully!"
