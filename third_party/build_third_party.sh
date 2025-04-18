#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Usage: build_third_party.sh <install_dir>"
    exit 1
fi

INSTALL_DIR=$(realpath "$1")
mkdir -p "$INSTALL_DIR"

echo "[INFO] Installing third-party dependencies into: $INSTALL_DIR"

# =====================================
# Build llama.cpp
# =====================================
LLAMA_INSTALL_SUBDIR="$INSTALL_DIR/llama.cpp"
if [ ! -d "$LLAMA_INSTALL_SUBDIR" ]; then
    echo "[INFO] Building llama.cpp..."
    pushd "$(dirname "$0")/llama.cpp"
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$LLAMA_INSTALL_SUBDIR"
    cmake --build . --config Release -j$(nproc)
    make install
    popd
    echo "[INFO] llama.cpp installed to $LLAMA_INSTALL_SUBDIR"
fi

# =====================================
# Build ONNX Runtime
# =====================================
ONNX_INSTALL_SUBDIR="$INSTALL_DIR/onnxruntime"
if [ ! -d "$ONNX_INSTALL_SUBDIR" ]; then
    echo "[INFO] Building ONNX Runtime..."
    ONNXRUNTIME_SRC_DIR="$(dirname "$0")/onnxruntime"
    pushd "$ONNXRUNTIME_SRC_DIR"
    ./build.sh --config Release --parallel --build_shared_lib --skip_tests \
        --cmake_extra_defines CMAKE_INSTALL_PREFIX="$ONNX_INSTALL_SUBDIR"
    popd

    INSTALL_DIR_FOUND=$(find "$ONNXRUNTIME_SRC_DIR/build" -type d -path "*/Linux/Release" | head -n 1)
    if [ -z "$INSTALL_DIR_FOUND" ]; then
        echo "Error: Could not find the ONNX Runtime build output directory."
        exit 1
    fi
    pushd "$INSTALL_DIR_FOUND"
    make install
    popd

    echo "[INFO] ONNX Runtime installed to $ONNX_INSTALL_SUBDIR"
fi

# =====================================
# Build SQLite
# =====================================
SQLITE_INSTALL_SUBDIR="$INSTALL_DIR/sqlite"
if [ ! -d "$SQLITE_INSTALL_SUBDIR" ]; then
    echo "[INFO] Building SQLite..."
    SQLITE_SRC_DIR="$(dirname "$0")/sqlite"
    pushd "$SQLITE_SRC_DIR"
    mkdir -p build && cd build
    ../configure --prefix="$SQLITE_INSTALL_SUBDIR" \
        --disable-tcl \
        CFLAGS="-DSQLITE_ENABLE_VECTOR"
    make -j$(nproc)
    make install
    popd
    echo "[INFO] SQLite installed to $SQLITE_INSTALL_SUBDIR"
fi

# =====================================
# Build cppjieba
# =====================================
CPPJIEBA_INSTALL_SUBDIR="$INSTALL_DIR/cppjieba"
if [ ! -d "$CPPJIEBA_INSTALL_SUBDIR" ]; then
    echo "[INFO] Building cppjieba..."
    CPPJIEBA_SRC_DIR="$(dirname "$0")/cppjieba"
    pushd "$CPPJIEBA_SRC_DIR"
    git submodule init
    git submodule update
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$CPPJIEBA_INSTALL_SUBDIR"
    make -j$(nproc)
    make install
    popd
    echo "[INFO] cppjieba installed to $CPPJIEBA_INSTALL_SUBDIR"
fi

# =====================================
# Build tokenizers-cpp
# =====================================
TOKENIZERS_CPP_INSTALL_SUBDIR="$INSTALL_DIR/tokenizers-cpp"
if [ ! -d "$TOKENIZERS_CPP_INSTALL_SUBDIR" ]; then
    echo "[INFO] Building tokenizers-cpp..."

    # Ensure Rust is available
    if ! command -v cargo >/dev/null 2>&1; then
        echo "[ERROR] Rust 'cargo' is required to build tokenizers-cpp but was not found."
        echo "Please install Rust from https://rustup.rs/"
        exit 1
    fi

    TOKENIZERS_CPP_SRC_DIR="$(dirname "$0")/tokenizers-cpp"

    if [ ! -d "$TOKENIZERS_CPP_SRC_DIR" ]; then
        echo "[ERROR] tokenizers-cpp source not found at $TOKENIZERS_CPP_SRC_DIR"
        echo "Please clone with: git clone --recursive https://github.com/mlc-ai/tokenizers-cpp.git"
        exit 1
    fi

    pushd "$TOKENIZERS_CPP_SRC_DIR"
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TOKENIZERS_CPP_INSTALL_SUBDIR"
    cmake --build . --config Release -j$(nproc)
    make install
    popd

    echo "[INFO] tokenizers-cpp installed to $TOKENIZERS_CPP_INSTALL_SUBDIR"
fi

# =====================================
# Merge headers & libs into INSTALL_DIR
# =====================================
echo "[INFO] Merging all headers and libraries to ${INSTALL_DIR}/include and lib..."

mkdir -p "$INSTALL_DIR/include"
mkdir -p "$INSTALL_DIR/lib"

for SUB in llama.cpp onnxruntime sqlite cppjieba tokenizers-cpp; do
    SUB_DIR="$INSTALL_DIR/$SUB"
    echo "[INFO] === Merging for $SUB ==="

    # Merge headers
    if [ -d "$SUB_DIR/include" ]; then
        echo "[INFO] Copying headers from $SUB_DIR/include..."
        cp -vr "$SUB_DIR/include/"* "$INSTALL_DIR/include/"
    else
        echo "[WARN] No include dir found for $SUB"
    fi

    # Determine lib dir
    if [ -d "$SUB_DIR/lib" ]; then
        LIB_SRC_DIR="$SUB_DIR/lib"
    elif [ -d "$SUB_DIR/lib64" ]; then
        LIB_SRC_DIR="$SUB_DIR/lib64"
    else
        echo "[WARN] No lib or lib64 directory found in $SUB_DIR"
        continue
    fi

    echo "[DEBUG] Using library source directory: $LIB_SRC_DIR"

    # List all files to be copied
    echo "[DEBUG] Libraries found in $LIB_SRC_DIR:"
    find "$LIB_SRC_DIR" -maxdepth 1 \( -type f -o -type l \) \
        \( -name "*.so*" -o -name "*.a" -o -name "*.dylib" \) \
        -exec ls -l "{}" \;

    # Copy all .so* files (including .so.1, .so.1.x, etc.) with dereferencing symlinks
    find "$LIB_SRC_DIR" -maxdepth 1 -type f -name "*.so*" -exec cp -vL "{}" "$INSTALL_DIR/lib/" \;

    # Optionally copy static libraries and dylibs if needed
    find "$LIB_SRC_DIR" -maxdepth 1 \( -name "*.a" -o -name "*.dylib" \) \
        -exec cp -vL "{}" "$INSTALL_DIR/lib/" \;

    # Copy symlinks as symlinks (not dereferencing them) if required
    find "$LIB_SRC_DIR" -maxdepth 1 -type l -exec cp -vs "{}" "$INSTALL_DIR/lib/" \;
done

echo "[SUCCESS] All third-party dependencies built and merged into: $INSTALL_DIR"