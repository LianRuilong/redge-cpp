#!/bin/bash

set -e  # 出错即退出

# Ensure rust env is available
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Check if cargo (Rust) is installed
if ! command -v cargo >/dev/null 2>&1; then
    echo "[ERROR] Rust 'cargo' is required to build tokenizers but was not found in PATH."
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi

# Step 1: 初始化 third_party 依赖仓库
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${SCRIPT_DIR}/third_party"
INIT_SCRIPT="${THIRD_PARTY_DIR}/init_third_party.sh"

if [ -f "$INIT_SCRIPT" ]; then
    echo "[INFO] Running third-party dependency initialization script..."
    bash "$INIT_SCRIPT"
else
    echo "[WARNING] third_party/init_third_party.sh not found, skipping third-party setup."
fi

# Step 2: 设置构建路径
BUILD_DIR="$(realpath "${SCRIPT_DIR}/out")"
INSTALL_DIR="${BUILD_DIR}/install"

echo "[INFO] Cleaning previous build directory: $BUILD_DIR"
# rm -rf "$BUILD_DIR"  # 如需清空则取消注释

echo "[INFO] Configuring project with CMake..."
cmake -B "$BUILD_DIR" -S "$SCRIPT_DIR" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

echo "[INFO] Building and installing (parallel build)..."
cmake --build "$BUILD_DIR" --target install -- -j$(nproc)
echo "[INFO] Files installed to: $INSTALL_DIR"

echo "[SUCCESS] Build and installation completed."
