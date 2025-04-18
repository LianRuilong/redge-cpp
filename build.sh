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

# Set up directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(realpath "${SCRIPT_DIR}/build")"
INSTALL_DIR="${BUILD_DIR}/install"

echo "[INFO] Cleaning previous build directory: $BUILD_DIR"
# rm -rf "$BUILD_DIR"

echo "[INFO] Configuring project with CMake..."
cmake -B "$BUILD_DIR" -S "$SCRIPT_DIR" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

echo "[INFO] Building and installing (parallel build)..."
cmake --build "$BUILD_DIR" --target install -- -j$(nproc)

echo "[SUCCESS] Build and installation completed."
echo "[INFO] Files installed to: $INSTALL_DIR"