#!/bin/bash

set -e  # 出错即退出

BUILD_DIR=build
INSTALL_DIR=${BUILD_DIR}/install

echo "[INFO] Cleaning previous build..."
rm -rf ${BUILD_DIR}

echo "[INFO] Configuring project with CMake..."
cmake -B ${BUILD_DIR} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}

echo "[INFO] Building and installing..."
cmake --build ${BUILD_DIR} --target install

echo "[INFO] Build and install completed."
echo "[INFO] Files installed to: ${INSTALL_DIR}"
