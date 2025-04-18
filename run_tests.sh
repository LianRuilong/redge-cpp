#!/bin/bash

set -e

INSTALL_DIR=out/install
BIN_DIR=${INSTALL_DIR}/bin
LIB_DIR=${INSTALL_DIR}/lib

echo "[INFO] Running tests in ${BIN_DIR}..."

if [ ! -d "${BIN_DIR}" ]; then
  echo "[ERROR] Binary directory not found: ${BIN_DIR}"
  exit 1
fi

# 遍历 BIN_DIR 下的所有可执行文件
for test_exec in "${BIN_DIR}"/*; do
  if [[ -x "${test_exec}" && ! -d "${test_exec}" ]]; then
    echo "--------------------------------------------------"
    echo "[TEST] Running $(basename "${test_exec}")"
    LD_LIBRARY_PATH=${LIB_DIR} "${test_exec}"
  fi
done

echo "[INFO] All tests completed."
