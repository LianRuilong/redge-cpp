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

# 搜索所有可执行的测试程序（包含子目录）
test_executables=$(find "${BIN_DIR}" -type f -executable)

if [ -z "$test_executables" ]; then
  echo "[WARN] No test executables found in ${BIN_DIR}"
  exit 0
fi

# 遍历所有测试可执行文件运行
for test_exec in $test_executables; do
  if [[ ! -x "$test_exec" ]]; then
    continue
  fi

  echo "--------------------------------------------------"
  echo "[TEST] Running $(basename "$test_exec")"
  if GTEST_FILTER="${GTEST_FILTER:-*}" LD_LIBRARY_PATH="$LIB_DIR" "$test_exec"; then
    echo "[PASS] $(basename "$test_exec")"
  else
    echo "[FAIL] $(basename "$test_exec")"
    exit 1
  fi
done

echo "[INFO] All tests completed."
