#!/bin/bash

set -e
set -o pipefail

echo "==> 初始化 third_party 依赖仓库..."

declare -A repos=(
    ["cppjieba"]="https://github.com/yanyiwu/cppjieba.git"
    ["onnxruntime"]="https://github.com/microsoft/onnxruntime.git"
    ["llama.cpp"]="https://github.com/ggml-org/llama.cpp.git"
    ["sqlite"]="https://github.com/sqlite/sqlite.git"
    ["tokenizers-cpp"]="https://github.com/mlc-ai/tokenizers-cpp.git"
    ["glog"]="git@github.com:google/glog.git"
    ["googletest"]="https://github.com/google/googletest.git -b v1.16.0"
)

for name in "${!repos[@]}"; do
    path="third_party/$name"
    url="${repos[$name]}"

    if [ -d "$path/.git" ]; then
        echo "==> $name 已存在，跳过..."
    else
        echo "==> 克隆 $name ..."
        git clone "$url" "$path"
    fi
done

echo "==> 所有 third_party 仓库初始化完成。"
