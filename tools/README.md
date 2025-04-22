# 关于如何使用工具

## onnx 模型转换工具

### 创建 Python 环境

```
sudo apt install python3
python3 -m venv env-redge
source redge-env/bin/activate
```

### 安装依赖包

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple optimum
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentence_transformers
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tiktoken
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentencepiece
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple accelerate
pip install -U sentence-transformers transformers onnxruntime
```

### 模型转换

```
python3 export_bge_to_onnx.py \
    --model_dir /home/lianrl/Work/baai/bge-small-zh-v1.5 \
    --output_path /home/lianrl/Work/baai/bge-small-zh-v1.5/onnx \
    --opset 14
```

```
optimum-cli export onnx   --model ~/Work/baai/multilingual-e5-small   ./build/multilingual-e5-small-onnx/   --task default
```

