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
pip install -r requirements.txt
```

> 若新增新的依赖包，直接编辑 requirements.txt 文件添加即可。

> 若因网络问题需更换安装源，可编辑 requirements.txt 中，-i {url} 部分。

### 模型转换

#### bge 模型


从 huggingface 克隆到本地再转换：

```
python3 export_bge_to_onnx.py --model {huggingface_model_name} --output {local_model__target_file_path}
```

使用样例：
```
python3 export_bge_to_onnx.py --model BAAI/bge-small-zh-v1.5 --output ./bge-small-zh-v1.5-onnx/model.onnx
```

从 huggingface 克隆到本地再转换：

```
python3 export_bge_to_onnx.py --model {local_cloned_model_dir} --output {local_model__target_file_path}
```

使用样例：

```
python3 export_bge_to_onnx.py --model ./bge-small-zh-v1.5 --output ./bge-small-zh-v1.5-onnx/model.onnx
```


#### e5 模型

从 huggingface 源转换：

```
optimum-cli export onnx --model {huggingface_model_name} {local_model__target_dir} --task default
```

使用样例：

```
optimum-cli export onnx --model intfloat/multilingual-e5-small ./multilingual-e5-small-onnx/ --task default
```

从 huggingface 克隆到本地再转换：

```
optimum-cli export onnx --model {local_cloned_model_dir} {local_model__target_dir} --task default
```

使用样例：

```
optimum-cli export onnx --model ~/Work/baai/multilingual-e5-small ./multilingual-e5-small-onnx/   --task default
```

### 模型验证

验证命令：

```
python3 verify_onnx.py {model_dir_path}

或者

python3 verify_onnx.py {model_dir_path} --text {input_text}
```

使用样例：
```
python3 verify_onnx.py ../resource/model/bge-small-zh-v1.5/ --text "我爱自然语言处理"

ONNX 模型推理成功！输出结果维度：
- last_hidden_state: shape=(1, 10, 512), dtype=float32
- 703: shape=(1, 512), dtype=float32
```
