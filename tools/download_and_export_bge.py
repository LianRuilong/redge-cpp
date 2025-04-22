import os
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import subprocess

# 1. 模型名称与保存路径
model_name = "BAAI/bge-small-zh-v1.5"
save_dir = Path("./bge_hf_model")
onnx_output_dir = save_dir / "onnx"

# 2. 下载并保存模型
print("[Step 1] Downloading model and tokenizer...")
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 3. 创建导出 ONNX 的输出目录
onnx_output_dir.mkdir(parents=True, exist_ok=True)

# 4. 运行 ONNX 导出命令
print("[Step 2] Exporting to ONNX...")
export_cmd = [
    "python3",
    "export_bge_to_onnx.py",  # 确保这个脚本在当前目录下
    "--model_dir", str(save_dir),
    "--output_path", str(onnx_output_dir),
    "--opset", "14"
]

result = subprocess.run(export_cmd, capture_output=True, text=True)

# 5. 输出结果
print("[ONNX Export Output]")
print(result.stdout)
if result.stderr:
    print("[ERROR]")
    print(result.stderr)
