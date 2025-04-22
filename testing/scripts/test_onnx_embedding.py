import sys
import numpy as np
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
import onnxruntime as ort
import os

# 参数：模型目录、文本、输出路径
model_dir = sys.argv[1]
text = sys.argv[2]
output_path = sys.argv[3]

tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))

# ✅ 设置 pad token，避免 padding 不一致
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 加载 ONNX 模型
session = ort.InferenceSession(os.path.join(model_dir, "model.onnx"))

# 编码文本
inputs = tokenizer(text, return_tensors="np", padding=False, truncation=False)

# ONNX 推理
outputs = session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
})

print("[Debug] PY input_ids:", inputs["input_ids"])
print("[Debug] PY attention_mask:", inputs["attention_mask"])

# 提取句向量（sentence embedding）
vec = outputs[1].squeeze()

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 保存向量到文本文件（与 C++ 保持一致）
np.savetxt(output_path, vec)