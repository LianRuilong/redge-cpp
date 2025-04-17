from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# 本地路径
model_dir = "model/multilingual-e5-small"
onnx_path = f"{model_dir}/model.onnx"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 创建 onnxruntime 推理 session
session = ort.InferenceSession(onnx_path)

# 输入文本
text = "你好，世界！"
inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)

# ONNX 推理
outputs = session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
})

# 查看输出 shape
print("ONNX Output:", outputs[0].shape)