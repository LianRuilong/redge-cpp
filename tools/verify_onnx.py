import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer
import onnxruntime as ort

def verify_onnx(model_dir, test_text="这是一个测试文本。"):
    model_path = os.path.join(model_dir, "model.onnx")
    tokenizer_path = model_dir  # 假设 tokenizer 文件也在同一目录下

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    inputs = tokenizer(test_text, return_tensors="np")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}

    ort_inputs = {k: v for k, v in inputs.items() if k in input_names}
    ort_outputs = session.run(None, ort_inputs)

    print("ONNX 模型推理成功！输出结果维度：")
    for name, output in zip(session.get_outputs(), ort_outputs):
        print(f"- {name.name}: shape={output.shape}, dtype={output.dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证 ONNX 模型是否可正常推理")
    parser.add_argument("model_dir", help="包含 model.onnx 和 tokenizer 的目录路径")
    parser.add_argument("--text", default="这是一个测试文本。", help="用于测试的输入文本")
    args = parser.parse_args()

    verify_onnx(args.model_dir, args.text)
    