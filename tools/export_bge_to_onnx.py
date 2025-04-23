import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import os

def export_onnx(model_path, output_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    text = "示例文本"
    inputs = tokenizer(text, return_tensors="pt")

    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state"]

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "last_hidden_state": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=14,
        )
    print(f"✅ ONNX 导出成功：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BGE model to ONNX")
    parser.add_argument("--model", required=True, help="模型路径或 HuggingFace 名称，例如 'BAAI/bge-small-zh-v1.5'")
    parser.add_argument("--output", required=True, help="ONNX 文件输出路径")
    args = parser.parse_args()

    export_onnx(args.model, args.output)
    