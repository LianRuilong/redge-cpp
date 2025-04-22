import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel

def export_onnx(model_dir, output_path, opset=14):
    print(f"[Info] Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()

    # 保存 tokenizer.json（用于 C++ 侧加载）
    tokenizer_path = os.path.join(output_path, "tokenizer.json")
    tokenizer.backend_tokenizer.save(tokenizer_path)
    print(f"[Success] Saved tokenizer.json to: {tokenizer_path}")

    # 准备 dummy 输入
    dummy_input = tokenizer("测试文本", return_tensors="pt", padding="max_length", max_length=16, truncation=True)
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    onnx_path = os.path.join(output_path, "model.onnx")
    print(f"[Info] Exporting ONNX to: {onnx_path}")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
        },
        do_constant_folding=True,
        opset_version=opset,
    )
    print(f"[Success] ONNX model exported to: {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export baai/bge-small-zh-v1.5 to ONNX")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the downloaded HuggingFace model dir")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the exported ONNX and tokenizer")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    export_onnx(args.model_dir, args.output_path, args.opset)
