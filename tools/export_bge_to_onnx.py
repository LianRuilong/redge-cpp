#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export a SentenceTransformer model (e.g., bge-small-zh-v1.5) to ONNX format with proper pooling.

Usage:
    python3 tools/export_bge_to_onnx.py source_path target_path
"""

import sys
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

def export_to_onnx(source_path: str, target_path: str, opset: int = 13):
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting from: {source_path.resolve()}")
    print(f"[INFO] Exporting to:   {target_path.resolve()}")
    print(f"[INFO] Task: feature-extraction, Opset: {opset}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(source_path)
    model = ORTModelForFeatureExtraction.from_pretrained(source_path, from_transformers=True)

    # Save the ONNX model and tokenizer
    model.save_pretrained(target_path)
    tokenizer.save_pretrained(target_path)

    print(f"[SUCCESS] Exported ONNX model to: {target_path.resolve()}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 tools/export_bge_to_onnx.py source_path target_path")
        sys.exit(1)

    source = sys.argv[1]
    target = sys.argv[2]

    export_to_onnx(source, target)

if __name__ == "__main__":
    main()