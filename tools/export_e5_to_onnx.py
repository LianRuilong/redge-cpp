#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export multilingual-e5-small model to ONNX format.

Usage:
    python3 tools/export_e5_to_onnx.py source_path target_path
"""

import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from transformers.onnx import export
from transformers.onnx import FeaturesManager
from transformers.utils import logging

def export_to_onnx(source_path: str, target_path: str, opset: int = 13):
    logging.set_verbosity_info()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(source_path, use_fast=False)
    model = AutoModel.from_pretrained(source_path)

    # Prepare paths
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    onnx_model_path = target_path / "model.onnx"

    # Infer task
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature="feature-extraction")
    onnx_config = model_onnx_config(model.config)

    # Dummy inputs
    dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework="pt")

    print(f"[INFO] Exporting to {onnx_model_path} ...")
    export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=opset,
        output=onnx_model_path,
        tokenizer=tokenizer,
        inputs=dummy_inputs
    )

    print(f"[SUCCESS] Exported ONNX model to: {onnx_model_path.resolve()}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 tools/export_e5_to_onnx.py source_path target_path")
        sys.exit(1)

    source = sys.argv[1]
    target = sys.argv[2]

    export_to_onnx(source, target)

if __name__ == "__main__":
    main()