import sys
import numpy as np
from transformers import PreTrainedTokenizerFast
import onnxruntime as ort
import os

# ========== DEBUG 控制 ==========
DEBUG = os.getenv("DEBUG", "0") == "1"

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[Debug]", *args, **kwargs)

# ========== 参数读取 ==========
model_dir = sys.argv[1]
text = sys.argv[2]
output_path = sys.argv[3]

# ========== 加载分词器 ==========
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
tokenizer.model_max_length = 512

# 设置 pad token（若未定义）
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ========== 加载 ONNX 模型 ==========
session = ort.InferenceSession(os.path.join(model_dir, "model.onnx"))

# ========== 编码文本 ==========
inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

debug_print("PY input_ids:", input_ids)
debug_print("PY attention_mask:", attention_mask)

# ========== 模型推理 ==========
output_names = [o.name for o in session.get_outputs()]
debug_print("Output names:", output_names)

# 查找句向量输出（非 last_hidden_state）
sentence_output_name = next((name for name in output_names if name != 'last_hidden_state'), None)

if sentence_output_name:
    # ✅ 使用模型提供的句向量输出
    outputs = session.run([sentence_output_name], {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })
    vec = outputs[0].squeeze()
else:
    # ❗模型只输出 last_hidden_state，手动做 mean pooling
    outputs = session.run(["last_hidden_state"], {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })
    last_hidden_state = outputs[0]  # shape: [1, seq_len, hidden_size]

    attention_mask_expanded = np.expand_dims(attention_mask, axis=-1)  # shape: [1, seq_len, 1]
    sum_embeddings = np.sum(last_hidden_state * attention_mask_expanded, axis=1)
    sum_mask = np.clip(attention_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    vec = sum_embeddings / sum_mask
    vec = vec.squeeze()

debug_print("Embedding shape:", vec.shape)

# ========== 保存向量 ==========
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.savetxt(output_path, vec)
