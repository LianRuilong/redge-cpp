import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

# 设置模型路径和文本
model_dir = "model/multilingual-e5-small"
onnx_path = f"{model_dir}/model.onnx"
text = "你好，世界！"

# 载入 tokenizer 和 onnx
tokenizer = AutoTokenizer.from_pretrained(model_dir)
session = ort.InferenceSession(onnx_path)

# 编码输入
inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
outputs = session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
})
embedding_py = outputs[1][0]  # sentence_embedding

# 保存 py 向量
np.save("embedding_py.npy", embedding_py)

# 加载 cpp 输出的 .txt 向量
embedding_cpp = np.loadtxt("temp_testing_result/embedding_cpp.txt")

# 验证维度一致
assert embedding_cpp.shape == embedding_py.shape, \
    f"Shape mismatch! C++: {embedding_cpp.shape}, Python: {embedding_py.shape}"

# 计算余弦相似度
cos_sim = cosine_similarity([embedding_cpp], [embedding_py])[0][0]
print(f"Cosine similarity between Python and C++ outputs: {cos_sim:.6f}")

# 可选：打印前几维对比
print("\n[Sample values comparison]")
for i in range(min(8, len(embedding_py))):
    print(f"[{i}] Python: {embedding_py[i]:.6f}, C++: {embedding_cpp[i]:.6f}")
