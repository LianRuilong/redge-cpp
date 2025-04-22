#include <chrono> 
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <numeric>

#include <gtest/gtest.h>

#include "text_embedding_factory.h"

// 读取向量
std::vector<float> read_vector_from_file(const std::string& filename) {
    std::vector<float> vec;
    std::ifstream in(filename);
    float value;
    while (in >> value) vec.push_back(value);
    return vec;
}

// 计算余弦相似度
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8f);
}

class EmbeddingBatchTest : public ::testing::TestWithParam<std::string> {};

TEST_P(EmbeddingBatchTest, CompareCppAndPythonEmbedding) {
    // 本地模型路径
    std::string model_path = "resource/model/multilingual-e5-small/";
    std::string text = GetParam();
    std::string safe_text = std::to_string(std::hash<std::string>{}(text)); // safe filename

    std::string cpp_out = "out/temp/embedding_cpp_" + safe_text + ".txt";
    std::string py_out  = "out/temp/embedding_py_" + safe_text + ".txt";

    auto onnx_start_time = std::chrono::high_resolution_clock::now();

    auto embedding = text_embedding::EmbeddingFactory::create(text_embedding::InferenceBackend::ONNXRUNTIME);

    ASSERT_TRUE(embedding);
    ASSERT_TRUE(embedding->load_model(model_path));

    auto embed_start_time = std::chrono::high_resolution_clock::now();

    std::vector<float> vec = embedding->embed(text);

    auto embed_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> embed_elapsed = embed_end_time - embed_start_time;
    std::cout << "[Profiling] embed elapsed : " << embed_elapsed.count() << " ms" << std::endl;

    embedding->unload_model();

    auto onnx_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> onnx_elapsed = onnx_end_time - onnx_start_time;
    std::cout << "[Profiling] onnx elapsed : " << onnx_elapsed.count() << " ms" << std::endl;

    // 保存 C++ 向量
    std::ofstream out(cpp_out);
    for (float v : vec) out << v << "\n";
    out.close();

    // Python 脚本调用
    std::string cmd = "python3 testing/scripts/test_onnx_embedding.py \"" + model_path + "\" \"" + text + "\" \"" + py_out + "\"";
    int ret = std::system(cmd.c_str());
    ASSERT_EQ(ret, 0) << "Python script failed!";

    std::vector<float> vec_py = read_vector_from_file(py_out);
    ASSERT_EQ(vec.size(), vec_py.size()) << "Vector sizes mismatch for input: " << text;

    float sim = cosine_similarity(vec, vec_py);
    std::cout << "[Sample] \"" << text << "\" => Cosine similarity: " << sim << std::endl;

    EXPECT_NEAR(sim, 1.0, 0.01) << "Mismatch for input: " << text;
}

INSTANTIATE_TEST_SUITE_P(
    EmbeddingSamples,
    EmbeddingBatchTest,
    ::testing::Values(
        "你好，世界！",
        "今天天气不错。",
        "OpenAI is building amazing models.",
        "如何前往火车站？",
        "The quick brown fox jumps over the lazy dog.",
        "我爱北京天安门。",
        "This is a test sentence.",
        "人工智能正在改变世界。",
        "Bonjour le monde!",
        "未来属于我们"
    )
);
