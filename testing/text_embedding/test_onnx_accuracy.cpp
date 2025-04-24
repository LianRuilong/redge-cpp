#include <chrono> 
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <numeric>
#include <filesystem>
#include <limits.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "logger.h"
#include "text_embedding_factory.h"

namespace fs = std::filesystem;

std::vector<float> read_vector_from_file(const std::string& filename) {
    std::vector<float> vec;
    std::ifstream in(filename);
    float value;
    while (in >> value) vec.push_back(value);
    return vec;
}

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8f);
}

std::string get_binary_dir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        return fs::path(std::string(result, count)).parent_path().string();
    }
    return "./";
}

class EmbeddingBatchTest : public ::testing::TestWithParam<std::string> {};

void run_embedding_test(const std::string& model_name, const std::string& model_path, const std::string& text) {
    std::string safe_text = std::to_string(std::hash<std::string>{}(text));
    std::string cpp_out = "out/temp/embedding_cpp_" + model_name + "_" + safe_text + ".txt";
    std::string py_out  = "out/temp/embedding_py_"  + model_name + "_" + safe_text + ".txt";

    LOG_INFO << "\n=== [" << model_name << "] Testing on text: \"" << text << "\" ===";

    auto onnx_start_time = std::chrono::high_resolution_clock::now();

    auto embedding = text_embedding::EmbeddingFactory::create(text_embedding::InferenceBackend::ONNXRUNTIME);
    ASSERT_TRUE(embedding);
    ASSERT_TRUE(embedding->load_model(model_path));

    auto embed_start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> vec = embedding->embed(text);
    auto embed_end_time = std::chrono::high_resolution_clock::now();

    embedding->unload_model();

    std::chrono::duration<double, std::milli> embed_elapsed = embed_end_time - embed_start_time;
    std::chrono::duration<double, std::milli> onnx_elapsed = embed_end_time - onnx_start_time;
    LOG_INFO << "[Profiling] embed elapsed : " << embed_elapsed.count() << " ms";
    LOG_INFO << "[Profiling] onnx elapsed  : " << onnx_elapsed.count() << " ms";

    std::ofstream out(cpp_out);
    for (float v : vec) out << v << "\n";
    out.close();

    std::string script_path = get_binary_dir() + "/scripts/test_onnx_embedding.py";
    std::string cmd = "python3 \"" + script_path + "\" \"" + model_path + "\" \"" + text + "\" \"" + py_out + "\"";
    int ret = std::system(cmd.c_str());
    ASSERT_EQ(ret, 0) << "Python script failed!";

    std::vector<float> vec_py = read_vector_from_file(py_out);
    ASSERT_EQ(vec.size(), vec_py.size()) << "Vector size mismatch!";

    float sim = cosine_similarity(vec, vec_py);
    LOG_INFO << "[Cosine Similarity] " << sim;

    EXPECT_NEAR(sim, 1.0, 0.01) << "Cosine similarity too low!";
}

TEST_P(EmbeddingBatchTest, CompareE5AndBGE) {
    std::string text = GetParam();

    run_embedding_test("multilingual-e5-small",  "resource/model/multilingual-e5-small/", text);
    run_embedding_test("bge-small-zh-v1.5", "resource/model/bge-small-zh-v1.5/", text);
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
