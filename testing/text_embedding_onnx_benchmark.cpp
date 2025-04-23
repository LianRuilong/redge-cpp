#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <numeric>
#include <mutex>
#include "text_embedding_factory.h"

struct ThreadResult {
    int thread_id;
    int total_requests;
    double total_time_ms;
};

std::mutex print_mutex;

void embedding_thread_worker(
    text_embedding::TextEmbedding* embedding,
    const std::string& text,
    int repeat_times,
    int thread_id,
    ThreadResult& result) {

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat_times; ++i) {
        std::vector<float> vec = embedding->embed(text);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    result = {thread_id, repeat_times, duration.count()};

    std::lock_guard<std::mutex> lock(print_mutex);
    std::cout << "[Thread " << thread_id << "] " << repeat_times
              << " requests finished in " << duration.count() << " ms. "
              << "Avg latency = " << (duration.count() / repeat_times) << " ms\n";
}

void run_qps_test(
    const std::string& model_name,
    text_embedding::TextEmbedding* embedding,
    const std::string& text,
    int thread_num,
    int repeat_per_thread) {

    std::cout << "\n========== Testing [" << model_name << "] with " << thread_num << " threads ==========\n";

    std::vector<std::thread> threads;
    std::vector<ThreadResult> results(thread_num);

    auto global_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < thread_num; ++i) {
        threads.emplace_back(embedding_thread_worker, embedding, text, repeat_per_thread, i, std::ref(results[i]));
    }
    for (auto& t : threads) t.join();
    auto global_end = std::chrono::high_resolution_clock::now();

    int total_requests = 0;
    double total_time_ms = 0.0;
    for (const auto& r : results) {
        total_requests += r.total_requests;
        total_time_ms = std::max(total_time_ms, r.total_time_ms);  // 取最长耗时
    }

    double qps = (total_requests / total_time_ms) * 1000.0;
    std::cout << "[Summary] Total Requests: " << total_requests
              << ", Max Elapsed Time: " << total_time_ms << " ms"
              << ", QPS: " << qps << "\n";
}

int main() {
    const std::string test_text = "人工智能正在改变世界。";
    const int thread_count = 8;
    const int repeat_per_thread = 100;

    // 测试 BGE 模型
    {
        const std::string bge_model_path = "resource/model/bge-small-zh-v1.5/";
        auto bge_embedding = text_embedding::EmbeddingFactory::create(text_embedding::InferenceBackend::ONNXRUNTIME);
        if (!bge_embedding->load_model(bge_model_path)) {
            std::cerr << "Failed to load BGE model.\n";
            return 1;
        }

        run_qps_test("bge", bge_embedding.get(), test_text, thread_count, repeat_per_thread);
        bge_embedding->unload_model();
    }

    // 测试 E5 模型
    {
        const std::string e5_model_path = "resource/model/multilingual-e5-small/";
        auto e5_embedding = text_embedding::EmbeddingFactory::create(text_embedding::InferenceBackend::ONNXRUNTIME);
        if (!e5_embedding->load_model(e5_model_path)) {
            std::cerr << "Failed to load E5 model.\n";
            return 1;
        }

        run_qps_test("e5", e5_embedding.get(), test_text, thread_count, repeat_per_thread);
        e5_embedding->unload_model();
    }

    return 0;
}