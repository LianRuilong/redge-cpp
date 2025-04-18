#include "text_embedding_factory.h"

#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <filesystem>

void save_vector_to_txt(const std::vector<float>& vec, const std::string& filename) {
    std::filesystem::path file_path(filename);
    std::filesystem::path dir = file_path.parent_path();

    // 自动创建目录（如果有需要）
    if (!dir.empty() && !std::filesystem::exists(dir)) {
        std::error_code ec;
        if (!std::filesystem::create_directories(dir, ec)) {
            std::cerr << "Failed to create directory: " << dir << ", error: " << ec.message() << std::endl;
            return;
        }
    }

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    for (float v : vec) {
        out << v << "\n";
    }
    out.close();
}

int main() {
    auto embedding = text_embedding::EmbeddingFactory::create(text_embedding::InferenceBackend::ONNXRUNTIME);
    if (!embedding) {
        std::cerr << "Failed to create embedding instance!" << std::endl;
        return 1;
    }

    if (!embedding->load_model("resource/model/multilingual-e5-small/")) {
        std::cerr << "Model loading failed!" << std::endl;
        return 1;
    }

    std::vector<float> vec = embedding->embed("你好，世界！");
    std::cout << "Vector size: " << vec.size() << std::endl;

    save_vector_to_txt(vec, "temp_testing_result/embedding_cpp.txt");

    embedding->unload_model();
    return 0;
}
