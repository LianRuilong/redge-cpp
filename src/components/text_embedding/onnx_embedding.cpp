#include "onnx_embedding.h"

#include <iostream>
#include <numeric>      // std::accumulate
#include <algorithm>    // std::transform
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {

// 打印模型输入输出的形状 
// [Model Inputs]:
// Input 0: input_ids
// Input 1: attention_mask
// [Model Outputs]:
// Output 0: token_embeddings
// Output 1: sentence_embedding
void print_model_io_info(Ort::Session& session) {
    // 打印输入名
    auto input_names = session.GetInputNames();
    std::cout << "[Model Inputs]:" << std::endl;
    for (size_t i = 0; i < input_names.size(); ++i) {
        std::cout << "  Input " << i << ": " << input_names[i] << std::endl;
    }

    // 打印输出名
    auto output_names = session.GetOutputNames();
    std::cout << "[Model Outputs]:" << std::endl;
    for (size_t i = 0; i < output_names.size(); ++i) {
        std::cout << "  Output " << i << ": " << output_names[i] << std::endl;
    }
}

} // namespace

namespace text_embedding {

OnnxRuntimeEmbedding::OnnxRuntimeEmbedding() : env_(ORT_LOGGING_LEVEL_WARNING, "TextEmbedding") {
    session_ = nullptr;
}

OnnxRuntimeEmbedding::~OnnxRuntimeEmbedding() {
    unload_model();
}

bool OnnxRuntimeEmbedding::load_model(const std::string& model_path) {
    try {
        std::string tokenizer_file = model_path + "tokenizer.json";
        std::string model_file = model_path + "model.onnx";

        std::cout << "Tokenizer file: " << tokenizer_file << ", model file: " << model_file << std::endl;

        if (tokenizer_file.empty() || model_file.empty()) {
            throw std::runtime_error("Tokenizer or model file path is empty");
        }
        
        load_tokenizer_from_json(tokenizer_file);

        Ort::SessionOptions session_options;
        session_ = new Ort::Session(env_, model_file.c_str(), session_options);
        std::cout << "Model loaded successfully: " << model_file << std::endl;

        // print_model_io_info(*session_);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

void OnnxRuntimeEmbedding::unload_model() {
    if (session_) {
        delete session_;
        session_ = nullptr;
        std::cout << "Model unloaded." << std::endl;
    }
}

std::vector<float> OnnxRuntimeEmbedding::embed(const std::string& text) {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized. Call load_tokenizer_from_json first.");
    }

    std::vector<int32_t> ids = tokenizer_->Encode(text);
    if (ids.empty()) {
        throw std::runtime_error("Tokenizer returned empty ids for text: " + text);
    }

    // 加上 special tokens
    if (bos_token_id_.has_value()) {
        ids.insert(ids.begin(), bos_token_id_.value());
    }
    if (eos_token_id_.has_value()) {
        ids.push_back(eos_token_id_.value());
    }

    std::vector<int64_t> input_ids(ids.begin(), ids.end());
    std::vector<int64_t> attention_mask(input_ids.size(), 1); // 简单填充全1

    // ===== 打印日志 =====
    std::cout << "[Debug] CPP input_ids: [";
    for (size_t i = 0; i < input_ids.size(); ++i) {
        std::cout << input_ids[i];
        if (i + 1 != input_ids.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "[Debug] CPP attention_mask: [";
    for (size_t i = 0; i < attention_mask.size(); ++i) {
        std::cout << attention_mask[i];
        if (i + 1 != attention_mask.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    // ===================

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());

    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());

    std::vector<const char*> input_names = {"input_ids", "attention_mask"};
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(std::move(input_tensor));
    input_tensors.emplace_back(std::move(mask_tensor));

    std::vector<const char*> output_names = {"sentence_embedding"};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                       input_names.data(),
                                       input_tensors.data(),
                                       input_tensors.size(),
                                       output_names.data(),
                                       1);

    float* float_array = output_tensors[0].GetTensorMutableData<float>();
    auto shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = shape_info.GetShape();

    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    return std::vector<float>(float_array, float_array + output_size);
}

void OnnxRuntimeEmbedding::load_tokenizer_from_json(const std::string& json_path) {
    namespace fs = std::filesystem;

    if (!fs::exists(json_path)) {
        throw std::runtime_error("Tokenizer file does not exist: " + json_path);
    }

    std::ifstream file(json_path);
    if (!file) {
        throw std::runtime_error("Unable to open tokenizer file: " + json_path);
    }

    std::string json_blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json_blob);
    if (!tokenizer_) {
        throw std::runtime_error("Failed to initialize tokenizer from: " + json_path);
    }

    // 修复：tokenizer_->TokenToId 返回 int，-1 表示找不到
    int bos_id = tokenizer_->TokenToId("<s>");
    if (bos_id != -1) {
        bos_token_id_ = bos_id;
    }

    int eos_id = tokenizer_->TokenToId("</s>");
    if (eos_id != -1) {
        eos_token_id_ = eos_id;
    }

    std::cout << "[Tokenizer] BOS ID: " << bos_id << ", EOS ID: " << eos_id << std::endl;
}

} // namespace text_embedding
