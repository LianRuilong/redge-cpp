#include "onnx_embedding.h"

#include <iostream>
#include <numeric>
#include <algorithm>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "logger.h"

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
    LOG_DEBUG << "[Model Inputs]:";
    for (size_t i = 0; i < input_names.size(); ++i) {
        LOG_DEBUG << "  Input " << i << ": " << input_names[i];
    }

    // 打印输出名
    auto output_names = session.GetOutputNames();
    LOG_DEBUG << "[Model Outputs]:";
    for (size_t i = 0; i < output_names.size(); ++i) {
        LOG_DEBUG << "  Output " << i << ": " << output_names[i];
    }
}

template <typename T>
std::string to_optional_str(const std::optional<T>& opt) {
    return opt ? std::to_string(*opt) : "n/a";
}

} // namespace

namespace text_embedding {

OnnxRuntimeEmbedding::OnnxRuntimeEmbedding()
    : env_(ORT_LOGGING_LEVEL_WARNING, "TextEmbedding"),
      session_(nullptr) {}

OnnxRuntimeEmbedding::~OnnxRuntimeEmbedding() {
    unload_model();
}

bool OnnxRuntimeEmbedding::load_model(const std::string& model_path) {
    try {
        std::string tokenizer_file = model_path + "tokenizer.json";
        std::string model_file = model_path + "model.onnx";

        LOG_DEBUG << "Tokenizer file: " << tokenizer_file << ", model file: " << model_file;

        if (tokenizer_file.empty() || model_file.empty()) {
            throw std::runtime_error("Tokenizer or model file path is empty");
        }
        
        init_tokenizer(tokenizer_file);

        Ort::SessionOptions session_options;
        session_ = std::make_unique<Ort::Session>(env_, model_file.c_str(), session_options);
        LOG_DEBUG << "Model loaded successfully: " << model_file;

        // print_model_io_info(*session_);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what();
        return false;
    }
}

void OnnxRuntimeEmbedding::unload_model() {
    std::shared_lock lock(tokenizer_mutex_);
    session_.reset();
    tokenizer_.reset();
}

std::vector<float> OnnxRuntimeEmbedding::embed(const std::string& text) {
    std::shared_lock lock(tokenizer_mutex_);

    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized. Call load_tokenizer_from_json first.");
    }

    auto input_ids = build_input_ids(text);
    auto attention_mask = std::vector<int64_t>(input_ids.size(), 1);
    auto input_tensors = prepare_input_tensors(input_ids, attention_mask);

    auto output_names_vec = session_->GetOutputNames();
    if (output_names_vec.empty()) {
        throw std::runtime_error("Model has no outputs.");
    }

    std::string selected_output = select_output_name(output_names_vec);

    if (!selected_output.empty() && selected_output != "last_hidden_state") {
        LOG_DEBUG << "[Debug] Using output name: " << selected_output;
        return extract_tensor_data(run_model({selected_output.c_str()}, input_tensors), selected_output);
    }

    LOG_DEBUG << "[Debug] Falling back to mean pooling over 'last_hidden_state'\n";
    return mean_pooling(run_model({"last_hidden_state"}, input_tensors), attention_mask);
}

void OnnxRuntimeEmbedding::init_tokenizer(const std::string& json_path) {
    namespace fs = std::filesystem;

    if (!fs::exists(json_path)) {
        throw std::runtime_error("Tokenizer file does not exist: " + json_path);
    }

    std::ifstream file(json_path);
    if (!file) {
        throw std::runtime_error("Unable to open tokenizer file: " + json_path);
    }

    std::string json_blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::shared_lock lock(tokenizer_mutex_);

    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json_blob);
    if (!tokenizer_) {
        throw std::runtime_error("Failed to initialize tokenizer from: " + json_path);
    }

    // 获取常见的特殊 token（尝试多种形式）
    std::vector<std::string> bos_candidates = {"[CLS]", "<s>"};
    std::vector<std::string> eos_candidates = {"[SEP]", "</s>"};

    for (const auto& token : bos_candidates) {
        int id = tokenizer_->TokenToId(token);
        if (id != -1) {
            bos_token_id_ = id;
            break;
        }
    }

    for (const auto& token : eos_candidates) {
        int id = tokenizer_->TokenToId(token);
        if (id != -1) {
            eos_token_id_ = id;
            break;
        }
    }

    LOG_DEBUG << "[Tokenizer] BOS ID: " << to_optional_str(bos_token_id_)
          << ", EOS ID: " << to_optional_str(eos_token_id_);
}

std::vector<int64_t> OnnxRuntimeEmbedding::build_input_ids(const std::string& text) {
    std::shared_lock lock(tokenizer_mutex_);

    std::vector<int32_t> ids = tokenizer_->Encode(text);
    if (ids.empty()) {
        throw std::runtime_error("Tokenizer returned empty ids for text: " + text);
    }

    if (bos_token_id_) ids.insert(ids.begin(), bos_token_id_.value());
    if (eos_token_id_) ids.push_back(eos_token_id_.value());
    
    return std::vector<int64_t>(ids.begin(), ids.end());
}

std::vector<Ort::Value> OnnxRuntimeEmbedding::prepare_input_tensors(const std::vector<int64_t>& input_ids,
                                                                     const std::vector<int64_t>& attention_mask) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(input_ids.data()), input_ids.size(), input_shape.data(), input_shape.size());
    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(attention_mask.data()), attention_mask.size(), input_shape.data(), input_shape.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    input_tensors.push_back(std::move(mask_tensor));
    return input_tensors;
}

std::string OnnxRuntimeEmbedding::select_output_name(const std::vector<std::string>& output_names) {
    for (const auto& name : output_names) {
        if (name.find("sentence") != std::string::npos || name.find("embedding") != std::string::npos) {
            return name;
        }
    }
    for (const auto& name : output_names) {
        if (name != "last_hidden_state") {
            return name;
        }
    }
    return "";
}

std::vector<Ort::Value> OnnxRuntimeEmbedding::run_model(const std::vector<const char*>& output_names,
                                                         const std::vector<Ort::Value>& input_tensors) {
    std::vector<const char*> input_names = {"input_ids", "attention_mask"};
    return session_->Run(Ort::RunOptions{nullptr},
                         input_names.data(), input_tensors.data(), input_tensors.size(),
                         output_names.data(), output_names.size());
}

std::vector<float> OnnxRuntimeEmbedding::extract_tensor_data(const std::vector<Ort::Value>& output_tensors,
                                                              const std::string& output_name) {
    auto& tensor = output_tensors[0];
    const float* float_array = tensor.GetTensorData<float>();
    auto shape_info = tensor.GetTensorTypeAndShapeInfo();
    auto shape = shape_info.GetShape();

    size_t output_size = 1;
    for (auto dim : shape) output_size *= dim;
    std::vector<float> result(float_array, float_array + output_size);

    LOG_DEBUG << "[Debug] Embedding shape: [";
    for (size_t i = 0; i < shape.size(); ++i)
        LOG_DEBUG << shape[i] << (i + 1 != shape.size() ? ", " : "");
    LOG_DEBUG << "]\n";

    return result;
}

std::vector<float> OnnxRuntimeEmbedding::mean_pooling(const std::vector<Ort::Value>& output_tensors,
                                                       const std::vector<int64_t>& attention_mask) {
    const float* float_array = output_tensors[0].GetTensorData<float>();
    auto shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = shape_info.GetShape();  // [1, seq_len, hidden]

    if (shape.size() != 3) {
        throw std::runtime_error("Unexpected output shape for last_hidden_state.");
    }

    int64_t seq_len = shape[1];
    int64_t hidden_size = shape[2];

    std::vector<float> pooled(hidden_size, 0.0f);
    int valid_count = 0;
    for (int64_t i = 0; i < seq_len; ++i) {
        if (attention_mask[i] == 0) continue;
        ++valid_count;
        for (int64_t j = 0; j < hidden_size; ++j) {
            pooled[j] += float_array[i * hidden_size + j];
        }
    }

    if (valid_count == 0) valid_count = 1;
    for (float& val : pooled) val /= valid_count;

    LOG_DEBUG << "[Debug] Embedding shape: [" << hidden_size << "]\n";
    return pooled;
}

} // namespace text_embedding
