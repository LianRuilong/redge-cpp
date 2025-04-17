#include "onnx_embedding.h"

#include <iostream>
#include <numeric>      // std::accumulate
#include <algorithm>    // std::transform
#include <string>
#include <vector>
#include <stdexcept>

OnnxRuntimeEmbedding::OnnxRuntimeEmbedding() : env(ORT_LOGGING_LEVEL_WARNING, "TextEmbedding") {
    session = nullptr;
}

OnnxRuntimeEmbedding::~OnnxRuntimeEmbedding() {
    unload_model();
}

bool OnnxRuntimeEmbedding::load_model(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        session = new Ort::Session(env, model_path.c_str(), session_options);
        std::cout << "Model loaded successfully: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

void OnnxRuntimeEmbedding::unload_model() {
    if (session) {
        delete session;
        session = nullptr;
        std::cout << "Model unloaded." << std::endl;
    }
}

std::vector<float> OnnxRuntimeEmbedding::embed(const std::string& text) {
    // 1. 编码输入文本（你需要按你的模型要求做预处理）
    std::vector<int64_t> input_ids = tokenizer.encode(text);
    size_t input_length = input_ids.size();

    // 2. 构造 shape 和数据
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_length)};
    size_t input_tensor_size = input_length;

    // 3. 创建 input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        input_ids.data(),
        input_tensor_size,
        input_shape.data(),
        input_shape.size()
    );

    // 4. 设置输入输出名（必须和模型一致）
    std::vector<const char*> input_names = {"input_ids"};
    std::vector<const char*> output_names = {"last_hidden_state"};  // 替换为你模型的实际输出名

    // 5. 运行模型
    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                       input_names.data(),
                                       &input_tensor,
                                       1,
                                       output_names.data(),
                                       1);

    // 6. 获取输出数据
    float* float_array = output_tensors[0].GetTensorMutableData<float>();

    // 7. 获取输出 shape
    Ort::TensorTypeAndShapeInfo output_shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_shape_info.GetShape();

    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    // 8. 拷贝成 vector<float>
    return std::vector<float>(float_array, float_array + output_size);
}
