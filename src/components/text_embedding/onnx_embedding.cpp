#include "onnx_embedding.h"
#include <iostream>

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
    if (!session) {
        throw std::runtime_error("Model not loaded!");
    }
    // TODO: Tokenize text and run inference
    return std::vector<float>{}; // Placeholder
}
