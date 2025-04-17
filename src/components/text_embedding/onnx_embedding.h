#pragma once

#include "text_embedding.h"
#include <onnxruntime/onnxruntime_cxx_api.h>

class OnnxRuntimeEmbedding : public TextEmbedding {
public:
    OnnxRuntimeEmbedding();
    ~OnnxRuntimeEmbedding() override;

    bool load_model(const std::string& model_path) override;
    void unload_model() override;
    std::vector<float> embed(const std::string& text) override;

private:
    Ort::Env env;
    Ort::Session* session;
    Tokenizer tokenizer;
};
