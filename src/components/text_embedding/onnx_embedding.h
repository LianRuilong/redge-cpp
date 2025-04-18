#pragma once

#include "text_embedding.h"

#include <memory>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

class OnnxRuntimeEmbedding : public TextEmbedding {
public:
    OnnxRuntimeEmbedding();
    ~OnnxRuntimeEmbedding() override;

    bool load_model(const std::string& model_path) override;
    void unload_model() override;
    std::vector<float> embed(const std::string& text) override;
    
    void load_tokenizer_from_json(const std::string& json_blob);

private:
    Ort::Env env;
    Ort::Session* session;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
};
