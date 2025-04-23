#pragma once

#include "text_embedding.h"

#include <memory>
#include <optional>
#include <shared_mutex>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

namespace text_embedding {

class OnnxRuntimeEmbedding : public TextEmbedding {
public:
    OnnxRuntimeEmbedding();
    ~OnnxRuntimeEmbedding() override;

    bool load_model(const std::string& model_path) override;
    void unload_model() override;
    std::vector<float> embed(const std::string& text) override;

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    std::optional<int32_t> bos_token_id_;
    std::optional<int32_t> eos_token_id_;
    mutable std::shared_mutex tokenizer_mutex_;

    void init_tokenizer(const std::string& json_path);

    std::vector<int64_t> build_input_ids(const std::string& text);
    std::vector<Ort::Value> prepare_input_tensors(const std::vector<int64_t>& input_ids,
                                                  const std::vector<int64_t>& attention_mask);
    std::string select_output_name(const std::vector<std::string>& output_names);
    std::vector<Ort::Value> run_model(const std::vector<const char*>& output_names,
                                      const std::vector<Ort::Value>& input_tensors);
    std::vector<float> extract_tensor_data(const std::vector<Ort::Value>& output_tensors,
                                           const std::string& output_name);
    std::vector<float> mean_pooling(const std::vector<Ort::Value>& output_tensors,
                                    const std::vector<int64_t>& attention_mask);
};

} // namespace text_embedding
