#pragma once

#include <vector>
#include <string>

class TextEmbedding {
public:
    virtual ~TextEmbedding() = default;

    // 加载模型
    virtual bool load_model(const std::string& model_path) = 0;

    // 卸载模型
    virtual void unload_model() = 0;

    // 文本向量化
    virtual std::vector<float> embed(const std::string& text) = 0;
};
