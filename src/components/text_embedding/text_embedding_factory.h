#pragma once

#include <memory>

#include "text_embedding.h"

enum class InferenceBackend {
    ONNXRUNTIME,
    MNN,
    TENSORRT
};

class EmbeddingFactory {
public:
    static std::unique_ptr<TextEmbedding> create(InferenceBackend backend);
};
