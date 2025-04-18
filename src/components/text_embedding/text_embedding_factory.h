#pragma once

#include <memory>

#include "text_embedding.h"

namespace text_embedding {

enum class InferenceBackend {
    ONNXRUNTIME,
    MNN,
    TENSORRT
};

class EmbeddingFactory {
public:
    static std::unique_ptr<TextEmbedding> create(InferenceBackend backend);
};

} // namespace text_embedding
