#pragma once

#include "text_embedding.h"
#include "onnx_embedding.h"
// #include "mnn_embedding.h"
// #include "tensor_rt_embedding.h"
#include <memory>

enum class InferenceBackend {
    ONNXRUNTIME,
    MNN,
    TENSORRT
};

class EmbeddingFactory {
public:
    static std::unique_ptr<TextEmbedding> create(InferenceBackend backend) {
        switch (backend) {
            case InferenceBackend::ONNXRUNTIME:
                return std::make_unique<OnnxRuntimeEmbedding>();
            case InferenceBackend::MNN:
                // return std::make_unique<MnnEmbedding>();
            case InferenceBackend::TENSORRT:
                // return std::make_unique<TensorRtEmbedding>();
            default:
                return nullptr;
        }
    }
};
