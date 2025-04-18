#include "text_embedding_factory.h"
#include "text_embedding.h"
#include "onnx_embedding.h"
// #include "mnn_embedding.h"
// #include "tensor_rt_embedding.h"

namespace text_embedding {

std::unique_ptr<TextEmbedding> EmbeddingFactory::create(InferenceBackend backend) {
    switch (backend) {
        case InferenceBackend::ONNXRUNTIME:
            return std::make_unique<OnnxRuntimeEmbedding>();
        case InferenceBackend::MNN:
            // return std::make_unique<MnnEmbedding>();
            break;
        case InferenceBackend::TENSORRT:
            // return std::make_unique<TensorRtEmbedding>();
            break;
        default:
            break;
    }
    return nullptr;
}

} // namespace text_embedding
