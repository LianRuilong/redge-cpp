#include "text_embedding_factory.h"
#include <iostream>

int main() {
    auto embedding = EmbeddingFactory::create(InferenceBackend::ONNXRUNTIME);
    if (!embedding) {
        std::cerr << "Failed to create embedding instance!" << std::endl;
        return 1;
    }

    if (!embedding->load_model("resource/model/multilingual-e5-small/model.onnx")) {
        std::cerr << "Model loading failed!" << std::endl;
        return 1;
    }

    std::vector<float> vec = embedding->embed("你好，世界！");
    std::cout << "Vector size: " << vec.size() << std::endl;

    embedding->unload_model();
    return 0;
}
