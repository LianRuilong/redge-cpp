# AI 能力服务（PC & 手机端）

## 项目概述
本项目旨在构建一个适用于 PC 和手机端的 AI 能力服务，提供可靠的大模型推理、知识库管理、图像处理、语音处理以及基础的 Agent 框架与能力，为上层应用提供稳定的 AI 支持。

## 核心功能
- **大模型推理**：支持本地大模型加载、推理请求处理及多模型管理。
- **知识库**：提供文档解析、向量化、索引构建和高效检索能力。
- **图像处理**：支持背景去除、图像增强、OCR 识别等功能。
- **语音处理**：支持语音识别、语音合成及关键词检测。
- **Agent 框架**：支持多任务 AI Agent 交互与执行。
- **API 服务**：提供标准化 API，便于应用程序集成。

## 目录结构
```bash
redge-cpp$ tree -L 2
.
├── base
├── components
│   ├── document_extractor      # 文档解析组件
│   ├── llm_inference           # 大模型推理组件
│   ├── text_embedding          # 文本向量化
│   └── text_reranking          # 文本重排序
├── docs                        # 项目文档
├── services
│   ├── infinite_rag            # 增量式 RAG 知识检索服务
│   └── semantic_router         # 语义路由服务
├── testing                     # 测试代码
└── third_party                 # 依赖的第三方库
    ├── hnswlib                 # 高效最近邻搜索库
    ├── libreoffice             # 文档解析工具
    ├── llama.cpp               # 轻量级 LLM 推理框架
    ├── onnxruntime             # ONNX 模型推理框架
    ├── opencv                  # 计算机视觉库
    ├── sqlite                  # 轻量级数据库
    └── TNN                     # 移动端深度学习推理库
```

## 技术架构
### 1. 核心组件
- **推理引擎**：使用 C++ 实现高效的模型推理，支持 ONNX、TensorRT、OpenVINO 等加速方案。
- **知识库**：基于 SQLite 或轻量级向量数据库，实现高效的知识管理。
- **图像处理模块**：基于 OpenCV、TNN 进行图像处理优化。
- **语音处理模块**：集成 Freeswitch / Unimrcp 或本地 TTS / ASR。
- **Agent 框架**：支持任务调度、上下文管理、多模态融合。
- **服务管理**：支持模块化加载、进程管理、资源监控。

### 2. 交互方式
- **本地 API**：通过 HTTP / WebSocket 提供 API 接口。
- **IPC 机制**：采用 DBus / ZeroMQ / gRPC 实现高效进程间通信。

## 依赖
- C++17 及以上
- CMake
- ONNX Runtime / TensorRT / OpenVINO（可选）
- SQLite / HNSWlib（可选）
- OpenCV（图像处理）
- Freeswitch / Unimrcp（语音处理，可选）
- gRPC / ZeroMQ（可选）

## 安装与部署
### 1. 构建
```sh
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 2. 运行
```sh
./ai_service --config config.json
```

## API 示例
```json
POST /api/inference
{
  "model": "llama2",
  "input": "请介绍一下人工智能"
}
```

```json
POST /api/knowledgebase/query
{
  "query": "计算机视觉的主要应用是什么？"
}
```

## 未来计划
- 增强多端兼容性，优化在移动端的推理性能。
- 增强 Agent 框架的可扩展性。
- 支持更多的 AI 处理能力，如视频分析。
- 提供 Python / JS SDK 方便集成。

---

