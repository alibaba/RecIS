<p align="center"><img width="30%" src="docs/_static/logo.png" alt="LOGO"></p>
<h1 align="center">  RecIS (Recommendation Intelligence System) </h1>

[中文版](README_zh.md)

RecIS: A unified deep learning framework specifically designed for ultra-large-scale sparse and dense computing. Built on the PyTorch open-source ecosystem, it provides a complete solution for recommendation model training and recommendation combined with multimodal/large model training. Jointly launched by Alibaba's AiCheng Technology and Taobao & Tmall's Advertising Technology and Algorithm Technology teams. Currently widely applied in Alibaba's advertising, recommendation, and search scenarios.

<p align="center">
    <img alt="Static Badge" src="https://img.shields.io/badge/made_by-XDL_Team-blue">
    <img alt="Static Badge" src="https://img.shields.io/badge/version-v1.0.0-green">
    <img alt="Static Badge" src="https://img.shields.io/badge/license-Apache--2.0-blue">
    <a href="https://arxiv.org/abs/2509.20883">
        <img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=FF4500">
    </a>
    <img src="https://img.shields.io/github/stars/alibaba/Recis?style=social" alt="stars">
    <img src="https://img.shields.io/github/issues/alibaba/Recis" alt="GitHub issues">
    <a href="./docs/_static/recis_wechat.png" target="_blank">
      <img src="https://img.shields.io/badge/WeChat-green?logo=wechat" alt="WeChat QR">
    </a>
</p>

## 🎯 Design Goals

**Unified Framework**
- Based on PyTorch open-source ecosystem, unifying sparse-dense framework requirements
- Meeting industrial-grade recommendation model training needs combined with multimodal and large model scenarios

**Performance Optimization**
- Optimizing memory access performance for sparse-related operators
- Providing sparse operator fusion optimization capabilities to fully utilize GPU
- Achieving or even exceeding TensorFlow-based performance

**Ease of Use**
- Flexible feature and embedding configuration
- Automated feature processing and optimization workflows
- Simple sparse model definition

## 🏗️ Core Architecture

RecIS adopts a modular design with the following core components:

<div align="center">

<img src="docs/_static/sys-recis.png" width="100%" alt="System Architecture">

</div>

- **ColumnIO**: Data Reading
  - Supports distributed sharded data reading
  - Supports feature pre-computation during the reading phase
  - Assembles samples into Torch Tensors and provides data prefetching
  
- **Feature Engine**: Feature Processing
  - Provides feature engineering and feature transformation processing capabilities, including Hash / Mod / Bucketize, etc.
  - Supports automatic operator fusion optimization strategies
  
- **Embedding Engine**: Embedding Management and Computation
  - Provides conflict-free, scalable KV storage embedding tables
  - Offers multi-table fusion optimization capabilities for better memory access performance
  - Supports feature admission and filtering strategies
  
- **Saver**: Parameter Saving and Loading
  - Provides sparse parameter storage and delivery capabilities in SafeTensors standard format

- **Pipelines**: Training Process Orchestration
  - Connects the above components and encapsulates training workflows
  - Supports complex training processes including multi-stage (training/testing alternation) and multi-objective computation

## 🚀 Key Optimizations

### Efficient Dynamic Embedding

The RecIS framework implements efficient dynamic embeddings (HashTable) through a two-level storage architecture:

- **IDMap**: Serves as first-level storage, using feature IDs as keys and Offsets as values
- **EmbeddingBlocks**: 
  - Serves as second-level storage, continuous sharded memory blocks for storing embedding parameters and optimizer states
  - Supports dynamic sharding with flexible scalability
- **Flexible Hardware Adaptation Strategy**: Supports both GPU and CPU placement for IDMap and EmbeddingBlocks

### Distributed Optimization

- **Parameter Aggregation and Sharding**: 
  - During model creation, merges parameter tables with identical properties (dimensions, initializers, etc.) into a single logical table
  - Parameters are evenly distributed across compute nodes
- **Request Merging and Splitting**: 
  - During forward computation, merges requests for parameter tables with identical properties and deduplicates to compute sharding information
  - Obtains embedding vectors from various compute nodes through All-to-All collective communication

### Efficient Hardware Resource Utilization

- **GPU Concurrency Optimization**: 
  - Supports feature processing operator fusion optimization, significantly reducing operator count and launch overhead
  
- **Parameter Table Fusion Optimization**: 
  - Supports merging parameter tables with identical properties, reducing feature lookup frequency, significantly decreasing operator count, and improving memory space utilization efficiency

- **Operator Implementation Optimization**: 
  - Implements vectorized memory access in operators to improve memory access utilization
  - Optimizes reduction operators through warp-level merging, reducing atomic operations and improving memory access utilization

## 🏆 Notable work based on RecIS
  - [MOON Embedding](https://arxiv.org/abs/2511.11305): MOON Embedding: Multimodal Representation Learning for E-commerce Search Advertising.
  - [MOON](https://arxiv.org/abs/2508.11999): Generative MLLM-based Multimodal Representation Learning for E-commerce Product Understanding.
  - [LUM](https://arxiv.org/abs/2502.08309): Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model.

## 📚 Documentation

- [Installation Guide](https://alibaba.github.io/RecIS/installation.html)
- [Quick Start](https://alibaba.github.io/RecIS/quickstart.html)
- [Project Introduction](https://alibaba.github.io/RecIS/introduction.html)
- [FAQ](https://alibaba.github.io/RecIS/faq.html)

## 🤝 Support and Feedback

If you encounter issues, you can:

- Check project [Issues](https://github.com/alibaba/RecIS/issues)
- Join our WeChat discussion group
  
<img src="./docs/_static/recis_wechat.png" width="20%" alt="Wechat">

## 📄 License

This project is open-sourced under the [Apache 2.0](LICENSE) license.
