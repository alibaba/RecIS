# 🚀 RecIS v1.1.0 Release Notes

We are excited to announce the release of **RecIS v1.1.0**. This version marks a significant milestone with the introduction of **Model Bank 1.0**, native **ROCm support**, and substantial performance optimizations for large-scale embedding tables.

---

## 🌟 Key Highlights

| Category | Description |
| --- | --- |
| **🏆 Framework** | **Model Bank 1.0** officially arrives; New **Negative Sampler** and **RTP Exporter** support. |
| **⚡ Performance** | Introduction of **Auto-resizing Hash Tables** and **Fused AdamW TF** CUDA operations. |
| **🌐 Compatibility** | Expanded hardware support for **AMD ROCm**; Fixed non-NVIDIA device kernel launches. |
| **🛡️ Robustness** | Improved multi-node synchronization and robust handling for empty tensor edge cases. |

---

## 📝 Detailed Changelog

### Bug Fixes

* **checkpoint:** fix mos version format, update use openlm api ([854bbb3](https://github.com/alibaba/RecIS/commit/854bbb3e59eb9cc24e641c59c56885f9ff40998f))
* **checkpoint:** refine torch_rank_weights_embs_table_multi_shard.json format ([d5e7a5c](https://github.com/alibaba/RecIS/commit/d5e7a5c71d1475a6413e879deffdb1aa3943487f))
* **checkpoint:** walk around save bug, deal with xpfs model path ([ae99728](https://github.com/alibaba/RecIS/commit/ae99728b265020035ecbd463e661c0c6f3fecbf7))
* **embedding:** fix empty kernel launch in non-nvidia device ([2e310d0](https://github.com/alibaba/RecIS/commit/2e310d09dac2ef9bb8e7518aeeb8bf615688b195))
* **embedding:** fix insert when size == 1 ([7702c9e](https://github.com/alibaba/RecIS/commit/7702c9e2f736d83fe04fac405cee19863edb89c8))
* **framework:** add an option for algo_config  for export ([0ad4c3f](https://github.com/alibaba/RecIS/commit/0ad4c3f8df2ebca2e90fd0a2910da9b91481afff))
* **framework:** fix bugs of invalid index, grad accumulation; add clear child feat ([1e7acf9](https://github.com/alibaba/RecIS/commit/1e7acf9a8a6e0b89336a33d4e34d7d2fef3712ce))
* **framework:** fix eval in trainer ([676a053](https://github.com/alibaba/RecIS/commit/676a05333ed2985c90d22cd3c349033b0ef443c7))
* **framework:** fix fg && exporter bugs ([3964ce2](https://github.com/alibaba/RecIS/commit/3964ce2f2faac1a53d882f8b9f7e45c6c042aba5))
* **framework:** fix load extra info not in ckpt ([a64cd00](https://github.com/alibaba/RecIS/commit/a64cd00cc4691ac78e0418074ab4cc3670436632))
* **framework:** fix loss backward ([7d9a41b](https://github.com/alibaba/RecIS/commit/7d9a41bbc49a5dfdc980ccd7309663c134661508))
* **framework:** fix some bug of model bank ([be196db](https://github.com/alibaba/RecIS/commit/be196dbaacc176a21f0b820c2a6f8fda48a7e2d3))
* **framework:** fix window io failover ([cde3049](https://github.com/alibaba/RecIS/commit/cde3049989bd67a38703e1458eb2f52eef9028c0))
* **framework:** reset io state when start another epoch ([f918f24](https://github.com/alibaba/RecIS/commit/f918f2409a0f5d9f4e4fa7feaf4476687a711435))
* **io:** fix batch_convert row_splits when dataset read empty data ([44661ab](https://github.com/alibaba/RecIS/commit/44661ab1b83c73b0e0726ec8b6921bd685a82659))
* **io:** fix None data when window switch ([e788b4d](https://github.com/alibaba/RecIS/commit/e788b4da8fffc097feb2cd6479fe90ba627121e8))
* **io:** fix odps import bug ([7c13f09](https://github.com/alibaba/RecIS/commit/7c13f0915da29c9b194acda214ceb04d126e2187))
* **io:** use openstorage get_table_size directly ([d5c0952](https://github.com/alibaba/RecIS/commit/d5c09521d9651d644590073df152c2c8d1d05366))
* **ops:** fix bug in fast atomic operations ([fea8d47](https://github.com/alibaba/RecIS/commit/fea8d475bc5ec62e82511fb3e89cae3f62b47849))
* **ops:** fix dense_to_ragged op when check_invalid=False ([#14](https://github.com/alibaba/RecIS/issues/14)) ([300a77b](https://github.com/alibaba/RecIS/commit/300a77b980e3b8d29a22fa96dc1b8749bb0c9aa7))
* **ops:** fix edge cases for empty tensors and improve CUDA kernel handling ([794be12](https://github.com/alibaba/RecIS/commit/794be124aa5cc026020f7cd57962d5d27daaf468))
* **ops:** fix emb segment reduce mean op ([3f82b9c](https://github.com/alibaba/RecIS/commit/3f82b9c51b4f1c8c93028c69082da71306d64477))
* **ops:** handle empty tensor inputs in ragged ops ([a39fc2a](https://github.com/alibaba/RecIS/commit/a39fc2ac353898a0c8650ffdaa83409c35a17c73))
* **optimizer:** step add 1 should be in-place ([cdb3632](https://github.com/alibaba/RecIS/commit/cdb3632af4aac90c1e33a3aa332fa13837110fb0))
* **serialize:** fix bug of file sync of multi node ([822af49](https://github.com/alibaba/RecIS/commit/822af49eb7a2f81a04a7956b6c82740a11ba7760))
* **serialize:** fix bug of load tensor ([e25eee4](https://github.com/alibaba/RecIS/commit/e25eee4bd9b634e20496e8a1254ebe1b0f95792d))
* **serialize:** fix bug when load by oname ([e5ca3d7](https://github.com/alibaba/RecIS/commit/e5ca3d759ba06feb18f4aff5f8dce958c742791b))
* **serialize:** fix bug when tensor num < parallel num ([a02aded](https://github.com/alibaba/RecIS/commit/a02aded482817ce2cc851bfdd9719e6340055ce6))
* **tools:** fix torch_fx_tool string format ([1d426f8](https://github.com/alibaba/RecIS/commit/1d426f88297dab5e0ab8dd8303d78c0565c3a80c))


### Features

* **checkpoint:** add label for ckpt ([5436b5b](https://github.com/alibaba/RecIS/commit/5436b5b4a42a777b53230ed49bd0776a8bd9c254))
* **checkpoint:** load dense optimizer by named_parameters ([a07dbaf](https://github.com/alibaba/RecIS/commit/a07dbaf97e6010c80c5bd71bd911baa0db844182))
* **docs:** add model bank docs ([ff0d23e](https://github.com/alibaba/RecIS/commit/ff0d23eed1fc370947bf7d87c31f03f54d413f9f))
* **embedding:** add monitor for ids/embs ([2f268eb](https://github.com/alibaba/RecIS/commit/2f268eb51294b045ba6ad1a9af7e2f01ba73ccd6))
* **embedding:** expose methods to retrieve child ids and embs from the coalesced hashtable; fix clear method of hashtable ([b5de207](https://github.com/alibaba/RecIS/commit/b5de207acbd47b7cbc0f3af0bef76d50bc7f2a9a))
* **framework,checkpoint:** change checkpointmanager to save/load hooks ([eb3b441](https://github.com/alibaba/RecIS/commit/eb3b44136bcdc7b417196a7e25c3b734d1bbb292))
* **framework:** [internal] add negative sampler ([8c21517](https://github.com/alibaba/RecIS/commit/8c2151703c69a6fe67b033e02410f9b44e4135e2))
* **framework:** add exporter for rtp ([b8af849](https://github.com/alibaba/RecIS/commit/b8af849e6c2aad475462a740a8b2ff8146910f18))
* **framework:** add skip option in model bank ([00828ce](https://github.com/alibaba/RecIS/commit/00828ce62030d57e1e1740dca96246b5bbde96df))
* **framework:** add some utility to RaggedTensor ([78eca0a](https://github.com/alibaba/RecIS/commit/78eca0a5f49309317df166e5f5f69ca779c21e68))
* **framework:** add window_iter for window pipline ([87886a0](https://github.com/alibaba/RecIS/commit/87886a0c0d5c09739cdce4e7be08cb72a629da5b))
* **framework:** collect eval result for hooks and fix after_data bug ([81d3723](https://github.com/alibaba/RecIS/commit/81d3723f4b4b8a353aeba53adad52186d77c1a24))
* **framework:** enable amp by options ([db5bbe7](https://github.com/alibaba/RecIS/commit/db5bbe7a01947d3061bd026ae1123eeb1665e236))
* **framework:** impl-independent monitor ([24a1631](https://github.com/alibaba/RecIS/commit/24a16314183dfb40eaed3ca5f4396530191271d9))
* **framework:** model bank 1.0 ([488672b](https://github.com/alibaba/RecIS/commit/488672b66145be15246770e16ba051e15c53f5c4))
* **framework:** support filter hashtable for saver, update hook for window, fix metric ([01eb2ae](https://github.com/alibaba/RecIS/commit/01eb2ae460767605a07e70b0a92780235356241c))
* **io:** add  adaptor filter by scene ([c3e6738](https://github.com/alibaba/RecIS/commit/c3e6738344638811e7d6757d52acda680d7653ee))
* **io:** add new dedup option for neg sampler ([61b2cb7](https://github.com/alibaba/RecIS/commit/61b2cb7b2889ea6de6682084472feb4e6e5c9a15))
* **io:** add standard fg for input features ([2deedff](https://github.com/alibaba/RecIS/commit/2deedfff3cf8eeea8131356985b94b24c2406736))
* **ops:** add fused AdamW TF CUDA operation ([05dba24](https://github.com/alibaba/RecIS/commit/05dba24656fa6ee66d4c54eabf37d8648e4ccf06))
* **ops:** add parse_sample_id ops ([78674cd](https://github.com/alibaba/RecIS/commit/78674cd1f86f1218bde55b9d2ff3d388baaaed46))
* **packaging:** support ROCm ([7a626d3](https://github.com/alibaba/RecIS/commit/7a626d3c28057ced47ec8ac4d52e7c87db342151))
* **serialize:** update load metric interface ([66b085d](https://github.com/alibaba/RecIS/commit/66b085db593cd341771b478673279355455427b7))
* update column-io to support ROCm device ([7907158](https://github.com/alibaba/RecIS/commit/790715863993b3f7c0e18db076c6680b49285f2c))


### Performance Improvements

* **embedding:** use auto-resizing hash table ([2f53f53](https://github.com/alibaba/RecIS/commit/2f53f5350c71d437afe4a4515ba8707e10b673cf))

---


# [1.0.0] - 2025-09-11

## 🎉 Initial Release

RecIS (Recommendation Intelligence System) v1.0.0 is now officially released! This is a unified architecture deep learning framework designed specifically for ultra-large-scale sparse models, built on the PyTorch open-source ecosystem. It has been widely used in Alibaba advertising, recommendation, searching and other scenarios.

## ✨ New Features

## Core Architecture

- **ColumnIO**: Data Reading
  - Supports distributed sharded data reading
  - Completes simple feature pre-computation during the reading phase
  - Assembles samples into Torch Tensors and provides data prefetching functionality
  
- **Feature Engine**: Feature Processing
  - Provides feature engineering and feature transformation processing capabilities, including Hash / Mod / Bucketize, etc.
  - Supports automatic operator fusion optimization strategies
  
- **Embedding Engine**: Embedding Management and Computing
  - Provides conflict-free, scalable KV storage embedding tables
  - Provides multi-table fusion optimization capabilities for better memory access performance
  - Supports feature elimination and admission strategies
  
- **Saver**: Parameter Saving and Loading
  - Provides sparse parameter storage and delivery capabilities in SafeTensors standard format

- **Pipelines**: Training Process Orchestration
  - Connects the above components and encapsulates training processes
  - Supports complex training workflows such as multi-stage (training/testing interleaved) and multi-objective computation

## 🛠️ Installation & Compatibility

## System Requirements
- **Python**: 3.10+
- **PyTorch**: 2.4+
- **CUDA**: 12.4

## Installation Methods
- **Docker Installation**: Pre-built Docker images for PyTorch 2.4.0/2.5.1/2.6.0
- **Source Installation**: Complete build system with CMake and setuptools

## Dependencies
- `torch>=2.4`
- `accelerate==0.29.2`
- `simple-parsing`
- `pyarrow` (for ORC support)

## 📚 Documentation

- Complete English and Chinese documentation
- Quick start tutorials with CTR model examples
- Comprehensive API reference
- Installation guides for different environments
- FAQ and troubleshooting guides

## 📦 Package Structure

- **Core Library**: `recis/` - Main framework code
- **C++ Extensions**: `csrc/` - High-performance C++ implementations
- **Documentation**: `docs/` - Comprehensive documentation in RST format
- **Examples**: `examples/` - Practical usage examples
- **Tools**: `tools/` - Data conversion and utility tools
- **Tests**: `tests/` - Comprehensive test suite

## 🚀 Key Optimizations

## Efficient Dynamic Embedding

The RecIS framework implements efficient dynamic embedding (HashTable) through a two-level storage architecture:

- **IDMap**: Serves as first-level storage, using feature ID as key and Offset as value
- **EmbeddingBlocks**: 
  - Serves as second-level storage, continuous sharded memory blocks for storing embedding parameters and optimizer states
  - Supports dynamic sharding with flexible expansion capabilities
- **Flexible Hardware Adaptation Strategy**: Supports both GPU and CPU placement for IDMap and EmbeddingBlocks

## Distributed Optimization

- **Parameter Aggregation and Sharding**: 
  - During model creation phase, merges parameter tables with identical properties (dimensions, initializers, etc.) into one logical table
  - Parameters are evenly distributed across compute nodes
- **Request Merging and Splitting**: 
  - During forward computation, merges requests for parameter tables with identical properties and computes sharding information with deduplication
  - Obtains embedding vectors from various compute nodes through All-to-All collective communication

## Efficient Hardware Resource Utilization

- **GPU Concurrency Optimization**: 
  - Supports feature processing operator fusion optimization, significantly reducing operator count and launch overhead
  
- **Parameter Table Fusion Optimization**: 
  - Supports merging parameter tables with identical properties, reducing feature lookup frequency, significantly decreasing operator count, and improving memory space utilization efficiency

- **Operator Implementation Optimization**: 
  - Operator implementations use vectorized memory access to improve memory utilization
  - Optimizes reduction operators through warp-level merging, reducing atomic operations and improving memory access utilization

## 🤝 Community & Support

- Open source under Apache 2.0 license
- Issue tracking and community support
- Active development by XDL Team

---

For detailed usage instructions, please refer to our [documentation](https://alibaba.github.io/RecIS/) and [quick start guide](https://alibaba.github.io/RecIS/quickstart.html).