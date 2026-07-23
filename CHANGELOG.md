# 🚀 RecIS v1.2.0 Release Notes

We are excited to announce the release of **RecIS v1.2.0**. This version brings **offline server** support, **MFU/MFU metrics** monitoring, **sparse Adagrad** optimizer, **trace writer v2**, and syncs **column-io 0.3.0** with comprehensive open-source cleanup.

---

## 🌟 Key Highlights

| Category | Description |
| --- | --- |
| **🏆 Framework** | **Offline Server** support; **Trace Writer v2**; **Sparse Adagrad** optimizer; **MFU metrics** monitoring. |
| **⚡ Performance** | Optimized checkpoint save latency; Removed useless unique in grad worker mean. |
| **🔧 Open-Source** | Synced **column-io 0.3.0**; Fixed INTERNAL_VERSION conditional compilation; Removed internal submodule dependencies; Fixed `from_list_string` empty array bug. |
| **🌐 Compatibility** | Support build on multiple Python 3.x versions; Python 3.13 compatibility fixes. |

---

## 📝 Detailed Changelog

### Features

* **ci:** parallelize twine upload (xargs -P 8) and fail-fast in yuque docs ([ad19034](https://github.com/alibaba/RecIS/commit/ad19034bebb1c82faef6e8eff992cd6d6ac62ddc))
* **ci:** recis ppu whl rename from [ppu] to [cuppu], same with column-io ([f002e2a](https://github.com/alibaba/RecIS/commit/f002e2ac013f27e898e2cfe693841a673ad7fdac))
* **embedding:** add custom combiner registry ([9013818](https://github.com/alibaba/RecIS/commit/90138185492df52f51c82d70dded365aff85b5a4))
* **embedding:** embedding engine add insert api ([43162fa](https://github.com/alibaba/RecIS/commit/43162fa2bc8be6ac718e2ed92ebd62cee05e644d))
* **embedding:** EmbeddingEngine no_reduce returns post-a2a embedding ([c366219](https://github.com/alibaba/RecIS/commit/c366219d887d6651bc04cfb01993121c046e7a66))
* **embedding:** enable pinned memory for cpu hashtable ([b2de2e7](https://github.com/alibaba/RecIS/commit/b2de2e7d9a989a34cb0a3b6f75a2eb09b1be344a))
* **embedding:** support fp16 & bf16 in constant generator. ([944967a](https://github.com/alibaba/RecIS/commit/944967a8823ae50985772c9dc3fb9f64d439376b))
* **feature_engine,ops,framework,embedding:** add topk & ragged index functions. ([d2f751b](https://github.com/alibaba/RecIS/commit/d2f751b121506e9616f89a748c9462965d4d75a1))
* **framework:** add flops_metric in monitor by profiler ([dfeae8b](https://github.com/alibaba/RecIS/commit/dfeae8b9001426ee95976f6d5684724ab29bcbc6))
* **framework:** add mos metric report and mean metric ops ([899e0c0](https://github.com/alibaba/RecIS/commit/899e0c0eb499e13a114f3f435b0773eb6ec57c00))
* **framework:** add stateful point type metric_sticky ([6f4b940](https://github.com/alibaba/RecIS/commit/6f4b940ff5105c7c2506e0f49fe9beede1434e7d))
* **framework:** Align checkpoint management with openlm_hub standards ([6b58e46](https://github.com/alibaba/RecIS/commit/6b58e4688df1f999b3e449707c33a73bc29b919d))
* **framework:** enable export without sparse ([925b8ac](https://github.com/alibaba/RecIS/commit/925b8acac0b78f491fe9e8a31de5282cf6a386bd))
* **framework:** impl sparse adagrad optimizer ([223db4f](https://github.com/alibaba/RecIS/commit/223db4fb6c2df71c7623dd4f733a749fbfd06cb6))
* **framework:** model bank load tf convert dense pkl ([7aa8420](https://github.com/alibaba/RecIS/commit/7aa8420fc93764412250db781a42995bc872ca2d))
* **framework:** offline_server支持更多的返回类型 ([a5d3101](https://github.com/alibaba/RecIS/commit/a5d3101fc0e75da70a9fee669e8bff9e234ea32a))
* **framework:** python3.13移除cgi修复及微调metric采集逻辑 ([a9824e2](https://github.com/alibaba/RecIS/commit/a9824e22491f178187a72fde78b23dc1979c96df))
* **framework:** support build logic on multiple py3.x version ([2198e1a](https://github.com/alibaba/RecIS/commit/2198e1a301edf8a046df6fd10c2fa22288b5dd98))
* **framework:** support offline server in recis ([df9f3b3](https://github.com/alibaba/RecIS/commit/df9f3b380247dcdcb36635ff7a67bcf22b585300))
* **framework:** supports MFU metrics (either auto-calc or manually configured) ([b626ef0](https://github.com/alibaba/RecIS/commit/b626ef034f8cc6dd3cc5907c93fc04b720067d29))
* **framework:** trace_writer v2 ([28436ca](https://github.com/alibaba/RecIS/commit/28436ca4a3e91116c2d333590e0054000b0b51ac))
* **framework:** 统一管理版本名称 & ⚠️(breaking) 切换fslib模块的默认编译标准为abi=1 ([8a14629](https://github.com/alibaba/RecIS/commit/8a14629d436fe0f7c09c5a943219a9167925801f))
* **framework:** update cmake to gt 3.28 ([c15d1cf](https://github.com/alibaba/RecIS/commit/c15d1cf119f343973b343c705d773b102ff79292))
* **framework:** upgrade fslib & open alog. ([94d0ad5](https://github.com/alibaba/RecIS/commit/94d0ad54bb61501cdb6f35ce7586f3bf57ce512a))
* **framework:** upgrade fslib to singleton version ([144de86](https://github.com/alibaba/RecIS/commit/144de86c941e0364292d7ef1eaa6d3e53be19fb4))
* **io:** add compress flag for _indicator handling ([bc98afb](https://github.com/alibaba/RecIS/commit/bc98afbe4e185d3e71162edee4a65be8120e0453))
* **io:** adapted to old column_io version ([20cd8b3](https://github.com/alibaba/RecIS/commit/20cd8b399565ba61453febb099507b105618b907))
* **io:** ComboOdpsDataset ([744ea95](https://github.com/alibaba/RecIS/commit/744ea951f72674e8a03fd42bbf31c2d6bd7ba806))
* **io:** create trace partition via tunnel when supported ([7c1b63d](https://github.com/alibaba/RecIS/commit/7c1b63d5d1652eb4f5fe6323ad20fe7ce3bb6b2b))
* **io:** support user-defined tensor modules in dataset base ([ec7b902](https://github.com/alibaba/RecIS/commit/ec7b9022a306eb3b2f454661405c584b3ab34b0c))
* **io:** Trace hook compatibility with multi-cloud clusters ([cf4702a](https://github.com/alibaba/RecIS/commit/cf4702a028cb4f4abbc8c4f78095f84ae5018f58))
* **io:** Trace hook compatibility with specical clusters ([d7dd61b](https://github.com/alibaba/RecIS/commit/d7dd61b3f8077612cafcd0d178b605fd802e5517))
* **ops:** add compute real length op ([4e73932](https://github.com/alibaba/RecIS/commit/4e739328dc875fd4ad6c60bc1c779bf6bbb2679f))
* **ops:** add fused djb2hash and sdbmhash op ([083cfdc](https://github.com/alibaba/RecIS/commit/083cfdc9ced1206fcf6a03213c1fc4b7a58ea7ed))
* **ops:** add multi hash ops for feature engine ([8db9ae8](https://github.com/alibaba/RecIS/commit/8db9ae81d450b489aa9a71f501fcff4f8c398e59))
* **ops:** add weight decay for adagrad, enable grad norm tool ([f3c06a1](https://github.com/alibaba/RecIS/commit/f3c06a108ffdd48e3f1a98df43a63dd5d3750018))
* **ops:** support ppu170 sdk. ([7fddf05](https://github.com/alibaba/RecIS/commit/7fddf0582fbafb671b1ba3fe811188b0d8828a1d))
* **ops,framework:** add mask op, update fg for search ([9747f81](https://github.com/alibaba/RecIS/commit/9747f81e588f02788824346018701650c443b1d9))
* **serialize:** support save and load model without sparse ([6d428b8](https://github.com/alibaba/RecIS/commit/6d428b8f45be8fa9be6bdc056ef7d7548f3066c9))

### Bug Fixes

* **checkpoint:** add storage prefix while update mos uri with xpfs. ([4bf4cb1](https://github.com/alibaba/RecIS/commit/4bf4cb1214fd4bde32ce0708790958ba4449e6d7))
* **checkpoint:** fix bug about load optim when use accelerate ([752953d](https://github.com/alibaba/RecIS/commit/752953d0f76872949e802f3fa02766b86c9a9308))
* **checkpoint:** fix bug of checkpointreader slowly ([63520cc](https://github.com/alibaba/RecIS/commit/63520cc21972666c77880a573e690ebfa33d4866))
* **embedding:** prevent CG probe index wraparound ([fde78f7](https://github.com/alibaba/RecIS/commit/fde78f7c9d8a91a863feea18e98b00b105cf23a7))
* **embedding:** remove duplicate ids & keep first while loading. ([a529bc8](https://github.com/alibaba/RecIS/commit/a529bc88b5ccde75c87fc05e4c8a251a7e68b663))
* **embedding,serialize:** fix nan values during checkpoint loading ([d7c35f3](https://github.com/alibaba/RecIS/commit/d7c35f34bf92e5005a40e761e2c71e29a9ff799a))
* **framework:** accelerator will warp named_optimizer to accelearte_optimizer ([0d4a189](https://github.com/alibaba/RecIS/commit/0d4a189a3903883b4685a4e82011e0ceac930501))
* **framework:** correct file system checks for checkpoint retrieval ([faa7f6c](https://github.com/alibaba/RecIS/commit/faa7f6c9842d1311e73199809eacf00779bf3bae))
* **framework:** create new version for model_export ([3bbdebd](https://github.com/alibaba/RecIS/commit/3bbdebd63a1d91c8d0471d53cb7087919a3be7ca))
* **framework:** fix bug of grad accumulate ([c739522](https://github.com/alibaba/RecIS/commit/c739522a532fcc6ce9e37dc5775aaa681c28b7a5))
* **framework:** fix bug when path not exists ([770990e](https://github.com/alibaba/RecIS/commit/770990ed08fb9d93ae0f35972e0df60514f3d636))
* **framework:** fix peak FLOPS for gpu cards ([c4b0d80](https://github.com/alibaba/RecIS/commit/c4b0d80917973f8c8755ecec59327b290acb37c4))
* **framework:** fix RTP torch_fx_tool typo bug ([bff6a4e](https://github.com/alibaba/RecIS/commit/bff6a4e7dcda7eb09b9d2b8053c12412a5faad9a))
* **framework:** log_writer compile error in higher gcc version ([30bb000](https://github.com/alibaba/RecIS/commit/30bb0003eed40fdc2ce41d2a6adc927aac099c00))
* **framework:** refine logger hook and fix mos import ([8b2bdf3](https://github.com/alibaba/RecIS/commit/8b2bdf30be41d5696b70277a6a0b0f3b925e8c4b))
* **framework:** sync after save ([474cb15](https://github.com/alibaba/RecIS/commit/474cb150827d8b15de908778f8257c581bc13d9f))
* **framework:** update search RTP export_torch_fx_tool ([94faa8b](https://github.com/alibaba/RecIS/commit/94faa8ba08b9c04ab2cc869050cd911465c3af5e))
* **io:** do not save io state for odps window io ([6daf86d](https://github.com/alibaba/RecIS/commit/6daf86dfa6b257755cf9aa973eb57837f94a9fed))
* **io:** fix bug of cannot append to ec file ([ae97a1a](https://github.com/alibaba/RecIS/commit/ae97a1a3e6d329819e1e1189524511d186a6e536))
* **io:** io_state is saved along with the checkpoint ([0592db9](https://github.com/alibaba/RecIS/commit/0592db9de735568fd01cf6f137ef9dee20c64171))
* **io:** remove lock and mp ([442e854](https://github.com/alibaba/RecIS/commit/442e85493e9e73566b88a46c26886eda11396995))
* **io:** return lake window end state ([3db246c](https://github.com/alibaba/RecIS/commit/3db246c589bbec5c10483b44c954fb682c53b101))
* **ops:** fix 3D cutoff bounds check ([f632547](https://github.com/alibaba/RecIS/commit/f63254796ebe8089862b3452af130f91ab0dcebe))
* **ops:** fix build issue in rocm ([fe82734](https://github.com/alibaba/RecIS/commit/fe827341eeab4af2ce859ab10ee04116644d7948))
* **ops:** fix launch failure of cutoff ([25f44ec](https://github.com/alibaba/RecIS/commit/25f44ec3a735d37be964765d182cf7b2bcfc105c))
* **ops:** fix runtime dispatch in segment_reduce forward/backward, segment_mean, ragged_tile_backward for FP16/BFloat16 support ([9c536ea](https://github.com/alibaba/RecIS/commit/9c536eafbf3e8f18f45bab6b58ae4dd65e9c8c53))
* **serialize:** fix issue of slow loading ([efd2bcb](https://github.com/alibaba/RecIS/commit/efd2bcb7e8205527b6266a79e9fcef8e92d182a3))
* **serialize:** read json file after writing done ([5cf10b8](https://github.com/alibaba/RecIS/commit/5cf10b8398316a44423dec6496a5f106d528001e))
* **framework:** do not report MOS metrics for checkpoints that have already been deleted ([13297df](https://github.com/alibaba/RecIS/commit/13297df25cf063a61b911b0897ce0395c3926209))

### Performance Improvements

* **framework,serialize:** optimize checkpoint save latency ([05046f5](https://github.com/alibaba/RecIS/commit/05046f5cedbc138b8633718534af975632e4f875))
* **ops,embedding:** rm grad worker mean useless unique && useless item in gen_segment_ids ([b79d61b](https://github.com/alibaba/RecIS/commit/b79d61bdc8eb18a165acf3b87b230b0447804085))

### Documentation

* **docs:** add note abput using output_dir in model_bank ([ac5239f](https://github.com/alibaba/RecIS/commit/ac5239faad1f1b39c825a0329a9a241c89d46f58))
* **docs:** add notable work based on RecIS ([b9a9e36](https://github.com/alibaba/RecIS/commit/b9a9e36478f70023ccc8169b566818b4f4a15f85))

### Build

* **build:** update middle version ([bcac5ec](https://github.com/alibaba/RecIS/commit/bcac5ec22d9c74f13b9ccd38ad9543f2151f424b))
* **ci:** fix package version for amd ([a8f10cb](https://github.com/alibaba/RecIS/commit/a8f10cbbca95e83dd571aee63e83b5270de08632))

### Open-Source Cleanup

* **column-io:** Sync column-io 0.3.0 from internal master with open-source cleanup
* **column-io:** Fix INTERNAL_VERSION conditional compilation (py_interface, dataset_impl, arrow_reader links)
* **column-io:** Fix cmake URLs to use GitHub public links (rapidjson, protobuf, googletest, zlib)
* **column-io:** Clean up internal references (ODPS endpoints, internal paths, kmonitor dependencies)
* **column-io:** Fix `from_list_string` empty array bug (prevent `all([])` returning True)
* **column-io:** Remove hardcoded Alibaba Cloud AccessKey from integration tests
* **column-io:** Replace hardcoded ODPS paths with environment variables
* **framework:** Remove internal submodule `torch_fx_tool` and conditionalize import
* **framework:** Conditionalize `openlm_hub` import in `mos.py`
* **ci:** Remove `.aoneci/` directory

---

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