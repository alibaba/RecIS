# 目录结构

```
cc/column-io/
├── README.md
├── arrow_reader/       # Arrow Record Batch Reader 适配层. 当前仅odps-combo-dataset依赖该层
├── CMakeLists.txt      # 顶层构建配置
├── dataset/            # Dataset 业务层抽象: 定义 Dataset/Iterator 接口及通用级联组合
├── dataset_impl/       # Dataset 数据源实现: ODPS 表、Lake 列存、本地 ORC 等数据源的实现
├── framework/          # 基础框架层: 数据容器Tensor、状态码、内(显)存分配等公共基础设施
├── lake/               # Lake 数据湖读取（仅内部版本）: FSLib、Scan/Stream/Table 多种 Reader 实现
├── monitor/            # 监控与指标上报: 日志写入、Metric 采集与 Socket 上报
├── odps/               # ODPS Algo SDK 封装（仅内部版本）
├── open_storage/       # ODPS OpenStorage SDK 封装（仅内部版本）
├── plugin/             # 插件机制: AOT 模型加载与 Torch Tensor 转换
└── py_interface/       # Python 接口层: Dataset/
```