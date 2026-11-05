# 版面 V2、V3 优化可行性检查

## 结论

两者有压缩和提速研究空间，但**公式模型的循环外提和解码缓存优化不能直接复用**：本机 V2、V3 都只有外层图，没有 Loop/If 子图。FP16 权重存储可复用方法，速度收益则需另行验证，不能从文件减半推断推理翻倍。

本次没有修改版面模型或默认路径，没有生成量化/压缩候选，也没有宣称精度或速度已经改善。

## 实际本地模型

| 模型 | 原文件 | 估计 FP16 存储后 | 状态 |
| --- | ---: | ---: | --- |
| PP-DocLayoutV2 | 204.06 MiB | 约 102.35 MiB | 估算，未生成与验收 |
| PP-DocLayoutV3 | 123.94 MiB | 约 62.11 MiB | 估算，未生成与验收 |

估算将不少于 128 个元素的 FP32 initializer 改为 FP16 存储，未计额外 Cast 与元数据。它会改变权重数值，不是无损压缩。若运行时还原为 FP32，CPU 计算量基本保留，内存也不一定减半。

原模型目录：`D:/CodeProjects/doc/RapidAI/models/`。

- V2 SHA256：`9fedca3a2ebfdce73fc36f7842a4cabe0a4fe8c0ee33a446ca358ed500907b29`。
- V3 SHA256：`b9a2759e51ee2cc9d98f10cea21caf5862043b882b5f57c683eaf6fa247196a3`。

输入均为 `image [batch,3,800,800]`、`im_shape [batch,2]`、`scale_factor [batch,2]`。输出不同：V2 框输出最后一维为 8；V3 为 7，另输出 `[N,200,200]` 的整数掩码。因此 V2 不能直接当作 V3 替换路径后使用。

## 结构与代表页面分析

原图计数：V2 有 111 个 Conv、167 个 MatMul、18 个 GridSample；V3 有 118 个 Conv、123 个 MatMul、18 个 GridSample。两者都包含特征采样、候选筛选和矩阵计算，不能把全部注意力或 GridSample 当成标准公式注意力直接替换。

对 `demo/pdfs/demo1.pdf` 第 0 页以 144 DPI 渲染 RGB，按现有 PPPreProcess 的 800×800 三次插值、FP32/255、NCHW 方式准备同一份输入，直接调用 ORT 1.26.0 CPU。每个模型预热一次，再跟踪一次；输出保存，两次重复输出逐项一致。

| 模型 | 本次主要算子时间，含 profiling 开销 |
| --- | --- |
| V2 | Conv 约 1845 ms；MatMul 约 1646 ms；FusedMatMul 约 440 ms |
| V3 | Conv 约 962 ms；其次 Add、Where、Cast；MatMul 约 67 ms |

这是**单页算子诊断**，不是 V2/V3 公平速度基准，也不是实际 PDF 全流程性能。两个模型顺序加载，系统仍有波动；不能把这里的绝对毫秒或原图节点数量当作部署速度。

优化后的外层图节点数：V2 为 1613，V3 为 1268。两者优化后 Identity 均为 0，说明原图中的大量 Identity 已被 ORT 自动消除；再写脚本删除这些节点，不能据此承诺新增速度收益。

## 可行的后续顺序

1. 优先 V3：它已在当前 Rust PDF 兼容入口接入，便于做真实流程回归。先生成独立 FP16 存储候选，检查分类、框坐标、置信度、掩码、阅读顺序及最终结构是否变化；不仅比较文件大小。
2. 速度方向优先检查卷积和特征计算。CPU INT8 需要合适的量化范围、必要的校准和代表性文档；它可能改变模型输出，不能默认为无损。纯存储压缩不能代替实际加速验证。
3. 检查适合标准融合的注意力子图，但不能直接套用 M 的固定节点匹配和 16×32 头维度。GridSample 等采样链需保留其语义。
4. 不降低 800×800 输入尺寸、不删检测类别、不提高筛选阈值或减少候选数量来冒充同等精度提速。
5. V2 需单独适配和验证：当前 Rust 兼容入口没有开放 `pp_doclayoutv2`，且其输出格式与 V3 不同。模型优化与接入兼容层是两项工作。

## 复现和保存

```text
.venv/Scripts/python.exe scripts/migration/inspect_layout_models.py --model D:/CodeProjects/doc/RapidAI/models/pp_doclayoutv2.onnx --model D:/CodeProjects/doc/RapidAI/models/pp_doclayoutv3.onnx --output build/<新的检查目录> --profile-pdf demo/pdfs/demo1.pdf
```

不传 `--profile-pdf` 时只读结构，不运行推理。`--page` 可指定代表页，默认 0。输出目录必须不存在，避免覆盖历史数据。工具在检查前后核对原模型哈希；模型输入和输出各自保存为 NPZ，可用于后续候选对照。

本次结果：`build/layout-v2-v3-profile-20261029/summary.json`，以及各模型子目录中的 `report.json`、原始 profiling 跟踪、单独保存的优化图与 `outputs.npz`；根目录的 `inputs.npz`、`input-source.json` 保存真实输入及来源哈希。

优化图为本机诊断资产，不是通用发布模型。没有跑版面优化候选的精度评估、实际 PDF 回归或 CUDA。
