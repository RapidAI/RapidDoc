# 公式投影外提候选模型验证

后续压缩实验及保留的半大小候选见 [公式模型压缩与提速候选验证](2026-10-29-formula-compression-candidates.md)。

## 改动范围

独立生成 `build/formula-projection-candidate-20261029/pp_formulanet_plus_m.hoisted.onnx`，将图片特征投影 `MatMul.0` 和固定偏置 `Add.20` 从解码循环移到外层图；对应两个 initializer 同步外移。

保留所有权重数值、其余节点、生成规则、词表和输入尺寸。没有量化、截断输出、降低分辨率或针对文档加规则。原模型与候选模型均为 593,908,338 字节，本次不减少模型文件大小。

原模型 SHA256：`52afa60ff51ad66cd7840a74188a058fe99081e617d3bd780117a2d67d94f50d`。

候选 SHA256：`a9d91509f48933088f9413ff22db61c3d25e0d588ede466a089c9845600056a2`。

候选通过 ONNX checker。构建前后、全部验证结束后，均核对原模型哈希不变。本次没有修改默认加载规则、重新编译 Rust 或安装 wheel；现有 demo 未显式指定候选时仍加载原模型。

## 冻结输入对照

使用此前保存的 74 个真实 FP32 输入，校验输入哈希。各轮独立进程、串行执行，采用相同 ORT 1.26.0 CPU、自动线程、CPU arena 和 ALL 图优化；关闭 profiling，各自进行相同短例预热。计时仅包含推理调用，不含预热、加载、输入读取和输出校验。

| 顺序 | 模型 | 74 个公式耗时 | 原始 token 校验 |
| --- | --- | ---: | --- |
| A1 | 原模型 | 81.530 秒 | 全部一致 |
| B1 | 候选 | 63.961 秒 | 全部一致 |
| B2 | 候选 | 68.587 秒 | 全部一致 |
| A2 | 原模型 | 67.259 秒 | 全部一致 |

原模型平均 74.394 秒，候选平均 66.274 秒，观察到耗时减少 **10.9%**。进程 CPU 时间均值分别为 819.695、734.570 秒。

原模型自身两轮差异明显，B2 也略慢于 A2，不能承诺每次都有相同收益。只读计数器记录到处理器性能指标和可用内存波动；未修改电源计划、线程亲和性、优先级或其他应用。

两组额外动态 batch 对照也通过：输入索引 `(0,1)` 与短长混合 `(16,51)`，batch=2，输出分别为 `[2,53]`、`[2,151]`，候选与原图的完整输出数组相同，包括结束后的填充。

## 内存

上述独立公式进程的两轮均值：

| 指标 | 原模型 | 候选 |
| --- | ---: | ---: |
| 进程峰值 working set | 1352.4 MiB | 1347.8 MiB |
| 结束时 private commit | 1822.3 MiB | 1818.2 MiB |

绝对值包括解释器、运行库、冻结输入和 Session，并非仅模型权重。差异仅约 4–5 MiB，结论是内存基本不变。

## 实际 Rust PDF 对照

使用 `.venv` 中现有 `rapid_doc_rust`，通过 `formula_config.model_dir_or_path` 显式选择原版或候选，分别解析同一个 `demo/pdfs/03.pdf`。仍为串行进程、相同导出范围和默认 batch=1；只有公式模型路径不同。

| 指标 | 原模型 | 候选 |
| --- | ---: | ---: |
| 公式阶段，74 个公式 | 45.355 秒 | 35.967 秒 |
| 全流程，含首次加载、导出和 JSON 回读 | 77.359 秒 | 57.764 秒 |

两者均通过既有参考校验：10 页、10907 个 Markdown 字符、39 个图片资产。Markdown、middle JSON、content list 结果一致，39 张图片的模式、尺寸及像素哈希一致；另直接比较了两次输出的 Markdown 字节及解析后的 JSON。

这是单组实际 PDF 对照。整体耗时减少约 25.3%，但其他阶段也有波动，例如表格模型 2.529→2.012 秒，而版面 6.757→7.351 秒；不能将整体降幅全部归因于公式改图，也未证明任意文档都有相同收益。整份 PDF 的峰值 working set 差异同样只有约 5 MiB。

## 本地试用候选

在现有 `RapidDoc(...)` 的 `formula_config` 中添加路径，保留其他已有配置：

```python
formula_config={
    "model_dir_or_path": (
        r"D:\CodeProjects\doc\RapidAI\RapidDoc\build\formula-projection-candidate-20261029"
        r"\pp_formulanet_plus_m.hoisted.onnx"
    ),
}
```

无需重新编译或安装 Python 包。删除该覆盖参数即可回到原来的默认加载行为。本次仅验证已记录的输入、batch=2 两组和一份 PDF，不等于所有输入都已证明逐位一致，也未包含 GPU 验证。

## 复现与产物

构建与验证工具仅用于开发，不增加运行时 Python 包的依赖：

```text
.venv/Scripts/python.exe scripts/migration/optimize_formula_projection.py --output build/<新的候选目录>
.venv/Scripts/python.exe scripts/migration/validate_formula_projection.py --directory build/<新的候选目录>
.venv/Scripts/python.exe scripts/migration/validate_formula_projection.py --directory build/<新的候选目录> --batch-smoke
```

工具拒绝覆盖已有候选和对照报告。PDF 对照工具增加可选环境变量 `RAPIDDOC_BENCH_FORMULA_MODEL`，未设置时保持原有行为。

`build/formula-projection-candidate-20261029/` 保存：`manifest.json`、四轮独立报告及只读遥测、`comparison.json`、`batch-smoke.json`、`measurement-summary.json`、`pdf-comparison.json`。

PDF 输出分别位于 `build/formula-projection-pdf-original-20261029/03/auto/`、`build/formula-projection-pdf-candidate-20261029/03/auto/`。未跑全量回归，未构建 Rust。
