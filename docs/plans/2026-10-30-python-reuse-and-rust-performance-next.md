# Python 复用收益验证与 Rust 后续性能边界

本轮只验证已有复用思路是否值得迁回 Python，并审查 Rust 还有哪些可行方向。没有修改 Python/Rust 生产解析代码、模型、阈值或候选算法，也未开始 Office 移植。

## Python 验证

对 `tests/fixtures/upstream_068/deep_structure_catalog.pdf` 首64页，通过原 Python RapidDoc、原配置和默认64页窗口采集 cProfile；表格/OCR回退保留，公式/印章关闭。输出相对于上一轮 Python 快照的 Markdown、content_list、middle_json、图片名称及解码像素完全一致。

| 复用候选 | 调用次数 | 采集中的累计秒数 |
|---|---:|---:|
| `_build_fragments` | 215 | 0.0183 |
| `_cluster_fragment_rows` | 114 | 0.0864 |
| `_transform_axis_lines` | 447 | 0.0106 |
| `copy.deepcopy`，包含初始化等调用 | 98,303，含递归 | 0.1424 |
| 已有 `_drawing_component_summaries` 缓存入口 | 145 | 0.0378 |
| 已有 `compound_baseline_flags` 缓存入口 | 117 | 0.0036 |

行准备/聚类/变换合计约0.115秒；即使消除这些准备成本，在当前样本中也没有明显整体收益，因此不增加新的生产缓存。Python 通过对象引用、dataclass浅层替换以及现有逐页释放机制，已经避免了 Rust 本轮修正的大部分深层副本。缓存还会增加对象驻留与失效维护成本。

这是一份耗时分析，不是内存分配追踪，不能据此宣称 Python 没有任何内存优化空间，也不能推出其他文档同样没有收益。当前没有证据支持把 Rust 约一半的内存下降复制到 Python；按本轮“收益不明显就不改”的要求保留现状。

采集后整段运行约94.15秒，包含大量函数追踪开销，**不能拿它与 Rust 90.60秒作速度对比**。两者最近无追踪对照仍是 Python36.31秒、Rust90.60秒。

证据：`build/python-native-reuse-profile.pstats`、`build/python-native-reuse-profile64/report.json`、`build/python-native-reuse-analysis.json`。可复现：

```powershell
.venv/Scripts/python.exe -m cProfile -o build/python-native-reuse-profile.pstats scripts/migration/benchmark_native_windows.py --engine python --start-page-id 0 --end-page-id 63 --output build/python-native-reuse-profile64-new --reference build/native-owned-materializers-python64
.venv/Scripts/python.exe scripts/migration/summarize_python_native_reuse.py --stats build/python-native-reuse-profile.pstats --output build/python-native-reuse-analysis.json
```

## Rust 仍有方案，但需要区分规模

当前64页分段：原生收集约19.54秒；页面准备约49.42秒，其中完整表格检测14.08秒、几何及角色收尾15.67秒。父子计时重叠，不可相加。模型不构成这个原生模式差距的主要部分，不能靠更换公式或版面模型修复此项差距。

审查确认：很多内部文字行仍是带完整 `chars` 的 `serde_json::Value`。真实参考首个页面的一行序列化长度约35KB；`closed_rule_pipeline::prepare`、`page_policies::parse` 等只读阶段仍会深拷贝行。Python 创建筛选列表通常只复制对象引用。前一轮已消除多个整页副本，但尚未消除这一层结构性区别。

可行方向按规模划分：

1. 局部只读结构改为借用行，按消费者逐个消除剩余深拷贝。风险较小，但尚未证明可将总耗时降到 Python 以下；不为几处零散复制反复全量构建/回归。
2. 将页面、文字行、字符、路径、表格候选改为 Rust 内部结构体；字符证据在页面内共享，JSON只留在公开诊断/输出边界。继续移植原 Python 规则和顺序，不另写简化算法。这是更完整的改造，需要独立一轮实现和差分验收，不能保证未经实测的提速倍数。
3. 页面并行不是当前直接可用的替代：全文几何、页边角色和前页续表网格存在依赖，直接并行整个页循环会改变语义；也可能抬高峰值内存。

## 与 Office 的顺序

PDF 功能、安装和本轮受影响回归已经有可用基线，性能目标仍未达成。可以先开展用户指定的 DOC/DOCX、XLS/XLSX、PPT/PPTX Rust 移植，把上述 PDF 内部结构重构单列后续，不把未达标改称完成。

Office 新代码应从开始就采用结构体、借用和清晰的数据生命周期，避免重复沿用 PDF 的 JSON 中间请求方式。保留现有 Python 0.6.8 合并后的实现作为行为参考，先迁 OOXML 公共读取与 DOCX，再推进 XLSX/PPTX及旧二进制格式；具体顺序仍须结合现有 Rust Office 代码与公共表格依赖确定，不在本次审查中开始改写。

Office实际依赖盘点、底层Rust库探针、持久Python参考和实施顺序见 [Office移植准备](2026-10-30-rust-office-preparation.md)。
