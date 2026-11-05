# 原生 PDF 慢点：函数级诊断

后续已实施第一轮整链数据传递优化并安装：129 页总耗时约下降 19%～22%，完整输出回归通过；范围和限制见 [实施记录](2026-11-01-native-value-transfer.md)。下文保留实施前诊断证据。

## 结论

当前主要成本是原生分析链在 Rust 内反复构造、复制和释放 `serde_json::Value`，不能归因于模型框架、Python 包装层或 PNG 编码。下一步应成段改造表格候选/页面容器/行内文字准备的内部传参；保留算法顺序、数字精度、输出和异常语义。并非仅替换几个末端 clone 就能解决。

此前独立阶段计时：129 页原生规则 PDF，Python 约 12.55 秒，Rust 约 39.48 秒；Rust 原生分析约 28.34 秒，渲染约 7.54 秒，页面组装约 2.72 秒。此文新增的是定位证据，没有安装性能修复，也没有达到与 Python 持平的目标。

## 实测证据

- 输入：`build/document-memory-20261101/inputs/windows-129.pdf`，129 页普通文字。公开 RapidDoc 接口，`docvortex_rule`、txt、关闭公式和表格模型、64 页窗口；并未跳过规则表格候选。
- Windows DbgHelp 对本工具启动的实际 Python 子进程主线程采样。2752 次有效采样，2749 次调用栈包含 `_native`。这属于主线程墙钟栈采样，包含等待与采样开销，不是 CPU 时间百分比。
- 1960/2749（71.30%）包含原生 `predict_range`；1177/2749（42.82%）包含 Value serializer；410/2749（14.91%）包含 JSON BTreeMap 子树 clone；508/2749（18.48%）包含 mimalloc 对齐分配。**调用栈分类重叠，不能相加。**
- PDFium 库内部占 125/2749（4.55%）。源码确有多次重开 PDF 的差异，但本输入下没有证据表明它是最大的热点；不能据此忽略渲染或推断其它 PDF 的分布相同。
- serializer 的最近业务调用方包括 `table_page::prepare_impl_owned`（177 次）、`native_pdf_document::collect_range_impl`（145 次）、`inline_output::payloads::prepare_parts`（92 次）、`strong_graphics::detect_source`（81 次）、`closed_rule_pipeline::page_sources`（68 次）。这些是采样次数，不是函数调用次数或精确秒数。
- 示例源码：`table_page.rs` 的 `json!(selected)`、`json!(updated)` 和 `json!({"source": analysis, ...})`；`strong_graphics` 空图形分支仍通过 json 宏复制整个 source；行内文字准备按行包装 line 和整页字框/origin 映射。内部 Value 已经存在，json 宏仍进入 serde 的 Value serializer，而不是移动现成树。

## 数字重建成本

`rapiddoc-office` 为公开 Office API 中超长编号的精度启用 serde_json `arbitrary_precision`，Cargo feature 合并使 PDF 同样使用该表示。Value 再序列化会经过 NumberValueEmitter 和数字文本处理；本次堆栈也采到对应路径。

独立 release 微实验：对已保存原生输出进行 4 轮交替测试，每轮 1000 次 `json!({"source": &value})` 与对象插入 `value.clone()`，均计入返回树的释放。

| serde_json 特性 | 宏包装中位秒 | 直接 clone 中位秒 | 比值 |
| --- | ---: | ---: | ---: |
| 当前完整特性，含 arbitrary_precision | 0.19416 | 0.11417 | 1.70 |
| 不含 arbitrary_precision 的对照构建 | 0.09892 | 0.07858 | 1.26 |

每种构建内部两种方式的完整序列化输出字节相同。两种构建之间未宣称数字类型或输出精度等价。微实验不是解析加速结果，禁止将其倍率外推到端到端。**不应直接关闭 arbitrary_precision/float_roundtrip**，那会牺牲既有兼容性。

## 复现与有效性

- 采样源码：`scripts/migration/native_stack_sample.cpp`；使用 Windows SDK DbgHelp、MSVC C++17 编译。工具只跟踪自己启动的 python 子进程，忽略 conhost，主线程挂起后取栈立即恢复，180 秒超时只终止自己的目标进程。
- 有效原始栈：`build/native-rule-stack-symbols-20261101.tsv`；汇总：`build/native-rule-stack-symbols-summary-20261101.json`。通过 `scripts/migration/report_native_stack_sample.py SOURCE --output OUTPUT` 重建。
- 必须在进程存活时加载匹配 PDB，禁用 deferred load。之前 `stack`、`stack-real` 采到了启动器/conhost，`stack-parser` 只有导出符号，均不用于此结论。Rust v0 名字保留原始 mangling；优化后的链接器可能折叠相同实现，因此不把可疑叶节点名称当成精确业务语义。
- Python cProfile：`build/native-rule-python-20261101.prof`，仅用于调用路径核对。原生分析执行 3 个窗口；全文 sources/preparation 是主要业务阶段。cProfile 将大量 Python 调用插桩，本次约 22.40 秒；Rust 栈采样约 43.23 秒；**不能用这两个插桩耗时计算性能比**。
- 两次诊断均完成 129 页，Markdown SHA256 均为 `e46ff1acbb2bea98fe3992afc2400a997f0887689f619197d2e3325e787c0230`。这只证明本次文本输出未变，不替代完整结构回归。
- 微实验源码：`rust/rapiddoc/examples/json_value_transfer.rs`；结果 `build/json-value-transfer-20261101.json`、`build/json-value-transfer-plain-20261101.json`。用现有 serde_json rlib 直接 rustc 编译，避免为诊断重新构建整个解析器。未修改模型、解析器运行配置或 `.venv` 安装包；未跑全量或 CUDA。

## 后续改造边界

先消除上述整条链的 Value 再序列化及整页映射重复搬运：已有 Vec/Map 直接转换为 Value、拥有者移交、只读上下文借用，将对外 JSON 诊断接口与内部传参区分。保持所有候选与仲裁规则，不按某份文档跳过分析。随后对相同安装包、输入和参数独立计时，并核对完整输出；再决定是否投入 PDFium 文档句柄复用。当前诊断不覆盖扫描件的模型/裁图瓶颈。
