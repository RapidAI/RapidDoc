# DOC 功能收尾与内存优化验收

本轮完成此前列出的内存归因、文本框/批注/代码块共同缺口、真实语料扩充、Office 阶段回归和安装验收。不是对所有 Word 二进制格式作无条件完整性保证。

## 实现

- 正文 story 按段消费，转换后及时释放段落，不再先保留全篇段落数组；字符映射的 piece index 使用 u32，并按实际可恢复字节预留容量。未增加字符数/页数限制。
- Python 绑定借用 middle 结构直接序列化，删除中间 serde_json::Value 对整份结构的复制。公开返回字段不变。
- Python/Rust 同时接入正文和页眉页脚文本框：按 FTXBXS 的 shape id 与 PlcfSpa 锚点绑定，忽略 reusable/备用记录，链式 story 只读一次。正文文本框保留内部表格、列表、样式及资源，在正文锚点后输出独立子流。
- 批注引用按 CP 定位，正文输出上标 `[comment N]`，批注正文按引用所在节附到 page_footnote。复用原有字段和富文本解析。作者、日期和审核状态没有输出。
- code 类型贯穿公开 middle 和四种导出，保留缩进、制表符、空白行及原始换行。Markdown 围栏长度随内容反引号调整；content_list 保留 code/text/text_format，v2 保留 code/code_content，不猜测编程语言。

## 正确性与覆盖

- Python：原有 DOC 测试 19 项，新增及共享 Office 输出/集成测试 45 项，均通过。
- Rust Office 阶段回归：127 项通过，0 失败、0 忽略，涵盖 DOC/DOCX/XLS/XLSX/PPT/PPTX 和共享资源/输出；未跑整个 Rust workspace 或 CUDA。
- 已安装 wheel：32 组二进制构造用例通过，含 5 组预期损坏错误；完整 middle（包括数字索引）及四种导出对照。新功能另有明确文本/结构预期，避免只有同源实现互相比较。
- 独立真实 DOC 从 1 份扩为 12 份：项目 demo 1 份，加 Apache POI 固定提交 `6327f056ba3cfdae326a0261a236460592b80f5c` 的 11 份。覆盖正文、表格/合并格/嵌套表格、列表、页眉页脚、脚注尾注、批注及浮动图片/图形相关文件。按 SHA256 去重；构造样本不计为真实语料。
- 真实文件逐项比较完整 middle、Markdown、content list 和资源内容。仅在图片像素/SVG 内容一致后规范化资源文件名，不放宽文本、样式或结构。另有固定真实参考覆盖四种 Rust 导出。

来源、SHA256 和许可保存在 `packaging/office-native-provenance.json` 及 `tests/fixtures/office_rust_reference_068/APACHE-POI-DOC-{LICENSE,NOTICE}.txt`；外部语料冻结到 `doc_external.json.gz`，日常测试无需联网。

## 分配归因

开发示例 `rust/rapiddoc-office/examples/doc_allocation.rs` 只用于定位分配，不链接进 Python wheel。6000 段文本压力样本：

| 阶段 | 优化前逻辑存活峰值 | 优化后 | 变化 |
|---|---:|---:|---:|
| 正文转换 | 18,174,187 B | 12,488,355 B | -31.3% |
| 绑定元数据导出/序列化 | 31,558,681 B | 14,205,615 B | -55.0% |

正文基线来自本轮初始计数；绑定对照在同一优化后程序中用 `--value` 重现原 Value 复制路径。绑定累计分配由 39,540,530 B 降到 22,085,464 B。计数是分配器请求的存活字节，包含阶段前仍存活对象，不等于 RSS、私有提交或物理内存节省百分比。

原始记录：`build/doc-allocation-{value,borrowed}-20261101.jsonl`、`build/doc-allocation-after-20261101.log`。进程级速度和内存使用独立进程另测。

## 进程级性能结果

阶段回归结束后，顺序启动独立 worker，交换执行顺序运行两轮，每轮冷调用一次、热调用三次。计时、冷解析内存、输出校验各用独立进程；方法与之前的 `2026-11-01-doc-performance-acceptance.md` 相同。时间不含输入读取、导入初始化、文件输出和验证；进程峰值包含导入依赖。

| 输入 | Python 热中位数 | Rust 热中位数 | 加速比 | Python 进程峰值工作集 | Rust 进程峰值工作集 |
|---|---:|---:|---:|---:|---:|
| 真实 docx_01.doc | 2.12985 s | 0.35291 s | 6.04 倍 | 333.49 MiB | 73.49 MiB |
| 生成的 6000 段文本 | 1.42913 s | 0.06228 s | 22.95 倍 | 348.88 MiB | 99.00 MiB |

两种输入的完整输出规范化快照一致。真实输入 Rust 热耗时与之前约 0.353 秒基本相同，不能将 Python 浮动造成的加速比增加解释为 Rust 明显提速；Rust 进程峰值从 79.4 降到 73.5 MiB，约低 7.4%。文本压力样本的 Rust 进程峰值从 114.3 降到 99.0 MiB，约低 13.4%，是重复文本压力结果，不代表复杂文档的同等改善。

真实输入解析阶段采样新增工作集：Python 22.21 MiB，Rust 36.62 MiB；新增私有提交：21.18/61.64 MiB。Rust 新增工作集比上一轮的 42.5 MiB 减少，但仍高于 Python，不能宣称解析算法的所有内存指标均已领先。父进程 2 毫秒目标间隔采样仍可能漏短峰，表格采用 Windows 保存的进程生命周期峰值；逻辑存活分配与 RSS 也不能直接换算。

完整原始报告和输入快照：`build/doc-performance-completion-20261101/{manifest,reports,summary}.json`，包含 20 个独立 worker 报告。仅两种输入做本次性能测量，12 份真实语料用于正确性验收。安装二进制 `_native.pyd` SHA256：`6a57bab3532959dddb33d46337d9bddbacbba0f6495d6d26ba8ca8e1ed047007`，在性能测试前已写入 manifest。

```powershell
.venv/Scripts/python.exe scripts/migration/accept_doc_performance.py --output build/doc-performance-new --rounds 2 --repeats 3 --stress-paragraphs 6000
```

## 复现与安装

项目 `.venv` 已安装 `build/office-doc-completion-20261101/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`；继续使用 `from rapid_doc_rust import RapidDoc` 和 `demo/rust_doc_demo.py`，无需切换解释器或新增参数。

```powershell
.venv/Scripts/python.exe scripts/migration/accept_doc_completion.py
./scripts/migration/test_office_native.ps1
.venv/Scripts/python.exe -m pytest tests/native_office/test_flash_legacy_doc.py tests/native_office/test_doc_extensions.py tests/test_office_structured_spans.py tests/test_office_native_integration.py
```

安装报告 `build/doc-completion-installed-20261101.json`；Office 日志和统计 `build/doc-office-stage-regression-20261101.{log,json}`；Python 日志 `build/doc-python{,-extensions}-regression-20261101.log`。最终字符容量/诊断调整后的 DOC compatibility/story 4 项受影响回归再次通过，记录 `build/doc-final-affected-regression-20261101.log`。统一安装验收可以直接从固定参考中的 Word/Table/ObjectPool 流重建输入，不依赖之前 Cargo 测试留下的临时文件。

## 边界

损坏或没有 shape 绑定的文本框会诊断并跳过，不能可靠定位时不擅自放进正文。文本框仍按逻辑锚点顺序导出，不重建 Word 的精确分页/浮动排版。页眉页脚沿用既有文字/公式投影，未承诺完整保留任意绘图对象。批注作者等元数据未纳入公开接口。12 份真实文件扩大了覆盖，不能替代未知复杂对象和非规范 DOC 的后续回归；遇到差异按格式规则修复，不按文件名特判。
