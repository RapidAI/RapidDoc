# 全文行内与媒体收尾

统一 release 构建 4 分 57 秒并安装至 `build/rust-wheel-env`。安装验收通过：915 空格规则、1,645 全文行内、35 原始页源整链、6 媒体链（3 个解码 PNG 正例及渲染失败回退）和 3 页实时 PDFium；严格字段类型、所有权、并发与开关门禁通过。源码 3 项相关测试通过。日志为 `build/native-document-inline-*.txt`；既有最终页及 118 模块来源门禁回归见主链。

本阶段继续现有 Python `_materialize_document_inline` 和 `materialize_table_media`，不改写解析算法。原始页源/几何→全文准备/最终页→词界与生成空格修复→链接、样式、上下标及 InlineSpan→表内媒体渲染的接续已实现。文件级证据采集/default PDF 和用户整篇差分仍需完成。

| Python 逻辑 | Rust/SDK 对应 |
| --- | --- |
| `prepare_spacing_lines`、可靠墨迹词界、中文生成空格与连字空格 | `inline_output/spacing.rs`、`spacing/characters.rs`；可靠词界复用已有完整 `spatial::tight` |
| `apply_spacing_lines`：自然语言唯一归块、公式空洞、断词及原文插删偏移 | `spacing.rs`；复用原 `assignments`、`project`、`matching::dehyphen` |
| `_materialize_document_inline` 严格逐页对齐、spacing 先于行内语义、诊断快照 | `native_inline_document.rs`；复用完整已有 `inline_output` |
| 原始全文准备后调用收尾 | `native_document::prepare_text_blocks` 的 `materialize_inline=True`；增加 `finalized_pages`，原 `block_pages/model_list` 检查点保留 |
| 全文行内→表格媒体 | SDK `finalize_native_document(pages, source, ...)`；调用原 Rust 表格媒体模块，输出逐页最终块 |

SDK 另开放 `prepare_native_spacing_lines`、`apply_native_spacing_lines`、`materialize_native_document_inline` 供明确阶段的调用和诊断。`finalize_native_document` 的 pages 是已有原始页源/几何，source 是供表内图片渲染的 PDF；不是直接接收文件的完整解析入口。所有新开关缺省 False，默认策略未提升。

参考为 885 完整空格准备/应用、30 集合/数值来源合同（4 原异常）、1,645 全文行内及诊断（7 原异常）、35 原始页源整链、6 媒体链，其中 3 个旋转单元格媒体正例包含原 PNG 解码像素和故意渲染失败的原文字回退。样本复用，不累加为独立文档数量。保留负词界偏移相对匹配起点的原行为、bool/int/float 集合相等与大整数精确比较，不把词界匹配改成简化分词。

早期媒体夹具使用单元格测试的稀疏行/Source，全文参考构造补出 baseline、signature_bboxes 而 Rust 明确阶段输入保持缺失，导致检查点形状不同。这些冻结文件与失败日志保留；canonical-source 版本在输入端按真实原生视觉行和 Source 合同补齐字段，未修改 Python 算法或旧输出。零媒体的原数字网格样本继续保留为反例，没有算作图片正例。

开发只跑受影响测试，阶段末统一构建安装。最终安装计数、实时 PDFium 和默认门禁结果见主链最新记录；无全量/CUDA、全文精度或速度/内存提升声明。
