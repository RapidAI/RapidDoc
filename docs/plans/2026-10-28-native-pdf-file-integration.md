# 原生 PDF 文件与 PdfModel 入口

本文件级节点统一 release 构建 8 分 41 秒，已安装至 `build/rust-wheel-env` 并验收通过。4 份/20 页完整采集及 PdfModel 输出、12 个可见签名，76 个实际可见页面、108 字形和 418 末尾清洗场景通过；两个用户示例另做完整原 `PdfModel.predict` 对照：论文模板 14 页、财报 9 页，文字、类型、几何、顺序、行内内容和全部 13 个表格一致。两个示例均无表内 PNG，不把它们计为媒体图片正例；论文中的 3 个 image 是原生版面块。后续外层 Markdown/图片资产已在[宿主节点](2026-10-28-native-pdf-host-integration.md)完成安装验收，共 23 页/19 资产，文件级接口不再是当前最外层入口。

受影响安装回归通过：58 个完整字符场景/7,650 原始记录、290 源行规则及实时视觉行、既有上下标/分数完整套件、118 模块来源与默认门禁。新增线程预算恢复、5 个实际整页分数和 3 个大候选合并集合源码测试通过。早期 debug 财报失败为调整前预算，已由最终 release 安装包整篇对照验证修复；失败日志保留。未跑全量/CUDA，未做速度/峰值内存结论。

日志：`build/native-pdf-file-wheel-build.txt`、`build/native-pdf-file-install-test.txt` 和 `build/native-pdf-file-*-install.txt`。两个实际安装输出保存在 `build/native-pdf-model-preview/示例1-论文模板.json`、`比亚迪财报.json`，可直接审阅。下一步是 RapidDoc 模型路由/适配器及 Markdown/资产的整体验收；原生文件采集、全文及 PdfModel 末尾不再待办。

本节点把真实 PDF 文件接到已完成的全文准备、最终页、行内语义、表格媒体和 PdfModel 末尾文字清洗。新增 `analyze_native_pdf(source, pdfium_library=...)` 返回 `pages`、`source_page_count`、`page_sizes`；`pages` 对应原 `PdfModel.predict` 的逐页 `model_list`。`collect_native_pdf_sources` 提供原始采集检查点。两者直接调用 Rust/PDFium，不调用 Python 解析器，不执行 OCR。

| Python 原逻辑 | Rust 实现 |
| --- | --- |
| `_collect_document_sources`：字符/视觉行、Form/绘图/图像/签名、空格/样式/链接，以及跨页图像和噪声裁决 | `legacy_model/native_pdf_document.rs`，复用现有完整阶段 |
| 可见文字模式、透明度、继承裁剪与字符裁剪、后绘制安全实心矩形遮挡 | `rapiddoc-pdf/character_primitives/visibility.rs`、`visibility/resources.rs`，复用 vendored PDFium 对象内核 |
| Widget 签名类型继承、隐藏标志、正常 AP、四旋转/页面裁剪 | `rapiddoc-pdf/signatures.rs` |
| `inline/glyph_weight.py` 条件渲染、同字体/同字符样本、灰度/最近邻/墨迹与宽高比证明 | `legacy_model/native_glyph_weight.rs` |
| `_analyze_native_document`、`PdfModel.predict` | `native_pdf_document::analyze/predict`，按原顺序接完整全文、表格媒体和最终清洗 |
| `content/normalization.py`：可见全角英数符号，公式保护、Span 原结构和 HTML 可见文本 | `legacy_model/native_model_normalization.rs`，复用现有兼容 HTML DOM |

`include_local_lines` 仅供文件采集在原位置取得装饰行处理前的空格证据；缺省关闭，不改变已有源行诊断输出。可见字符接口和完整字符诊断接口分开，后者仍保留隐藏字符。稀疏可见字符的代理对恢复读取原始相邻 Unicode，不假定过滤后的数组下标连续。

全文文件使用按页数增长且有上限的线程局部资源预算：工作每页 5,000 万且最多 10 亿、节点 800 万、字符串 64 MB、输出 128 MB；退出、错误及 unwind 均恢复原诊断限制，其他线程独立。单页字符、图形、输入文件及渲染像素的原有上限保留。

财报整篇验收暴露了早期分数字符证据把“整页”误套入 1,500 字符单行诊断上限。实际 Python 路径在五页分别传入 1,563、2,087、1,796、1,872、2,135 个字符。新增 `fraction_members_page` 明确使用上层原有 20,000 字符整页合同，逐字符几何/规则内核和 1,000 万交叉工作限制不变；原 `fraction_members` 诊断仍限制 1,500。五个实际请求另冻结为 `native-page-fractions.json`，输出均与 Python 对照，旧诊断拒绝行为保留。

独立进程使用方式：

```python
from rapid_doc_rust import analyze_native_pdf

result = analyze_native_pdf("demo/pdfs/示例1-论文模板.pdf")
model_list = result["pages"]
```

此接口返回原生 PdfModel 输出；现在可由 `parse_pdf_models(..., docvortex_mode="rule", native_only=True)` 继续原宿主 OCR、Markdown 和资产链，见[宿主接入](2026-10-28-native-pdf-host-integration.md)。SDK 自带运行库缺失时需显式指定 `pdfium_library`；不要在同一进程导入 pypdfium2。

财报另有 187/339/229 个待合并候选，分别累计 23,375/41,019/23,444 个成员引用，合并后为 1/1/2 个候选。另冻结 `native-pdf-candidate-capacity-canonical-members.json`，三个实际集合均与原合并函数一致。文件级成员总量每页 20 万、最多 100 万，原诊断 20,000 不变。原导出未处理草稿/共享成员集合的失败日志及不完整参考保留，不作有效证据。

原财报整篇在已计费 89,581,230、再申请 4,898,450 时触发最初 9,000 万全文工作预算；文件级预算现在按完整多阶段累积量而非单个诊断阶段计费，所有阶段继续共享 Work。后续整篇验收在统一 release 安装包运行，避免反复执行慢速 debug 整篇。

冻结证据：`native-visible-text.json` 的 76 个实际 PDF 页面（包含 20 个全排除页面）、`native-glyph-weight.json` 的 108 个字形规则、`native-model-normalization.json` 的 418 个末尾清洗场景、`native-pdf-document.json` 的 4 份/20 页文件链与 12 个可见签名框。`native-pdf-acceptance.json` 单独冻结两个用户示例的原 `PdfModel.predict`，不覆盖任何既有参考。PNG 以解码后的尺寸、模式和像素核对；不要求压缩字节相同。

源码、安装和两个示例的最终结果见本文顶部。`parse_pdf_models` 的原生宿主模式/适配器已接通，最终安装结果见[宿主节点](2026-10-28-native-pdf-host-integration.md)；两个示例通过也不等于全部 PDF 精度、速度或峰值内存均更优，兼容性审计的普遍声明继续 False。增强阅读顺序和 CUDA 实测继续按用户要求后置；开发不跑全量回归。
