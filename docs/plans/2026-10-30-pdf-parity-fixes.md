# PDF 验收差异修复

## 原因和改动

本轮修复原验收留下的标点/空格、两个空文本 JSON 条目及 OCR 置信度差异。没有按文件名分支，没有替换输出文字、折叠空白、补造空条目或更换模型。

1. **正文检测默认值错误。** 原 Python `AtomModelSingleton` 创建正文 OCR 时默认 `det_db_box_thresh=0.3`，表格 OCR 显式使用 0.5；Rust 正文曾错误复用 0.5。现在由 `detection::Config::body()` 保留正文 0.3/扩框 1.8，表格保持 0.5/1.6。一个句号候选因此不再被提前排除；原 Python 将它识别为 `0`，本轮按原流程对齐，并非人为把标点改成数字。
2. **跨页检测分组顺序。** Python 对整个页面批次按首次出现的目标分辨率分组，然后逐页收集识别裁图。Rust 原本只按单页分组。现在共享 Rust 引擎在正文识别前恢复相同的全局分组顺序，保留组内框顺序及已有原生文字前缀，重建识别索引。没有识别结果的检测任务也参与分组排序。兼容封装将用户 `pdf_pages_batch` 原值传给核心，取消静默截为 16；默认仍与 Python 一样为 64。
3. **NumPy 置信度归约。** 逐字符概率先按原规则保留小数，再使用 NumPy FP64 的分组/递归求和顺序计算均值，避免舍入边界上的顺序求和差异。
4. **多边形重叠过滤。** 原 Python 在矩形重叠超过 0.7 时还会检查真实多边形覆盖率。Rust 漏了这一步，误删凹多边形表格旁的脚注区域。现在复用现有 GEOS 动态运行库执行原有 intersection/union、small-area 比率及无效多边形 buffer(0) 修复规则；没有关闭重叠过滤，也没有在导出层添加空条目。

修改位于共享 Rust 流程，其他语言绑定可以复用。Python 仅负责既有参数适配、模型配置和验证脚本。`RAPIDDOC_PARITY_TRACE` 只在显式设置时输出本地诊断数据；正常解析不保存额外裁图或模型 JSON。

## 已验证

- 受影响 3 份、29 页：`05.pdf`、`GBT3620.1-2016.pdf`、`small_ocr.pdf` 的 Markdown、内容 JSON、中间 JSON、图片名称及像素均与冻结 Python 基线严格一致。PNG 编码字节可以不同，不影响像素。
- 原始正文裁图逐项比对：05 的 537 张、small_ocr 的 229 张，像素和输入顺序全部一致。证据为 `build/pdf-parity-final-affected-20261030/crop-order-proof.json`。
- Rust 定向测试覆盖全局分组（含空识别任务）、NumPy 舍入边界、正文/表格不同默认阈值及原检测边界；安装包测试覆盖凹多边形过滤、默认配置及页面批次传递。
- 最终 wheel 已安装原 `.venv`，148 个文件逐项匹配；冻结清单的 17 个模型 SHA 未变。wheel：`build/pdf-parity-final-wheels-20261030/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`，SHA256 `b09bd6aff1ebd136c82195b690fef06561b81e7a7aa3c7cf448d87dbd4506c7c`。

阶段效果回归已结束：最终安装包重新解析全部 12 份去重文件、101 页，逐文件与冻结的原 Python 快照比较，Markdown、内容 JSON、中间 JSON、图片名称/像素全部严格一致。受影响 3 份与其余 9 份分别在同一实例中连续调用，覆盖模型复用；没有因图片编码字节不同而忽略其他字段。汇总为 `build/pdf-parity-final-summary-20261030.json`，详细结果分别位于 `build/pdf-parity-final-affected-20261030/`、`build/pdf-parity-final-regression-20261030/`。

安装核对：`build/pdf-parity-final-installed-receipt.json`。原 Python 生产源码未修改，模型未修改。6 项 Rust 定向测试、18 项安装包 Python 单测、8 组实际 PDFium 同进程共存及 `pip check` 通过；共存结果为 `build/pdf-parity-final-pdfium-20261030/summary.json`。本轮完整回归用于关闭已发现的效果差异，不代表穷尽所有文档或自定义模型/配置组合。不以诊断运行时间给出新的性能承诺，速度/峰值内存仍沿用原测量的限制，未运行 CUDA。

## 后续顺序

先闭合 PDF 效果回归，再按已有结构审查做适度领域归组、常规排版和契约整理，保留冻结基线逐步验证；随后集成 Office。不会在这一步重写解析算法、重划全部 crate，或仅按文件行数合并模块。
