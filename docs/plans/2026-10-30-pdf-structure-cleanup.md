# PDF Structure Cleanup Implementation Plan

**Goal:** 整理已验收的 PDF 实现，保持解析、公开 API、数值与模型行为不变。

**Architecture:** 保留现有 crate。`rapiddoc::processing` 为规则处理的规范入口，按文档、版面、文字、表格、图形、公式、OCR 归组；`page_assembly` 为页面组装，旧 `legacy_model`/`legacy_page` 路径保留兼容重导出。推理层按领域存放实现，保持既有模块导出与所有权边界。Python 绑定按职责分文件，根文件保留统一注册和异常/序列化辅助。

**Tech Stack:** Rust、Cargo/rustfmt、PyO3/maturin、现有 Python 对照脚本。

## 执行与验证

1. 冻结整理前的本项目 Rust 源码、模块映射、已安装 Python 导出/签名及完整 PDF 对照结果；不复制模型或构建缓存。
2. 将 `rust/rapiddoc/src/legacy_model.rs` 和子模块归入 `processing/`，按领域物理归组；将 `legacy_page` 归为 `page_assembly`，原 Rust 路径保留。移动文件与伴随 JSON 资源时校验绝对路径位于项目内。
3. 将 `rust/rapiddoc-inference/src` 的模型前后处理归入 OCR/layout/formula/table/image/runtime 目录；不改模块可见性和会话实现。
4. 将 `bindings/python/src/lib.rs` 的 PDF 事实、资产、表格、公式、OCR、模型会话和诊断适配分文件，Python 函数/类型名称、参数、注册顺序保持不变。
5. 对自有 crate 和绑定源码运行常规 rustfmt，排除 vendor/实验代码；核对移动前后的实现、常量、字符串及数据资源，不改变算法。
6. 执行编译、受影响的资源/检测/分组/置信度和安装 API 测试；阶段末 release 构建并安装原 `.venv`，对全部 12 份、101 页冻结基线验证 Markdown、两个 JSON 和图片名称/像素；再验证 PDFium 共存及安装文件/模型 SHA。
7. 更新包职责和查找说明，记录验证结果及限制。此轮不接入 Office，不重新设计通用文档类型，不转换整条 JSON 契约，不承诺性能收益，不运行 CUDA。

## 实施结果

- 256 个原路径归组或移动，原模块的每份实现和规则数据只有一份。`processing/` 的七个领域包含 document 28、formula 14、graphics 16、layout 21、ocr 4、table 58、text 53 个 Rust 文件；原公开模块名和旧路径均保留兼容。
- `legacy_page` 的规范入口改为 `page_assembly`。涉及 `pub(in ...)` 的可见性指向同一实际祖先模块；没有扩大对外公共 API，也没有复制算法以保留旧路径。
- 含子模块的组件采用标准 `mod.rs` 入口；Unicode、字体折叠、表格 HTML 等相邻 JSON 资源随实现移动，编译检查覆盖路径解析。
- 推理实现按 runtime/image/OCR/layout/formula/table/seal 归组，公开模块、会话和原生内核保持兼容。
- Python 绑定拆为七个领域适配文件。根 `lib.rs` 从 2,570 行减为 173 行；这只是迁移实现和恢复排版，没有删除功能。Python `_native` 的 94 个导出以及 SDK 的导出/签名与整理前一致。
- 454 个自有 Rust 源文件通过 rustfmt 检查，未格式化 vendor。迁移生成器和兼容审计的源码路径已更新，560 处完整文件字面量指向有效文件；119 个模块的兼容清单已刷新并通过 `--check`。
- 包职责、领域查找入口和 Office 事实/结果边界说明位于 `rust/README.md`。正式 JSON 契约、FP32/FP64 往返和已有规则不变，Office 尚未接入。

## 验证与交付

整理前源码/API 快照为 `build/pdf-structure-before-20261030/`，移动清单为 `build/pdf-structure-moves-20261030.json`。`audit_pdf_structure.py` 对 446 个 Rust 文件以及拆分前的 104 个绑定项逐项核对，实现、常量和字符串均一致；20 份 JSON 资源逐字节一致。核对允许模块声明/等价祖先路径、资源引用的同内容重定位，以及 rustfmt 的空白、闭包排版和可选尾逗号变化，不替换输出文字或数值。核对结果为 `build/pdf-structure-source-audit-20261030.json`。

`cargo check -p rapiddoc-python`、13 项 Rust 定向测试（检测/资源/CTC/分组/表格）、18 项安装包 Python 单测、`pip check` 通过。最终 wheel 已安装原 `.venv`：`build/pdf-structure-wheels-final-20261030/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`，SHA256 `5c8e63b9e7ba3010b2a2062677f7ebb8aae96f0804c90a06a1c36a86ac1e0eb8`。148 个安装文件匹配，17 个模型 SHA 未变，Python API 核对通过；证据为 `build/pdf-structure-installed-receipt-20261030.json`。

全 12 份去重 PDF、101 页效果回归通过：Markdown、内容 JSON、中间 JSON、图片名称/像素与冻结的原 Python 基线严格一致；同时与整理前 Rust 快照比较，这些内容以及 PNG 编码字节也全部一致。详细结果位于 `build/pdf-structure-acceptance-20261030/`，汇总为 `build/pdf-structure-summary-20261030.json`。同一实例顺序解析全部 12 份，覆盖模型复用。性能和峰值内存未做新的基准测试，未运行 CUDA。错误诊断中源码路径/行号随文件移动和排版变化，不把它们作为稳定 API；错误码与异常传播规则不变。

8 组实际 PDFium 同进程共存验证全部通过，含导入顺序、显式库路径、环境变量、并发及 Python 销毁重建。结果为 `build/pdf-structure-pdfium-20261030/summary.json`；整个阶段的交付汇总为 `build/pdf-structure-final-receipt-20261030.json`。本次整理完成，后续可开始 Office 集成。
