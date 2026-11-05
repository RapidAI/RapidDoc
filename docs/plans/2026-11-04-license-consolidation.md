# 发布许可证文件精简

本次只整理分发声明，不改 Rust/Python 解析代码、模型、运行库和字体。

## 结果

- 最新 Windows wheel 的三个 `licenses/` 子树：300 → **7 个文件**。
- SDK 主目录：265 → 2；PDFium：31 → 3；ONNX 算子：4 → 2。
- 每个子树使用 `THIRD_PARTY_NOTICES.txt` 集中展示声明，`third-party-notices.json` 保存原组件路径、原文 SHA-256 和字节位置。同样文字只出现一次，不同版权原文不合并。
- PDFium 的 FreeType 声明使用非 UTF-8 编码，原文件单独保留，不猜测编码或改写原文。
- Rust Office 源码目录的生成副本：237 → **21 个文件**。218 个依赖声明副本合为两份文件，19 个直接来源声明仍留在对应 crate 中。
- 其他原始 vendor 许可证、字体声明、GEOS 的 LGPL 声明/源码归档保持原样；上述 7 个文件不是整个仓库所有许可证的总数，也不意味着所有许可义务只有七项。

这里减少的是文件及重复正文，未将无法确认的版权声明、NOTICE 或构建依赖声明直接抹掉。MIT 要求保留版权及许可文本，Apache-2.0 要求提供许可副本并保留相关归属/NOTICE；集中展示保留完整可读文本及组件对应关系。参考：

- https://opensource.org/license/mit
- https://www.apache.org/licenses/LICENSE-2.0

## 可持续打包

`scripts/migration/license_bundle.py` 提供按内容去重、逐字节校验、旧路径读取和重复执行检查。`compact_release_notices.py` 可整理暂存目录，或直接重打已有 wheel，不需要重新编译 Rust。

Office 构建、PDFium 暂存、Linux release wheel 和依赖声明收集脚本均接入合并流程。原来源清单中的依赖路径通过 `notice_archive` 指向集中索引；`read_notice()` 可验证原路径对应的完整字节。迁移声明同步及兼容性/许可检查工具已适配。

已模拟下一次 Office 暂存步骤，确认不会再次留下数百个文件。

## 验证与交付

- 10 项测试通过：去重、不同版权保护、损坏索引、非 UTF-8 文件、嵌套合并、重复执行、wheel 文件不变，以及 DOCX 正文/表格图片公开调用。
- 新 wheel 与原 wheel 中全部 Python 文件、Pyd、DLL 和 SO 逐字节一致；变化仅限许可证子树、许可证哈希清单及 wheel RECORD。
- RECORD 全量校验通过，PDFium 清单中的许可证哈希通过，迁移声明原始 120 条路径校验通过；安装后的二进制也与 wheel 一致。
- 新 wheel 已安装项目 `.venv`：`build/compact-notices-wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`。
- 收据：`build/license-consolidation-20261104.json`。未重编译 Rust，未跑全量解析/CUDA，未同步服务器。

额外执行现有 `verify_lopdf_vendor.py` 时，其源码审计在 `src/bookmarks.rs` 与历史 UPSTREAM 哈希不一致处停止。这不是本次许可证整理改动；未修改 vendor 源码或覆盖历史记录。该组件的许可证已单独核对，合并前后原文字节一致。后续发布审计需单独核对这处既有来源差异。
