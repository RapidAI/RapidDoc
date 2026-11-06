# Changelog

## Native SDK 0.1.1

- 新增 Linux ARM64（glibc）与 Windows ARM64 CPU 构建和跨绑定安装验证。
- Linux/macOS ORT 运行库去重，保留 provider、许可证及必要来源记录；Linux Node 平台包实测减少约 27%。
- Windows x64 npm 平台包改为 `@hzkitty/rapid-doc-win32-x64`，主包仍为 `rapid-doc`。
- 补充 Python PyPI 描述、npm 平台 README，以及 Node/Rust 的中英文文档。
- Java ORT 使用独立打包版本 `1.23.2-rapiddoc.1`，原生 ORT 仍为 1.23.2。
- 原 Python `rapid-doc` 保持 0.9.12；未修改的 vendored Rust crate 保持已有版本。

## 0.9.12

原 Python `rapid-doc` 发布版本。

- 同步 DocVortex 0.6.13～0.6.15 的 PDF 表格字体框、跨页表格边界和相关文字处理修复。
- 保留原 Python API、模型配置和 Office 解析入口。
- 完成发布前的 PDF、Office 和表格回归检查。

## 0.1.0

首个共享 Rust 引擎及 Python、Java、Go、Node.js 绑定版本。各渠道发布状态见 [README](README.md#选择解析引擎与语言-sdk)，部分渠道尚在完成首次发布。

- 同步 DocVortex 0.6.13～0.6.15 的相关 PDF 和跨页表格修复。
- Python PDF 表格增加 inflated glyph 墨迹校准，降低字体布局框过高导致的错误合行。
- 跨页 HTML 表格合并复用正文边界规则，保留行内样式、链接和原始 HTML 节点。
- Rust PDFium 对超出 OpenType 合法范围的字重值归一为未知，避免异常字重参与标题和粗体判定。
- Rust、Python、Java、Go、Node.js 使用统一的输入适配和 PDF/Office 解析核心。
- 通过发布前的 Python、Rust、Java、Go、Node.js 回归检查。
- 新增 macOS 14+ Intel x64 / Apple Silicon ARM64 CPU 包；四个平台通过安装、CPU ONNX 推理和七格式跨绑定一致性检查。

已知限制是 Rust 侧尚未完整读取嵌入字体的 OS/2 字重表。当前会拒绝异常字重估算，完整字体证据恢复留待后续版本。
