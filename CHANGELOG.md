# Changelog

## 0.9.12

原 Python `rapid-doc` 发布版本。

- 同步 DocVortex 0.6.13～0.6.15 的 PDF 表格字体框、跨页表格边界和相关文字处理修复。
- 保留原 Python API、模型配置和 Office 解析入口。
- 完成发布前的 PDF、Office 和表格回归检查。

## 0.1.0

首个 Rust 引擎及 Python、Java、Go、Node.js 绑定发布候选版本。

- 同步 DocVortex 0.6.13～0.6.15 的相关 PDF 和跨页表格修复。
- Python PDF 表格增加 inflated glyph 墨迹校准，降低字体布局框过高导致的错误合行。
- 跨页 HTML 表格合并复用正文边界规则，保留行内样式、链接和原始 HTML 节点。
- Rust PDFium 对超出 OpenType 合法范围的字重值归一为未知，避免异常字重参与标题和粗体判定。
- Rust、Python、Java、Go、Node.js 使用统一的输入适配和 PDF/Office 解析核心。
- 通过发布前的 Python、Rust、Java、Go、Node.js 回归检查。

已知限制是 Rust 侧尚未完整读取嵌入字体的 OS/2 字重表。当前会拒绝异常字重估算，完整字体证据恢复留待后续版本。
