# DocVortex 0.6.8：#36 与 PDF 选择性同步

后续状态：下文记录的第 54 页表头拆格已由 [表题合并修复](2026-10-30-caption-grid-merge-fix.md) 解决；原 328 页 Python 整文档回归已通过，旧 expected failure 标记已移除。全文 Rust 诊断入口另有内存问题，未计为整篇通过。

## 范围

按用户确认采用上游 #36 实现，并同步已审查的五项 PDF 行为。
基准上游为 `2de5f42b1884495313d9df63868e6a2f3d85fea0`，未整包覆盖，未升级模型或运行库。
Python 参考与共享 Rust 处理层同时修改，未把解析算法新增到 Python SDK 封装。

| 内容 | 实现与边界 |
| --- | --- |
| Office #36 | `errors.py`、DOC 图片、共享 OfficeArt、PPT 图片直接采用上游实现；保留本地回归，补充 DIB 加 BMP 文件头后的预算检查。单张图片拒绝和整文档错误仍分别处理 |
| `8a3ff6c` 损坏表格文字 | 输入与输出两阶段检查；至少两个异常且占比达到 5% 时拒绝原生候选，让原有后备识别继续处理。不是页数、字符数或公式容量限制，不新增整篇错误 |
| `72656be` 层级参数表 | 复用既有稀疏恢复器独立证明子表，再组合候选；所有后备候选也必须保留已证明的子单元格。空白/歧义和缺少物理边界的负例保持拒绝 |
| `2c6c0ad` 旋转表头 | 物理占格与阅读副本分开；保留方向、advance 和基线原点，用同向基线证据处理半周歧义。只有阅读和上下标分类使用正向副本，不移动原字符格位 |
| `d878cba` 上下标冲突 | 匹配来源保留至仲裁：完整行证据优于片段，重复路径不投票；同级冲突局部弃权；合法且有所有权的原生标签优先。诊断仅记录最终实际物化的区间 |
| `a4c8b2e` 私用区 | 活跃 Python/Rust span 路径已经支持三段 PUA，保持原文；对齐保留的 vendor `text_content` 内核及上游测试，不改成删除私用符号 |

保留本地横向文字 advance 间距、标点分行、占格、表格/图形仲裁及版面流程。
跨页表格 `ac4e28c` 与斜向水印 `72c1151` 仍单独审查；增强阅读顺序和 CUDA 仍按用户要求暂缓。

## 独立参考与回归

- 上游表格及行内冲突测试通过 **146 项**；包导入路径适配为裁剪源，渲染断言使用 RapidDoc 自己的实际渲染器。
- Python Office 与 native PDF 受影响回归 **589 项通过**，未包含下面的已知整文档失败。#36 的 **24 项**属于此集合，不重复相加。
- 从上游测试捕获真实输入，在 pytest 恢复 monkeypatch 后重新调用更新后的 Python 计算参考。**64 组表格、21 组行内样式、47 组表格 HTML，共 132 组**在 Rust 单元入口和安装后的 Python SDK 均通过；核对完整候选、诊断、源字符归属、HTML、Span 与实际物化诊断，不只比较纯文字。
- 参考压缩持久化在 `tests/fixtures/upstream_068/pdf_reference.json.gz`，生成器为 `scripts/migration/export_upstream_pdf_reference.py`，不依赖已经清理的历史 `build/migration` 夹具；现有旧基线未覆写。
- 私用区 vendor 内核 **13 项单元测试通过**，包括三段私用区计数与原文保留。
- #36 更新后的 **20 份 Office 文档**与同步前完整基线一致，涵盖六种格式及图片字节。
- 安装后 PDFium 共存 **8 组通过**，含导入顺序、外部库、并发与 Python 关闭场景。

## 保留的已知失败

引入的上游 328 页目录整文档测试未通过：第 54 页宿主候选把表题也纳入表框，
闭合网格因无法认领表题而被拒绝，随后稀疏候选把纵向 `Day 1` 拆到两个单元格。
升级前源码重放同一页也失败：旧结果 `1y / Da`，新阅读方向恢复后为 `y 1 / Da`。
这不是本轮新增的文字丢弃，也不能算作已修复的整页效果。

原上游断言保留，以 strict xfail 标记；不降低文字覆盖阈值、不清空格子，也不使用文件名或页号修改生产算法。
后续需处理宿主表题与物理表框的候选边界，不能仅改旋转字形排序掩盖这个问题。
记录：`build/pdf-upstream-catalog.log`、`pdf-upstream-catalog54-before.json`、
`pdf-upstream-catalog54.json`、`pdf-upstream-catalog54-diagnostic.json`。
实际原始字符/线/矩形、升级前文字及期望标签另持久保存在
`tests/fixtures/upstream_068/catalog54_caption_grid.json.gz`，清理构建目录后仍可重放。

## 安装与整文档验收

已完成：12 份、101 页的更新后 Python 与原基线一致；安装后 Rust 与更新后 Python
及升级前 Rust 均一致，Rust 图片编码字节也未变化。下列已知宿主失败单独保留，不计入通过。

最终 wheel：`build/pdf-upstream-wheels-final-20261030/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`。
已安装到项目原 `.venv`，148 个包文件与 wheel 一致，绑定适配器源码未变，`pip check` 通过。
最后仅重打包来源审计信息，原生二进制与通过 132 项安装验证的版本相同。

使用 `scripts/migration/verify_pdf_upstream_delivery.py` 对登记的 12 份 PDF 串行验收：
先比较更新后 Python 与原结构整理基线，再比较安装后 Rust 与更新后 Python。
完整结果及交付状态以 `build/pdf-upstream-final-receipt-20261030.json` 为准；
输出快照分别保存到 `build/pdf-upstream-python-20261030` 和 `build/pdf-upstream-rust-20261030`。
比较 Markdown、内容列表、中间结构和图片像素/名称；Python/Rust PNG 编码字节差异单独记录。
本轮不重新承诺速度、内存优势，也未运行全工作区测试或 CUDA。

来源清单：Python `PDF_UPSTREAM_068.json` / `OFFICE_UPSTREAM_068.json`、
Rust vendor `UPSTREAM.json`、`tests/native_pdf/upstream_068_tests.json`。


## 上游本身的独立核实

使用本机 `D:/CodeProjects/doc/DocVortex` 的 v0.6.8 源码（`2de5f42`），
临时构建与源码匹配的协议 42 原生扩展后，公开 `parse(..., page_range="54", parse_mode="txt")`
输出完整的 `Day 1` 和 `Day 2`。上游原始 328 页回归
`test_full_catalog_rotated_row_headers_keep_reading_order` 也通过（1 passed，88.69 秒）。
因此此处是 RapidDoc 集成链路与最新上游的行为差异；“升级前已经存在”只指 RapidDoc，
不能理解为上游本身也有同样缺陷。

本机上游目录已有的 `.pyd` 仍是协议 34，与源码要求的协议 42 不匹配。
本次使用隔离构建的扩展，未替换上游源码目录或现有安装。
证据：`build/docvortex-upstream-catalog54-probe.json`、
`build/docvortex-upstream-catalog-full-test.log`；重放脚本为 `build/probe_docvortex_catalog54.py`。
