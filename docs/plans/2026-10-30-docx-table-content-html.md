# DOCX 普通表格内容、图片与 colspan 校正

继续移植当前 Python 表格路径，未以简化纯文本算法替代 Mammoth。本阶段提供公式补入和最终链接校验之前的内部 HTML，尚未接入公开 RapidDoc Office 入口。

## 已实现

- `rust/rapiddoc-office/src/docx/table_html.rs`：Mammoth 1.12.0 的直接 run 格式、样式名匹配、标题、列表嵌套/折叠、链接/字段、书签、换行、内容控件、删除段落拼接、文本框额外内容、嵌套表格与表头。Mammoth 不继承 Word run 字体格式，故未用上一阶段 Word 样式继承替代它。
- `table_markup.rs`：原有空节点过滤、强制保留节点和相邻 HTML 路径折叠；折叠移动子节点，不复制整棵子树。保留原 HTML 转义和属性排序。
- 同文件移植本地 colspan 校正：众数及首次出现顺序、左侧优先缩减、rowspan 保护、嵌套遍历、异常回退保持原规则。直接处理现有树，避免 HTML 序列化后重新解析；列宽运算使用 BigInt。
- `table_images.rs`：按 part 关系延迟读取 ZIP 图片；位图/SVG 保留原始字节、MIME、替代文本和图片链接，直接生成原 data URI，无 PNG 编解码。包含 VML 图片路径。
- 保留行/单元格来源范围，不为每格重新扫描全文；合并掉的延续格仍先读取，以保留字段与删除段落状态。样式索引建立后释放临时样式 XML 树。
- Symbol 正文保留原异常分支，匹配签名仍容忍无效字符；支持原 Python 十六进制数字的符号、分隔符、Unicode 数字和大整数。

WMF/EMF（含 MIME 错标但载荷可识别）、外部图片、公式及脚注保留资源接口；未接入时明确失败，不丢弃内容后冒充成功。没有新增运行时 Python 解析代码；参考生成器只供开发使用。

## 验证

生成器 `scripts/migration/export_docx_table_html_reference.py` 直接调用原 Mammoth reader/转换器和本地 colspan 函数，未维护镜像裁判算法。持久参考为 `tests/fixtures/office_rust_reference_068/docx_table_html.json.gz`。

- 90 组内容案例：79 组 HTML 与原 Mammoth 字节一致，8 组检查原异常，3 组含公式且检查未接入资源时明确失败；不把这 3 组计为公式效果通过。
- 上述包含 12 组实际 ZIP 表格上下文，涉及主 part 关系和真实图片。缺少包上下文的图片案例只检查失败，不计为图片效果验收。
- 普通成功案例再比 colspan 校正后的 HTML DOM，忽略 BeautifulSoup 与原 writer 的空标签/引号序列化差异；另有 12 组众数并列、rowspan 短路、异常回退、负/零跨度、45 位跨度和嵌套表格边界。
- 本阶段 crate 的 17 组测试通过；后续只重跑受影响表格测试。原 559 段落、29 表格结构/签名、36 符号和 116 匹配参考保留。
- 格式、生成器语法检查通过；未跑 PDF 全量、CUDA或Office性能基准，未重新安装 wheel。

## 未完成

1. 含公式单元格的原 Word 富文本重建、OMML / Equation XML / MTEF、WMF/EMF 和图表。
2. 最终安全链接规范化、嵌入 Mammoth style-map、自定义映射与脚注完整上下文。
3. 正文输出、章节/分页、目录、页眉页脚、资产与最终 schema/Markdown，以及完整损坏包容错。
4. 完整格式贯通后接公开 Python API、更新原 `.venv`，做整文档效果、时间和峰值内存验收。

本轮未修改 Python PDF/Office 生产代码、模型或 `.venv` 安装包。来源及 BSD 许可记录在 `packaging/office-native-provenance.json` 与 crate 的 `licenses/MAMMOTH.txt`。历史阶段：[DOCX 上下文与表格结构](2026-10-30-docx-context-and-table-structure.md)。

后续更新：OMML、Equation XML、含公式单元格与最终链接校验见 [公式阶段记录](2026-10-30-office-omml-and-docx-formulas.md)。上面的未完成列表为本阶段历史记录，其他未覆盖能力仍待完成。
