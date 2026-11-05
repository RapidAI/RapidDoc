# Office OMML 与 DOCX 含公式表格

本阶段继续移植当前 Python 的公式规则，新增原生 OMML、Equation XML、公式载体选择和含公式单元格重建。仍不是完整 DOCX 转换器，未更新原 `.venv`。

## 实现范围

- `rust/rapiddoc-office/src/equation/omml.rs`：按本地 `omml.py` 处理分式、根式、重音、定界符、上下标、函数、矩阵、数组/公式标签、上下限、组合符号、数学字体及原异常分支。保留现有 Python 不支持节点的处理规则，不宣称支持完整 OMML 标准。保留 raw `.latex` 与 `str()` 的双空格折叠区别。
- `omml_data.json`：冻结当前 `latex_dict.py` 和 pylatexenc 2.10 的逐字符行为。覆盖默认字典键、所有 NFC 变化的 Unicode 标量和本地覆盖规则，共 2,654 个非恒等映射、约 50KB；其他字符按原规则保留。生成器检查 Unicode16、pylatexenc版本及默认规则类型，防止新规则被遗漏。运行时无 Python 或 pylatexenc 调用，后续 Excel/PPT 可复用。
- `equation/equation_xml.rs`：按原 Word 2003 XML 结构、唯一段落/公式检查、禁用注释/DTD等规则解码 VML equationxml；保留 SHA256 缓存与去重计费。这里的单项/累计 128MiB 来自现有 Python Office 资产限制，不是额外 PDF 字符/公式预算。
- `docx/formulas.rs`：保留 OMML、Equation XML、OLE MTEF、图片 MTEF 的载体优先级、AlternateContent/object 分支、文本框排除、文字/换行顺序及错误 OMML 回退。MTEF 分支仍留待接入，不能把未实现当作没有公式。
- 含公式单元格按原 Word 格式重建，普通表格仍按 Mammoth 格式；保留段落、样式继承、超链接和行列数不匹配时的原注入保护。内部流程已贯通：Mammoth树生成/折叠 → 公式注入 → colspan校正 → 最终链接校验，未增加中间 JSON 或 HTML 重解析。
- `links.rs`：移植 Office 链接校验，包括允许协议、相对/fragment、控制字符、IPv6/IPvFuture及 Unicode NFKC authority 检查。非法链接仅移除链接包装，保留富文本。保留现有 BeautifulSoup 对表格 ASCII 纯空白的规范化，避免 tab/空格差异。

## 明确的修复差异

原 Python 用 `<eq>{latex}</eq>` 直接拼接 HTML，公式 `a<b` 会被 BeautifulSoup 当作标签，破坏公式内容。Rust 在 HTML 输出边界转义公式文字，保留 `<`、`&` 等原值。OMML/LaTeX 本身没有改写，也没有降低识别精度。

参考生成器同时保留原始结果与仅增加公式边界转义的结果；其余仍调用原 Python 的提取、样式、注入和规范化代码。测试分别比较未改动的公式 token 和安全输出 DOM，不把这项差异藏入新基线。原生产 Python 尚未应用这项修复。

## 验证

- 192 组 OMML：含 46 个真实公式、24 个预期异常；比较 raw LaTeX 和最终字符串。
- 2,654 个非恒等字符映射及未知字符保留对照。
- 45 组 token/段落/表格案例，含一个真实 DOCX 表格上下文；覆盖分支优先级、混合格式/链接、隐藏 run、内容控件、错误公式和嵌套/合并注入保护。
- 13 组 Equation XML：结构错误、注释/DTD、重复公式、编码声明及重复解码的原计费行为。
- 49 组 Office 链接校验，包含 Unicode NFKC 禁用字符。
- `cargo test --locked -p rapiddoc-office`：22组通过。格式、数据重生成检查和生成器语法检查通过；未跑 PDF 全量、CUDA或完整 Office 性能基准。

持久参考：`tests/fixtures/office_rust_reference_068/omml.json.gz`、`docx_formulas.json.gz`。生成器在 `scripts/migration/` 下：`generate_office_omml_data.py`、`export_office_omml_reference.py`、`export_docx_formula_reference.py`。来源/SHA与 MIT/BSD 声明见 `packaging/office-native-provenance.json` 和 crate 的许可证目录。

## 后续

继续接 OLE/图片 MTEF、WMF/EMF、图表及完整文档资源；补齐嵌入 style-map、脚注、正文分页/目录/页眉页脚、最终 schema/Markdown和包损坏容错，再接公开 Python Office API及整文档效果/速度/峰值内存验收。当前资源接口对未接入分支明确失败；没有以省略这些内容换取“DOCX完成”。

后续更新：OLE/图片MTEF与DOCX资源接入见 [MTEF阶段记录](2026-10-30-office-mtef-and-docx-embedded.md)。本页后续列表保留为当时状态，普通矢量渲染和完整文档入口仍未完成。
