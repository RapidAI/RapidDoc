# DOCX段落上下文与表格结构移植记录

状态：共享底座之后，已实现DOCX语义上下文及表格结构；尚未实现完整DOCX转Markdown，不接入默认RapidDoc入口。

## 本轮实现

`rust/rapiddoc-office/src/docx/`：

- `Source`从实际ZIP关系定位主文档、样式、编号。XML有单一所有者，树视图借用缓冲区；部件不按段落重复解压。缺少样式部件时使用与Python python-docx1.2.0一致的默认样式，保留MIT来源声明。
- `Styles`建立样式ID索引，按原规则解析默认样式、basedOn继承、直接run格式、字符样式和段落样式。显式false终止继承；Hyperlink字符样式的特殊处理不变。继承循环有访问记录，默认样式/字体属性结果在当前文档缓存，未使用的错误属性不提前解析。
- 标题ID、名称、冒号等级、outlineLvl及TOC层级按原Python规则处理。Unicode十进制数字使用固定Unicode16数据，避免Rust regex与Python Unicode版本差异改变国际化标题层级。
- `Numbering`解析abstractNum、num、pStyle反查层级、startOverride及原有编号格式集合；直接空numPr不与样式numPr混合，numId=0取消编号。计数推进重置同numId的更深层级，切换列表保持各自状态。起点和计数用BigInt，不新增64位累计编号上限；40位起点及连续推进已与Python参考对照。
- `fields`按原顺序展开run、超链接、内容控件及透明容器，解析分段、嵌套和未闭合复杂字段；保留字段结果、原有内部TOC链接抑制、链接组边界空格、空白样式与可见空白处理。普通隐藏run与链接内隐藏run的既有差异保留，不擅自修正Python行为。
- 表格`layout`移植Mammoth的横纵合并处理、孤立延续格、删除行、表头和异常结构回退。记录单元格原XML字节范围及未放置对象，供后续完整内容转换消费，不分配稠密字符/单元格网格。
- 表格`signature`移植项目当前匹配规则：特殊字符/Symbol、纵合并延续格内容排除、MC Fallback、简单域、复选内容控件首个文本排除；保留嵌套表格/图片计数及前缀匹配规则。
- 正常部件关系读取已支持ZIP根目录中的主文档及相对样式路径；独立OPC normalizer路径函数的历史语义和原参考不变。

本轮没有改生产Python代码、模型、PDF规则或 `.venv` 安装包。新增Python脚本只生成测试参考，不参与Rust运行时解析。

## 验证

`cargo test --locked -p rapiddoc-office`：14组测试通过，包含此前共享底座测试和本轮新增测试。

1. 当前Python DocxConverter生成5份文档的559个段落参考，包括预期异常案例；对比原始文字、格式、继承标题/编号/目录属性、字段指令和分组链接结果。4个实际ZIP输入另行通过Source加载，覆盖Strict与非标准主part。
2. 编号起点覆盖、父级前进/子级重置、列表切换、缺失定义回退及大整数计数对照。诊断参考中起点/计数用十进制字符串保存，避免测试JSON读写截断；不是更改RapidDoc公开返回类型。
3. 29个表格结构及签名、36组符号、116组匹配参考。结构参考调用安装版本Mammoth的原始reader与合并算法；输入仅清空单元格富内容以隔离跨度算法，没有写一份Python镜像算法作为裁判。
4. 缺省下划线值、非法布尔/下划线、未知/缺失vertical alignment、循环样式、内容控件、孤立字段、嵌套字段与未闭合字段等保留原返回/异常分支。
5. 格式检查、Unicode生成数据检查及参考生成脚本语法检查通过。只测受影响crate，没有PDF全量、CUDA或整体性能回归。

持久参考：

- `tests/fixtures/office_rust_reference_068/docx_context.json.gz`
- `tests/fixtures/office_rust_reference_068/docx_tables.json.gz`
- 生成器：`scripts/migration/export_docx_context_reference.py`、`export_docx_table_reference.py`。

Word上下文字段和Mammoth表格字符处理存在真实差异，例如不间断连字符和简单域；分别移植和验证，不能用一个通用纯文本提取函数替代两条路径。

## 当前未完成与下一步

- 单元格完整HTML、富文本/列表、嵌套表格最终写出及colspan规范化。
- 图片、图表和嵌入对象的资产写出；OMML、Equation XML、MTEF与图片公式解析。
- 正文章节/分页、标题列表/目录组合、文本框、页眉页脚和最终schema/Markdown输出。
- 完整包损坏恢复与缺失内容类型修复、公开Python API安装及整文档效果/速度/峰值内存验收。

当前段落/表格结构差分通过，不代表完整DOCX解析效果与Python已经相同。新接口只提供Rust内部语义上下文；Python `rapid_doc_rust.RapidDoc` 的Office入口仍待完整转换链接入。

后续继续接完整表格内容和公式/图片，保持当前Python输出作为基线，不以省略复杂内容换取完成进度。

后续更新：普通单元格内容、图片和colspan校正的最新实现见 [表格内容实施记录](2026-10-30-docx-table-content-html.md)。以上未完成列表保留为本阶段历史记录。
