# 完整文字块组装接入

已完整迁移现有 Python `text_assembly/assembly.py::_build_text_blocks`，包括其调用的全部段界、脚注、空间/正文/数学补合并。不是逐行输出或精简段落算法。

| Python 职责 | Rust 对应 |
| --- | --- |
| `line_layout` 的正文/语义连接、净空/表格/横线屏障 | `native_text_assembly/connections.rs`、父模块 `State` |
| `rows` 的悬挂分组、局部栏对象身份、强调首行、文献新条目 | `grouping.rs` |
| 页顶/圆点/斜体引文/图注、结构/孤立缩进/视觉重置/公式显示行/首页字段段界 | `breaks.rs` |
| 自然段、中文正文与条目、行首字体重置、显式列表/元数据 | `prose.rs` |
| `footnotes` 全部分组脚注、机构/编号/上标、几何锚点和续行切分/排序/紧框 | `footnotes.rs`；复用原 `compact::footnote` 上标证明 |
| `_build_text_blocks` 脚注→按旋转/栏带分组→原连接优先级→块材料化 | `assembly.rs` |
| `common` 内部块元数据拼接、栏角色与走廊跳块判断 | `block_common.rs` |
| 短同基线前缀、空间组件、列表引导、未终止正文、重叠行块 | `block_merging.rs` |
| 宿主/无宿主/残余行内数学碎片、恢复段落闭合、公式上下文块 | `math_blocks.rs` |
| 连续最终页语句至完整文字块；原始 source/geometry 全文链 | `native_finalization::prepare_blocks_with_work`、`native_document::prepare_text_blocks` |

保留原调用顺序、参考组/段落组 0 的 Python 假值行为、重复来源的最后值与对象身份、同分稳定顺序、doc_title 换行、字体/视觉行集合与脚本 sidecar、各层 bbox 及紧框修复标志。数学分量继续使用原 CPython 整数 set 迭代；新增稀疏集合初始化与逐项 remove，不改变原 difference_update。布局适配只使用布尔/整数等价的数值投影，输出保留原行号和字体标志类型；缺省来源框沿用 `_LineItem.__post_init__`，不更改几何规则。内部块还未做最终输出归一化，保留 `_text_lines` 等后续必需证据。

SDK 新入口：

- `build_native_text_blocks(lines, page_size, ...)`：已准备行的完整文字块构建。
- `prepare_native_final_page_blocks(page, ...)`：已有 prepared page→最终公式/标题/行→完整文字块，增加 `text_blocks`。
- `prepare_native_final_document_blocks(pages)`：原始 source/geometry→全文文献→逐页完整文字块，增加 `block_pages`；`pages` 保留文献检查点。

源码验证：791 完整组装、5,537 次完整补合并回放、150 定向块边界（109 个产生变化）、24 浅快照/值相等删除合同（5 个原异常）、16 首次布尔/整数集合代表类型及 6 同几何键首项合同、626 行段界/分组、72 Unicode/CJK/身份补充合同（14 个 CJK 段落正例）、348 脚注（137 块）、204 连续页（含原零正文高度异常）、35 原始页源整链。测试有复用，不能相加当成独立文档。只跑受影响测试；共享集合的原 pop/difference 轨迹仍通过。安装验收状态见[主链记录](2026-10-27-form-runtime-integration.md)。

精确剩余位置已推进至 `_finalize_prepared_page` 中紧随文字块构建的 `merge_overlapping_member_blocks(...)`：后续成员/多行标题/图注/局部列表重建、视觉注释与 tight 输出框、完整排序/归一化/行内证据，然后默认 PDF 与用户整篇效果、耗时和峰值内存验收。完整 `_build_text_blocks`、最终行状态、公式接线不再待办。未进行全量或 CUDA；enhanced 阅读顺序后置，不宣称全文等价或性能提升。


## 成员、图注及局部列表的完整接续

`refine_blocks=True` 继续执行现有 Python 最终页从 `merge_overlapping_member_blocks` 至 `split_local_list_blocks` 的全部语句，额外返回 `refined_text_blocks`。默认 False 保留原 `text_blocks` 检查点；全文 `block_pages` 的原文献 `pages` 仍保留。

| Python 职责 | Rust 对应 |
| --- | --- |
| `continuity._reassemble_member_lines`、文献/首页成员组和重叠小块恢复 | `native_text_assembly/member_blocks.rs`；按真实行和水平顺序、重复来源最后值重建，复用完整布局分离反证 |
| 多行标题、页边碎片、首页作者列、重复弱标题续文 | `annotations.rs`，保留结构标题带/紧框/字体、稳定同分顺序 |
| 图像/数字网格图注、完整多轮续接、并排图像组、结构间隙与标题尾行反证；来源/附注分割 | `captions.rs` |
| 镜像图片环绕正文、前缀回收、重新调用完整文字块构建及永久几何屏障 | `caption_wrap.rs`，保留浅快照对象身份和原重新分段流程 |
| 完整连续区的递增小写字母及实心矢量圆点局部列表 | `local_lists.rs`；保留否定条件及原 list 占位块/子项元数据 |
| 上述完整调用顺序与共享 Work | `native_finalization::prepare_blocks_with_work`；止于 `absolute_blocks` 赋值前 |

源码对照为 642 成员/标题等规则、558 图注/环绕/来源/列表与完整后处理、10 Python 空白合同、211 连续页（含原零正文高度异常），及 35 原始 source/geometry 整链。新增原生矢量/字母列表和第二页正例，并有描边、矩形和偏离同排反例。语料有复用，不能累加为独立文档。旧规则导出器错用模块入口、后续 PreparedPage 漏传 bullet_paths 的冻结文件保留，另补 canonical/paths 参考；没有修改 Python 参考算法或过滤失败。后处理页参考执行原 `_finalize_prepared_page` 连续语句，而不是替代算法。

本段当前统一构建安装，结果见主链。精确剩余位置已推进至 `absolute_blocks = prepared.fixed_blocks + formula_blocks + index_blocks + text_blocks`：之后视觉注释恢复/绑定、聚合后紧框、完整排序/归一化/行内证据，最后默认 PDF 和整篇质量、耗时、峰值内存验收。上文“从成员块重建开始”为本文件早期检查点，现已完成。无全量/CUDA或质量/性能提升声明。


成员/图注/局部列表节点已构建安装通过：211 连续页、35 文档链、1,210 规则/合同和 5 页实时 PDFium。详见主链记录。后续 tight/归一化与文献/窄图形栏序内核正在独立验证，尚未跳过视觉注释接入主流程。


## 最终输出内核与栏序守护

`native_final_output.rs` 按原 `pipeline.py` 移植 `_apply_post_aggregation_tight_bboxes`、`_normalize_output_block`、`_normalize_output_line_items`、`_preserve_narrow_graphic_column_order`，以及 `continuity.order_body_above_reference_band`。1,116 输出/紧框/行框及 132 栏序守护参考通过（45 个排序变化正例）。保留候选字段弹出、旋转逆变换、千分位偶数舍入/最小一刻度、无效行整组为空、控制字符、图像角度、公式内容、文献/段界/父框等元数据合同。仅提供诊断 `native_final_output`，主链没有提前应用这些操作；完整视觉注释与分组排序仍按原位置待接。不能把独立输出内核完成记作完整最终 PDF 输出完成。


输出/栏序内核统一 release 构建安装通过：1,248 组对照与 118 模块来源/默认门禁通过；内核仍保持独立，完整视觉注释和分组排序未接通前不提升最终页/default PDF 完成标志。详见主链及 `build/native-final-output-*-install.txt`。
