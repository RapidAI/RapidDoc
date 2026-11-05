# 最终页面行状态接续

本节点完整执行现有 Python `pipeline._finalize_prepared_page` 从首句到 `_build_text_blocks` 之前的连续语句。先复用已完成的公式检查点，再执行最终标题/语义收尾、同排合并、文献再次分组及首页机构再次分组。没有替换为逐行输出或简化段落算法。

Python 参考目录为 `rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/`。目录不重构；新增代码放回对应职责，以下表帮助定位：

| Python 文件/函数 | Rust 文件/函数 |
| --- | --- |
| `layout_evidence.py` 的构建、corridor、separated | `native_layout.rs` |
| `structural.py` 显式编号/字体重置/正文同字号/非首页题名 | `native_titles/page_structural.rs` |
| `native_boundaries.suppress_continuous_sentence_titles` | `native_titles/final_rules.rs::suppress` |
| `line_merging._demote_runin_title_fragments` | `native_titles/final_rules.rs::runin` |
| `structural._demote_regular_repeated_item_titles` | `native_titles/final_rules.rs::repeated` |
| `panels.classify_panel_titles` | `native_titles/final_rules.rs::panels` |
| `structural._classify_bold_numbered_heading_rows` | `native_precontainers/headings.rs::classify_mode`，复用现有完整编号/极端修复实现；新增 `final_rules.rs::roman` |
| `structural._restore_wrapped_bold_title_tails` | `native_titles/final_rules.rs::wrapped`，复用 `native_page_preparation/compact.rs::freeze` |
| `structural._restore_short_heading_with_image_displaced_prose` | `native_titles/final_rules.rs::displaced` |
| `structural._classify_short_cjk_section_leads` | `native_titles/final_rules.rs::short_cjk` |
| `continuity.group_front_matter_lines` | `native_titles/final_rules.rs::front` |
| `line_merging` 的指标标题/标题已定视觉行/编号碎片/后语义合并 | `native_page_preparation/semantic.rs` |
| `_merge_dense_split_visual_row` / `_merge_same_baseline_group` | 复用 `native_page_preparation/baseline.rs::materialize_mode`，保留各自空白/状态规则 |
| 最终页面连续顺序及文献再次分组 | `native_finalization.rs::prepare_lines_with_work`，复用 `native_references` |
| 原始页源/几何→全文文献→最终行状态 | `native_document.rs::prepare_final_text` |

SDK：`prepare_native_final_page_lines(page, page_index=0, body_profile=None, canonical_body_profile=None, title_profile=None)` 消费已准备页；`prepare_native_final_document_lines(pages)` 消费已有 source/geometry 文档输入，额外返回逐页 `text_pages`。原 `pages` 仍保存文献阶段状态，避免把不同检查点混写。输入所有权不转移。

受影响源码对照：404 结构规则、378 收尾规则、416 最终合并、204 连续页（包含原零正文高度异常）、35 原始页源整链及 15 Unicode/标签字符左缘合同。语料有复用，不累计为独立文档。嵌套 `_LineItem`、`_TitleStylePrototype` 的初版导出漏还原类型，冻结文件保留，另生成 typed/canonical 参考；修正的是 JSON 与 Python 对象之间的测试边界，没有修改参考算法或过滤失败用例。零高度在现有 Python 实际执行到除法的位置返回错误；重复来源仅最终页面路径保持对象身份，原独立严格入口不放宽。

统一 release 构建（3 分 57 秒）及安装已完成。新安装套件通过 204 连续页、35 原始页源整链、354 组件、15 Unicode/标签合同及五页实时 PDFium；原标题、完整页准备、最终公式和 118 模块兼容性四组安装套件通过。日志和明细见[主链记录](2026-10-27-form-runtime-integration.md)。无全量/CUDA。

## 精确剩余位置

下一句是 `_build_text_blocks(...)`，需要完整迁移其段落边界、悬挂分组、语义/正文连接、分组脚注、文字块及全部补合并；然后继续图注/标题/文献成员块重建、局部列表、视觉注释、最终排序、归一化和行内证据接入，再接默认 PDF 与用户整篇质量/耗时/峰值内存验收。上述行状态已完成，不再笼统列成“最终标题未完成”。默认 PDF 完成及整篇等价、效果/速度/内存提升标志仍为 False；CUDA 与 enhanced 阅读顺序后置。
