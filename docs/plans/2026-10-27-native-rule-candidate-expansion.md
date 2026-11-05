# 规则临时候选组装为正式候选

复用已迁移核心/表题准备和完整自动表注收集，在同一Work内实现旧 `_expand_rule_table_candidate` 的普通、非prepared参考路径。保持完整local框按核心/表题/core行/表注顺序并集、注释原坐标、表体与注释ID互斥，再还原bbox/core_bbox方向坐标，返回score=0及原候选默认标志。

不接受预选表注行或引用结论；has_possible_table_notes显式保留旧负向门控。规则边界、核心行及表题种子仍是上游事实，规则候选产生器、prepared/owned优化与连通网格扩框另待。冻结旧完整函数，实际PDF实时输入对照，阶段末一次wheel/相关回归，无全量/CUDA。

## 实现与验证范围

- 新API复用已有核心/表题与完整自动表注函数，不传入任何预选表注或引用结论；子调用共享已有Work。has_possible_table_notes默认为True，False时保留旧负向门控，不读取未消费line内容。
- 原函数完整普通路径：rule框与core行并核心，自动收集caption/note并构造原坐标注释，local框按caption/core/footnote原序并集后以core首项union，core成员减去注释成员；bbox/core_bbox从upright逆变换，local_bbox不旋转。90/180/270方向坐标差保留int/float类型，其他角度沿旧返回原局部框。score=0.0，inferred_grid/样式/权威网格标志沿旧默认。
- 冻结155旧完整 `_expand_rule_table_candidate` 调用参考，139带注释；包含局部0/90/180/270/其他方向、整数/混合坐标、caption/core/note来源重合、原框/局部框差异、has_possible_table_notes正负门控。未使用prepared_core/prepared_candidate或owned分支，普通路径语义完成不等于已实现这些存储优化。
- 20页实际PDF实时提取字符/字形/路径/绘图和视觉行后组装，7带表题或表注。规则框由绘图bbox或首填充框fallback构造，core按首框内行中心、fixture Table种子及单片段local=原框是明确上游查询，自动表注由新规则判定。实际页面0/90/270、查询angle0；不宣称完整规则候选产生、聚行或caption种子自动识别。
- 普通JSON、各4096rows/lines/选择索引、5000边界及子阶段20k片段/字形/coreIDs、finite abs<=1e9、positive median_height和严格boolean负门控；JSON/10M累计Work/8MB输出，错误传播、无跨调用缓存，输入不修改。

## 下一项

正式候选组装及方向还原已消掉。下一项连通网格扩框；随后接规则临时候选自动产生/执行、补充候选产生器、一般页面候选编排，再接diagram与默认PDF。当前不作整文档质量/性能已完成的结论。

## 验收完成

- 两个Rust单元测试、一个20页PDF全组装集成测试通过，含已有Work总量不足时不得在子调用重置预算的测试。
- 阶段末一次release wheel和一次隔离安装；native_rule_candidate_expansion、native_table_footnotes、native_table_candidate_merge、compatibility、smoke五组安装套件通过；验证完整候选字段/原序行框/数值类型、方向坐标还原、体/注释ID分离、负向门控、深层所有权、并发与异常恢复。安装环境未导入旧Python解析器/PDFium。
- fmt、106模块审查、固定36＋3＋1上游哈希通过，全量runner登记124组，本轮未全跑，无CUDA。
- 日志build/native-rule-candidate-expansion-rust.txt、build/pdf-rule-candidate-expansion-rust.txt、build/native-rule-candidate-expansion-wheel.txt、build/native-rule-candidate-expansion-install-test.txt、build/native-rule-candidate-expansion-regression.txt。

后续状态：连通网格扩框与自动闭合候选完整链已在[闭合物理网格阶段](2026-10-27-native-closed-rule-pipeline.md)接通；旧“下一项”描述为当时状态。普通横线区间自动产生尚待。
