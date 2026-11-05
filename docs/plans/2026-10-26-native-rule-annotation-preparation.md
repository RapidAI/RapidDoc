# 规则表格边界、表题和注释准备

以规则边界、局部视觉行和明确core_row_indices准备核心框/体成员；迁移 `_collect_caption_rows` 与 `_clip_visual_row_to_corridor` 完整表题续行规则，再迁移 `_build_table_annotation` 的表体来源/几何排除、重复来源框原序并集和空注释规则。表注行先明确给出footnote_row_indices，自动表注收集仍待。

普通JSON行/片段保留原bbox与local_bbox区别，annotation行框用有序键框对跨Rust桥接、Python返回整数键字典。冻结旧完整helper及核心参考，补实际PDF实时文字/边界→原视觉行片段查询。仅定向测试，阶段末一次wheel/相关回归，无全量/CUDA。未接完整临时候选产生/物化、表注自动识别或默认PDF。

## 实现与验证范围

- 新接口返回rule_bbox/core_local_bbox/core_line_indices、裁剪后的caption_rows和annotations，尚不输出完整正式候选。规则边界和core_row_indices是明确上游事实；core成员取片段line_index，核心框按原先规则框、再核心行框并集顺序保留坐标赢家类型。
- 表题从caption_source_index首次出现的row开始，横中心位于规则走廊±2em的片段保留并按local左缘稳定排序；使用既有CPython补偿sum重算center_y，再以局部框union重算行框，保留visual_row_id。续行中心到表顶、相邻间隙2em和末行距表顶2em均按旧规则。
- 注释记录去除核心来源；caption还排除local底部越过核心顶部且横覆盖>=5%的片段，footnote不加这项几何排除。重复来源以原bbox按首次出现顺序union；输出注释整体框按行框字典插入顺序union。原bbox和local_bbox不能混用。空注释不输出。
- 136旧完整helper及核心组合参考，124表题行/136注释正例；覆盖首种子缺失/偏栏、2em临界、5%交叠/严格顶边、重复与负source ID、原/局部坐标不同、核心成员排除、相等数值不同int/float赢家类型。
- 12页实际PDF（0/90/270）实时提取字符/路径/绘图与视觉行后构造明确查询，2表题/7注释正例。每条视觉行投影为单片段row，local暂等于原bbox；core从首填充框选行，Table/Note fixture文字选择种子和表注行，属于测试输入，不称自动表题/表注检测或完整表格聚行。未将180度纳入该语料。
- 普通JSON、positive finite median_height、finite abs<=1e9、4096行/合计20k源片段/5000规则/各4096选择索引，表注选择累计20k片段；既有JSON/10M累计Work/8MB输出。核心成员/片段裁剪/排序/注释原序归并计费，接受已有Work，无跨调用缓存。

## 下一项

自动表注行收集已完成，见 `2026-10-26-native-table-footnotes.md`；下一项把核心/表题/表注记录组装为正式候选并还原方向坐标，随后接连通网格扩框。页面级临时候选产生、补充表格产生器编排和默认PDF仍待；此阶段不作整文档质量/性能结论。

## 验收完成

- 两个Rust单元测试、一个12页PDF实时查询集成测试通过。
- 阶段末一次release wheel和一次隔离安装；native_rule_annotation_preparation、native_table_candidate_merge、native_table_replacements、compatibility、smoke五组安装套件通过，验证原序行框字典/数值类型、原与局部坐标、表体成员排除、深层所有权、并发、预算/资源错误及恢复；运行时不导入旧Python解析器/PDFium。
- fmt、104模块审查、固定36＋3＋1上游哈希通过，全量runner登记122组，本轮未全跑，无CUDA。
- 日志build/native-rule-annotation-preparation-rust.txt、build/pdf-rule-annotation-preparation-rust.txt、build/native-rule-annotation-preparation-wheel.txt、build/native-rule-annotation-preparation-install-test.txt、build/native-rule-annotation-preparation-regression.txt。
