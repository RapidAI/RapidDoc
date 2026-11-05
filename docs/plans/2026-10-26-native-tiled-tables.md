# 原生相接填充单元格表格候选

迁移旧 `_detect_tiled_fill_table_candidates` 的完整阶段：真实 CJK 字框校准、相接矩形组件、行列轨证明、无底纹表头和推断线网格、相邻题名转移。SDK返回独立候选；不修改输入，不以外部布尔值替代文字或网格证明。

冻结旧函数完整输出及集合删除/弹出顺序，定向Rust验证；真实PDF链保留旧旋转和字体支持域。阶段末一次wheel及相关安装回归，不跑全量/CUDA。接入柱图时由本模块输出表框，完整柱组与默认PDF编排仍待。

## 验收

- 完整迁移旧 `_detect_tiled_fill_table_candidates`，不以外部 inflated/网格布尔值替代实际证据。保留原DrawingLine的start/end、稳定候选顺序、数值类型和重复source ID语义，提供独立候选/题名/推断网格。
- 冻结168旧完整输出参考，140候选正例、2相邻题名转移；8条独立CPython set(range) pop/difference_update轨迹覆盖5000成员和删除后缩容。来源文件和参考未覆写。
- 12页实际四方向PDF实时读取字符/路径/绘图，再构造视觉行并检测，4候选及表格填充块排除正例、1表头推断网格正例。测试字体是自行生成的矩形字形，加高额外字形形成loose框失真，而CJK tight框正常；未注入检测/字框结果。该样本用于几何规则验证，不代表客户PDF语料或完整文档质量。
- 3个定向Rust单元测试、1个12页PDF集成测试通过。阶段末一次release wheel、一次隔离安装；native_tiled_tables、native_bar_geometry、compatibility、smoke四组安装套件通过，检查完整字段/类型/顺序、输入所有权、并发、资源错误与恢复，运行时未导入旧Python解析器或字体生成库。
- 20k源字符/合计20k行字符、4096行各20k码点、各5000路径/绘图/排除框、合计20k推断线；finite abs<=1e9、既有JSON/10M累计Work/8MB输出。集合槽位扫描、文字、轨道、排序与题名操作累计计费，允许上层共用已有预算；无跨调用缓存。
- fmt、95模块审查和固定36＋3＋1上游哈希通过。全量runner登记113组，本轮未全跑，无CUDA。日志build/native-tiled-tables-rust.txt、build/pdf-tiled-tables-rust.txt、build/native-tiled-tables-wheel.txt、build/native-tiled-tables-install-test.txt、build/native-tiled-tables-regression.txt。

## 后续

该函数已可生成 `prepare_native_bar_geometry` 所需的 `table_bboxes`；旧接口继续要求显式传入表框。本阶段不包含同文件其他filled-grid族。下一步迁移柱组聚类、外部数字/图题保护及轮廓/栅格/标签扩框，随后完成强图形产生器集合和默认PDF编排。
