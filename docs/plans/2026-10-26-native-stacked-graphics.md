# 堆叠柱图和图形尺度校准

沿旧graphics.py的 `_native_graphic_em`、`_detect_stacked_column_graphics` 及 table_filled_grid.py 的CJK指标判定。读取独立页源几何，返回校准尺度、异常CJK证据和柱系列候选框；不是完整强图形检测，也不跳过表格/重叠仲裁。

冻结独立旧阈值/正负参考，迁移柱宽、共同基线、列间隔、同列叠层、数值刻度、横轴/标签扩框；再运行定向差分和实际PDF页源输入。阶段末一次wheel和相关安装回归，无全量/CUDA。源数据不修改，资源预算累计传播。

已完成304组旧输出对照，162柱图正例、210尺度校准。包含20锚点数量/CJK范围、松紧框比例/普通中位数、柱宽/间距/共同基线75%、高柱不截断、同列上层矩形、Unicode刻度/逗号小数、横轴和填充路径标签。图体候选按原path顺序输出，可重复；不在此阶段合并或认领，不改变输入。

41页实际PDF调用SDK实时读取字符/路径/绘图线并生成视觉行后接新接口。它们没有inflated或柱图正例，只作为真实负例链；正例来自独立合成页源，不宣称已验证实际异常CJK柱图的完整解析效果。原冻结PDF及旧JSON不覆盖。

接口 `detect_native_stacked_graphics(source, *, em, fallback_em)` 返回graphic_em、inflated_cjk_metrics、stacked_column_bboxes。em/fallback是显式调用参数，尚未实现完整图形尺度编排。普通JSON，候选/尺度输出为浮点PDF point；最多20000字符、4096行（每行20000码点）、5000路径和绘图线、合计20000矩形，有限数值abs<=1e9，CJK tight锚点必须有有效loose框。沿既有JSON/10M累计Work/8MB结果预算，预付中位数/排序和同列矩形匹配扫描，文本按字节计费，接受上层已有预算；资源错误明确传播。

两个定向Rust单元测试通过；阶段末一次release wheel构建并隔离安装。native_stacked_graphics、native_raster_charts、compatibility、smoke共四组安装套件通过，覆盖字段/类型/顺序、所有权/并发、输入/叠层扫描限额和错误恢复。fmt、87模块来源审查及固定36 core＋3 paths＋1 script哈希保持。未跑全workspace、全部105组runner、未受影响渲染、全量clippy或CUDA；两个legacy完成标志False。

日志：build/native-stacked-graphics-rust.txt、build/native-stacked-graphics-wheel.txt、build/native-stacked-graphics-install-test.txt和build/native-stacked-graphics-regression.txt。完整强图形/表格/重叠仲裁、跨页准备、候选检测及默认PDF编排仍待，不作整文档质量/速度/峰值内存结论。
