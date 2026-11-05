# Final two native table candidate producers

连续迁移aligned-numeric与captioned-mixed-numeric两个完整旧分支，不改阈值或以预选区域替代判定。aligned复用已迁移字符文字准备、数值重分组及最终字符网格确认，补齐numeric_region_bounds；mixed保留多层表头、原生年份字形切分、附近脚注和重复表头单行续表。最后按旧numeric替换及existing_bounds次序联合页面编排。

仅跑受影响旧函数差分、区域边界与真实PDF源重建；阶段末统一一次wheel和相关安装回归。固定CJK字体运行时失败样本及原冻结参考保留，不提升默认PDF完成标志。不需要GPU或用户补充资料。

## 本轮实现完成

`aligned_numeric`与`captioned_mixed`两个完整候选产生器已接通。aligned复用既有字符文字/数值重分组与最终区域网格恢复，新增完整numeric_region_bounds；只做旧检测器实际消费的区域确认，不产生未消费的诊断HTML。mixed保留短文本首列、叶级与跨列表头、原生年份字形切词、真实竖线/表头横线、贴邻同字体表注、已知完整重复表头证明的单行续表。旧表题来源被排除导致KeyError的输入仍失败，并映射为SDK结构化invalid_input；不默默接受或丢弃。

页面编排按原次序：year、aligned及已有候选覆盖过滤、metric，再year＋metric＋aligned完整替换；existing_bounds形成后unruled＋mixed＋pair＋clipped。全部候选产生分支及其相对次序已接通；尚未执行最终整页caption join、prose tail、Form/续表外置等策略，因此不把该接口称完整_detect_table_candidates等价，不改变默认PDF。

## 证据与边界

- 157个完整函数参考（46 aligned/67 mixed接受）＋15个特殊路径（9正例）＋1331个全产生器编排参考（980正例），安装统计1503个结果事实＋1个旧缺失来源异常。多个集合复用输入，不能当作独立文档。
- 另11个numeric_region_bounds直接Rust事实，验证上方正文、贴邻左列续行、物理边界等，不重复计入1503。
- 55页实际PDF实时字符/视觉行/绘图源重建：52个既有/新增查询＋3个独立近底边aligned正例；3个aligned及2个mixed接受正例。远底边的最初3个aligned负例原样保留；mixed跨列表头实际样本未接受，其接受路径由旧源级特殊参考覆盖。
- 未嵌入CID固定CJK字体运行时差异仍保留原失败和ignored回归，不计入这55页。当前55页中含此前独立嵌入字体指标页；Form/图像/签章实际输入为空，不能据此宣称提取已完成。
- 输入所有权独立；JSON/共享10M Work/有限几何/4096行/5000路径/20k字形成员网格/8MB输出边界保留，新增两个完整路径的共享预算耗尽及恢复验证。默认legacy_native_table_recovery/legacy_table_execution和固定CJK字体capability仍为False；新native_table_candidate_producers仅标明产生器可用。

## 后续整合

候选产生器不再列入待办。剩余是：最终整页caption分组/稀疏链合并、无框正文尾段裁剪、Form最终排除、续表标记外置、短中文标题、末表轨道提取；固定CJK字体运行时；diagram和默认PDF接入；完整PDF→Markdown对照与性能/内存验证。enhanced阅读顺序和CUDA仍按用户要求延后。

阶段验收完成：20个受影响结构链Rust单元测试、1个55页PDF实时源重建集成测试、2个既有对齐数值恢复/正则集成测试通过。阶段末一次release wheel及一次隔离安装；新全产生器、已有五类数值产生器、对齐数值恢复、完整替换、compatibility、smoke共6组安装套件通过。fmt、110模块兼容性审查及固定36＋3＋1上游哈希通过；runner登记128组，本轮未全跑，无CUDA。旧未嵌入CID字体失败仍保留且不计55页通过。日志build/native-all-candidate-producers-install-test.txt、build/native-all-candidate-producers-regression.txt。
