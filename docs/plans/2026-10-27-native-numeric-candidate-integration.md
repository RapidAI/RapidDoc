# Remaining native candidate integration

当前更新：最后aligned及captioned-mixed两个产生器已在[后续阶段](2026-10-27-native-final-candidate-integration.md)接通；候选产生器全部从待办移除。以下“剩余两类/七类”均为历史记录。最终整页策略与默认PDF等仍待。


继续旧Python迁移：先连续实现partial-rule-grid、year-header与metric-pair三个不依赖额外恢复器的完整候选分支，并按旧partial→open、year→metric替换仲裁次序接入结构页面；随后推进其余aligned/unruled/mixed/pair分支。复用自动聚行、既有完整替换仲裁及JSON/Work边界，不改旧规则阈值或默认PDF完成标志。

每个完整函数直接冻结旧参考，含接受/拒绝、数字词法、方向/语义、边界与顺序；组合参考检验替换次序，实际PDF重建源检验真实分支。只跑受影响测试，合并阶段末一次wheel安装，不逐helper构建，不全量/CUDA。既有冻结参考/PDF和上游哈希不改写。

## 已实现并接入

本轮连续完成partial_grid、year_header、metric_pair、unruled_numeric、captioned_pair五个完整旧产生器。复用既有自动聚行、Unicode数字/字形工具与完整替换仲裁，未更改规则阈值：部分网格仅补共同竖线末端底边且保留原全部drawing；年份序列必须小步递增、后续表头构成硬隔断；指标值满足80%数值与两端对齐；无框数值表要求粗体覆盖和每列原生alpha字形；双列文字表要求深色表头及每行右列原生字形。输出保留旧评分、注释、成员/网格权威标志与int/float类型。

`detect_native_supplemental_tables(..., kind=...)`增补上述五个kind；`detect_native_structural_table_candidates`共同编排partial→open恢复、year→metric完整覆盖替换仲裁，既有区域排除后补无框数值/双列文字，最终再裁切表单。aligned-numeric与captioned-mixed-numeric仍缺，不把该入口称完整一般检测器。

## 对照与运行范围

- 222旧部分网格/年份/指标参考，其中47/50/45接受正例；144无框数值/双列文字参考，其中59/34正例。1159联合编排参考（848正例），安装套件合计1525事实；部分共享输入，不能当作独立文档。中间1015编排参考保留用于定位，未重复计数。
- 实时PDF源重建验证46页：原46记录排除1个未嵌入CID指标页，另加1页独立嵌入字体指标表（45＋1）。五个新增分支各有1个真实提取接受正例；46页联合编排33个正例。实际Form/图片/签章框为空；其排除行为由旧源级事实覆盖，不冒称原生提取已接通。
- 共享Work、输出/几何/字形上限，输入不修改；原序注释、所有权、并发与错误恢复由定向安装套件验收。阶段末统一一次wheel，不逐helper构建，不全量/CUDA。

## 新发现的字体运行时边界

未嵌入`STSong-Light`的CID指标PDF暴露了旧固定CJK替代字体与新Rust系统默认字体的差异。首字顶部：旧`82.625`，当前Rust `82.58001708984375`。旧`pdfium_guard()`安装`font_runtime.py`固定DroidSansFallbackFull（资源SHA256为`8a4dea0899424438af25a6f1f6eb61e5d111d85367eece0a5d30170861ae6b2e`），当前Rust未接入。普通Python ctypes读取在未安装该字体提供器时与Rust相同，已定位为运行时差异，不能改候选阈值掩盖。

保留原冻结`pdf-numeric-candidates.json`与原PDF；`native_numeric_candidates::unembedded_cid_requires_fixed_font_runtime`为显式ignored可执行失败回归。原CID源直接送入新候选函数的对照仍通过，但只算源级差分，不能算实时几何等价。另建嵌入字体正例验证指标分支，不替换原记录。新增capability `native_fixed_cjk_font_runtime=False`，接默认PDF之前必须完成该字体运行时并取消此ignore。

剩余：两类候选、最终整页caption/tail/Form/续表策略与末表轨道自动提取、固定CJK字体运行时、diagram/默认PDF、全文统一回归。无需GPU或用户补充信息。

安装验收完成：16个结构链Rust单元测试＋2个完整替换回归通过；实时PDF集成46页通过，1个未嵌入CID失败回归明确ignored并单独重现确认。阶段末一次release wheel和一次隔离安装；新数值候选、原结构链、完整替换、compatibility、smoke共5组安装套件通过。fmt、109模块兼容性审查与固定36＋3＋1上游哈希通过；runner登记127组，本轮未全跑，无CUDA。日志build/native-numeric-candidate-pipeline-install-test.txt、build/native-numeric-candidate-pipeline-regression.txt，已知字体失败日志build/native-cid-font-runtime-known-failure.txt。
