# 非表格容器：代码块

后续更新：[目录预认领与Form选择](2026-10-27-native-page-containers.md)已完成并通过5组安装验收；本记录中的目录/Form选择剩余项属于历史状态，当前待完整Form采集/角色归一化及图像图形物化等后续接线。

迁移旧code_blocks.py完整入口：填充背景/等宽字形、横线限定结构清单、算法标题与真实栏走廊，统一字符去重/空间投影（保留空行）/回退文本和来源行认领。

复用已有文本栏推断及空间投影；旧表格投影保持infer_tight_spaces=True、无额外空行，代码使用False/保留空行。词法判定保留Python Unicode数字、单词边界和IGNORECASE特例，东西文字宽组从固定Unicode数据生成。所有分支共享Work，来源号须唯一。

独立入口及正式页准备中的两次代码认领顺序已接通；不声称完整页准备完成。以旧完整函数、实际PDF及受影响旧投影/栏/表格回归验证，阶段末wheel（整数边界修复后补构建），无全量/CUDA。

## 实现与验证

- 完整filled/ruled+algorithm两个入口复用真实栏推断，不注入栏走廊或代码候选。词法缓存只属于当前源，保持Python单词边界/Unicode数字/IGNORECASE行为。
- 代码使用spatial::project_options(false,true)，旧表格入口仍为(true,false)；空行数量和字宽槽均按旧round规则。原512行栏入口不变，代码布局内有界支持4096唯一来源行。
- Unicode16宽字符122范围固定在wide.json，export_code_unicode.py可重现/验证；字体名复用旧完整casefold表。
- prepare_native_code_table_page在规则代码后才检测表格，在表格认领后才处理底色代码；末表轨道同时考虑代码固定块。共用Work与原始/分析行副作用，旧table-only接口输出保持。
- 227完整代码函数参考＋63顺序组合，合成无轨/无真实行身份旧误判保留，不替换基线；另3物理网格混排源同时保留代码和表格。
- 67不同实际PDF页，新增8页有16个独立builder查询（3代码正例）；组合页阶段3代码页/41表格页。8新增页与67组合页重叠，不相加报75页。
- 9受影响Rust单元及2PDF集成通过；格式化、116模块审查与原固定36＋3＋1哈希保持。runner登记134组安装套件及独立PDF渲染，未全跑，无CUDA。

安装验收已完成：新代码容器、旧表格页提交、文字栏、多行投影、表格块提交、compatibility及smoke共7组均通过；整数修复后补构建并复验，同阶段共2次release构建。最终安装日志 `build/native-code-integer-boundaries-*-install-test.txt`，边界Rust日志 `build/native-code-integer-boundaries-rust.txt`。Rust日志 `build/native-code-containers-related-rust.txt`、`build/native-code-containers-all-pdf-rust.txt`，另保留各子阶段导出/测试日志。剩目录/Form/图片图形提交及后续几何公式文本接线，不开启默认PDF完成标志。

整数边界检查：新增独立4例旧Python参考，不改既有冻结语料；修复RGBA减法与缺失char_idx时来源去重键的i64环绕，改用i128保持旧Python整数结果。修复前已复现dark-min漏检和相隔2^58来源号误去重，另外2边界作为对照。仅重跑受影响Rust/安装测试并补构建，不启动全量。
