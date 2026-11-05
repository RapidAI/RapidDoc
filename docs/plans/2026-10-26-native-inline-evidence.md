# Native inline style and link evidence Implementation Plan

**Goal:** 移植旧视觉行的字体/绘图样式和链接注解区间检测。

**Architecture:** 复用固定vendor inline_styles数值内核及已验证字体元数据解析，适配字符原序/索引、严格框和紧凑Unicode匹配文本。链接按旧中心/面积与冲突规则形成半开区间。统一JSON接口返回style_lines/link_lines；按输入规模共享工作预算。字形像素粗体推断、完整PDF链接注解读取及输出投影另续，不提升默认完成标志。

**Tech Stack:** Rust、serde_json、固定Unicode16、现有PyO3 JSON入口。

1. 独立冻结实际旧检测函数输出，覆盖字体/短run/项目符号、绘图/分数线保护、链接冲突和真实PDF视觉行链。
2. 实现有界适配和链接算法，开发只跑新定向Rust对照。
3. 阶段结束构建一次wheel，验证实时字符链和相关安装回归；不跑全量/CUDA。

执行完成：275基础旧事实后追加10多字符项目符号事实，复现了按原片段而非规范化片段判定标记造成的误加粗。修正后两个定向Rust测试通过285事实及已有Work保留检查；首次构建后因此追加一次必要修正构建，没有运行全量测试。

最终wheel隔离安装通过285逐值/类型/区间/顺序参考、57实时PDF字符→视觉行→局部准备→样式/链接链（2样式/4链接正例）、所有权/并发、计数/累计扫描预算和错误后恢复；链接注解仍为旧PDF冻结事实。原source_lines、native_table_materialization、compatibility、smoke四组相关回归通过，共五组安装套件。fmt、79模块来源审查及固定36 core＋3 paths＋1 scripts哈希通过；未跑全workspace、全部97组runner、未受影响渲染、全量clippy或CUDA。

日志build/native-inline-evidence-rust.txt、build/native-inline-evidence-edge-before-fix.txt、build/native-inline-evidence-final-wheel.txt、build/native-inline-evidence-install-test.txt、build/native-inline-evidence-regression.txt。下一步SDK链接注解读取、像素字重补证及证据输出投影；完整页源/候选检测和默认PDF集成仍待，未作整文档质量/性能结论。
