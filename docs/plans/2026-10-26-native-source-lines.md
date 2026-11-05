# Native source-line preparation Implementation Plan

**Goal:** 接通旧页源收集中的首字母段、悬挂字母段、薄栅格分隔线和装饰文字规则。

**Architecture:** 输入现有视觉LineItem普通JSON，保持原序与字符/几何。严格依旧收集顺序运行两个局部段落函数，按em_height上中位数补极薄水平图片线，再转投页顶私用区重复装饰字符。输出独立lines/drawing_lines。字体样式/链接/水印/噪声、跨页准备和候选检测留后续，不替代完整页源。

**Tech Stack:** Rust、serde_json、固定Unicode16、既有PyO3 JSON入口。

1. 独立执行旧Python函数冻结正反例、边界和既有真实视觉行链，不覆盖原语料。
2. 移植局部段落判定和线规则，保留字段/类型/顺序与旧副作用，统一工作预算。
3. 开发只跑新阶段定向测试；阶段完成一次wheel构建与相关安装回归，无CUDA/全量测试。

执行完成：两个定向Rust测试通过290旧完整事实及已有Work保留检查；本阶段只构建一次release wheel并隔离安装。新SDK逐值/类型/全部字段和顺序核对290事实，57组实时PDF字符→视觉行→局部准备链、所有权/并发、输入计数、扫描预算、错误后恢复通过。现有native_visual_lines、native_table_block_commit、compatibility、smoke四组回归通过，共五组安装套件。

fmt、78模块审查和固定36 core＋3 paths＋1 scripts哈希通过；未跑全workspace、全部96组runner、未受影响渲染、全量clippy或CUDA。日志build/native-source-lines-rust.txt、build/native-source-lines-wheel.txt、build/native-source-lines-install-test.txt和build/native-source-lines-regression.txt。图片/绘图为明确阶段输入；完整样式/链接/图片水印/噪声/跨页准备、候选检测和默认PDF集成仍待，不作整文档质量/性能结论。
