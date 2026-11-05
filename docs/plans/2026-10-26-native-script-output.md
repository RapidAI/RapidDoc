# Native script output Implementation Plan

**Goal:** 接入新script_lines的最终Span投影与物化诊断，沿旧归属/回退/公式保护。

**Architecture:** 复用现有角色分类/分式保护生成完整sidecar，再接字符投影和样式/链接输出；增加整行主归属、脚本紧框备用归属、独立行匹配、晚期公式区域过滤及精确物化诊断。输入身份按原数组位置区分，不使用可能重复的source_index作对象标识。原apply默认行为保持，显式script_lines才运行脚本阶段；新增trace接口返回blocks和materialized_ranges。

**Tech Stack:** Rust、普通JSON、已有PyO3入口与固定Unicode16。

1. 冻结独立旧完整输出/诊断，覆盖主/备用归属、短脚本救回、公式和歧义、类型/顺序与实际PDF链。
2. 复用现有匹配核心实现脚本投影/诊断及共享预算，开发仅定向对照。
3. 阶段末一次wheel构建及相关安装回归，无全量/CUDA；完整页源/候选检测与默认集成仍待。

执行完成：四个相关单元测试和一个实际PDF集成测试通过，覆盖原297样式/链接输出、249完整脚本输出/诊断、516 sidecar构造和预算保留。44预分类分式成员输入保留于原560行精炼参考，不把它们重命名为原始检测输入；两个故意冲突sup/sub的作者随机组合由旧严格Span校验拒绝，未计入249成功事实。

阶段末一次release wheel构建并隔离安装，249/516逐值/类型/字段/顺序、所有权/并发/限额/错误恢复和默认兼容通过。57实时PDF负例链以及四方向专用真实PDF读取→视觉行/局部准备→脚本检测/证据→Span与诊断通过，7处实际物化范围；无字符/角色/脚本sidecar结果注入。块仍是明确视觉行派生输入，不替代整页块检测。

原native_inline_output、native_script_refinement、compatibility、smoke四组安装回归通过，共五组安装套件。fmt、82模块审查和固定36 core＋3 paths＋1 scripts哈希通过；未跑全workspace、全部100组runner、未受影响渲染、全量clippy或CUDA。日志build/native-script-output-rust.txt、build/native-script-output-pdf-rust.txt、build/native-script-output-wheel.txt、build/native-script-output-install-test.txt和build/native-script-output-regression.txt。像素字重、图片/噪声/跨页页源准备及候选检测/默认流程集成仍待，不作整文档质量/性能结论。
