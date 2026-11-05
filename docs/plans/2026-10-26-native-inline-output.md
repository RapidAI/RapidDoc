# Native inline output Implementation Plan

**Goal:** 把已提取的原生样式/链接证据按旧规则投影为输出块InlineSpan。

**Architecture:** 输入明确的finalized blocks及style_lines/link_lines，按几何唯一归块、码点文本对齐、公式空洞/断词/上下文回退和冲突过滤生成原字符串区间，再物化文字/公式/链接Span。旧detector-owned平坦sup/sub标记按原规则处理；本阶段不接script_lines检测和投影，不改变默认PDF入口。归一化单位bbox沿旧规则转page point，保留其他块字段。

**Tech Stack:** Rust、serde_json、固定Unicode16、冻结HTML实体表、已有PyO3 JSON入口。

1. 独立执行旧apply_pdf_text_links/styles及materialize_pdf_inline_spans冻结完整输出，包含公式、歧义、断词、Unicode和真实字符/注解链。
2. 移植有界文本投影/匹配/区间和Span物化，开发只跑定向差分。
3. 阶段末构建wheel、真实PDF链和相关安装回归，不跑全量/CUDA。

执行完成：两个定向Rust测试通过297旧完整输出及已有Work不重置检查。差分中修正链接子文字typed dump保留styles空数组的结构差异；未闭合公式查找逐次计费，避免扫描绕过预算。阶段结束只构建一次release wheel，隔离安装通过297完整块/字段/类型/码点偏移、57实时PDF字符＋注解→视觉行/局部准备/检测→Span链、所有权/并发、块数/未闭合公式预算及错误后恢复。

原pdf_link_annotations、native_inline_evidence、compatibility、smoke四组相关回归通过，共五组安装套件。fmt、81模块审查及固定36 core＋3 paths＋1 scripts哈希通过；未跑全workspace、全部99组runner、未受影响渲染、全量clippy或CUDA。日志build/native-inline-output-rust.txt、build/native-inline-output-wheel.txt、build/native-inline-output-install-test.txt和build/native-inline-output-regression.txt。

块为明确视觉行派生阶段输入；新script_lines匹配/诊断、像素字重及完整页源/候选检测/默认PDF集成仍待。链接证据目标限定PDF四协议，不支持Python私有区间对象；不据此声称整文档质量或性能等价。
