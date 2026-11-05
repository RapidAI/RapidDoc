# Native PDF link annotations Implementation Plan

**Goal:** Rust直接读取PDF外部URI链接注解，串入现有视觉行链接检测。

**Architecture:** 复用唯一PDFium锁/运行库和RAII句柄，按旧枚举顺序保留source_index；隐藏注解/非URI动作/非法UTF-8和目标跳过。QuadPoints优先，全部无效时回退Rect，按原CropBox/旋转转换、裁页、去重排序。URI策略移植旧sanitize_hyperlink_target及其urllib解析边界，不使用浏览器URL规范化。普通坏链接跳过，资源错误传播；默认解析标志保持False。

**Tech Stack:** 现有PDFium raw bindings、Rust标准库IPv6解析、固定Unicode16 NFKC、PyO3。

1. 独立冻结旧URI策略和实际PDF注解（隐藏/动作/多Quad/旋转/CropBox/坏目标及已有PDF），不覆盖语料。
2. 实现有界URI策略、可见性/原生读取与坐标处理，开发只测受影响部分。
3. 阶段结束构建wheel，实时读取注解与字符→视觉行→页源准备→链接证据串联验证，相关回归，无全量/CUDA。

执行完成：URI策略定向Rust测试通过207旧事实，原生读取定向Rust测试通过36实际PDF页/26有效注解及坏文档/页号错误。阶段末只构建一次release wheel并隔离安装；新SDK通过36页注解、57组实时字符＋实时注解→视觉行/局部准备/inline证据链（无注解结果注入）、路径/bytes/流所有权、并发、加密/NUL/错误后恢复和两份作者URI/Quad限额PDF。作者限额输入不计作旧Python等价证据。

原pdf_character_primitives、native_inline_evidence、compatibility、smoke四组回归通过，共五组安装套件。fmt、80模块审查及固定36 core＋3 paths＋1 scripts哈希通过；未跑全workspace、全部98组runner、未受影响渲染、全量clippy或CUDA。日志build/pdf-link-uri-rust.txt、build/pdf-link-annotations-rust.txt、build/pdf-link-annotations-wheel.txt、build/pdf-link-annotations-install-test.txt和build/pdf-link-annotations-regression.txt。

后续仍需像素字重补证、样式/链接最终输出投影、图片/噪声/跨页页源准备与候选检测/默认集成，不作完整PDF或性能结论。
