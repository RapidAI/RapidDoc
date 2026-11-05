# Native table block commit Implementation Plan

**Goal:** 串联旧_materialize_table_blocks的恢复/媒体/弱表撤销/投影/原子提交路径。

**Architecture:** 复用既有prepare/annotations/完整引擎/语义HTML/spatial投影，在table_block下添加commit与media子模块。源字符校准每页只做一次，原始字符仍负责投影。计算成功前不发布注释或认领；候选去重使用包含注释的原bbox。保留临时媒体URI与图内文字回退，PNG实际编码阶段后续接入。

**Tech Stack:** Rust、serde_json、现有Work预算及原样旧算法。

1. 独立旧Python冻结完整块级结果、候选/行副作用及媒体几何/占位结果；覆盖原生接受、投影、空内容、重复候选、注释事务、弱表撤销、真实PDF原语和媒体。
2. 接通跨阶段Work及每页校准缓存；复用已有空间投影，不重新发明文本规则。
3. 实现媒体选择/遮罩/唯一物理cell归属及临时URI插入；失败则继续原文字恢复。
4. 暴露materialize_native_table_blocks，返回表体、注释、已认领来源与独立更新后的候选/行。支持有界普通JSON源；资源限额错误传播，不伪装普通恢复失败。
5. 串行Rust全量/fmt/clippy/CUDA编译、构建安装wheel、独立事实和相关回归；更新审查与文档。真实PNG编码、完整页源检测、默认集成与既定延期项继续后续。

执行结果：5步已完成。781完整旧事务（192原生、512投影、102首页撤销、109注释回滚、24媒体、4恢复异常）、八候选页缓存和96独立媒体事实冻结；2个非str Char原始引擎输入的PageSource域排除理由单列，原引擎基线保留。231项Rust测试、fmt/clippy、CUDA feature编译、12组安装套件和75模块来源审查通过；固定36＋3＋1源码哈希未变。最终wheel验证16实时PDF/角度串联（视觉行由glyph重建）、副作用/类型/所有权/并发/累计限额/异常恢复；96媒体事实在Rust中40附着56拒绝，完整wheel覆盖24媒体案例。未全跑93套件/36像素或实测GPU。日志build/native-table-block-commit-*.txt、build/native-table-media-rust.txt与build/rust-current-*.txt。
