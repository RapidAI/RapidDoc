# Native visual lines Implementation Plan

**Goal:** 沿旧Python字符分行规则构建原生视觉行，作为完整页源的基础。

**Architecture:** 复用固定vendor text_pipeline/inline_pairs数值内核；字符仍为普通JSON，所有原字段保留。包装层补齐旧LineItem默认字段、排版度量、递归上下标合并、稳定来源号及最终行序。候选检测和完整页源收集另续，不开启默认完成标志。

**Tech Stack:** Rust、固定Unicode16表、serde_json、已有PyO3 JSON入口。

1. 冻结独立旧Python合成字符与真实PDF字符的完整视觉行，不覆盖已有语料。
2. 实现有界字符适配与固定Unicode属性、视觉run构建及旧行物化/上下标合并。
3. 开发只跑新增视觉行定向Rust测试，修正实际差分；无CUDA检查。
4. 阶段结束构建一次wheel，验证新SDK和受影响排版/脚本组件；记录范围与剩余工作。

执行完成：新增定向Rust测试通过317旧事实/977行；release wheel本阶段构建一次并隔离安装。安装接口逐值/类型/全部公开字段核对317事实，57组实时PDF分别读取代理对恢复或去重字符后重建视觉行，输入不变、并发、限额、组行+配对共享Work及错误后恢复通过。五组相关回归（table_script_preparation、native_inline_clusters、native_table_block_commit、compatibility、smoke）通过，共六组安装套件。

77模块来源审查、fmt和固定36 core＋3 paths＋1 scripts哈希通过。日志build/native-visual-lines-rust.txt、build/native-visual-lines-wheel.txt、build/native-visual-lines-install-test.txt及build/native-visual-lines-regression.txt。本轮未跑全workspace、全95组runner、未受影响的渲染对照、全量clippy或CUDA检查。完整页源其他准备、候选检测与默认集成仍待，不能将本阶段解释成完整PDF等价或性能改善。
