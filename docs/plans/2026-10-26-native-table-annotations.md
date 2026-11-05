# Native table annotations Implementation Plan

**Goal:** 移植旧注释块物化和首页弱表撤销，衔接已完成的块级准备。

**Architecture:** 在既有table_block模块下增加output.rs，复用字符校准、候选分离、Work和物理行连接。准备入口可自动执行注释物化，再用真实失败集合计算表体框及投影成员。首页撤销入口显式返回判定与修改后的行，仅供结构恢复失败路径使用。

**Tech Stack:** Rust、serde_json、已固定Unicode16 casefold/NFKC和文本连接、PyO3 JSON分派。

1. 独立旧Python冻结注释分组/输出/失败集合/成员/元数据和弱表撤销输入、完整副作用，保留原基线。
2. 实现字体族、原生文本清理、tight框裁页、注释排序/分组/内容/元数据和外置事务；共享外层Work，不提交最终表格块。
3. 实现原metadata/text_role谓词与首页规则，保留真网格保护与标题副作用；Unicode16明确固定。
4. 暴露prepare_native_table_annotations及restore_native_front_matter_panel，保持现有prepare_native_table_block结果不变，增加差分、实时PDF、所有权/并发/累计预算测试。
5. 串行运行Rust测试、fmt/clippy、CUDA编译、构建安装wheel和相关套件；更新审查、归属及接续文档。投影降级、媒体归属、最终原子提交和默认接入仍为后续工作。

执行结果：5步完成。904旧注释事务＋24类型、240首页面板/102撤销及32角色/元数据事实独立冻结；16实际PDF/角度无注释负例实时读取（非空glyph视觉行实时重建）。229项Rust测试、fmt/clippy、CUDA feature编译、10组安装套件和74模块来源审查通过；固定36＋3＋1上游哈希未变。最终wheel已安装，所有权/并发、4000字体组准备和1600注释各自成功但组合超10M预算失败、异常恢复和许可证检查通过。未全跑92套件/36像素组或实测GPU。日志build/native-table-annotations-*.txt及build/rust-current-*.txt。
