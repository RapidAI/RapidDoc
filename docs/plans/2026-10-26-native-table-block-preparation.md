# Native table block preparation Implementation Plan

**Goal:** 移植旧 table_materialization 的块级准备，保持原字符来源、候选注释和认领范围。

**Architecture:** 在已有 legacy_model 下添加 table_block.rs；普通 JSON 输入 source/candidate，不执行最终输出认领。按旧顺序计算汉字字体组框校准、表内脚注分离、同基线/旋转成员扩展、失败注释并回表体。显式校准覆盖自动校准；不修改调用方输入。

**Tech Stack:** Rust、serde_json、现有 Work 资源预算、PyO3 统一 JSON 分派。

1. 新建 scripts/migration/export_native_table_block.py，独立运行旧 Python 私有函数，冻结四旋转、阈值边界、迭代扩展、注释失败、自动校准与实际 PDF 原语输入；拒绝覆盖基线。
2. 新建 rust/rapiddoc/src/legacy_model/table_block.rs，移植上述函数和必要的几何/字号/表注谓词；限制 JSON、输入规模、有限坐标与累计比较预算。
3. 在 bindings/python/src/lib.rs 和 python/rapid_doc_rust/__init__.py 暴露 prepare_native_table_block；返回 candidate、projection_line_indices、body_bbox、calibrated_char_bboxes、native_chars。failed_annotations 是明确的阶段输入，尚不执行注释文本物化。
4. Rust 差分测试和 bindings/python/tests/native_table_block_install.py 核对完整值/类型、所有权、并发、资源错误及真实 PDF 读取链；验证输出字符可继续进入已有完整恢复与整表语义渲染。
5. 串行执行 fmt/clippy、相关和全量 Rust 测试、CUDA feature 编译、构建安装 wheel、相关安装套件；更新来源审查、许可证和接续文档。

本轮不将准备阶段称为完整块级物化。注释输出、首页弱表撤销、投影降级、图片归属、最终原子提交与默认 PDF 接入继续后续实现。保持既定目录、包名、解析策略和延期项；不提交或修改用户现有改动。

执行结果：上述5步已完成。独立冻结612准备＋12类型及16旧准备→恢复/HTML链，Rust实现和Python接口已加入并安装最终wheel；228项Rust测试、fmt/clippy、CUDA feature编译、9组相关安装套件及73模块来源审查通过。固定36＋3＋1上游哈希未变。实际PDF非空行由实时字符重建glyph视觉行，空行阶段输入4组；12恢复非空。共同预算、所有权、并发和错误后恢复检查通过。全量91套件/36像素组未全跑，GPU未实机测试。日志build/native-table-block-*.txt及build/rust-current-*.txt。
