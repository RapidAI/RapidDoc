# PDF 图片区域裁决

后续进展：图片证据自动读取已完成，见 [PDF图片证据自动读取](2026-10-26-pdf-image-infos.md)。本文保留本阶段验收时的边界记录。

目标：按旧 `_collect_document_sources` 顺序迁移全文重复水印、正文页平滑背景过滤和标题下已证明的图片白边裁剪。复用现有 JSON 桥，接口 `prepare_native_image_regions(pages)`，各页包含 `page_size`、`lines`、`image_infos`；输出排序后的水印指纹和逐页图片 bbox，不修改输入。

先冻结旧 Python 参考，覆盖面积/页数/正文/字号/字重/交叠阈值及重复 bbox；再实现累计预算和定向差分；阶段末一次 wheel 构建及相关安装回归。图片指纹和像素证据自动读取、噪声处理、完整页源/候选检测和默认 PDF 集成仍待。无 CUDA、全量测试或整文档质量/性能结论。

已完成：328组旧Python决策逐值对照，33组水印正例、30个含平滑图删除的页、78处白边裁剪输出；64组实际PDF页源/文档用例，含11个图片证据输入（按用例计数，不宣称独立图片数）。相同bbox的最后证据可覆盖前项，包括None及后来被水印过滤的图；面积判定规范化框，但最终保留原始框。

两个定向Rust测试通过；阶段末一次release wheel构建并隔离安装，新增native_image_regions及原native_source_lines、compatibility、smoke共四组安装套件通过。64组安装链实时读取字符并准备行，图片指纹/平滑/白边证据仍使用冻结阶段输入；字段/数值类型/顺序、所有权、并发、输入/扫描预算和错误恢复通过。支持普通JSON，最多10000页、每页4096行/每行20000码点/5000图片，有限数值abs<=1e9，复用既有JSON、10M累计Work和8MB结果预算；接受上层既有工作预算。

fmt、83模块来源审查和固定36 core＋3 paths＋1 script哈希通过。未跑全workspace、全部101组runner、未受影响渲染、全量clippy或CUDA，两个legacy完成标志保持False。日志：build/native-image-regions-rust.txt、build/native-image-regions-wheel.txt、build/native-image-regions-install-test.txt、build/native-image-regions-regression.txt。
