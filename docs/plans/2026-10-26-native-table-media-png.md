# Native table media PNG Implementation Plan

**Goal:** 将旧表内媒体占位符编码为PNG data URI，并保留页级失败回退。

**Architecture:** 在table_block/images.rs实现自有PNG裁剪/旋转/替换及按页事务，复用Pixels模式/ICC/tRNS处理。PDF入口按旧scale=2、长边3500上限每个有媒体的页只渲染一次；默认媒体bbox为归一化坐标，可显式传page_sizes衔接上一阶段point框。编码使用Rust PNG，压缩字节可不同，验证原模式/尺寸/像素及HTML其他部分。

**Tech Stack:** Rust image/PDFium、已有Pixels、PyO3、普通JSON。

1. 独立旧Python冻结合成PNG/真实PDF渲染、裁剪旋转像素、HTML、部分成功后失败和元数据清理。
2. 复用Pixels裁图并增加保模式旋转，移植crop floor/ceil、页内剩余文字回退和全局占位符替换。
3. 增加受旧长边限制的RGB PDF渲染入口，保持原render_pdf_page行为；无媒体页不读PDF不渲染。
4. 暴露materialize_native_table_media_page和materialize_native_table_media，返回独立blocks/pages；资源异常明确传播。
5. Rust/安装wheel差分与像素对照、相关回归、fmt/clippy/CUDA编译，更新审查/归属/文档。完整来源/候选检测和默认页级接入继续后续。

执行记录：五项完成。232项Rust workspace测试、fmt、严格clippy与CUDA feature编译通过，release wheel已重新构建并隔离安装。182单页/502引用和45真实PDF/125引用均通过逐值/类型及解码模式、尺寸、像素、ICC/tRNS对照；包含部分成功后的页内回退、point刻度、所有权、并发、限额与错误后恢复。通用PDF渲染的18 RGBA＋18 RGB组重新对照旧像素全部一致，使用--existing-corpus避免改写冻结PDF。

另10组相关安装回归全部通过：完整块事务、注释、块准备、整表物化、页面资产、内联资产、表格资产、model_document、compatibility和smoke；共11组安装套件及独立36像素对照，未全跑94组runner。回归日志build/native-table-media-png-regression.txt。model_document的冻结提供器与真实CPU模型烟测不能替代完整原生PDF效果证明。

日志：build/native-table-media-png-install-test.txt、build/native-table-media-png-render-regression.txt、build/native-table-media-png-wheel.txt、build/rust-current-*.txt。76模块审查与固定36 core＋3 paths＋1 scripts源码哈希保持；PNG压缩字节/URI哈希名差异已声明。默认完成标志仍False，未作整文档效果、速度、峰值内存或GPU实机结论。
