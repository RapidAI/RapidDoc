# Office 图片回退与 DOCX 包级容错

本阶段完成损坏矢量图片在 Office 层的占位回退、DOCX 实际表格输出接线和包级修复。仍未把完整 DOCX 正文组装或公开 Office API 标为完成。

## 图片回退

`rust/rapiddoc-office/src/image.rs` 移植原 `office/image.py` 的矢量格式判断、EMU 尺寸提示、标准占位图和 SVG→PNG→占位选择。元文件渲染入口仍严格返回错误；只有 Office 调用边界选择占位，保留成功图片的 partial/diagnostics，以及失败原因。

标准占位图按原 320×180 尺寸、边框/背景、三行文案、字体逐级缩小、换行/居中绘制和默认 JPEG 参数生成。字体尝试 DejaVuSans.ttf、Arial.ttf、LiberationSans-Regular.ttf，最后使用原 Pillow 默认字体子集和固定 10px；不采用元文件字体的家族替代规则。生成结果由进程共享缓存；失败不缓存，重复坏图片部件也不重复渲染或追加诊断。

DOCX `table_images::Images` 已使用该链路。坏矢量图不会因 Rust 渲染错误中断整表；VML 和 DrawingML 保留文字及替代说明。普通表格位图继续保留原始载荷，不擅自转码。`Images::fallbacks()` / `Shared::image_fallbacks()` 提供部件、错误分类及原因。

普通正文位图转码、SVG 栅格化及完整正文图片调度仍需在正文阶段完成；不能将当前表格图片能力等同于完整正文转换。

## 包级容错

`docx/normalize.rs` 移植原 `package_normalizer.py` / `opc.py` 的规则，并由 `Source::open` 实际调用：

- 从根关系构建可达图。关系流损坏继续报错；关系 XML 无法解析时将图标记为不完整，不据不完整图丢弃其他坏成员。
- 逐成员验证解压及 CRC。只有原规则允许的 `word/media/`、`word/embeddings/` 或可信可达图之外的坏成员可以跳过；正文、样式等关键坏成员继续失败。
- 删除缺失、非法、已跳过成员的内部关系；保留外部关系，关系类型及目标解析规则沿原实现。
- 沿 officeDocument 及已知 Office 关系补充/修正 Content Types Override，不把所有 `.xml` 一律当作正文。碰到不能解析的相关 XML 时不做不完整的类型修复。
- 修补基于共享 Package 覆盖层。CRC 验证以流方式读取，不持有所有解压成员，也不重打包或复制整个 ZIP。这里只减少中间缓冲；没有运行速度/峰值内存基准，不据此宣称具体收益。
- XML 覆盖写回 UTF-8 时同步更新声明，防止原 UTF-16/旧编码声明与新字节不一致。

原生 Strict namespace 读取能力继续保留，无须为 python-docx 改写整个包的 namespace。此前已支持的“主文档位于 ZIP 根目录”也继续保留；关系修复在这个场景使用实际源 part 解析，避免照搬 Python 历史辅助函数的相对路径缺口。没有顺带修改 Python 实现。

`Source::normalization` 返回跳过成员、移除关系和修复类型的报告；`Package` 的其他格式默认读取策略不变。

## 验证

- 35 组占位图：多个尺寸、空文案和多行文案，每个 RGBA 字节与原 Pillow 一致；标准 JPEG 解码像素一致，并验证共享缓存复用。
- 259 组矢量回退策略、7 组 EMU 尺寸提示参考。
- 32 份包级参考：CRC、关键/可降级/不可达成员、关系图不完整、缺失或越界目标、循环关系、缺失/错误 Content Types、UTF-16/UTF-16BE/windows-1252。25 份成功修复、7 份保留原致命损坏行为，存活成员、关系及类型映射一致。
- DOCX 实际表格图片扩展为 **298 份包、596 次放置**：原 251 份成功渲染加 47 份损坏/不支持图片；HTML、图片、诊断和同 part 缓存通过。47 份回退保留原 JPEG data URI 和整表前后文字。
- 阶段 Office crate **52 组测试通过**；最终缓存/共用检测整理后只复查受影响测试。未运行全量 PDF/CUDA，也未替换 `.venv` 中的 rapid_doc_rust。

## 剩余

完整 DOCX 正文编排仍需继续移植：列表/目录及锚点、分节分页、正文图片、文本框、图表、页眉页脚，以及统一结果输出；其他 Office 转换器和公开调用入口仍未完成。图片及字体目前仍是 Windows/BASIC 的构造参考验收，独立真实矢量语料、跨平台/复杂排版和少见损坏编码行为的风险沿上一阶段记录保留。
