# 矢量渲染输出与 DOCX 资源接入

后续已接通普通GDI文字/字体及对应DOCX图片，见[文字与DOCX渲染链路](2026-10-30-metafile-text-and-docx-rendering.md)。下文保留本阶段原状态。

本轮连续推进超采样最终像素、PNG/SVG编码、统一渲染入口和DOCX图片接线，不把单个内部算法完成作为交付终点。**无文字/EMF+的GDI路径与位图已经能从DOCX包内读取并输出SVG到表格HTML；完整矢量渲染阶段仍未完成。**

## 链路

`DOCX part → 共享ZIP/Arc载荷 → WMF/EMF校验与GDI回放 → 原样式/裁剪/ROP → 自适应栅格与SVG → data URI → 表格HTML`

- `metafile/filters.rs`：按 Pillow 12.2.0 Resample.c 实现Lanczos系数、22位定点两次采样、预乘alpha/还原；按BoxBlur.c和UnsharpMask.c接入原固定参数 `radius=.6, percent=100, threshold=2`。这是原后端使用的固定锐化参数，不新增图像质量配置。
- `raster.rs`：纯路径按原预算选择4×/2×/1×，含图片固定1×；超采样后Lanczos缩小和锐化与原顺序一致。没有把固定1倍结果冒充原最终输出。
- `output.rs`：生成RGBA PNG，含fallback输出的DPI元数据；SVG保留路径、stroke/fill属性、裁剪定义/嵌套顺序、图片affine与源裁剪、内嵌PNG和不可见fallback metadata。复杂ROP或copy/and以外裁剪按原规则使用整图PNG包装。
- SVG沿用原64MiB消费者预算，fallback倍率选择与原相同；普通SVG先生成最多2×fallback，ROP整图包装使用原最终栅格路径。
- `metafile::render/office_image`：统一纯Rust文件入口，返回媒体类型、尺寸、partial和诊断。Office默认200DPI、先SVG、仅资源错误重试PNG。无需PNG/JSON在Python和Rust之间往返。
- `docx/table_images.rs`：按MIME或文件魔数识别的WMF/EMF开始调用原生入口；同一part在表格重复出现时复用SVG结果。普通位图仍保持原表格路径的原始载荷。
- `VectorReport`和共享资源入口保留每个已渲染part的尺寸、partial和诊断；重复资源不重复记录。尚未完成的公开解析器将用这些结果输出进度/诊断，当前未虚构生产日志。
- 端到端参考发现已有VML alt移植差异：原Mammoth Office namespace表不定义`o`前缀，因此namespaced `o:title`未作为alt输出。Rust对齐此分支；DrawingML docPr descr/title路径继续保留alt，测试同时覆盖两种放置方式。

PNG采用已有生态依赖 `png = 0.18.1` 编码，**压缩字节不要求和Pillow相同**。SVG比较保留所有非图片标记原文，PNG替换为尺寸、RGBA像素SHA和pHYs元数据后严格比较。这样区分无损压缩差异与解析效果差异，不拿base64不同当作识别下降，也不隐藏像素差异。

## 验证

- 6组独立Lanczos/锐化图像，覆盖上下采样、单轴不变、透明像素和1像素边界，与原Pillow逐字节一致。
- 233组已有成功的构造文件分别验证最终自适应渲染和资源计费、SVG结构、内嵌PNG像素/尺寸/DPI；最终PNG解码后与原最终像素严格一致。12组SVG浮点格式边界一致。
- 15份构造DOCX，30次VML/DrawingML矢量图放置：ZIP资源→原生渲染→表格HTML，对比原DocxConverter的Mammoth完整上下文路径。除了PNG压缩字节外，HTML与SVG标记、嵌入图片RGBA像素和元数据一致。
- 重复part的渲染复用，以及尺寸/partial/诊断与原renderer一致；包含未知ROP和复杂裁剪回退案例。
- Office crate的43组测试通过；最后报告接口调整及参考补充只重跑受影响8组，通过。Rust格式与参考脚本语法检查通过。未跑PDF全量、CUDA、Office总体性能或新的wheel安装。

参考生成器 `scripts/migration/export_office_path_pixels.py` 和 `export_docx_vector_reference.py`，参考资源 `metafile_paths.json.gz` 与 `docx_vectors.json.gz`。来源与SHA见 `packaging/office-native-provenance.json`，收据见 `build/office-metafile-output-receipt.json`。

## 未完成边界

- 文字/字体选择、字形度量、旋转文字、背景与装饰，以及EMF+对象/绘图尚未接入。遇到原支持但Rust尚未完成的记录仍明确返回Unsupported，**不会跳过文字、丢预览或新增占位图来冒充完成**。
- 原Python对损坏矢量图的标准占位图及完整诊断策略仍需在完整Office资源层接回；当前错误保持显式传播，未宣称损坏图片行为全面一致。
- 新PNG编码字节不同；没有承诺SVG原始字节或Office文件结果全量一致。当前语料为构造图像，独立真实WMF/EMF仍待验收。
- DOCX完整正文/公开Python兼容入口，以及DOC/XLS/XLSX/PPT/PPTX仍未贯通；`.venv`未更新。当前进展不能当作完整Office迁移完成，速度/内存也未重新基准。

后续继续按完整矢量资源阶段推进，重点是字体/文字与EMF+，随后统一做真实Office资源验收；不再把内部函数拆成多个用户确认点。
