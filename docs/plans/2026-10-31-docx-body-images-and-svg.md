# DOCX 正文图片与 SVG

本阶段把正文图片加载、转码、引用去重、替代说明及浮动/行内顺序接入已经完成的正文编排。不是只实现独立编码函数；原 Python 完整 `DocxConverter.convert()` 的 pages 已包含本阶段图片并进行对比。**图表、剩余附注资源和公开 Office/Python API 仍待完成。**

## 接入行为

- 先排除公式预览图，再按同一顶层块中的 relationship ID 去重；不同段落中的同一图片仍分别输出。载荷和转码结果按部件路径缓存，复用 Arc，避免每次放置都解压/解码/转码。
- 浮动图片所在段落先输出文字，再输出图片；行内图片沿原流程先图片后段落。包含分节的原段落流程不另改顺序。
- DrawingML 读取 `wp:docPr/@descr`；VML 读取 `v:shape/@alt`，按原规则去除首尾空白。关系缺失或普通损坏图片按原策略跳过，并保留诊断；已识别但尚未连接的少见图片格式明确 Unsupported，避免以“不认识”误当损坏后静默丢弃。
- 元文件共用先前的 SVG/PNG/占位链路；正文及表格共用部件渲染缓存、partial、矢量诊断和失败报告。普通表格位图继续保留原始载荷，不套用正文转码。

## 位图模式与引擎

`image/raster.rs` 保留原 Pillow 模式选择：RGB 等普通图片转 JPEG；RGBA/LA 与带透明信息的 P 模式转 PNG。不能以“解码器总是返回 RGBA”为理由一律输出 PNG。

- PNG/JPEG 复用此前验证的 PNG/Pillow 转换和 libjpeg-turbo。PNG 16 位灰度按 Pillow 规则夹至 255，非按高字节近似；透明色与 ICO 载体的 mode/info 差别保留。PNG 同一次解析返回像素与原模式信息，避免为选择输出格式再次解析。
- 发布包 `image 0.25.10` 处理已测 BMP、GIF、WebP、TGA、PNM、ICO 路径，关闭不需要的默认格式和 rayon；GIF 保留第一帧及原透明声明语义。补充 Pillow 支持的单色 TGA 和灰度 PFM 边界。
- `image/tiff.rs` 使用发布 `tiff 0.11.3` 的压缩/strip 解码，补充 Pillow 的 LA、整型/浮点灰度、调色板、CMYK、预乘 alpha 和 8 种方向处理。调色板 TIFF 的压缩解码通过一个只覆盖 photometric SHORT 的只读视图复用原解码器，源载荷不复制，也没有重写压缩算法。
- 对比发现通用 TIFF JPEG 解码与 Pillow 不同，已将 JPEG 压缩 TIFF 的 tables/strips 拼成合法 JPEG 交给已有 libjpeg-turbo；RGB、L、CMYK 和多 strip 已测像素一致。没有放宽像素容差。YCbCr JPEG TIFF 路径有实现但尚无通过的独立参考，不能据此声明该分支已验收。
- 普通照片不继承元文件 8,192 单边/16M 像素限制；沿现有 Pillow 默认 decompression bomb 错误阈值 178,956,970 像素。9,001×1 的 RGB/RGBA 普通图片已验证能输出。元文件原有限制继续只作用于元文件。

## SVG

`image/svg.rs` 移植 `_svg_raster.py` 的固有尺寸、viewBox 回退、长度单位、最小长边 512、单边/总像素收缩、DTD/损坏输入拒绝及全透明回退规则。

固定发布 `resvg 0.48.0`，Cargo.lock 中 `usvg 0.48.1`、`tiny-skia 0.12.0`，与下载的已安装 `resvg_py 0.5.0` sdist/Cargo.lock 相同。运行阶段直接调用 Rust 引擎，未使用 Python 绑定或复制 SVG 绘制内核。原 `skip_system_fonts=True` 使用空字体库，未转路径的文字继续遵循原行为；渲染质量、DPI、尺寸 FitTo 和默认值对齐原适配层。

保留原 resvg 默认 href resolver：不发起 HTTP 请求，但本地路径解析能力与原绑定一致；不把原注释中的“完全不加载外部资源”当作已经验证的事实。复杂嵌入图片与少见 SVG 仍需扩大真实语料验收。

## 验证与交付状态

- **200 组位图参考**：原 Pillow 输出 MIME、尺寸和每个 RGBA 字节。覆盖 1/L/LA/P/RGB/RGBA/整数/浮点/CMYK 的可保存模式、透明声明、压缩 TIFF、8 种方向、JPEG TIFF 及多 strip、畸形输入、超元文件单边的普通图片。
- **62 组 SVG 参考**：不同尺寸/单位/viewBox、路径、圆、渐变、透明及字体缺失、三种 XML 编码、损坏/DTD/外来根节点，输出尺寸和逐像素 SHA256 与原 Python 一致。
- **254 份 DOCX 完整输出参考**：253 份成功、1 份保留损坏输入错误；261 个逻辑页，包含 148 个正文图片块。覆盖行内/浮动顺序、重复引用、不同段落复用、VML 替代说明、元文件及占位、SVG/空图回退，并继续覆盖先前正文、目录、表格、文本框和页眉页脚。
- 阶段 Office crate **56 组测试通过**；随后 PNG 解析共享与 JPEG TIFF 补充只复查受影响的元文件/位图/SVG/正文测试，最后 TIFF 补充 200 组像素参考通过。`cargo fmt --check` 通过。没有全量 PDF/CUDA 或性能/峰值内存基准。
- 生成/验证脚本：`export_office_raster_reference.py`、`export_office_svg_reference.py`、扩展的 `export_docx_body_reference.py`、`test_office_native.ps1`。新增引擎及依赖许可声明由 `collect_office_image_notices.py` 记录于 `packaging/office-image-dependency-notices.json`，74 个已解析依赖均保留声明，适配层许可另保存 `licenses/RESVG-PY.txt`。
- 未重建生产 wheel，未替换 `.venv` 中的 rapid_doc_rust；现有 PDF 默认入口未变。

## 剩余边界

DOCX 主要剩图表（含嵌入工作簿回退）、表格附注/嵌入样式映射等资源、统一结果导出和 Python 原 API 接入。图片方面仍有少见载体/编码未验收，例如 palette BigTIFF、部分高位深/多平面 TIFF、PCX/PSD/JPEG2000 等；不能将已测常用模式称为完整 Pillow 替代。原图片库版本、跨平台字体/复杂排版及真实客户语料风险继续保留；本轮没有以简化算法换取速度，也没有宣称尚未测量的性能收益。
