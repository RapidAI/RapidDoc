# WMF/EMF 路径、裁剪及混合栅格化

后续已接通最终超采样过滤、PNG/SVG输出和受支持GDI图片的DOCX表格资源流程，见[输出与DOCX资源接入](2026-10-30-metafile-output-and-docx-resources.md)。下文保留本阶段原状态。

接续[位图绘制](2026-10-30-metafile-bitmap-rendering.md)，本阶段接通路径和位图同画布绘制。依据本机 metafile-render 0.3.0、pyclipper 1.4.0 和 Pillow 12.2.0 行为迁移，运行时不调用 Python。

## 已实现

- `metafile/clipper.rs`：浮点坐标按原256倍率和 ties-even 量化；填充规则、孔洞、路径偏移、join/cap、miter limit 和 arc tolerance 保持原值。
- 采用固定发布依赖 `clipper-sys = 0.8.0` 内置的 Clipper 6.4.2。没有切换 Clipper2，也没有重写简化多边形算法。发布包的 offset 包装会先对开放路径做布尔操作，与原 Python 直接调用不同，因此 `native/clipper_bridge.cpp` 只适配原 Clipper API 并按原 PolyTree 深度优先顺序回传轮廓。
- 适配层保留未修改的6.4.2声明头，C++实现仍由发布依赖提供，不复制或二开内核。异常在C++边界转为错误；Rust点数预算超出通过回调返回，原始累加值保留。
- `metafile/mask.rs`：移植 Pillow Draw.c 多边形填充的整型坐标转换、float32边交点、水平边合并、端点重复、断角连接、扫描行及边界取整。没有换成另一套抗锯齿算法。
- `metafile/paths.rs`：evenodd/nonzero 填充、孔洞覆盖、连续虚线分段、开放/闭合描边、线帽/连接、hatch画刷、fill/stroke alpha、ROP之前的layer合成，以及copy/and/or/xor/diff裁剪序列。
- `image_session.rs`：图片和裁剪mask共用原64MiB预算与LRU顺序。不可变Arc命中不复制整张mask；clip键保存原几何、矩阵和尺寸，不依赖JSON序列化。
- `metafile/raster.rs::render_once`：原单次渲染流程，支持路径/位图/裁剪混合命令、文档坐标映射、透明白画布、绘制顺序、ROP2/ROP3及原工作量校验。显式倍率1和2已验证。
- 修复源裁剪为空时多执行一次clip的移植差异，和原 Python 一样直接返回空layer；避免改变缓存命中与计费。

构建增加C++适配编译（`cc`），当前Windows/MSVC已编译运行验证；其他目标尚未构建。Clipper内核为Boost许可，包装为ISC许可，声明分别保存于 `licenses/CLIPPER.txt` 和 `CLIPPER-SYS.txt`；Pillow扫描行移植沿用 `licenses/PILLOW.txt`。

## 验证

参考生成器 `scripts/migration/export_office_path_pixels.py` 直接调用原 Python backend/Pillow/pyclipper，保存 `tests/fixtures/office_rust_reference_068/metafile_paths.json.gz`。

- 305组Pillow多边形边界，包括固定退化/水平边案例与确定随机轮廓，像素一致。
- 164组复合填充、216组线帽/连接/宽度/虚线描边、32组hatch、25组裁剪栈和72组完整路径layer，像素与点数计费一致。
- 24组虚线片段结构一致，坐标按1e-12相对容差比较；非数值字段严格比较。
- 492组文件渲染查询：466组成功输出像素一致，26组异常类型一致。包含原几何文件参考与构造的混合位图/路径/SaveDC/RestoreDC/裁剪记录；每份按1倍和2倍显式倍率运行。成功案例同时比较flatten计费、缓存命中/未命中和内存计费。
- 6组原路径预算边界，包含跨C++回调中止，错误和预算累加值一致。
- Office crate全部40组测试通过；Rust格式、生成器语法检查通过。未运行PDF全量、CUDA或整体性能测试。

来源与SHA见 `packaging/office-native-provenance.json`，阶段收据见 `build/office-metafile-path-raster-receipt.json`。

## 尚未交付

当前是原 `_render_pillow_once` 对应的单次渲染层；纯矢量文档的自适应倍率选择之后，最终Lanczos缩小和UnsharpMask尚待实现，不能把固定1倍输出作为默认渲染结果。文字/字体、EMF+绘图、标准化PNG输入、SVG/PNG编码、诊断及资源输出还需继续接通。

现有语料是构造矢量文件，尚未独立真实WMF/EMF效果验收，未宣称整张图或编码字节全面兼容。DOCX公开入口、完整正文和其他Office格式仍待贯通，`.venv`未更新。
