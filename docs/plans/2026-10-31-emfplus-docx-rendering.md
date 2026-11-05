# EMF+ 回放与 DOCX 矢量图片接入

本阶段接通 `DOCX part → EMF+ Only → Rust typed commands → PNG/SVG → 表格 HTML`。不是只增加对象解析辅助函数；对象、状态、实际绘制、最终编码、重复部件缓存与 DOCX 输出均有原 Python 参考验证。

## 实现

- `rust/rapiddoc-office/src/metafile/emfplus/` 按原 metafile-render 0.3.0 移植有界游标、对象、图片和回放。覆盖对象槽替换、跨记录重组、路径/RLE 类型、画刷、画笔选项、字体、StringFormat、ImageAttributes。
- 状态包含 Save/Restore、容器、世界/页面/单位变换、矩形/路径裁剪与偏移。GetDC 的 GDI 状态和栈仅在临时窗口内生效，下一条 EMF+ 记录恢复；Dual 仍沿原 GDI 路线。
- 绘制接通 Clear、矩形、椭圆、弧/饼、线/多边形/贝塞尔、对象路径、位图、DrawString 和 DriverString，复用原有路径、字体和混合绘制后端。Clear 保留对已有透明度的清除行为及带裁剪/非首条操作时的 SVG 栅格 fallback。
- 图片支持原 24/32 位 BGR/BGRX/BGRA/预乘 BGRa、负 stride、PNG/JPEG，包含透明色和 16 位灰度到 RGBA 的 Pillow 转换规则。PNG 标准化载荷由 Arc 共享，同一图片多次绘制不重复收取对象解码预算。
- 已知原不支持的曲线、区域、效果、复杂文字等沿原 warning/partial/停止规则处理，没有自行以简化绘图替代。Header/EOF、被中断对象、错误引用与类型、越界数据保留错误分类；停止绘制后仍验证记录边界。
- 普通对象直接借用记录字节；只有跨记录对象需要拼接。生产路径不调用 Python，不以 JSON 传递绘图指令。

## JPEG 引擎与构建

纯 Rust `jpeg-decoder 0.3.2` 在噪声 JPEG 对比中出现 1 色阶差异，已移除，未放宽像素容差。采用发布包 `turbojpeg 1.5.1 / turbojpeg-sys 1.2.0`，静态编译其 libjpeg-turbo 3.1.0；CMYK 按 Pillow Convert.c 转换。参考 Pillow 使用 libjpeg-turbo 3.1.4.1，本次样例逐像素一致，不代表所有版本/损坏 JPEG 均等价。

本机 CMake 4.4.4、NASM 3.01、MSVC 构建，已核对 `WITH_SIMD_INT=1`。`.cargo/config.toml` 指向小型 CMake toolchain，避免 libjpeg-turbo 默认强制静态 CRT 与 Rust 默认动态 CRT 冲突；没有屏蔽链接警告。保留显式 `+crt-static` 选择。使用发布依赖，没有复制或修改 JPEG 内核。

开发验证入口：`scripts/migration/test_office_native.ps1`；可追加 `--test office_emfplus` 仅跑受影响测试。CMake/NASM 仅为构建工具，最终运行不需要 Python JPEG 模块、CMake/NASM 或手工 JPEG DLL。NASM 未配置时脚本提示性能差异，不能把无 SIMD 构建用作速度验收。

This software is based in part on the work of the Independent JPEG Group.
绑定、libjpeg-turbo、IJG 声明保存于 Office crate 的 licenses 目录。

## 验证

参考脚本 `scripts/migration/export_emfplus_reference.py` 调用已安装的原 metafile-render/Pillow，生成固定 fixtures；不以新 Rust 实现生成期望结果。

- **255 份 EMF+ 构造文件**：212 份成功回放，43 份错误；严格比较命令、几何、文字布局、诊断、partial、位置及错误分类。
- **424 组 PNG/SVG 输出查询**：比较每个 RGBA 字节；SVG 非图片标记严格一致，嵌入 PNG 按解码像素、尺寸和 pHYs 比较。包含渐进 JPEG、不同质量/色度采样、CMYK、透明色、16 位灰度、原生位图与裁剪。
- 非图片输入预算与 Python 精确比较；图片按实际标准化 PNG 字节与 RGBA 计费，并验证重复绘制不重复收取。**编码器不同导致 PNG 字节数不同，极靠近累计字节阈值时触发位置可能不同**；没有伪称 PNG 压缩字节或此类边缘触发位置相同，也没有为 Rust 新增限制。像素、对象、状态栈等上限沿原 metafile-render，和此前取消的额外 PDF 字符/公式限制无关。
- **251 份 DOCX，502 次图片放置**：39 份原 GDI 参考及 212 份 EMF+；同一部件同时以 VML/DrawingML 放置，比较原 Mammoth 表格 HTML、嵌入 SVG/像素与诊断，验证同 part 缓存复用。
- 保留原 **377 份 WMF/EMF、1,779 个记录快照**。旧的“EMF+ 一律 Unsupported”临时断言已替换成原 Python 实际结果。
- Office crate **49 组阶段回归通过**；最后新增样例/借用调整后复查受影响的 EMF+、元文件与路径/DOCX 测试。没有全量 PDF/CUDA 回归或性能基准。

## 剩余工作与边界

- 完整 DOCX 正文组装、损坏矢量图在 Office 层的原占位策略、DOC/XLS/XLSX/PPT/PPTX 转换器及公开 Office API 尚未完成。此次没有替换 `.venv` 的 rapid_doc_rust 安装包；现有 PDF 入口不受影响。
- 当前为 Windows/BASIC 字体环境的构造参考验收；独立真实矢量语料、其他平台、RAQM/复杂排版仍需验收。字体和 JPEG 引擎版本不同的风险继续保留。
- EMF+ 压缩图片的可绘制格式按原实现限定 PNG/JPEG；已验证 GIF/BMP/TIFF/WebP 的原诊断跳过行为。Pillow 可识别的其他少见格式及损坏文件识别细节尚未完整逐一验收，不能将本轮结果解释为整个 Pillow 格式识别器的替代。
