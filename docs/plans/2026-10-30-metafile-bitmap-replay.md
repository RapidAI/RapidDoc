# WMF/EMF 位图记录、透明度和逻辑合成

后续进展：通用DIB解码、采样和纯位图文档绘制已接通，见[位图绘制记录](2026-10-30-metafile-bitmap-rendering.md)。下文保留本阶段交付时的边界。

沿用本机 metafile-render 0.3.0 的原算法，继续上一阶段几何回放。此阶段完成位图命令及部分像素内核，**尚未完成完整图像渲染，不接入公开 DOCX 入口**。

## 改动

- `metafile/model.rs` 的命令集合支持路径与 DIB 图片，像素载荷使用不可变 Arc；记录源位置、裁剪栈、源矩形、目标四角、stretch mode、ROP 和 constant alpha 都保留。
- `metafile/bitmap.rs` 接入 WMF StretchDIB 与 EMF BitBlt、StretchBlt、SetDIBitsToDevice、StretchDIBits、AlphaBlend；保留原空载荷/未知 ROP 诊断、异常先后顺序和 signed extent。坐标相加使用 i64，避免 Python 整数相加移植后发生 i32 溢出。
- DIB CORE/INFO 尺寸、调色板/bitfield 掩码偏移、像素行对齐及截断校验、解码内存计费按原规则移植。原 Python 的 16M 像素和 128MiB 位图预算未扩大或另加限制；压缩 DIB 校验不提前冒充完整解码。
- `Playback` 对位图计费并据目标四角推导无显式尺寸文件的画布。普通路径的序列化形状、状态及计费不变。
- 32-bit AC_SRC_ALPHA 的 BGRA 预乘透明度还原：保留上下行方向、alpha=0 的原通道值和超出预乘范围的截断行为。此 helper 的调用前提由后续解码分派控制，并非所有 32-bit DIB 都默认含 alpha。
- 源矩形裁剪、镜像和可见交集到目标平行四边形的映射已实现。
- `metafile/composite.rs` 实现14种逐通道逻辑操作以及原 Pillow masked paste 的 alpha/取整规则。尚未分派到最终画布后端；普通 source-over 合成仍待接入。

## 验证

参考生成器直接调用安装版本的 GDI handler、DIB 校验、raw-alpha、裁剪及合成实现，不依赖新写算法作为 oracle。

- 原几何参考扩展为227份构造文件、896个逐记录快照；新增61份位图记录文件，覆盖全部六种记录、损坏偏移/载荷、ROP、负尺寸与坐标相加边界。
- 75组 DIB 载荷覆盖不同 bit depth/compression、CORE、掩码、损坏尺寸和截断。raw-alpha 对全部256种 alpha、有效/无效预乘通道和行方向逐字节比较；这不代表通用压缩 DIB 已解码。
- 9组裁剪比较实际像素序列、镜像、交集比例和目标坐标。
- 14种逻辑操作各比较1,792个 RGBA 像素，结果逐字节一致。
- `cargo test --locked -p rapiddoc-office`：31组通过；Rust格式和两份参考脚本语法检查通过。未运行 PDF 全量、CUDA或整体性能比较。

参考文件为 `tests/fixtures/office_rust_reference_068/metafile_geometry.json.gz` 与 `metafile_bitmap.json.gz`；来源/SHA 更新于 `packaging/office-native-provenance.json`，验证记录在 `build/office-metafile-bitmap-receipt.json`。

## 剩余与边界

通用 DIB/PNG 解码、nearest/bicubic 仿射采样、source-over、最终裁剪/栅格与 SVG 后端、文字字体及 EMF+绘图仍待完成。当前像素测试是独立内核验证，**不是整张 WMF/EMF 渲染效果验收**；尚无独立真实矢量文件效果验收。

DOCX 资源流程仍明确报告未接入矢量渲染，不新增占位图掩盖未实现的原受支持内容。完整正文和公开 Office API 仍待贯通，`.venv` 未更新。
