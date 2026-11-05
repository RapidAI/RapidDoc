# 原生 DIB 解码与位图绘制

后续已接通路径与位图的混合单次绘制及最终裁剪，见[混合绘制记录](2026-10-30-metafile-path-raster.md)。下文保留本阶段交付时的状态。

本阶段延续[位图回放](2026-10-30-metafile-bitmap-replay.md)，把 DIB 命令推进到像素输出。原运行时参考为本机 metafile-render 0.3.0 与 Pillow 12.2.0，Rust 运行期间不调用 Python。

## 实现

- `metafile/dib_decode.rs`：按 Pillow 的 BMP 插件及 unpack 规则实现 CORE/INFO/V2/V3/OS2/V4/V5 头、1/4/8 位调色板、16/24/32 位颜色、支持的 bitfields、RLE4/RLE8、上下行方向与透明度分派。保留原校验/损坏输入行为；例如不把 compression=6 自动解释成原 Pillow 不支持的普通 BMP，但原 AC_SRC_ALPHA 分支仍可读取它。
- 保留非规范输入兼容：palette 可跨 header/bits 读取，灰度 palette 的原模式切换，缺失 palette 项的黑色，RLE 越行截断、delta 空隙、绝对模式/奇数字节对齐、原 RLE4 奇数计数行为，以及16/24位 BMP 中被忽略的 alpha mask。没有给特定文档打补丁。
- 不拼装第二份完整 BMP；header 与 bits 保持独立，掩码/调色板按原文件偏移读取。RGBA 输出由统一类型验证尺寸/长度。
- `metafile/sampling.rs`：保留 Pillow 最近邻的纯缩放浮点累加、通用16.16定点和大变换浮点分支；双三次保留原核、边缘延拓、像素中心与取整。RGBA 先预乘、插值再还原，保留透明边缘处理。源裁剪、镜像、constant alpha 和目标四角变换串联。
- `metafile/composite.rs`：补齐原七位定点 source-over 及 ROP2/ROP3 分派，未知 ROP 沿用原回退。
- `metafile/image_session.rs`：每次渲染独立64MiB LRU解码缓存，预算沿用原实现。命中返回不可变 Arc，后续裁剪/透明度处理只修改私有图片，避免缓存污染和原缓存命中时的完整 RGBA 副本。
- 内部 `render_bitmaps` 接通文件回放后的纯位图文档：文档坐标映射、透明白画布、解码/采样、按命令顺序合成。含路径或裁剪的文档明确返回未接入，不静默删除内容。该入口不是完整 renderer，也未接公开 DOCX API。

## 验证

参考脚本直接执行原解码器、Pillow变换、合成和栅格后端，未以新算法充当参考。

- 834组 DIB 解码用例：RGBA 字节/尺寸或错误类型一致，涵盖支持头/位深/掩码、调色板及RLE兼容边界。
- 428组仿射采样：包含旋转/倾斜/翻转/缩放/平移、退化/大坐标和透明图片；输出逐字节一致。
- 100组解码→裁剪/镜像→constant alpha→采样链路，输出逐字节一致。
- 65,536对源/目标 alpha，覆盖全部alpha组合，source-over逐字节一致；27组ROP2/ROP3分派一致。上一阶段的14种逻辑操作参考继续通过。
- 43份构造的纯位图 WMF/EMF：原 Python 完整 `render_pillow` 与 Rust 的文件回放→画布输出像素一致，画布16×12。**这些不是独立真实 Office 矢量文件，也不覆盖混合图形。**
- 7次缓存请求对比原 Python 的命中/未命中、内存计费、LRU淘汰和图片字节；清理后计费归零。
- Office crate 的36组测试通过；补充16/24位alpha掩码边界后只重跑受影响的8组位图测试，通过。Rust格式及参考脚本语法检查通过。未运行PDF全量、CUDA或整体性能测试。

参考：`metafile_bitmap.json.gz`、`metafile_bitmap_pixels.json.gz`；生成器：`export_office_bitmap_reference.py`、`export_office_bitmap_pixels.py`。SHA/来源见 `packaging/office-native-provenance.json`，收据见 `build/office-metafile-bitmap-render-receipt.json`。

Pillow的C内核依据固定版本的 [Geometry.c](https://github.com/python-pillow/Pillow/blob/12.2.0/src/libImaging/Geometry.c)、[AlphaComposite.c](https://github.com/python-pillow/Pillow/blob/12.2.0/src/libImaging/AlphaComposite.c)、Convert.c和Unpack.c移植。MIT-CMU声明保存在 `rust/rapiddoc-office/licenses/PILLOW.txt`，区别于 metafile-render 的MIT声明。

## 剩余

路径填充/描边与最终几何裁剪、文字/字体、EMF+绘图及其标准化PNG输入、SVG/PNG编码和资源输出仍待接通。混合文档尚不能渲染；当前纯位图像素验证不代表完整WMF/EMF效果或编码字节验收。完整DOCX正文与公开Office入口尚未完成，`.venv` 未更新；未宣称整体速度/内存目标已达成。
