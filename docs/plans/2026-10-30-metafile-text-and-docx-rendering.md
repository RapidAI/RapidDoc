# WMF/EMF 文字、字体与 DOCX 图片完整链路

本轮连续实现 GDI 文字记录、字体选择/度量、字形绘制、旋转/背景/装饰、多行文字、SVG及DOCX资源输出。**含普通GDI文字的WMF/EMF已接通表格图片输出；EMF+绘图仍未移植，完整Office公开入口尚未完成。**

## 已接通链路

`DOCX part → WMF/EMF记录 → 字体/字符集/字符位置 → 路径、位图、文字混合绘制 → 原超采样/缩放/锐化 → PNG/SVG → DOCX表格HTML`

- `metafile/text.rs`：WMF TextOut/ExtTextOut、EMF ExtTextOutA/W、UTF16损坏/代理对、ANSI字符集、显式dx/dy、TA_UPDATECP、整串对齐、字体高度/旋转、opaque/clipped、glyph-index原占位行为与文字包围盒。字符数、点数及裁剪栈预算都来自原metafile-render，不新增PDF字符或公式限制。
- `metafile/model.rs`：增加具名TextCommand和TextAlignment，保持typed Rust数据从回放到后端，不经过Python或JSON命令往返。
- `metafile/fonts.rs`：字体目录惰性索引、原别名与粗斜体候选顺序、字体名Unicode casefold、256项字体实例LRU、同路径字体字节Arc共享、FreeType尺寸/字距/包围盒及灰度/单色字形掩码。字体文件不随字号重复保留整份字节；不修改系统字体。
- 默认字体复用Pillow 12.2.0原Aileron子集（12,676字节），不替换成另一套默认字体。生成器 `generate_metafile_font_data.py` 可从固定参考版本重新提取；字体作者声明No Rights Reserved，说明保存于 `licenses/AILERON.txt`。
- `metafile/text_render.rs`：字体锚点、显式字符位置的整串平移、多行默认spacing与行宽对齐、opaque背景、下划线/删除线、围绕原reference旋转、原预乘双三次旋转和裁剪。文字mask混合按Pillow 12.2.0 Paste.c执行，保留透明目标像素的RGB处理，不使用近似的source-over代替文字mask填充。
- SVG输出保留原text-anchor、dominant-baseline、font-family、旋转、逐字位置、字体装饰和背景标记，并沿用完整PNG fallback。
- 字体替代诊断按请求名和实际family/style去重，保留原记录位置、info级别及partial行为，与解析诊断共同执行原256条截断规则。
- 原生公开渲染入口的PNG使用请求DPI；SVG内部fallback继续使用原96×倍率DPI。原PNG入口此前误用了内部fallback规则，现已区分。

## 字符集兼容

没有直接把Python的`shift_jis`替换成浏览器/WHATWG的Windows-932。验证中确认它们对扩展字符、坏字节及残留输入的消费规则不同。

`generate_metafile_codecs.py`冻结原Python单字节与DBCS映射，并保存GB18030的207段四字节范围。Rust读取压缩常量并按原坏字节消费规则解码；没有运行时Python调用。DBCS数据117,971字节压缩，展开约1MiB，进程只初始化一次。单字节映射覆盖原cp125x/cp874/cp437等字符集。

## 字体引擎选择与边界

使用发布依赖 `freetype-sys = 0.20.1`，不fork或复制FreeType内核，也不加载Python的 `_imagingft` 或依赖用户手工放置DLL。本机Windows/MSVC编译其内置FreeType 2.13.2。0.23.0发布包的bundled构建在本机因错误zlib头路径失败，未给用户环境加绝对路径补丁。

参考Pillow使用FreeType 2.14.3，且本机 `raqm=False`、默认BASIC布局。因此本轮验证并移植的是**现有Windows Python的BASIC布局行为**。432组实际字体选择/度量/掩码和72组文字层像素一致，**不代表两个FreeType版本对所有字体均无差别**。Linux/macOS字体目录规则已实现，但尚未构建验收；依赖RAQM/HarfBuzz的复杂排版仍需单独补齐与平台验收，不能据本机结果宣称跨平台完整等效。

FreeType绑定许可MIT、内核选择FTL，声明保存 `licenses/FREETYPE-SYS.txt` 与 `licenses/FREETYPE.txt`。Portions of this software are copyright © 2023 The FreeType Project (https://freetype.org). All rights reserved.

## 验证

参考均由安装版本的原Python handler、Pillow及字体解析器生成；不以新算法作oracle。

- 377份构造WMF/EMF、1,779个逐记录快照，包含本轮新增150份文字/字体文件。比较完整状态、命令、当前位置、包围盒、计费、partial及异常。
- 432组字体查询：Arial、Times New Roman、SimSun及不存在字体，粗斜体、9/17/32字号、不同锚点与小数起点；实际family/style、替代原因、advance、bbox、mask尺寸/偏移及每个灰度字节一致。
- 72组文字layer：多行、顶部/底部/基线、左右居中、显式位置、透明色、背景、裁剪、不同角度和较粗装饰线；RGBA像素、SVG标记与点数计费一致。
- 28组字体规则、224组固定/随机字符集输入，以及四种DBCS全部262,144组双字节输入的序列解码摘要，与Python一致。
- 786组文件单次绘制查询：760组成功像素、26组异常行为一致；380组最终超采样/SVG/内嵌PNG参考一致。PNG压缩字节仍不要求与Pillow相同，比较解码像素、尺寸及元数据。
- 39份构造DOCX、78次VML/DrawingML图片放置，包内资源到HTML的输出参考通过；同part复用及字体替代诊断也与原入口一致。
- PNG入口200DPI的pHYs为7874，SVG内fallback保持原规则；重复字体替代诊断去重通过。
- `cargo test --locked -p rapiddoc-office` 全部48组通过；Rust格式和受影响生成器语法检查通过。没有跑PDF全量、CUDA、整体速度/内存基准或重装wheel。

生成器：`export_metafile_text_reference.py`、`export_office_metafile_reference.py`、`export_office_path_pixels.py`、`export_docx_vector_reference.py`。参考：`metafile_text.json.gz`、`metafile_geometry.json.gz`、`metafile_paths.json.gz`、`docx_vectors.json.gz`。来源/SHA见 `packaging/office-native-provenance.json`，收据见 `build/office-metafile-text-receipt.json`。

## 尚未完成

EMF+对象/绘图、原Windows原生PNG后端的auto选择、损坏矢量图的标准占位策略与独立真实矢量语料验收仍待完成。当前EMF+Only绘图仍明确报未接入，不忽略文字/图元或以占位图冒充成功。

DOCX完整正文与公开Python入口，以及DOC/XLS/XLSX/PPT/PPTX迁移仍需继续；本轮未修改生产Python解析器、模型或 `.venv`。本阶段完成不能当作整个Office迁移完成，也没有承诺整体性能目标已经达到。
