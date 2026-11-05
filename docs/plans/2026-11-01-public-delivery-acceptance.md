# 七种格式公开接口验收

本轮承接 DOC 收尾后的第 1 项：核对已安装的 PDF、DOC、DOCX、XLS、XLSX、PPT、PPTX 统一入口、默认参数、输入形式、导出、资源、writer/落盘和常见异常。性能及大文件内存趋势另作独立测量，不从正确性检查的耗时推断。

## 范围与方法

脚本 `scripts/migration/accept_public_delivery.py` 顺序启动独立 Python/Rust worker，交替两端执行顺序，冻结输入 SHA256 和已安装 `_native.pyd` 指纹。通过 `RapidDoc()` 默认配置调用，不手动指定 DLL、模型文件或算法阈值。

- 六种 Office 格式：项目各 1 份完整真实文档。逐项覆盖 path、bytes、BytesIO、批量、六种格式混合批量、自定义资源前缀、data URI、独立 image/md writer、磁盘输出、关闭 JSON 转储，以及 mm_markdown、nlp_markdown、content_list、content_list_v2。
- PDF：扫描件 small_ocr、论文模板、含印章各检查第 0–1 页，以及第 1 页配合 `pdf_pages_batch=1` 的窗口。页码结束值按公开接口含末页语义检查；总计 9 个选定页面调用/端，不是对整份 PDF 或全语料的新回归。
- 对照完整 middle、Markdown/content list、资源像素/SVG；只有确认像素/SVG 一致后才替换资源名。data URI 先还原为对应资源身份，不要求图片压缩编码字节一致，不忽略文本、结构、坐标或块索引。
- 默认 parse_method、formula/table 开关、lang、图片路径/模式和 batch 大小逐项一致。Rust Office 路由禁止调用模型加载器。
- 异常检查另起 worker：缺失文件、lang 数量不匹配、空文件、截断/损坏容器。要求错误输入拒绝解析；空文件对照 Python 的返回或拒绝及完整输出。保存异常类、错误码和消息作诊断，不要求两端异常类/措辞完全相同。

## 检查发现并修复

第一轮 72 组正向检查中 61 组精确一致，PPT 的 11 组暴露同一组链路问题，未将其计为通过：

1. 外部文字应按 slide persist reference 分组，并作为连续 records 解析；原入口将全文分组打散后逐 record 解析，破坏 TextHeader/文本/样式关联。现在按 slide/master 绑定，保留错误传播。
2. 原入口把整个 shape 的段落拼在一起，未投影完整列表、标题、样式和换行。现在沿用 Python 的逐段规则、层级与类型切换、标题提升、可见空白、链接和表格 HTML。复用已验证的原生 XYCut++，未另写简化排序。
3. ExObjRefAtom 的对象 id 位于 4 字节 payload 起始位置，之前误读偏移 8。图表数据复用 BIFF/MSGraph 解析，保留预览；不把二进制 Workbook 当作 OOXML XML。
4. 嵌入 OLE 存在完整的 zlib sync-flush 流。按 Python 接受完整 sync-flush 或正常结束流，同时验证声明尺寸、输入全部消费，拒绝截断、尾随输入及尺寸不符。不是放宽到任意不完整压缩流。
5. 对齐表格优先仲裁、背景/小图过滤、文本与图片共存、原生/图片公式优先级、备注归属、缺失/重复 persist 页和文本恢复。隐藏页按 Python 行为保留，不擅自改变内容范围。

异常对照还发现空 DOC/XLS/PPT 原先被当作一页有效空文档，空 PPTX 多一页。现在将文件或命名 stream 的格式提示传入原生绑定，只在空输入时使用；无格式提示的低层空字节调用保留原行为。非空文档仍由内容识别。

## 回归证据

- `ppt_output.json.gz`：21 组完整二进制输入，17 组预期错误；包含 1 份独立真实 PPT 和构造/恢复输入。完整 middle、资源像素和四种导出对照。
- `ppt_projection.json`：9 组独立段落投影参考，包括软换行、链接、可见空白、上下标、相邻样式、深层/变类型列表、普通段落边界和标题；同时对照正文树和表格 HTML。
- 压缩流单测同时检查正常结束/sync-flush 的接受和截断、尾随数据、错误声明尺寸的拒绝。
- 真实 PPT 语料仍只有 1 份，构造用例不能算真实文档数量，也不能据此声称所有未知 PPT 均已保真。

最终运行结果、安装路径和日志在本报告验收结果段记录；旧的失败报告保留，避免覆盖问题证据。

## 验收结果

- 七种格式正向检查 **73 组通过**：Office 67 组、PDF 6 组。结果 `build/public-delivery-final-20261101/summary.json`。PPT 真实样本恢复为与 Python 相同的 6 个逻辑页、3 个资源，完整结构和导出精确一致。
- 缺失文件、lang 数量、空文件、损坏输入检查 **36 组通过**，结果 `build/public-errors-final-20261101/summary.json`。这些是 9 个输入文件上的场景数量，不代表 36 份独立语料。
- `__init__`、`__call__`、`warmup` 公共参数顺序和默认值一致；Rust 仅额外提供可选的 cpu_spinning/cpu_dynamic_block_base，不改变原参数。记录 `build/public-signatures-20261101.json`。
- Office 阶段回归 **130 项通过**（58 个测试执行单元，0 失败、0 忽略）；完成最后的 PPT 改动后另有 **21 项受影响检查通过**。日志和统计分别为 `build/public-office-stage-regression-20261101.{log,json}`、`build/public-final-affected-20261101.{log,json}`。未跑全 workspace、完整 PDF 语料或 CUDA。
- 最后复查修正 PPT 原生公式的旧投影：块公式必须使用 interline_equation span，不能沿用行内 span。原生与图片公式统一入口，补充公开类型和 `$$` 分隔符断言。此后仅重跑受影响检查和安装后的 PPT，其他六种格式代码未改，不重复 PDF 推理验收。R2 的结果在最终交付 receipt 中记录。

完整七格式检查针对 `build/office-public-delivery-20261101` 安装包；最终 PPT 公式修正版本为 `build/office-public-delivery-20261101-r2`。两次二进制指纹分别保存，不能将前一次的整套报告冒充后一次整套重跑。源码与参考哈希记录于 `packaging/office-native-provenance.json` 的 public_delivery_20261101；原始失败证据在 `build/public-delivery-20261101` 和 `build/public-errors-20261101`。

**最终 R2 已安装到项目 `.venv`**。13 项受影响原生回归通过；安装后的 PPT 11 组正向和 4 组异常检查通过，报告分别为 `build/public-ppt-r2-installed-20261101/summary.json`、`build/public-ppt-errors-r2-installed-20261101/summary.json`。已核对 R1/R2 的 Python 运行时代码文件字节一致、R2 wheel 内来源哈希与当前源码一致，以及已安装二进制指纹一致。统一交付记录：`build/public-delivery-receipt-20261101.json`。

本轮没有重新给出速度倍数或内存结论。下一阶段使用隔离进程统一整理各格式性能及大型输入的内存趋势，优先修复可测量的问题；DOC 既有性能报告仍是它自己的样本与版本记录。

## 复现

```powershell
.venv/Scripts/python.exe scripts/migration/accept_public_delivery.py --output build/public-delivery-new
.venv/Scripts/python.exe scripts/migration/accept_public_delivery.py --output build/public-errors-new --errors-only
```

输出目录必须不存在。每个文档/引擎保存独立 report 和各场景完整快照，summary 保存差异路径和示例。`--resume` 只适用于同一二进制/输入，构建安装新版后必须使用新目录。局部可用 `--formats doc docx xls xlsx ppt pptx` 或 `--formats pdf`。

用户调用示例 `demo/rust_document_demo.py`，可编辑 FILES 后在现有 `.venv` 直接运行。Office 输出是逻辑页，不承诺 Word 的物理分页；PDF 会按默认配置加载识别模型。
