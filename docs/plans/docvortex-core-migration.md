# DocVortex 核心迁移与 Office 升级执行计划

**目标：** 将 RapidDoc 已使用的 DocVortex PDF 规则版面、表格能力迁入项目；升级 docx/xlsx/pptx，增加 doc/xls/ppt 原生解析；移除 fasttext-predict 系列必装依赖；由 RapidDoc 的 CPU/GPU extras 决定安装哪一种 ONNX Runtime。

**架构：** 将必要的上游代码放入私有目录 `rapid_doc/_vendor/docvortex_core/`，保留内部模块结构，通过薄适配层接入现有 pipeline 和 Office middle_json。保持 RapidDoc 的公开输出契约，按能力分批切换；不移植整个 DocVortex 应用、OCR 或导出系统。

**执行约束：** 不兼容被替换的旧解析代码、旧标签字符串协议、fasttext 接口或外部 DocVortex 安装方式。每个格式完成迁移后直接删除旧实现，最终只维护一套解析路径。doc/xls/ppt 是需要支持的文件格式，仍迁入其原生解析器。

**技术栈：** Python、pypdfium2、现有 Office 库、Pydantic、olefile；ONNX Runtime 继续按推理后端选择。首先采用 PDF 规则核心的 Python 实现，Rust 加速另行评估。

> 已执行：内置 PDF 核心、六格式 Office、输出适配、文件路由和依赖清理均已落地。实测与支持边界见 [迁移记录](../migrations/docvortex-baseline.md)。下文保留原规划；实际测试文件名与命令以迁移记录为准。

## 一、结论与已核实事实

### 1. 可以移植，CPU Runtime 的安装依赖和运行时导入要分开处理

- RapidDoc 的 `pyproject.toml` 已将 `onnxruntime` / `onnxruntime-gpu` 分别放在 `cpu` / `gpu` extras；DocVortex 放在独立的 `docvortex` extra。
- 本地 DocVortex 源码版本是 **0.5.16**，HEAD 为 `f5da4e69022f5b0a935fb64579d7f0f7ec7d08b6`。
- 该源码的 `pyproject.toml` 把 `onnxruntime>=1.20.1,<2` 列为必装依赖，同时还依赖 PyPI `magika`。
- 当前全局 Python 中安装的是 DocVortex **0.5.12**，不是上述源码版本；其 Magika **0.6.2** 的包元数据也明确要求 CPU 包 `onnxruntime`。只删 DocVortex 自身的一条依赖，仍可能被 Magika 间接装回。
- RapidDoc 已自带 `rapid_doc/model/magika/`，无需为了迁移再次依赖 PyPI `magika`。
- CPU/GPU 包都供代码通过 `import onnxruntime` 使用；GPU 包包含大部分 CPU 功能，同一环境应只安装其中一种。[ONNX Runtime 官方说明](https://onnxruntime.ai/docs/get-started/with-python)

因此，目标是**不强制安装 CPU 发行包**，而不是删掉所有 `import onnxruntime`。现有 Magika/OCR/模型仍可使用用户选择的 Runtime。当前 Magika 显式使用 `CPUExecutionProvider`，不必为文件类型识别强行启用 GPU。

`--no-deps` 可用于临时实验，不作为正式安装方案；长期应保证正常依赖解析也不会装入两种 Runtime。

### 2. 现有 PDF 入口适合保留，替换其内部实现

当前生产调用集中在 `rapid_doc/model/docvortex_adapter.py`：

| 功能 | 当前上游入口 | 接入位置 |
| --- | --- | --- |
| 文档级规则版面 | `document.pdf.PDFDocument`、`analyzers.native.PdfModel.predict()` | `pipeline_analyze.py`、`rapid_layout.py` |
| 区域表格恢复 | `analyzers.pdf.prepare_table_page()`、`recover_table_region()` | `analyze_utils.py` |

上游表格模块实际是 **`src/docvortex/analyzers/pdf.py`**，接口仍然存在。

迁移时保留 `docvortex_rule`、`docvortex_mode=off/rule/rule_plus_s`、`rule_table_engine=docvortex` 等配置，以及现有结果转换、公式/印章补充和模型回退策略。首先改变适配器底层导入，不同时重构外部配置。

### 3. 六种 Office 都可以借用，但新版输出不能直接接到旧后处理

- RapidDoc 的统一入口 `backend/office/office_analyze.py` 目前只分派 docx/xlsx/pptx。
- `main.py` 的路径和 bytes 输入、`cli/gradio_app.py` 都存在旧版 Office 的预转换；仅新增解析器不会让旧文件真正走原生解析。
- 新版 DocVortex 返回结构化 spans，标题区分 `doc_title` / `paragraph_title`；图片可能只有 `image_base64` 而没有 `content`。
- 当前 `office_magic_model.py` 的文本解析主要接收标签字符串，按旧类型处理标题，并会跳过缺少 `content` 的图片。直接替换 converter 会丢标题/图片，或在文本解析时报错。
- RapidDoc 当前 DOCX 还带有章节编号、目录锚点、富文本、图表等本地处理，不能按“上游更新了”就整体覆盖。

建议直接升级解析器和 Office 后处理，以新 spans 为唯一文本输入协议；图片、标题、列表、图表、公式逐项映射到 RapidDoc 的现有 middle_json，不保留旧标签字符串解析分支。表格 HTML、图片载荷和公式 LaTeX 仍按各自类型处理。

### 4. fasttext-predict 可以移除，推荐改掉实际业务判断

RapidDoc 当前依赖 `fasttext-predict` / `fasttext-predict-v2`，通过本地 `model/fast_langdetect/` 使用；实际生产调用只有两处：

| 调用方 | 实际用途 | 推荐替代 |
| --- | --- | --- |
| `backend/pipeline/para_split.py` | 用中日韩分类选择段落右边界阈值 | 用字符脚本特征和行几何选择阈值 |
| `backend/pipeline/pipeline_middle_json_mkcontent.py` | 决定拼接文本时是否加空格 | 用相邻可见字符、标点和实际行边界决定 |

DocVortex 的 `foundation/_text.py` 已用 `resolve_text_line_boundary()`、`merge_text_line_contents()` 完成无语言模型的连接：处理汉字/假名、韩文空格、英文断词、URL、标点等。

**自然语言识别与代码语言识别不同。** `guess_language_by_text()` 是 Magika 的代码/内容类型识别，不能拿它替换 fasttext 的自然语言判断。

不要把原来的 `detect_lang()` 简单改成“含汉字就返回 zh”；混排段落中仍可能需要英文词间空格。替换两个内部调用后直接删除旧接口、fasttext 模块和模型资源，不提供兼容 shim；新规则只服务文本连接和段落判断，不承担原来的 176 种语言分类。

## 二、规模、边界与工作量

对本地源码做了 AST 静态导入分析：

| 入口集合 | 静态涉及 Python 文件 | 源码行数约 |
| --- | ---: | ---: |
| PDF 文档 + 规则版面 pipeline | 136 | 69,280 |
| PDF 区域表格公开模块 | 92 | 37,986 |
| 六种 Office converter | 78 | 27,493 |

这些集合有重叠，**不能相加**。统计包含函数内导入和类型分支，不含完整的动态导入、资源和所有包初始化副作用，既不是精确迁移清单，也不是必须复制的行数。PDF 表格和版面还共享文档、字符几何、字体、图形原语和文本规则，因此不能只复制一个 `table.py`。

上游 `_compute_backend.py` 支持 `auto/python/rust`，Python 回退存在。去掉 Rust 构建可以降低首期工作量，但不能据此保证与上游默认 Rust 后端同速；性能必须单独测量。

首期范围：

- PDF 规则版面、区域表格、必要的 PDFDocument/字符与图形证据。
- 六种 Office converter，以及被它们使用的 OLE、富文本、图片、图表、公式和表格投影公共能力。
- 与此范围相关的测试、样例和版权说明。

暂不移植：DocVortex CLI/API/result 对象体系、完整渲染导出、OCR、EPUB/HTML/OFD/ODF/RTF 解析、整套 Rust 工程。某个公共模块若被目标入口依赖，按实际闭包处理，不能仅按目录名排除。

粗估 **15–30 工程人日**，主要不确定性是 PDF 依赖裁剪、Office 输出适配和旧二进制格式覆盖。先完成 P0 后再按真实清单修正估算。

## 三、关键设计决定

### D1：使用私有核心 + 现有适配器

新建 `rapid_doc/_vendor/docvortex_core/`，尽量保留上游内部模块布局，重写绝对包导入，精简各级 `__init__.py`。记录源文件路径、上游提交、文件哈希和本地改动原因。

收益：避免把整个上游应用依赖和公开对象体系带进来，后续比较上游修复有来源可查。代价：RapidDoc 需要维护这部分代码，升级时人工核对差异。

禁止通过修改 `sys.path`、伪造 `sys.modules['docvortex']` 或运行时下载源码实现迁移。

### D2：解析输出升级，公开输出保持 RapidDoc 契约

Office 后处理只接受新版结构化文本 spans，删除旧标签字符串分支。结构化数据直接映射，不先统一转为纯文本，也不把新 spans 转回旧标签字符串。每种格式的 converter 与后处理在同一迁移批次完成验证，避免新旧协议混用。

映射至少包括：标题级别和编号、样式、超链接、行内公式、列表/目录层级、图片和表格内图片、图表、锚点、页脚/注释。复用现有 image_writer 和 Markdown 导出。

### D3：旧 Office 直接原生解析，删除预转换和备用解析路径

doc/xls/ppt 统一使用迁入的原生解析器，不新增 `legacy_office_mode`，不保留为兼容旧解析链而设置的 LibreOffice 转新版模式或自动转换回退。

加密、不支持版本、损坏和资源超限按明确错误返回，不能把空解析结果当作成功。删除 `convert_legacy_office_to_modern()` 及其专用辅助逻辑；`office_converter.py` 若仍承担独立的 Office 转 PDF 功能，仅保留该功能，不因移除旧解析链而误删。

### D4：先验证 Python 核心，Runtime 和 Rust 分别选择

私有 PDF 核心先使用 Python 后端，不依赖 `docvortex._native`，也不修改用户进程的全局 `DOCVORTEX_COMPUTE_BACKEND`。保留可测量的计算接口；只有基准表明 Python 性能不满足目标时，再为热点引入 Rust。

Runtime 仍由 CPU/GPU extras 选择。全项目“Office 在完全无 Runtime 环境运行”属于额外目标：当前 RapidDoc 全局初始化 Magika，若要做到这一点，还需处理顶层导入与惰性初始化；不能把本次上游核心验证当成整包已支持。

## 四、分阶段执行清单

每个阶段独立提交；先建立行为测试或基线，再改代码，执行阶段验收后更新本计划。无需一开始运行全部演示脚本。

### P0：固定源码和行为基线（0.5–1 日）

**新增：** `docs/migrations/docvortex-source-manifest.json`、`docs/migrations/docvortex-baseline.md`、`tests/fixtures/docvortex_migration/`。

- [x] 对目标入口生成真实文件/资源/第三方依赖清单，补查动态导入、包初始化和 `PdfModel` 的输出规范化调用。
- [x] 固定 0.5.16 的提交与相关文件哈希；区分安装版 0.5.12 和源码版，不混用输出作为基线。
- [x] 记录 RapidDoc 当前规则路由与三个现代 Office 的 Markdown/middle_json 基线；记录上游六格式的语义基线。
- [ ] 纳入代表性 PDF：双栏、旋转文字、跨页表格、有框/无框表格、公式、扫描页与混合页。
- [ ] Office 基线覆盖现有 demo、空文件、目录/编号、合并单元格、隐藏 sheet、Strict OOXML、图片、图表和公式。
- [x] 每个引入文件保留原有版权标注；核对上游 pyproject 声明的多许可证与文件实际来源，准备第三方说明。

**验收：** 清单、来源、输入和期望行为可复现。不得重置或清理两个仓库已有改动。

### P1：移除 fasttext 的生产依赖（1–2 日）

**修改：** `rapid_doc/utils/language.py`、`rapid_doc/backend/pipeline/para_split.py`、`rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py`、`pyproject.toml`。

**新增：** `rapid_doc/utils/text_boundary.py`、`tests/test_text_boundary.py`、`tests/test_paragraph_script_rules.py`。

- [x] 迁移上游文本边界规则，补齐字符脚本统计；不引入 Magika/ONNX/fasttext。
- [x] 用实际行边界连接规则替换 Markdown 的整段语言判断，保留公式、超链接和样式边界。
- [x] 用脚本占比及行几何替换段落阈值分类，分别验证韩文和中英混排。
- [x] 测试汉字、假名、韩文、英文断词/硬连字符、URL、多行 URL、数字范围、标点、空文本和 Unicode 代理字符。
- [x] 删除 `detect_lang()`、两种 fasttext 依赖、模型打包声明、`FTLANG_CACHE` 初始化、`model/fast_langdetect/` 和模型资源；只保留仍被新规则使用的 Unicode 工具。

**验收命令：** `python -m pytest -q tests/test_text_boundary.py tests/test_paragraph_script_rules.py`。

**验收：** 阻断 fasttext 导入仍能处理 PDF 文本；基线未出现丢字、错误空格或异常段落合并。真正需要通用语言分类时另加可选后端，不恢复必装依赖。

### P2：迁入必要的公共核心（1–2 日）

**新增：** `rapid_doc/_vendor/__init__.py`、`rapid_doc/_vendor/docvortex_core/`、`rapid_doc/_vendor/docvortex_core/UPSTREAM.md`、`rapid_doc/_vendor/docvortex_core/licenses/`、`tests/test_native_core_imports.py`。

**修改：** `pyproject.toml`。

- [x] 依据 P0 清单迁入公共 schema/spans、几何、图片载荷、链接校验等必要模块；暂保留被引用的 Pydantic 类型，不同时重写 schema。
- [x] 精简包入口和绝对导入，避免加载上游应用 API、导出、OCR 和非目标格式。
- [x] 私有 `foundation/language.py` 只保留必要 Unicode 工具；代码语言识别通过惰性适配复用 RapidDoc 自带 Magika，不能依赖 PyPI Magika。
- [x] 建立 Python 计算接口，确认每个 Rust 调用点都有正确回退，禁止“空结果假装回退成功”。
- [x] 按实际直接使用补充 Pydantic、Pillow 等依赖声明；Office 需要 olefile、metafile-render、resvg_py 时逐项加入并验证 wheel 支持，不复制整个上游依赖列表。

**验收命令：** `python -m pytest -q tests/test_native_core_imports.py`。

**验收：** 核心导入不需要外部 docvortex、fasttext、PyPI magika 或 Rust 扩展；与宿主模型相关的识别保持惰性。

### P3：迁移 PDF 区域表格（2–4 日）

**上游来源：** `src/docvortex/analyzers/pdf.py`、`analyzers/native/pdf/_table_recovery/`、`table_materialization.py`、相关字符/图形证据模块。

**修改：** `rapid_doc/model/docvortex_adapter.py`；必要时修改 `rapid_doc/backend/pipeline/analyze_utils.py`、`rapid_doc/utils/pdf_text_tool.py`。

**新增：** `tests/test_native_pdf_tables.py`；**复用：** `tests/test_rule_table_routing.py`、`tests/test_rule_table_pdfium_integration.py`、`tests/test_table_merge_continuation.py`。

- [x] 先保留上游必要的 PDFPage/PDFTablePage 证据契约，验证后再判断是否可复用 RapidDoc 原语；不要直接把当前 pdftext chars 当作完全等价输入。
- [x] 替换 `recover_docvortex_table()` 的底层导入，保持 HTML、confidence、diagnostics 的接受与回退行为。
- [ ] 验证 point/归一化/像素坐标、旋转和裁剪；每页准备一次证据，同页表格复用。
- [ ] 验证关闭 PDF 句柄后证据仍可用，异常时资源释放，缓存不跨文档污染。
- [ ] 用合并单元格、无框表格、填充细线、上下标和倾斜/旋转样例对比上游。

**验收命令：** `python -m pytest -q tests/test_native_pdf_tables.py tests/test_rule_table_routing.py tests/test_rule_table_pdfium_integration.py tests/test_table_merge_continuation.py`。

**验收：** 原生恢复不再导入外部 DocVortex；失败仍按当前配置进入规则/模型备用路径，非表格区域不会误接受。

### P4：迁移 PDF 文档级规则版面（3–6 日）

**上游来源：** `src/docvortex/document/pdf/`、`analyzers/native/pdf/pipeline.py` 及其闭包；包括 `PdfModel.predict()` 后的文本规范化。

**修改：** `rapid_doc/model/docvortex_adapter.py`；必要时修改 `rapid_doc/model/layout/rapid_layout.py`、`rapid_doc/backend/pipeline/pipeline_analyze.py`。

**新增：** `tests/test_native_pdf_layout.py`、`tests/test_native_pdf_fallback.py`。

- [x] 迁移完整文档级分析，保留跨页标题、页眉页脚和阅读顺序上下文，不能拆成独立逐页调用替代。
- [x] 保留现有类别映射、`original_order`、旋转文本和原生表格 HTML 行为。
- [x] 验证 `rule`、`rule_plus_s` 的公式/印章补充及扫描/混合页处理。
- [ ] 验证空版面、异常页眉页脚、整页误判图片等模型回退条件，并确认模型只在需要时初始化。
- [x] 表格和版面迁移都完成后，直接删除 `docvortex` extra 及其安装提示，不保留空 extra 兼容别名；规则功能作为内置能力提供。
- [ ] 测量冷启动、热调用、每页耗时和峰值内存；分开比较现有 RapidDoc、上游 Python、上游 Rust，不承诺纯 Python 同速。

**验收命令：** `python -m pytest -q tests/test_native_pdf_layout.py tests/test_native_pdf_fallback.py tests/test_native_pdf_tables.py`。

**里程碑：** 现有 PDF 规则功能完全内置，正常安装规则功能不会再由 DocVortex/Magika 拉入 CPU Runtime。

### P5：升级 docx/xlsx/pptx，逐格式切换（3–6 日）

**上游来源：** `analyzers/native/office/docx/`、`xlsx/`、`pptx/`，共享 `rich_text.py`、`image.py`、`chart.py`、`equation/`、`spreadsheet/` 等实际依赖。

**修改：** `rapid_doc/model/docx/main.py`、`rapid_doc/model/xlsx/main.py`、`rapid_doc/model/pptx/main.py`、`rapid_doc/backend/office/office_magic_model.py`、`model_output_to_middle_json.py`；必要时修改 `rapid_doc/backend/utils/office_image.py` 和 `html_image_utils.py`。

**新增：** `tests/test_office_structured_spans.py`、`tests/test_office_modern_parity.py`、`tests/test_office_assets.py`。

- [ ] 将文本后处理改为结构化 spans，删除旧标签字符串解析；显式映射标题类型、图片字段、超链接、样式和行内公式。各格式切换时同步调整相关调用和测试。
- [ ] 先切 DOCX：保留 RapidDoc 本地目录锚点、编号、文本框、复杂字段、表格、图表和公式行为。
- [ ] 再切 XLSX：迁入共享 SpreadsheetProjector，验证合并单元格、日期/数值格式、公式缓存、隐藏 sheet、图片、chartsheet，以及既有 xlsm 支持。
- [ ] 再切 PPTX：验证主题/布局样式继承、组合形状、坐标/阅读顺序、备注、表格、图表和公式。
- [ ] 复用并升级 package normalizer，覆盖 Strict/Transitional 和非标准 part 路径。
- [ ] 每个格式通过语义对比后，直接替换并删除其旧 converter、专用工具和无用入口；不增加新旧解析器选择开关，不同时维护两套公共图片/公式/表格工具。

**验收命令：** `python -m pytest -q tests/test_office_structured_spans.py tests/test_office_modern_parity.py tests/test_office_assets.py`。

**验收：** 现有 demo 和上游相关 fixture 均可解析；Markdown、图片文件与 middle_json 不丢内容。只比较该格式支持的语义，不用字典完全相等代替质量验证。

### P6：增加 doc/xls/ppt 原生解析（3–6 日）

**上游来源：** `analyzers/native/office/legacy/`、`doc/`、`xls/`、`ppt/`、`errors.py`、`limits.py`、相关 MTEF/OfficeArt 模块。

**修改：** `rapid_doc/backend/office/office_analyze.py`；**新增：** `tests/test_office_legacy_doc.py`、`tests/test_office_legacy_xls.py`、`tests/test_office_legacy_ppt.py`、`tests/test_office_legacy_errors.py`。

- [x] 先迁入有界 OLE/binary/OfficeArt 读取与统一错误分类、资源限制。
- [x] 接入 DOC 的 FIB、piece table、样式/字段/书签、section、表格、图片和内嵌公式。
- [x] 接入 XLS 的 BIFF、SST、富文本、合并、日期/数值格式、公式缓存、图表/图片；复用 P5 的表格投影。读取缓存值不等于重新计算公式。
- [x] 接入 PPT 的记录树、文本样式、幻灯片/形状、表格、图片、公式和图表。
- [x] 移植上游生成 fixture 的辅助模块与相关测试，并改为导入私有核心；加入真实样例，防止仅合成文件通过。
- [ ] 覆盖加密、缺 stream、损坏记录、超限、未知版本、空文档及不支持的嵌入对象，明确支持边界。

**验收命令：** `python -m pytest -q tests/test_office_legacy_doc.py tests/test_office_legacy_xls.py tests/test_office_legacy_ppt.py tests/test_office_legacy_errors.py`。

**验收：** 解析核心不调用 LibreOffice；页语义按 DOC section、XLS sheet、PPT slide 定义，不承诺恢复 Word 的精确印刷分页或六格式视觉完全一致。

### P7：统一文件识别和全部入口路由（1–2 日）

**修改：** `rapid_doc/utils/guess_suffix_or_lang.py`、`rapid_doc/main.py`、`rapid_doc/cli/common.py`、`rapid_doc/cli/gradio_app.py`、`rapid_doc/utils/office_converter.py`；必要时修改 `rapid_doc/utils/empty_office.py`、`rapid_doc/cli/__init__.py`。

**新增：** `tests/test_office_detection.py`、`tests/test_office_input_routes.py`、`tests/test_office_native_errors.py`。

- [x] 文件识别采用 PDF signature / OOXML 包结构 / OLE stream 优先，Magika 兜底；迁入 Strict 和非标准包判型修复。
- [x] 验证加密 OOXML 的 OLE 容器不被误识别为普通 DOC/XLS/PPT，扩展名和真实内容不一致时有明确行为。
- [x] 将 doc/xls/ppt 加入 Office 格式集合、读取与分派路径，清除 main 和 Gradio 中的无条件预转换。
- [x] 路径、bytes、流、URL 输入，以及同步/异步和批处理都经过统一策略；覆盖已存在的空现代 Office 行为。
- [x] 删除旧 Office 预转换函数、专用临时文件逻辑和无用配置；不增加转换模式开关，保留原始文件名/来源和解析器信息，不支持或失败时返回明确错误。

**验收命令：** `python -m pytest -q tests/test_office_detection.py tests/test_office_input_routes.py tests/test_office_native_errors.py`。

**验收：** 阻断 soffice 调用仍能通过六种 Office 的正常入口解析；旧解析链与自动转换分支已删除，失败不会静默返回空结果。

### P8：安装、打包、回归和文档收尾（1–2 日）

**修改：** `pyproject.toml`、`README.md`、`docs/analyze_param.md`；新增迁移说明与 `tests/test_dependency_contract.py`。

- [x] 在干净环境检查 wheel 元数据和完整依赖解析，确保没有 `docvortex`、PyPI `magika`、fasttext 系列必装依赖。
- [x] 分别验证 `rapid-doc[cpu]` 和 `rapid-doc[gpu]`，两种环境各只安装对应一种 Runtime；不要组合 cpu+gpu extras。
- [x] 检查已有 `test = rapid-doc[cpu]` 的语义：GPU 验证环境不要再安装 test extra，否则它会主动拉入 CPU 包；测试工具可单独安装。
- [ ] 检查 `core`、`npu` extras 的依赖链，避免其他传递依赖装回 CPU 包；确认 pypdfium2 版本要求、Pydantic API 和 Python 3.10–3.14 的新依赖兼容性。
- [x] 构建 wheel/sdist，确认新私有包、必要资源和版权文件入包；所有资源定位都从 RapidDoc 自身出发。
- [x] 在 wheel 安装环境运行六格式与 PDF smoke，阻断外部 docvortex 导入；不使用开发机器的全局包掩盖缺失依赖。
- [x] 更新配置、旧 Office 原生解析支持边界、旧接口/extra 移除说明和迁移来源；检查已替换实现、兼容分支与废弃资源均已删除，再完成完整回归。

**验收命令示例（届时在独立环境执行）：**

```powershell
python -m pytest -q tests/test_dependency_contract.py tests/test_native_core_imports.py
python -m build
python -m pip install --dry-run --ignore-installed --report install-gpu.json ".[gpu]"
python -m pip install --dry-run --ignore-installed --report install-cpu.json ".[cpu]"
```

Dry-run 验证安装清单，真实运行还需分别安装构建出的 wheel 并执行 smoke；CUDA provider 的实际运行只在具备匹配 CUDA/cuDNN 的环境验证。

## 五、执行顺序与完成标准

推荐顺序：**P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8**。

先解除当前 PDF 功能的外部 DocVortex 依赖，再升级 Office；XLS 原生投影依赖新的公共表格组件，应在 P5 之后接入。不要在同一批次同时替换 PDF 文本算法、全部 Office converter 和公开输出 schema。

最终完成条件：

- [x] 正常 GPU 安装不会被这些迁入功能拉入 CPU Runtime；`cpu`、`gpu` 选择有效。
- [x] 现有规则版面/表格配置无需安装外部 DocVortex 即可运行，回退和混合模式通过回归。
- [x] 六种 Office 均通过统一入口原生解析，旧 converter、标签字符串文本协议及旧 Office 预转换路径已删除，不保留兼容开关。
- [x] 输出兼容 RapidDoc；关键文字、标题、列表、表格、链接、图片、图表和公式不丢失。
- [x] 生产调用和 wheel 不再必需 fasttext 模块或语言模型资源。
- [x] 性能、内存、支持边界、上游来源和版权文件有记录。

## 六、规划阶段的上游预验证（历史记录）

本次使用本地 **0.5.16 源码**，强制 `DOCVORTEX_COMPUTE_BACKEND=python`，并在独立 Python 进程中阻断 `onnxruntime`、`magika`、`fasttext` 导入：

| 验证 | 结果 |
| --- | --- |
| `demo/ms_office_docs/docx_01.docx` / `docx_01.doc` | 均成功，3 个逻辑页 |
| `demo/ms_office_docs/xlsx_01.xlsx` / `xlsx_01.xls` | 均成功，3 个可见 sheet |
| `demo/ms_office_docs/pptx_01.pptx` / `pptx_01.ppt` | 均成功，6 个 slide |
| 临时生成的 3×3 表格 PDF，文档级版面 | 成功识别一个 table block |
| 同一 PDF，关闭文档后的区域表格恢复 | 成功，包含单元格 22，confidence=1.0 |
| 上游 `test_text_utils.py`、`test_text_boundary_rules.py` | **43 passed** |
| 上述六格式和简单 PDF 的模型依赖导入尝试 | 0 次 |

以上为实施前的上游预验证。实施后的安装、语义检查、回归、基准结果见迁移记录；Rust 性能和 CUDA provider 的实际推理仍需专项验证。

规划阶段未修改生产代码；实施阶段已迁入所需代码，并仅在隔离环境安装测试依赖，未修改 DocVortex 仓库或卸载全局 Runtime。剩余扩大覆盖工作保留为未勾选项，不表示旧解析路径仍在运行。
