# Python 绑定历史迁移记录

本文归档自 2026-11-04 整理前的 `bindings/python/README.md`，仅供追溯开发过程，不再持续维护。文中的阶段状态、构建路径及旧预览接口可能已失效；当前安装与调用方式以 [Python 使用说明](../../bindings/python/README.md) 为准。迁移检查点现由 `scripts/migration` 开发工具显式启用，不随正式 wheel 发布。

---

# 当前预览包安装（2026-11-02）

2026-11-04：Python 正式入口现调用统一 Rust SDK，原有 `RapidDoc(...)` 参数和返回值保留。模型配置、运行库选择及预加载归共享 Rust 管理，预加载与解析复用会话；原生 Rust 使用见[SDK 示例](../../rust/rapiddoc/README.md)，验证见[记录](../../docs/plans/2026-11-04-shared-rust-sdk.md)。

2026-11-04：正式包已移除旧迁移预览集合，`RapidDoc(...)` 和现有 CLI 不变。旧 `parse_pdf_models` 等检查点仅供 `scripts/migration` 开发工具显式启用，不再随 wheel 发布；常用诊断保留在 `_diagnostics.py`。见[清理记录](../../docs/plans/2026-11-04-preview-package-cleanup.md)。

2026-11-04：版面框、span 框和行顺序绘图已下沉共享 Rust PDF 层；`f_draw_layout_bbox`、`f_draw_span_bbox` 调用不变，Python SDK 不再依赖 pypdf/reportlab。生产绘图不经过 Python 页面 JSON 往返，后续语言绑定共用同一实现。见[实现与验证](../../docs/plans/2026-11-04-shared-rust-drawing.md)。

最新包补齐整页方向矫正开关，并修复退化裁图，见 [方向矫正](../../docs/plans/2026-11-02-document-orientation.md)及[裁图修复](../../docs/plans/2026-11-02-degenerate-perspective-crops.md)。
本机 Windows 包还调整了阶段进度条，见 [本机日志验证](../../docs/plans/2026-11-02-local-progress-display.md)；本次未更新 Linux 包。
算法块的原生 Magika 分类已接入，见 [修复与评测日志](../../docs/plans/2026-11-02-magika-algorithm-and-eval-logs.md)。

包名 `rapid-doc-rust`，导入名 `rapid_doc_rust`。Windows x64 CPU、Linux x64
CUDA 的独立环境安装和自动运行库加载已验收；不需要安装原 RapidDoc、
RapidOCR、pdftext、pypdfium2、Shapely，也不需要手动指定 DLL/SO 路径。

Windows CPU 使用已构建的预览 wheel：

```powershell
python -m pip install build/progress-wall-clock-windows-wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl
```

Linux CUDA 使用单独的预览构建（包名和 API 相同，依赖为 GPU 版 ORT）：

```bash
python -m pip install build/doc-orientation/linux-wheels/rapid_doc_rust-0.1.0-1cuda-cp310-abi3-manylinux_2_35_x86_64.whl
```

该 Linux 包要求 glibc >= 2.35 和兼容 NVIDIA 驱动，会安装已验收版本的
GPU ORT、cuBLAS、cuDNN 与 CUDA Runtime。建议新建环境；同一环境选择
一种 ORT 发行包，避免 `onnxruntime` 和 `onnxruntime-gpu` 覆盖相同文件。
这两个 wheel 是本地预览产物，尚未发布 PyPI，使用前需取得对应文件。

```python
from rapid_doc_rust import RapidDoc

engine = RapidDoc(output_dir="output")
result = engine("input.pdf")
print(result.markdown)
```

默认 CPU；Linux CUDA 包可用 `MINERU_DEVICE_MODE=cuda python app.py` 启动。
Office 六种格式使用同一入口，不加载模型。已有模型通过可选环境变量
`RAPID_MODELS_DIR` 复用；未指定时查找原 Python 模型目录，独立安装则
默认下载到 `~/.cache/rapid_doc/models`。原始模型按 SHA 校验；CUDA 优化图
在初始化时由 Rust 在内存生成并复用会话，不生成优化 ONNX 磁盘缓存。

[干净环境验收与边界](../../docs/plans/2026-11-02-clean-release-acceptance.md)。
以下旧日期的待完成项和构建指令保留作历史记录，以此处最新记录为准。

# 当前 PDF 与 Office 接入

公开入口为 `from rapid_doc_rust import RapidDoc`，支持 PDF、DOC、DOCX、XLS、XLSX、PPT、PPTX。六种 Office 格式由 Rust 原生解析，不加载识别模型；PDF 自动定位运行库和模型。统一示例：[demo/rust_document_demo.py](../../demo/rust_demo.py)。

2026-11-01 的统一验收补齐 PPT 幻灯片外部文字绑定、逐段/列表/标题投影、排序和嵌入图表及其预览。隐藏页遵循当前 Python 输出行为，不再自行过滤。空文件按输入格式对齐 Python：DOCX/XLSX 保留空逻辑页，PPTX 无幻灯片，空 DOC/XLS/PPT 拒绝解析。验收范围和结果见[七种格式公开接口验收](../../docs/plans/2026-11-01-public-delivery-acceptance.md)。

详见 [DOC 收尾记录](../../docs/plans/2026-11-01-doc-completion-acceptance.md)、[DOCX 交付记录](../../docs/plans/2026-10-31-docx-public-output.md)、[XLSX 交付记录](../../docs/plans/2026-10-31-xlsx-public-delivery.md)、[PPTX 交付记录](../../docs/plans/2026-10-31-pptx-public-delivery.md)。以下保留 PDF 历史记录。

# Python Rust 开发预览

2026-10-28 当前节点：原 PDF 宿主路由已接入 Rust 原生全文，`parse_pdf_models` 支持 `off/rule/rule_plus_s`、`native_only` 和内置原生区域表格恢复，贯通模型回退、OCR、公式、段落、跨页表格、Markdown、content_list 和图片资产。已完成 release 构建安装（8 分 51 秒）；本阶段安装对照、输出位置和限制统一见[宿主接入验收](../../docs/plans/2026-10-28-native-pdf-host-integration.md)。跨页 OCR 批次、BGR 渲染和边界坐标类型均按原 Python 保留，未改冻结参考或降低算法精度。`docvortex_mode` 默认仍为 `off`，原生模式显式传 `rule`；增强阅读顺序、CUDA 与同条件性能测量后置，不能据局部样本宣称全部 PDF 等价或普遍更快/更省内存。下方“宿主/Markdown 待接”为历史检查点。

当前：完整原生文件入口 `analyze_native_pdf(source)` 已实现，返回 `pages`（原 PdfModel 的 model_list）、`source_page_count` 和 `page_sizes`；`collect_native_pdf_sources(source)` 返回采集检查点。无须传入 raw_pages，不执行 OCR。统一 release 安装验收进行中；默认 `parse_pdf_models` 的模型/OCR/Markdown 路由尚未自动接入本入口。见[完整文件节点与用法](../../docs/plans/2026-10-28-native-pdf-file-integration.md)。以下为历史记录。

最新：全文行内与表内媒体已接通并安装验收。`prepare_native_final_document_blocks(raw_pages, materialize_inline=True)` 返回 `finalized_pages`；`finalize_native_document(raw_pages, pdf_source, ...)` 继续表格媒体并返回逐页最终块。raw_pages 为已有原始页源/几何，后者的 pdf_source 供媒体渲染；完整 PDF 文件采集入口仍待接。915 空格规则、1,645 行内编排、35 全文链、6 媒体链/3 解码 PNG 和 3 页实时 PDFium 通过。见[接口与对应表](../../docs/plans/2026-10-28-document-inline-integration.md)。以下为历史记录。

最新安装结果：完整最终页已构建安装通过，211 页级用例、35 原始页源链、1,458 规则及 3 页实时 PDFium 完整输出；使用 `prepare_native_final_page_blocks(..., finalize_output=True)` 或全文同名开关取得 `model_list`。剩余外层全文行内/媒体接线、默认 PDF 与整篇验收。下方“安装验收进行中”为此前检查点；[完整记录](../../docs/plans/2026-10-28-final-page-output.md)。

2026-10-28：最终页及全文文字块入口新增 `finalize_output=True`，继续完整图注带/图例、视觉区域绑定、分组阅读排序、tight 和最终归一化，额外返回 `model_list`。211 页/35 原始页源链源码对照通过，统一安装验收状态见[主链](../../docs/plans/2026-10-27-form-runtime-integration.md)，[Python→Rust 对应表](../../docs/plans/2026-10-28-final-page-output.md)。外层全文行内收尾、默认 PDF 和整篇质量/耗时/内存验收仍待；以下为历史记录。

2026-10-28：最终页/全文文字块入口新增 `refine_blocks=True`，继续完整成员、标题、图注、镜像环绕正文、来源附注和局部列表，额外返回 `refined_text_blocks`。默认 False 保留基础文字块检查点。211 连续页、35 原始页源整链和 1,210 规则/空白合同通过；统一安装状态及精确剩余位置见[主链记录](../../docs/plans/2026-10-27-form-runtime-integration.md)。视觉注释及最终排序/输出/默认 PDF 尚待。以下为历史记录。

2026-10-28：新增 `build_native_text_blocks`、`prepare_native_final_page_blocks`、`prepare_native_final_document_blocks`，完整迁移现有 Python 文字块构建与全部段界/脚注/补合并；最终文档接口返回 `block_pages`，不是归一化后的最终 PDF 输出。204 连续页和 35 原始页源整链通过。统一构建安装状态及后续成员/图注/排序/输出边界见 [文字块节点](../../docs/plans/2026-10-28-text-block-integration.md)。以下为历史阶段记录。

完整逐页准备节点已统一构建并安装到 `build/rust-wheel-env`。`prepare_native_document_pages(pages, complete_page=True)` 完成全文几何/容器/脚注之后的整页合并、上/下标、证据冻结与字符压缩、页辅助复核和最终末表查询；`finish_native_geometry_page` 也支持该参数，并可传 `tight_bboxes/origins`。默认 False 保留原阶段入口。976 组合并、499 组完整页准备、身份/异常与实时 PDF 对照通过；新入口及四组旧安装套件通过。全文语义、其余公式、最终编排与默认 PDF 入口继续后续迁移。以下为历史阶段记录。

源码 `prepare_native_document_pages(pages, classify_footnotes=True)` 和 `finish_native_geometry_page(..., classify_footnotes=True)` 继续图像脚注、表后字段保护和页面脚注分组，自动包含前置 classify_text；默认 False 保留旧检查点。164 组场景、468 组连续旧语句和 1 页实时脚注正例通过。文本簇合并/上下注标与后续最终编排仍待，未重新打包安装。以下为历史阶段记录。

源码新增 `prepare_native_document_pages(pages, classify_text=True)`，`finish_native_geometry_page` 也支持该参数：容器后继续图题、术语/附录、引用、展示标题和异常 CJK 段组恢复。默认 False 保留原检查点。92 组分类、304 组连续文档、来源身份/快照及 3 页实时 PDFium 对照通过；脚注、后续公式和最终编排仍待，未重新打包安装。

源码新增 `prepare_native_document_pages(pages)`：原始 source/geometry 全文→前置分类/容器认领→几何重拆/标签回收/行首分式，页面可附 style_lines/link_lines。61 组分类、211 组连续旧语句（含 4 旧异常）、3 页实时 PDFium 对照通过。仍待后置辅助/公式/最终文字编排与默认 PDF 接入；未重新打包安装。以下为历史阶段记录。

2026-10-27 源码新增 `prepare_native_page_marginals` 和 `prepare_native_document_sources`：完整容器前页边保护/重复分隔线，以及几何→页边→Form→分隔线的文档前缀。207 组前缀和三页实际 PDFium 对照通过；尚未重新构建安装 wheel，默认完整 PDF 入口仍待后续分类接线。

2026-10-27 源码新增完整全文几何生成/容器前修复接口：`build_native_document_geometry_plan`、`prepare_native_document_geometry`、`apply_native_geometry_plan`、`finish_native_geometry_page`。146 组文档前缀和 10 页实际 PDFium 对照通过；页边及后续文字分类和默认 PDF 仍待。尚未重新构建安装本阶段 wheel，当前已安装预览不包含这些接口。

2026-10-27 源码已新增实际 Form 结构、全文 Form 角色、图像容器和矢量公式链；`prepare_native_document_containers` 仍是阶段接口，完整几何/公式/文字分类及默认 PDF 迁移尚未结束。本阶段尚未重新构建安装 wheel，已安装版本不包含这些新增接口。开发验证及剩余范围见 [主链记录](../../docs/plans/2026-10-27-form-runtime-integration.md)。

2026-10-27：新增 `extract_native_index_blocks`、`select_native_form_regions`、`prepare_native_container_table_page`；目录保护与Form选择已接上现有代码/表格页编排。来源映射使用`[{"bbox":[...],"source_indices":[...]}]`传递，返回副本保留收紧别名和旧行分类副作用。能力`native_index_containers/native_form_selection/native_container_table_page=True`。

416组旧结果＋2旧异常、75实际PDF页（4目录/41表格/3代码）、8相关Rust单元和1集成通过；阶段末1次构建/安装及5组受影响安装套件已通过。实际PDF的Form归属为空，Form选择正例仅源级语料，完整Form采集/角色归一化、图形图片物化与后续文本公式/默认PDF仍待。以下为历史接口记录。

`build_native_code_blocks(source, *, kind="filled", excluded_bboxes=(), claimed_line_indices=())`输出`code_blocks/claimed_line_indices`。`filled`检测底色等宽代码，`ruled`包含横线限定和算法标题清单；须保留源行的字形/字体/真实视觉行身份，不能用固定图框代替证明。代码空间投影保留空行，回退文本沿旧控制字符清理规则。

`prepare_native_code_table_page`接受与`prepare_native_table_page`相同的输入证据，执行横线/算法代码→表格提交→底色代码，并额外返回`rule_code_blocks/code_blocks`。上游非代码/表格容器证据仍由excluded_bboxes/prior_fixed_blocks/prior_claimed_line_indices提供；来源号须唯一，不改变原table-only入口。

能力标记`native_code_containers/native_code_table_page=True`。227完整代码＋63阶段参考、9相关Rust测试及67不同实际PDF页已通过，实际3页代码/41页表格；7组受影响安装测试已通过（修复整数边界后补构建复验）。目录/Form/图像图形提交和后续文本公式仍待，完整PDF开关保持False。以下为历史接口记录。


`prepare_native_table_page(source, *, excluded_bboxes=(), prior_fixed_blocks=(), prior_claimed_line_indices=(), table_header_separator_bboxes=(), tight_bboxes=None, origins=None, calibrated_char_bboxes=None)` 已接通正式页准备的表格事务链。它自动执行强图形/数值反证避让、候选检测、原始成员恢复和提交，返回 `source/table_blocks/annotation_blocks/claimed_line_indices/remaining_lines/fixed_blocks/table_bboxes/strong_graphic_bboxes/next_table_grid`，无需传候选或HTML结果。

`source`需保留真实字符、视觉行、绘图线和路径等既有PageSource字段，行source_index须唯一；重复号沿现有提交接口拒绝。已有非表格排除框通过excluded_bboxes传入，prior_fixed_blocks参与固定块顺序与末表证明，prior_claimed_line_indices保护先前已认领的行。字形框/原点映射为真实提取证据。所有输出为独立对象，失败不提交部分结果。

`native_table_page_commit=True`只标记表格部分贯通。144组阶段组合、2个合同拒绝、31个受影响Rust测试及59实际PDF页通过（41页表格/6页续表轨道）；阶段末一次release wheel及8组隔离安装套件通过；新增3组分析/原始行副作用对照，共147组有效输入＋2个合同拒绝。目录/代码/Form选择、图形图像块与后续公式文本处理仍需整体接线，默认PDF完成开关保持False。以下为历史接口记录。


`detect_native_strong_graphics(source)` 已接通完整旧强图形检测：从源行自动推导字号与CJK尺度，按旧序组合十类图体，再执行网格排除、包含/重叠仲裁。返回 `{"source": ..., "graphic_bboxes": ...}`；source 是独立投影，保留柱图认领caption时的行语义/标题/段落变化，传入对象不修改。栅格坐标图core检测不额外执行描述分组。

`native_strong_graphics=True`。493组完整旧结果＋37异常、8个受影响Rust测试及59页PDF通过，新增四方向真实轮廓PDF有2接受/2拒绝；阶段末一次release wheel及7组隔离安装套件已通过。正式整页提交和默认PDF仍待，完成标志保持False。以下为历史接口记录。


`detect_native_diagram_components(source, *, em)` 已完成旧原生复杂图体组件：自动从真实源计算表格屏障，连接复杂路径与相邻图片，拒绝正文区域，并认领时间轴图标和窄列说明。需要保留 `source.chars` 的 bbox/tight_bbox、完整源行/绘图线/路径/图片框；`em` 为旧上游图形字号，仍由调用方传入。

`native_diagram_components=True`仅代表完整组件可用，尚未切换默认整页PDF。226组完整函数结果、11旧异常和55页实际源重建通过Rust验证；实际55页均为CJK指标门控负例，图体/时间轴正例来自独立源级参考，不能宣称真实PDF正例或全文等价。阶段末一次release wheel及6组隔离安装套件已通过。以下为历史接口记录。


固定 CJK 字体运行时已接通，`native_fixed_cjk_font_runtime=True`。PDFium 首次使用时安装与旧 Python 相同的 DroidSansFallbackFull 字体及缓存身份；字体和精确编码表编入 Rust，无需旧 rapid_doc 包或 Python 回调。原未嵌入 CID 字框差异已消除，并提升为常规回归。

`pdfium_font_runtime_info(*, pdfium_library=None)` 初始化运行库并返回字体策略、SHA256及真实替代/完整复制计数；只返回值，不暴露句柄。`classify_pdf_font_name(face: bytes, *, charset=1)` 按旧五编码及明确别名判断固定替代，返回字符集或 None；字体名最多4096字节。导入 SDK 不会初始化 PDFium；仍须与 pypdfium2 分进程使用。

源码验证：4771分类、3209解码序列、8页新增真实四语言CID PDF、16像素参考及已有字符/表格回归通过；阶段末一次release wheel及7组隔离安装套件通过，其中32组RGB/RGBA像素逐一匹配旧版。完整 PDF 默认接入和兼容性完成标志仍未开放。以下为历史接口记录，较早字体False/CID差异记录已由本阶段补齐。


`detect_native_table_candidates(source, *, excluded_bboxes=())` 已接通旧表格完整候选检测：全部产生器、完整表替换、表题分组、正文尾段裁剪、Form 排除、续表标题和短中文题名收尾。`detect_native_structural_table_candidates` 保留为最终后处理之前的候选接口。候选检测不执行最终表格恢复/图形仲裁，也没有切换默认 PDF 流程。

`apply_native_table_page_policies(source, candidates, *, operation="all")` 可独立验证后处理；可选 `join/trim/forms/externalize/bind`。输入及输出注释的 `line_bboxes` 为保留插入顺序的整数键字典，调用不会修改传入对象。`trailing_native_table_grid(source, prepared)` 从 `prepared.fixed_blocks` 和 `prepared.remaining_lines` 推导页末表格最后物理行的下一页列轨；未确认末表时返回 None，不要求调用方提供轨道结论。

能力标志 `native_complete_table_candidates/native_table_page_policies=True`；完整迁移开关仍为 False。1533 组结果事实、3 个旧异常和 58 页实际 PDF 已通过 Rust 对照；阶段末 release wheel 和4组隔离安装套件已通过。固定 CJK 字体仍缺，原未嵌入 CID 失败不计通过。以下为此前接口的历史阶段说明。


`detect_native_supplemental_tables(..., kind=...)`新增最后两个完整分支`aligned_numeric`及`captioned_mixed`。前者要求原始`source.chars`，从整页数值组推导区域，再以区域原生字符网格确认；后者保留原生表头字形，支持多层表头、临近表注和已知完整表头的单行续表。保留原字段`formula_candidate_only/caption_start/semantic_type/font_signature`等及所有字形几何；只给行框不能替代字符恢复证明。

`detect_native_structural_table_candidates`现已联合全部候选产生分支和完整替换仲裁，但尚未执行最终整页caption join/prose tail/Form/续表收尾，因此不是完整默认PDF入口。`native_table_candidate_producers=True`只标记产生器可用，原完成开关保持False。1503结果事实＋1异常、11区域边界、55页实际PDF已通过Rust对照；固定CJK字体运行时仍未接通，原未嵌入CID失败不计通过。


`detect_native_supplemental_tables`现另支持`partial_grid`、`year_header`、`metric_pair`、`unruled_numeric`、`captioned_pair`五个kind；`detect_native_structural_table_candidates`已联合编排并执行完整覆盖替换仲裁。来源行还应保留`formula_candidate_only/compact_formula_cluster`，原生文字列证明必须保留字形和字体元数据。新增独立数值候选入口capability `native_numeric_candidate_pipeline=True`，通用检测器仍缺aligned及captioned-mixed两个分支与最终整页策略。

`native_fixed_cjk_font_runtime=False`明确当前未安装旧固定CJK替代字体。未嵌入CID中文字体可能产生字框差异，保留1个ignored失败回归；嵌入字体指标表和其余实际样本合计46页已通过实时源重建验证，不能据此宣称未嵌入CJK已等价。默认PDF接入前须补齐该字体运行时；源级候选差分不替代实时提取证明。


新增一般结构表格入口：`detect_native_rule_table_candidates(source, *, angle=0, median_height=None, excluded_bboxes=())`完成普通横线全部区间枚举、自动核心选择、稳定列/连续行/正文拒绝、评分、表题/表注和连通扩框。`detect_native_supplemental_tables(source, *, kind, excluded_bboxes=())`支持`open_vertical`、`short_caption`、`shaded_header`、`blank_answer`、`page_clipped`五个完整旧分支，分别为开边竖线、独立表题短表、浅色表头/裁剪阶段网格、空答案表、右界裁切表单。

`detect_native_structural_table_candidates(source, *, excluded_bboxes=())`按旧顺序联合上述分支与filled/tiled/闭合网格：草稿跨正文排除后正式组装/扩框/合并，完整开边替换优先，裁切表单排除已有区域。返回普通候选字典及有序整数键注释字典，所有权独立。source沿已有视觉行/矢量源；补充行元数据`dominant_font_weight/paragraph_terminal/semantic_type/caption_start`，Path保留`fill_rgba/segment_count/fill_visible/stroke_visible/form_depth`。空答案判断必须保留表头实际字形及字体签名；阶段网格需要真实不可见裁剪路径和图像框，不能手工写入恢复结论。

此入口是已移植结构分支的组合，并非完整一般检测器：partial-grid及其余6类数值/标题产生器、最终页面caption/tail/续表策略、diagram和默认PDF仍待。当前41页实际PDF中空答案为负例，空答案/裁剪阶段网格正例由旧合成源对照提供；图片/签章/Form实际提取未在该套件证明。原默认完成标志保持False。


新增闭合物理网格完整链四个入口：

- `inspect_native_rule_table_geometry(source, *, angle=0, median_height=None)`：自动聚行、表题种子、连通网格及列轨。
- `detect_native_closed_rule_tables(source, *, angle=0, median_height=None, excluded_bboxes=())`：从原始行/规则自动生成闭合表格正式候选，包含表题和自动表注。
- `expand_native_connected_table_grids(source, candidates, *, angle=0, median_height=None, excluded_bboxes=())`：已有候选按真实连通网格扩框、补成员、减注释成员。
- `detect_native_physical_table_candidates(source, *, excluded_bboxes=())`：填充/铺片优先、按方向生成闭合线表、合并和位置排序。

`source`包含`page_size`、`lines`、`drawing_lines`，页面编排还消费`path_infos`和`chars`；可传`form_bboxes/image_bboxes/signature_bboxes`。行沿`build_native_visual_lines`字段，包括text/bbox/source_index/angle/visual_row_id/effective_height/chars和字体元数据；绘图路径沿`read_pdf_vector_geometry`输出。使用原坐标，内部按angle旋转。`median_height=None`自动从局部片段高度计算。检测无需预选core/表题/注释行，返回独立候选及原序整数键`line_bboxes`字典。续表证明仍需显式`source.previous_table_grid={page_index,angle,positions,tolerance}`和当前`source.page_index`，positions/tolerance为页宽归一化列轨信息，末表轨道自动提取尚待。

这四个入口完成闭合物理分支，尚未覆盖一般横线区间、开放/阴影/数值/标题候选等完整页面检测与最终策略，不改变默认PDF入口。24页实际PDF的Form/图片/签章输入为空；排除行为由旧规则事实覆盖。


新增 `expand_native_rule_table_candidate(rows, lines, rule_bboxes, core_row_indices, *, page_size, median_height, angle=0, caption_source_index=None, has_possible_table_notes=True)`，把已完成核心/表题/自动表注组装为正式候选：完整local并集、注释原坐标、体/注释成员分离、bbox/core_bbox方向还原及score0.0默认字段，全部子阶段共享Work。155旧完整普通函数参考（139带注释）与20页实际PDF查询（7带注释）通过Rust验证。本次消掉“正式候选组装与方向还原”；下一项连通网格扩框，自动规则候选产生及页面编排仍待。

返回 `candidate` 字典，字段沿旧候选：bbox/local_bbox/core_bbox、angle/score、line_indices、annotations、inferred_grid及样式/网格默认标志。注释行框是原序整数键字典，frame/core_row_indices/caption_source_index仍为上游事实；接口不接受预选表注行，自动收集表注。has_possible_table_notes=False保留旧负向门控，不读取未消费的行内容。该入口对应普通非prepared/non-owned参考路径，不涉及旧Python对象存储优化或规则临时候选自动产生。

新增 `collect_native_table_footnotes(rows, lines, rule_bbox, core_line_indices, *, page_size, median_height, angle=0)`，完整接通自动表注收集：明确Note/Source/注、辅助Unicode短标记及表体紧凑token/小且上移字形引用、正文字号参照和连续行字体/字号/缩进/距离限制。返回独立footnote_rows及有序annotation，不再要求提供表注行索引。177旧参考/75标记事实与20页实时PDF查询通过Rust验证，包含实际辅助与上标引用正例。本次消掉“自动表注行收集”；下一项把核心、表题和表注组装为正式表格候选并还原方向坐标，页面级候选自动产生仍待。

自动表注接口的rows沿已有局部视觉行/片段字段；lines保留text、bbox、source_index、angle、effective_height、em_height、style_scale_repaired、font_signature、font_coverage和chars（字形bbox/char）。规则框和core来源IDs仍是上游事实，接口独立判定表注起点、辅助标记引用和连续行，无显式注释/引用布尔输入。返回annotation为空时为None，有注释时line_bboxes为原序整数键字典。

新增 `prepare_native_rule_table_annotations(rows, rule_bboxes, core_row_indices, *, median_height, caption_source_index=None, footnote_row_indices=())`，完成临时规则表格的核心边界/成员、表题续行收集和注释记录构造。保留原坐标与local_bbox区别、横向走廊裁剪、表体成员排除、重复来源框合并和注释字典顺序，输入不修改。136旧参考与12页PDF实时查询通过Rust验证。本次消掉核心/表题/注释记录准备；表注行目前显式传入，下一项是自动表注行收集，完整临时候选产生/转正式仍待。

`rows` 为明确的局部视觉行：每行含 `fragments`、`bbox`、`center_y`、可选 `visual_row_id`；片段含 `text`、原坐标 `bbox`、`local_bbox`、来源 `line_index`、可选 `visual_row_id`。`rule_bboxes` 是局部规则边界框；core/footnote索引指向rows数组，caption_source_index则是片段来源ID。返回 `rule_bbox`、`core_local_bbox`、`core_line_indices`、裁剪后的 `caption_rows` 和 `annotations`；注释框使用原坐标，行框字典为整数键并保留插入顺序。这里不自动生成视觉行或识别表题/表注种子，也未输出完整正式表格候选。

新增 `exclude_native_prose_spanning_rule_drafts(source, candidates)`，完成跨正文规则草稿排除：下方网格尺寸/距离、贯穿竖轨和上下横边界，加上草稿成员中至少三排符合词法/字重/语义门槛的长正文，才拒绝草稿；caption草稿与非0方向保留。返回独立原序候选和拒绝原数组位置。151旧参考（112拒绝）与20页PDF实时查询（2拒绝）通过Rust验证。本次消掉“跨正文规则草稿排除”；草稿产生/惰性物化仍待，下一项是物化所需的规则边界、成员和注释准备。

候选投影使用 `kind="table"` 或 `kind="draft"`。table提供bbox/可选core_bbox/angle；draft提供angle（旧context.angle）、median_height（旧context尺度）、caption（None表示不存在）、boundaries（边界bbox数组）、member_source_indices（旧rows.fragments的line_index）。source提供drawing_lines及lines的text/bbox/source_index/angle/caption_start/formula_candidate_only/semantic_type/dominant_font_weight。字段是上游事实，不是冲突结论；接口不调用draft.materialize，也不检测caption存在性。

新增 `merge_native_table_candidates(candidates)`，完成已物化候选合并：分数稳定逆序、同角度首个20%重叠目标、100分强网格拒绝弱候选扩张、动态bbox/local/core并集，以及来源合并、同类注释归并和表体成员优先的注释冲突消解。返回独立candidates，保留目标其他元数据；Python注释行框字典经有序键框对跨Rust桥接，返回整数键字典，原输入不修改。158旧对照（142变化）及28页PDF/44实时检测后查询（24变化）通过Rust验证。本次消掉的是“已物化候选合并”；下一子任务是跨正文规则草稿排除，惰性草稿物化和完整候选产生器仍待。

新增 `prefer_complete_native_table_replacements(base, recovered, replacements)`，完整迁移旧替换仲裁：替代片段与旧核心至少90%较小框交叠，按交集并面积验证共同覆盖，重叠片段不重复计数；先累计所有旧表的拒绝结论，再用最终通过的替代候选删除旧base/recovered。返回原序独立三组候选，保留全部注释/来源/其他元数据，不修改输入。157旧完整参考（153变化/78拒绝）和28页实际PDF/44组实时填充检测后重建查询（24非空变化）通过Rust验证；查询中的半表拆分是测试构造，不是完整年份/数值表候选产生器。一般候选编排、diagram和默认PDF仍待。

新增 `inspect_native_closed_grids(source, candidates, *, allow_two_by_two=False)`，按原候选顺序返回 `closed_grid_flags`。完整迁移旧候选角度/core回退、较大Form排除、原source ID成员、正行高中位数、贯穿横纵轨道、两位小数去重/端点容差和文字占用行列；缺少正行高且实际需要求中位数时传播输入错误。176旧参考（117正例）、24舍入事实与16页实际PDF/32组实时重建查询（7正例）通过定向Rust验证。候选是显式上游输入，真实Form提取未包含；一般表格候选、diagram组件和默认PDF仍待。

闭合网格 `source` 提供 `form_bboxes`、`drawing_lines`（bbox/orientation）和 `lines`（bbox/source_index/effective_height）；候选提供 `bbox`、可选 `core_bbox`、`angle` 和 `line_indices`（原source ID集合）。默认要求至少6个成员、3条纵轨/4条横轨、2列/3行文字占用；显式2x2模式为至少4个成员、3条纵轨/3条横轨、2列/2行占用。成员按source ID匹配，重复源行仍参与计数，不增加框内过滤或全格占用条件。该接口只证明给定候选，不能替代完整表格候选生成。

新增 `detect_native_filled_grid_tables(path_infos, *, page_size, excluded_bboxes=())`，完整迁移旧外框填充行带表格候选：页面比例、最大单元格与半行底纹排除、水平行带数量/覆盖/间隙、垂直覆盖与空带、稳定选优及重叠去重。返回独立完整 `candidates`，保留原bbox数值类型，输入不修改；该函数不依赖CJK字框，也不检查rectangle_bboxes数量。165旧完整参考（101正例）、6整数/混合float算术事实及28页实际PDF实时路径链（4正例）通过定向Rust验证。旋转后只有两条水平带的样本沿旧拒绝；一般表格/闭合网格证明、diagram组件及默认PDF编排仍待。

新增 `detect_native_bar_graphics(source, *, em)`，完成旧原生柱图候选产生器：真实表框/柱组、图题保护及行语义修改、两轮接触柱体/复杂轮廓/栅格/标签扩框，最后执行实际有字网格排除。返回独立 `source` 与原序 `bar_bboxes`，所有子阶段共用累计Work，原输入不修改。另提供 `protect_native_graphic_captions(lines, candidate_bboxes, *, median_height)`，返回独立 `lines` 和 `protected_source_indices`；仅沿旧英文fig/figure数字种子启动。122完整柱图参考、151图题参考及40页实际PDF链已通过定向Rust验证；完成该产生器不等于完整默认PDF编排完成。

柱图 `source` 沿现有页源字段：`page_size`、`chars`、`path_infos`、`drawing_lines`、`image_bboxes`、`form_bboxes` 和 `lines`。普通行提供 `text`/`bbox`/`source_index`/`angle`；为保持多行图题和字高规则，应保留 `font_signature`、`dominant_font_weight`、`effective_height`、`em_height`、`style_scale_repaired`、`semantic_type`、`paragraph_group`、`title_suppressed` 和 `visual_row_id` 等现有元数据。多行强图题可能在返回source中被设为caption、抑制标题并分配段落组；调用方继续使用返回source才能保留旧副作用。完整共享中英文强图题API尚未迁移，当前只实现旧英文种子所需的强图题判定支持域。

新增 `prepare_native_bar_groups(source, *, em)`，在同一累计Work中自动接入相接填充表格排除与柱体准备，再执行共同零轴选择、稳定分类方向排序/切簇、位置和长度差/真实间隔证明、数字行几何准入及零轴补边。返回 `table_bboxes`、`admitted_bars` 和 `groups`；每组含原准入数组 `bar_indices`、原行数组 `numeric_line_indices`、`horizontal`、`baseline`、`core_bbox`、补边后的 `bbox` 和 `label_anchor`。em必须正且有限，输入不修改；此接口尚不执行图题保护、两轮图体扩展及最终有字网格排除，不能称完整柱图检测。187旧参考（152组正例）、30旧round事实和20页实际PDF链（6正例页/10柱组/4表格排除）已通过定向Rust验证。

新增 `detect_native_tiled_tables(source, *, excluded_bboxes=())`，完整迁移旧相接填充单元格候选阶段：真实CJK宽松/紧致框中位数校准、相接矩形组件、行列轨证明、无底纹表头推断网格，以及相邻表格题名转移。返回独立完整 `candidates`，其 `bbox` 可传给柱图准备的 `table_bboxes`；不修改 source。168旧参考（140正例/2题名转移）、12页实际PDF链（4表格及柱体排除正例/1推断网格正例）通过定向Rust验证。测试字体为自行生成的矩形字形，不代表客户文档质量证明；完整其他填充网格族、柱组/图体扩展及默认PDF仍待。

`source` 提供 `page_size`、`chars`（CJK 字符的 `bbox`/`tight_bbox`）、`path_infos`、`drawing_lines` 和 `lines`（`text`/`bbox`/`source_index`/可选 `chars`）。候选返回 `bbox`/`local_bbox`/`core_bbox`、`angle`/`score`、`line_indices`、`annotations`、`inferred_grid` 和网格权威标志；推断网格保留原绘图线的附加端点元数据。可用以下方式连接柱体准入：

```python
tables = detect_native_tiled_tables(source)["candidates"]
prepared = prepare_native_bar_geometry(
    source["path_infos"], source["drawing_lines"], em=em,
    table_bboxes=[table["bbox"] for table in tables],
)
```

该连接验证表格填充块排除；完整柱组及图体扩展仍待。

新增 `prepare_native_bar_geometry(path_infos, drawing_lines, *, em, table_bboxes, axis_queries=())`，返回与原 path 同序的 `compound_baselines`、含 `path_index`/`rectangle_index`/`bbox`/方向标志的 `admitted_bars`，以及与查询同序的 `zero_axis_bboxes`。查询包含 `bounds`、`horizontal` 和 `baseline`；可直接使用 `read_pdf_vector_geometry` 的路径和绘图线。`table_bboxes` 为必填上游填充表格框；当前接口只处理基线/柱体准入/零轴扩边，尚非完整柱图检测。170 旧参考和 8 页实际 PDF 链通过。

新增 `detect_native_raster_diagrams(source, *, em, frame_bboxes=())`，source提供page_size、drawing_lines、image_bboxes，以及网格/文字框证据所需lines/path_infos/form_bboxes。返回 `raster_diagram_bboxes` 和与查询frame_bboxes同序的 `plain_text_frame_flags`。连接图候选内部执行真实有字网格保护；frame查询可来自绘图组件bbox。144旧对照、20页实际PDF链；em显式输入，不修改页源，完整原生diagram/bar集合及默认PDF编排仍待。

新增 `detect_native_concentric_graphics(source, *, em)`，source提供page_size、lines（text/bbox/angle）、path_infos，返回 `concentric_bboxes`。按旧编号英文fig/figure图题、百分比标签、同心描边轮廓及引线证明候选，不修改输入。随后可交 `arbitrate_native_graphic_candidates` 执行有字网格保护/重叠仲裁。196旧参考、8页实际PDF全实时链；em显式输入，完整强图形集合及默认PDF编排仍待。

新增 `arbitrate_native_graphic_candidates(source, candidate_bboxes, *, median_height)`，返回 `graphic_bboxes` 和 `grid_rejections`（按面积稳定排序后的候选位置/网格框证据）。`inspect_native_graphic_grids(source, regions, *, median_height)` 返回各原区域的 `grid_bboxes`。source提供page_size、lines（bbox/angle/visual_row_id）、drawing_lines、form_bboxes；网格由实际绘图和文字证明，先排除表格再做图形包含/重叠合并。候选可来自已迁移的组件/路径等产生器，完整强图形集合和默认PDF编排仍待。178旧对照和8页实时PDF链已纳入验证，数值均值使用缓存精确比率依赖，无Python图像/数值运行库。

新增 `detect_native_isolated_graphics(source)`，source提供page_size、lines、path_infos、image_bboxes。返回 `em`、`path_groups`（原path数组位置）、`vector_bboxes` 和 `meaningful_raster_indices`（原图片框数组位置）。按旧正文尺度/路径段数面积/覆盖/颜色证据判定，无行时矢量em默认10，小栅格必须有正向正文尺度；不修改输入，不执行完整强图形/表格仲裁。229旧对照及8页专用真实PDF链已纳入验证，完整默认PDF编排仍待。

新增 `detect_native_drawing_graphics(drawing_lines, path_infos, *, page_size, median_height)` 返回 `tolerance`、`complex_path_indices`、`components`、`graphic_bboxes`。每个组件含原 `line_indices`、`bbox`、`horizontal_count`、`vertical_count`。可直接传入 `read_pdf_vector_geometry` 输出，保留原组件顺序、重复线和框数值类型；不修改输入。最多各5000线/路径及累计扫描预算，median_height显式传入。239旧对照及8页专用PDF链已纳入验证；完整强图形/表格仲裁和默认PDF编排仍待。

新增 `detect_native_path_graphics(path_infos, *, page_size, median_height)` 返回 `complex_path_indices`（原数组位置）、`container_bboxes`、`axis_bboxes`。可直接传入 `read_pdf_vector_geometry` 的路径和page_size，median_height为调用方现有图形尺度；候选保留旧顺序和重复，框union保留原数值赢家/类型。最多5000路径及5000输出框、累计扫描预算。263旧对照和8页实际四方向正负链；完整强图形/表格仲裁和默认PDF编排仍待。

新增 `detect_native_stacked_graphics(source, *, em, fallback_em)` 返回 `graphic_em`、`inflated_cjk_metrics`、`stacked_column_bboxes`。source可提供chars（原字框及tight_bbox）、lines、path_infos、drawing_lines。em为现有柱图距离尺度，fallback_em为图形尺度原值；接口给出旧尺度证据和原顺序柱图候选，不修改source、不执行完整图形/表格仲裁。候选输出float PDF point。304旧对照含162柱图正例；41实际PDF输入仅负例，完整默认流程仍待。

新增 `group_native_raster_chart_descriptions(source)`，source提供lines及image_bboxes（可缺省为空），可接已完成图片过滤和噪声处理的页源。返回独立 `source`、`graphic_bboxes` 和 `axis_groups`（vertical/line_indices，身份按原行数组位置）。按旧等差刻度和图片邻接证明图体，归组上方同字体的长说明/短续行，只修改paragraph_group。272组旧对照、16页真实链参考；完整强图形检测及默认PDF编排仍待。

原生噪声：`prepare_native_text_noise(source, *, visual_bboxes=())` 返回数组位置候选和需读颜色的字符索引；`read_pdf_text_paint(pdf, indices, *, page_index=0, pdfium_library=None, password=None)` 返回以字符串索引为键的RGBA数组映射，描边/裁剪文字和无效索引不提供证据；`apply_native_text_noise(source, *, paint=None, visual_bboxes=())` 返回 `source`、`excluded_source_indices`、`requested_char_indices`。也可用 `filter_pdf_native_text_noise(source, pdf, ...)` 自动串联，无候选时不访问PDF。source包含lines/chars，删除沿旧来源号/字符索引语义，返回独立值；缺少颜色则保留原文。强图形框仍需调用方提供，此阶段不包含完整图形检测/图注归组或默认PDF编排。

新增 `read_pdf_image_infos(source, *, page_index=0, pdfium_library=None, password=None)`，返回 `page_size`、`page_bbox`、`rotation`、`image_infos`；每张图包含视觉 `bbox`、可空原始流 `fingerprint`、`smooth_background` 和可空 `blank_top_bbox`。可直接给 `prepare_native_source_lines` 和 `prepare_native_image_regions` 使用，取消图片证据手动传入的需求。无需Python图片运行库。每图原始流超过16MiB不生成指纹；累计64MiB原始流/64M解码像素及共享缩放预算超限报错。该组合已验证局部页源链，完整默认PDF编排仍待。

`prepare_native_image_regions(pages)` 对已提取的图片证据执行旧全文水印/正文背景/标题白边决策，返回 `watermark_fingerprints`（排序列表）和 `image_bboxes`（逐页列表）。每页提供 `page_size`、`lines`、`image_infos`；图片字段为 `bbox`、可空 `fingerprint`、`smooth_background`、可空 `blank_top_bbox`。行需文字和框，可带 `angle`、`effective_height`、`dominant_font_weight`。保留输入所有权和原始框数值类型。328组旧对照；图片证据自动读取仍待，此接口不表示默认PDF流程已接通。

脚本证据生成、最终投影和诊断接口已安装验证：249完整输出/诊断、516 sidecar、57实时PDF负例链及四方向专用PDF/7处实际脚本通过；五项受影响Rust测试和五组安装套件通过。旧apply默认保持，字段/类型/顺序、所有权/并发/限额/错误恢复核对通过；fmt、82模块审查和固定36＋3＋1哈希保持。本阶段一次wheel构建，无全workspace/全部100组runner/未受影响渲染/全量clippy/CUDA。日志build/native-script-output-install-test.txt和build/native-script-output-regression.txt。

新增prepare_native_script_lines(lines, *, page_size, tight_bboxes=None, origins=None, all_chars=(), drawing_lines=())，从视觉行和原始几何生成script_lines；复用既有角色/分式分类，单行≤1500字符、side-map≤20000，分式原语沿已有1500字符/5000规则上限。已有视觉行来源号和紧框类型保留。

apply_native_inline_evidence增加可选script_lines=None，省略保持旧默认；传数组时执行完整脚本投影，空数组也清理旧_inline_math_regions。新增trace_native_inline_evidence(blocks, *, page_size, style_lines=(), link_lines=(), script_lines=())返回{"blocks":...,"materialized_ranges":...}，诊断含实际通过映射的原字符串/证据区间、来源、角色、框、角度与稳定body计数。主/备用归属、独立短脚本匹配及晚期公式保护按旧规则；输入保持不变。

249旧完整输出/诊断、516旧payload事实和五项受影响Rust测试通过；57实际PDF负例链，另四方向专用实际PDF产生7处物化脚本。44预分类分式输入仍由原560行精炼参考独立核对。完整页源/块候选检测/默认PDF流程仍待，标志False；开发仅定向测试，阶段末相关安装回归，无全量/CUDA。

样式/链接到InlineSpan输出接口已安装验证：297旧完整输出、57实时PDF字符＋注解链和五组相关安装套件通过；两个定向Rust测试、fmt、81模块审查及固定36＋3＋1哈希保持。一次阶段wheel构建，无全workspace/全部99组runner/未受影响渲染/全量clippy/CUDA。日志build/native-inline-output-install-test.txt和build/native-inline-output-regression.txt。

新增apply_native_inline_evidence(blocks, *, page_size, style_lines=(), link_lines=())，接收finalized块（单位bbox或page point）和detect_native_inline_evidence输出，返回内容已物化为文字/公式/超链接InlineSpan的新块。按旧几何归属、公式空洞/断词/上下文和冲突规则投影，不修改输入。已有detector-owned平坦sup/sub标记会按旧规则转换；新script_lines检测结果的投影暂未接入。

输入限额4000块/各4096证据行/每行5000良构范围，坐标abs≤1e9、8MB输出与既有JSON/共享10M Work预算，超限明确报错。link_lines目标须符合PDF来源四种外链协议，任意Python私有区间对象不在JSON接口支持域。297旧完整输出含57实际PDF链已通过定向Rust对照；这些块仍是明确阶段输入，不等于整页块检测或完整默认PDF迁移。

PDF链接注解接口已安装验证：207旧URI/36实际页26注解、57实时字符＋实时注解→视觉行/局部准备/inline链及五组相关安装套件通过，取消该链的注解注入。流所有权/并发/加密/错误恢复及作者URI/Quad资源限额通过；两个定向Rust测试、fmt、80模块审查和固定36＋3＋1哈希保持。本阶段一次wheel构建，无全workspace/全部98组runner/未受影响渲染/全量clippy/CUDA。日志build/pdf-link-annotations-install-test.txt和build/pdf-link-annotations-regression.txt。

新增read_pdf_link_annotations(source, *, page_index=0, pdfium_library=None, password=None)：直接读取外部URI注解，返回target/bboxes/source_index独立字典，保持跳过隐藏/非法动作/目标后的原枚举号。QuadPoints优先、去重排序、无效时回退Rect，bbox为应用CropBox/旋转后的视觉PDF point。结果可直接传detect_native_inline_evidence的annotations，取消手工注入注解。

遵循旧URI策略（http/https/mailto/tel及Unicode16/urllib边界），不访问链接。每页枚举20000/有效5000、单URI1MB/累计URI8MB、单链接5000 Quad/累计区域20000，资源超限明确抛出。207旧URI/36实际PDF页参考已通过定向Rust对照；本接口不负责像素字重、最终样式/链接投影、候选检测或完整默认PDF解析，两个legacy标志仍False。

原生样式/链接检测已安装验证：285旧事实、57实时PDF字符链及五组相关安装套件通过；两个定向Rust测试、fmt、79模块审查和固定36＋3＋1哈希通过。多字符标记差异已由新增旧参考复现并修正，最终wheel安装通过，未跑全workspace/全部97组runner/未受影响渲染/全量clippy/CUDA。日志build/native-inline-evidence-install-test.txt和build/native-inline-evidence-regression.txt。

新增detect_native_inline_evidence(lines, *, drawing_lines=(), annotations=())，返回独立style_lines/link_lines。字符使用原font/char_idx/bbox/tight_bbox，drawing_lines为bbox/width/orientation，annotations为target/bboxes（外部URI注解阶段输入）。样式输出bold/underline/strikethrough，链接冲突字符保留普通文字；所有区间按规范化文本的Unicode码点计数，不是UTF-8字节。

285旧完整参考含57实际PDF字符→视觉行→局部准备→检测链；SDK注解读取、像素字重补证及将证据投影到最终输出仍待。支持4096行/每行20000字符/单片段256字节，绘图/注解各5000、总链接区域20000、8MB输出及既有JSON/共享10M Work上限；有限坐标abs≤1e9、source_index为i64/布尔，资源异常明确传播。两项定向Rust测试通过；默认PDF完成标志仍False，CUDA检查暂停。

页源局部准备接口已安装验证：290旧完整事实、57实时PDF读取链及五组相关安装套件通过；两个定向Rust测试、fmt、78模块审查与固定36＋3＋1哈希通过。本阶段只构建一次wheel，未跑全workspace/全部96组runner/未受影响渲染/全量clippy/CUDA。日志build/native-source-lines-install-test.txt和build/native-source-lines-regression.txt。

新增prepare_native_source_lines(lines, *, page_size, drawing_lines=(), image_infos=())，输入上一阶段视觉行，输出独立{"lines":...,"drawing_lines":...}。依旧规则恢复下沉首字母正文/悬挂字母段，追加极薄水平栅格图片和页顶重复私用区装饰文字分隔线；保留字符、来源、框和行序，仅修改旧分组/标题抑制等字段。image_infos只需提供bbox，drawing_lines为bbox/width/orientation；它们仍是明确阶段输入。

290旧完整事实含57实际PDF字符→视觉行→本阶段链、22装饰线和16薄图线输出。支持4096行/每行20000字符、消费的drawing_lines/image_infos各5000、输出10000线/8MB及既有JSON限制，共享10M工作预算超限明确报错。完整页源其他准备/候选检测/默认PDF接入尚待，两个legacy标志仍False；开发定向验证，阶段结束构建一次wheel，CUDA检查暂停。

视觉行新接口已安装验证：317旧事实/977行、57实时PDF读取链及五组相关回归通过（共六组安装套件）。定向Rust/fmt/77模块审查及固定36＋3＋1哈希通过；本阶段构建一次wheel，未跑全workspace/全95组runner、未受影响渲染、全量clippy或CUDA。日志build/native-visual-lines-install-test.txt及build/native-visual-lines-regression.txt。

新增build_native_visual_lines(chars, *, page_size, page_rotation=0, supported_angles=(0.,90.,270.))：从原生字符字典自动重建视觉行，返回包含原字符、稳定source_index、row/run、source_bbox和排版字段的独立LineItem JSON。包含旧分行/方向筛选/视觉间隙、控制字符/Zs清理及跨粗行上下标链合并。可衔接read_pdf_character_primitives的restored_chars，或read_pdf_native_text_geometry的去重chars；page_size/rotation使用读取结果对应字段。

最多4096字符、单字符256字节、单行排版1500字符，有限坐标abs≤1e9；既有JSON限制和组行/配对共享10M预算超限明确报错。317旧事实/977行含57实际PDF输入；页源的其他准备、候选检测及默认PDF接入仍待，不能据此视为整页完整解析。按用户要求暂停CUDA检查，开发只运行受影响测试，阶段结束统一构建/相关回归。

本轮11组相关安装套件及36组PDF像素对照通过，未全跑94组runner；日志build/native-table-media-png-regression.txt。完整页级来源/候选检测及高层默认接入仍待实现。

媒体PNG检查点已验证：232项Rust测试、fmt/严格clippy/CUDA feature编译及76模块审查通过。已安装wheel对照182 PNG页/502引用和45真实PDF/125引用，解码模式/尺寸/像素/ICC/tRNS与其余HTML/JSON一致；通用PDF渲染18 RGBA＋18 RGB组重新对照通过。固定36＋3＋1上游源码哈希保持。下方各旧检查点的测试数字及当时待办为历史记录。

新增媒体PNG入口：`materialize_native_table_media_page(blocks, page_png=None, *, page_size=None)`使用调用方PNG，None表示图像不可用并回退文字；`materialize_native_table_media(pages, source, *, page_sizes=None, pdfium_library=None, password=None, max_pixels=16_000_000)`对每个有媒体的PDF原页渲染一次（RGB，scale2/长边3500）。pages须按原PDF页号排列，未参与页放[]。无媒体不读取来源。返回独立blocks/pages，保留已生成图片，对页内失败后的剩余占位符转义文字回退，清理非空_table_images。

默认媒体bbox为单位坐标；从materialize_native_table_blocks返回的point框衔接时，单页提供page_size，整PDF提供对应page_sizes（空页可为None）。仅媒体裁剪框使用旧千分刻度/ties-even/最小刻度转换，其余字段保持。支持8bit L/LA/RGB/RGBA及ICC/tRNS，源PNG64MB/16MP、单裁图4MP、累计64MP裁图/64MB PNG/64MB输出和10M文字扫描上限，资源错误不会隐藏为回退。PNG压缩字节/data URI以及URI哈希文件名可与Pillow不同，解码模式/尺寸/像素/元数据及其余HTML保持旧参考。

独立冻结182 PNG页/502引用、45真实PDF/角度/事务链/125引用。验证：`build/rust-wheel-env/Scripts/python.exe -I bindings/python/tests/native_table_media_png_install.py .`，安装测试不依赖Pillow。完整页源/候选检测和默认集成仍待完成，两个legacy标志继续False。

完整块事务最终验证：231项Rust测试、fmt/clippy/CUDA feature编译、75模块来源审查及固定36＋3＋1哈希通过。已安装wheel通过781完整旧事务、八候选缓存、16实时PDF/角度串联，以及类型/副作用/所有权/并发/累计预算/异常恢复检查；12组相关安装套件通过。96独立媒体事实由Rust核对，安装包通过24完整媒体事务。日志build/native-table-block-commit-install-test.txt与build/native-table-block-commit-regression.txt。未全跑93套件/36像素组或实测GPU；实际PNG编码、页级自动接入和默认启用继续待办。

新增完整块计算接口 `materialize_native_table_blocks(source, candidates, *, tight_bboxes=None, origins=None, calibrated_char_bboxes=None)`。source/candidates为已检测的_PageSource/_TableCandidate中立字典，支持可选path_infos、form_bboxes、image_bboxes及page_index；坐标保持阶段原单位。返回table_blocks、annotation_blocks、claimed_line_indices和独立lines/candidates副作用。稳定排序/完整候选框去重、真实注释、原生/语义HTML、弱表撤销、原始字符投影和原子认领已接通。候选逆向框保留旧行为；字符和校准框仍有序。来源认领始终指LineItem.source_index。

媒体选择/遮罩/唯一cell归属返回临时`docvortex-table-image:` URI及_table_images的原文字回退元数据，尚需后续PNG解析阶段；页级来源/候选检测和默认PDF接入仍待。781旧完整事务、1组八候选缓存、96媒体几何事实和16实时PDF/角度计算串联独立冻结；实际媒体检测/PNG效果未据此证明。128候选、每类媒体集合5000、4000块/8MB块序列化、4MB HTML和原JSON限额；新编排共用10M Work，资源异常传播而非降级投影。两个legacy完成标志仍False。验证入口：`build/rust-wheel-env/Scripts/python.exe -I bindings/python/tests/native_table_block_commit_install.py .`。

注释检查点的10组相关安装套件全部通过，见build/native-table-annotations-regression.txt；未全跑92组或36像素组。模型PDF编排中的冻结输入与真实CPU烟测分别保留，不能据此称完整原生PDF迁移完成。

注释检查点验证：229项Rust测试、fmt/clippy/CUDA feature编译、74模块来源审查及固定36＋3＋1哈希通过。最终安装wheel的928注释/类型、240首页面板、16实时PDF负例及所有权/并发/累计预算/错误后恢复/许可证检查通过；日志build/native-table-annotations-install-test.txt、build/native-table-annotations-wheel.txt。保留GPU实机、投影降级、媒体和最终提交待办；下方历史检查点数字按各自阶段记录。

新增 `prepare_native_table_annotations(source, candidate, *, calibrated_char_bboxes=None)`，自动完成块准备和注释物化，返回已有准备字段及annotation_blocks、externalized_line_indices、failed_annotations，使用真实失败集合回并表体并扣除外置成员。`prepare_native_table_block` 的显式失败输入接口保留。新注释输出保留字体重启、物理行连接、tight框、局部行框/行高/字体集合等元数据；字体集合为排序唯一JSON列表。

`restore_native_front_matter_panel(source, bbox)` 只能在结构恢复失败后调用，返回restore与独立lines；不会自行感知或覆盖成功恢复。source.page_index=0时按旧两类元数据/侧栏正文/摘要和真网格保护判定，成功时保留标题标记副作用。Unicode NFKC/casefold及词边界固定16.0.0。904注释＋24类型、240首页面板/102撤销和16实时PDF无注释负例独立冻结；正例为明确阶段输入，尚不证明整文档质量提升。验证入口：`build/rust-wheel-env/Scripts/python.exe -I bindings/python/tests/native_table_annotations_install.py .`。输入限额沿旧准备，最多4000输出块/8MB输出序列化，准备与注释共享10M预算；最终投影降级、媒体和原子提交仍待接入。

块级准备检查点共9组已安装套件通过（新准备、整表、二维簇、脚本准备、引擎、去重、CPU PDF编排、来源审查、smoke）；日志build/native-table-block-regression.txt。未运行全部91套件或36像素组。最终注释输出、投影回退、图片归属与原子认领提交尚待接入。

块级准备的 `projection_line_indices` 是 `_LineItem.source_index` 集合，既不是字符 `char_idx`，也不是 lines 的列表位置。`calibrated_char_bboxes` 的键才是字符 `char_idx`；缺少该字段的字符不应用校准映射。

2026-10-26 新增 `prepare_native_table_block(source, candidate, *, failed_annotations=(), calibrated_char_bboxes=None)`。source 含 page_size、lines、chars、drawing_lines；candidate 为旧_TableCandidate中立字典，annotations与line_indices集合映射为JSON列表。返回候选、projection_line_indices、body_bbox、校准框映射及native_chars；不修改输入，显式校准覆盖自动校准。只执行字符框校准、表内脚注分离、成员扩展和失败注释边界回并，failed_annotations仍需由后续注释物化阶段提供，尚不提交输出认领。

612旧准备＋12类型事实包含53脚注分离、207成员扩展和46校准案例；16实际PDF／角度的准备→恢复→整表HTML参考中12非空。PDF行是既有glyph视觉行的显式输入，字符/绘图可实时从SDK读取。4096唯一来源行、20000字符/成员/框映射、5000规则、2000注释、abs坐标≤1e9及原JSON上限；各准备步骤共享10M预算。完整PDF默认开关保持False。新增安装验证：`build/rust-wheel-env/Scripts/python.exe -I bindings/python/tests/native_table_block_install.py .`。

块级准备wheel验证：228项Rust测试、fmt/clippy/CUDA feature编译和73模块来源审查通过。624准备/类型事实、16实时PDF/角度串联（12非空）、所有权/并发/累计10M预算/限额/异常恢复和许可证检查通过。非空PDF行由SDK实时字符重建glyph视觉行，4空行阶段输入用作负例，仍不构成完整行检测器验证。全量runner现有91组，本轮未全跑；GPU尚未实机验证。

历史整表wheel验证：227项Rust测试、13组相关安装包套件及72模块来源审查通过；526旧整表、724恢复、32实时PDF组合及预算/异常边界均通过。

2026-10-26 新增整表接口：

```python
from rapid_doc_rust import recover_native_table_result, render_native_table_html_with_scripts

# table_input 为已有字符/绘图接口输出组成的 NativeTableInput 中立字典。
value = recover_native_table_result(
    table_input, tight_bboxes=tight_bboxes, origins=origins,
    preserve_font_styles=False,
)
if value is not None:
    content, result = value  # 语义 HTML；result 保留原始 html 与内部结果
```

`render_native_table_html_with_scripts(result, table_input, *, tight_bboxes=None, origins=None, preserve_font_styles=False)` 可单独渲染中立结果。缺少任一侧映射时按旧行为返回原HTML，包括 preserve_font_styles=True。恢复错误为 `NativeTableRecoveryError`（RapidDocError子类，code及stage="recovery"），渲染错误仍为 RapidDocError。输入、结果与映射均为普通 JSON 字典；侧映射键可为整数来源ID或其十进制字符串。

新链路已有526旧整表HTML、724原始输入恢复、32实际PDF/角度/样式组合参考；168上下标正例为中立结果阶段事实。资源限额沿用组件限制，新增共享10M编排预算与4MB整表HTML上限。PDF块级接受/回退/资产物化仍待接入，`legacy_native_table_recovery`及`legacy_table_execution`继续False；下方为历史检查点。

2026-10-26 二维簇与多行cell角色：

新增`merge_native_inline_clusters(lines, *, page_size, table_bboxes=())`与`classify_native_table_script_cell(lines, *, page_size, tight_bboxes=None, origins=None, fraction_members=())`。完整移植旧二维inline簇的栏内连接、表格/图题/尺度跨栏保护、行内/紧凑公式簇判断、跨run成对角标补证、宿主评分和物化；宿主支持量复用CPython补偿sum，保留float(source_index)的平局和按源序拼接、结构归属、公式区域及排版重填。

合并返回`lines`与`groups`：lines是36个消费字段的普通JSON投影（32默认来自原_LineItem），groups保存每个输出簇的原输入行位置；不按source_index或字符ID解释。投影不包含私有缓存及其他LineItem字段，native_math_words用排序去重列表表示。单元格接口实际执行二维合并→脚本判定→词首抑制→连续同源角色冲突聚合，最终输出char_idx来源角色；跨行冲突不能先逐行删除body状态再合并。

冻结910组旧完整二维合并（262实际合并、18成对补证、155旧区间索引分支）、750组完整cell角色（428多行输入）及480叶子HTML链（160多行，无脚本标签）；另88组带标签正例由显式视觉行阶段输入构成，不当作实际PDF正例。48组原始字符→真实旧glyph生成→准备→合并/角色→HTML调用链均无脚本，用作误标负例。16空/控制字符簇与完整行边界、9分式尺度及48完整分段参考定位并修正了空字符串被错误缓存为不可打印的问题；保留旧Python空文本对邻接、可见列表和统计的影响。

空或非空批次≤512行，source_index为唯一i64、角度0/90/180/270；合计chars≤1500、表格屏障≤5000，输入及合并后单行公式区域≤2000，有限坐标abs≤1e9。序列化输入成本8MB及既有JSON节点/2MB文本上限保持；栏推断、连接、宿主、配对和脚本规模预收费共享10M Work，行内精炼保留自身预算。通过503行密集布局和转义文本膨胀验证超限明确报错、后续调用恢复，不降级成“普通未接纳”。

验证记录：225项Rust workspace测试、fmt、严格clippy和CUDA feature编译通过；新wheel已安装，新增合并/角色/HTML及空字符事实的逐值/类型、原输入成员完整性、所有权/并发、10M累计预算、转义文本序列化8MB限制、异常后恢复和许可证检查通过。9组相关安装包回归通过（新二维簇＋栏带＋表格准备＋脚本精炼＋基础脚本/叶子HTML＋完整表格引擎＋model_document＋来源审查＋smoke）。model_document仍为旧固定版面/公式和真实CPU OCR/SLANet的PDF多页编排对照；不据此承诺整文档效果改善。71模块来源审查、36 core＋3 paths＋1 scripts原样源码哈希通过；本轮未重跑全部89组安装包和36像素对照，历史全量记录保留。日志：build/native-inline-clusters-rust.txt、build/native-inline-clusters-install-test.txt、build/native-inline-clusters-regression.txt、build/native-inline-clusters-regression-run.txt、build/native-inline-clusters-wheel.txt和build/rust-current-*.txt。GPU未作实机测试。

下一步：接整表render_native_table_html_with_scripts、recover_table_result及PDF级原生/模型回退编排，再做整文档效果和性能验收。二维簇与多行cell角色阶段现已接通；完整整表HTML编排和默认替换尚未完成，legacy_native_table_recovery/legacy_table_execution仍False。visible-only、enhanced阅读顺序模型及GPU实机运行继续延后。

2026-10-26 完整原生栏带推断：

新增`infer_native_text_lanes(line_geometry, *, local_page_width, median_height, recalculate_intervals=True)`。完整移植旧`_infer_text_lanes`编排，保留锚点资格、贪心中位数聚类、局部双栏候选/稳定平局、成员分配和可选栏宽扩张、跨栏续行、重复缩进及同高短尾批量结算。复用固定vendor原样rule_text.rs的三个数值内核，Rust适配入口另校验空集合、非有限数值和非法栏索引；没有新写栏位启发式替换旧规则。

冻结1128组旧完整调用参考，含182组局部双栏命中、24组重复缩进状态变化、47组跨栏短尾迁移；语料包含随机字体/尺度/语义、正文/通栏/局部双栏和已有旧表格视觉行。分支记录通过观察旧函数调用/返回获得，不替换分类器或注入栏位；每条输出的成员位置、顺序、栏边界、is_span和类型逐项对照。

输入为非空≤512条`(line JSON, 已转正bbox)`，source_index为唯一i64、同一行只出现一次；几何有限正向且abs≤1e9，字体名≤4096字节，字体签名使用普通二元JSON。字重只要求有限，不套坐标上限。返回`left/right/is_span/line_indices`，line_indices是原输入列表位置，不能按source_index或字符来源ID解释；保留旧空栏及成员顺序。内核规模预收费与外层簇比较、前序/并行行扫描、迁移共享10M Work预算，沿宿主JSON节点/2MB文本限制。

验证记录：222项Rust workspace测试、fmt、严格clippy和CUDA feature编译通过；新wheel已安装，新增1128组完整栏带逐值/类型对照、成员完整性、重复调用、所有权/并发、输入限制、503行有效密集布局累计10M预算拒绝与错误恢复通过。8组相关安装包回归通过（新栏带＋表格准备＋脚本精炼＋基础脚本/叶子HTML＋完整表格引擎＋model_document＋来源审查＋smoke）；model_document为旧固定版面/公式与真实CPU OCR/SLANet的PDF多页编排对照。70模块来源审查、随包许可证及36 core＋3 paths＋1 scripts原样源码哈希通过。本轮运行定向安装包回归，未重跑全部88组；历史87组全量及36组像素对照记录保留于上一检查点。日志：build/native-text-lanes-rust.txt、build/native-text-lanes-install-test.txt、build/native-text-lanes-regression.txt、build/native-text-lanes-regression-run.txt、build/native-text-lanes-wheel.txt和build/rust-current-*.txt。GPU未作实机测试，未承诺整文档质量/速度/内存提升。

下一步直接接旧二维重叠簇的连接、宿主选择与合并，再完成多行cell角色聚合、整表语义HTML与recover_table_result。当前栏带推断已完成，二维行合并仍未完成；默认PDF模型路径和legacy_native_table_recovery/legacy_table_execution=False不变。visible-only、enhanced阅读顺序模型及GPU实机运行继续延后。

2026-10-26 表格视觉行和排版准备：

本轮新增`measure_native_line_typography`、`prepare_native_table_script_cells`和`classify_native_table_script_line`。按旧Python计算裁页/旋转后的字高、字宽、主字体及覆盖率、主字重、行首强调/异族宽度和缩小引用句尾；字体平局保留首项，Unicode16全部1557个casefold映射固定随包。保留旧版空文本isprintable为真、合法字框仍参与统计的细节。

表格准备保持char来源首项/glyph来源末项、visual_row/x/y/glyph_id稳定顺序、多行才填排版、cell边界横线排除和分式来源保护；单行专用分类继续调用实际脚本流程，再抑制偏移词首并聚合同源冲突。准备接口返回fraction_rules和cells，每格包含glyphs/chars/visual_lines/fraction_members；visual_lines尚未二维合并。fraction_members与角色字典均用char_idx来源ID，不是列表位置。

988组旧排版度量（836常规＋112边界＋40既有em字段值/类型保留）、171组完整准备，以及3978组实际准备→单行分类→叶子HTML旧参考已冻结；其中168组上下标输出、43个分式保护cell、159组规则筛选场景（其中138组实际排除cell边界网格线）。单行路径不需要二维合并，不把这些对照当作多行整表自动语义完成。普通NativeTableInput/Result JSON要求cell(row,col)唯一；单行/单格≤1500字符/字形，页面≤20000字符/3000字形/2000cell/5000线，累计选择字形≤10000，有限坐标abs≤1e9。字体flags支持i64/u64及可转换文本，超范围报资源错误。网格比较与跨cell分式乘积共享10M预算，重复字形/字符累计序列化成本8MB在克隆前限制；宿主JSON预算不变，MIT/Unicode归属随包。

验证记录：220项Rust workspace测试、fmt、严格clippy和CUDA feature编译通过；最终wheel已安装到build/rust-wheel-env，87组安装包回归及18 RGBA＋18 RGB旧PDF精确像素对照全部通过。新增5137组事实（988排版＋171准备＋3978单行串接）逐值/类型、所有权、并发、共享10M工作预算及重复克隆8MB上限、异常恢复和随包许可证核对通过。69模块来源审查与36 core＋3 paths＋1 scripts原样源码哈希通过。日志：build/table-script-preparation-rust.txt、build/table-script-preparation-install-test.txt、build/table-script-preparation-wheel.txt、build/rust-current-*.txt和build/wheel-all-tests.txt。GPU仅编译检查，未作实机运行或性能/内存改善承诺。

下一步：移植旧text-lane推断与二维簇合并，接通多行cell角色聚合、整表render_native_table_html_with_scripts和recover_table_result，再接PDF级回退编排。默认PDF模型路径不变；legacy_native_table_recovery/legacy_table_execution仍False。visible-only、enhanced阅读顺序、GPU实机运行继续延后。

2026-10-26 单行语义与分式保护：新增`refine_native_script_tokens`、`classify_native_script_runs`、`classify_native_script_line`和`native_fraction_member_sources`。按旧Python移植引用/数学索引精炼、公式区域连续分段、首字母纠正和结构补证，单行路径实际调用基础分类后再精炼，不注入预分类角色。分式几何复用原样vendor的rule_text.rs；最终分子/分母恢复body，body_counts/formula_flags保持旧版最终覆盖前值。

冻结1838组精炼、2757组完整分段、560组完整单行、843组分式（255正例）以及20组真实分式检测→单行保护链；20组均实际改变角色，覆盖0/90/180/270度、来源ID不等于列表位置、旧空间索引分支及prepared零高分数线边界。数学标识政策按旧函数固定Unicode16的232区间。来源SHA随参考数据记录，独立wheel不导入旧Python。

接口约定：roles输出按chars位置；单行输出chars按旧规则排序后与roles对应；fraction输出及fraction_members输入均为char_idx来源ID。单行普通JSON支持chars、angle、inline_math_regions、compact_formula_cluster/restored_inline_cluster；side-map键是来源ID字符串。最多1500字符、5000规则、20000 side-map/保护ID、单字符文本256字节；坐标有限且abs≤1e9。精炼比较10M预算，分式规则×字符10M上限，沿既有JSON节点/2MB文本预算；超限显式错误。MIT/Unicode归属随wheel分发。

验证记录：218项Rust workspace测试、fmt、严格clippy与CUDA feature编译通过；新wheel已安装到build/rust-wheel-env，全部86组安装包回归及18 RGBA＋18 RGB旧PDF精确像素对照通过。新增6018组旧版事实（1838精炼＋2757分段＋560单行＋843分式＋20联动）逐值/类型、所有权、并发、输入限制和许可证检查通过。68模块来源审查及36 core＋3 paths＋1 scripts原样源码哈希通过。日志：build/script-refinement-rust.txt、build/script-refinement-install-test.txt、build/script-refinement-wheel.txt、build/rust-current-*.txt和build/wheel-all-tests.txt。CUDA仅编译检查，GPU运行仍待服务器验收。

剩余优先项：二维行合并、单元格视觉行/字体构建、表格网格线过滤/角色聚合/偏移词前缀、整表`render_native_table_html_with_scripts`和`recover_table_result`编排，再接原生/模型回退。当前完成单行语义，尚未自动生成整表语义HTML；默认PDF模型流程不变，legacy_native_table_recovery与legacy_table_execution仍False。visible-only、enhanced阅读顺序和GPU实机测试继续延后。

2026-10-25 语义HTML基础组件：新增`classify_native_script_roles`、`native_numeric_script_indices`与`render_native_table_styled_cell`。前两者移植旧_script_geometry基础几何分类及数值上标补充，按loose/tight/origin、保护位置和原始字体身份判断；第三个是给定来源角色的旧单元格片段渲染，保留转义、紧框补空格、None来源样式继承和文字不匹配回退。渲染只生成固定sup/sub/b/i标签，preserve_font_styles默认False。

基础分类复用本地DocVortex b2ef8d0的一份原样scripts.rs，补齐CPython整数set槽位等距平局；原36份core/3份paths不变。1536基础分类参考含64实际PDF/模式几何、72等距平局，另302显式已有角色的数值补充（42正例）；142975非零Unicode16标志与全部1112064标量的零/非零标志逐项核对。1288叶子渲染参考含1174上下标输出、626字体标签输出；另390字体元数据转换参考，修正了大整数标志低位丢失、浮点字符串误作int及大字重受无关坐标上限限制造成的118条差异。叶子中的角色是该阶段显式输入，不把这些结果作为整表自动分类已经完成的证据。

接口约定：分类输出与输入chars的位置一一对应；protected_body_indices和数值补充输出均为列表位置，不是char_idx。渲染roles与chars_by_source的键则是glyph.source_index来源ID（普通JSON键为字符串）。基础几何接口限1500字符、单文本≤256字节、字符/字体≤1MB、side-map≤20000；渲染叶子限1500字形，沿既有JSON节点/2MB键文本及单元格parts工作预算，最终HTML≤4MB，回退分支也执行输出预算。

本轮最终验证：215项Rust workspace测试、严格clippy及CUDA feature编译通过；最终wheel已安装到build/rust-wheel-env，全部85组安装包回归与18 RGBA＋18 RGB旧PDF精确像素对照通过。1536基础分类/72集合平局、302显式角色数值补充、1288单元格HTML及390字体转换、所有权/并发/输入预算/HTML回退超限/许可证核对通过；全Unicode标志在Rust逐码点验证。67模块来源审查及原36 core＋3 paths、新增1 scripts原样源码哈希通过。日志：build/script-geometry-rust.txt、build/table-styled-cell-rust.txt、build/table-style-metadata-rust.txt、build/native-table-script-primitives-*.txt、build/rust-current-*.txt和build/wheel-all-tests.txt。

剩余：inline token精炼、分式保护、二维行合并、按cell自动生成角色、整表render_native_table_html_with_scripts和recover_table_result编排。visible-only与enhanced阅读顺序仍延后；默认PDF模型流程不变，完整PDF级原生恢复标志仍False。


2026-10-25 完整字符去重：新增`deduplicate_native_characters(chars)`和`read_pdf_native_text_geometry(source, *, page_index=0, pdfium_library=None, password=None, include_geometry=True)`。前者接收SDK已恢复代理对的普通字符JSON；后者直接读取PDF并完成代理恢复→同字形映射保护→重复绘制/平移副本来源归并→隐藏OCR副本抑制，返回字符与`loose_bboxes/tight_bboxes/origins`索引（JSON索引键为字符串）。核心复用原样vendor的text_snapshot.rs/dedup.rs，字体结构相等与signed-zero、固定Unicode16空白/NFKC/Han、Unicode17部首政策保持；仅同字形的异常Han映射规范化，不改写其他正文。

冻结840条旧完整去重参考（包含64组实际PDF/扩展模式及6字体/角度相等平局），4964 NFKC非恒等码点与1010角度cos/sin/round事实。64组实际PDF整链几何参考另行从旧函数独立读取，不从Rust回推。PDFium自身在实际PDF读取时已过滤部分重复绘制，真实Han多码值合并有正例；其他删除路线以独立旧完整字符输入核对，不把这些输入冒充实际PDF删除正例。

去重边界为5000条记录、总来源≤20000、单映射≤256字节、文字/字体合计≤8MB、绘制候选访问≤200万及隐藏配对保守上界≤2000万；索引唯一、来源非空、几何有限且abs≤1e15。超限/输入错误传播，原raw接口20000字符上限不变。`docvortex_character_dedup/docvortex_pdf_native_text_geometry=True`；旧raw API与默认PDF模型路径不变。visible-only裁剪/遮挡及适配器语义上下标HTML仍待接入，PDF级完整恢复标志仍False。

本轮最终验证：213项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；最终wheel已安装到build/rust-wheel-env，全部84组安装包回归与18 RGBA＋18 RGB旧PDF精确像素对照通过。840去重输入逐值/类型、64实际PDF整链几何、16组实时读取字符和绘图→完整引擎评估/恢复/诊断（不注入冻结字符）、并发/所有权/输入预算/错误恢复及随包许可证核对通过。66模块来源审查、36份core与3份paths原样源码哈希通过。日志：build/character-dedup-*.txt、build/rust-current-*.txt、build/wheel-all-tests.txt及build/wheel-all-run-output.txt。

下一步：移植table_materialization.recover_table_result及table_text_styles.render_native_table_html_with_scripts，复用已就绪的字符/紧框/原点和NativeTableResult，补齐语义上下标HTML后再接默认模型回退编排；visible-only裁剪/遮挡另行验收，enhanced阅读顺序继续延后。


实际PDF原语可以显式串接已有引擎（区域单位为视觉页面左上原点的PDF point，非图像像素）：

```python
from rapid_doc_rust import (
    read_pdf_native_text_geometry, read_pdf_vector_geometry,
    recover_native_table_primitives,
)
text = read_pdf_native_text_geometry(pdf_path, page_index=page_index)
vectors = read_pdf_vector_geometry(pdf_path, page_index=page_index)
result = recover_native_table_primitives(dict(
    table_bbox=table_bbox, page_size=text["page_size"], angle=0,
    chars=text["chars"], drawing_lines=vectors["drawing_lines"],
    rectangles=vectors["path_infos"],
))
```

`pdf_path/page_index/table_bbox`由调用方指定。结果是原语级`NativeTableResult`或None；适配器语义上下标HTML和默认模型回退编排仍待接入。

2026-10-25 字符原语：新增`read_pdf_character_primitives(source, *, page_index=0, pdfium_library=None, password=None, include_geometry=True)`。Rust直接读取旧DocVortex去重前的`chars`，并分别返回代理对恢复后的`restored_chars`；原始码值、来源索引、字体、页内对象编号、渲染模式、生成标记、原点、书写方向和可选loose/tight字框完整保留。基础布局沿旧整数ceil页面尺寸，扩展几何沿浮点CropBox；含隐藏文字时才按旧规则读取透明度可见标记，页面上没有隐藏模式时不新增该字段。

58组真实PDF/扩展模式、7650原始记录及恢复结果已独立冻结，包括50组既有PDF与8组补充ToUnicode/旋转/Form/隐藏透明文字；补充平面代理对、一个字形多码值、孤立高低代理均由实际PDFium读取。上限20k字符/1MB单字体名/8MB累计字体名/32MB宿主JSON，text/page/document按逆序在全局锁内释放，不输出原生地址。该接口不执行DocVortex重复绘制/隐藏副本去重，也不启用visible-only页面裁剪与遮挡过滤；不能拿它替代完整`get_chars_with_geometry`或默认PDF表格恢复。下一步接通这些旧规则，再接适配器语义HTML；完整恢复标志仍False。

本轮最终验证：211项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel已安装到build/rust-wheel-env，全部83组安装包回归与18 RGBA＋18 RGB旧PDF精确像素对照通过。新增58组字符/7650原始记录及代理恢复逐值/类型/可选字段、并发/所有权/密码和错误恢复通过；绘图41页及16组引擎串接再次通过。65模块来源审查、原36份core及新增3份paths源码哈希通过。日志：build/pdf-character-primitives-*.txt、build/rust-current-*.txt、build/wheel-all-tests.txt与build/wheel-all-run-output.txt。


2026-10-25：新增`read_pdf_vector_geometry(source, *, page_index=0, pdfium_library=None, password=None)`，接受路径、bytes或二进制流，页号从0开始。使用随包PDFium读取旧DocVortex视觉页面线与路径：

```python
from rapid_doc_rust import read_pdf_vector_geometry
geometry = read_pdf_vector_geometry("demo/pdfs/比亚迪财报.pdf", page_index=0)
lines = geometry["drawing_lines"]
paths = geometry["path_infos"]
```

返回自有JSON，无原生句柄；调用不关闭传入流。保留来源索引/填充颜色/Form层级/内部矩形，矩形摘要可以显式传给原语级引擎的`rectangles`。41页实际PDF/428条线/3694路径与旧版逐项对照，16组实时绘图＋冻结旧原始字符串接引擎。对象≤20万、路径与裁剪段合计≤200万、Form旧深度15、返回JSON≤32MB。`docvortex_pdf_vector_geometry=True`。DocVortex字符恢复/去重和语义上下标HTML仍待迁移，PDF级恢复标志仍False，默认预览不变。


2026-10-25：原语级完整表格引擎已接通。`evaluate_native_table_primitives(table_input)`返回文字、生成/保留候选、选中项、物理冲突与拒绝原因；`recover_native_table_primitives(table_input)`返回旧`NativeTableResult`普通JSON或None；`diagnose_native_table_primitives(table_input)`重放各路线并返回完整旧诊断。输入包含`table_bbox/page_size/angle/chars/drawing_lines/rectangles`，均是已有原生字符与页面原语，Rust内自行构建文字，不接受预制候选替代实际生成。

724条完整旧引擎参考全部从原始字符开始（547恢复、六类来源、9物理冲突、16旧DocVortex实际PDF/角度原语），包含aligned覆盖、欠分割/列下界、物理冲突/rect解围和身份保留。引擎一次调用的文字构建、候选生成及诊断重放共享实际1000万工作预算；独立阶段公开接口仍各有自己的预算，资源/输入错误传播。原语输入早期计数最多20000，相交原语>5000返回旧`primitive_limit`；进入候选生成后仍沿各路线5000原语上限。`native_table_engine=True`；Rust读取PDF原语和适配器语义上下标HTML仍待接入，PDF级`legacy_native_table_recovery=False`，默认PDF模型流程未切换。

资源说明修正：下文表格阶段原写2000万工作预算，现已按`legacy_page::Work`实际1000万修正；运行时上限没有提高或降低。公式ftfy的独立2000万预算不变。

2026-10-25：`build_native_table_banded_numeric_candidate(table_input,text)`移植旧分带数值候选，返回`candidate/diagnostics/html`（无候选时candidate/html为None）。保留重复数值行的众数首遇优先、外线/填充表头证据、公共字形间隙、三层分组表头和整字落格。发布沿原100%捕获、零歧义、score≥0.98门，不额外提高为100%顺序门；原95/96顺序正例保留。

所有格由`table_input.chars`原始来源/显式空白重新空间投影，表体去空白文字必须一致；表头按原行为可重建。严格源ID无效时传播错误，不用fallback ID掩盖；负源ID/缺省源ID/重复键的CPython来源set顺序已对照。1751旧候选参考（262非空、16旧实际PDF/角度原语）、1843数值语法/12共同间隙/164有符号set事实及3非法ID错误。每类5000drawing/rectangle、3000字形、20叶子列/2000物化格、20000原始字符及共享10M保守工作预算。`native_table_banded_numeric=True`，完整恢复仍False；剩余完整_evaluate、Rust PDF原语提取和语义HTML未接入。

2026-10-25：`generate_native_table_multiline_candidates(table_input,text)`接通旧多行稀疏主分支，失败后按原顺序尝试`grouped_rule_band`整行分组标题回退；返回`candidates/diagnostics/html`。保留关键列runs/短key runs、物理行带、居中多行续文、填充记录首列rowspan、两层表头、走廊与严格完整性门。回退标题由`table_input.chars`原始来源、邻接显式空白和紧字框空间投影复核，不直接拿普通格文字替代。

`project_native_table_spatial_text(chars,table_bbox=...,angle=0)`提供相同旧表格空间投影，返回保留内部等宽网格空白的字符串；`chars`为已有原生字符JSON（char为str/null、bbox为PDF point，可选char_idx/tight_bbox/font/writing_angle/origin），不读取PDF。1136条完整候选参考中548非空、54走实际分组回退，另356空间投影/114紧框间距事实；16旧实际PDF/角度仍为给定原语参考。每类5000drawing/rectangle、3000字形、20推断列/2000物化格、20000原始字符、单行投影列≤100万/累计≤800万字符及共享JSON/10M保守预算。`native_table_multiline/native_table_spatial_projection=True`，完整原生恢复仍False；还缺banded_numeric、完整_evaluate、Rust PDF原语提取和语义HTML。

2026-10-25：`generate_native_table_sparse_hybrid_candidates(table_input,text)`移植旧物理竖线/矩形边缘与重复文字对齐融合路线，返回`candidates/diagnostics/html`。包含最长连续稠密行选择、物理轨道优先、两层表头rowspan/colspan、粗token仅在强物理边且不横切glyph时跨列，以及100%捕获/顺序、零歧义和0.98发布门；不自行提高低分候选。

799条旧参考逐值/类型对照（571非空、16旧实际PDF/角度原语），覆盖分段竖线75%门、局部分隔20%/80%门、60%逐列/80%总体支持和行轨/最终完整性拒绝。每类5000drawing/rectangle、3000字形、推断最多200稠密列（物理假设可多一列）、2000物化格及共享10M保守工作预算；只接给定原语与NativeTableText，不打开PDF。`native_table_sparse_hybrid=True`；完整原生恢复仍False，尚缺完整多行稀疏/grouped回退、banded_numeric和完整引擎/PDF原语提取。

2026-10-25：`generate_native_table_rule_band_candidates(table_input,text)`生成旧三线表换行表头/至少两描述列同步重启的rowspan候选，返回candidates/diagnostics/同序html。普通单行表头三线表继续交其他旧路线，不额外放宽。builder接受后仍须100%文字捕获/顺序、零歧义和原0.98门，不能以候选分数代替这些发布条件。

`inspect_native_table_sparse_columns(table_input,text,header_bottom=0.)`返回共享物理线/矩形、hypothesis/hypothesis_diagnostics、stable_gutters/gutter_diagnostics、逐视觉行occupancy及overlapping_formula_rows；hypothesis包含evidence/x_tracks/physical_boundaries/filled_band_count。物理边界不会跳过墨迹跨界检查。

388候选参考、386共享事实（16旧实际PDF/角度）、195 median聚类；每类最多5000 drawing/rectangle、3000字形、20叶子列/2000物化格及共享保守预算。输入须为连续row_index和有效visual_row/字形引用。native_table_rule_band/native_table_sparse_columns=True；完整sparse_multiline/grouped回退、其他剩余候选、完整引擎、Rust PDF原语提取与语义HTML仍未接通，完整恢复仍False。

补充金额/状态PDF参考：另8条由两份实际PDF经旧DocVortex提取的角度输入，2条非空候选；与主218条分开冻结，独立wheel逐项通过。合计226条候选/重分行、125非空、16旧PDF/角度事实；Rust此处未执行PDF提取。

2026-10-25：`regroup_native_table_numeric_text(text)`按旧数值路线重分视觉行和token，保留字形ID/来源/spacing及原字号尺度。`build_native_table_aligned_numeric_candidate(table_input, text, regroup=True)`默认先重分行，再执行数值列/表头/逻辑表体/跨度及来源完整性验证；返回text、numeric_groups、candidate、html，无候选时后二者为None。regroup=False仅接受连续row_index，须使用已准备好的NativeTableText。

218条旧候选/重分行及532 regex参考包含123非空候选和8旧实际PDF/角度事实。保留旧数字/括号语法，SDK无生产Python依赖；最多5000 drawing/3000字形/2000物化格，finite坐标/spacing/字号abs≤1e9和共享保守工作预算。该入口不读取PDF、不含完整引擎的aligned覆盖选择或语义上下标；native_table_aligned_numeric=True，完整原生恢复仍False。

2026-10-25：`generate_native_table_text_candidates(table_input, text)`生成旧sparse_grid少线/底纹、text_grid无线表和key_value两列候选。输入与矢量入口相同的普通JSON页面原语/NativeTableText；输出candidates、diagnostics及同序html。保留60%锚点、续行/稠密歧义、表头colspan和拒绝rowspan等规则。375旧参考含8实际PDF/角度事实，最多5000原语/3000字形、200列/2000物化格，三路共享保守工作预算。

`inspect_native_table_engine_policy(table_input, text, vector_attempts=(), candidates=())`只返回旧引擎早期门和已有诊断守卫事实：primitive_count/input_rejection_gate、alias空行/物理漏行、minimum_columns、column_consistent原索引、尝试网格冲突及resolved_rect_undercount原索引。table_input还须含page_size；诊断使用矢量入口输出，候选使用已有原生候选字段。这里最多20000输入原语，按实际相交数检查旧5000门；与生成器资源限制分开。351旧守卫参考，不运行完整_evaluate或其他候选生成。

native_table_text_candidates/native_table_engine_policy=True；完整原生恢复仍False。aligned_numeric及其他稀疏候选、完整引擎、PDF DocVortex原语提取和语义上下标仍未接通。

补充9条完整路线边界：部分续行在真纵隔断/无隔断时的旧区别、alias冲突和残留窄带拒绝，已在独立wheel逐项对照；完整矢量样本共268条，轨道阶段204条。主参考未覆写。

2026-10-25：`generate_native_table_vector_candidates(table_input, text)`执行给定PDF原语的完整旧矢量路线：装饰竖线剔除、drawing/rect分路、轨道硬门、raw/supported假设、拓扑/物理证据、落格/评分和普通HTML。返回`candidates`、`diagnostics`及与候选同序的`html`数组；候选尚需交`select_native_table_candidates`等后续完整引擎逻辑决定是否使用。

`table_input`含table_bbox（PDF point）、数值angle、drawing_lines（bbox/width/orientation）与rectangles（bbox/segment_count/fill_visible/stroke_visible）；text使用旧DocVortex字符事实的`build_native_table_text`输出。`inspect_native_table_vector_tracks(table_input,text,include_drawing=True,include_rectangles=False,prune_unsupported_horizontal=False)`单独返回轨道和首个拒绝诊断，直接检查阶段不预先剔除装饰竖线。

259条完整矢量和204轨道阶段对照（含8旧实际PDF/角度原语）。Unicode16 isdigit固定，共享一次调用10M保守预算及调用内证据缓存；5000输入原语/3000字形、200规范轨道/10000原子格/2000物化格上限不变。该入口不打开PDF，不含其他文本/稀疏候选、完整_evaluate或语义上下标；native_table_vector_generation=True，legacy_native_table_recovery仍False，默认解析流程不改。

2026-10-25：`build_native_table_vector_candidate(tracks, text, evidence_label='line_grid', diagnostics=None)`接旧规范轨道→隔断连通/合并格→物理证据→文字落格/评分/普通HTML。返回topology、candidate、diagnostics、html；拒绝时candidate/html为None，拓扑阶段通过而后续拒绝仍返回topology。diag为输入副本，未知字段保留。

tracks使用旧_VectorTracks的普通JSON字段：snap_tolerance、local_width、rules、canonical_x_tracks/canonical_y_tracks（coordinate/aliases）、x_tracks/y_tracks、narrow_empty_threshold、is_line_grid/is_single_row_shape/is_single_column_shape、rows/cols。坐标/轨道长度必须一致且递增；text使用build_native_table_text输出。该输入假定已通过完整轨道生成阶段，不能把任意检测线的轨道直接当作已验证假设。

409＋12旧参考与104精确均值对照覆盖拓扑和候选。最多10000原子格/2000物化格、199行列/各200轨道、每轨道200alias、5000规则线/3000字形，保守共享工作预算；非空文字行全部无token明确失败。native_table_vector_topology=True，完整原生恢复仍False；尚需轨道完整拒绝门、多假设、PDF原语提取及语义上下标物化。

追加等距边界：14条实际旧最近物理线参考表明，等距时Python整数索引set的遍历顺序会影响外缘折叠。已按CPython3.14的有界整数插入/扩容/遍历语义保留该行为，204条集合遍历参考通过；PSF许可和来源随wheel分发。旧370条轨道参考未覆写。

2026-10-25：新增物理网格前段接口：

- `prepare_native_table_rule_evidence(table_input, snap_tolerance=..., join_gap=..., evidence_halo=0., include_drawing=True, include_rectangles=True)`：table_input含table_bbox、数值angle、drawing_lines和rectangles，返回fragments、rules、inferred与supported。后两者均为[x_tracks, y_tracks, removed_horizontal_tracks]。drawing包含bbox（旧width/orientation可保留但此阶段按几何判方向）；矩形包含bbox/segment_count/fill_visible/stroke_visible。页面坐标为PDF point。
- `canonicalize_native_table_tracks(positions, glyph_centers=..., threshold=..., extent=..., evidence_halo=..., minimum_track_count=..., rules=None, orientation='', separator_extent=0., snap_tolerance=0., ...)`：positions用前一步浮点轨道，rules用其合并线，返回tracks（coordinate/aliases）、alias_conflict、outer_collapse_count及unique；保留外缘、窄带及物理双边界规则。

336＋370旧函数参考覆盖本阶段。阈值由后续旧候选编排提供，此处不自行选择假设、评分或完成恢复。最多5000输入原语/20000展开片段、200规范轨道/3000字形中心和共享工作预算，非有限坐标/负容差明确失败；仅普通JSON数值和布尔字段。native_table_vector_evidence=True，完整原生恢复仍False。

2026-10-25：新增给定网格的原生候选接口：

- `inspect_native_table_candidate(text, rows=..., cols=..., specs=..., use_grid_index=False)`：返回网格校验、可选轨道索引和逐字归属/歧义。
- `build_native_table_candidate(text, source=..., rows=..., cols=..., specs=..., structure_support=..., row_stability=..., column_stability=..., ...)`：返回candidate、diagnostics和普通html；三个结构质量指标必须由调用方显式提供。文字使用`build_native_table_text`的输出，spec包含row/col/rowspan/colspan/bbox。
- `render_native_table_candidate(candidate)`：普通HTML序列化，保留跨度、行/列顺序和旧quote=False转义；不含自定义语义上下标渲染。

327条旧候选与22条额外边界逐项对照网格、索引、归属、拒绝原因、文字/来源和评分，另4条HTML参考。候选构建与最终接纳分离，低分候选仍由已有`select_native_table_candidates`决定是否接纳。仅支持给定物理网格，不负责从PDF发现网格；`native_table_candidate_builder=True`，`legacy_native_table_recovery=False`。物理网格发现、完整评估和Rust DocVortex字符几何提取仍待实现。最多2000规格/10000逻辑格/3000字形，坐标abs≤1e9，逻辑坐标/跨度abs≤1000，共享JSON、内容及工作量预算继续生效。

发行包 `rapid-doc-rust`，导入 `rapid_doc_rust`；原生扩展 `_native` 使用 Python 3.10+ stable ABI。

原生表格文字阶段已增加`prepare_native_table_glyphs(chars, table_bbox=..., angle=0)`、`build_native_table_text(...)`与`build_native_table_cell_text(owned_glyphs, median_height=...)`。前两者接收旧DocVortex普通JSON字符（char=str/null，bbox/loose_bbox为PDF point四坐标），返回局部字形和带原char_idx的视觉行/token；后者仅接收已归属到某个单元格的字形，返回text/来源parts/分组索引，None来源代表算法插入间隔。不会把整表所有字形自动当作一个单元格。

362条旧文字构建（含16实际PDF/角度输入）、1934条旧单元格文字/来源片段和841条跨行分隔参考通过；Unicode16属性内置，SDK不需要Python Unicode处理或NumPy。仅给定字符几何阶段，Rust DocVortex PDF几何提取、物理候选发现和语义HTML物化仍未完成，默认解析路线不变。输入≤2万字符、输出/单元格≤3000字形，来源索引abs≤1e12、坐标abs≤1e9、单元格文字≤1MB，共享JSON和工作预算继续生效。

原生表格新检查点：`plan_native_table_rule`/`accept_native_table_rule`移植旧捷径守卫和结果接纳，`select_native_table_candidates`只仲裁**已经生成的候选**。完整物理候选生成与语义HTML物化尚未内置，`legacy_native_table_recovery=False`，默认模型路径不变。

`parse_pdf_models(..., native_table_recoverer=adapter)`提供可选适配点。适配器须实现`recover_table(pdf_bytes, *, page_index, bbox, pdfium_library, password)`，返回可JSON序列化的`{'html': str, 'confidence': number}`或None；bbox为PDF point。守卫通过且置信度达到`rule_table_score_threshold=0.9`才接纳，成功后跳过表格模型，None/低分/普通恢复异常回模型，ImportError继续抛出。`use_rule_table=False`禁用，`native_table_routes`返回逐表诊断。该接口供后续Rust后端或显式调用方适配器接入，不会自动导入旧Python解析器，也没有用实验网格代替旧恢复算法。

本阶段另有2条实际旧五页组合参考，wheel使用冻结的旧PDF原生结果作为适配器响应对照编排；这不代表Rust已经运行原生候选生成。原有6条模型组合对照仍保留。

旧版PDF对照在独立进程冻结。恢复器应遵守现有PDFium限制：同一进程不能混用pypdfium2和本wheel对同一PDFium库的访问。

2026-10-25追加：多页入口现在支持UNET和旧默认双模型路由。单UNET传`table_variant='unet', table_model=unet_session`。双模型传`table_variant='unet-slanet-plus'`（或`unet-slanet-1m`）、`wired_model=unet_session, table_model=slanet_session, classifier=paddle_session, secondary_classifier=qanything_session`；`use_compare_table=False, cls_conf_threshold=0.9`对应旧默认仲裁。空OCR跳过分类/结构，低于阈值才比较两个候选。仍不含原生规则表格捷径及可选空格补识别器，不能替换完整Python默认解析。

`recognize_legacy_unet_table_png`支持已有转义三列OCR与全图OCR回退，返回`html/raw_html/cells/matches`；`recognize_legacy_table_auto_png`共用一次OCR/提示输入并执行旧双模型路由，返回最终HTML、分类、原HTML候选、选中分支及选优指标。`select_legacy_table_candidate`的OCR文字须已经转义，和旧安全预览`select_table_candidate`的未转义输入不同，避免实体重复编码。

新增多页模型预览`parse_pdf_models`，直接接收PDF并返回`markdown/content_list/pages/assets/source_page_count/page_range`。须提供V3矩形版面、DB/CTC及实际任务需要的公式/SLANet/方向模型；模型都是现有`OnnxSession(runtime, model, sha256)`，可复用会话。`native_text=True`明确选择原生优先，False强制OCR；页范围从0开始、end_page不包含。保留模型原阅读顺序。

本地开发模型已准备时，可在项目根目录用已安装wheel的隔离Python运行真实PDF转换：

```powershell
.\build\rust-wheel-env\Scripts\python.exe -I .\scripts\migration\preview_pdf.py "demo/pdfs/示例1-论文模板.pdf" "demo/pdfs/比亚迪财报.pdf"
```

输出到新建的`build/rust-pdf-preview/<时间戳>/<文档名>/document.md`、`result.json`及`images/`；根目录`summary.json`记录各文档成功/错误，失败另存`error.txt`且进程返回非零。可用`--start-page 0 --end-page 1`先测第一页，`--output <新目录>`指定输出。默认144DPI、CPU单线程、原生文字优先、公式开启、UNET/SLANet双模型；版面/公式均真实推理，没有固定参考响应或旧Python解析器适配器。脚本依赖现有`build/migration`模型清单与本地模型文件，默认ORT位于`.venv`，可用`--runtime`覆盖；不自动下载模型。完整原生表格恢复尚未接入，本入口仍为开发预览。

```python
from rapid_doc_rust import parse_pdf_models

result = parse_pdf_models(
    'input.pdf', layout_model=layout_session,
    detector=det_session, recognizer=rec_session,
    table_model=slanet_session, table_variant='slanet-plus',
    formula_model=formula_session, orientation=orientation_session,
    native_text=True,
)
print(result['markdown'])
# result['assets']为{'images/<文件名>': bytes}，调用方自行保存到同名相对路径。
# result['pages']/result['content_list']为可JSON序列化数据，不包含图片bytes。
```

这是显式模型配置的预览；尚无原生表格结构捷径、可选空格补识别、seal/checkbox、自动方向/文档分类或Office。缺少必要模型会报错，不静默丢弃任务。正文OCR按页运行，后置OCR按整个文档运行，尚未证明任意文档与旧全批识别等价。返回资产与后置OCR各限128MB，默认128页、每页1600万/总1.28亿渲染像素，模型裁图仍限400万像素，共享JSON预算继续生效。

独立wheel已与实际旧五页组合对照SLANet/UNET/双模型三种配置、两种文字模式，共6条组合，JSON/Markdown/content_list和18份图片像素一致；该对照固定版面/公式响应，OCR/分类/表格结构实际CPU运行，另有真实V3 PDF冒烟。不是完整解析器或质量/性能验证。最终表格入口`recognize_legacy_slanet_table_png`返回`html`以及原有`raw_html/structure/match`诊断；`normalize_legacy_table_html`可独立执行旧最终CJK/HTML归一化。

兼容性审查（2026-10-24）：本包尚不兼容完整 Python 解析器，也未证明整文档质量/速度/内存优于旧版。`organize_page_text`、`recognize_pdf_page_hybrid` 必须显式传入 `experimental=True`；原生表格优先默认关闭。`compatibility_report()` 提供模块分类、来源哈希和证据边界，详细结果见 `docs/plans/rust-compatibility-audit.md`。原 `rapid_doc` 生产解析流程未切换到这些实验。

此外提供原生 PDF 文字、页面 OCR、已知区域网格恢复、PP-DocLayoutV3 矩形、Paddle/Qanything 表格分类，以及 SLANet/UNET 表格裁图、单页表格、正文/表格有序页面块、保守原生/OCR 融合和块内段落视图。wheel 可携带本机验证的 PDFium；ORT 和模型需显式提供。完整默认文档路径和 Office 仍在开发。

```python
from rapid_doc_rust import parse_pdf_text

output = parse_pdf_text("document.pdf")
print(output.text)
print(output.pages[0]["characters"])
```

`start_page` 从零开始、`end_page` 不包含在结果中；默认遇到扫描页返回带 `code="model_missing"` 的 RapidDocError，显式 `allow_partial=True` 才允许返回部分文字及诊断。提供的流不会被关闭。当前预览不支持与 pypdfium2 共用进程，差分请使用独立进程。

OnnxSession 的 `run()` 是单输入 FP32 的底层开发接口；需要运行库路径、模型路径及 SHA256。`cuda` 需构建时启用对应特性，注册失败明确报错；服务器 GPU 推理尚未验证。

`run_named({name: {"shape": [...], "values": [...]}})` 支持多个命名 FP32 输入，名称必须与模型完全一致；输出保留 `dtype`（float32/int32/int64）、shape 和 values。输入/输出分别按合计元素预算检查，int64 不经浮点转换。已用合成模型和真实 PP-DocLayoutV3 三输入模型验证。

`layout_session.layout_png(source, threshold=0.5)` 已接通 PP-DocLayoutV3 的矩形模式，返回 `cls_id/label/score/coordinate/order`。RGB/cubic/NCHW 前处理的 15 组 FP32 张量逐位一致，15 组后处理规则和 3 张 PNG 与 Python 基线匹配。已覆盖分类阈值、同类/跨类 NMS、大背景图过滤、large 类包含淘汰、顺序和裁剪；最多 1000 候选。相同评分明确优先后出现的框，不依赖 NumPy 不稳定排序的偶然结果。尚未迁移 mask 多边形、类别自定义阈值或其他布局模型版本。

当前 cubic 前处理匹配已验证的 OpenCV 5.0.0 / Intel IPP 2026 AVX2 行为，极小图走整数回退。没有链接 IPP/OpenCV 动态库；其他 OpenCV 后端和平台的逐位等价尚未验证。CLI `layout-png` 需要 `--runtime --model --sha256`，可设置 `--threshold` 和 `--device`。

`table_session.table_structure_png(source, variant="slanet-plus")` 返回 SLANet-plus 或 `slanet-1m` 的结构 token、评分和原图像素单元格框。7 组前处理用例中有效张量逐位一致；2 组序列合同和 6 张两模型 PNG 输出与 Python 基线匹配。保留重复单元格 token、EOS 截断、零 bbox 过滤及 plus 的坐标回缩；不执行 CTC 重复消除。变体与模型不符明确报错，空结构不返回 NaN。CLI 为 `table-structure-png`。

`match_table(structure, ocr_lines)` 将同图像的 OCR 四角匹配到格框，返回 `matches.cells`（每格的原始 OCR 索引）、`matches.unmatched`、`cell_texts` 和 HTML。连续面积 IoU/距离及首格平局规则匹配旧实现；不相交文字不强行填格。单元格文字按 OCR 来源顺序用空格连接并保留原有空白，文字中的 HTML 全部转义，不解释 `<b>` 等标签。输出 HTML 为规范化表格片段，格式不保证与旧版 HTML 字节一致。

`recognize_table_png(source, structure_model=table_session, detector=detector, recognizer=recognizer, variant="slanet-plus")` 接通显式表格裁图的结构→OCR→匹配→HTML，CLI 为 `recognize-table-png`，另需 `--structure-model/--structure-sha256`。尚不自动执行 wired/wireless 分类、UNET 或整页表格候选发现。

8 组匹配合同含混合四/八值框、空输入、平局、零面积、阈值和跨 chunk 数量；4 组 wired/wireless 的两模型完整裁图流程通过安装验证。空白图的实际模型输出分别出现 section 标签错配和反序轴框，完整接口明确报 `invalid_input`；`table_structure_png` 仍保留原始输出供诊断。缺格、重叠、未闭合或 token/格框数量不匹配也不自动修复。

`unet_session.table_line_mask_png(source)` 返回 UNET 原始线类别图 `width/height/classes`，类别 0/1/2 为背景/横线/竖线。此接口的坐标在长边 1024 的模型缩放图内，尚未回缩到原图或恢复单元格。14 组前处理张量逐位一致，包含 cubic/area、极小图、整数/非整数缩小；3 张 PNG 的 int64 类别图逐项一致。后续线几何接口见下，不能把原始 mask 当作已识别表格。

`table_line_geometry(mask, source_width=w, source_height=h, morph_close=True)` 将上述 mask 缩放回原图，完成横/竖线隔离、闭运算、8 连通域、中心线、补线和端点延长，返回按源规则排列的 `rows/columns` FP32 线段。垂直闭运算始终执行，`morph_close` 只控制水平闭运算，与旧实现一致；最多 400 万源图像素、边长 4096、2000 连通域，连接工作量受预算约束。20 组缩放/闭运算及连接线输出逐位一致，包含真实模型结果。

`connect_table_lines(rows, columns, **config)` 可独立复用连接规则，配置含 `row_gap/column_gap/angle/more_rows/more_columns/extend/max_lines/max_work`；84 组线段合同逐位一致。连通域与旋转外接框另有 48 组逐位对照，保持 OpenCV BBDT 的 2x2 块标签顺序。

`recover_table_polygons(polygons, row_threshold=10, column_threshold=15)` 接收已按行排列的逆时针格框（左上/左下/右下/右上），返回原框索引对应的 `logic_points` 闭区间和基准行列数。使用双精度坐标，固定合并容差为 10；不重新排序或自动补齐缺格。30 组拓扑合同覆盖跨行/跨列、轨道插入、稀疏格和真实 UNET 框；旧规则生成空/反序跨度的用例明确报错。

`unet_session.table_cells_png(source, rotated_fix=True)` 已串接 PNG→UNET→线段→2px LINE_AA 绘线→角度估计→最近邻旋转→闭合格发现→逆变换→同行排序→逻辑格。`polygons` 为原图像素逆时针四角，`rotated_polygons` 在校正后的图像中，二者与 `topology.logic_points` 索引一致。没有检测到格框时返回空列表和 `topology=None`；无效轨道或反序跨度明确报错，关闭旋转修正可能导致倾斜输入触发这种校验。

24 组抗锯齿图、36 组旋转图逐像素一致；12 组倾斜网格区域合同及 5 张实际 PNG（含 ±3° 倾斜）的格框/逻辑格与 Python 对照通过。旋转的 SIMD/标量舍入合同对应当前 OpenCV 5.0.0 / AVX2 参考环境，其他后端未验证。

`recognize_wired_table_png(source, structure_model=unet_session, detector=detector, recognizer=recognizer, rotated_fix=True, min_confidence=0.0, blank_cell_rec=True)` 接通显式有线表格裁图的 UNET 格框→OCR→跨格文字拆分→空格补识别→边缘噪声裁剪→安全 HTML。返回 `cells/ocr_lines/blank_rec/matches/html`。`match_wired_table(cells, ocr_lines, blank_cells=())` 可单独匹配已有格框和文字；`recognizer.recognize_blank_table_cells_png(source, polygons, needed=None)` 可单独检查补识别结果。所有像素和文字计算均在 Rust 内执行，不需要 Python NumPy/OpenCV/ORT。

命令行使用 `python -m rapid_doc_rust recognize-wired-table-png crop.png --runtime <ORT库> --structure-model <UNET模型> --structure-sha256 <哈希> --det-model <检测模型> --det-sha256 <哈希> --rec-model <识别模型> --rec-sha256 <哈希>`，输出同一 JSON。可添加 `--no-rotated-fix` 或 `--no-blank-cell-rec`，也支持通用 `--device/--threads/--min-confidence`。

59 组匹配/拆分/行合并合同、10 组 HTML/边缘裁剪、6 组彩色裁图/对比度和 5 张实际 PNG 完整流程通过对照。匹配保留原 OCR 索引；补识别片段索引为 `null`，`blank_rec` 保留拒绝原因。文字按纯文本转义，重叠逻辑格明确报错，未匹配文字保留。Unicode 分词属性固定为 Unicode 16.0.0，不随宿主 Python 版本变化。补识别保留旧实现的通道交换和过滤规则；可用 `blank_cell_rec=False` 关闭。该接口显式选择有线分支；分类路由见下。尚不做整页候选发现，也不保证旧 HTML 字节等价。

`classifier.classify_table_png(source, variant="paddle", output_type=None)` 返回 `label/score`。`paddle` 对应 `paddle_cls.onnx` 的 BGR/Lanczos4/中心裁图，默认输出按概率处理；`qanything` 对应 `q_cls.onnx` 的历史通道灰度语义及 Pillow L/BICUBIC 缩放，默认按 logits 做一次 softmax。自定义导出模型必须按实际输出指定 `output_type="probabilities"` 或 `"logits"`；不根据文件名或 ONNX 图自动猜测。两种模型各 13 组张量逐位一致，6 组评分合同、6 组实际 CPU PNG、并发与错误恢复通过；两个标签相同评分时选择 wired。CLI 为 `classify-table-png`，参数是 `input --runtime --model --sha256 [--variant paddle|qanything] [--output-type probabilities|logits]`。

`recognize_table_auto_png(source, classifier=paddle_session, secondary_classifier=q_session, wired_model=unet_session, wireless_model=slanet_session, detector=detector, recognizer=recognizer, wireless_variant="slanet-plus", use_compare_table=False, cls_conf_threshold=0.9)` 对显式表格裁图先分类，再调用对应 UNET/SLANet 流程。`secondary_classifier` 可省略；提供时采用旧 TableCls 共识规则：标签不一致选 wireless，评分取两者最小值，`classification.members` 保留两次预测。

评分严格小于阈值，或 `use_compare_table=True` 时，两个分支共用一次检测/区域 OCR，再按旧 `select_best_table_model` 的文字命中、非空格数和规模规则选优；有线分支的空格补识别仍独立执行。`route` 是最终选中分支，可能与 `classification.label` 不同；`comparison` 返回两个候选计数及原因，未比较时为 `None`。候选缺模型、结构无效或超出预算均明确报错，不吞掉错误。只需按分类调用单个分支时设置 `cls_conf_threshold=0.0`；否则低评分场景须同时提供两个结构模型。支持 `rotated_fix/min_confidence/blank_cell_rec/classifier_output_type`。4 组实际双模型输出、默认双分类器比较、一次 OCR 共享、缺模型以及高分/等于阈值/FP32 边界通过安装验证；分类评分转为完整双精度数值再序列化，避免把 FP32 的 `0.8999999761581421` 改成 `0.9` 后跨越阈值。整页表格发现和正式兼容 API 仍在开发。

`select_table_candidate(ocr_texts, wired_html=..., wireless_html=...)` 可独立检查选优。`ocr_texts` 是未转义 OCR 字符串，Rust 内部使用输出器同一转义方式做字面子串匹配；保留重复/空 OCR 和旧 `<td`/`<th` 前缀计数语义（因此 `thead` 也影响旧计数）。返回 `route/reason/wired/wireless`，计数字段为 `physical_cells/blank_cells/non_blank_cells/ocr_hits`。197 组合同与旧 Python selector 的决策/计数一致，覆盖六个规则原因及四组实际候选。输入限规范闭合表格 HTML：表格/行/格标签、整数跨度、`br` 和输出器生成的转义文本；未知实体或任意浏览器 HTML 明确拒绝。两候选合计最多 1600 万字节、OCR 最多 1000 行/100 万字节，文字检索有 5000 万字节扫描预算。HTML 是内部输出格式，不执行浏览器修复或加载外部资源。

`recognize_page_tables_png(source, layout_model=layout_session, classifier=paddle_session, wired_model=unet_session, wireless_model=slanet_session, detector=detector, recognizer=recognizer, secondary_classifier=q_session)` 接通整页 PP-DocLayoutV3 rect→`table` 候选→裁图→分类/双结构选优。保留布局列表顺序及 `layout_index`，不把正文区域识别成表格。返回 `width/height/layout/tables/crop_pixels`；每个表格含原布局 `region`、整数裁图 `crop`、裁图坐标下的完整 `recognition` 和页面坐标下的 `page_geometry`。后者包括单元格多边形、OCR 行、UNET 匹配片段及 `crop_to_page` 仿射矩阵；匹配索引与局部结果一致。保留模型角点顺序，SLANet 四值框转为左上/右上/右下/左下四角；不再次截断模型预测。

默认 `crop_grid=5`，按旧流程向外扩到 5 像素网格再裁页面边界，可设 1；允许 1～64。页面默认/上限 1600 万像素，单个裁图上限 400 万像素；默认 `max_tables=32/max_total_crop_pixels=32_000_000`，配置上限为 1000 表格/12800 万累计像素。所有候选先规划和检查预算，再逐一识别；超限或任一候选失败均报错，不截断成部分成功。无表格时不调用分类/OCR/结构模型。支持裁图接口的阈值、分支、补识别等显式选项。13 组旧 RGB 裁图合同、单表/双表/空白三页（共 3 表）、独立裁图等价、坐标投影、预算及裁图并发通过。

`recognize_pdf_page_tables(source, page_index=0, dpi=144.0, layout_model=layout_session, classifier=..., wired_model=..., wireless_model=..., detector=..., recognizer=...)` 渲染指定 PDF 页后调用上述流程，额外返回原 `page_index/dpi`。输入支持路径、bytes 或二进制流；坐标均为渲染图像素，DPI 改变像素尺寸，暂未输出 PDF point 坐标。两个 PDF 页/DPI 场景通过，含非零原页号与空白页。当前为单页模型路线，尚未做自动方向策略、标题/脚注归属、原生表格规则融合、正文解析或跨页合并；整份文档兼容入口仍未完成。

`recognize_page_png(source, layout_model=..., detector=..., recognizer=..., classifier=..., wired_model=..., wireless_model=...)` 将正文 OCR 和表格组成同一页 `blocks`，按模型 `order` 排序，同 order 保持原布局数组顺序。原始 `layout`、`layout_index`、`region` 保留；文字块包含 `crop/local_lines/lines/text`，`lines` 为页面像素坐标。表格块包含 `recognition/page_geometry/lines/html/text`，与单页表格接口共用既有识别流程。页面 `text` 以双换行连接非空块文字，表格文字取检测 OCR 行，含空格补识别的单元格最终内容以 `html` 为准。这里的 text 是逐行预览，不是完整文档或 Markdown 导出。

当前文字标签为 `abstract/algorithm/content/doc_title/figure_title/paragraph_title/reference/reference_content/text/vertical_text/vision_footnote`。其他区域（含图片、公式、页眉、页脚和 footnote）保留为 `kind="unsupported", text=None`，并写入 `diagnostics`，不会误报识别完成。`figure_title` 暂作为普通文字，不归属到图表；竖排和旋转页面没有自动方向策略。重叠文字区域不去重，空布局不回退整页 OCR；版面漏检的文字仍可能缺失。

`max_regions=256` 限制待识别文字和表格的合计区域数量；单裁图 400 万像素、默认合计 `max_total_crop_pixels=32_000_000`，硬上限 12800 万；页面 1600 万像素、区域配置上限 1000。使用与表格相同的 5px 网格裁图（可设 `crop_grid=1`）。所有选中裁图共用一次预检预算，超限在 OCR/表格模型调用前报错。文字/表格需要检测和识别会话，表格额外需要分类及实际路由所需结构会话；仅空页或不支持区域时不调用这些会话。任一识别失败向外报错，不返回部分成功结果。

`recognize_pdf_page(source, layout_model=..., detector=..., recognizer=..., page_index=0, dpi=144, **options)` 渲染指定 PDF 页并调用上述块接口，返回原页号/DPI。输出仍为渲染图像素；未执行原生 PDF 文字/表格融合、PDF point 投影、复杂段落重排、标题/脚注归属或跨页合并。完整兼容解析入口仍待实现。

`recognize_pdf_page_hybrid(source, experimental=True, layout_model=..., detector=..., recognizer=..., page_index=0, dpi=144, **options)` 在上述有序页面块上增加正文逐行原生/OCR 融合。布局和 DB 检测仍执行；只对需要回退的行调用识别模型。原生字符使用 PDFium 的 `visual_tight_bounds`，按实际渲染宽高与视觉 point 尺寸的比例映射到像素，避免非整数 DPI 取整偏移。原 `recognize_pdf_page` 保持纯模型流程。

当前采用保守的预览策略：页面未旋转、检测行接近水平、字符 Unicode/索引可靠、按源索引向右排列、无图片框重叠、原生字符外框覆盖检测行宽至少 80%/高至少 50% 才复用原文。相同检测行内混有扫描文字时，图片重叠会让整行回退 OCR。覆盖不足、原生文字不连续或不能映射 Unicode 也回退；资源超限和模型错误继续报错，不视为普通回退。这些是当前新预览的明确策略，不宣称与旧全文解析规则逐项等价。

融合的文字行增加 `method/reason`；原生行 `confidence=None`，并提供含中间空格的 PDFium `character_indices`，OCR 行保留识别置信度、token 和时间步。`min_confidence` 仅过滤 OCR 行。`native_routes` 保留未过滤的选择记录，`native_page` 保留原始 PDF 字符/几何事实（其中 visual 字段仍是视觉左上 point），输出 `lines` 和布局仍是页面像素。只要全部检测行可复用原生文字，`recognizer` 可以省略；需要 OCR 时缺少识别会话明确报错。表格的原生优先选择见下文，纯模型接口保持原行为。

`max_native_work=10_000_000` 控制一页所有文字区域累计的“检测行数 ×（原生字符数 + 图片数）”工作量，上限同默认值；预算耗尽不重置、不跳过后续区域。原生事实 JSON 上限 32MB、最多 10 万字符/1 万图片。依然受页面/裁图预算约束。当前未补回布局或 DB 漏检的原生文字、未验证任意字体编码/隐藏文字层、未做重叠布局去重、完整段落重排和跨页整合；原生事实单独保留以供检查，不能把该接口视为完整兼容解析器。

融合接口默认 `prefer_native_tables=False`，保持所有表格走模型。明确设置 `experimental=True, prefer_native_tables=True` 时，先对 table 布局候选恢复原生物理网格，可靠时使用原生单元格/文字，其余继续模型。同一页可包含两种表格；此选择策略未证明与旧解析器等价。显式开启后，纯原生表格页可省略 `detector/recognizer/classifier/wired_model/wireless_model`（布局模型仍需要）；正文检测或模型表格仍需要相应会话。

当前原生表格策略要求未旋转、候选扩展区域无图片、Unicode/字符方向可靠、存在非空原生文字、闭合外边界以及矩形合并格。轨道容差 1 point、分隔边覆盖阈值 90%；恢复网格与布局区域 IoU 至少 80%，所选非空字符必须全部归属单元格。`native_table_margin_pt=6.0` 在布局框周围留固定 point 余量查找外边线，可设 0～32，不因失败自动放宽。缺边、空原生文字、图片重叠、网格范围不匹配等会保留原因并走模型；资源超限、无效来源事实和模型运行错误直接报错，不返回部分成功。

表格块新增 `table_selection={method, reason}`；原生分支 `recognition.route="native"`，包含 `table/x_tracks/y_tracks/html/text`，坐标系为 `visual_points`，单元格保留 PDFium `character_indices`。`page_geometry.cell_polygons` 使用实际不均匀轨道与跨度，坐标为页面像素，`point_to_page` 记录 point→像素矩阵；不会按等宽行列伪造格框。`lines=[]` 表示文字位于原生单元格；块 `text` 按物理单元格连接，HTML 保留 rowspan/colspan 并转义原文。块 `crop` 仍记录原布局裁图规划，原生网格可在固定余量内超出该框；原生分支不实际裁图或运行表格模型。

`native_table_routes` 记录各候选选择原因，`native_table_work` 记录累计工作量。`max_native_table_work=10_000_000` 为默认/硬上限，按候选扫描的页面事实及 `2 × 原子格数 × 分隔线数` 覆盖查询估算累计，拒绝的候选也收费；单网格最多 2000 原子格/2000 分隔线。原生表格候选仍受页面块裁图预算约束。当前不恢复无边线表格，不从填充细条推测线，不自动修复缺边，不处理旋转原生网格；尚未补回布局漏表、合并跨页表格或完成全文规则兼容。

`assemble_model_page_with_assets(source, model, image_writer=..., page_width=..., page_height=..., page_index=0, scale=1.0, formula_enabled=True)` 接收页 PNG 和已识别的旧模型 JSON，按旧裁图规则生成图片、表格、公式 PNG，在页面组装时回填 image_path，再归块/排序。source 支持路径、bytes 或二进制流；image_writer 为目录路径或提供 write(name, bytes) 的对象。返回 page_info 副本，空页为 None；不修改模型输入。无资产需求仍使用下方原 assemble_model_page，默认行为不变。

文件名按旧规则由页原始模式像素的 MD5 大写、span 类型、页号和未缩放 bbox 取整值生成 SHA256，平铺保存为 PNG；裁图坐标 bbox×scale 向零取整，越界补零，保留 L/LA/RGB/RGBA 模式及透明通道。支持 8-bit PNG，P/16-bit 明确 unsupported。337 组实际旧 writer 页面/调用的完整 JSON、名称、顺序和解码像素精确一致；PNG 压缩字节可能不同。资产页→编号→段落的 337 组 JSON 和 1011 组旧输出对照通过。

另有 11 组 ICC/EXIF/透明色参考：按旧 PNG save 保留 ICC 色彩配置及 RGB/L 的 tRNS 透明色，默认不继承 EXIF；透明色不改变 Pillow 原始模式或页 MD5。元数据与像素均精确对照，单 ICC 配置≤1MB，页与原图的 ICC 合计≤8MB。

需要原图优先时设置 extract_original_image=True，并提供 original_images=[{"bbox": [...], "source": original_png}, ...]；bbox 是页面单位，仅 image 按 IoU≥extract_original_image_iou_thresh（默认 0.9）使用最后一个匹配原图，表格和公式仍裁页图。这里只接收显式提供的 PNG，不自动从 PDF 提取原图。未归块的候选仍按旧顺序写出，同名写入保留旧覆盖顺序。

所有原生计算通过后才开始 writer I/O，计算失败不写文件；目录模式使用临时文件后单文件原子替换，限制为平铺 SHA256 文件名。自定义 writer 和 I/O 异常原样抛出，已完成的写入不保证回滚。页/单原图 PNG 输入≤64MB、解码≤1600 万像素；原图最多 256 张，合计≤6400 万像素/128MB PNG。单裁图资产≤400 万像素、最多 1000 次写出，累计≤3200 万像素/128MB 编码字节；另受旧页面 JSON/几何预算约束。PDF 原图自动提取尚未支持，不触发模型推理或正式解析路由。

表内 UUID 图片使用 `table_fill_images=[{"uuid":"...", "bbox":[x0,y0,x1,y1], "source":png_path_or_bytes_or_stream}]`，对应旧 page_dict.table_fill_image_list 的已提取原图。按旧顺序在 MagicModel 过滤后、普通裁图前写出，HTML 中所有 UUID 子串替换成 `<img src="目录名/文件名" alt="Image" />`；空目录名仍保留 `/文件名`，与旧逻辑一致。

旧函数把含进程地址的函数对象字符串加入名称哈希，新版有意改为稳定种子 `table_fill/{页原始像素MD5大写}_{页号}_{四个未缩放bbox坐标向零取整}`，再 SHA256+.png，避免与普通 table 裁图冲突。相同页/bbox 的不同 UUID 仍按旧顺序覆盖同名图片，不按 UUID/内容去重。57 组实际旧页面/调用和 171 组输出保留原始名称证据，明确投影新名称后比较其他字段、像素、元数据和顺序；不宣称旧文件名等价。

表内图仅支持 8-bit L/LA/RGB/RGBA PNG，保留 ICC/tRNS，默认不继承 EXIF；UUID 非空≤256 UTF-8 字节，最多 256 张、单张输入≤64MB、输入合计≤128MB、命中后解码缓存≤6400 万像素。与裁图共享每张≤400 万像素、1000 项/3200 万输出像素/128MB 输出预算，ICC 共享≤8MB。HTML 查找额外工作预算1000万字节，展开后单 HTML≤2MB。计算失败显式抛错且不写文件，区别于旧函数吞异常；I/O 失败不回滚已写文件。

HTML（包括表格 HTML）内的 `src="data:image/...;base64,..."` 已在普通裁图之前处理，保存解码原字节，以完整 URI 的 SHA256 命名，jpeg 扩展名为 jpg；前缀沿用 writer._parent_dir 的目录名，缺失时为 images。新增 `replace_inline_base64_img_src(markup, image_writer, page_index=0, image_path_prefix='')` 可独立处理 HTML。只匹配旧代码的双引号、小写 src、无等号两侧空格形式；失败编码保留原文。9667 组旧工具、12 组完整页面及 36 组旧输出串联参考覆盖原字节、HTML、调用顺序。

自定义 writer 通过 `_mineru_written_image_paths` 跨调用去重内联图片，成功后才标记；无法设置属性的 writer 仍重复写出。普通裁图保持旧重复写入顺序。目录方式每次可覆盖同名文件，内容相同。内联 HTML≤2MB、前缀≤4096 字节、最多 1000 项；只允许安全平铺扩展名（1–32 个 a-z、0-9、-、_），WMF/EMF 转换明确 unsupported。畸形 padding 对齐本机旧 CPython 3.14.5，不承诺与其他小版本的畸形输入一致；参见 [CPython 基线](https://github.com/python/cpython/blob/v3.14.5/Modules/binascii.c)。

```python
from rapid_doc_rust import (assemble_model_page_with_assets, optimize_formula_numbers,
                           split_paragraphs, render_model_pages)

page = assemble_model_page_with_assets(
    "rendered-page.png", old_model_json, image_writer="output/images",
    page_width=600, page_height=800, scale=2.0,
)
if page is not None:
    pages = split_paragraphs(optimize_formula_numbers([page]))
    markdown = render_model_pages(pages, image_path_prefix="images")
```

`assemble_model_page(model, page_width=..., page_height=..., page_index=0, scale=1.0, formula_enabled=True)` 接收旧模型 JSON `{"layout_dets": [...]}`，移植未修改 page_model_info_to_page_info 中 **OCR 已识别、ocr_enable=True、image_writer=None** 的规则链。模型项包含 category_id、8 值 poly、score；OCR 项（category_id=15）另含 text，主块需有效 original_order。poly 与 polygon_points 是模型图像像素坐标，按 scale 转到调用方提供的页面单位；输出含 preproc_blocks/page_idx/page_size/discarded_blocks，空页面返回 None，输入不变。

Rust 内部已迁移 MagicModel 的坐标转换/多边形舍入、置信度和重叠处理、图表标题/脚注距离关联、native foreground 归属约束、span 去重/过滤/分配、横排/竖排行以及已有模型排序。保留旧 image_footnote 被标成 image_caption 等实际行为，不使用新几何启发式替换。286 组实际旧函数完整输出及 18 组 int/float/bool 等值去重与多边形舍入边界精确一致；大整数不会因转 float 误判重复。JSON 整数范围为 i64/u64，其他数字为有限 FP64；不接受任意 Python 对象。

`assemble_model_pages(pages, optimize_formula_numbers=False, merge_tables=False)` 接收页面请求列表，每项为 model/page_width/page_height 及可选 page_index/scale/formula_enabled。默认 page_index 为列表索引，空页包装为空块页，再调用旧 para_split；返回 page_info_list 副本，382 组旧页面→段落参考精确一致。`optimize_formula_numbers=True` 在 para_split 前执行旧编号关联，`merge_tables=True` 在段落后执行旧跨页表格；两个选项均默认 False。它仍不等同于完整 result_to_middle_json，不执行后置 OCR、模型推理或正式解析路由。

`assemble_model_page_with_native_text(source, model, native_page=..., image_writer=..., page_width=..., page_height=..., ...)` 提供旧 `ocr_enable=False` 分支。source 是8-bit RGB页图PNG；其他选项与 assemble_model_page_with_assets 相同，仍处理普通资产/内联图片/显式表内图。native_page 可通过 `read_native_pdf_page(pdf, page_index=0)` 自动生成，返回旧get_page的分组事实JSON：`blocks[].lines[]` 有页面单位 `bbox`、弧度 `rotation` 和 `spans[]`，后者含 `text`、`chars[]`，字符含 `char`、整数 `char_idx` 和页面单位 `bbox` 数组。原bbox对象换为等坐标数组，模型poly仍是图像像素并由scale转换。

`read_native_pdf_page` 接收路径、bytes或流，支持 `pdfium_library/password/quote_loosebox`，默认按旧 pdftext 0.7.1 的0.7/0.1阈值去重、分span/行、上下标和块，保持PDF源顺序。每页≤10万字符、字体名累计≤32MB、单名≤1MB；不执行OCR或改变版面模型 original_order。`render_pdf_page(..., color_mode='RGB')` 可直接产生所需RGB PNG，默认仍为RGBA；二者均无Python图像依赖。完整模型推理路由仍待接入；PDF原图导出见下方接口。

`extract_original_pdf_images(pdf, page_index=0, max_depth=15, max_pixels=32_000_000, render=True, scale_to_original=True)` 自动提取原图，返回 `{images,diagnostics}`。每个image含 `uuid/bbox/width/height/mode/source`，source为独立PNG bytes，images可直接传入 `original_images` 或 `table_fill_images`。按旧对象遍历顺序、宽高任一小于5页面单位过滤、原始分辨率比例缩放及mask渲染；bbox保留旧 `get_bounds` 加页高翻转的语义，嵌套Form及旋转页不悄悄替换为新的视觉坐标策略。

当前pypdfium2 5.10.1移除了旧函数调用的get_pos，旧get_ori_image会吞异常并全部返回空列表。Rust使用对应get_bounds恢复提图；参考明确用get_pos=get_bounds兼容别名运行实际旧函数，非未适配空输出的完全等价声明。随机UUID改为稳定的 `pdf-image-{page_index}-{accepted_index}`，同页不同配置应由调用方分别管理引用。对象级解码失败沿用跳过并增加诊断；资源超限和不支持位图模式整次失败。导出L/RGB/RGBA PNG，单图原始及预计/实际渲染分辨率≤1600万像素、累计输出≤3200万像素，最多256候选、PNG单张≤64MB/累计128MB、深度≤32和页面总对象预算有界。保留pypdfium2缩放来源和Apache许可。

`assemble_pdf_model_page(pdf, model, image_writer=..., page_index=0, dpi=144.0, native_text=True, extract_original_image=False, ...)` 组合读取旧原生分组、RGB渲染、可选原图提取和现有单页资产组装，返回 `{page,ocr_images,diagnostics}`。原图提取须显式开启；仍使用旧image span最后达到IoU阈值的图片，不改变table/formula普通裁图。`native_text=False` 适用于model已含OCR结果的分支；默认返回待补识别图片，交给post_process_model_ocr。model仍由调用方提供，尚非完整自动模型路由；此组合接口沿用原生读取的每页10万字符上限。

`predict_model_layout(page_png, layout_model=session, layout_shape_mode='rect', threshold=0.5)` 执行真实PP-DocLayoutV3 ONNX布局推理，再按旧RapidLayoutModel/BatchAnalyze映射为 `{layout_dets:[...]}`。必须显式选择rect；auto/poly/quad尚未接入，拒绝以矩形替代。`convert_model_layout(regions, layout_shape_mode='rect', ...)` 可独立转换已有PP-DocLayoutV2/V3类别结果；`filter_model_layout(layout_dets, use_custom_ocr=False)` 独立保留旧reference剔除、6像素小框、严格0.5/0.7重叠和inline/image/seal/chart例外，原顺序与未知字段保留。默认忽略标签、Python三位score round及formula_enabled/formula_level过滤与旧版一致；不生成缺失order，enhanced仍延期。可传markdown_ignore_labels覆盖默认列表，filter_overlaps=False用于显式跳过该过滤stage；use_custom_ocr只控制旧inline重叠分支，不表示已接入VL模型。

`prepare_model_crops(page_png, model, page_index=0, formula_bbox_expand_px=2)` 返回 `{model,ocr,tables,formulas,images}`，按旧规则为任务附加layout_index、source图片ID和useful_list：OCR四周白边50像素，表格先floor/ceil到5像素网格再按NumPy截断，公式按旧round/扩框及邻框阻挡规则裁图。model中的公式bbox、表内图片layout_image_list同步保留；表内图仍按80%包含判定，跨表重复出现时重复裁图。PIL/ndarray负载换独立PNG和ID；随机UUID改为稳定table-image-page-image-table，原RGB像素另行精确核对。

`prepare_pdf_model_inference(pdf, layout_model=session, layout_shape_mode='rect', ...)` 自动读取原生事实、渲染RGB、执行布局模型并生成上述计划，另返回native_page、page_image和scale。**这是推理准备入口，尚不执行OCR/公式/表格模型或回填结果。** 待执行器按旧公式→OCR→表格→后处理顺序串接；当前生产rapid_doc入口未切换。

```python
plan = prepare_pdf_model_inference(
    'input.pdf', layout_model=layout_session, layout_shape_mode='rect', dpi=144,
)
# plan['ocr']/['tables']/['formulas'] 的 source 引用 plan['images'] 中的独立RGB PNG。
```

此任务规划仅8-bit RGB页图、矩形模型布局；多边形、预填原生/VL布局、旧crop_img越界赋值和空图明确拒绝。每次≤1000布局/裁图，页图≤64MB/1600万像素，单裁图≤400万/累计≤3200万像素与128MB PNG；JSON≤2MB字符/键、200000节点/64层。公式扩框参数0..1024，formula_level为0/1/2。固定响应隔离转换的258组/86过滤与60计划/409裁图，与3组真实CPU模型对照分别记录，不宣称完整自动解析或整文档质量/性能提升。

`derive_native_ocr_boxes(native_page, region, useful_list=..., scale=1.0, masks=())` 移植旧txt_spans_bbox_extract，返回 `{region,boxes}`。masks中的bbox必须已转换为裁图坐标。保留正交方向容差、span中心/标点边缘判定、按行中心聚类、NumPy pairwise均值、首span字号和相邻合并；Python两位round、公式重叠严格阈值及±1切框沿用旧规则。公式切框结果为FP32，此后逆变换也保持旧NumPy FP32运算，不统一转FP64。空框会在返回的region副本设置need_ocr_det，输入原对象不变。

`prepare_model_ocr(plan, native_page=None, scale=None, ocr_enable=False, use_det_mode='auto', page_index=0)` 接收prepare_model_crops/prepare_pdf_model_inference的计划。native_page和scale默认从plan读取；原生模式缺少事实时明确报错。返回原计划副本，追加model/native/detect和新的images引用：model含旧原生空文本OCR span，native保留原生检测框，detect为需要实际OCR检测的任务，按旧64像素分辨率组及组内原顺序排列。每项含source（遮罩补白PNG）、crop_source（原始未遮罩裁图）、useful_list、masks、ocr_enable和target_shape；后续识别应使用原始裁图。保留旧auto/txt/ocr控制与rotate_label回退；PDF本身rotation不代替rotate_label。

```python
plan = prepare_pdf_model_inference(
    'input.pdf', layout_model=layout_session, layout_shape_mode='rect', dpi=144,
)
ocr_plan = prepare_model_ocr(plan, page_index=0)
# ocr_plan['detect'][*]['source']引用ocr_plan['images']，尚未运行检测/识别模型。
```

此准备阶段**不执行检测或识别**，可将结果继续交给下方execute_model_ocr。182组实际旧原生框、54组native/检测输入计划、75张实际旧检测器输入像素精确对照；这组准备参考中的固定空检测响应仅隔离输入，不作为实际检测质量证据。旧merge临时span字段不写回调用方。JSON/裁图预算同上；原生共享工作量≤1000万、每页扫描≤10万span、产出≤2万框、每任务≤1000遮罩，重复覆盖的总遮罩面积≤6400万像素。多页图片映射合并时须使用各自page_index。

`execute_model_ocr(ocr_plan, detector=det_session, recognizer=rec_session, batch_size=6, det_batch_size=1, lang=None, page_index=0)` 执行旧默认OCR任务，返回含已回填model的计划副本。检测使用项目ocr_patch实际生效的固定limit_side_len=960、max缩放、unclip_ratio=1.8、dilation=True、box_thresh=0.5；依赖原始源码中的960/1500/2000分档已被项目补丁覆盖，不使用该分档。独立 `OnnxSession.detect_model_png` 提供此路径；已有detect_png仍沿用原预览默认参数。

检测结果按旧内部排序后再执行utils.sorted_boxes、merge_det_boxes和公式遮罩切分，保留NumPy2弱标量FP32比较/算术、严格0.6/0.8/行宽4倍门槛及角框追加顺序。`organize_model_detection_boxes(boxes, masks=())` 可单独运行。`prepare_model_recognition(plan, detections, lang=None, page_index=0)` 从原始未遮罩crop_source导出旧识别裁图，返回processed、recognition及images，图片ID为rec-page-index；detections为按detect任务顺序提供的四点框列表。crop在宽3过滤之前生成，模型poly在crop之后做倾斜矫正，竖行旋转高宽比门槛为2；通用预览裁图的1.5门槛保持。仅8-bit RGB PNG，tRNS/ICC/EXIF不改变原RGB像素。

识别沿用已验证的共同padding批处理，`apply_model_ocr_recognition(model, [(text,score),...])` 按旧模型级回填：三位score，原始score严格<0.5标记category_id=16；七种特定文字在score<0.8且宽<高时同样标记。与后置page OCR的严格>0.5接受规则分开。np_img和lang只从本次被选中的模型span移除，其他字段保留。固定响应的112组规则与5条真实CPU det→rec→回填（含1280/2048宽图）分别验证。

```python
ocr_plan = prepare_model_ocr(plan, page_index=0)
result = execute_model_ocr(
    ocr_plan, detector=det_session, recognizer=rec_session, page_index=0,
)
model = result['model']  # 可继续进入页面组装；此调用不执行公式、表格模型。
```

目前det_batch_size只接受1；rec批处理仍拒绝同宽高比跨批改变padding的歧义。输入模型返回副本，缺图先检查；共享流仅读一次且不关闭，重复任务仍重复推理。旧批失败逐图重试尚未复现，错误显式传播且原模型不部分回填。JSON/工作量预算同上；最多1000检测框/任务和1000裁图尝试，尝试含宽度过滤掉的裁图也计入单图400万/累计3200万像素预算；PNG输入/输出各≤128MB。只覆盖旧默认小模型路径，完整公式/表格执行、多页路由及GPU实机仍待完成，不宣称完整解析质量/性能更优。

```python
native_page = read_native_pdf_page('input.pdf', page_index=0)
page_png = render_pdf_page('input.pdf', page_index=0, dpi=144, color_mode='RGB').png
# 传给下方组装调用时 scale=2.0；page_width/page_height 使用 native_page['size']。
# 或直接 assemble_pdf_model_page('input.pdf', model, image_writer='output/images',
#                              extract_original_image=True)
```

`prepare_formula_input(source, height=384, width=384)` 提供旧PP-FormulaNet Plus输入预处理，height/width为仅限关键字参数；默认适用S/M，L须指定768。返回dict：shape为NCHW，tensor为小端float32原始字节，rgb为预处理后RGB原始字节，rgb_size为[宽,高]，另含crop_box/resized_size/dtype/layout；无需NumPy/Pillow/OpenCV。仅支持8-bit RGB PNG；目标每边2..2048、源≤400万/中间≤1600万像素、每轴≤65536、工作量≤2.56亿，超限明确拒绝。88实际旧RGB逐字节一致；灰度张量使用公开OpenCV FMA式，与旧IPP绝对误差≤1e-6（本机最大2.38418579e-7）。**仅预处理，不运行或解码公式模型，也未证明网络输出等价。**

`OnnxSession.recognize_formula_batch_png(sources, batch_size=1)` 执行PP-FormulaNet Plus原图PNG→旧预处理→ONNX→BPE/ByteLevel解码→旧LaTeX修复→默认ftfy，保持输入顺序。batch_size为关键字参数，范围1..64；模型必须是单通道384/768的单输入、单int64 token输出，character元数据为旧字典格式。模型需显式路径/SHA，SDK不下载。实际CPU对照为旧配置的M模型、6张合成图共6批，含希腊、全角和中文，不能推广为真实文档质量、S/L或GPU已验证。

公式所用 **ftfy6.3.1默认处理已迁移**，支持合法Unicode的乱码恢复、实体/字宽/连字/引号/控制字符及Unicode16 NFC，不再限制ASCII/基本汉字；原18组拒绝分支已全部精确通过，无Python ftfy依赖。29517实际旧ftfy及29517旧完整公式post_process参考逐字符串精确一致。capabilities中formula_model_inference=True、formula_full_ftfy=True、formula_ftfy_version='6.3.1'；完整解析器仍未完成。`FormulaDecoder(character).decode(token_rows)`用于旧模型character JSON解码；`fix_formula_text(text)`独立执行完整旧LaTeX→ftfy后处理，`repair_formula_latex(text)`只执行前半段。LaTeX的normalize仍不启用，ftfy原本启用的默认NFC单独保留。

只提供旧公式使用的默认ftfy选项，不提供自定义配置/说明步骤或孤立Python surrogate输入。保留HTML auto跨行状态和数字分号实体等旧细节；单输入/输出64KB、临时文本256KB、修复/编码128轮/递归64/共享文本工作量2000万，超限明确报错而非返回部分结果。模型跨batch共享该文本预算并累计限制输出2MB。

公式任务可以跨页统一执行：`execute_model_formulas(plans, recognizer=formula_session, batch_size=1)`，formula_session可直接使用上述OnnxSession；plans为prepare_model_crops返回的计划列表。也可提供实现`recognize_formula_batch_png(png_list, *, batch_size)`并返回等长str/None序列的适配器，适配器须保证自身模型前后处理兼容，不能使用OCR识别器替代。

公式会话输入总计128MB/源3200万像素、累计张量3200万元素/预处理2.56亿工作量，持会话锁并释放GIL；同一流仅读一次且不关闭。字典8MB、词表10万/额外token1000、每行8192/总128000token、单文本64KB/输出2MB，ByteLevel解码前预检词长，超限拒绝。

表格的独立OCR阶段：`recognize_table_text_png(source, detector=det_session, recognizer=rec_session, masks=(), batch_size=6, retry_padding=8, retry_short_side=32)`。所有配置均为关键字参数，masks为表裁图坐标的bbox列表。返回`{boxes, ocr_result, retried}`；ocr_result是旧表格模型使用的`[boxes, texts, scores]`三列，文本已按旧规则归一化和HTML转义，**不能直接当作现有预览matcher的原文line字典**。不做正文的merge/宽3/低分过滤；检测取遮罩图，识别取原图，裁图旋转门槛2。表格DB固定960/unclip1.6；薄图对齐0视为空检测，短边<32时仅在空结果后加8px白边重试，重试不再切公式。

此入口要求调用方已完成原生路线和方向选择；**不包含完整旧表格解析**，OCR word-box、公式图片与结构模型的完整编排待接入，capabilities中legacy_table_text_execution=True、legacy_table_model_result=True、legacy_table_execution=False。失败显式传播，尚不复现旧批失败逐图/word-box回退。单RGB PNG≤64MB/400万像素，1000框/遮罩、重复mask6400万、批识别裁图3200万像素/128MB；padding0..1024、retry_short_side0..65536。同一源图只解码一次生成批识别裁图，流不关闭且只读一次。

`prepare_table_hints(page, table, useful_list=..., formulas=(), checkboxes=(), scale=1.0, formula_enabled=True, image_enabled=True, extract_original_image=False)`返回`page/table/formulas/fill_images`副本。公式按旧裁图映射后只排除完全在外的项；可将返回formulas作为文字阶段的masks。图片在允许原图提取时，优先取完整包含且IoU<0.9的原图，任意原图入选即不补布局图；否则取全部layout_image_list，保留旧累积table_fill_image_list。页面rotate_label为90/180/270时禁止原图。图片是JSON元数据，PIL/PNG按uuid另持有，不保证跨调用Python对象别名。

`prepare_table_model_payload_png(source, ocr_result, fill_images=(), formulas=(), skip_text_in_image=True, inline_delimiters=('$','$'))`将已定方向的RGB图和旧转义三列OCR准备为结构模型输入，返回`{png, ocr_result, needs_ocr}`。已有文字不重复转义，先逐图涂白/插uuid/删除图片包含的先前文字和图片占位，再加F32公式框和旧转义公式/checkbox。图片按旧int取整且包含右下边界涂白，区别于检测用公式遮罩。`[]/None`返回needs_ocr=True，调用方仍须执行旧predict全图OCR分支；`[[],[],[]]`保持旧truthy语义可插提示。这里不执行结构网络，也不生成HTML。241旧提示、343实际predict结构输入/精确像素与4真实PDF/CPU提示链通过；后者2页×强制/非强制OCR，提示元数据为给定测试输入，框/文本/RGB精确且分数实测误差0。

提示/结构输入的RGB PNG≤64MB/400万像素，输入JSON深度64/20万节点/2MB字符串；图片及模型输入公式各≤1000项、OCR总行≤20000、包含比较≤1000万次、重复白矩形≤6400万像素、累计图片元数据≤2000项，分隔符每侧≤64字节。输入不修改，流仅读一次且不关闭，tRNS不合成白底。`legacy_table_hints/legacy_table_model_payload=True`，`legacy_table_execution=False`；完整结构/HTML兼容编排仍待完成。

可改用`prepare_table_model_input_png(source, ocr_result, detector=det_session, recognizer=rec_session, fill_images=(), formulas=(), skip_text_in_image=True, inline_delimiters=('$','$'), batch_size=6)`完成空OCR回退后再准备提示。返回原payload字段及`can_infer/full_ocr_used`：`None/[]`先执行旧全图det+rec，仍为空时返回`ocr_result=None, needs_ocr=False, can_infer=False`，跳过图片/公式；已有三列（包括三空列或低分结果）直接沿用，此时不要求det/rec session。此入口接收已定方向图，等同旧调用上下文`skip_table_orientation=True`，不执行direct predict额外方向猜测或结构网络。

独立`recognize_table_full_ocr_png(source, detector=det_session, recognizer=rec_session, masks=(), batch_size=6)`返回`{boxes, ocr_result}`，与旧行级OCR不同：原图检测不先涂公式遮罩、不做薄图白边重试；无merge/宽3过滤，公式切框后识别，**先按默认0.5筛分再旧归一化/转义**。空文本只要分数足够仍保留。检测为空时不要求识别器，全部被公式切空则按旧直接识别器的空列表失败显式报错；结果数量不符也显式报错。默认ch/non-seal的真实直接TextRecognizer参考单列，不含RTL、OCR词框或失败逐图重试；RGB与裁图预算沿用上述1000框/400万源像素/3200万累计裁图/128MB。243旧策略、81阈值边界、8真实直接-rec CPU和16旧predict回退输入通过，框/文本/RGB精确，实测最大分数误差0。`legacy_table_full_ocr=True`，完整表格解析仍未完成。

旧SLANet matcher可调用`match_legacy_slanet_table(structure, ocr_result, width=..., height=...)`，返回`raw_html/pred/matched/dt_boxes/logic_points`。OCR是旧转义三列，结构是模型token与4/8值单元格框；保留旧1-IoU→distance→首项规则、只裁下界的min/只裁上界的max、过滤表格上方OCR及原输入顺序。HTML保留原token与文本，不再次转义；多内容按旧粗体/空白规则合并，移除完整thead/tbody chunk。逻辑点保留旧跨度解析行为，不以安全预览的严格网格代替。

`recognize_legacy_slanet_raw_png(source, ocr_result, structure_model=slanet_session, variant='slanet-plus', detector=..., recognizer=..., fill_images=(), formulas=())`接通已定方向图→模型输入/可选全图OCR回退→真实plus/1m结构网络→旧matcher。variant可为slanet-plus/slanet-1m；无字时不要求或调用结构模型，返回raw_html/structure/match为None。结构provider需提供`table_structure_legacy_png`；OnnxSession已实现该方法，将F32单元格坐标完整提升为F64再序列化，避免预览短小数回读后改变边界。原有table_structure_png合同不改。真实样本结构坐标逐值精确，模型置信度沿用F64均值，与旧F32均值差≤1e-5，仅作为元数据。

这两个入口输出的是**raw_html**，尚未执行RapidTableModel最终的CJK/BeautifulSoup归一化；不包含UNET、分类器/双模型仲裁、原生结构捷径和完整表格API。54旧几何、156原HTML/逻辑、96格式→匹配、6真实plus/1m结构样本以及6 CPU OCR串联（2无字跳过结构）通过。输入JSON预算沿用深度64/20万节点/2MB字符串，≤1000 OCR/10000 cell/20000 token、1000万工作量/10万逻辑占用格；输出在拼接前检查4MB预算，跨度只支持abs≤1000的ASCII整数。空cell或三空列进入matcher会明确失败，列数不符严格报错，结构错误直接传播，输入不修改、流只读一次且不关闭。`legacy_slanet_matcher/legacy_slanet_raw_inference=True`，`legacy_table_execution=False`。

综合文字入口`recognize_table_text_stage_png(source, native_page=page, table=table, useful_list=..., detector=det_session, recognizer=rec_session, orientation=orientation_session, scale=1.0, force_ocr=False, ocr_enable=False)`先做旧方向准备，再在表格方向为0、页面未旋转且未强制OCR时填充PDF原生文字；**只在整体原生结果为空时OCR，不补齐部分空行**。返回方向准备结果及`ocr_result/text_source`（native/ocr/none），不运行表格结构模型。native_page须为`read_native_pdf_page`等接口提供的规范旧JSON事实。

可独立调用`extract_table_native_text_png(source, boxes, native_page=page, table=table, useful_list=..., scale=1.0, word_boxes=False)`查询原生三列结果；True时输出旧逐字符框，和OCR模型词框不同。source为表格RGB裁图，useful_list为旧8项坐标映射。保留宽3/斜框纠正、原生字符间距、连字处理和稀疏文字/对比度；旧全局bbox作用于表裁图的行为也保留。每个JSON字符出现位置按独立pdftext.Bbox投影，普通JSON正文填充合同不改。共享JSON/字符工作预算和400万源像素，原生补识别裁图累计3200万像素/128MB，逐字符输出最多20000框；非法参数/资源异常直接抛出，不吞异常后悄悄回退OCR。262旧几何/262填充/10实际PDF填充含字符框、28固定响应整体回退及10真实PDF→CPU链（2原生/8 OCR）通过；后者识别分数实测误差0。`legacy_table_native_text=True`，原生表格结构捷径、OCR word-box、公式图片和结构模型仍待接入。

方向准备可单独调用`prepare_oriented_table_text_png(source, page=page, table=table, detector=det_session, orientation=orientation_session, masks=(), scale=1.0)`，返回最终`png/boxes/rotate_label/angles/pdf_not_rotate/classifier_used/retries`。page为旧原生页面JSON（span的bbox对象投影成四坐标），table带原poly。优先用原生文字角度；无角度时按旧竖版/竖框门槛才要求方向模型，并保留旧BGR模型输入。90/270旋转后重新检测、180不旋转，公式遮罩仍沿用原坐标；`retries`依次记录每轮检测是否触发薄图白边。`plan_table_direction(page, table, width=..., height=..., boxes=..., scale=1.0)`可只查询投票和分类调用计划。角度字符串支持ASCII无下划线float，其他数字格式请先转JSON number；共享JSON预算与400万像素限制，过大角度明确失败。320旧规则、18像素、76固定响应方向流程及28真实CPU检测/方向流程通过，后者2次实际分类；没有运行OCR识别/结构模型。`legacy_table_direction=True`，完整表格流程仍未完成。

可独立调用`organize_table_detection_boxes(boxes, masks=())`和`apply_table_ocr_results(boxes, results)`调试框整理/行级结果。`apply_table_prediction(table, html_code, formulas=(), fill_images=(), scale=1.0)`接收已经由模型返回的HTML，按旧_process_single_table末段返回表格副本；formulas须传整页公式/checkbox原列表，不根据位置重新过滤。它删除layout_image_list、提取首<table>至末</table>，有效时保存除scale向零取整的formula_boxes/img_boxes；没有运行结构模型。131框/75恢复/209策略/64末段固定响应与遮罩/裁图像素对照；8条真实CPU DB→CTC链含成功薄图恢复另列，框/文字精确，分数误差上限1e-5（本机实测0），不代表完整表格质量。

入口保持页和区域顺序，一次调用适配器，仅用非空结果更新latex，返回计划副本；空串/None保留旧值，不修改公式空白或命令。已有模型结果可用`apply_model_formula_results(plans, results)`独立回填。结果数量不符会报错，避免旧zip的静默截断。先预检全部RGB PNG，单图400万/累计3200万像素、输入128MB、1000任务/1000页/20000布局；共享流只读一次且不关闭。100组实际旧固定响应仅验证策略，不宣称实际公式推理或整文档质量。

下方页面组装入口返回 `{page, ocr_images}`；空页 page=None，ocr_images 是内存PNG字节映射。原生字符不足的 span 会写入 `np_img='ocr-页号-序号'`，用这些图片进行后置 OCR，再编号、段落与跨页表格：

```python
result = assemble_model_page_with_native_text(
    page_png, model, native_page=native_page, image_writer='output/images',
    page_width=width, page_height=height, page_index=0, scale=scale,
)
if result['page'] is not None:
    pages = post_process_model_ocr(
        [result['page']], recognizer=recognizer, images=result['ocr_images'],
    )
    pages = split_paragraphs(optimize_formula_numbers(pages))
    pages = merge_cross_page_tables(pages)
```

`fill_model_text_spans(source, spans, native_page=..., blocks=..., discarded_blocks=(), scale=1.0, page_index=0)` 是独立调试stage，blocks是旧 prepare_block_bboxes 数组（前四项坐标、第8项类型），返回 `{spans,ocr_images}`。两种入口均返回副本；多页合用图片映射时需使用不同page_index。只实现旧return_word_box=False路径；字符索引为绝对值≤10亿的整数。正交方向容差严格小于0.1度，保留旧纵排、字符中轴/标点、显式空白、连字和长度不足回退规则。

补识别裁图按 NumPy 切片而非 PIL 补零：向零取整，负坐标从末尾计数、超界截断，空图删除。OpenCV灰度/NumPy标准差/两位小数round后对比度低于0.17删除，等于0.17保留。OCR PNG只保存原RGB像素，主动移除ICC/透明色/EXIF以避免解码时改变旧np_img字节。198组旧span、32组完整页面、295个旧对比度、15条真实CPU串联/45输出提供对照；包含5组真实PDF事实、元数据和正负零归属边界。

页图≤64MB/1600万像素；单OCR裁图≤400万像素，含被丢弃的低对比度裁图尝试合计≤3200万像素，最多1000张、编码≤128MB。输入JSON总字符/键≤2MB、200000节点/64层，最多1000 span、10000有效原生行/100000字符；坐标绝对值≤100万，scale=(0,1000]。单span网格区间≤100001格、全局工作量≤1000万，极小高度/超范围网格明确resource_limit。NumPy pairwise求和许可位于包内licenses/NUMPY-PAIRWISE.txt。

`merge_cross_page_tables(pages)` 独立处理已有 para_blocks 并返回副本：从后往前检查相邻页首尾表格，保留旧续表 caption/脚注和宽度规则、重复表头、rowspan/colspan 对齐与回退、稀疏续行、cell_merge、跨页脚注及表后误挂 caption 恢复。被合走的子块按旧行为清空 lines 并标记 lines_deleted；未知字段和 preproc_blocks 保留。此显式 stage 不读取 MINERU_TABLE_MERGE_ENABLE。313 组旧完整页面 JSON、864 组旧输出及 48 组单页→段落→合并链/144 组输出作为精确参考，包含原 5 个续表回归。

支持正确显式闭合、单表、非嵌套单元格 HTML、标准 void 标签、合法分号结尾 HTML5/数值实体（实体文本≤32字节），正跨度为 ASCII 整数 1..1000。嵌套/多表、错误 HTML 自动修复、script/style/template、未知/缺分号实体、非法控制字符实体明确 unsupported。单表合并后≤1000行/1000逻辑列，输入 HTML≤20000节点/64层、每次跨度占位扫描≤200000项；输入 JSON 字符/键合计≤2MB、200000节点/64层、最多1000页、每页初始1000块，全流程工作预算1000万，复制内容也收费。计算失败不返回部分合并页面，输入保持不变。

`optimize_formula_numbers(pages)` 独立迁移旧 `_optimize_formula_number_blocks`，返回副本，仅修改 preproc_blocks。优先前一公式，再按旧条件检查后一公式，最后按同排重叠/右侧间距/中心距离关联编号；转换全角标点并剥离外层括号后追加 LaTeX tag。同名已有 tag 不重复追加，冲突或无可靠关联的编号按旧逻辑转 text，空编号删除；不跨页关联。应先于 split_paragraphs 调用，函数本身不重算已有 para_blocks。

后置 OCR 提供 `apply_model_ocr_results(pages, results)` 与 `post_process_model_ocr(pages, recognizer=..., images=..., batch_size=6)`。前者输入旧收集顺序的 `[(text,score), ...]`，后者把 span 的 `np_img` 当作非空≤256 UTF-8 字节的图片 ID，从 `images={id: PNG路径/bytes/二进制流}` 加载已裁出的单行图片，在 Rust 批识别后回填。仅收集主块 text/title/formula_number、table/image 的 caption/footnote 子块，以及所有 discarded_blocks；其中有 np_img 键就收集，不按图片值真伪过滤，不遍历既有 para_blocks 或 image_body/table_body。

成功后删除选中 span 的 np_img，严格 `score > 0.5` 才保留文字并按旧 `.3f` 舍入 score，否则 content=''、score=0.0。有限数值不额外强制 0..1，沿用旧规则。结果数量/格式错误或推理失败抛出且不修改原页面；这是与旧函数原地先 pop 再推理不同的失败所有权合同。忽略块及未知字段保留。同一图片 ID 的流读取一次，重复引用仍在原位置重复预测；空计划不要求模型或图片映射。必须先后置 OCR，再编号、段落、跨页表格和输出：

```python
pages = post_process_model_ocr(pages, recognizer=recognizer, images=crop_images)
pages = optimize_formula_numbers(pages)
pages = split_paragraphs(pages)
pages = merge_cross_page_tables(pages)
```

`OnnxSession.recognize_batch_png(sources, batch_size=6)` 按旧宽高比排序，批内共同 padding，返回原 sources 顺序；只支持当前 3×48 CTC 识别合同，最小宽320、最大4096。若同分宽高比跨批且填充宽度不同，会明确 unsupported，因为 NumPy 非稳定排序可能改变具体图片的 padding；不会悄悄换成新排序或单张推理。batch_size 必须与要对照的旧配置一致。12 组旧 CPU 参考含11组支持、1组明确拒绝；支持项逐图张量位模式一致，真实模型文字精确、置信度绝对误差≤1e-5。73组旧回填与11条真实CPU响应串联/33输出另行精确对照。

回填最多1000页、1000有效文本块/10000行/20000 span/1000图片、JSON字符/键合计2MB、200000节点/64层。批大小1..64，单PNG≤64MB/1600万像素，总输入≤128MB/3200万像素；重复引用按展开后收费。批张量受 max_elements（默认1600万）约束，保持 GIL 释放和会话锁；PNG透明色按现有白底BGR解码合同处理。PDF原生字符填充/补识别裁图的生产入口尚未迁移，此处需显式提供裁图，不自动调用正式 Python 解析器。

718 组完整旧后处理 JSON、382 组原模型→公式编号→段落串联、639 组原模型→公式编号→Markdown/content_list 精确一致；新的输出对照保守排除 169 组处理后仍包含算法的链，算法 Magika 分类依旧 unsupported。此处理不运行公式识别模型。单独入口最多 1000 页、总计 1000 块/10000 行、200000 节点、64 层、2MB 字符串及 32MB native JSON，坐标绝对值≤100 万，几何与文字扫描合计限 1000 万工作量；非法输入和超限明确报错。

`render_model_pages(pages, make_mode="mm_markdown", image_path_prefix="")` 接收已完成段落的 page_info_list，输出旧 Markdown 字符串或 content_list 数组。支持 mm_markdown、nlp_markdown、content_list 及固定默认 $/$$ 公式分隔符；保留正文转义、全角字母/数字转换、断词、列表换行、标题/图表/公式、0～1000 bbox 与丢弃块的旧规则。726 组旧完整输出及从原模型组装到输出的串联精确一致。算法代码语言分类依赖 Magika，171 组包含算法的链已明确排除，遇到算法文本块报 unsupported，不伪造 txt 标签；自定义公式分隔符和 content_list_v2 尚未迁移。

```python
from rapid_doc_rust import assemble_model_pages, render_model_pages

pages = assemble_model_pages([
    {"model": old_model_json, "page_width": 600, "page_height": 800, "scale": 2.0}
], optimize_formula_numbers=True)
markdown = render_model_pages(pages, make_mode="mm_markdown", image_path_prefix="images")
content_list = render_model_pages(pages, make_mode="content_list", image_path_prefix="images")
```

这些显式 stage API 不直接接受 recognize_page_png 的预览块，也不执行模型推理或原生 PDF 字符填充。assemble_model_page/assemble_model_pages 不写图片，image_path 按旧无 writer 逻辑为空；实际写出使用上文 assemble_model_page_with_assets。含 data:image/ 的 HTML 须先完成独立资产处理，当前明确 unsupported。主块缺少模型顺序时明确 unsupported，enhanced/XY-cut 保持延期。单页最多 1000 模型项、200000 节点、64 层、2MB 字符串，native 请求上限 32MB；坐标绝对值≤100 万，页面尺寸 [1,100 万]、scale 为 (0,1000]。多页最多 1000 页并共享 1000 万几何操作预算，再受既有段落总块/行/文本预算限制；输出另限 3200 万累计文字扫描量。非法输入、超限或未迁移行为不返回部分成功。

`sort_model_blocks(blocks, page_width=..., page_height=..., footnote_blocks=None)` 移植旧 `utils/block_sort.py` 中模型带 `original_order` 的完整分支，不需要 experimental 参数。接收旧 stage 的块 JSON，包含 `type/bbox/lines`，图表子块另需 `group_id`；不直接接受 `recognize_page_png` 的预览块结构。保留块顺序优先于 span 顺序、缺失顺序索引为块数、同序稳定、全页行号从 1 开始、旧虚拟行与 real_lines/virtual_lines 处理、图表按 group_id 回组及中位数索引。父块框取首个 body，不推测联合框或图表归属。

返回排序后的新块数组，输入不变。1,206 组完整旧输出及 308 组“旧排序→旧段落”串联输出逐字段/类型一致，后者含 8 组实际 CPU OCR 行。所有块/span 均无有效 original_order 时抛出 `RapidDocError(code="unsupported")`，不会替换成实验几何排序；旧 enhanced/XY-cut 分支仍待移植。

2026-10-24 范围决定：现阶段采用模型自带阅读顺序，enhanced/XY-cut 延期至其他任务完成、用户明确要求后再实现；不是当前集成的前置条件。下一步补齐旧页面组装和输出，接口现有缺失顺序行为保持不变。

合同限有效 Unicode、正宽高 bbox、正页面尺寸，行高截断后中位数须为正，有限坐标绝对值不超过 100 万。顺序和数值 group_id 限 ±(2^53−1)，group_id 支持 JSON 标量并保持 Python 数值键相等语义。上限 1000 块、10000 输入行（含 real_lines）、20000 输入行与新增虚拟行合计、200000 JSON 节点、64 层、2MB 字符串及 32MB JSON。错误输入或超限明确报错；与旧 Python 对非法输入的异常行为不作等价承诺。

```python
from rapid_doc_rust import sort_model_blocks, split_paragraphs

ordered = sort_model_blocks(old_stage_blocks, page_width=600, page_height=800)
pages = split_paragraphs([
    {"page_idx": 0, "page_size": [600, 800], "preproc_blocks": ordered}
])
```

`split_paragraphs(pages)` 是旧 `backend/pipeline/para_split.py` 的独立移植接口，不需要 experimental 参数。接收旧流程已排序/分组的 `page_info_list` JSON，每页含 `page_idx/page_size/preproc_blocks`；text 块含 `bbox/lines`，行含 `bbox/spans`，text span 含 `type/content`。它不直接接受页面块预览结果，也不执行 OCR、排序或导出。

返回新的页列表，保留原 `preproc_blocks` 和未知字段，写入与旧函数相同的 `para_blocks`。支持旧 text/list/index 分类、列表行首尾标记、跨块/跨页合并、cross_page/lines_deleted；原 Python 函数原地写页对象，本接口返回副本。850 组完整旧 JSON 输出逐字段一致，包含浮点几何/元数据和 759 个 Unicode 大写边界字符；使用 float_roundtrip 保持 JSON 数值解析精度。该证据不代表完整解析器集成或整文档质量已验收。

输入为有效 Unicode、正文字块/行 bbox、唯一整数页号和 JSON 字段；最多 1000 页/块、10000 行、200000 节点、2MB 字符串与 32MB JSON。非文字块整体保留，无效结构/几何与超限明确报错；不接受模型/PIL/NumPy 等任意 Python 对象。页内源索引与排序必须由上游旧流程提供，原生产解析器未自动切换。

```python
from rapid_doc_rust import split_paragraphs

processed_pages = split_paragraphs(page_info_list)
# page_info_list 保持不变，processed_pages[*]["para_blocks"] 为旧规则输出。
```

`organize_page_text(page, experimental=True, split_columns=True, line_order="source", block_order="source", max_block_order_work=2_000_000)` 接受 `recognize_page_png/recognize_pdf_page/recognize_pdf_page_hybrid` 的结果，返回独立的 `paragraphs/text/diagnostics/source_line_count`，不修改输入。段落含 `block_index/layout_index/column_index/line_indices/bbox/kind/text/break_reason`；坐标保持 `page_pixels`，行索引相对原块。只在块内合并，保守分栏要求每栏至少两行且垂直重叠。默认 source 保持各栏内的源行顺序及输入块顺序，关闭分栏则保持整个块的源行顺序。间距、缩进、短尾、行高和列表起始可切断段落；标题跳过正文缩进/短尾规则，算法保留原行和缩进。表格/unsupported 作为整体保留且不计入正文源行数；空文字块提供诊断。尚不支持列表嵌套语义或跨页段落。

`block_order="geometry"` 显式重排完整布局块，支持清晰单栏、并排多栏和有留白隔开的通栏头尾；段落索引仍指向原输入，表格/unsupported 整体保留。额外返回 `block_order={indices,reason,work_used}`，source 模式不增加该字段。reason 为 `geometry/already_ordered` 表示已确定几何顺序；`overlapping_regions/invalid_region_geometry/unsupported_direction/ambiguous_geometry` 表示整页保留原序。区域重叠不去重、不部分重排；零面积、越界、竖排或非水平行也不猜测块序。

排序按页面短边 0.5% 的投影留白分隔，并排栏须垂直重叠；通栏宽度须至少为当前区域总宽的 80%。先剥离通栏头尾再处理栏区，避免双栏逐行交错；页中通栏等无法确定的混排保留原序。`max_block_order_work` 为 1～200 万，累计限制区域验证、成对重叠及排序成本估计；超限返回 resource_limit，不能作为普通几何回退。块数仍最多 1000。该策略用于从左到右的横排页面，不自动判断语言书写方向。不提供标题/脚注归属、RTL、重叠仲裁或跨页顺序，不宣称完整旧阅读顺序等价。

`line_order="geometry"` 显式启用栏内从上到下排序，保留原行索引及块顺序；标题也可整理，算法/竖排标签保留原序。整栏存在空行、非水平行、同排片段或垂直重叠超过较小行高 20% 时保留该栏原序。诊断 `lines_reordered` 表示发生排序；`line_order_preserved_overlap/blank/non_horizontal/algorithm/vertical_text` 表示相应保留原因，按块去重。关闭分栏时不会自动横向拼接同排文字。排序后的累计前缀扫描量另受全页 3200 万字节预算，不跨块/跨页重排，也不推测通栏混排顺序。

列表条目可合并同栏内的悬挂续行，保留原标记与源行索引。按首行高度近似 em，续行左缘缩进须为 0.5～2 em，多条续行左缘偏差最多 0.25 em；行高比最多 1.15、行间隙 -0.3～0.7 em，且前行不能是明显短尾。新列表标记、空行、返回正文左缘或不满足几何条件会切断；新增原因 `list_continuation_boundary` 表示续行条件不满足。此策略不恢复齐左续行、嵌套语义或矢量圆点，不宣称完整旧列表解析等价。

`merge_text_lines(lines)` 单独提供旧换行连接规则：CJK/标点、英文断词、样式边界与跨行 URL；固定 Unicode 16.0.0，仅接受有效 Unicode 字符串。最多 1000 行、1MB 去首尾空白后的 UTF-8 文字、3200 万累计前缀扫描字节。段落接口另有全页 1000 块/10000 行/2MB 文字预算，验证全部几何和预算后返回；不以截断或部分成功掩盖错误。这些规则不替代完整旧段落解析器。

```python
from rapid_doc_rust import organize_page_text, merge_text_lines

paragraphs = organize_page_text(page, experimental=True)
print(paragraphs["text"])
# 需要恢复栏内乱序行时显式启用；检查 diagnostics 中的歧义保留原因。
ordered = organize_page_text(page, experimental=True, line_order="geometry", block_order="geometry")
assert merge_text_lines(["文档", "解析"]) == "文档解析"
```

`session.recognize_png(path_or_bytes)` 对一张裁出的文本行执行 BGR/resize/padding、识别模型和 CTC 解码；适用于当前 PP-OCR 识别合同（默认 3x48x320）。`session.detect_png()` 需要 DB 检测模型，使用固定 side=960/max、阈值 0.3、box_threshold=0.5、unclip_ratio=1.6 和 2x2 膨胀，返回原图像素四角与评分。`recognize_regions_png(source, regions)` 按给定四角顺序执行透视裁图、竖行旋转和逐行识别。

```python
from rapid_doc_rust import OnnxSession, ocr_png, ocr_pdf_page

# runtime 为本机 ORT 动态库；模型路径和 SHA256 来自你的本地模型清单。
detector = OnnxSession(runtime, det_model, det_sha256)
recognizer = OnnxSession(runtime, rec_model, rec_sha256)
lines = ocr_png("page.png", detector=detector, recognizer=recognizer)
page = ocr_pdf_page("scanned.pdf", page_index=0, detector=detector, recognizer=recognizer)
print(page.text)
```

`ocr_png` 的检测、同行排序、裁图、识别均在 Rust 内执行，Python 只读输入和筛选置信度。`ocr_pdf_page` 显式渲染选定页再执行 OCR，返回页号、像素尺寸、逐行结果与按行拼接的文字；坐标为渲染图像素。它尚不与原生 PDF 文字合并，不执行方向分类、印章、版面或段落分析。

`orientation_session.classify_orientation_png(source)` 使用 `rapid_orientation.onnx` 返回 `rotation_degrees` 和原始 `score`；RGB、短边 256、中心裁 224 的处理与当前 RapidOrientationModel 一致，5 组张量逐位、4 张 PNG 分类/并发差分通过。这个接口只返回模型分类，不实施原解析器的宽高比/竖行启发式，也不会自动旋转图像。

`parse_pdf_ocr(source, detector=detector, recognizer=recognizer)` 在 Rust 中执行原生/OCR 页面路由：原生文字页保留原文，无原生文字且需要 OCR 的页才调用模型，空白页不调用模型。纯原生文档可省略两个会话，需要时缺模型明确报错。支持 `start_page/end_page`、每页/累计像素和 OCR 页数预算。

结果 `pages[*].text` 是逐页路由后的文字，`pages[*].ocr_lines` 四角采用旋转裁剪后的页面左上 point 坐标；`document` 保留原生事实，已处理的 `ocr_required` 变为 `ocr_completed`。不要将它与 `ocr_pdf_page` 的像素坐标混用。尚不补识别已有少量原生文字页面内的图片块，也没有完整布局和阅读顺序推断。

检测前处理 18 组用例的有效张量逐位一致，24 组 DB 概率图用例的文本框坐标精确一致、评分容差 1e-5。6 组透视/边界/竖行裁图样本逐像素一致；3 张真实文本/旋转/空白 PNG 和对应扫描 PDF 页通过安装验证。这些样本不代表所有复杂扫描件已达到完整 RapidDoc 的解析效果。

命令行新增 `ocr-png` / `ocr-pdf-page`，需要 `--runtime`、`--det-model`、`--det-sha256`、`--rec-model`、`--rec-sha256`；后者支持 `--page`、`--dpi` 和 `--pdfium`。默认 CPU，不下载模型。

`render_pdf_page(source, page_index=0, dpi=144, max_pixels=16_000_000)` 返回自有 PNG 字节和宽高。`recover_pdf_table(source, bounds, page_index=0)` 的 bounds 为旋转裁剪后的左上 point 坐标，返回表格单元格和 HTML；这是已知区域恢复，不包含整页表格检测。

```powershell
.venv\Scripts\python.exe -m maturin build --release --manifest-path bindings/python/Cargo.toml --out build/wheels
```

测试必须安装真实 wheel 后执行 `tests/smoke_install.py`，不把从源码导入视为安装验证。GPU 实机验证由用户后续在服务器执行。

全量隔离安装验证：`.venv/Scripts/python.exe scripts/migration/check_wheel.py --python build/rust-wheel-env/Scripts/python.exe`，包含接口、CLI 与 PDF RGBA 对照；详细日志写入 `build/wheel-all-tests.txt`。


全文辅助文本预览：`prepare_native_document_auxiliary(pages)` 使用与 `prepare_native_document_pages` 相同的 source/geometry 输入，连续执行完整逐页准备、上一页末表轨道传递、重复矢量装饰、全文页眉/页脚/页码伴随/脚注尾段/表前和背景正文恢复、有效/校准两套正文画像及延迟图像脚注。额外返回 `body_profile`、`canonical_body_profile`。`classify_native_document_auxiliary(pages)` 可只处理已准备页面。旧逐页诊断接口继续保持原合同。

此节点在标题画像之前停止；全文标题原型/结构标题、文献区与条目、其余公式和最终文字/行内编排尚待接入，不能将它当作默认 PDF 转换入口。当前验证与安装状态见 [主链记录](../../docs/plans/2026-10-27-form-runtime-integration.md)。


全文文字分类预览：`prepare_native_document_text(pages)` 使用相同 source/geometry 输入，包含完整逐页准备、跨页末表、辅助分类和双正文画像，再完成全文标题原型、校准结构/旧尺度仲裁及未知字重重复标题，额外返回 `title_profile`。对应旧 `_classify_document_text` 全阶段。`classify_native_page_titles` 提供完整旧页面标题分类；`infer_native_document_title_profile` 与 `classify_native_document_titles` 可消费已准备页面。标题探测使用副本，不改源输入。文献、最终公式/文字/行内编排和默认 PDF 接入仍待；不能据此宣称整篇转换完成。安装状态见主链记录。


文献阶段预览：`prepare_native_document_references(pages)` 沿同一 source/geometry 输入连续完成全文文字/标题分类、文献区域、编号条目分组及小字号强调标题，停在最终页面编排之前。`classify_native_document_references(pages)` 单独消费已准备页面；`group_native_reference_lines(page)` 复用完整悬挂条目分组，包括作者年代区域，供最终编排再次调用。跨页文献、附录/译文和作者简介等屏障、脚注来源组回收均沿旧算法。最终公式/文字/行内编排、文献块重建排序、默认 PDF 转换仍待，安装状态见主链记录。

公式恢复预览：`recover_native_formula_blocks(lines, page_size, table_bboxes=None, drawing_lines=None)` 完整移植旧 `_build_formula_like_blocks`，包含拆行/栏内锚点、混排正文、独立数学分量、完整分量替换、正文宿主回退、编号补认领、墨迹尾部及右缘整带仲裁，返回 `formula_blocks`、`remaining_lines` 和保存副作用的 `lines` 副本。`prepare_native_formula_input` 使用相同参数，先执行原密集拆行恢复和同基线合并，返回 `formula_blocks`、`remaining_lines`、`formula_input`。行使用原生视觉行或已准备页的行状态；不在 Python 调用旧解析器，输入不修改。预分类/目录分流、canonical 重放、未映射字形墨迹和最终文字/行内/文献块装配仍由后续页面主链衔接；这两个接口不等于默认整页或整篇 PDF 输出。安装状态见主链记录。

最终公式检查点：`prepare_native_page_formulas(page)` 已将以上内核接入原页面优先级：预分类语义/目录→公式输入/恢复→排除语义来源后的 canonical 重放与精确框库存替换→清快照→唯一未映射墨迹认领→恢复原样式→回退目录及语义分离。返回 `formula_blocks`、`index_blocks`、`semantic_lines`、`remaining_lines` 和已清空的 `canonical_formula_source_lines`。`prepare_native_document_formulas(pages)` 从原始 source/geometry 连续完成全文文献准备并返回额外的逐页 `formula_pages`；原 `pages` 保存文献阶段状态，供调用方明确续接。最终标题/文字/行内聚合、文献块重建排序和默认 PDF 尚未接入。

最终行状态：`prepare_native_final_page_lines(page, page_index=0, body_profile=None, canonical_body_profile=None, title_profile=None)` 从公式检查点连续执行全部最终标题收尾、同行文字合并、文献再次分组和首页机构再次分组，止于 `_build_text_blocks` 之前。返回公式/目录块、最终 `remaining_lines`、空语义暂存/快照及脚注来源组。`prepare_native_final_document_lines(pages)` 从原始 source/geometry 运行全文文献准备并新增逐页 `text_pages`，原 `pages` 保存文献阶段状态。输入不修改。此节点已完成最终行状态；段落块构建与块级重建、排序、归一化和最终行内接线尚未完成，不是默认整篇 PDF 输出。对应关系见[源码映射](../../docs/plans/2026-10-28-final-text-line-integration.md)。
