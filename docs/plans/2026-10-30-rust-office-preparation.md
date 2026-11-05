# Rust Office 移植准备：依赖、复用边界与实施顺序

日期：2026-10-30。范围仅 DOC、DOCX、XLS、XLSX、PPT、PPTX；不加入其他格式。本轮为准备工作，不改生产解析器、公开API、模型或已安装Rust wheel。

## 当前基线

- 行为基线是项目内已合并DocVortex 0.6.8改动及本地修复的Python实现，不是重新下载的原始0.6.8，更不是此前0.5.16快照。上游合并来源见 `rapid_doc/_vendor/docvortex_core/OFFICE_UPSTREAM_068.json`。
- Office目录含78个Python模块、25,234行；静态扫描这些模块及共用SVG助手，直接涉及13个第三方导入根。标准库、间接依赖与项目自有算法分开记录，不把整个RapidDoc依赖表当作Office依赖。
- 库存及源码SHA：`packaging/office-migration-inventory.json`。发现26条六格式样本路径，按SHA去重为20个文件；同一文件的副本不重复宣称覆盖。
- 当前生产Rust实现尚无Office转换器；`rust/vendor/docvortex-core`主要是PDF/图像计算内核，不能视为已有六格式Rust解析器。
- 现有Python Office清单有20个验收案例；本轮另行冻结当前输出，保留原golden，临时输出见 `build/office-rust-preparation-python-reference/`，持久压缩参考与输入/输出SHA见 `tests/fixtures/office_rust_reference_068/manifest.json`。这一步是建立移植参考，不是Rust解析通过验收。

## Python依赖与Rust替换关系

| Python包/导入根 | 实际用途 | Rust路线 | 结论与缺口 |
|---|---|---|---|
| `python-docx` / `docx` | DOCX包、段落/run、关系、样式、编号、字段、表格 | `zip` + `quick-xml`，小型上下文使用 `roxmltree`；移植本地DOCX规则 | 有通用XML库，不代表现有DOCX语义已经被替代。`docx-rs`、`ooxmlsdk`列为对照候选，暂不整包依赖 |
| `pypptx-with-oxml` / `pptx` | PPTX形状、组合变换、母版/布局、样式继承、表格/图表/公式 | 同一OPC/XML底座 + 本地PPTX规则 | 当前依赖是带oxml能力的fork，不能按普通python-pptx的最低能力移植 |
| `openpyxl` | XLSX读取、单元格显示/富文本、合并、隐藏信息、图片/图表；XLS的中间工作簿适配 | 优先中立 `Workbook/Sheet/Cell/Style` 结构；XML元数据补全；`calamine`作值读取候选 | 已验证calamine能读样本值；尚未验证能覆盖本项目样式、锚点、图表、OLE/OMML等。不能直接替换后宣称等价 |
| `mammoth` | DOCX顶层表格HTML转换，依赖完整文档的关系、样式、编号上下文 | 按本地表格处理流程移植所需子集，直接从结构化单元格输出 | 不重写完整Mammoth。嵌套表、列表、图片、链接和被本地补回的公式必须保留 |
| `beautifulsoup4` / `bs4` | DOCX表格HTML的解析、定位与补入内容 | 新流程直接生成结构化表格及HTML；确有旧HTML输入时再评估HTML parser | 不以正则解析任意HTML，也不默认引入完整浏览器DOM |
| `lxml` | 命名空间、元素查询/修改、XPath、规范化、OMML | `quick-xml`流式读写 + `roxmltree`只读查询 + 明确的节点变换 | 二者不等于XPath/XSLT完整替代。按现有XPath逐个移植；命名空间按URI识别，不能写死前缀 |
| `olefile` | DOC/XLS/PPT和嵌入公式的OLE/CFB流、属性元数据 | `cfb` + 本地兼容适配 | cfb只替换容器；FIB/piece table、BIFF、PPT records、属性集和大小写查找语义仍需实现/对照 |
| `Pillow` / `PIL` | 位图读写、透明度、大小、占位图绘字与字体度量 | 复用 `image`，按Office需要开启JPEG/GIF/BMP/TIFF/WebP；字体/绘制另行验证 | 现有Rust image只开启PNG。编码字节、像素、字体替换需分别验收，不能认为换codec必然像素一致 |
| `metafile-render` | WMF/EMF含EMF+及GDI/ROP等渲染，失败回退 | `emf-core`/其WMF后端作候选；缺失行为参考当前renderer所需模块移植 | **高风险待验证**，不能把位图提取当作完整矢量渲染。详见下文 |
| `resvg_py` | SVG资源光栅化、输出资源边界 | 直接使用 `resvg` / `usvg` / `tiny-skia` | 同一库系列可复用；需核实Python绑定底层版本、字体/系统字体开关及外部资源策略，不能只选最新版后声称像素等价 |
| `pylatexenc` | OMML字符到LaTeX的编码与符号映射 | 移植实际使用的编码选项和映射，不引入无关TeX引擎 | 精确保留命令边界、转义、Unicode、定界符和矩阵输出；不能只做字符替换 |
| `pydantic` | 转换上下文、枚举、单元格等模型校验 | Rust结构体、枚举、校验函数；边界用已有serde | 不必寻找“Rust版Pydantic”，也不把校验逻辑全部删掉 |
| `loguru` | 日志与降级/错误诊断 | 复用 `rapiddoc-core::progress`，原生宿主日志接口与Python侧展示适配 | Rust负责事件，绑定层负责展示；不重复造一套Python解析调度 |

已查阅的官方接口：[quick-xml](https://docs.rs/quick-xml/latest/quick_xml/)、[roxmltree](https://docs.rs/roxmltree/latest/roxmltree/)、[cfb](https://docs.rs/cfb/latest/cfb/)、[calamine](https://docs.rs/calamine/latest/calamine/)、[docx-rs](https://docs.rs/docx-rs/latest/docx_rs/)、[ooxmlsdk](https://docs.rs/ooxmlsdk/latest/ooxmlsdk/)、[image](https://docs.rs/image/0.25.10/image/)、[resvg](https://docs.rs/resvg/latest/resvg/)。选型结论是对本项目需求的判断，不是上述库承诺的完整兼容性。

### 标准库与间接依赖

- `zipfile` → `zip`；`zlib` → 复用现有 `flate2`；`struct` → checked slices及 `from_le_bytes`，不另写OLE文件系统。
- 旧DOC/XLS代码页 → 现有 `encoding_rs` 可作为底层；UTF-16、非法代理、字段控制字符及原错误处理需要明确实现，不把GBK/Shift-JIS文本误判UTF-8。
- `decimal`、`fractions`、日期和Excel数字格式 → 复用已有数值工具/`num-rational`，按 `xls/number_format.py` 移植显示规则，包括1900/1904日期系统等；不统一用f64默认格式化替换原显示结果。
- `mammoth`间接依赖 `cobble`，`openpyxl`间接依赖 `et-xmlfile`，PPTX库安装依赖 `XlsxWriter` 等。不因安装依赖而把其全部写出能力移植进来。
- 当前 `metafile-render` 还依赖 `pyclipper`。裁剪语义随矢量后端整体验证，不能任意换一套布尔几何算法就假定边界一致。
- Office共享的富文本span、链接、图片资源、schema和现有XYCut顺序也属于迁移范围。普通Office顺序按原实现保留，不启动用户已延后的PDF增强阅读顺序任务。

## 不能靠换包解决的自有实现

1. **DOC**：现有 `doc/` 的FIB、piece table、SPRM、段落/字符样式、字段、书签、列表、图片投影。已有 `cfb` 能读出流，不代表能解释Word二进制记录。
2. **XLS**：现有BIFF records/SST/CONTINUE、数字格式、合并/隐藏、图表及OfficeArt；移植中立工作表投影器，避免再生成一整套openpyxl兼容对象后重复扫描。
3. **PPT**：现有persist/user-edit链、文本与样式记录、分组坐标、OfficeArt、图表和公式，保留原记录次序和降级行为。
4. **OOXML业务规则**：Strict/Transitional URI、异常/非标准part路径、relationships、MC AlternateContent、样式/编号/主题继承、字段与超链接、表格合并、图表数据和嵌入对象。
5. **公式**：现有OMML、Equation XML、MTEF及v5、OLE Equation Native、WMF/GIF AppsMFCC comment重组；以本地输出和异常行为为基线。

不扩展为任意Office排版/编辑器，不引入LibreOffice、Word自动化、Python子进程或先转PDF作为默认解析链。公式缓存值与公式文本分别保留，不为迁移额外增加Excel重算引擎。

### 公式与矢量库实际缺口

不能简单写成“Rust没有库”：存在 [mt-mtef](https://docs.rs/mt-mtef/latest/mt_mtef/)、[mt-omml](https://docs.rs/mt-omml/latest/mt_omml/) 等候选。但其文档列出了跳过记录或规范化/信息丢失情况，必须用本地OMML/MTEF参考验证，未达到原输出语义前以移植现有子集为主。

`metafile-render` 本机安装为0.3.0。另查上游main提交 `c3c0ab99917f1f42f4aa4b224cf572d093f11a53`：`rust/Cargo.toml` 为 `metafile-render-native 0.4.1`、`publish=false`、PyO3 cdylib；Rust源码说明只承担点列/ROP字节内核，GDI状态与诊断仍在Python。**它不是可以直接Cargo引入的完整Rust renderer**。源码审查证据存于 `build/office-preparation-research/`；来源：[上游manifest](https://github.com/myhloli/metafile-render/blob/c3c0ab99917f1f42f4aa4b224cf572d093f11a53/rust/Cargo.toml)、[Rust内核](https://github.com/myhloli/metafile-render/blob/c3c0ab99917f1f42f4aa4b224cf572d093f11a53/rust/src/lib.rs)。

[emf-core](https://docs.rs/emf-core/latest/emf_core/)提供EMF/WMF到SVG的候选路径，也不视为已与当前renderer等价。后续应先比较真实矢量资源、字体、透明度、ROP/裁剪和EMF+结果，再决定直接采用、局部补齐或按所需Python模块移植。原有明确的占位/降级行为保留，但不能用新增占位掩盖以前能够渲染的内容。

额外对照：[umya-spreadsheet](https://docs.rs/umya-spreadsheet/latest/umya_spreadsheet/)是完整XLSX对象模型候选；[office_oxide](https://docs.rs/office_oxide/latest/office_oxide/)宣称统一六格式API。它们可以作为能力对照，当前尚无本地差分证明，暂不替代自有规则；不根据名称、功能表或作者性能宣传决定主链。

## 已执行的Rust基础库探针

独立项目：`experiments/office-dependency-probe/`，不加入生产workspace、不修改已安装 `.venv`。

| 库 | 锁定试验版本 | 许可证 | 当前验证 |
|---|---|---|---|
| zip | 9.1.3 | MIT | 三种OOXML打开ZIP并枚举part |
| quick-xml | 0.42.0 | MIT | 流式解析主XML |
| roxmltree | 0.21.2 | MIT OR Apache-2.0 | 从根relationships定位主part |
| cfb | 0.15.0 | MIT | DOC/XLS/PPT枚举CFB streams |
| calamine | 0.37.0 | MIT | XLS/XLSX均读出4个sheet，非空单元格数24/66/30/2 |

本机 `rustc 1.98.1` 编译并运行通过。上述库最高声明MSRV为1.88；这是候选底座的构建验证，不代表全项目MSRV降为1.88。最终生产依赖只在实施时按实际调用引入，保留lockfile，不提前启用所有codec/重算/写出功能。

六格式输入为 `demo/ms_office_docs` 中的各一个样本。DOCX读取47个ZIP entry、PPTX62个、XLSX46个；DOC/XLS/PPT分别枚举14/3/5个stream。结果见 `build/office-dependency-probe.json`。这里只验证容器、XML、单元格读入口，不声称表格、公式、样式或Markdown保真。

```powershell
.venv/Scripts/python.exe scripts/migration/collect_office_preparation.py --output packaging/office-migration-inventory.json
$env:CARGO_TARGET_DIR = "$PWD/build/office-preparation-target"
cargo run --locked --manifest-path experiments/office-dependency-probe/Cargo.toml -- demo/ms_office_docs/docx_01.docx demo/ms_office_docs/docx_01.doc demo/ms_office_docs/xlsx_01.xlsx demo/ms_office_docs/xlsx_01.xls demo/ms_office_docs/pptx_01.pptx demo/ms_office_docs/pptx_01.ppt
```

## 结构与决策记录（准备阶段）

### ADR-O1：单独Office crate，底层能力复用

状态：拟采用，尚未添加生产依赖。

建议 `rust/rapiddoc-office` 管理六格式，按职责组织 `package`、`docx`、`spreadsheet`、`presentation`、`legacy`、`equation`、`media` 和内部模型。依赖现有core错误/进度/基础契约，Rust facade负责输出适配。不是每种Python包或每个小函数拆一个crate；文件是否拆分由实现体积和职责决定。

优点是XML/OLE、公式、表格与资产集中复用，Office不依赖PDFium才能读正文。代价是需明确Office逻辑页/sheet/slide与现有PDF page schema的适配，不能生搬PDF字符模型。备选“所有代码堆进rapiddoc或按格式建六crate”暂不采用。

### ADR-O2：成熟容器库 + 原规则移植

状态：基础库探针通过，语义适配待实施。

采用ZIP/XML/CFB成熟底座，不自写完整容器。没有经过差分验收的高层库不接管默认解析规则。保留本地损坏包规范化、字段、表格、公式和异常处理；对照库不成为运行时Python回退。代价是需要移植业务规则，但避免新库默认行为悄然改变效果。

### ADR-O3：结构体与受控生命周期

状态：实施约束。

内部使用结构体/枚举、借用和必要的 `Arc<[u8]>` 等共享载荷；关系、样式、编号按文档建立一次索引。按sheet/slide及可释放的部件消费，避免解压全部媒体或把每个单元格格式化/编码多次。JSON只用于公开输出与诊断边界。跨阶段不构造包含整份文档的 `serde_json::Value` 请求。

large worksheet用流式XML，少量styles/relationships/公式片段可用只读树；UTF-16等不能仅开启quick-xml encoding就认为支持，应先规范解码，其官方文档明确说明需中间解码层。复用XML底座时保留原损坏/不支持内容诊断，不静默吞掉未识别节点。

### ADR-O4：许可证、原行为与验收

状态：实施约束，具体复制文件逐项记录。

复制/改写第三方源码前记录repo、版本/提交、文件、许可证和修改范围，随分发保留声明；仅参考已核实授权的部分。Mammoth、pylatexenc、OMML来源链不能仅靠DocVortex主仓库许可证概括。

沿用Python已有Office边界与错误类别，包括资源/损坏/OLE/包缺失错误；不额外发明Python没有的字符/公式数量上限。注意Python Office本身已有 `limits.py`，不是照搬此前PDF中已取消的额外Rust预算。

## 实施顺序与受影响验证

1. **O00 基线与底座**：固定20案例/输入SHA，补齐六格式能力矩阵；实现生产OPC/XML/CFB、关系与typed输出契约。准备探针不能替代这一阶段。
2. **O01 共享语义**：富文本/链接、资产、表格HTML、公式入口与嵌入对象，先冻结代表性公式/图片参考。
3. **O02 DOCX**：包规范化、段落与样式/编号/字段；完整表格上下文、公式、图片与结构化span对齐。选择DOCX先做是为验证共享模型，并非只做纯文本。
4. **O03 XLSX**：中立sheet模型、现有region/projector规则、合并/隐藏、显示格式、图表、图片/公式锚点。避免先接calamine值再把样式作为永久缺失项。
5. **O04 PPTX**：母版/布局/样式继承、组合变换、顺序、标题/列表、图表与嵌入公式。
6. **O05 DOC/XLS/PPT**：复用前述模型和CFB，按现有二进制parser逐项移植；不是用文本提取库代替原能力。
7. **O06 公开入口、分发与阶段验收**：原Python兼容API接入，薄绑定，原 `.venv` 更新；逐格式输出/资产/错误差分，以及独立进程速度与峰值内存比较。

开发只跑受影响格式/部件的参考与回归；阶段完成再统一构建、安装和六格式检查。核心风险是公式/矢量渲染、表格复杂上下文、样式继承、损坏包容错和二进制记录；已知风险应成为差分案例，不作为删功能或降精度的理由。CUDA与PDF增强阅读顺序仍不在本次范围。

实施更新：O00共享底座已开始实现并通过受影响测试，当前状态与未完成范围见 [共享底座实施记录](2026-10-30-office-foundation-implementation.md)。准备阶段的“尚未实现”描述保留为当时记录，不能视为完整Office已经完成。
