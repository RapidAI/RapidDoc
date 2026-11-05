# Rust 兼容性审查

2026-10-28 最新安装：最终公式节点及原始页源接线已 release 构建安装，243 页面、6 正向证明、35 整链和两页实时 PDFium 通过；旧公式、旧目录/Form/容器及 118 模块来源/默认门禁全部通过。无全量/CUDA，最终文字/块装配及默认 PDF 未完成；所有整篇等价/提升声明保持 False。详见[主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-28 后续：最终页面公式前缀已连续接入，含 canonical 原闭包/精确框库存替换、清快照、墨迹、样式恢复及前后目录；1037 原语句、6 定向正例、35 原始页源整链通过。重复来源仅在原页面调用路径按对象身份处理，旧独立目录合同保持。118 模块来源清单已补 prepared.rs；最终装配与默认 PDF 仍待，所有整篇等价/提升声明保持 False。安装状态见[主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-28 最新：原完整公式恢复及 dense/baseline 输入链已迁移，来源清单增加 `native_formula_components` 各实现及复用内核。787 前缀、1059 恢复/分量、1023 输入序列和受影响旧合并/完整页合同通过；保留有限的零宽/倒置框和重复来源，未放宽旧公开严格适配器。仍为 118 模块，默认整篇等价及性能/效果提升标志保持 False。最终页面 canonical 等接线和默认 PDF 未完成；安装状态见[主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-28 当前：文献区域/条目分组和前置小字号强调标题已迁移，`prepare_native_document_references` 连通原始页源→全文文字/标题→文献/紧凑标题，抵达最终页面编排前。637 文献序列、24 分组、22 文字合同、35 原始页源整链通过，含三页 PDFium 的跨页文献和附录停止正例；已统一 release 构建安装，3 组相关安装套件通过（新文献、旧标题、兼容性审计），含三页实时 PDFium、严格类型、所有权与并发。剩余最终公式/文字/行内编排、文献块重建排序、默认 PDF 和整篇验收；文献区域/分组内核不再待办。无全量/CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 当前：完整页面标题分类和全文标题原型/校准结构仲裁/未知字重重复标题已实现，`prepare_native_document_text` 连通原 `_classify_document_text` 全阶段。验证为 769 完整页、521 原型、590 全文标题、34 原始页源整链及 43 边界/4 集合轨迹（另 744 前缀）；已统一 release 构建安装到 build/rust-wheel-env，3 组相关安装套件通过（新全文/标题、旧辅助入口、兼容性审计），包含实时 PDFium、严格类型、所有权和并发。剩余文献、最终公式/文字/行内编排、默认 PDF 和整篇验收；标题推断阶段不再待办，最终收尾的标题调用归最终编排。无全量/CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 当前：全文辅助文本/正文画像节点已贯通原始页源、完整逐页准备、顺序末表继承、矢量重复装饰和全部辅助分类/双正文画像。546 组连续旧语句、11 组定向证据、34 组完整文档链通过；新入口 `prepare_native_document_auxiliary`，已统一 release 构建并安装到 build/rust-wheel-env，新文档链、旧完整页和兼容性审计 3 组安装套件通过，含实际 PDFium 新入口和并发/所有权验证。剩余明确为全文标题原型/结构/未知字重标题、文献区与条目、其余公式和最终文字/行内编排、默认 PDF 及用户整篇验收。无全量/CUDA，无性能/质量提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 最新：本轮已完整贯通逐页准备节点：同基线/二维簇/同基线合并→上/下标 sidecar→字符释放前证据冻结→PreparedPage 构建→末尾页辅助复核和末表查询；并接在已有全文几何、容器和脚注链之后。`prepare_native_document_pages(..., complete_page=True)` 可调用，旧阶段默认保持。976 组合并、499 组完整页（含 6 旧异常）、12 组身份/快照、3 组字体旧异常和实时 PDF 完整页通过。已在整个节点统一 release 构建并安装到 build/rust-wheel-env，5 组相关安装套件通过（含新入口/严格类型/并发/所有权/实时 PDF 和旧入口兼容）。剩余全文/跨页语义、其余公式恢复和最终文字编排、默认 PDF 及用户整篇效果/性能验收；不再把完整逐页准备列为待办。无全量或 CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 原主链条目补入容器前页边保护、原文字匹配/页码、网格排除的重复分隔线及全文前缀。207 连续文档和三页实际 PDFium 源对照通过，源码清单仍为 118 模块。后续页内分类和默认 PDF 未完成，不宣称整文档等价或性能提升。

2026-10-27 全文几何计划生成已实现，原条目纳入 planner/X/Y、固定 Unicode 类别及未修改的旧计算内核。覆盖生成器与容器前修复前缀、实际 PDFium 输入对照，保留字体转换错误及来源副作用。模块数仍为 118，无默认 PDF 或质量/速度/内存提升声明，本阶段未安装验收。

2026-10-27 几何阶段补充：现有 `native_form_image_formula_chain` 条目纳入计划应用、原内核视觉重拆和字符身份证据分片源码，模块数仍为 118。旧计划输入的阶段对照与全文计划生成明确分开；来源副作用按原始对象关系保留。完整计划生成和默认 PDF 等价尚未完成。

2026-10-27 新增 `native_form_image_formula_chain` 开发中条目，源码清单共 118 模块，记录实际 Form、全文角色/图像/矢量公式与几何后文字恢复的边界；尚无本阶段安装验收。完整几何与后续公式/文字分类、默认 PDF 仍待，全文等价及性能/质量提升声明保持 False。详见 [当前主链记录](2026-10-27-form-runtime-integration.md)；下方旧清单数量和待办为历史记录。

2026-10-27 目录预认领与 Form 选择已接通：`extract_native_index_blocks`完整移植目录提取（含标题条件、点线、无页码编号、标签续行、语义变化与tight输出），`select_native_form_regions`保留字形归属/边缘ink、保留Form面积例外、证据收紧和来源映射别名。`prepare_native_container_table_page`按明确目录预认领→Form选择→横线/算法代码→表格检测/提交→活跃Form筛选→底色代码执行，目录块按旧次序留在阶段末尾；旧两个页接口范围保持。

验证：242目录＋104Form选择＋70连续编排，共416组旧结果及2个4301位编号旧ValueError；8个受影响Rust单元和1个PDF集成通过。75个不同实际PDF页（新增8＋既有67），4页目录/41页表格/3页代码；Form框与归属元数据在实际PDF测试中为空，Form正例只来自源级对照，不能声称PDF Form采集全链完成。初轮参考导出器的Form框tuple和显式angle投影问题以独立canonical语料修正，原冻结文件保留；只统计正确类型语料。

阶段末1次release wheel、1次隔离安装完成；新目录/Form/连续编排、旧代码容器、旧表格页提交、compatibility、smoke共5组安装通过。审查117模块，runner登记135组安装套件及独立PDF渲染，本轮不全跑，无CUDA。详见[目录/Form阶段](2026-10-27-native-page-containers.md)。

当前剩余：Form完整结构采集/角色归一化及Form/图形/图片块物化、后续几何公式文本分类整体接线，然后默认PDF与整文档差分。目录预认领和Form选择不再列为未实现，原完成标志False。下方为历史阶段记录。

2026-10-27 非表格代码容器及代码/表格顺序已接通：`build_native_code_blocks(source, kind="filled"|"ruled")`完整迁移底色等宽代码、横线限定清单和算法标题清单；由真实字形/行距/缩进或行号槽/文字栏走廊证明，保留字符来源去重、代码空行和旧回退清理。`prepare_native_code_table_page`按旧顺序先认领横线/算法代码建立表格屏障，随后完整表格检测与提交，最后认领底色代码；统一固定块/行认领/续表轨道。原table-only入口继续保留。

验证：227组旧完整代码builder结果（197＋26＋4整数边界）及63组代码/表格阶段组合（60＋3物理网格混排）通过。67个不同实际PDF页已完成Rust实时源对照，其中8页新增代码PDF覆盖四旋转/横线/算法/正文/表格；16组独立builder查询有3组代码正例，组合页准备有3页代码/41页表格。原无竖轨/行身份的合成表被旧代码逻辑误判的结果保持，另3个具备物理轨道和真实行身份的混排源同时保留代码和表格，不用改旧结果宣称效果提高。

9个受影响Rust单元（代码2＋页准备4＋旧空间投影2＋原栏推断1）和2个PDF集成已通过；旧表格紧字框投影仍用原选项，代码新选项保留空行/不额外推断tight空格。Unicode16东西文宽122范围已冻结并有重现校验脚本。阶段末release wheel安装验收已完成：新代码容器＋旧表格页提交、文字栏、表格多行投影、表格块提交、compatibility、smoke共7组通过；发现整数边界后补构建并复验，同一阶段共2次构建；兼容性清单116模块，runner登记134组安装套件及独立PDF渲染验证。本轮无全量/CUDA。

当前剩余：目录、Form选择、图形/图片块提交及后续几何/公式/文本分类整体接线，随后默认PDF与整文档差分。代码检测/物化及表格前后认领顺序不再列为待办；这仍不是完整_prepare_page_source，原完成标志False。详见[代码容器](2026-10-27-native-code-containers.md)。下方代码待办均为历史记录。


2026-10-27 正式整页准备的表格事务链已贯通：新增 `prepare_native_table_page(source, ...)`，从分析视图保护已分类/上游已认领行，自动计算强图形和aligned数值反证、排除确认页眉分隔线、检测完整候选，再回原始行执行字符结构恢复/注释物化/整组提交，最后生成数值网格标记、认领/剩余行与末表续表轨道。调用方不再传表格候选或恢复结论；上游非表格排除框、已提交块和认领证据仍显式传入。检测/恢复/提交/轨道共享Work，失败不发布部分块。

验证：144组唯一来源号有效输入的旧函数连续组合结果通过，另2组重复来源号按既有Rust提交合同明确拒绝（旧Python原结果保留，不算差分通过）；59页实际PDF源→检测→恢复→提交全部通过，41页生成表格、6页生成续表轨道。四排三列数值网格真标记、protected分析视图与原始成员回收、页眉线过滤、已有块/认领/字形映射与恢复失败均有独立参考。贯通时修正旧表注准备对反向绘图线框的过早拒绝；保留反向校准框和重复来源号等原接口拒绝边界。

31个受影响Rust单元测试（新页链2＋块准备/提交5＋完整候选链24）与59页集成通过。安装验收通过：阶段末一次release wheel及一次隔离重装，新表格页链、旧块准备/注释/提交、整页候选、强图形、compatibility、smoke共8组安装套件通过；新增3组分析/原始行副作用安装对照也通过，共147组有效输入对照＋2个合同拒绝。日志 `build/native-table-page-commit-wheel.txt`、`build/native-table-page-commit-install-test.txt`、`build/native-table-page-commit-regression.txt`。兼容性清单115模块，runner登记133组安装套件及独立PDF渲染验证，本轮只跑受影响验证，无全量/CUDA。

当前剩余已缩小到正式页准备的非表格部分：目录/代码/Form选择、图形与图像块提交，以及后续几何修复/公式/文本分类的整体接线；随后默认PDF及整文档差分。这里完成的是表格事务链，不冒充完整_prepare_page_source；`legacy_native_table_recovery/legacy_table_execution`仍为False。详见[表格页准备提交](2026-10-27-native-table-page-commit.md)。以下为历史阶段，“正式整页准备待接”需按上述已完成表格部分理解。


2026-10-27 强图形总编排已实现：新增 `detect_native_strong_graphics(source)`，调用方不传预选图体或字号。沿旧行字体尺度中位数与CJK字宽校准，顺序执行独立矢量、栅格连通图、原生diagram、堆叠柱、栅格坐标图、柱图、同心轮廓、复杂容器、路径坐标轴、绘图线组件十类产生器；统一有字多列网格排除、面积稳定排序、包含拒绝、重叠合并及最终坐标排序。柱图标题语义副作用保留并传入后续分支，返回独立source投影；栅格坐标图仅执行core检测，不误执行描述段落分组。

验证：493组旧完整函数结果（335源级正例、1组标题副作用）及37组真实旧异常对照通过；8个受影响Rust单元测试和59页实际PDF实时源集成通过。既有55页无强图体，新加4方向复杂轮廓PDF有2页接受/2页拒绝，不称4正例。首轮导出器缺少PDFPathInfo默认rectangle_bboxes导致的122个constructor错误不计为检测器异常；原冻结文件保留，另建canonical完整语料纠正投影，正式验收仅使用493＋37。

安装验收通过：阶段末一次release wheel及一次隔离重装，新强图形入口及堆叠柱、栅格坐标图、图形仲裁、diagram、compatibility、smoke共7组安装套件通过。日志 `build/native-strong-graphics-wheel.txt`、`build/native-strong-graphics-install-test.txt`、`build/native-strong-graphics-regression.txt`。兼容性清单114模块，runner登记132组安装套件及独立PDF渲染验证；无全量/CUDA。

当前剩余：正式整页准备和图表提交编排、默认PDF接线及整文档差分。强图形总编排、diagram、固定CJK及完整表格候选已完成；`legacy_native_table_recovery/legacy_table_execution`保持False。不宣称全文等价或质量/速度/内存改善。详见[强图形总编排](2026-10-27-native-strong-graphics.md)。以下为各阶段历史记录，旧“总编排待接”不再是当前状态。


2026-10-27 原生 diagram 组件完整迁移：`detect_native_diagram_components(source, *, em)` 已沿旧函数接通异常CJK指标门控、自动铺片/填充/完整表格检测与闭合占格排除、根层复杂路径连接、正文屏障、两轮图片/路径扩框、重叠合并及时间轴图标/彩色横轨/窄列说明。表格判定由真实输入自动计算，未要求上游注入结果；候选检测与组件共享Work预算。

验证：226组旧完整函数结果（主语料200组含108图体正例，补充26组含tight字框的时间轴/正文路径）、11组旧异常映射通过；2个受影响Rust单元测试（结果/异常及已有预算/恢复）和55页实时PDF源重建集成通过。55页均为inflated指标门控负例，不能算真实diagram正例；主语料缺tight框的旧异常未覆盖，另建补充语料验证正例。源级合成不代表整文档质量提升。

安装验收通过：阶段末一次release wheel及一次隔离重装，新diagram、整页候选、闭合占格、固定字体、compatibility、smoke共6组安装套件通过。日志 `build/native-diagram-components-wheel.txt`、`build/native-diagram-components-install-test.txt`、`build/native-diagram-components-regression.txt`。兼容性清单113模块，runner登记131组安装套件及独立PDF渲染验证；不跑全量/CUDA。

当前剩余：正式整页准备/图表仲裁的整体编排与默认PDF接线、整文档差分；“diagram组件未实现”已消除，完整图形总编排仍需接入这个已完成组件。固定CJK和完整表格候选均已完成；`legacy_native_table_recovery/legacy_table_execution`保持False。详见[diagram组件记录](2026-10-27-native-diagram-components.md)。以下为历史状态，旧待办不代表当前仍缺同一组件。


2026-10-27 固定 CJK 字体运行时已实现：唯一 PDFium 运行库初始化后、开文档前安装旧 DroidSansFallbackFull 字体提供器。保持固定字体 SHA256、缓存身份、系统字体代理、C 回调/system 库入口 ABI、可重入锁及进程所有权；回调永久错误以 runtime_unavailable 传播。原 provider Rust 源码逐字节移植并固定哈希；低频 Python 分类回调替换为 Rust 的五编码严格解码和旧 Unicode casefold，发行包不依赖 rapid_doc/pypdfium2 或 Python 回调。

`native_fixed_cjk_font_runtime=True`。原未嵌入 metric_pair CID 字框及候选失败已消除，已从 ignored 提升为常规回归；旧冻结参考和失败日志保留。新增 `pdfium_font_runtime_info()` 返回字体身份与真实替代计数，`classify_pdf_font_name(bytes, charset=...)` 可独立验证字体名判定。固定字体/编码表编入 Rust，Font/DocVortex/CPython 许可证随 wheel 分发。

源码验证：4771组旧字体分类、3209组独立严格解码序列；6个受影响库单元测试（含回调重入、缓冲区、系统代理、重复释放、panic永久故障与资源哈希）；8页新增实际未嵌入CID PDF覆盖四语言及0/90文字方向，字符字框和16个RGBA像素参考通过。既有58字符查询、47数值候选查询（含原CID失败）及58完整候选页回归通过。不同套件重叠页不相加算独立文档数。

安装验收通过：阶段末一次release wheel及一次隔离重装，新字体、字符、数值候选、全产生器、整页后处理、compatibility、smoke共7组安装套件通过；8页CJK字符及32组RGB/RGBA像素逐一匹配旧版，原CID失败已通过安装包实时源与候选验证。日志 `build/native-fixed-font-wheel.txt`、`build/native-fixed-font-install-test.txt`、`build/native-fixed-font-regression.txt`。112模块兼容性清单，runner登记130组安装套件及独立PDF渲染验证；不跑全量/CUDA。

当前剩余：diagram缺口、正式整页准备/图表仲裁与默认PDF接线、整文档差分；字体不再列为当前缺口。`legacy_native_table_recovery/legacy_table_execution`仍为False，不宣称全文等价或质量/速度/内存改善。详见[固定字体运行时记录](2026-10-27-native-fixed-cjk-font-runtime.md)。以下为历史阶段记录，其中字体False与CID未通过是当时状态。


2026-10-27 整页表格候选收尾：已完成旧 `_detect_table_candidates` 全部产生器与最终后处理，新增 `detect_native_table_candidates`；此前 `detect_native_structural_table_candidates` 保留为后处理之前的投影入口。表题分组（含稀疏横线链）、正文尾段裁剪、Form 排除、续表标题外置、异常 CJK 字框短标题和末排序均沿旧次序执行；页末续表列轨已从最后物理行推导，独立入口 `trailing_native_table_grid(source, prepared)` 接收正式已组装块及剩余行，尚未自动接入默认页准备。

验证：103 组后处理（含四方向/Unicode）、1404 组实际旧完整检测器、26 组末表轨道，共 1533 组结果事实和 3 个旧异常；24 个受影响 Rust 单元测试及 58 页实际 PDF 集成通过。新增 3 页后处理正例使用显式入参候选投影，其中 2 页在完整检测器也改变输出，原有 55 页输出保持。其中新增 28 组边界事实另有定向通过日志。Form 排除的正例来自显式 source.form_bboxes；实际 58 页该字段为空，不代表 PDF Form 读取已验证。短 CJK 标题的 inflated 指标正例为源级合成，不能替代字体运行时验证。

安装验收：阶段末一次 release wheel、一次隔离重装通过；新整页套件及旧全产生器、compatibility、smoke，共4组安装套件通过。日志 `build/native-table-page-policies-wheel.txt`、`build/native-table-page-policies-install-test.txt`、`build/native-table-page-policies-regression.txt`。兼容性清单 111 模块，runner 登记 129 组安装套件及独立 PDF 渲染验证；本轮只跑受影响测试，不执行全量或 CUDA。固定上游 36＋3＋1 文件哈希未改动。

当前剩余：固定 CJK 字体运行时（未嵌入 CID 失败仍保留）、diagram 组件、正式整页准备/图表仲裁与默认 PDF 入口接线，以及阶段末整文档差分。候选检测完成不代表完整 PDF 迁移完成；`legacy_native_table_recovery/legacy_table_execution` 仍为 False，无质量/性能提升或全文等价声明。详见 [整页候选收尾记录](2026-10-27-native-table-page-policies.md)。以下为各阶段历史记录，旧记录中的“待完成”不是当前待办。


2026-10-27 最后两类候选已接通：aligned-numeric与captioned-mixed-numeric完成，包含原生字符区域确认、完整区域边界、多层表头、原生年份切词、贴邻脚注和重复表头单行续表。所有候选产生分支及numeric过滤/完整替换/existing_bounds相对次序均已联合；1503结果事实＋1旧异常、11区域边界和55页实时PDF通过Rust定向验证，阶段验收完成：20个受影响结构链Rust单元测试、1个55页PDF实时源重建集成测试、2个既有对齐数值恢复/正则集成测试通过。阶段末一次release wheel及一次隔离安装；新全产生器、已有五类数值产生器、对齐数值恢复、完整替换、compatibility、smoke共6组安装套件通过。fmt、110模块兼容性审查及固定36＋3＋1上游哈希通过；runner登记128组，本轮未全跑，无CUDA。旧未嵌入CID字体失败仍保留且不计55页通过。日志build/native-all-candidate-producers-install-test.txt、build/native-all-candidate-producers-regression.txt。候选产生器已从待办删除，当前重点转为最终整页策略、固定CJK字体运行时、diagram/默认PDF和全文回归。详见[本轮验收](2026-10-27-native-final-candidate-integration.md)。未嵌入CID字体已知失败仍独立保留；默认PDF完成标志False。

2026-10-27 数值/部分网格整合：新增partial-grid、year-header、metric-pair、unruled-numeric、captioned-pair-text五个完整分支并接入结构页面和完整覆盖替换仲裁。1525参考事实、46页实时PDF源重建通过Rust定向检查，安装验收完成：16个结构链Rust单元测试＋2个完整替换回归通过；实时PDF集成46页通过，1个未嵌入CID失败回归明确ignored并单独重现确认。阶段末一次release wheel和一次隔离安装；新数值候选、原结构链、完整替换、compatibility、smoke共5组安装套件通过。fmt、109模块兼容性审查与固定36＋3＋1上游哈希通过；runner登记127组，本轮未全跑，无CUDA。日志build/native-numeric-candidate-pipeline-install-test.txt、build/native-numeric-candidate-pipeline-regression.txt，已知字体失败日志build/native-cid-font-runtime-known-failure.txt。一般候选由7类剩余减少到aligned-numeric和captioned-mixed-numeric两类。发现未嵌入CID字体的旧固定CJK运行时尚未接入，保留1个明确ignored失败回归，不计入46页通过数；新增独立嵌入字体指标正例，不改写原参考。详见[本轮记录](2026-10-27-native-numeric-candidate-integration.md)。默认PDF完成标志保持False。

2026-10-27 一般表格连续整合：新增完整普通横线候选＋开边竖线、表题短表、浅色表头/裁剪阶段网格、空答案表、右界裁切表单，共6个产生分支，已按旧相对顺序接入结构页面编排。1454个安装对照事实已冻结，Rust定向对照通过；普通36页PDF与扩展41页PDF源重建验证（41包含36，28整合正例）。本轮验收完成：18个受影响Rust单元测试（结构链14＋正文排除/合并4）、36页普通候选及41页结构编排PDF集成测试通过。统一一次release wheel和一次隔离安装，新结构套件＋旧闭合链、正文排除、候选合并、正式组装、compatibility、smoke共7组安装套件通过。fmt、108模块审查和固定36＋3＋1上游哈希通过；runner登记126组，本轮未全跑，无CUDA。日志build/native-structural-table-pipeline-install-test.txt、build/native-structural-table-pipeline-regression.txt。具体剩余7类候选、最终整页策略、diagram/默认入口及完整回归，详见[连续整合记录](2026-10-27-pdf-general-integration.md)。旧条目中的“下一项”是历史状态，不能当作当前待办。

2026-10-27 闭合物理网格整链已实现：页面行自动聚成局部表格行、Unicode表题选择、连通横边/贯穿列轨、两横边稀疏表与相邻页全列轨证明、完整表题/表注组装、已有候选连通扩框，以及filled/tiled优先和各角度闭合候选的页面分支编排。冻结519旧事实（129×3＋132），另24页实际PDF实时源重建后检查四接口，8闭合/16物理候选正例；无预选core/表题/表注结论输入。详见[本阶段](2026-10-27-native-closed-rule-pipeline.md)。普通横线区间候选产生是下一项；本轮没有完成整个一般表格检测器。上一页网格元数据仍显式传入，实际Form框为空，默认PDF/CUDA仍未接通或验证。6组安装套件（新增整链、正式组装、图形仲裁、候选合并、compatibility、smoke）通过；阶段末一次release wheel及一次隔离安装。fmt、107模块兼容性审查和固定36＋3＋1哈希通过；全量runner登记125组但本轮未全跑，无CUDA。日志build/native-closed-rule-pipeline-install-test.txt、build/native-closed-rule-pipeline-regression.txt。

2026-10-27 正式规则候选组装阶段验收：155旧完整普通组装参考（139带注释）和20页实际PDF完整组装查询（7带注释）通过。两个Rust单元测试、一个PDF集成测试及一次release wheel后5组安装套件通过，涵盖方向逆变换/原序类型/体注释分离/负向门控/所有权/并发/已有预算；fmt、106模块审查与固定36＋3＋1哈希保持。本次消掉核心/表题/自动表注正式组装及方向还原；下一项连通网格扩框，规则临时候选自动产生/执行仍待。未跑全量/CUDA。日志build/native-rule-candidate-expansion-install-test.txt、build/native-rule-candidate-expansion-regression.txt。

新增 `expand_native_rule_table_candidate(rows, lines, rule_bboxes, core_row_indices, *, page_size, median_height, angle=0, caption_source_index=None, has_possible_table_notes=True)`，把已完成核心/表题/自动表注组装为正式候选：完整local并集、注释原坐标、体/注释成员分离、bbox/core_bbox方向还原及score0.0默认字段，全部子阶段共享Work。155旧完整普通函数参考（139带注释）与20页实际PDF查询（7带注释）通过Rust验证。本次消掉“正式候选组装与方向还原”；下一项连通网格扩框，自动规则候选产生及页面编排仍待。

2026-10-27 自动表注收集阶段验收：177旧完整收集/角度/链参考（129正例）、75标记事实及20页实际PDF实时字形文字查询（6正例，含辅助紧凑引用和上标引用）通过。2个新单元测试、2个共享行准备回归测试和1个PDF集成测试通过；一次release wheel后5组安装套件通过，fmt、105模块审查及固定36＋3＋1哈希保持。本次消掉自动表注行收集，无需预选表注行；下一项组装正式表格候选并还原方向坐标，完整页面级候选产生仍待。未跑全量/CUDA。日志build/native-table-footnotes-install-test.txt、build/native-table-footnotes-regression.txt。

新增 `collect_native_table_footnotes(rows, lines, rule_bbox, core_line_indices, *, page_size, median_height, angle=0)`，完整接通自动表注收集：明确Note/Source/注、辅助Unicode短标记及表体紧凑token/小且上移字形引用、正文字号参照和连续行字体/字号/缩进/距离限制。返回独立footnote_rows及有序annotation，不再要求提供表注行索引。177旧参考/75标记事实与20页实时PDF查询通过Rust验证，包含实际辅助与上标引用正例。本次消掉“自动表注行收集”；下一项把核心、表题和表注组装为正式表格候选并还原方向坐标，页面级候选自动产生仍待。

规则表核心/表题/注释准备阶段验收：136旧完整helper/核心参考（124表题行/136注释正例）、12页实际PDF实时查询（2表题/7注释正例）通过。2个定向Rust单元测试、1个PDF集成测试及一次release wheel后5组安装套件通过；fmt、104模块审查和固定36＋3＋1哈希保持。本次消掉核心边界/成员、表题续行及注释记录准备；下一项自动表注行收集，完整临时候选转正式尚未接通。未跑全量/CUDA。日志build/native-rule-annotation-preparation-install-test.txt、build/native-rule-annotation-preparation-regression.txt。

新增 `prepare_native_rule_table_annotations(rows, rule_bboxes, core_row_indices, *, median_height, caption_source_index=None, footnote_row_indices=())`，完成临时规则表格的核心边界/成员、表题续行收集和注释记录构造。保留原坐标与local_bbox区别、横向走廊裁剪、表体成员排除、重复来源框合并和注释字典顺序，输入不修改。136旧参考与12页PDF实时查询通过Rust验证。本次消掉核心/表题/注释记录准备；表注行目前显式传入，下一项是自动表注行收集，完整临时候选产生/转正式仍待。

跨正文规则草稿排除阶段验收：151旧完整判断参考（112拒绝）、20页实际PDF实时文字/规则投影查询（2拒绝）通过。2个新单元测试、2个复用词边界回归测试和1个PDF集成测试通过；一次release wheel后5组安装套件通过，fmt、103模块审查及固定36＋3＋1哈希保持。本次消掉跨正文规则草稿排除；下一项为物化所需边界/成员/注释准备，草稿产生/惰性物化仍待。未跑全量/CUDA。日志build/native-prose-rule-conflicts-install-test.txt、build/native-prose-rule-conflicts-regression.txt。

新增 `exclude_native_prose_spanning_rule_drafts(source, candidates)`，完成跨正文规则草稿排除：下方网格尺寸/距离、贯穿竖轨和上下横边界，加上草稿成员中至少三排符合词法/字重/语义门槛的长正文，才拒绝草稿；caption草稿与非0方向保留。返回独立原序候选和拒绝原数组位置。151旧参考（112拒绝）与20页PDF实时查询（2拒绝）通过Rust验证。本次消掉“跨正文规则草稿排除”；草稿产生/惰性物化仍待，下一项是物化所需的规则边界、成员和注释准备。

已物化候选合并阶段验收：158旧完整支持域参考（142变化），28页实际PDF/44组实时检测候选查询（24变化）通过。两个Rust单元测试、一个PDF集成测试及一次release wheel后五组安装套件通过，涵盖注释字典顺序/表体优先/数值类型/完整元数据/所有权/并发/预算恢复；fmt、102模块审查和固定36＋3＋1哈希保持。具体消掉“已物化候选合并与注释冲突消解”；下一项为跨正文规则草稿排除，draft惰性物化/一般产生器编排仍待；未跑全量/CUDA。日志build/native-table-candidate-merge-install-test.txt、build/native-table-candidate-merge-regression.txt。

新增 `merge_native_table_candidates(candidates)`，完成已物化候选合并：分数稳定逆序、同角度首个20%重叠目标、100分强网格拒绝弱候选扩张、动态bbox/local/core并集，以及来源合并、同类注释归并和表体成员优先的注释冲突消解。返回独立candidates，保留目标其他元数据；Python注释行框字典经有序键框对跨Rust桥接，返回整数键字典，原输入不修改。158旧对照（142变化）及28页PDF/44实时检测后查询（24变化）通过Rust验证。本次消掉的是“已物化候选合并”；下一子任务是跨正文规则草稿排除，惰性草稿物化和完整候选产生器仍待。

完整表格替换仲裁阶段验收：157旧完整函数参考（153变化/78替代拒绝）与28页实际PDF/44组实时填充检测后重建查询（24非空变化）通过。两个定向Rust单元测试、一个PDF集成测试及一次release wheel后五组安装套件通过；fmt、101模块审查和固定36＋3＋1哈希保持。候选产生器仍是上游阶段，完整一般检测编排、diagram和默认PDF仍待；未跑全量/CUDA。日志build/native-table-replacements-install-test.txt、build/native-table-replacements-regression.txt。

新增 `prefer_complete_native_table_replacements(base, recovered, replacements)`，完整迁移旧替换仲裁：替代片段与旧核心至少90%较小框交叠，按交集并面积验证共同覆盖，重叠片段不重复计数；先累计所有旧表的拒绝结论，再用最终通过的替代候选删除旧base/recovered。返回原序独立三组候选，保留全部注释/来源/其他元数据，不修改输入。157旧完整参考（153变化/78拒绝）和28页实际PDF/44组实时填充检测后重建查询（24非空变化）通过Rust验证；查询中的半表拆分是测试构造，不是完整年份/数值表候选产生器。一般候选编排、diagram和默认PDF仍待。

闭合网格阶段验收：176旧完整证明参考（117正例）、2旧缺正行高异常、24舍入事实及16页实际PDF/32组实时重建查询（7正例）通过。2个新闭合证明单元测试、2个复用面积适配回归单元测试和1个PDF集成测试通过；一次release wheel后4组安装套件通过，fmt、100模块审查与固定36＋3＋1哈希保持。候选仍为明确上游输入，实际Form提取未迁移；一般候选编排、diagram组件和默认PDF仍待。未跑全量/CUDA。日志build/native-closed-grids-install-test.txt、build/native-closed-grids-regression.txt。

新增 `inspect_native_closed_grids(source, candidates, *, allow_two_by_two=False)`，按原候选顺序返回 `closed_grid_flags`。完整迁移旧候选角度/core回退、较大Form排除、原source ID成员、正行高中位数、贯穿横纵轨道、两位小数去重/端点容差和文字占用行列；缺少正行高且实际需要求中位数时传播输入错误。176旧参考（117正例）、24舍入事实与16页实际PDF/32组实时重建查询（7正例）通过定向Rust验证。候选是显式上游输入，真实Form提取未包含；一般表格候选、diagram组件和默认PDF仍待。

外框填充行带表格阶段验收：165旧完整参考（101正例）、6独立整数/混合float面积算术事实、28页实际PDF实时路径链（4正例）通过。两个定向Rust单元测试和一个PDF集成测试、一次release wheel及四组安装套件通过；fmt、99模块审查和固定36＋3＋1哈希保持。已核对wheel内99模块兼容性记录。未跑全量/CUDA。一般表格候选/闭合网格前置、diagram组件、强图形统一入口和默认PDF仍待。日志build/native-filled-grid-install-test.txt、build/native-filled-grid-regression.txt。

新增 `detect_native_filled_grid_tables(path_infos, *, page_size, excluded_bboxes=())`，完整迁移旧外框填充行带表格候选：页面比例、最大单元格与半行底纹排除、水平行带数量/覆盖/间隙、垂直覆盖与空带、稳定选优及重叠去重。返回独立完整 `candidates`，保留原bbox数值类型，输入不修改；该函数不依赖CJK字框，也不检查rectangle_bboxes数量。165旧完整参考（101正例）、6整数/混合float算术事实及28页实际PDF实时路径链（4正例）通过定向Rust验证。旋转后只有两条水平带的样本沿旧拒绝；一般表格/闭合网格证明、diagram组件及默认PDF编排仍待。

完整柱图候选阶段验收：151旧完整图题保护参考（146保护/107行语义修改）、122旧完整柱图参考（121候选正例/1source修改/1网格排除）与40页实际PDF全实时链（10候选正例/1图题修改/1网格排除）通过。4个定向Rust单元测试、1个PDF集成测试及一次release wheel后的6组安装套件通过；fmt、98模块审查和固定36＋3＋1哈希保持。完整_detect_native_bar_graphics候选产生器已迁移，默认PDF编排、其余强图形产生器和跨页准备仍待；未跑全量/CUDA。日志build/native-graphic-captions-install-test.txt、build/native-bar-graphics-install-test.txt、build/native-bar-graphics-regression.txt。

新增 `detect_native_bar_graphics(source, *, em)`，完成旧原生柱图候选产生器：真实表框/柱组、图题保护及行语义修改、两轮接触柱体/复杂轮廓/栅格/标签扩框，最后执行实际有字网格排除。返回独立 `source` 与原序 `bar_bboxes`，所有子阶段共用累计Work，原输入不修改。另提供 `protect_native_graphic_captions(lines, candidate_bboxes, *, median_height)`，返回独立 `lines` 和 `protected_source_indices`；仅沿旧英文fig/figure数字种子启动。122完整柱图参考、151图题参考及40页实际PDF链已通过定向Rust验证；完成该产生器不等于完整默认PDF编排完成。

柱组前段阶段验收：187旧原函数AST/临界参考（152正例/2上游表框排除）、30旧round事实与20页实际PDF全实时链（6正例页/10柱组/4表格排除）通过。2个柱组定向Rust单元测试、1个复用集合轨迹测试及1个PDF集成测试通过；一次release wheel后5组安装套件通过，fmt、96模块审查和固定36＋3＋1哈希保持。未跑全量/CUDA。图题保护、两轮图体扩展、最终有字网格排除及默认PDF仍待。日志build/native-bar-groups-install-test.txt、build/native-bar-groups-regression.txt。

新增 `prepare_native_bar_groups(source, *, em)`，在同一累计Work中自动接入相接填充表格排除与柱体准备，再执行共同零轴选择、稳定分类方向排序/切簇、位置和长度差/真实间隔证明、数字行几何准入及零轴补边。返回 `table_bboxes`、`admitted_bars` 和 `groups`；每组含原准入数组 `bar_indices`、原行数组 `numeric_line_indices`、`horizontal`、`baseline`、`core_bbox`、补边后的 `bbox` 和 `label_anchor`。em必须正且有限，输入不修改；此接口尚不执行图题保护、两轮图体扩展及最终有字网格排除，不能称完整柱图检测。187旧参考（152组正例）、30旧round事实和20页实际PDF链（6正例页/10柱组/4表格排除）已通过定向Rust验证。

相接填充表格阶段验收：168旧完整候选参考（140正例、2题名转移）及8条CPython集合弹出/删除/缩容轨迹通过；12页实际PDF实时字符/路径/绘图/视觉行→候选→柱体排除链通过（4表格/排除正例、1表头推断网格正例）。三个定向Rust单元测试、一个PDF集成测试、一次release wheel后的四组安装套件通过；fmt、95模块审查及固定36＋3＋1哈希保持。未跑全量/CUDA。完整其他填充网格族、柱组/图体扩展、强图形入口/跨页和默认PDF仍待。日志build/native-tiled-tables-install-test.txt、build/native-tiled-tables-regression.txt。

新增 `detect_native_tiled_tables(source, *, excluded_bboxes=())`，完整迁移旧相接填充单元格候选阶段：真实CJK宽松/紧致框中位数校准、相接矩形组件、行列轨证明、无底纹表头推断网格，以及相邻表格题名转移。返回独立完整 `candidates`，其 `bbox` 可传给柱图准备的 `table_bboxes`；不修改 source。168旧参考（140正例/2题名转移）、12页实际PDF链（4表格及柱体排除正例/1推断网格正例）通过定向Rust验证。测试字体为自行生成的矩形字形，不代表客户文档质量证明；完整其他填充网格族、柱组/图体扩展及默认PDF仍待。

柱图几何准备阶段验收：170 旧参考（123 准入正例）、8 页实际 PDF 实时路径/绘图→准入→重建零轴查询链（8 准入及零轴正例）通过。两个定向 Rust 单元测试、一个 PDF 集成测试及一次 release wheel 后四组安装套件通过；fmt、94 模块审查和固定36＋3＋1哈希通过。未跑全量/CUDA。table_bboxes 必填；完整填充表格检测、柱组/图体扩展和默认 PDF 仍待。日志 build/native-bar-geometry-install-test.txt、build/native-bar-geometry-regression.txt。

第94模块 `native_bar_geometry`：170 旧参考（123 准入正例）与 8 页实时 PDF 路径/绘图→准入→重建零轴查询链（8 正例）。沿旧75%共同基线、颜色/段数/尺寸/方向、95%表格覆盖排除及半线宽扩边；原位置/顺序/JSON类型保持。table_bboxes 必填，实际 PDF 旧上游表格检测为空；未迁移填充表格产生器、完整柱组/图体扩展或默认 PDF 编排。

栅格连接图阶段验收：144旧参考（63连接图/46文字框正例）、20页实际PDF全实时链（5连接图/1文字框，另1网格排除正例）通过；两个定向Rust单元测试和一个PDF集成测试、一次release wheel与五组相关安装套件通过。fmt、93模块审查及固定36＋3＋1哈希保持。无全量/CUDA，完整native diagram/bar产生器、跨页准备和默认PDF仍待。日志build/native-raster-diagrams-install-test.txt、build/native-raster-diagrams-regression.txt。

第93模块 `native_raster_diagrams`：144旧连接图/文字框参考，63连接图/46文字框正例；20实际PDF全实时字符/图像/绘图/路径/视觉行及组件查询框链，5连接图/1文字框、1网格排除正例。保留旧先网格排除、内部短横线/图片覆盖和普通框证据，组件与网格预算共享。旋转页按旧angle0文字网格支持域，完整native diagram/bar产生器/跨页准备/默认PDF仍待。

同心轮廓阶段验收：196旧参考（145正例）、8页实际PDF候选→网格仲裁链（1正例），两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel及四组相关安装套件通过。fmt、92模块审查和固定36＋3＋1哈希保持。无全量/CUDA，完整强图形产生器集合、跨页准备和默认PDF仍待。日志build/native-concentric-graphics-install-test.txt、build/native-concentric-graphics-regression.txt。

第92模块 `native_concentric_graphics`：196旧完整候选事实（145正例）、8页实际PDF全实时文字/路径/视觉行→候选→网格仲裁链（1正例）。保留旧英文fig/figure语法、Unicode数字/空白/I忽略大小写、百分比不按角度过滤、根层描边/同心几何、引线扩框及包含拒绝；原图题角度/180度视觉行支持域保持。完整候选产生器集合和默认PDF仍待，完成标志False。

图形/有字网格仲裁阶段验收：178旧参考（120网格排除正例）、8页实际PDF候选生成→仲裁链（1表格排除正例），两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel及四组相关安装套件通过。fmt、91模块审查和固定36＋3＋1哈希保持。无全量/CUDA；完整强图形产生器集合、跨页准备和默认PDF仍待。日志build/native-graphic-arbitration-install-test.txt、build/native-graphic-arbitration-regression.txt。

第91模块 `native_graphic_arbitration`：178旧网格及原尾段AST最终输出事实，120网格排除正例；8实际PDF候选生成→网格保护→仲裁实时链，1表格排除正例。完整迁移相关网格证明，原候选产生器和median_height仍显式输入。均值使用num-rational精确二进制比率，旧fmean/CPython sum和覆盖+=顺序保持；大整数/混合float面积排序、包含/重叠阈值及union类型保持。完整强图形/跨页/默认PDF仍待，完成标志False。

独立图形阶段验收：229旧参考、41既有实际PDF输入和8页专用实时链（3矢量/1小栅格正例）、两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel和四组相关安装套件通过。fmt、90模块审查及固定36＋3＋1哈希保持。无全量/CUDA；完整强图形/表格仲裁、跨页准备和默认PDF仍待。日志build/native-isolated-graphics-install-test.txt、build/native-isolated-graphics-regression.txt。

第90模块 `native_isolated_graphics`：229旧完整分组/矢量/小栅格事实，137矢量/7小栅格正例；41既有实际PDF源输入和8页专用四方向实时链，3矢量/1小栅格正例。保持旧增量合并顺序、CPython hypot临界距离、字号/颜色/页角/正文栅格覆盖和small-raster阈值，None填色计入颜色集合。组复制和距离扫描预算累计；180度文字覆盖保持旧支持域，完整图形/表格仲裁和默认PDF仍待。

绘图组件阶段验收：239旧事实、41既有实际PDF输入和8页专用实时链（2正例）通过；两个定向Rust单元测试和一个PDF集成测试、一次release wheel和四组相关安装测试通过。fmt、89模块审查及固定36＋3＋1哈希保持。无全量/CUDA；完整强图形/表格仲裁、跨页准备及默认PDF仍待。日志build/native-drawing-graphics-install-test.txt、build/native-drawing-graphics-regression.txt。

第89模块 `native_drawing_graphics`：239独立旧组件/统计/候选事实（134正例），41既有PDF实时线/路径输入及8页专用四方向实时链（2正例）。稳定扫描/并查集/原行首次出现顺序、bbox类型、组件大小/横纵计数/复杂路径证明按旧规则。CPython hypot仅可见性扩大，运算原样。接口无缓存，完整图形/表格仲裁、跨页页源/默认PDF仍待，完成标志False。

复杂路径阶段验收：263旧参考、41既有实际PDF输入和8页专用实时链通过（4容器/2轴正例）；两个定向Rust单元测试和一个PDF集成测试、一次release wheel及四组相关安装套件通过。fmt、88模块审查及固定36＋3＋1哈希保持。无全量/CUDA；完整强图形/表格仲裁、跨页准备和默认PDF仍待。日志build/native-path-graphics-install-test.txt、build/native-path-graphics-regression.txt。

第88模块 `native_path_graphics`：263旧完整路径候选事实（60容器/73轴正例）、41实时既有PDF输入；8页专用实际PDF路径→候选链含4容器/2轴正例。纵轴原序全扫描执行旧最终距离条件，保留原候选顺序/重复来源号排除和union数值类型。有限double point算术/显式median_height，合计5000输出框在构造时检查；完整图形/表格/重叠仲裁与默认PDF仍待。

堆叠柱候选阶段验收：304旧事实、41页真实PDF负例读取、两个定向Rust测试通过；一次release wheel及四组相关安装套件通过。fmt、87模块来源审查和固定36＋3＋1哈希保持。真实PDF无异常CJK/柱图正例，此限制保留；无全量/CUDA，完整图形/表格仲裁及默认编排仍待。日志build/native-stacked-graphics-install-test.txt和build/native-stacked-graphics-regression.txt。

第87模块 `native_stacked_graphics`：304独立旧尺度/候选事实，162柱图正例、210尺度校准；41实际PDF页仅负例，无实际inflated/柱图正例，不混淆合成页源与真实正例。保留旧CJK范围/普通中位数/阈值、柱系列和图体扩框顺序，候选坐标输出float points。完整图形/表格/重叠仲裁及尺度编排仍待，完成标志False。

栅格图归组阶段验收：272组旧参考、16页全实时PDF链、两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel构建及五组相关安装套件通过。fmt、86模块来源审查、固定36＋3＋1哈希保持。无全量/CUDA，完整强图形/跨页准备/候选检测/默认PDF编排仍待。日志build/native-raster-charts-install-test.txt、build/native-raster-charts-regression.txt。

第86模块 `native_raster_charts`：272独立旧完整输出和刻度/图体事实，223图体/151归组正例；16实际PDF全实时页源/图像/噪声/归组链，10图体/2归组正例。最大面积保持首项平局，大整数与混合浮点面积独立精确比较，已有/负分组及同core相邻链覆盖保留。仅修改paragraph_group，旧文字/几何/字符/语义保持；明确JSON/字体/有限数值/累计工作限制。完整强图形检测/跨页准备/默认PDF未完成，完成标志False。

噪声阶段验收完成：236组旧决策、32页实际PDF全实时链（4页删除正例）、两个定向Rust单元测试和一个PDF集成测试通过；一次wheel构建，五组相关安装套件通过。fmt、85模块审查和固定36＋3＋1哈希保持。无全量/CUDA，图注归组/强图形框自动检测/完整默认编排仍待。日志build/native-text-noise-install-test.txt及build/native-text-noise-regression.txt。

第85模块 `native_text_noise`：236旧候选/请求/完整清理副作用事实，199候选/178删除正例；32实际PDF颜色及全实时页源链（4删除正例）。固定rule_text保持原样；Unicode数字/空白/英文词与CJK特征、字体分组、正文/目录/图缝几何、填色模式/浅灰范围/参考亮度与80%比例、重复来源号及缺失char_idx的清理行为保持。支持普通JSON/i64索引/有限几何及限定签名/RGBA，字符可见性重复扫描按字节计费，资源错误传播。强图形框输入不代表完整图形检测已迁移，图注归组/默认PDF仍待。

验证完成：一个定向Rust实际PDF测试、一次release wheel构建及六组相关安装套件通过；10页全实时链与85页图片事实均匹配旧参考，公式88组回归保持。fmt、84模块审查及固定36＋3＋1哈希通过。无全量/CUDA。日志build/pdf-image-infos-install-test.txt和build/pdf-image-infos-regression.txt。

第84模块 `pdf_image_infos`：85旧PDF页/57图、13处平滑背景/20处白边结果精确对照；10页自动字符/绘图/图片读取到局部页源和全文图片决策的链路，取消新链的图片注入。第83模块328决策参考继续独立保留。每图指纹原流16MiB、累计64MiB流读取和64M解码像素、既有256M缩放预算是明确支持边界，资源错误传播；无完整PDF质量/性能等价结论，默认完成标志False。

图片区域阶段验收完成：两个定向Rust测试，四组相关安装套件（native_image_regions/native_source_lines/compatibility/smoke）及一次wheel构建通过；fmt、83模块审查和固定36＋3＋1哈希通过。图片证据自动提取仍待，无全量/CUDA。日志build/native-image-regions-install-test.txt、build/native-image-regions-regression.txt。

第83模块 `native_image_regions`：328组独立旧图片区域决策（33组水印正例、30页含平滑图删除、78处白边输出），64组实际PDF页源/文档用例含11个图片证据输入。保持统计先于背景过滤、不同页计数/面积阈值、空指纹、正文Unicode strip计数、普通中位数及重复框最后证据覆盖。图片指纹/像素证据尚未由SDK自动提取；安装链只实时计算文字/行，不把冻结图片输入计为自动提取。来源和资源限制列于审查JSON，两个legacy完成标志仍False。

第82模块最终wheel验收：249完整输出/诊断、516 sidecar、57实时PDF负例链及四方向专用PDF/7处实际脚本通过；无角色/sidecar结果注入，五项受影响Rust测试及五组相关安装套件通过。字段/类型/码点/顺序、重复来源身份、默认兼容/所有权/并发/限额和错误恢复核对通过；44预分类分式事实仍独立保留原参考。fmt、82模块审查及固定36＋3＋1哈希通过，一次构建，无全workspace/全部100组runner/未受影响渲染/全量clippy/CUDA。日志build/native-script-output-install-test.txt和build/native-script-output-regression.txt。

第82模块native_script_output：完整script_lines构造、主/紧框备用归属、独立短脚本匹配、晚期公式保护、Span与诊断接通。249旧完整输出/诊断、516旧payload事实；44预分类分式输入保持在原560行参考中。57实际PDF链无脚本范围，仅负例；四方向专用实际PDF真实读取/检测产生7处物化脚本。诊断顺序和原bbox类型、重复来源号/原对象位置身份、字段和码点偏移按旧规则。apply省略新参数保持原默认；显式空数组执行旧清理。投影和准备各自累计预算与既有叶子支持域明确；不把这些局部链称完整页源/候选检测或默认PDF等价，默认完成标志保持False。

第81模块最终wheel验收通过：297旧完整输出、57实时PDF字符＋注解到Span链、五组相关安装套件及两个定向Rust测试通过；字段/类型/码点、既有脚本标记、所有权/并发/限额/扫描预算/错误恢复匹配。fmt、来源审查及固定36＋3＋1哈希通过，一次阶段wheel构建，无全workspace/全部99组runner/未受影响渲染/全量clippy/CUDA。日志build/native-inline-output-install-test.txt和build/native-inline-output-regression.txt；明确块输入和未接新script_lines/完整默认PDF的边界保持，完成标志False。

第81模块native_inline_output：297独立旧完整Span输出，57实际PDF字符/注解→视觉行/局部准备/检测链；块为明确视觉行派生输入。几何归属/类型过滤、公式/断词/上下文回退和冲突合并保留旧顺序及码点偏移，物化文字/公式/链接和已有detector-owned平坦sup/sub；未接新script_lines匹配/诊断。链接子文字styles默认数组与顶层省略差异按旧typed dump保留。冻结HTML实体/无效引用和Unicode16属性，接受良构JSON范围/PDF四种外链，不支持任意Python私有区间对象；既有JSON/10M共享预算及明确输入输出上限，资源错误传播。完整块检测/默认PDF和质量性能结论仍待，标志False。

第80模块已安装验收：36实际PDF页/26注解、57组字符＋注解均实时读取的视觉行/局部准备/inline链、所有权/并发/加密/NUL/错误恢复及URI/Quad资源限额通过；两个定向Rust测试（含207 URI）及五组安装套件通过。该新链取消注解注入，原第79模块的阶段参考仍独立保留。fmt、来源审查和固定36＋3＋1哈希通过；只构建一次wheel，无全workspace/全部98组runner/未受影响渲染/全量clippy/CUDA。日志build/pdf-link-annotations-install-test.txt和build/pdf-link-annotations-regression.txt，默认完成标志保持False。

第80模块pdf_link_annotations：旧URI策略/可见性/动作/Quad与Rect回退、CropBox/旋转/裁页/稳定枚举来源号已移植。207旧URI和36实际PDF页/26有效注解为独立参考；两个定向Rust测试通过。URI按CPython urllib和Unicode16 NFKC边界保留原目标，未改用浏览器URL策略；URI/枚举/Quad预算为显式支持限制，普通坏链接跳过而资源异常传播。输入所有权和唯一运行库锁、RAII句柄沿既有边界；仅注解读取，未据此提升默认完成标志或声称完整PDF/性能等价。

第79模块最终wheel验收通过：285旧事实/57实时PDF字符链（链接注解为旧PDF阶段输入），完整区间/字段/类型/顺序、所有权/并发、限额/共享预算/错误后恢复及五组相关安装套件通过。两个定向Rust测试、fmt、来源审查和固定36＋3＋1哈希通过；未跑全workspace/全部97组runner/未受影响渲染/全量clippy/CUDA，日志build/native-inline-evidence-install-test.txt和build/native-inline-evidence-regression.txt。默认完成标志仍False。

第79模块native_inline_evidence：复用固定inline_styles与已验证字体元数据，普通JSON适配严格框/索引排序/紧凑Unicode文本，保留短粗体/项目符号过滤、绘图最佳归属/分数线和墨迹保护及链接冲突/区间。285旧完整事实含10多字符标记边界；57实际PDF字符链有2样式和4链接正例，注解为独立旧PDF读取的阶段输入。修正已由新增参考复现的规范化标记差异，未修改vendor。i64/布尔来源、既有JSON及共享10M预算/明确输入输出限额，错误不降级；SDK注解读取、像素字重及输出投影和完整页源/候选检测/默认启用尚待，完成标志False。

第78模块已安装验证：290旧事实、57实时PDF读取链、全部字段/类型/顺序、所有权/并发/限额/工作预算和错误后恢复通过；两个定向Rust测试及五组相关安装套件通过。fmt、78模块审查和固定36＋3＋1哈希保持。本阶段构建一次wheel，未跑全workspace/全部96组runner/未受影响渲染/全量clippy/CUDA，日志build/native-source-lines-install-test.txt和build/native-source-lines-regression.txt；默认完成标志保持False。

第78模块native_source_lines：290独立旧局部准备事实，首字母/悬挂字母段、22装饰文字线和16薄图片线输出，57真实PDF字符→视觉行→局部准备链。保留稳定原序、原字符和几何、旧分组/标题抑制/文字副作用与绘图规范化/追加顺序；上中位数em、canonical样式修复、Unicode16私用区规则按旧路径。每次候选扫描按页规模/文本累计收费，并接受既有Work；普通JSON/限额为显式支持域，资源异常不降级。图片/绘图区域为阶段输入，不据此证明完整页源、候选检测或默认PDF等价。CUDA检查暂停。

第77模块安装验收通过：317旧事实/977行，57实时PDF原字符读取→视觉行，全部公开字段/类型/来源、所有权/并发/共享预算和错误后恢复匹配；另五组相关回归通过，共六组安装套件。定向Rust测试、fmt、来源审查与固定36＋3＋1哈希通过。本阶段构建一次wheel，未跑全workspace/全95组runner/未受影响渲染/全量clippy/CUDA；日志build/native-visual-lines-install-test.txt和build/native-visual-lines-regression.txt。

第77模块native_visual_lines：新增字符→完整公开LineItem的普通JSON接口；317独立旧事实/977行含57实际PDF字符输入。固定core分行和互为最近上下标配对、Unicode16属性/Sm范围；保留来源号、原字符、source_bbox、排版与原角色标记。开发仅定向测试，阶段完成再统一构建/相关回归，CUDA检查依用户要求暂停。完整页源其他准备、候选检测和默认PDF启用尚待，两个legacy完成标志保持False。

相关安装回归10组及新增媒体PNG共11组通过，另36组渲染像素对照通过；94组runner未全跑。范围含原完整块事务、注释、准备、整表物化、三类资产、model_document、compatibility、smoke，日志build/native-table-media-png-regression.txt。模型编排烟测仍不代替完整原生页检测与效果证明。

媒体PNG已安装wheel验证通过：182 PNG页/502引用＋45真实PDF/125引用，比对解码模式/尺寸/像素/ICC/tRNS与其他HTML/JSON。232项Rust测试、fmt/严格clippy/CUDA feature编译、76模块审查及固定36＋3＋1哈希通过；通用PDF的18 RGBA＋18 RGB组旧像素对照重新通过。编码字节差异仍显式保留，不据此提升两个legacy完成标志或宣称整文档质量/性能改善。日志build/native-table-media-png-install-test.txt、build/native-table-media-png-render-regression.txt。

2026-10-26 第76模块native_table_media_png标记intentional_output_difference：真实PNG编码/裁剪/逆时针旋转、页内部分成功保留和后续文字回退、元数据清理及point千分刻度已移植。182 PNG页/502引用、45真实PDF/角度/链125引用独立冻结；与旧Pillow比解码mode/尺寸/像素/ICC/tRNS及其他HTML/JSON，PNG压缩字节/data URI/URI哈希文件名明确允许不同。原通用render_page的算式保持，新增旧scale2/3500长边RGB入口。无媒体不读取来源；资源错误传播，支持8bit四模式及既有限额。区域为阶段输入，未完成完整页级检测/默认启用，不作性能/整文档质量结论。

完整块事务最终wheel验证通过：781旧事务、八候选缓存及16实时PDF串联逐值/类型/副作用/所有权/并发/预算/异常恢复核对，相关12组安装套件通过。231项Rust测试、fmt/clippy/CUDA feature编译、75模块来源审查及固定36＋3＋1哈希通过；96独立媒体事实由Rust对照，wheel覆盖24完整媒体案例。未全跑93套件/36像素组，未实测GPU；完整页级检测、实际PNG和默认集成仍未据此完成。日志build/native-table-block-commit-regression.txt。

2026-10-26 第75模块native_table_block_commit：781独立旧完整事务、八候选页缓存、96媒体几何/占位事实及16实时PDF原语串联。保留score稳定排序、完整候选框去重、原字符投影、注释整组发布/回滚及来源认领、候选/标题副作用，复用既有引擎和投影。逆向候选框与数值字符索引按旧路径保留；修正行级坏侧映射异常被容错的差异。2个非str Char原始引擎输入不属于PageSource域，排除原因独立记录。资源超限作为显式支持域差异传播，不伪装普通失败。媒体区域为明确阶段输入，只生成临时URI/原文字；PNG、完整页检测、默认接入与质量/性能结论继续待验，完成标志False。

注释相关10组已安装套件全部通过，见build/native-table-annotations-regression.txt。未全跑92套件或36像素组；模型编排中的冻结提供器和真实CPU烟测与新阶段的实时PDF负例分开记录，仍不承诺完整PDF等价或性能改善。

注释检查点最终验证通过：229项Rust测试、fmt/clippy、CUDA feature编译、74模块来源审查和固定36＋3＋1哈希。已安装wheel通过928注释/类型事实、240首页面板、16实时PDF负例、元数据/类型/输入不变/并发/累计Work和异常后恢复检查。日志build/native-table-annotations-install-test.txt；保留默认关闭与实际PDF注释正例尚未覆盖的证据边界。

2026-10-26 第74模块native_table_annotations：完整旧注释排序/视觉分组/文本清理/物理行连接/tight元数据/外置失败集合与首页弱表撤销。904事务/327输出/141分组/183tight/98失败案例，另24类型事实、240首页面板/102撤销及32角色/字段事实。NFKC复用固定Unicode16 crate，casefold与word/space沿固定表；保留坐标高度类型及标题副作用。16实时PDF仅作无注释负例，明确阶段正例不冒充PDF效果。旧显式准备接口保持，新接口自动物化并共享预算；媒体/投影降级/原子提交与默认启用仍待，两个legacy完成标志False，许可证随包。

块级准备相关9组安装套件全部通过，日志build/native-table-block-regression.txt。全部91组及36像素对照未在本轮重跑；模型PDF编排仍含既有冻结提供器与真实CPU烟测，不能替代本原生块级阶段或整文档质量证明。

块级准备最终wheel的624组事实和16实时PDF/角度链（12非空）已通过逐值/类型、字段/来源保留、所有权/并发、累计预算/限额/异常恢复及许可证检查。228项Rust测试、fmt/clippy/CUDA feature编译和73模块来源审查通过，固定36＋3＋1源码哈希保持。非空PDF行由实时SDK字符重建glyph视觉行，4空行输入用作阶段负例；并非完整PDF行检测或最终块提交证明。

2026-10-26 新增第73模块 native_table_block_preparation：612旧准备＋12坐标类型事实，53脚注分离、207成员扩展、46字符框校准；16实际PDF/角度准备→完整恢复→语义HTML链、12非空。只移植表格块准备函数，不提交最终块或认领；failed_annotations为明确阶段输入。行是已有glyph视觉行投影，实际SDK字符与绘图证据单独核对。保留Python数值等价字体分组、源集合/框类型、校准覆盖顺序和分轮同基线闭包，共享10M工作预算。默认PDF完成标志仍False，完整注释输出/弱表撤销/图片/投影回退继续后续实现；MIT与既有Unicode/CPython归属随包。

最新验证：227项Rust测试、fmt/clippy/CUDA feature检查及13组相关wheel套件通过。新整表526／恢复724／实时PDF组合32逐值类型、所有权、并发、恢复错误子类/渲染错误原类、累计10M预算及4MB输出检查通过；72模块来源审查和36＋3＋1上游固定哈希通过。全量runner为90组，本轮只跑相关13组；旧像素与全量记录保留，未作GPU实机或整文档质量/性能结论。

2026-10-26 新增第72模块 `native_table_materialization`：完整旧语义 HTML 与 `recover_table_result` 普通 JSON 路径。526 旧整表 HTML、724 原始输入恢复和32实际 PDF/角度/字体组合独立冻结；168 脚本正例来自中立结果阶段输入，实际 PDF 的效果证据单独统计。保留原HTML回退、原始result.html、字体/跨度/转义、None及恢复/渲染异常区别。复用已有准备/二维/角色/叶子实现，跨新编排共享外层10M预算，保留内核自身限额和4MB输出限制。内部整数非直角与公共四直角契约分开。新增 MIT notice 随wheel。PDF块认领/资产/修复、owned加速、visible-only与默认回退未接入，两个legacy完成标记仍False。下方为历史审查。

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


2026-10-25 字符原语：新增`read_pdf_character_primitives(source, *, page_index=0, pdfium_library=None, password=None, include_geometry=True)`。Rust直接读取旧DocVortex去重前的`chars`，并分别返回代理对恢复后的`restored_chars`；原始码值、来源索引、字体、页内对象编号、渲染模式、生成标记、原点、书写方向和可选loose/tight字框完整保留。基础布局沿旧整数ceil页面尺寸，扩展几何沿浮点CropBox；含隐藏文字时才按旧规则读取透明度可见标记，页面上没有隐藏模式时不新增该字段。

58组真实PDF/扩展模式、7650原始记录及恢复结果已独立冻结，包括50组既有PDF与8组补充ToUnicode/旋转/Form/隐藏透明文字；补充平面代理对、一个字形多码值、孤立高低代理均由实际PDFium读取。上限20k字符/1MB单字体名/8MB累计字体名/32MB宿主JSON，text/page/document按逆序在全局锁内释放，不输出原生地址。该接口不执行DocVortex重复绘制/隐藏副本去重，也不启用visible-only页面裁剪与遮挡过滤；不能拿它替代完整`get_chars_with_geometry`或默认PDF表格恢复。下一步接通这些旧规则，再接适配器语义HTML；完整恢复标志仍False。

本轮最终验证：211项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel已安装到build/rust-wheel-env，全部83组安装包回归与18 RGBA＋18 RGB旧PDF精确像素对照通过。新增58组字符/7650原始记录及代理恢复逐值/类型/可选字段、并发/所有权/密码和错误恢复通过；绘图41页及16组引擎串接再次通过。65模块来源审查、原36份core及新增3份paths源码哈希通过。日志：build/pdf-character-primitives-*.txt、build/rust-current-*.txt、build/wheel-all-tests.txt与build/wheel-all-run-output.txt。


2026-10-25 新检查点：Rust端DocVortex绘图原语读取已接通。`read_pdf_vector_geometry`直接读取PDF页，返回视觉坐标`drawing_lines/path_infos`以及`page_size/page_bbox/rotation`；路径保留来源索引、Form层次、填充颜色和内部矩形。复用固定上游三份原样Rust源码，在现有PDFium全局锁内借用同库20个符号，不再次初始化引擎；对象/段计数先受限，再读取几何。旧深度15、裁剪/旋转、非均匀矩阵、零面积框过滤和共线合并保持。独立基线覆盖41页实际PDF、428条合并线、3694路径；505 hypot位值、507 round3和180合并事实。16组实时绘图＋冻结旧原始字符已接完整表格引擎对照。

验证：210项Rust workspace测试、最终PDF定向复测、fmt、严格clippy及CUDA feature编译通过；新版wheel已安装到build/rust-wheel-env，82组完整安装包回归和18 RGBA＋18 RGB精确像素对照通过。新增41页逐值/类型、16组实时绘图引擎串接、所有权/并发/密码与错误恢复及随包许可证核对通过。64模块来源审查、原36份core与新增3份paths原样源码哈希通过。日志：build/pdf-vector-primitives-*.txt、build/rust-current-*.txt、build/wheel-all-tests.txt和build/wheel-all-run-output.txt。

剩余：旧DocVortex字符多码值恢复/几何/去重、适配器语义上下标HTML和PDF级恢复；现有pdftext字符接口不能替代。`docvortex_pdf_vector_geometry=True`，`legacy_native_table_recovery/legacy_table_execution=False`，默认PDF流程未切换。


2026-10-25 新原语级完整引擎：724实际旧原始字符/页面原语→_evaluate/recover/diagnose参考（547恢复、六来源、9物理冲突、16旧实际PDF/角度完整原语），不mock文字或候选。保留完整调用顺序、列/行/alias守卫、物理冲突/rect解围、aligned覆盖及身份保留。共用Work实际1000万预算，诊断重放不重置；旧表格预算说明误写2000万已修正，运行时上限未改，ftfy独立预算不变。审查现63模块；此处不读PDF、不含任意对象属性coerce适配或语义上下标HTML，PDF级恢复/默认切换仍未完成。

2026-10-25 新banded_numeric：1751旧候选/诊断/重投影普通HTML（262非空、16旧PDF/角度原语），1843数字规则/12共同净空/164有符号来源set/3非法源ID异常。保留众数平局首遇、三层分组表头、表头可重建/正文文字完整性差异；100%捕获和零歧义不变，未新增100%顺序门（原95/96正例接纳）。原set索引函数不改；源ID适配扩展有界负数/重复输入，PSF许可继续随包。审查现62模块，完整_evaluate/PDF原语提取/语义HTML仍未接入。

2026-10-25 新完整多行稀疏：1136候选/诊断/普通HTML（548非空、54成功grouped回退）、356空间投影和114紧框间距事实；16旧实际PDF/角度为给定原语。主失败才回退，整行组标题严格依原始字符来源/显式空白及紧框投影复核，不用普通格文字替代；完整恢复仍False，尚缺banded_numeric/完整_evaluate/PDF原语提取及语义HTML。审查现61模块，未提高质量/性能声明。

2026-10-25 新sparse_hybrid：799条旧候选/诊断/普通HTML参考（571非空、16旧实际PDF/角度原语）。保留连续稠密行平局、物理优先/去重、75%竖线覆盖、60%逐列/80%总体支持及精确mean、两层表头和20%/80%局部分隔、物理token跨列与最终完整性门。新增Python显式入口，不自动进入默认PDF路径，完整原生恢复仍False；审查现60模块。剩余完整sparse_multiline/grouped回退、banded_numeric、完整_evaluate及Rust PDF原语提取不在本项内。

2026-10-25 当前三线表：generate_native_table_rule_band_candidates与inspect_native_table_sparse_columns，386＋2候选/386共享事实（16旧实际PDF/角度）/195 median事实；两个补充保留builder接受但最终严格发布拒绝的区别。共享列/矩形/走廊不是完整sparse_multiline/grouped回退；未提升完整_evaluate、原生恢复或质量/性能声明。

补充金额/状态PDF参考：另8条由两份实际PDF经旧DocVortex提取的角度输入，2条非空候选；与主218条分开冻结，独立wheel逐项通过。合计226条候选/重分行、125非空、16旧PDF/角度事实；Rust此处未执行PDF提取。

2026-10-25 当前数值候选：regroup_native_table_numeric_text及build_native_table_aligned_numeric_candidate按旧执行链移植，218旧参考/532 regex边界逐项对照，包含123非空候选、8旧实际PDF/角度事实。默认regroup；False限制连续row_index，有限几何/advance/尺度与预算明确；源中未调用的numeric_region_bounds未移植。aligned候选的引擎覆盖/冲突决策尚未接入，完整恢复仍False。

2026-10-25 当前：generate_native_table_text_candidates移植少线/无线/key-value，375旧参考（8旧实际PDF/角度事实）覆盖原轨道、续行、分组表头colspan和发布门。inspect_native_table_engine_policy以351旧参考覆盖早期几何/相交计数和已有诊断守卫；不是完整_evaluate。首版2条参考因浅拷贝错配，旧文件保留，stable深拷贝快照另存；没有改Rust迁就错配。完整引擎仍缺数值及其他稀疏候选执行，恢复标志仍False。

补充9条完整路线边界：部分续行在真纵隔断/无隔断时的旧区别、alias冲突和残留窄带拒绝，已在独立wheel逐项对照；完整矢量样本共268条，轨道阶段204条。主参考未覆写。

2026-10-25 当前矢量编排：新增generate_native_table_vector_candidates及inspect_native_table_vector_tracks，248完整生成＋204轨道硬门＋11补充逐项对照。8补充输入为实际旧DocVortex PDF原语/角度，含文本构建后字形，Rust只消费冻结原语，不把pdftext几何冒充DocVortex。raw/supported/rect共用预算和调用内证据缓存，保留旧装饰/halo/晶格/稠密基线/假设选择的拒绝顺序。矢量路线完成不等于完整引擎；其他旧候选、完整_evaluate、语义HTML和Rust PDF原语提取待完成。

2026-10-25 当前拓扑/候选检查点：409旧_build_vector_topology/_materialize_vector_candidate主参考、12门槛/拓扑补充、104 statistics.mean逐值通过。复用固定vendor连通分量和精确二进制和；≤1-0.8缺边、>5%歧义、单行/列100%文字完整性、alias影响空行与最终空行发布门按旧顺序。仅给定规范轨道，尚无完整轨道假设编排/多候选或PDF原语提取，不提升完整恢复/质量性能声明。

追加等距边界：14条实际旧最近物理线参考表明，等距时Python整数索引set的遍历顺序会影响外缘折叠。已按CPython3.14的有界整数插入/扩容/遍历语义保留该行为，204条集合遍历参考通过；PSF许可和来源随wheel分发。旧370条轨道参考未覆写。

2026-10-25 当前物理证据：新增336条旧线段准备/合并/raw-supported轨道及370条旧规范化/alias/双边界对照；全为合成/边界参考。复用固定vendor合并线段；保留零halo的5000展开片段门，halo分支仍允许多片段但受共享工作预算。接口不猜测阈值、不生成最终拓扑或置信度，完整恢复仍False；物理行评分、多假设和PDF原语提取未完成。

2026-10-25 当前候选检查点：给定网格校验/索引、字形落格、单元格来源、拒绝诊断、评分和普通HTML已按实际旧candidate.py对照（327＋22候选、4 HTML）。保留重叠同分取后项、重复字形ID覆盖映射但单元格仍累加、缺轨道退回穷举，以及旧索引/穷举在不一致物理规格上的差异；不额外修正规则。普通HTML仅转义&/</>，不做表头猜测或语义上下标。结构质量明确由上游提供；完整物理候选发现、评估和Rust DocVortex字符几何提取未完成，完整恢复能力仍False。以下旧检查点按各自阶段理解。

2026-10-24。以本仓库现有 Python 解析效果为基准。当前 Rust 是开发预览，不能替换完整 Python 解析器，也没有整文档质量更好、速度更快或内存更低的验证结论。

## 本轮修正

- `organize_page_text` 和 `recognize_pdf_page_hybrid` 必须显式传入 `experimental=True`，否则在读取输入前拒绝调用。它们包含尚未证明等价的几何、列表或原生/OCR 仲裁策略。
- `prefer_native_tables` 默认改为 `False`。即使开启融合实验，也必须另行设置 `prefer_native_tables=True` 才优先采用新的原生网格策略。
- 原有生产包 `rapid_doc` 未接入这些实验。数值内核和旧换行规则保留；不删除候选算法，也不把它们宣称为旧逻辑的改进替代品。
- wheel 分发 `compatibility.json`，可通过 `compatibility_report()` 查看 63 个模块的来源哈希、证据范围和偏差。`capabilities()['compatibility']` 明确不报告完整 Python 兼容或整文档质量/性能验证。

原生文字新检查点：362旧pending/完整字形-行-token（含16实际PDF字符几何/角度），1934旧cell文字/片段与841跨行分隔对照。保留advance与visible框分工、Unicode16 printable/decimal/Latin-NFKC及原来源/None间隔，复用未改动vendor分行/细分。输入为给定JSON字符，尚非Rust DocVortex PDF几何提取、归属/候选评分或HTML物化；默认模型路径和legacy_native_table_recovery=False不变。

原生捷径新检查点：547实际旧触发/接纳策略、425已有原生候选仲裁helper对照；多页恢复器适配点用独立冻结的实际旧PDF结果验证接纳和模型回退。该适配器不是Rust恢复后端；原生候选生成、完整候选诊断和HTML物化仍未完成，legacy_native_table_recovery=False，默认实验网格未提升。原有6模型组合之外新增2旧原生策略组合，只证明编排，不提升整文档质量/性能声明。

最新UNET/双模型：旧已转义文字使用独立raw HTML输出，安全预览保持；旧原HTML选优不再重复转义OCR，复用既有决策优先级并新增有限BeautifulSoup文本解析。28 HTML、5 UNET/5 OCR→UNET、237选优、12调用顺序及10实际CPU路由（5强制比较）对照；6旧五页组合的JSON/Markdown/18RGB资产像素一致。对应默认无空格补识别器模型配置，尚非原生结构捷径/旧全部异常回退；版面/公式响应固定，不提升整文档质量/性能声明。

2026-10-25 最新：表格最终归一化新增12936旧字符串/2270 HTML/6真实模型HTML对照。新增多页`parse_pdf_models`，2条五页实际旧组合的JSON、Markdown/content_list、6份RGB资产精确一致；实际CPU OCR/SLANet，固定版面/公式响应，另真实V3模型PDF冒烟。仅显式V3 rect+SLANet配置，正文OCR按页执行，跨页批组成与旧全批可能不同；后置OCR按文档执行。完整默认UNET/分类/双模型、原生表格结构、自动方向/分类、印章/checkbox、Office和整文档质量/性能仍未验证；模块状态为partial_pipeline。原正文误用表格参数的参考保留，适用正文1.8/表格1.6参考另存，未覆写期望。

2026-10-25 追加旧SLANet matcher/raw网络链：54几何、156 HTML/逻辑、96格式→匹配，6实际plus/1m结构样本和6 CPU OCR串联（2空字跳过结构）。原逆IoU阈值/distance/首项、OCR过滤/原序拼接/粗体及完整chunk过滤、整数诊断类型和逻辑解析quirks保留；不接预览安全网格输出。raw_html尚无最终BeautifulSoup/CJK归一化，不承诺UNET/双模型/完整表格；空cell/三空列和异常显式失败、OCR数量严格、资源/ASCII跨度边界明确。结构网络已在非空分支执行，完整表格执行仍False。

2026-10-24 追加空OCR全图回退：243旧策略/81筛分转义边界、8实际直接TextRecognizer CPU det+rec及16实际predict模型输入参考。默认ch/non-seal/no-merge、原图无预遮罩/白边重试，先0.5过滤再转义；仍空跳过图片/公式并返回can_infer=False，三空列不再识别。已选方向/skip_table_orientation=True，不含direct predict额外方向猜测或RTL/word-box/异常重试。masks切空显式报错保留旧空识别失败，数量不符新增严格报错；结构网络未执行，不提升完整表格/文档声明。

2026-10-24 追加公式/图片模型输入：241旧提示和343实际RapidTableModel输入/精确像素参考、4真实PDF/CPU提示链。公式/布局图片metadata为给定输入，结构网络只捕获输入；不证明完整表格质量。保留旧原图优先/IoU门槛/旋转限制、顺序图片占位与包含删除、inclusive int涂白、F32公式/checkbox及旧转义。JSON元数据由uuid关联外部资产，不保留跨调用Python对象别名；空OCR needs_ocr标记尚未执行旧全图OCR，三空列按旧truthy处理。资源/输入异常显式抛出，完整表格执行仍False。

2026-10-24 追加原生表格文字优先：262旧坐标映射/262原生填充/10实际PDF填充（含逐字符框）、28实际旧整体回退策略，以及10真实PDF→CPU链（2原生/8 OCR）。保留方向/force/ocr_enable守卫、部分原生结果不补OCR、旧全局bbox裁图和对比度语义；表格字符框按独立Bbox对象投影，普通JSON正文原合同不改。无效输入显式报错而非旧吞异常；OCR word-box、原生表格结构捷径、公式图片和结构模型编排尚未完成。

2026-10-24 追加表格方向准备：320旧原生角度/分类门槛、18旋转/通道像素、76实际旧固定响应流程（含4条旋转后沿用原遮罩坐标）及28真实CPU检测/方向流程。2条确实调用方向模型，其余按旧角度/门槛跳过；保留BGR模型输入、180不转、90/270重检测。JSON bbox适配、有限预算及ASCII无下划线float角度字符串为接口范围；无原生表格捷径、原生文字优先及实际结构模型编排，不提升完整表格兼容声明。

2026-10-24 追加表格行级OCR/HTML回填：131框/75重试恢复/209 OCR策略/64实际_process_single_table末段固定响应、遮罩/白边/4原图识别裁图精确对照，8真实CPU DB→CTC另列且含薄图成功恢复。表格unclip1.6、无merge，薄图对齐0按旧None空结果触发8px重试，识别取原图；结果三列文字已按旧规则转义，不接实验matcher/默认候选策略。HTML只执行旧末段，不代表结构模型已执行。分数误差≤1e-5；None投影[]、数量异常报错、旧批失败逐图/word-box回退未复现。原生路线、方向/原生文字、公式图片插入和结构模型编排仍待接入，legacy_table_execution=False；非完整表格等价。

2026-10-24 追加默认ftfy：29517实际ftfy6.3.1＋29517实际旧公式完整post_process精确一致；含全部有效Unicode标量/NFD、实体、10编码/乱码、CESU8/Java NUL、多行HTML auto和Unicode16 NFC。原1352 tokenizer参考全部通过，移除18组旧恒等子集门槛，不覆写旧期望；实际M另补3张Unicode合成图2批/跨页链，累计6批6图仍不是实际文档质量证明。Rust独立内核不调用Python ftfy，源SHA和实际规则/映射冻结，NFC版本锁定16；只实现旧公式所用默认选项和合法Unicode，不含自定义配置/孤立Python surrogate。共享文本工作量20M、128轮/64递归/单字段64KB/临时256KB、模型总文本2MB，超限失败而非截断。S/L与GPU仍未实测。

2026-10-24 追加公式解码/M模型：673实际旧LaTeX修复、1352 tokenizer用例、4额外token配置参考；tokenizers与旧版同为0.22.2。按旧修复与EOS规则解码，保留Python混合list移动索引、align*未转义量词和未启用normalize等细节。通用ftfy尚未迁移，只准入修复后ASCII/基本汉字且无HTML实体的恒等子集，其他明确unsupported；不是完整公式兼容。4批实际M模型CPU参考来自3张合成图，不构成真实文档公式质量或S/L/GPU证据。新原生会话可进入跨页公式回填，生产Python解析器不切换，formula_full_ftfy=False。

2026-10-24 追加公式预处理：88实际旧PP-FormulaNet输入RGB逐字节一致；FP32张量使用公开OpenCV FMA式，与本机旧5.0 IPP的运算次序不同，绝对误差≤1e-6、独立wheel最大2.38418579e-7。不是张量逐位或真实网络输出等价证明。移植Pillow12.2的双线性/双三次/整数reduce及极窄高图纵向优先次序，先放大小边再thumbnail，不替换为OpenCV或image crate滤镜。S/M384、L768，RGB源4M/中间16M像素、每轴65536、目标2..2048、256M工作量；超限拒绝。公式tokenizer/LaTeX/ftfy解码与实际网络仍待实现。

2026-10-24 追加跨页公式任务：100组实际旧_run_formula_recognition固定响应差分，仅验证页/任务顺序、原latex保留与非空回填；不代表公式网络推理。execute_model_formulas通过显式PNG识别器适配器执行任务，先校验全部输入再调用模型，失败不修改原plans。旧zip静默截断改为数量不符报错；完整任务顺序校验、预算和PNG输入预检为新增合同。内置PP-FormulaNet预处理/tokenizer解码尚未迁移，formula_model_inference=False；表格/正式多页API仍未完成。

2026-10-24 追加模型OCR执行：131旧框整理、32识别计划/16裁图、112回填策略与5条真实CPU det1→rec6→模型链精确对照。实际运行时ocr_patch固定960，依赖原始分档被覆盖；定位差异后修正实现，未放宽原断言。pipeline crop旋转门槛2与预览1.5分开，F32弱标量、裁图/宽度过滤/框矫正时序、三位score及<0.5/<0.8策略保留。RGB tRNS不合成白底。新错误传播/原输入保留与旧批失败逐图重试有区别，det批>1与rec跨批同分歧义明确拒绝；完整公式/表格/多页API未完成。

2026-10-24 追加原生OCR计划：182实际旧检测框、54 native/检测输入计划、75实际旧检测器输入RGB像素。共享NumPy pairwise均值、稳定顺序、首span字号、严格阈值、round2、公式切框FP32和逆变换精度保留；need_ocr_det/模式/方向回退及64像素分组遵循旧分支。固定空检测响应仅隔离输入，不冒充实际检测推理证据；旧临时span字段不回写调用方，返回副本。后续det框整理、真实det/rec执行及回填仍待接入，未提升完整质量/性能声明。

2026-10-24 追加旧模型准备：258组类别转换/86过滤、60实际旧任务计划/409 RGB裁图，3组真实CPU布局输出精确对照；predict_model_layout和prepare_pdf_model_inference接通真实布局推理与任务准备。必须显式rect，auto/poly/quad不以矩形代替；不改变默认形状或模型order。PIL/ndarray换ID与PNG，随机表内图UUID投影稳定值并保留原值，crops列intentional_output_difference；其余计划/像素精确。后续OCR/公式/表格任务执行及正式全解析仍待完成，未提升质量/性能声明。

2026-10-24 追加PDF原图提取：102组实际旧参考/432图片像素、17完整页面资产；仅对旧函数运行环境加get_pos=get_bounds别名，原算法保持。当前未适配旧函数因API移除全空，Rust恢复提图列intentional_output_difference；随机UUID原值另存，Rust为稳定pdf-image-page-index。对象解码失败跳过但增加诊断，资源超限/不支持模式明确失败。保留旧局部Form/旋转bbox语义；新assemble_pdf_model_page仍要求调用方提供模型结果，不是完整自动解析器。

2026-10-24 追加PDFium旧分组桥接：read_native_pdf_page按旧pdftext 0.7.1读取并去重字符、分span/行/上下标/块，28完整PDF页面JSON与268边界参考；5组PDF组装/5条真实CPU链改用新读取和RGB渲染。整数/双精度字段、quote_loosebox、CropBox和四方向旋转保留；Unicode16属性固定；每页10万字符、字体名累计32MB。默认阈值不提供自定义，正式模型路由/原图导出仍未完成。下方“桥接未实现”为历史检查点记录。

2026-10-24 追加原生字符填充与OCR裁图：198组旧span、32组完整页面（5组真实PDF事实）、295组旧对比度、15条实际native→真实CPU OCR→编号→段落链/45输出。保留NumPy负切片/裁图过滤、FP64 pairwise和NumPy round、字符稳定归属与int/float字段类型；补测复现并修复total_cmp将正负零重排而改变字符归属的问题，不放宽断言。bbox对象换等坐标数组，np_img换ID+PNG并单独比较原RGB字节；RGB输入、无word_box。尚未实现Rust原始PDFium字符到旧分组事实的自动桥接，正式解析入口未切换。

2026-10-24 追加后置 OCR 与批识别：73 组实际旧后处理，模型用固定响应隔离选取/回填；12 组另行运行实际旧 CPU 批识别，11 组支持项逐原图共同-padding 张量 FP32 位模式一致、模型文字精确/置信度误差≤1e-5，1 组同宽高比跨批改变 padding 明确 unsupported。11 条真实 CPU 响应→旧回填→编号→段落链及33输出精确对照。新 API 返回副本，失败不 pop 调用方字段；np_img 用 JSON 图片引用而非 NumPy 缓冲。正式原生字符填充/裁图来源尚未接入，不提升整文档兼容/质量声明。

2026-10-24 追加旧跨页表格合并：313 组完整旧页面 JSON（288 包含原 99 组，另有 25 空表边界）、864 组完整旧输出、48 组实际单页→编号→段落→合并链及 144 输出。移植原跨度、表头、列映射/回退、稀疏续行、caption/脚注和对象身份规则；HTML 不做输出归一化。输入限定显式正确闭合、非嵌套单表 HTML，错误 HTML 修复/脚本样式/部分非法实体/非正跨度明确 unsupported。批组装 merge_tables 默认 False，正式 PDF/RapidDoc 路由仍未接入；不据此宣称完整文档质量或性能提高。

2026-10-24 追加表内 UUID 图片：57 组实际旧单页/writer、171 组串联输出，原始 JSON/调用及含函数地址的命名种子均保留。新版以 table_fill/页像素 MD5/页号/取整 bbox 稳定命名，审查状态明确为 intentional_output_difference；仅投影捕获到的新旧文件名，其余 HTML、JSON、模式/像素、ICC/透明色和调用顺序精确比较。错误改为显式抛出，不吞异常返回缺图成功。8-bit PNG/非空短 UUID 等资源边界见 README，不包含整文档推理与跨页表格合并。

2026-10-24 追加 HTML 内联资产：9667 组实际旧工具的 HTML、原始字节与文件名，以及 12 组完整旧页面/writer 顺序参考。保留 writer 生命周期去重和不能设置属性时不去重，写入失败不记成功；普通裁图继续重复写入。畸形 padding 合同为本机 CPython 3.14.5，其他小版本的畸形输入差异不作等价承诺。安全扩展名和 WMF/EMF unsupported 为明确边界；table_fill_image_list UUID 转换仍待实现。旧表内 UUID writer 用函数对象字符串（含地址）命名，后续需先记录此非确定性，不掩盖为精确差分。

## 证据分类

| 分类 | 当前范围 | 不能据此推断 |
| --- | --- | --- |
| 旧 Python 抽样差分 | PDF 字符/几何事实与固定 PDFium 渲染、DB/CTC/方向/布局/表格数值、换行连接、表格选优 | 任意输入、其他平台/后端或整文档效果等价 |
| 有意的输出差异 | 安全 HTML 转义与规范化、非法拓扑拒绝、明确资源错误 | 旧 HTML 字节一致或所有旧异常行为兼容 |
| 部分流程接通 | 页面块/表格裁图组装、rect 布局 | 完整 padding/公式遮盖、漏检补偿、资产、标题脚注归属 |
| 新增实验规则 | 段落几何/列表续行、栏内行序/块序、原生文字与表格仲裁 | 旧段落器、enhanced XY-cut 或完整原生规则等价 |
| 尚未实现 | 完整 RapidDoc API/Markdown、中间 JSON、内容列表、资产、六种 Office 等 | 正式迁移完成 |

详细文件映射和证据入口由 `scripts/migration/audit_compatibility.py` 生成到 `packaging/compatibility-audit.json`；`--check` 验证哈希和禁用实验的默认值，不自动提升模块状态。

## 已确认的偏差

运行未修改的 `para_split.py` 和 `block_sort.py`，冻结到 `build/migration/compatibility.json`；再通过真实安装的 wheel 生成 `build/migration/compatibility-differences.json`。

- 35 组适配段落结构合同中，33 组的分类、段界或来源行组合不同。例如一组双栏行，旧段落器保留为一个 list 块，新策略拆成两段 text。
- 15 组已有模型顺序合同中，6 组不同。旧 `sort_blocks_by_xycut_plus` 优先使用 `original_order`，而显式几何模式会按位置重排。
- 5 组算法/竖排/非文字等输入不在该适配器范围，明确列为 skipped，不计为通过。

这些是针对已构造页面合同的结构对照，不是整份真实文档的准确率，更不是“33/35 个文档效果变差”。块序对照使用输入数组秩作为明确的模型 original_order；这验证了原实现的模型顺序优先分支，不能替代无模型顺序时的 enhanced 排序评估。当前实验不能进入兼容默认路径。

## 旧段落内核的后续移植

旧 `para_split.py` 现已作为独立 `split_paragraphs(pages)` 移植，保留 text/list/index 分类、行首尾标记、倒序跨块/跨页合并及全部 JSON 字段。636 组旧完整输出及 214 组补充边界输出共 850 组精确一致；覆盖 759 个大写边界字符、混合脚本、原数字类型、浮点几何/元数据与实际 CPU OCR 行。上述 33/35 等偏差仍描述实验几何段落器，没有用新内核的结果冒充实验已修好。

边界测试证实旧 JSON 默认浮点解析的 1 ULP 偏差会改变列表行标记。旧 Python 在 Rust 舍入后的输入上产生同样差异；启用 float_roundtrip 后恢复精确一致，没有放宽断言。基线脚本为 `export_legacy_paragraphs.py`、`export_legacy_paragraph_boundaries.py`，均拒绝覆盖已冻结文件。

等价范围限于资源预算内的有效 Unicode、正文字几何、唯一整数页号和 JSON 数据；新 API 返回副本，旧 Python 原地写入。尚未接入完整旧排序/模型对象/导出，不代表完整 API 兼容；实验入口仍须显式开启。

## 旧模型顺序分支的后续移植

`legacy_block_sort.rs`/`sort_model_blocks` 迁移旧 sort_blocks_by_bbox 有 original_order 的整条分支，包括虚拟行、全页行号、图表回组和中位数索引；返回 JSON 副本。1,206 组完整旧输出和 308 组旧 sort→para_split 串联输出（含 8 组实际 CPU OCR）逐字段、数组顺序、数字类型与值精确一致。上述实验几何排序的 6/15 偏差仍然保留，不冒充实验规则已修好。

`export_legacy_block_sort.py` 和 `export_legacy_sort_paragraphs.py` 在未修改的旧实现上冻结基线并记录来源哈希，拒绝覆盖已有文件。输入限制与预算见 Python README；无有效 original_order 时明确 unsupported，旧 enhanced/XY-cut 尚未迁移，不通过实验几何回退掩盖缺项。本结果不代表完整文档 API、质量或性能验收。

## 旧模型页面组装与输出

`legacy_page.rs`、`legacy_page/magic.rs` 移植 MagicModel 及旧单页转换的纯 JSON 规则，输入为已识别 OCR 的 raw layout_dets。286 组实际未修改单页函数输出、18 组数值边界与 382 组页面→para_split 串联精确一致；对应旧 ocr_enable=True、image_writer=None 分支，不执行原生 PDF 字符填充或资产 writer。新 API 返回副本，旧函数修改模型数据。

补测已复现将 2^53 与 2^53+1 都转换为浮点后，旧字典去重的保留项发生改变；修复为整数精确比较、整数/浮点精确等值比较，覆盖 bool、u64 边界与多边形 round(x,2)，未放宽输出断言。

`legacy_page/output.rs` 移植默认公式分隔符下的旧 union_make，726 组完整 Markdown/content_list 与 raw-model→组装→段落→输出串联精确一致。171 组包含算法标签的链保守排除并记录原因，算法文本依赖 Magika 语言分类，当前明确 unsupported；没有用固定 txt 代替旧模型。输出器不写资产，不支持自定义公式分隔符或 content_list_v2。

参考由 export_legacy_page.py、export_legacy_page_numbers.py、export_legacy_page_chain.py、export_legacy_output.py 冻结并记录源码哈希，拒绝覆盖旧文件。仍缺公式编号/后置 OCR/跨页表格后处理、实际图片写出、模型推理与正式 PDF API 的完整集成，不提升 document parity/quality/performance 声明。enhanced/XY-cut 保持延期。

## 旧公式编号后处理

`legacy_page/formula.rs` 与 optimize_formula_numbers 移植未修改 `_optimize_formula_number_blocks`，718 组完整 JSON 精确一致；382 组实际旧模型页面→编号→段落及 639 组模型→编号→Markdown/content_list 对照通过。覆盖邻块优先、严格的“后一公式之后不能紧跟编号”条件、同排阈值/间距边界、稳定首个最近候选、既有 tag 相同或冲突、空编号与孤立编号、全角字符及此前结果对象被后续编号修改的行为。

参考由 export_legacy_formula_numbers.py 与 export_legacy_formula_output.py 在旧实现上冻结，记录源码哈希且拒绝覆盖。独立入口返回副本，只改 preproc_blocks；批组装通过 optimize_formula_numbers=True 显式接入，既有默认 False 保留。新输出对照按编号处理后的块排除 169 组算法链，原输出检查点的 171 组排除记录不改写。不运行公式识别模型，仍缺资产、后置 OCR、跨页表格及正式 API，不宣称完整解析器或整文档性能已验收。

## 旧图片裁图与资产 writer

legacy_page/assets.rs 在旧 cut_image_and_table 所在的页面组装位置生成 PNG 和回填路径，计算全部完成后交由目录或 write(name,bytes) 宿主 writer 写出。337 组实际旧单页函数与 writer 的完整 JSON、文件名、调用顺序和解码像素精确一致；另有 337 组资产页→公式编号→段落及 1011 组默认 Markdown/content_list 对照。保留页原模式像素 MD5、bbox 取整和 SHA256 命名、越界补零、原图最后匹配、未归块仍写出与同名顺序覆盖；PNG 压缩编码字节不要求相同。

export_legacy_assets.py 和 export_legacy_assets_output.py 冻结未修改旧实现的结果并记录来源哈希，拒绝覆盖。安装包在无 Pillow/NumPy 环境下用独立标准 PNG 滤波解码比较像素，同时验证目录、回调、写入失败、预算和并发。现阶段支持 8-bit L/LA/RGB/RGBA PNG，原图需显式提供；不含 HTML 内联/表内图片和 PDF 原图自动提取，不提升完整解析器兼容或整文档质量/性能声明。writer I/O 失败不承诺回滚已写文件。

export_legacy_assets_metadata.py 追加 11 组实际旧 writer 合同，比较 ICC 字节/透明色、EXIF 是否继承、原始 PNG 模式/页 MD5 与像素。确认旧默认 PNG save 保留 ICC 与 RGB/L 透明色但不继承 EXIF，Rust 按同样行为实现；没有用“保留更多元数据”改变旧显示行为。单 ICC≤1MB、页和原图 ICC 合计≤8MB。

## 耗时与内存实测

`scripts/bench/compare_text_join.py` 在两个独立进程中顺序执行相同的 13,930 组旧换行合同，1 次预热、5 次重复。Python 与 wheel 均为 Python 3.14.5；每次输出哈希一致。

| 指标 | Python | Rust wheel |
| --- | --- | --- |
| 一批换行的中位耗时 | 约 0.0695 秒 | 约 0.0113 秒 |
| 进程生命周期峰值 RSS | 42,024,960 字节 | 42,094,592 字节 |

该样本批次约快 6.14 倍，包含 Python/Rust 绑定的数据转换，计时不含导入、输入读取或哈希。RSS 包含导入、输入合同和宿主 Python，两者接近，没有证明内存下降；不是内核分配量或整文档峰值。原始重复值、来源路径、版本和输入/输出哈希见 `build/bench/text-join-comparison.json`。未测完整解析器，不能推导全流程性能结论。

## 后续迁移顺序和准入条件

1. 旧模型顺序、段落、页面组装、公式编号、资产/writer、跨页表格、后置OCR、原生字符填充、PDFium到旧pdftext分组、PDF原图提取、实际布局/OCR及默认Markdown/content_list stage已有实现；公式预处理/解码、旧默认ftfy、M模型CPU与跨页回填已接通。UUID文件名为明确有意差异。下一步旧表格执行/回填及正式多页API，并扩充真实文档对照。2026-10-24 按用户要求将无模型顺序的enhanced/XY-cut（含其专属辅助内核）延期，其他任务完成后等待用户明确要求才启动；缺少模型顺序保持unsupported，不用作者几何合同替代旧参考。
2. 冻结旧规则的输入、输出和源文件哈希，覆盖正文、列表、中文/英文、多栏、跨页及已修复的真实回归；不能仅证明文字没丢就认为段落结构一致。
3. 正式默认路径使用已验证的旧行为。候选改进单独开关，记录质量差分，未完成功能不得静默使用候选算法冒充旧实现。
4. 整文档对照必须包含顺序、段落/列表、表格结构和正文、公式、资产及导出，而后在相同模型/线程/设备/输入下用独立进程测耗时和峰值内存。CPU 与服务器 GPU 分开验收。
