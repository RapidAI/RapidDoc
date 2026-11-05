# Rust 运行库验证记录

2026-10-28 当前节点：原 PDF 宿主路由已接入 Rust 原生全文，`parse_pdf_models` 支持 `off/rule/rule_plus_s`、`native_only` 和内置原生区域表格恢复，贯通模型回退、OCR、公式、段落、跨页表格、Markdown、content_list 和图片资产。已完成 release 构建安装（8 分 51 秒）；本阶段安装对照、输出位置和限制统一见[宿主接入验收](2026-10-28-native-pdf-host-integration.md)。跨页 OCR 批次、BGR 渲染和边界坐标类型均按原 Python 保留，未改冻结参考或降低算法精度。`docvortex_mode` 默认仍为 `off`，原生模式显式传 `rule`；增强阅读顺序、CUDA 与同条件性能测量后置，不能据局部样本宣称全部 PDF 等价或普遍更快/更省内存。下方“宿主/Markdown 待接”为历史检查点。

2026-10-28 最新安装：最终公式节点已 release 构建安装，243 页面/6 正例/35 原始页源整链、两页实时 PDFium，以及旧公式/目录容器/兼容性套件通过。无全量/CUDA；剩余最终标题/文字/行内/文献块装配排序和默认 PDF、整篇验收。详见[主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-28 后续：最终页面公式连续接线通过 1037 原语句、6 目录/canonical/墨迹正例、35 原始页源整链。新增 `prepare_native_page_formulas`/`prepare_native_document_formulas`，止于 `title_container_bboxes` 前，后续标题/文字/行内及块排序仍待。前一公式内核已安装并通过 203+227、两页实时 PDFium 和旧完整页/兼容性套件；本次接线安装状态见[主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-28 最新：公式完整识别/仲裁及 dense/baseline 输入链已实现；受影响 787 前缀、1059 恢复/分量、1023 输入序列和旧合并/完整页合同通过，含重复来源、零宽/倒置框的原行为；两页实际 PDFium 正例参考已冻结。统一构建/安装结果见[主链记录](2026-10-27-form-runtime-integration.md)。最终页面 canonical/语义目录/墨迹接线、文字行内/文献块重建排序、默认 PDF 与整篇验收仍待。无全量/CUDA，无效果/性能提升声明。下方为历史记录。

2026-10-28 当前：文献区域/条目分组和前置小字号强调标题已迁移，`prepare_native_document_references` 连通原始页源→全文文字/标题→文献/紧凑标题，抵达最终页面编排前。637 文献序列、24 分组、22 文字合同、35 原始页源整链通过，含三页 PDFium 的跨页文献和附录停止正例；已统一 release 构建安装，3 组相关安装套件通过（新文献、旧标题、兼容性审计），含三页实时 PDFium、严格类型、所有权与并发。剩余最终公式/文字/行内编排、文献块重建排序、默认 PDF 和整篇验收；文献区域/分组内核不再待办。无全量/CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 当前：完整页面标题分类和全文标题原型/校准结构仲裁/未知字重重复标题已实现，`prepare_native_document_text` 连通原 `_classify_document_text` 全阶段。验证为 769 完整页、521 原型、590 全文标题、34 原始页源整链及 43 边界/4 集合轨迹（另 744 前缀）；已统一 release 构建安装到 build/rust-wheel-env，3 组相关安装套件通过（新全文/标题、旧辅助入口、兼容性审计），包含实时 PDFium、严格类型、所有权和并发。剩余文献、最终公式/文字/行内编排、默认 PDF 和整篇验收；标题推断阶段不再待办，最终收尾的标题调用归最终编排。无全量/CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 当前：全文辅助文本/正文画像节点已贯通原始页源、完整逐页准备、顺序末表继承、矢量重复装饰和全部辅助分类/双正文画像。546 组连续旧语句、11 组定向证据、34 组完整文档链通过；新入口 `prepare_native_document_auxiliary`，已统一 release 构建并安装到 build/rust-wheel-env，新文档链、旧完整页和兼容性审计 3 组安装套件通过，含实际 PDFium 新入口和并发/所有权验证。剩余明确为全文标题原型/结构/未知字重标题、文献区与条目、其余公式和最终文字/行内编排、默认 PDF 及用户整篇验收。无全量/CUDA，无性能/质量提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 最新：本轮已完整贯通逐页准备节点：同基线/二维簇/同基线合并→上/下标 sidecar→字符释放前证据冻结→PreparedPage 构建→末尾页辅助复核和末表查询；并接在已有全文几何、容器和脚注链之后。`prepare_native_document_pages(..., complete_page=True)` 可调用，旧阶段默认保持。976 组合并、499 组完整页（含 6 旧异常）、12 组身份/快照、3 组字体旧异常和实时 PDF 完整页通过。已在整个节点统一 release 构建并安装到 build/rust-wheel-env，5 组相关安装套件通过（含新入口/严格类型/并发/所有权/实时 PDF 和旧入口兼容）。剩余全文/跨页语义、其余公式恢复和最终文字编排、默认 PDF 及用户整篇效果/性能验收；不再把完整逐页准备列为待办。无全量或 CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 容器前页边与连续文档前缀完成：207 组前缀对照，三页实时 PDFium 源（6 页眉、3 页脚、3 页码、3 重复分隔线排除）通过。源码入口 `prepare_native_page_marginals` / `prepare_native_document_sources`；当前无新版 wheel、安装验收、全量或 CUDA。

2026-10-27 全文几何生成器已通过 75+42+13+19 场景对照（包含 4 旧异常；复用语料不累计为独立文档），146 组生成/容器前修复前缀、10 页实际 PDFium 全行状态与原几何侧表→全文计划对照。原风险/run/样式内核源码未修改；完整接口尚未发布安装。当前不跑全量或 CUDA。

2026-10-27 几何续接：已有计划应用、视觉重拆、样式/链接对齐和后容器连续阶段已做定向旧版对照（66/27/113 场景及 5 组来源副作用）。10 页实际 Form 几何由旧生成器构建计划，仅验证 Rust 消费计划；未验证 Rust 计划生成。仍无本阶段安装、全量或 CUDA 验收。

2026-10-27 最新开发验证及未发布边界见 [Form/图像/矢量公式主链](2026-10-27-form-runtime-integration.md)。源码清单 118 模块；本阶段只有受影响 Rust 对照和实际 Form PDF 集成，尚未构建 release wheel、运行安装验收、全量回归或 CUDA。下文安装记录均为历史版本。

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

柱图几何准备：`prepare_native_bar_geometry` 提供共同基线、准入/方向和零轴扩边，固定上游 compound_baselines 内核保持不变。170 旧参考（123 准入正例）、8 实际 PDF 实时链（8 准入及零轴正例）；table_bboxes 为必填上游输入，尚非完整柱图检测或默认 PDF 路线。

栅格连接图阶段验收：144旧参考（63连接图/46文字框正例）、20页实际PDF全实时链（5连接图/1文字框，另1网格排除正例）通过；两个定向Rust单元测试和一个PDF集成测试、一次release wheel与五组相关安装套件通过。fmt、93模块审查及固定36＋3＋1哈希保持。无全量/CUDA，完整native diagram/bar产生器、跨页准备和默认PDF仍待。日志build/native-raster-diagrams-install-test.txt、build/native-raster-diagrams-regression.txt。

栅格连接图/普通文字框新增 `detect_native_raster_diagrams`，144旧事实（63连接图/46文字框正例）、20实际PDF实时输入链（5连接图/1文字框、1有字网格排除正例）。组件和网格复用同一累计预算，无结论注入；完整native diagram/bar集合与默认PDF仍待。验证见 `2026-10-26-native-raster-diagrams.md`。

同心轮廓阶段验收：196旧参考（145正例）、8页实际PDF候选→网格仲裁链（1正例），两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel及四组相关安装套件通过。fmt、92模块审查和固定36＋3＋1哈希保持。无全量/CUDA，完整强图形产生器集合、跨页准备和默认PDF仍待。日志build/native-concentric-graphics-install-test.txt、build/native-concentric-graphics-regression.txt。

同心轮廓阶段新增 `detect_native_concentric_graphics`，196旧参考（145正例）及8页实际PDF全实时候选/网格仲裁链（1正例）。编号英文图题/百分比/短引线和原顺序保持；完整候选集合/跨页/默认PDF仍待。验证见 `2026-10-26-native-concentric-graphics.md`。

图形/有字网格仲裁阶段验收：178旧参考（120网格排除正例）、8页实际PDF候选生成→仲裁链（1表格排除正例），两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel及四组相关安装套件通过。fmt、91模块审查和固定36＋3＋1哈希保持。无全量/CUDA；完整强图形产生器集合、跨页准备和默认PDF仍待。日志build/native-graphic-arbitration-install-test.txt、build/native-graphic-arbitration-regression.txt。

有字网格保护/候选仲裁新增两个SDK入口，178旧参考（120网格排除正例）、8页实际文字/绘图/组件候选→仲裁链（1表格排除正例）。精确均值/行中心与覆盖顺序沿旧实现，网格结论由实际页源计算；完整候选产生器集合/跨页准备/默认编排仍待。验证见 `2026-10-26-native-graphic-arbitration.md`。

独立图形阶段验收：229旧参考、41既有实际PDF输入和8页专用实时链（3矢量/1小栅格正例）、两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel和四组相关安装套件通过。fmt、90模块审查及固定36＋3＋1哈希保持。无全量/CUDA；完整强图形/表格仲裁、跨页准备和默认PDF仍待。日志build/native-isolated-graphics-install-test.txt、build/native-isolated-graphics-regression.txt。

独立矢量/小栅格阶段新增 `detect_native_isolated_graphics`，229旧参考（137矢量/7小栅格正例）、41既有实际PDF源输入和8页专用实时链（3矢量/1小栅格正例）。原组序/几何类型/覆盖策略保持，距离及组扩展复制均累计计费，验证见 `2026-10-26-native-isolated-graphics.md`。完整强图形/表格仲裁和默认PDF仍待。

绘图组件阶段验收：239旧事实、41既有实际PDF输入和8页专用实时链（2正例）通过；两个定向Rust单元测试和一个PDF集成测试、一次release wheel和四组相关安装测试通过。fmt、89模块审查及固定36＋3＋1哈希保持。无全量/CUDA；完整强图形/表格仲裁、跨页准备及默认PDF仍待。日志build/native-drawing-graphics-install-test.txt、build/native-drawing-graphics-regression.txt。

绘图组件阶段新增 `detect_native_drawing_graphics`，239旧参考（134图形正例）、41既有实际PDF输入及8页专用实时链（2图形正例）。CPython hypot复用算术不改，组件原序/类型保持。验证见 `2026-10-26-native-drawing-graphics.md`，完整强图形/表格仲裁和默认PDF仍待。

复杂路径阶段验收：263旧参考、41既有实际PDF输入和8页专用实时链通过（4容器/2轴正例）；两个定向Rust单元测试和一个PDF集成测试、一次release wheel及四组相关安装套件通过。fmt、88模块审查及固定36＋3＋1哈希保持。无全量/CUDA；完整强图形/表格仲裁、跨页准备和默认PDF仍待。日志build/native-path-graphics-install-test.txt、build/native-path-graphics-regression.txt。

复杂路径阶段新增 `detect_native_path_graphics`，263旧事实、41既有实际PDF输入及8页专用四方向PDF实时链（4容器/2轴正例）。保留原序/重复/union类型，输出数量和扫描均有累计限额。完整图形/表格仲裁及默认编排仍待，验证见 `2026-10-26-native-path-graphics.md`。

堆叠柱候选阶段验收：304旧事实、41页真实PDF负例读取、两个定向Rust测试通过；一次release wheel及四组相关安装套件通过。fmt、87模块来源审查和固定36＋3＋1哈希保持。真实PDF无异常CJK/柱图正例，此限制保留；无全量/CUDA，完整图形/表格仲裁及默认编排仍待。日志build/native-stacked-graphics-install-test.txt和build/native-stacked-graphics-regression.txt。

新增CJK图形尺度和堆叠柱候选阶段 `detect_native_stacked_graphics`，304旧参考（162柱图正例/210尺度校准），41实际PDF实时字符/路径输入仅负例。接口显式em/fallback，不修改source；完整强图形/表格仲裁及默认PDF仍待。验证记录见 `2026-10-26-native-stacked-graphics.md`。

栅格图归组阶段验收：272组旧参考、16页全实时PDF链、两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel构建及五组相关安装套件通过。fmt、86模块来源审查、固定36＋3＋1哈希保持。无全量/CUDA，完整强图形/跨页准备/候选检测/默认PDF编排仍待。日志build/native-raster-charts-install-test.txt、build/native-raster-charts-regression.txt。

栅格图说明阶段新增 `group_native_raster_chart_descriptions`，固定rule_graphics刻度内核不改。272旧事实（223图体/151归组正例）及16页全实时字符/图片/图像过滤/噪声/归组链（10图体/2归组正例），仅改独立source.paragraph_group。阶段验证见 `2026-10-26-native-raster-charts.md`。完整强图形/跨页准备/默认PDF编排仍待。

噪声阶段验收完成：236组旧决策、32页实际PDF全实时链（4页删除正例）、两个定向Rust单元测试和一个PDF集成测试通过；一次wheel构建，五组相关安装套件通过。fmt、85模块审查和固定36＋3＋1哈希保持。无全量/CUDA，图注归组/强图形框自动检测/完整默认编排仍待。日志build/native-text-noise-install-test.txt及build/native-text-noise-regression.txt。

原生噪声阶段接通候选→按需纯填色RGBA→来源号/字符索引清理，新增四个SDK入口。236组旧决策、32页实际PDF实时链（4删除正例）。不注入颜色或候选结果；强图形框仍为显式输入，图注归组/完整默认编排待后续。验证详见 `2026-10-26-native-text-noise.md`。

验证完成：一个定向Rust实际PDF测试、一次release wheel构建及六组相关安装套件通过；10页全实时链与85页图片事实均匹配旧参考，公式88组回归保持。fmt、84模块审查及固定36＋3＋1哈希通过。无全量/CUDA。日志build/pdf-image-infos-install-test.txt和build/pdf-image-infos-regression.txt。

图片证据读取阶段新增 `read_pdf_image_infos`，原生流指纹和像素背景/白边证据现从PDF自动读取。85实际旧PDF页、57图片、13平滑背景/20白边；新增10页全实时字符/绘图/图片→局部页源→图片裁决链。Pillow算术共享至core，公式预处理纳入本轮受影响回归；所有FFI仍使用唯一运行库锁。记录见 `2026-10-26-pdf-image-infos.md`。

图片区域阶段验收完成：两个定向Rust测试，四组相关安装套件（native_image_regions/native_source_lines/compatibility/smoke）及一次wheel构建通过；fmt、83模块审查和固定36＋3＋1哈希通过。图片证据自动提取仍待，无全量/CUDA。日志build/native-image-regions-install-test.txt、build/native-image-regions-regression.txt。

图片区域阶段新增 `prepare_native_image_regions(pages)`，旧全文水印统计→逐页背景过滤→标题白边裁剪保持原顺序。328组旧决策、64组实际PDF页源用例；图片证据仍是显式输入，实时链只读取文字并准备行。受影响测试/阶段末wheel验证记录见 `2026-10-26-native-image-regions.md`，无CUDA或完整PDF性能结论。

脚本证据/输出最终wheel已验证：249完整输出/诊断、516 sidecar、57实时PDF负例链和四方向专用PDF/7处实际物化脚本通过，五项受影响Rust测试及五组相关安装套件通过；含原完整行内输出/精炼/compatibility/smoke回归。fmt、82模块审查和固定36＋3＋1哈希保持；一次阶段构建，无全workspace/全部100组runner/未受影响渲染/全量clippy/CUDA。日志build/native-script-output-install-test.txt和build/native-script-output-regression.txt。

脚本输出阶段：249旧完整脚本输出/诊断、516旧sidecar构造事实（44预分类分式输入另留原560参考），57实际PDF负例链；另四方向专用实际PDF产生7处真实物化脚本。五项受影响Rust测试通过，含原297样式/链接参考与真实PDF集成。新增第82审查模块/100组全量runner；只在阶段结束构建wheel/相关回归，无全量/CUDA。

行内输出最终wheel已验证：两个定向Rust测试、297旧完整Span输出、57实时PDF字符＋注解链和五组相关安装套件通过，含字段/类型/码点、所有权/并发、未闭合公式扫描预算/输入限额/错误恢复。阶段末一次构建；fmt、81模块审查及固定36＋3＋1哈希通过。无全workspace/全部99组runner/未受影响渲染/全量clippy/CUDA；日志build/native-inline-output-install-test.txt和build/native-inline-output-regression.txt。

样式/链接最终投影阶段：297旧完整InlineSpan输出/57实际PDF字符和注解链，两个定向Rust测试通过。包含公式空洞/断词/歧义回退、几何归属、链接冲突/跨行合并、Unicode偏移和已有平坦脚本标记；新script_lines投影另续。新增第81模块审查/99组全量runner，按开发定向/阶段末构建相关回归执行，无全量/CUDA。

PDF注解最终wheel已验证：207旧URI、36实际页/26注解的两个定向Rust测试、57实时字符＋实时注解链和五组相关安装套件通过；原字符读取/inline证据/compatibility/smoke回归通过，另检查流所有权/并发/加密/错误恢复及作者URI/Quad资源输入。fmt、80模块审查、固定36＋3＋1哈希保持；本阶段一次wheel构建，无全workspace/全部98组runner/未受影响渲染/全量clippy/CUDA。日志build/pdf-link-annotations-install-test.txt和build/pdf-link-annotations-regression.txt。

PDF外链注解阶段：新增read_pdf_link_annotations，207旧URI事实与36实际PDF页/26有效注解的定向Rust对照通过；覆盖可见性/动作/UTF8/URI、Quad优先/去重/回退、CropBox/旋转和原枚举索引。新增第80模块审查/98组全量runner；两份作者PDF仅作URI/Quad资源限额检查。唯一运行库锁和局部句柄生命周期沿既有规则；像素字重/最终输出投影及完整页源/候选检测/默认集成仍待，无全量/CUDA。

原生样式/链接最终wheel已验证：285旧事实、57实时PDF字符链和五组相关安装套件通过，含2样式/4链接正例、码点区间/类型/顺序、所有权/并发/限额/累计预算/错误后恢复。两个定向Rust测试、fmt、79模块审查及固定36＋3＋1哈希通过；多字符标记边界的真实差异触发一次必要重建。未跑全workspace/全部97组runner/未受影响渲染/全量clippy/CUDA；日志build/native-inline-evidence-install-test.txt和build/native-inline-evidence-regression.txt。

原生样式/链接阶段：285旧完整事实（含10多字符项目符号边界）、57实际PDF字符→视觉行→页源局部准备链，含2样式/4链接正例。链接注解仍为旧PDF冻结输入，字形像素粗体补证、SDK注解读取与输出投影另续。两项定向Rust测试通过；首次构建后发现并复现项目符号规范化差异，修正后额外重建一次，无全量/CUDA。新增第79模块审查/97组全量runner。

页源局部准备阶段已验证：两个定向Rust测试、290旧事实、57实时PDF字符→视觉行→局部准备链和五组相关安装套件通过（新source_lines、原visual_lines、block_commit、compatibility、smoke）。fmt、78模块审查及固定36＋3＋1哈希保持；阶段末只构建一次wheel，未跑全workspace/全部96组runner/未受影响渲染/全量clippy/CUDA。日志build/native-source-lines-install-test.txt和build/native-source-lines-regression.txt。

页源局部准备阶段：prepare_native_source_lines接通首字母/悬挂字母段、薄水平栅格图及重复私用区装饰文字转线，290旧完整事实含57实际PDF链，22装饰线/16薄图线输出。两项定向Rust测试通过，覆盖已有Work不重置；第78审查模块/96组全量runner，本轮仅阶段相关验证，CUDA暂停。完整样式/链接/图像/噪声/跨页准备和候选检测仍待。

原生视觉行本阶段已完成：定向Rust测试、317旧事实/977行的安装对照、57实时PDF字符→视觉行与五组相关回归（共六组安装套件）、77模块审查/fmt/固定36＋3＋1哈希通过；wheel只在阶段结束构建一次。日志build/native-visual-lines-install-test.txt、build/native-visual-lines-regression.txt。未跑全workspace、全95组runner、未受影响渲染、全量clippy或CUDA。

原生视觉行阶段：317完整旧事实/977行，含57实际PDF字符输入；复用固定vendor分行/配对与Unicode16表。开发阶段仅定向测试，阶段结束一次wheel构建与相关回归。用户已明确暂停CUDA编译/GPU验证；不再每轮跑全workspace、全安装runner或未受影响的PDF渲染对照。页源其他准备和候选检测继续后续。

本轮11组相关安装套件全部通过，含新增媒体PNG及原块事务/注释/准备/物化、三类资产、model_document、compatibility、smoke；另36组像素对照。全量94组runner未全跑，日志build/native-table-media-png-regression.txt。

媒体PNG最终wheel：232项Rust测试、fmt/严格clippy/CUDA feature编译、76模块审查及固定36＋3＋1哈希通过；182 PNG页/502引用与45真实PDF/125引用的解码模式/尺寸/像素/ICC/tRNS及其余HTML/JSON一致。通用PDF渲染36组（18 RGBA＋18 RGB）已重新核对旧像素，全部通过。使用只读冻结PDF选项--existing-corpus。日志build/native-table-media-png-install-test.txt和build/native-table-media-png-render-regression.txt；GPU仅编译验证。

2026-10-26 媒体PNG：182旧PNG页/502引用（32页内部分成功失败、14ICC/tRNS），45真实PDF/旋转/越界页/旧事务链125引用，包含3500长边和4组占位→实际编码。核对解码模式/尺寸/像素/元数据及其他JSON/HTML，Rust/Pillow压缩字节、URI和URI哈希名允许差异。渲染仅有媒体页、scale2/3500 RGB，Point入口沿旧千分刻度；新增76模块来源审查、94组安装runner，页级检测/默认集成仍待。

完整块事务检查点验证通过：231项Rust测试、fmt/clippy、CUDA feature编译，75模块来源审查和固定36＋3＋1哈希。安装wheel核对781整流程＋八候选缓存、16实时PDF串联、类型/副作用/所有权/并发/共享预算/异常恢复；12组相关安装套件通过。96独立媒体事实由Rust直接对照，完整wheel覆盖24媒体案例。见build/native-table-block-commit-*.txt、build/native-table-media-rust.txt和build/rust-current-*.txt；93套件和36像素组本轮未全跑，GPU未实机测试。

2026-10-26 完整块计算：新增native_table_block_commit套件，781旧事务（192原生/512投影/102弱表撤销/109注释回滚/24媒体/4恢复异常）＋八候选缓存及96媒体几何（40附着56拒绝），16实际PDF/角度原语→完整计算串联。行重建自glyph视觉行；媒体区域是明确阶段输入，非图像检测或PNG实测。来源审查75模块、runner93组；新编排共享10M，资源错误传播，PNG编码/完整页检测/默认集成仍待。

注释检查点10组安装回归通过：native_table_annotations、native_table_block、native_table_materialization、native_inline_clusters、table_script_preparation、native_table_engine、character_dedup、model_document、compatibility、smoke。日志build/native-table-annotations-regression.txt；92组全量和36像素对照未在本轮全跑。

注释最终wheel已安装并验证928注释/类型、240首页面板及16实时PDF负例；所有权/并发、准备与注释各自成功但组合超10M预算失败、异常恢复和许可证检查通过。229项Rust workspace测试、fmt/clippy/CUDA feature编译、74模块来源检查及固定36＋3＋1哈希通过。日志build/native-table-annotations-install-test.txt、build/native-table-annotations-wheel.txt、build/rust-current-*.txt；GPU未实机运行。

2026-10-26 注释与弱表撤销：新增native_table_annotations套件，904完整旧注释事务＋24坐标/高度类型、240首页面板/102撤销、32角色/元数据事实；16实际PDF/角度实时原语及glyph视觉行串联为无注释负例。正例为显式LineItem阶段数据。来源审查74模块，安装runner92组。新准备/注释共享10M预算，输出4000块/8MB和既有连接限额；默认PDF接入和GPU实机未完成。

块级准备相关安装回归共9组通过：native_table_block、native_table_materialization、native_inline_clusters、table_script_preparation、native_table_engine、character_dedup、model_document、compatibility、smoke。日志build/native-table-block-regression.txt；未重跑全部91组或36像素对照。

块级准备验证已通过：228项Rust测试、fmt/clippy、CUDA feature编译；安装wheel的624旧准备/类型事实和16实时PDF/角度串联（12非空），含来源/字段保留、并发、累计10M预算超限、输入限额和异常后恢复。非空PDF行实时由SDK字符重建glyph视觉行，4空行阶段输入为负例。73模块来源检查及固定36＋3＋1哈希通过。日志build/native-table-block-install-test.txt、build/native-table-block-wheel.txt及build/rust-current-*.txt；GPU仅编译，未实机运行。

2026-10-26 新增块级准备套件 native_table_block：612旧准备＋12类型事实、16实时PDF/角度准备→恢复→语义HTML链，含共享预算分阶段各自成功而组合超限的守卫测试。全量安装runner现有91组。该阶段为普通JSON纯CPU准备，实际PDF行由已有glyph视觉行显式投影；不作为完整PDF行检测、完整块物化或GPU运行证明。来源审查增至73模块，固定36＋3＋1上游源码保持。

最新验证：227项Rust测试、fmt、严格clippy、CUDA feature编译及13组相关已安装wheel套件通过；72模块来源审查与36＋3＋1固定源码哈希通过。1500字符分别恢复/渲染成功而累计串联超限，4MB转义HTML超限与异常后恢复均通过。新整表实测详见build/native-table-materialization-install-test.txt，相关回归见build/native-table-materialization-regression.txt，构建见build/native-table-materialization-wheel.txt。未重跑现有全部90组或36像素组，未做GPU实机和性能/内存改善承诺。

2026-10-26 整表语义 HTML / 结果恢复：新增 `render_native_table_html_with_scripts` 与 `recover_native_table_result`，后者返回 None 或 `(content, result)`。旧 Python 基线包含 526 完整整表渲染（168 自动上下标，明确中立结果阶段输入）、724 原始字符/绘图恢复（547 非空）和 32 实际 PDF/角度/字体组合（24 非空、4 内容变化）。SDK 实时读取 chars、drawing_lines、path_infos、tight_bboxes 和 origins 后进入完整恢复与语义渲染；不混载旧 pypdfium2 与 SDK PDFium。外层10M预算跨新编排阶段共享，恢复/渲染错误边界明确；默认 PDF 块级接入仍待完成，GPU 仅编译验证。下方为历史检查点。

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


2026-10-25 原语级完整引擎：724未mock的原始字符/线/矩形→旧_evaluate、NativeTableResult与全路线诊断，547恢复覆盖六来源、9物理冲突、16旧实际PDF/角度完整原语。所有阶段和诊断重放共用现有Work实际1000万；原阶段公开API预算独立，文档之前2000万的表格预算误记已修正，ftfy独立预算保持。仍未从Rust读取DocVortex PDF原语或接适配器语义HTML，PDF级恢复与默认切换未完成。

验证：208项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel在独立环境完成全部81组安装包回归和18 RGBA＋18 RGB旧PDF像素精确对照。724完整引擎评估/恢复/诊断逐值及类型对照、候选身份保留、所有权/并发与共享预算（失败后可再次恢复）通过；63模块来源审查和36份vendor哈希通过。日志：build/rust-current-*.txt、build/native-table-engine-rust.txt、build/native-table-engine-budget.txt、build/wheel-all-tests.txt及build/wheel-all-run-output.txt。

2026-10-25 分带数值：1751实际旧候选/诊断/重投影普通HTML（262非空、16旧实际PDF/角度原语），1843数字语法、12共同字形净空、164有符号set及3非法源ID异常；原众数首遇平局和95/96顺序接纳正例保留，正文/表头投影规则不统一改写。给定原语/字符，完整引擎和PDF原语提取仍未接入。

验证：206项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel在独立环境完成分带数值、多行稀疏、融合、三线、数值对齐、文本三路、引擎守卫、完整矢量、拓扑、物理证据、候选、文字、兼容性与多页PDF十四组定向安装回归。1751候选逐值/类型/诊断/重投影普通HTML、所有权/并发/预算及后段仲裁通过，另1843数字规则/12净空/164有符号set事实在Rust核对。62模块来源审查、36份vendor哈希通过；最终wheel包含更新PSF说明及同步来源哈希。全量runner现80组，本轮未重复上一检查点79组全量及PDF像素回归。日志：build/rust-current-*.txt、build/native-table-banded-numeric-*-rust.txt/build/native-table-banded-numeric-rust.txt及build/native-table-banded-numeric-*-final.txt。

2026-10-25 完整多行稀疏：1136条旧主分支及grouped回退参考（548非空、54成功回退、16旧实际PDF/角度原语），356空间投影/114紧框间距；主失败才回退，整行组标题按原始来源/显式空白与紧框投影复核。给定原语和原始字符，不读PDF、不执行完整_evaluate或语义HTML。

验证：203项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel在独立环境完成全部79组安装包回归及18 RGBA＋18 RGB旧PDF像素精确对照。1136候选/诊断/普通HTML及356空间投影逐值/类型对照，Rust另核114紧框间距事实；所有权/并发/资源门及后段仲裁通过。61模块来源审查与36份vendor哈希通过。日志：build/rust-current-*.txt、build/native-table-multiline-rust.txt、build/native-table-projection-rust.txt、build/wheel-all-tests.txt及build/wheel-all-run-output.txt。

2026-10-25 sparse_hybrid：799条实际旧候选/诊断/普通HTML参考（571非空、16旧实际PDF/角度原语），包含物理线/矩形优先、重复文字对齐、两层表头和严格完整性发布。只给定原语，不执行完整原生引擎或PDF原语提取。

验证：200项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel在独立环境完成全部78组安装包回归及18 RGBA＋18 RGB旧PDF像素精确对照。799条hybrid参考逐值/类型/诊断/普通HTML、所有权/并发/资源门及后段仲裁通过；60模块来源审查与36份vendor哈希通过。日志：build/rust-current-*.txt、build/native-table-sparse-hybrid-rust.txt、build/wheel-all-tests.txt及build/wheel-all-run-output.txt。

验证：199项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；独立安装wheel完成三线表、数值对齐、文本三路、引擎守卫、完整矢量、拓扑、物理证据、候选、文字、兼容性、多页PDF十一组定向回归。59模块来源审查与36份vendor哈希通过。全量runner现77组，本轮未重复此前70组全量与PDF像素回归。日志：build/rust-current-*.txt及build/native-table-rule-band-*-final.txt/build/native-table-rule-band-final.txt。

2026-10-25 新rule_band：386主＋2发布门候选、386共享列/矩形/空白走廊事实、195旧median聚类。换行表头与同步描述列rowspan，100%捕获/顺序和零歧义发布不放宽；共享local_rules经数值路线回归。给定原语，未执行完整多行稀疏引擎或Rust PDF提取。

验证：196项Rust workspace测试、fmt、严格clippy、CUDA feature编译通过；独立安装wheel完成数值对齐、文本三路、引擎守卫、完整矢量、拓扑、物理证据、候选、文字、兼容性、多页PDF十组定向回归。主218条/532 regex在Rust逐项对照，wheel含补充8实际金额/状态PDF角度输入共226条（125非空、16旧PDF/角度事实）；所有权/并发/资源限制/后段仲裁通过。58模块来源审查和36份vendor哈希通过；审查范围补充后重建同一二进制wheel，数值/兼容性最终复测通过。全量runner现76组，本轮未重复此前70组全量与PDF像素回归。日志：build/rust-current-*.txt及build/native-table-aligned-numeric-*-final.txt/build/native-table-aligned-numeric-final.txt。

补充金额/状态PDF参考：另8条由两份实际PDF经旧DocVortex提取的角度输入，2条非空候选；与主218条分开冻结，独立wheel逐项通过。合计226条候选/重分行、125非空、16旧PDF/角度事实；Rust此处未执行PDF提取。

2026-10-25 新数值对齐：218旧中心重分行/金额分组/完整候选（123非空，8旧实际PDF/角度事实）、532旧regex边界。重用已有token/单元格来源/HTML；保留decimal与isdigit区别、独立括号可选、表头跨度/状态行跨度和最终来源跨格拒绝。只给定原语，未执行完整引擎或Rust PDF提取。

验证：194项Rust workspace测试、fmt、严格clippy、CUDA feature编译通过；独立安装wheel完成文本三路、引擎事实守卫、完整矢量、拓扑、物理证据、候选、文字、兼容性和多页PDF九组定向回归。375候选/351守卫逐值对照及输入所有权、并发、资源限制、后段仲裁集成通过；57模块来源审查与36份vendor哈希通过。全量wheel runner现75组，本轮未重复此前70组全量与PDF像素回归。日志：build/rust-current-*.txt、build/native-table-text-grid-*-final.txt/build/native-table-text-grid-final.txt及build/native-table-engine-policy-final.txt。

2026-10-25 新文本三路与引擎守卫：375少线/无线/key-value生成参考（8旧实际PDF/角度事实）、351输入/矢量诊断守卫逐项对照。包含续行/稠密行/表头列跨度和60%门，输入早期阶段在旧text=None响应下隔离；不宣称完整候选引擎或PDF读取。

验证：192项Rust workspace测试、fmt、严格clippy、CUDA feature编译通过；独立安装wheel完成完整矢量生成、拓扑、物理证据、候选、文字、兼容性和多页PDF七组定向回归。268条完整矢量＋204轨道阶段（含8旧实际PDF/角度原语）逐值/类型对照、完整888 isdigit属性与所有权/并发/共享预算/后段仲裁集成通过。55模块来源和36份vendor哈希通过。本轮未重复此前70组wheel全量与PDF像素回归，全量runner现73组。日志：build/rust-current-*.txt及build/native-table-vector-pipeline-*-final.txt/build/native-table-vector-pipeline-final.txt。

补充9条完整路线边界：部分续行在真纵隔断/无隔断时的旧区别、alias冲突和残留窄带拒绝，已在独立wheel逐项对照；完整矢量样本共268条，轨道阶段204条。主参考未覆写。

2026-10-25 新矢量候选生成：248旧完整路线＋204完整轨道硬门＋11补充（8旧PDF/角度事实输入），独立于此前给定规范轨道的421条；原始/支持假设、drawing/rect分路及诊断逐值/类型对照。Rust本阶段不读取PDF，888 Unicode16 isdigit属性固定，完整原生恢复仍False。

验证：188项Rust workspace测试、fmt、严格clippy、CUDA feature编译通过；独立安装wheel完成拓扑/候选、物理证据/轨道、原生候选、文字、兼容性与多页PDF六组定向回归，421条旧拓扑/候选逐值/类型对照、104精确均值及所有权/并发/资源限制/最终仲裁集成通过。54模块来源审查、36份vendor哈希通过。全量wheel runner已扩到72组，本轮未重复此前70组全量与PDF像素回归。日志：build/rust-current-*.txt及build/native-table-vector-grid-*-final.txt/build/native-table-vector-grid-final.txt。

2026-10-25 新拓扑/物化：409主参考＋12边界＋104精确均值，覆盖5%歧义、alias恢复、L形/退化拒绝、单行/列物理门、字形跨界、空行补证、文字完整性和候选HTML。参数来自给定已验证规范轨道，不读取PDF；完整矢量假设编排和恢复仍False。

验证：185项Rust workspace测试、fmt、严格clippy、CUDA feature编译全部通过；独立安装wheel完成物理证据/轨道、原生候选、原生文字、兼容性及多页PDF五组定向回归（720条新旧物理/轨道对照，另204整数set参考）。53模块来源审查和36份vendor哈希通过。本轮未重复此前70组wheel全量及像素回归；已将新套件加入全量runner（现71组）。日志：build/rust-current-*.txt、build/native-table-vector-*-final.txt及build/native-table-vector-final.txt。

追加等距边界：14条实际旧最近物理线参考表明，等距时Python整数索引set的遍历顺序会影响外缘折叠。已按CPython3.14的有界整数插入/扩容/遍历语义保留该行为，204条集合遍历参考通过；PSF许可和来源随wheel分发。旧370条轨道参考未覆写。

2026-10-25 新物理证据：336旧页面原语→片段/合并线/raw-supported轨道、370旧规范轨道参考。含旋转/halo/反向框、矩形85%门、装饰线过滤、别名冲突/双边界和唯一性；纯JSON给定原语，不读取PDF、不证明完整候选或整文档效果。

2026-10-25 新增给定网格候选：327主参考＋22额外边界覆盖逻辑格校验、可选轨道索引、逐字归属/歧义、拒绝诊断、文字/来源、顺序分和质量分；4普通HTML参考。独立wheel同时验证严格数值/类型、输入所有权、并发、资源限制和已有来源仲裁集成。给定物理网格来自调用方，尚未执行Rust物理网格发现或完整PDF原生恢复。

原生文字阶段新增362实际旧字形/行/token（16实际PDF/角度输入）、1934单元格文字/来源片段及841跨行分隔对照；Unicode16内置、复用vendor分行/细分，独立wheel数值/类型与所有权/并发/预算通过。只是给定字符/字形，不是Rust PDF几何提取或完整原生候选生成；默认路线不变。

原生表格策略检查点：547旧捷径/425候选仲裁helper对照及额外2旧五页编排参考通过；后者在wheel使用冻结旧原生结果的显式适配器，不是Rust原生候选生成实测。legacy_native_table_recovery=False；缺后端仍模型。接纳跳过表格模型、空/普通异常回模型、ImportError传播已验证。

2026-10-25接通旧UNET与默认双模型路由：5真实UNET/5 CPU OCR→UNET、10实际Paddle+Qanything/UNET/SLANet（5强制比较），最终HTML与旧版对照；6旧五页组合的JSON/Markdown/18RGB资产像素一致（版面/公式固定）。默认无空格补识别器；原生结构捷径和完整生产配置仍未迁移。

最新验证（给定网格候选阶段）：181项Rust测试、fmt、严格clippy、CUDA feature编译；70组独立安装wheel全量回归及18 RGBA＋18 RGB精确像素对照全部通过；52模块来源和36份vendor哈希通过。日志见build/rust-current-*.txt及build/wheel-all-tests.txt。
2026-10-25 最新：已接最终表格HTML归一化与`parse_pdf_models`多页预览，独立wheel对照2条旧五页组合（原生优先/强制OCR），JSON/Markdown/content_list及6份RGB资产精确；OCR/SLANet真实CPU运行，版面/公式固定，另真实V3 PDF冒烟。12936旧CJK/2270 HTML/6真实最终HTML对照通过。完整默认表格、自动方向/分类、印章/checkbox/Office仍待迁移，GPU实机未验证。

2026-10-23。本机无 GPU；按用户指示，GPU 实机测试留到项目完成后在服务器执行，不阻塞 CPU 开发和 GPU 支持代码实现。

本轮首次并行编译遇到 Windows `os error 1455`（页面文件不足）；`cargo test/clippy/check -j 1` 后通过，release wheel 使用 `CARGO_BUILD_JOBS=1` 构建。此错误不是代码测试断言失败，未调整系统页面文件设置。

2026-10-24 兼容性审查：下表为模块抽样/作者合同证据，不代表完整 Python 等价或整文档质量/性能提升。段落与原生/OCR 融合现在须 `experimental=True`，原生表格优先默认 False；具体结构偏差和局部耗时/RSS 结果见 `rust-compatibility-audit.md`。未经整文档对照的策略不得进入兼容默认路径。

最新接通模型OCR执行：真实det1→旧框整理→pipeline识别裁图→rec6→旧模型回填，131框规则、32计划/16裁图、112策略、5实际CPU链精确对照。对齐项目ocr_patch固定960和pipeline旋转门槛2，预览默认保持；det批>1/旧失败逐图重试、公式/表格与多页API待完成，不提升整文档质量/性能声明。

追加跨页公式任务执行：100组实际旧固定响应对照，仅验证任务展开/非空latex回填。execute_model_formulas调用显式PNG识别器适配器，不含PP-FormulaNet数值预处理、tokenizer解码或内置推理；空结果保留原值，结果数量不符显式拒绝，输入预检与失败原模型不变为新增合同。formula_model_inference=False。

追加公式输入预处理：88实际旧参考，RGB逐字节一致；OpenCV公开FMA与旧IPP张量绝对误差≤1e-6，独立wheel最大2.38418579e-7。Rust移植Pillow双线性/双三次/整数reduce与crop_margin/黑边/16倍补齐，SDK无Pillow/NumPy/OpenCV运行时依赖；S/M384、L768。仅输入阶段完成，tokenizer及实际网络输出尚未验证，formula_model_inference仍为False。

追加公式解码/M模型：673旧修复、1352 tokenizer用例、4额外token配置及4批旧M模型CPU参考（3张合成图，batch1/2）。新增原生推理和跨页回填路径，tokenizers0.22.2同旧版。通用ftfy尚未迁移，只支持修复后ASCII/基本汉字且无HTML实体子集，其他明确unsupported；当前formula_model_inference=True、formula_full_ftfy=False，S/L及GPU未实测，不代表完整公式或整文档等价。

最新默认ftfy已接通：29517实际旧ftfy与29517实际旧公式post_process精确对照，全1352 tokenizer用例含原18个拒绝分支通过。Unicode16 NFC/3501实体/10编码/乱码/多行HTML状态按旧默认迁移，无Python ftfy运行时依赖；新增Unicode合成图M模型2批，累计6批6图，未证明实际文档质量/S/L/GPU。formula_full_ftfy=True；只限旧默认选项/合法Unicode，128轮/递归64/共享20M工作量、64KB字段/256KB中间文本超限明确失败。

追加旧表格行级OCR/HTML回填：131框/75恢复/209策略/64旧HTML末段、精确像素及8真实CPU链。保留固定960/unclip1.6、无merge、对齐0空检测、8px薄图重试、原图裁图及旧转义/不筛分策略；分数误差限1e-5。仅已选方向的OCR与模型结果末段，原生路线/方向/原生文字优先、公式图片填充及结构模型编排待接入，legacy_table_execution=False，不提升完整表格/文档质量声明。

追加旧表格方向准备：320组旧投票/分类门槛、18组精确像素、76条实际旧固定响应方向链（含4条旋转公式遮罩）与28条实际CPU检测/方向准备链（2次真实分类模型调用）。保留原生角度优先/BGR输入/180不旋转/90及270重检测，返回最终PNG和框；不包含原生捷径、原生文字填充、结构模型编排。legacy_table_direction=True，legacy_table_execution=False。

追加旧表格原生文字优先：262坐标/262填充/10实际PDF填充与逐字符框、28整体回退策略；真实PDF→CPU链10条（2原生/8 OCR）。综合接口接方向准备→原生文字→整体为空时OCR，不补部分漏行；独立Bbox对象语义和全局bbox裁图保留，普通JSON正文原合同不改。OCR word-box/原生结构捷径/公式图片/结构模型仍待接入。

追加公式/图片模型输入：241旧提示、343实际RapidTableModel结构输入及精确像素、4真实PDF→CPU→提示链（给定公式/布局图元数据）。保留原图优先/IoU<0.9/页面旋转限制/顺序占位与包含删除/inclusive int白矩形/F32公式及旧转义；空OCR needs_ocr待后续全图OCR，不冒充结构模型已执行。

追加空OCR全图回退：243旧策略/81分数边界、8实际CPU直接det+rec及16实际模型输入回退；原图检测无预遮罩/白边重试、先0.5筛分再转义，不影响原行级OCR/既有结果。仍空时can_infer=False并跳过所有提示；三空列直接复用；只限已选方向、默认ch/non-seal，不运行结构网络。

2026-10-25 追加旧SLANet raw matcher/网络链：54几何/156 HTML逻辑/96格式→匹配、6真实plus/1m结构样本与6 CPU OCR串联（2无字跳过结构）。保留原逆IoU/distance/首项、原序拼接/粗体/完整chunk过滤/整数诊断类型/逻辑quirks；raw_html尚未做最终BeautifulSoup/CJK归一化，不连接预览安全输出器，完整表格仍False。

最终验证：165项Rust测试、fmt、严格clippy、CUDA feature编译、63组独立wheel验证、18组RGBA＋18组RGB精确像素对照通过；43模块来源检查、36份DocVortex vendor哈希通过。6真实plus/1m结构样本及6 CPU OCR串联（2无字跳过网络）的raw HTML/匹配/逻辑/诊断类型严格一致，结构坐标逐值一致；置信度差≤1e-5，最终CJK/BeautifulSoup归一化未执行，不提升完整表格质量/性能声明。

2026-10-24 用户范围调整：本期使用版面模型 original_order，enhanced/XY-cut 延期，其他任务完成后由用户明确要求再实现；缺少模型顺序仍明确 unsupported。下一步旧表格执行/回填及正式多页API，并继续扩充真实文档对照，不把延期功能计为已完成。

| 项目 | 实测结果 | 尚未证明 |
| --- | --- | --- |
| 模型OCR执行 | 131框规则、32计划/16RGB裁图、112回填策略、5实际CPU det1→rec6→模型回填，含1280/2048大图与RGB tRNS | 仅旧补丁固定960默认；det批>1、旧逐图失败重试、公式/表格/多页与GPU实机未完成 |
| 原生OCR框/检测输入 | 182旧框、54 native/检测计划、75实际旧输入像素精确；真实PDF原生/扫描页准备安装验证 | 固定空检测响应只隔离输入；det框整理、实际det/rec执行与回填待接入 |
| 旧模型布局/任务准备 | 258类别转换、86过滤、60旧计划/409 RGB裁图；3真实CPU布局精确对照和PDF准备入口 | 显式rect；auto/poly/quad、OCR/公式/表格任务执行及多页正式API未完成；UUID稳定投影 |
| PDF原图与资产集成 | 102组get_pos兼容别名旧参考/432图片像素、17完整旧资产页面；原始解码失败诊断/巨大图及257候选资源边界 | 旧未适配函数全空为已知API问题，Rust恢复提图；UUID稳定投影；保留旧Form/旋转bbox语义；正式模型路由待完成 |
| PDF原生分组/字符填充/OCR裁图 | 28完整旧PDF页面、268去重/分组边界；198旧span、32完整页、295对比度、15真实CPU串联/45输出，裁图原RGB逐字节对照 | 默认旧分组阈值、RGB PNG、无word_box；正式模型路由、PDF原图导出待完成 |
| 后置 OCR/批识别 | 73 组旧回填、11 组受支持的真实 CPU 批推理逐图张量位模式/文字对照、11链/33输出 | 1组歧义同分跨批padding明确拒绝；置信度误差≤1e-5，原生PDF裁图生产入口未接入，GPU未实测 |
| 旧跨页表格 | 313 组完整页面 JSON、864 输出，48 组实际页面→段落→合并链/144 输出 | 严格闭合非嵌套 HTML；错误 HTML 修复等 unsupported；正式 PDF/API 未集成 |
| 表内 UUID 图片 | 57 组实际旧页面/writer 与 171 组输出，稳定名称投影后比较 JSON/顺序/像素/元数据 | 稳定命名和显式错误为有意差异；不宣称旧文件名等价，不含 PDF 原图提取/跨页表格 |
| HTML 内联资产 | 9667 组旧工具 HTML/原字节/名称、12 组完整旧页面/writer 顺序 | 畸形 padding 仅 CPython 3.14.5 基线；WMF/EMF、table_fill_image_list、PDF 原图提取待完成 |
| Windows MSVC Rust | rustc/cargo 1.98.1，实际编译链接成功 | Linux/macOS 构建 |
| DocVortex core | 离线 locked 测试 12 + 8 项通过，target-dir 在本项目 build 内 | 与 RapidDoc 每个规则的完整差分 |
| pdfium-render 0.9.4 | 固定 pdfium_7350 + thread_safe；3 份文字基线、7 份几何样本通过 | 任意曲线精确裁剪、全部原生规则和渲染精度 |
| Python 扩展 | PyO3 0.28.2、abi3-py310；Windows x64 wheel 实际安装；已携带 PDFium | 其他 Python 版本实机测试、ORT runtime 打包 |
| CPU ONNX Runtime | ort rc.15/api-26，ORT 1.26.0；3 模型 × 2 batch 张量、PP-OCRv6 det/rec 和方向分类前后处理及显式页面 OCR 已通过抽样差分 | 自动方向策略、布局/公式全模型及复杂 OCR 语料 |
| OCR 图像计算 | imageproc 0.25.0 轮廓；Rust calipers/圆角扩框/mask/透视 cubic；24 组 DB、6 组裁图、3 PNG/3 PDF 页安装验证 | 任意扫描件、其他 OpenCV 版本/平台的全量等价 |
| PP-DocLayoutV3 | 三输入 FP32、有类型 FP32/int32/int64 输出；15 组前处理、15 组矩形规则、3 PNG | mask 多边形、其他模型版本、完整布局路由 |
| SLANet/plus | 7 组前处理、2 序列合同、6 PNG 结构；8 匹配合同、4 完整裁图 OCR/HTML、2 空白坏输出拒绝 | 标题/脚注归属与跨页合并；旧 HTML 字节等价 |
| UNET 原始线图 | 14 组 cubic/area 与 normalize 张量逐位一致；3 张 PNG 类别图逐项一致 | 任意输入的线图质量；恢复/路由见后续行 |
| UNET 线几何 / 多边形拓扑 | 20 组回缩/闭运算及连接线、84 组补线、48 组连通域框逐位一致；30 组源索引合并格合同 | 任意输入的结构完整性 |
| UNET 图像到格框 | 24 组 AA/36 组旋转灰度图逐像素一致、12 倾斜网格合同、5 PNG 完整格框路线 | 其他 OpenCV 后端的数值等价 |
| UNET 裁图 OCR / HTML | 59 组匹配/拆分/合并、10 组安全 HTML/边缘裁剪、6 组彩色补识别裁图与对比度、5 PNG 完整流程；空格补识别、并发及预算拒绝通过 | 原生规则融合及完整文档版面；其他语料的质量与旧 HTML 字节等价 |
| 表格分类 / 裁图路由 | Paddle Lanczos4 与 Qanything 灰度/Pillow bicubic 各 13 张量逐位一致；6 解码、6 CPU PNG、4 共识合同及单/双分类器到 UNET/两种 SLANet 路由通过；评分保留 FP32 转 Python float 的精确数值 | 批量正式 API；自定义导出模型须显式指定输出类型 |
| 表格双结构选优 | 197 组旧 selector 决策/计数、全部六种原因、4 个实际双模型比较、默认双分类器比较及阈值边界通过；两个分支共用一次检测/区域 OCR | 规范 HTML 之外的浏览器修复、更多复杂整页表格质量评估 |
| 单页模型表格 | 整页布局→5px 网格裁图→分类/选优→页面像素几何；13 RGB 裁图合同、单表/双表/空白三页（3 表）与独立裁图等价、2 PDF 页/DPI、预算/并发通过 | 自动方向、标题/脚注归属、原生表格融合、PDF point 映射、正文/跨页整合；仅 rect 模式 |
| CUDA | 待服务器实测，不是开发阻塞项 | EP 初始化与实际 GPU 推理 |
| 正文/表格页面块 | `recognize_page_png/recognize_pdf_page` 共用文字/表格裁图预算，按模型 order 稳定排列；3 个独立旧 CPU 正文参考、表格接口等价、未支持区域/同序稳定性、预算及 2 PDF 页/DPI 通过 | 完整段落重排、漏检补偿、原生融合、图表标题/脚注归属、跨页合并；未迁移旧正文全部 padding/遮盖规则 |
| 原生/OCR 正文行融合 | 6 页/7 组页与 DPI 独立 CPU OCR/PDF 字符参考通过（含 137 DPI）；8 原生行/4 次实际 OCR 识别、同一行混合扫描、裁图偏移、旋转、空白、流和累计预算通过 | 漏检补偿、任意字体/隐藏文字层质量、原生表格融合、完整段落重排、跨页整合；当前为保守水平行策略 |
| 原生表格优先 / 模型回退 | 8 页/10 组页与 DPI；4 组/6 张纯原生表无识别模型、实际不均匀轨道/合并格/字符索引/转义/像素投影、拒绝候选累计预算、同页原生/扫描表、扫描模型裁图等价和真实布局→原生表通过 | 旋转原生网格、无边线/填充条规则、漏检/重叠去重、任意字体/隐藏层、跨页整合；当前为保守描边网格选择 |
| 换行 / 块内段落 | 13,930 组旧 Python 文本连接参考精确一致；7 组几何合同、实际 OCR/原生文字/原生表格页面、源行完整保留、并发和预算通过；Unicode 16.0.0 固定表 | 完整旧段落规则、跨块阅读顺序、复杂列表/嵌套、标题脚注归属、重叠去重和跨页整合；当前为显式保守后处理 |
| 列表悬挂续行 | 19 组几何合同 × 3 比例及并发；数字/字母/括号/CJK/符号标记、双栏隔离、段界拒绝、源行索引、真实扫描 OCR 和 72/137 DPI 无识别模型的原生 PDF 通过 | 嵌套层级、字体签名/矢量圆点、齐左续行和跨块列表；保守几何策略，不宣称完整旧列表器等价 |
| 显式栏内几何行序 | 14 组作者合同 × 3 比例、源序默认兼容、乱序单/双栏与列表、歧义诊断、跨块隔离、并发及排序后累计工作预算；打乱真实 OCR/72/137 DPI 原生页面结果后恢复参考文本 | 跨块/跨页阅读顺序、通栏混排、同排片段组装与重叠去重；默认 source，仅 geometry 选项启用 |
| 显式页面块顺序 | 16 合同 × 3 比例、通栏头尾/单栏/双栏/三栏、完整块和源索引保留、整页歧义回退/预算/并发；真实扫描双栏与 72/137 DPI 原生页；720 种排列及 1000 区域通过 | 页中通栏混排、重叠仲裁、标题脚注归属、RTL/竖排、跨页顺序；默认 source，geometry 为保守几何策略 |
| 旧 para_split 内核 | 850 组未修改 Python 的完整 JSON 输出逐字段一致；含 FP64 解析边界、759 个大写边界字符、实际 CPU OCR 行、列表/目录标记、跨块/跨页、元数据、并发和资源限制 | 完整旧阅读顺序/模型/导出集成；非 JSON 对象、无效几何和非法 Unicode 不在等价合同内 |
| 旧模型顺序排序 | 1,206 组完整旧 block_sort JSON 逐字段/类型一致；308 组 sort→para_split 串联含 8 组实际 CPU OCR；虚拟行、全页行号、图表回组、中位数索引、稳定同序、并发和预算验证 | 无 original_order 的 enhanced/XY-cut、图表归属发现、完整模型/导出集成；该分支无模型顺序时明确 unsupported |
| 旧模型页面组装 | 286+18 组实际旧单页函数完整 JSON、382 组页面→段落串联；坐标/分组/过滤/横竖排行/丢弃块/模型顺序，含 8 组实际 CPU OCR；大整数等值去重和十进制舍入边界通过 | 仅 OCR 已识别、writer=None；原生字符填充、资产、后置 OCR、公式编号及跨页表格后处理、正式推理 API 未接通 |
| 旧 Markdown/content_list | 726 组完整旧输出与 raw-model→页面→段落→输出串联；默认公式分隔符、正文/标题/公式、图表与列表规则 | 171 组 Magika 算法链明确排除，运行遇到算法文本报 unsupported；资产写出、自定义公式分隔符、content_list_v2 尚未迁移 |
| 旧公式编号后处理 | 718 组完整旧后处理 JSON、382 组模型→编号→段落、639 组模型→编号→输出精确一致；邻块优先、同排候选、稳定并列、既有 tag 冲突与孤立编号；独立副本、幂等/并发/预算验证 | 不运行公式识别模型，不跨页配对；处理后的输出保守排除 169 组算法链；批组装需显式 optimize_formula_numbers=True，其他文档后处理仍未完成 |
| 旧图片裁图与 writer | 337+11 组实际旧页面/writer 的 JSON、哈希命名、顺序、解码像素与元数据一致；337 组资产页→段落及 1011 组输出；L/LA/RGB/RGBA、负数/越界裁图、原图最后匹配、目录/回调、同名覆盖、并发和失败/预算；按旧规则保留 ICC/tRNS、默认不继承 EXIF | PNG 压缩字节不承诺一致；仅 8-bit PNG、原图须显式提供；HTML 内联/表内图片、PDF 原图自动提取与正式推理 API 尚未接通 |

PDFium 来源为当前 venv 的 `pypdfium2_raw/pdfium.dll`；staging 将其复制为包内独立名称 `rapiddoc_pdfium.dll` 并附带许可。版本和 SHA256 见 packaging/pdfium-runtime-manifest.json。探针与 Python 参考实现使用独立进程，Rust 不接受 Python 句柄。

`firecrawl-pdfium` 0.1.0 的本地 crate 源码复核：sys 函数表没有 FPDFText_GetFontSize、FPDFText_GetFontWeight、FPDFPageObj_GetType、FPDFFormObj_GetObject、FPDFPageObj_GetStrokeColor。直接作为唯一 binding 需要较广扩展。因此目前探针优先使用 pdfium-render；firecrawl 的自有像素缓冲和运行库加载设计保留作参考，不引入第二套 PDFium 所有者。

原始字符和路径保留 PDFium 左下 point 坐标，新增 visual_* 字段用于旋转裁剪后的左上 point 坐标。七份样本的差分范围及未覆盖项记录在 build/pdf-geometry-report.json，不代表全部字体或任意 clip 完全等价。

验证命令：

```powershell
cargo test --workspace --locked -j 1
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -j 1 -- -D warnings
cargo build -p rapiddoc-runtime-probe --locked
.venv\Scripts\python.exe scripts/migration/check_rust_probe.py --binary target/debug/rapiddoc-runtime-probe.exe --pdfium .venv/Lib/site-packages/pypdfium2_raw/pdfium.dll
.venv\Scripts\python.exe -m maturin build --release --manifest-path bindings/python/Cargo.toml --out build/wheels
build/rust-wheel-env/Scripts/python.exe -I bindings/python/tests/smoke_install.py
```

当前 wheel 支持原生 PDF 文字、区域网格、页面渲染、多输入 FP32 OnnxSession、显式 PNG/PDF 页 OCR、原生/OCR 页面路由、布局矩形和表格裁图 OCR/安全 HTML。capabilities 按具体接口报告能力；总解析器 ocr/layout/tables/images 仍为 false，表示完整自动路由与输出尚未完成。构建时 MSVC 输出创建导入库的信息触发 linker_messages warning，但构建成功。

新增表格方向参考由旧Python环境运行`scripts/migration/export_table_direction.py`、同脚本`--portrait`与`--empty-portrait`，最后运行`scripts/migration/export_table_direction_masks.py`生成；依赖此前表格OCR/方向参考和模型文件。输出分开冻结到build/migration/table-direction*.json，已有文件拒绝覆盖，不按Rust结果改写期望。

表格原生文字参考依次运行`scripts/migration/export_table_native.py`、同脚本`--typed-detector`、`scripts/migration/export_table_native_cpu.py`；依赖此前原生页面/字符填充和表格OCR参考。初版整数mock检测参考保留，typed-detector补充真实DB返回的F32列表类型；10实际PDF填充=5页×行/字符框，10真实CPU链=5页×强制/非强制OCR，不代表10个独立文档。输出已有时拒绝覆盖。

公式/图片结构输入参考由`scripts/migration/export_table_hints.py`与`export_table_hints_cpu.py`生成，复用既有PDF页面/图像/CPU模型；后者2页×强制/非强制OCR，共4条真实文字链，公式/布局图片是给定元数据，结构模型仅捕获输入。输出build/migration/table-hints*.json已有时拒绝覆盖；PIL对象/图片字节另由宿主持有，JSON只保留uuid/元数据。

空OCR回退参考由`scripts/migration/export_table_full_ocr.py`生成，复用已有文字/旋转/空白/薄图与CPU模型；`legacy_table_models.create_full_models`配置实际TextRecognizer.__call__，避免将旧行级rec适配器当作可调用模型。输出已有拒绝覆盖；8真实CPU=4图×两种提示，16predict回退=4图×四种OCR状态，结构模型仅捕获输入。默认ch/non-seal，不含RTL/word-box。

旧SLANet matcher参考由`scripts/migration/export_legacy_table_match.py`生成到build/migration/legacy-table-match.json，已有拒绝覆盖。6结构样本=3合成图×plus/1m；6 CPU OCR串联中2无字跳过结构。坐标以F32完整提升F64序列化，6样本逐值一致，置信度旧F32均值与现F64均值差≤1e-5；raw HTML/匹配/逻辑/诊断类型严格比较，最终BeautifulSoup/CJK处理仍未执行。


## CUDA 内存优化（当前方式）

初始化时由 Rust 从原始模型在内存完成 OCR、版面、公式、SLANet、UNET 优化，随后复用会话。不生成优化 ONNX 文件，不读取旧清单/旧缓存，不需要安装或激活优化包。`python -m rapid_doc_rust.cuda status` 仅查看当前策略和随包算子库。

原模型复用 `RAPID_MODELS_DIR` 或原 Python 的模型位置。独立安装且没有可写 Python 模型目录时，下载到 `~/.cache/rapid_doc/models`；不再查找 `rapid_doc_rust` 专用模型缓存。自定义算子二进制随运行库分发，仍需 CUDA/ORT 环境。

详细设计与实测见 [Rust 内存优化](2026-11-02-native-memory-models.md)。之前的清单安装命令和磁盘包部署为已废弃历史方案。
