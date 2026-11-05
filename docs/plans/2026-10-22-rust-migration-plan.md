# RapidDoc Rust Migration Implementation Plan

2026-10-28 当前节点：原 PDF 宿主路由已接入 Rust 原生全文，`parse_pdf_models` 支持 `off/rule/rule_plus_s`、`native_only` 和内置原生区域表格恢复，贯通模型回退、OCR、公式、段落、跨页表格、Markdown、content_list 和图片资产。已完成 release 构建安装（8 分 51 秒）；本阶段安装对照、输出位置和限制统一见[宿主接入验收](2026-10-28-native-pdf-host-integration.md)。跨页 OCR 批次、BGR 渲染和边界坐标类型均按原 Python 保留，未改冻结参考或降低算法精度。`docvortex_mode` 默认仍为 `off`，原生模式显式传 `rule`；增强阅读顺序、CUDA 与同条件性能测量后置，不能据局部样本宣称全部 PDF 等价或普遍更快/更省内存。下方“宿主/Markdown 待接”为历史检查点。

2026-10-28 当前：文献区域/条目分组和前置小字号强调标题已迁移，`prepare_native_document_references` 连通原始页源→全文文字/标题→文献/紧凑标题，抵达最终页面编排前。637 文献序列、24 分组、22 文字合同、35 原始页源整链通过，含三页 PDFium 的跨页文献和附录停止正例；已统一 release 构建安装，3 组相关安装套件通过（新文献、旧标题、兼容性审计），含三页实时 PDFium、严格类型、所有权与并发。剩余最终公式/文字/行内编排、文献块重建排序、默认 PDF 和整篇验收；文献区域/分组内核不再待办。无全量/CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 当前：完整页面标题分类和全文标题原型/校准结构仲裁/未知字重重复标题已实现，`prepare_native_document_text` 连通原 `_classify_document_text` 全阶段。验证为 769 完整页、521 原型、590 全文标题、34 原始页源整链及 43 边界/4 集合轨迹（另 744 前缀）；已统一 release 构建安装到 build/rust-wheel-env，3 组相关安装套件通过（新全文/标题、旧辅助入口、兼容性审计），包含实时 PDFium、严格类型、所有权和并发。剩余文献、最终公式/文字/行内编排、默认 PDF 和整篇验收；标题推断阶段不再待办，最终收尾的标题调用归最终编排。无全量/CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 当前：全文辅助文本/正文画像节点已贯通原始页源、完整逐页准备、顺序末表继承、矢量重复装饰和全部辅助分类/双正文画像。546 组连续旧语句、11 组定向证据、34 组完整文档链通过；新入口 `prepare_native_document_auxiliary`，已统一 release 构建并安装到 build/rust-wheel-env，新文档链、旧完整页和兼容性审计 3 组安装套件通过，含实际 PDFium 新入口和并发/所有权验证。剩余明确为全文标题原型/结构/未知字重标题、文献区与条目、其余公式和最终文字/行内编排、默认 PDF 及用户整篇验收。无全量/CUDA，无性能/质量提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 最新：本轮已完整贯通逐页准备节点：同基线/二维簇/同基线合并→上/下标 sidecar→字符释放前证据冻结→PreparedPage 构建→末尾页辅助复核和末表查询；并接在已有全文几何、容器和脚注链之后。`prepare_native_document_pages(..., complete_page=True)` 可调用，旧阶段默认保持。976 组合并、499 组完整页（含 6 旧异常）、12 组身份/快照、3 组字体旧异常和实时 PDF 完整页通过。已在整个节点统一 release 构建并安装到 build/rust-wheel-env，5 组相关安装套件通过（含新入口/严格类型/并发/所有权/实时 PDF 和旧入口兼容）。剩余全文/跨页语义、其余公式恢复和最终文字编排、默认 PDF 及用户整篇效果/性能验收；不再把完整逐页准备列为待办。无全量或 CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 容器前页边保护和重复分隔线已接在全文几何之后、Form 归一化前后各自原位置；完整文档前缀 207 组与三页实时 PDFium 对照通过。继续逐页通讯脚注/标题、后置辅助文本/公式和最终默认 PDF 接入。详情见 [主链记录](2026-10-27-form-runtime-integration.md)。

2026-10-27 全文几何计划生成器及容器前文档修复前缀已完成旧版对照；不再列为后续任务。继续页边/公式/文字分类和最终默认 PDF 接入，阶段开发只跑受影响检查。详情见 [主链记录](2026-10-27-form-runtime-integration.md)。

2026-10-27 几何接续已完成计划应用、视觉重拆、样式/链接分片及与图像回收/分式恢复的连续编排；计划生成器仍待，详见 [当前主链记录](2026-10-27-form-runtime-integration.md)。继续逐分支复用旧算法，只做受影响验证。

2026-10-27 最新接续：Form 实际结构/全文角色及图像、矢量公式容器链已接通；几何后的拆行/标签回收/行首分式恢复已实现，几何修复本身及后续公式/文字分类仍待。继续按完整旧算法迁移，只跑受影响测试，主流程节点再统一打包。准确进展见 [当前主链记录](2026-10-27-form-runtime-integration.md)，下方待办均保留为历史快照。

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

新增 `expand_native_rule_table_candidate(rows, lines, rule_bboxes, core_row_indices, *, page_size, median_height, angle=0, caption_source_index=None, has_possible_table_notes=True)`，把已完成核心/表题/自动表注组装为正式候选：完整local并集、注释原坐标、体/注释成员分离、bbox/core_bbox方向还原及score0.0默认字段，全部子阶段共享Work。155旧完整普通函数参考（139带注释）与20页实际PDF查询（7带注释）通过Rust验证。本次消掉“正式候选组装与方向还原”；下一项连通网格扩框，自动规则候选产生及页面编排仍待。

新增 `collect_native_table_footnotes(rows, lines, rule_bbox, core_line_indices, *, page_size, median_height, angle=0)`，完整接通自动表注收集：明确Note/Source/注、辅助Unicode短标记及表体紧凑token/小且上移字形引用、正文字号参照和连续行字体/字号/缩进/距离限制。返回独立footnote_rows及有序annotation，不再要求提供表注行索引。177旧参考/75标记事实与20页实时PDF查询通过Rust验证，包含实际辅助与上标引用正例。本次消掉“自动表注行收集”；下一项把核心、表题和表注组装为正式表格候选并还原方向坐标，页面级候选自动产生仍待。

新增 `prepare_native_rule_table_annotations(rows, rule_bboxes, core_row_indices, *, median_height, caption_source_index=None, footnote_row_indices=())`，完成临时规则表格的核心边界/成员、表题续行收集和注释记录构造。保留原坐标与local_bbox区别、横向走廊裁剪、表体成员排除、重复来源框合并和注释字典顺序，输入不修改。136旧参考与12页PDF实时查询通过Rust验证。本次消掉核心/表题/注释记录准备；表注行目前显式传入，下一项是自动表注行收集，完整临时候选产生/转正式仍待。

新增 `exclude_native_prose_spanning_rule_drafts(source, candidates)`，完成跨正文规则草稿排除：下方网格尺寸/距离、贯穿竖轨和上下横边界，加上草稿成员中至少三排符合词法/字重/语义门槛的长正文，才拒绝草稿；caption草稿与非0方向保留。返回独立原序候选和拒绝原数组位置。151旧参考（112拒绝）与20页PDF实时查询（2拒绝）通过Rust验证。本次消掉“跨正文规则草稿排除”；草稿产生/惰性物化仍待，下一项是物化所需的规则边界、成员和注释准备。

新增 `merge_native_table_candidates(candidates)`，完成已物化候选合并：分数稳定逆序、同角度首个20%重叠目标、100分强网格拒绝弱候选扩张、动态bbox/local/core并集，以及来源合并、同类注释归并和表体成员优先的注释冲突消解。返回独立candidates，保留目标其他元数据；Python注释行框字典经有序键框对跨Rust桥接，返回整数键字典，原输入不修改。158旧对照（142变化）及28页PDF/44实时检测后查询（24变化）通过Rust验证。本次消掉的是“已物化候选合并”；下一子任务是跨正文规则草稿排除，惰性草稿物化和完整候选产生器仍待。

新增 `prefer_complete_native_table_replacements(base, recovered, replacements)`，完整迁移旧替换仲裁：替代片段与旧核心至少90%较小框交叠，按交集并面积验证共同覆盖，重叠片段不重复计数；先累计所有旧表的拒绝结论，再用最终通过的替代候选删除旧base/recovered。返回原序独立三组候选，保留全部注释/来源/其他元数据，不修改输入。157旧完整参考（153变化/78拒绝）和28页实际PDF/44组实时填充检测后重建查询（24非空变化）通过Rust验证；查询中的半表拆分是测试构造，不是完整年份/数值表候选产生器。一般候选编排、diagram和默认PDF仍待。

新增 `inspect_native_closed_grids(source, candidates, *, allow_two_by_two=False)`，按原候选顺序返回 `closed_grid_flags`。完整迁移旧候选角度/core回退、较大Form排除、原source ID成员、正行高中位数、贯穿横纵轨道、两位小数去重/端点容差和文字占用行列；缺少正行高且实际需要求中位数时传播输入错误。176旧参考（117正例）、24舍入事实与16页实际PDF/32组实时重建查询（7正例）通过定向Rust验证。候选是显式上游输入，真实Form提取未包含；一般表格候选、diagram组件和默认PDF仍待。

新增 `detect_native_filled_grid_tables(path_infos, *, page_size, excluded_bboxes=())`，完整迁移旧外框填充行带表格候选：页面比例、最大单元格与半行底纹排除、水平行带数量/覆盖/间隙、垂直覆盖与空带、稳定选优及重叠去重。返回独立完整 `candidates`，保留原bbox数值类型，输入不修改；该函数不依赖CJK字框，也不检查rectangle_bboxes数量。165旧完整参考（101正例）、6整数/混合float算术事实及28页实际PDF实时路径链（4正例）通过定向Rust验证。旋转后只有两条水平带的样本沿旧拒绝；一般表格/闭合网格证明、diagram组件及默认PDF编排仍待。

新增 `detect_native_bar_graphics(source, *, em)`，完成旧原生柱图候选产生器：真实表框/柱组、图题保护及行语义修改、两轮接触柱体/复杂轮廓/栅格/标签扩框，最后执行实际有字网格排除。返回独立 `source` 与原序 `bar_bboxes`，所有子阶段共用累计Work，原输入不修改。另提供 `protect_native_graphic_captions(lines, candidate_bboxes, *, median_height)`，返回独立 `lines` 和 `protected_source_indices`；仅沿旧英文fig/figure数字种子启动。122完整柱图参考、151图题参考及40页实际PDF链已通过定向Rust验证；完成该产生器不等于完整默认PDF编排完成。

新增 `prepare_native_bar_groups(source, *, em)`，在同一累计Work中自动接入相接填充表格排除与柱体准备，再执行共同零轴选择、稳定分类方向排序/切簇、位置和长度差/真实间隔证明、数字行几何准入及零轴补边。返回 `table_bboxes`、`admitted_bars` 和 `groups`；每组含原准入数组 `bar_indices`、原行数组 `numeric_line_indices`、`horizontal`、`baseline`、`core_bbox`、补边后的 `bbox` 和 `label_anchor`。em必须正且有限，输入不修改；此接口尚不执行图题保护、两轮图体扩展及最终有字网格排除，不能称完整柱图检测。187旧参考（152组正例）、30旧round事实和20页实际PDF链（6正例页/10柱组/4表格排除）已通过定向Rust验证。

新增 `detect_native_tiled_tables(source, *, excluded_bboxes=())`，完整迁移旧相接填充单元格候选阶段：真实CJK宽松/紧致框中位数校准、相接矩形组件、行列轨证明、无底纹表头推断网格，以及相邻表格题名转移。返回独立完整 `candidates`，其 `bbox` 可传给柱图准备的 `table_bboxes`；不修改 source。168旧参考（140正例/2题名转移）、12页实际PDF链（4表格及柱体排除正例/1推断网格正例）通过定向Rust验证。测试字体为自行生成的矩形字形，不代表客户文档质量证明；完整其他填充网格族、柱组/图体扩展及默认PDF仍待。

2026-10-26 增量：原生柱图共同基线、柱体准入/方向及零轴补边完成定向 Rust 对照及一次 wheel/四组安装回归，170 旧参考与 8 页实时 PDF 链。上游 table_bboxes 必填；接下来迁移完整填充表格产生器、柱组聚类与标签/图体扩展，再接强图形入口和默认 PDF。

2026-10-26 增量：栅格标签连接图及普通文字框阶段接通，复用绘图组件/真实网格保护。144旧参考及20页实际PDF全实时链，旧旋转支持域保持；完整native diagram/bar产生器、跨页准备和默认PDF编排仍待。

2026-10-26 增量：编号图题、百分比及同心轮廓/引线候选接通，196旧对照和8页真实PDF到网格仲裁链。沿旧英文图题和旋转支持域，完整强图形候选集合、跨页准备和默认PDF编排继续待办。

2026-10-26 增量：图形候选共用网格保护和末尾仲裁接通，迁移有字物理列/行带、Form排除和70%并集覆盖；178旧参考、8实际PDF候选生成到仲裁链。完整候选产生器集合、跨页页源准备和默认PDF编排继续待办。

2026-10-26 增量：独立无文字矢量图体及小栅格判定接通，229旧对照、41既有实际PDF输入和8页专用实时链。路径原序增量合并、字号/覆盖/颜色例外沿旧规则，完整强图形/表格仲裁、跨页准备与默认PDF编排继续待办。

2026-10-26 增量：旧绘图线组件连接、统计和复杂路径图形候选接通，239旧参考、41既有实际PDF输入及8页专用实时链。原组件顺序/类型/距离舍入保持；完整强图形/表格仲裁、跨页准备和默认PDF编排继续待办。

2026-10-26 增量：二维复杂路径、填充容器/相交坐标轴候选迁移，263旧事实、41既有实际PDF输入及8页专用四方向实时链。原候选顺序/重复及union类型保留，完整强图形/表格仲裁、跨页准备和默认PDF编排继续待办。

2026-10-26 增量：补齐旧CJK图形尺度校准与堆叠柱候选阶段，304旧对照，41实际PDF输入负例链；正例为独立合成页源。完整强图形/跨页页源和候选检测/默认PDF流程仍待，未作整文档质量/性能结论。

2026-10-26 增量：栅格图刻度/图体证明及说明续行归组接通，272旧参考及16实际PDF读取→局部源/图像过滤/噪声→归组链。沿旧顺序/面积平局/分组覆盖，不改文字或几何。完整强图形检测、跨页页源准备、候选检测和默认PDF编排继续待办。

2026-10-26 增量：原生浅灰数字噪声接通旧几何候选、实际PDF按需纯填色RGBA和完整来源号/字符清理。236旧决策与32页真实PDF链，支持无候选不访问PDF。强图形检测/栅格图注归组、完整跨页准备/候选检测/默认PDF编排仍待。

2026-10-26 增量：新增PDF图片证据自动读取（原始流SHA-256、Form/裁剪/旋转图片框、平滑背景/白边），85旧页参考及10页实时读取到图片裁决链。原Pillow RGB重采样提至core复用且保持公式算术，相关公式安装测试纳入验证。噪声/完整页源和跨页准备/候选检测/默认PDF仍待。

2026-10-26 增量：迁移旧图片区域裁决（全文水印、正文背景过滤、标题白边裁剪），新增阶段接口 `prepare_native_image_regions` 和328组旧参考/64组实际PDF页源用例。像素证据仍为显式输入；PDF图片读取、噪声、完整页源及候选检测/默认集成继续待办。验证范围见 `2026-10-26-native-image-regions.md`。

脚本证据生成和输出阶段验证完成：249完整输出/诊断、516 sidecar、57实时PDF负例链及四方向专用PDF/7处实际脚本通过；五项受影响Rust测试和五组安装套件通过。fmt/82模块审查/固定36＋3＋1哈希保持，本阶段一次wheel构建，无全workspace/全部100组runner/未受影响渲染/全量clippy/CUDA。新script_lines投影已完成，后续重点转到像素字重与完整页源/图片噪声/跨页规则、候选检测和默认PDF流程集成。

2026-10-26 新script_lines证据生成、最终Span投影与物化诊断已接通；249旧完整输出/诊断、516旧payload事实，原44预分类分式参考独立保留。57实际PDF负例链，另四方向专用PDF真实读取产生7处脚本；五项受影响Rust测试通过。新增第82审查模块，原apply默认不变；接下来继续完整页源/像素字重/图片噪声/跨页规则与候选检测/默认流程集成，按定向/阶段相关验证执行，无CUDA。

样式/链接Span输出阶段验证完成：297旧完整输出、57实时PDF字符＋注解链、五组安装套件及两个定向Rust测试通过，fmt/81模块审查与固定36＋3＋1哈希保持。本阶段一次wheel构建，无全workspace/全部99组runner/未受影响渲染/全量clippy/CUDA；继续新script_lines投影、像素字重和完整页源/候选检测及默认流程集成。

2026-10-26 样式/链接证据到InlineSpan输出已实现，297独立旧完整输出含57实际PDF链；新第81审查模块。接通几何/文本投影与回退、冲突合并及文字/公式/链接Span，已有平坦脚本标记可物化，新script_lines投影仍待。块为明确阶段输入，后续继续脚本/像素字重、完整页源和候选检测及默认流程集成；仅定向/阶段相关验证，不跑CUDA。

PDF注解读取阶段验证完成：两个定向Rust测试、207旧URI/36实际页26注解、57实时字符＋注解链及五组相关安装套件通过；fmt/80模块审查/固定36＋3＋1哈希保持。阶段末一次wheel构建，未跑全workspace/全部98组runner/未受影响渲染/全量clippy/CUDA。接下来像素字重/证据输出投影及页源其他准备、候选检测与默认PDF集成。

2026-10-26 PDF外链注解自动读取已实现，207 URI与36实际PDF页/26有效注解的旧参考定向Rust对照通过；新增第80审查模块。正在验证字符和注解均实时读取的57条视觉行/页源局部准备/inline检测链，取消该链的注解注入。像素字重补证、证据最终输出投影、完整页源/候选检测和默认集成继续后续，按定向/阶段相关回归执行，无CUDA。

原生样式/链接阶段验证完成：285旧事实、57实时PDF字符链、五组相关安装套件及两个定向Rust测试通过，fmt/79模块审查和固定36＋3＋1哈希保持。修正已复现的多字符标记差异后最终wheel隔离安装通过；未跑全workspace/全部97组runner/未受影响渲染/全量clippy/CUDA。继续SDK注解读取、像素字重和证据输出投影及完整页源/候选检测。

2026-10-26 原生字体/绘图样式及链接证据检测已接通，285旧完整事实含57实际PDF字符链。复用原样固定内核，修正多字符项目符号规范化差异；SDK实时注解读取、像素字重补证和输出投影仍待。新增第79模块来源审查，按定向测试→阶段构建/相关回归执行，发现实际差异后必要重建；无全量/CUDA。

页源局部准备阶段验证完成：290旧完整事实、57实时PDF字符→视觉行→局部准备链及五组相关安装套件通过；两个定向Rust测试/fmt/78模块来源审查和固定36＋3＋1哈希通过。阶段末只构建一次wheel，未跑全workspace/全部96组runner/未受影响渲染/全量clippy/CUDA。接下来继续完整页源其他准备与候选检测。

2026-10-26 页源局部准备已接通首字母正文、悬挂字母段、薄栅格图和私用区装饰文字转线。290独立旧参考含57实际PDF视觉行链；新增第78模块来源审查。完整样式/链接/图片水印/噪声/跨页准备和表格候选检测继续后续；开发只定向测试，阶段末统一构建相关回归，不运行CUDA检查。

视觉行阶段验证完成：317旧事实/977行、57实时PDF读取链和六组相关安装套件通过；定向Rust/fmt/77模块审查及固定36＋3＋1哈希通过。阶段结束只构建一次wheel，未跑全workspace/全95组runner/未受影响渲染/全量clippy/CUDA，遵循新的验证策略。下一步仍为完整页源其他准备与候选检测。

2026-10-26 继续页源基础链：原生字符→视觉行已实现，317独立旧事实/977行包含57实际PDF字符输入，新增第77审查模块。接下来接页源其他准备和完整表格候选检测。验证策略按用户要求调整：开发只测受影响部分，阶段完成再一次构建和相关回归；暂停CUDA编译和GPU验证，重要集成节点才做全量。

媒体PNG及相关回归共11组安装套件通过，另36组PDF像素对照通过；未全跑94组runner。日志build/native-table-media-png-regression.txt，下一步完整页源/候选检测。

媒体PNG验证完成：232项Rust测试、fmt/严格clippy/CUDA feature编译、76模块审查与固定36＋3＋1哈希通过。release wheel已安装，182 PNG页/502引用和45真实PDF/125引用全部对照通过，通用PDF渲染36组旧像素重新核对一致。完整页源/候选检测及默认集成仍待，GPU实机/enhanced依既定安排延期。

2026-10-26 媒体PNG阶段：两个新入口接通调用方PNG和按原页号的PDF渲染、floor/ceil裁剪、保模式旋转/元数据及部分页失败文字回退；point框按旧千分刻度转换。182独立PNG页/502引用与45真实PDF/链125引用冻结，76模块审查/94安装runner。PNG压缩字节、URI和URI哈希名为明确差异，像素和其他结构按旧参考核对。完整页源/候选检测与默认集成接下来实现；GPU/enhanced及完成开关依旧保持。

完整块事务检查点已完成验证：231项Rust测试、fmt/clippy/CUDA feature编译、75模块来源审查及固定36＋3＋1哈希；最终wheel通过781旧整流程＋八候选缓存、16实时PDF串联及共享预算/异常恢复，12组相关安装套件通过。96独立媒体事实由Rust核对，wheel覆盖24媒体事务。93套件/36像素组本轮未全跑，GPU未实机。下一步为实际PNG编码与页级自动接入，默认开关保持False。

2026-10-26 完整块计算已串联：materialize_native_table_blocks接通原生恢复、语义HTML、真实注释、弱表撤销、原字符投影、媒体归属占位与原子发布/行认领。781旧完整事务、八候选缓存、96媒体事实和16实时PDF原语链独立冻结，75审查模块/93安装runner；阶段输入不等同完整候选/行/媒体检测。页级校准缓存、新阶段共享预算与资源错误传播保留；PNG实际编码和页级检测/默认集成接下来实现，GPU/enhanced等延期不变。

注释检查点10组相关安装回归已通过，见build/native-table-annotations-regression.txt。全量92套件和36像素组未在本轮全跑。继续投影降级、媒体归属及最终原子提交。

注释检查点验证完成：229项Rust测试、fmt/clippy/CUDA feature编译、74模块来源审查与固定36＋3＋1哈希通过；最终wheel已验证928注释/类型、240首页面板、16实时PDF负例及所有权、并发、累计预算和异常恢复。下一步仍为投影降级、媒体归属及最终原子提交。

2026-10-26 注释物化与首页弱表撤销已实现：prepare_native_table_annotations自动计算注释块/外置/失败集合，使用同一准备预算回并边界和扣除成员；restore_native_front_matter_panel是仅供恢复失败后调用的独立入口，保留原元数据/正文/真网格保护和标题副作用。904注释＋24类型、240面板/102撤销及16实时PDF负例独立冻结；NFKC/casefold固定Unicode16。审查74模块、runner92组。接下来投影降级、媒体归属和原子提交；默认完成标志及既定延期不变。

块级准备检查点的9组相关安装回归全部通过，见build/native-table-block-regression.txt。全量runner91组及36像素对照本轮未全跑。接续任务为注释块物化与首页弱表撤销。

块级准备验证通过：228项Rust测试、fmt/clippy/CUDA feature编译、73模块来源审查及固定36＋3＋1哈希；已安装wheel的624准备/类型事实、16实时PDF/角度串联及累计预算、限额和异常恢复通过。非空PDF行实时从SDK字符生成glyph视觉行，4空行阶段输入为负例。仍保留完整注释输出/弱表撤销/投影降级/图片归属与原子提交待办。

2026-10-26 块级准备：新增 prepare_native_table_block，移植汉字框校准、嵌入脚注外置、同基线/旋转成员扩展和失败注释表体回并。612独立旧准备＋12类型事实、16实际PDF/角度准备→恢复/HTML链，12非空。source行由已有glyph视觉行显式投影，不宣称完整行检测。failed_annotations仍是后续阶段输入，准备结果不提交认领；注释块物化、首页弱表撤销、投影降级、图片归属和原子提交接下来实现。73模块来源审查/91组安装runner，默认开关与延期项保持。

最新验证：227项Rust workspace测试、fmt、严格clippy、CUDA feature编译和13组相关安装包回归通过；72模块来源审查及36＋3＋1固定源码哈希通过。新整表套件通过526整表、724恢复、32实时PDF组合及累计预算/错误边界检查。90组全量runner和36像素对照本轮未全跑；GPU实机仍待服务器。

2026-10-26 整表语义 HTML / 结果恢复已接通：新增 `render_native_table_html_with_scripts` 和 `recover_native_table_result`（None 或 `(content, result)`）。沿用旧恢复→准备→二维→角色→叶子→整表流程、原HTML回退、raw result.html及异常阶段边界；共享外层Work预算并保留各内核限制。独立冻结526整表、724原始输入恢复、32实际PDF/旋转/样式组合；168自动上下标正例为中立结果阶段事实，不据此承诺整文档改善。接下来迁移 PDF 块级物化/字符认领/注释图片/跨表修复和默认回退；owned批处理、visible-only、enhanced阅读顺序及GPU实机依既定安排延期。两个legacy完成开关仍False；下方为历史进展。

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


2026-10-25 当前：原始字符/线/矩形→完整旧引擎、NativeTableResult及全路线诊断已接通；724完整参考（547恢复、六类来源、9物理冲突、16旧实际PDF/角度完整原语）逐项对照。候选身份、aligned撤销/表头覆盖、列下界/漏行/alias、有限回退顺序保持；整个调用含诊断重放共用Work实际1000万预算。下一步Rust PDF原语提取、适配器语义HTML及PDF级接入，legacy_native_table_recovery仍False。

2026-10-25 已完成：banded_numeric分带数值候选；1751旧候选/诊断/重投影HTML（262非空、16旧PDF/角度原语）、1843数字规则/12共同净空/164有符号来源set及3异常参考；保留众数首遇平局、三层分组表头及95/96顺序旧正例。

2026-10-25 已完成：完整sparse_multiline主分支及grouped_rule_band回退；1136旧参考（548非空、54成功回退）、356空间投影/114紧框间距事实，16旧实际PDF/角度为给定原语。保留来源/邻接显式空白和严格文本一致性。

2026-10-25 已完成：sparse_hybrid物理线/矩形与文字对齐融合候选；799旧候选/诊断/普通HTML参考（571非空、16旧实际PDF/角度原语），保留物理假设优先、两层表头和完整性门。

2026-10-25 已完成：rule_band三线表候选和共享稀疏列/空白走廊证据已移植；386＋2候选、386共享事实（16旧PDF/角度）、195 median聚类逐项对照。数值路线_local_rules抽为共享函数，不改原逻辑。

2026-10-25 当前：aligned_numeric重分行与实际候选执行链已移植，218旧参考（123非空/8实际PDF角度事实）及532 regex边界；保留金额右缘/表头净空和行列跨度、来源完整性及旧括号quirk。尚不含未调用numeric_region_bounds，不读PDF/不执行完整_evaluate。下一步rule_band/sparse_hybrid/multiline/banded_numeric及完整引擎控制；完整恢复仍False。

2026-10-25 当前：旧text_grid.py少线/无线/key-value三路已接通，375条旧参考（8实际PDF/角度事实输入）；引擎输入早期门/已有诊断守卫351条参考。后者不是完整_evaluate，首版2条可变诊断错配保留，独立stable快照重建。下一步aligned_numeric/regroup，再接rule_band/sparse_hybrid/multiline/banded_numeric与完整引擎；原生恢复仍False。

2026-10-25 当前：给定页面原语→完整矢量候选已串联（轨道完整硬门/halo/矩形晶格/稠密基线/装饰竖线及raw-supported/line-rect编排）。248完整生成＋204轨道阶段、11补充（8旧实际PDF/角度原语）逐项对照，Unicode16完整888 isdigit码点核对；整次调用共享预算。下一步为完整引擎早期仲裁及其他文本/稀疏候选，再接Rust DocVortex原语提取/语义HTML。完整原生恢复仍False。

2026-10-25 当前：已接规范轨道→缺边连通/矩形拓扑→单行/列与原子行物理证据→文字落格/评分/普通HTML和发布门。409＋12实际旧参考、104精确均值对照。下一步串_build_vector_tracks的完整拒绝门和多假设选择，之后接完整PDF恢复；仍是给定规范轨道输入，不是自动PDF矢量候选生成。

追加等距边界：14条实际旧最近物理线参考表明，等距时Python整数索引set的遍历顺序会影响外缘折叠。已按CPython3.14的有界整数插入/扩容/遍历语义保留该行为，204条集合遍历参考通过；PSF许可和来源随wheel分发。旧370条轨道参考未覆写。

2026-10-25 当前：物理线段/矩形准备、raw/supported轨道推断和规范化已移植；336＋370实际旧函数参考对照，原裁剪/旋转/halo/均值吸附/断线连接、装饰短横过滤、别名冲突和独立双边界保留。仅给定页面原语和轨道，完整矢量拓扑及物理证据评分、多假设候选生成仍未接通；下一步接这部分，再接完整恢复与Rust PDF原语提取。最新范围见rust-migration-status.md顶部。

2026-10-25 当前接续：已移植candidate.py给定网格校验、轨道索引、字形归属、单元格文字/来源、拒绝诊断、顺序一致性与质量评分，以及普通HTML。327主参考＋22边界参考及4 HTML逐项对照；结构质量由上游显式提供，最终来源门槛沿用已有仲裁，不在构建器提前代替。下一步移植旧向量线/矩形/文本的物理网格发现和多候选生成，之后接完整_evaluate、语义上下标HTML和Rust DocVortex字符几何；legacy_native_table_recovery仍False。以下为历史检查点，当前范围以本段及rust-migration-status.md顶部为准。

> **For agents:** 按任务逐项实施；可使用本机 `executing-plans` 技能。先阅读同目录 `2026-10-22-rust-feasibility.md`。本文件是分阶段执行计划；P00 已启动，实际进度与验证证据见 rust-migration-status.md。

更新日期：2026-10-25。

最新：旧原生字形/视觉行/token与单元格文字/来源片段已移植，362文字构建（16真实PDF字符几何/角度）、1934单元格及841跨行分隔对照。复用vendor分行/细分内核，保留advance框、标点分行补偿、Unicode16分类及URL/单位/连字符旧规则。下一步为candidate.py网格验证、字符落格与评分；完整物理候选生成、Rust DocVortex字符几何提取和HTML物化未完成，legacy_native_table_recovery仍False。

最新检查点：原生表格捷径守卫/接纳和已有候选helper仲裁已移植，547旧策略/425候选对照。显式恢复器适配点已接多页入口，以冻结旧PDF恢复结果验证接纳跳过模型、None/错误回模型、ImportError传播。**候选生成、完整_evaluate及HTML物化未移植，legacy_native_table_recovery=False**；下一步从原生字符归一化/分行和物理候选生成继续，不把实验网格作为默认实现。

最新：旧UNET与双模型分类/比较已接到多页入口。28旧原HTML、5真实UNET及5实际OCR→UNET、237原HTML选优/12阈值调用顺序、10实际CPU双模型路由（5强制比较）；三种表格配置×两种文字模式共6旧五页组合，JSON/Markdown/18份资产像素精确对照。原生表格结构捷径、可选空格补识别/失败重试与其他生产配置继续接入，完整legacy_table_execution仍False。

2026-10-25 新检查点：最终CJK/HTML归一化及`parse_pdf_models`多页预览接通；2条实际旧五页组合的JSON/Markdown/content_list与资产像素精确对照，正文OCR/SLANet真实CPU运行，版面/公式响应固定，另实际V3 PDF冒烟。仅V3 rect+SLANet-plus/1m显式配置，默认UNET/双模型、原生表格结构、自动方向/分类、印章/checkbox和Office尚未覆盖；不能替换生产Python。下一步优先完成默认表格路由，详见rust-migration-status.md顶部。

2026-10-25 接续：旧SLANet matcher与raw网络链已接通，54几何/156 HTML逻辑/96格式匹配及6真实plus/1m结构样本、6 CPU OCR串联（2无字跳过结构）。旧匹配/文本拼接/逻辑quirks保留；新增legacy结构坐标完整F32→F64序列化，原预览合同不改。尚缺最终CJK/BeautifulSoup归一化、UNET旧matcher、分类/双模型仲裁及完整API；legacy_table_execution=False。详见rust-migration-status.md顶部。

2026-10-24 已完成空OCR全图回退：接通提示输入，243旧策略/81阈值、8真实CPU直接-rec及16实际predict输入；保留原图无预遮罩/白边重试、默认0.5筛分、三空列不再识别、仍空跳过图片/公式。已选方向/default ch限定。

2026-10-24 已完成公式/图片提示与结构模型输入：241提示、343实际predict输入/精确像素及4真实PDF/CPU提示链；保留原图优先/IoU/页面旋转限制、顺序占位与包含删除、inclusive int白矩形、F32公式和旧转义。低级payload的空OCR needs_ocr待处理合同保留，新增综合入口可自动完成回退。

2026-10-24 已完成原生文字优先：方向准备和行级OCR回退已接通；262几何/262填充/10实际PDF填充（含字符框）、28旧回退策略、10实际PDF→CPU链（2原生/8 OCR）。部分原生结果不补OCR，保留旧全局bbox裁图与独立Bbox对象语义。

2026-10-24 已完成方向准备：320投票/门槛、18精确像素、76固定响应流程和28真实CPU检测/方向流程（2次实际方向网络调用）；保留原生角度优先、BGR输入与180不转/90及270重检测。先前行级OCR/HTML末段的8真实CPU检测识别链另列，未混同方向链证据。

2026-10-24 接续：公式所用默认ftfy6.3.1已迁移到Rust，29517旧ftfy及29517旧完整post_process精确对照，原18组解码拒绝分支已全部通过；另2批Unicode M图参考，累计6批6张合成图仍不能代表实际文档质量。下一步表格执行/回填及正式多页API，并扩充真实文档对照；详见rust-migration-status.md顶部。未实现的延期功能仍保持原范围。

2026-10-24 接续：公式token解码/旧LaTeX修复及M模型CPU推理已接通，原生会话可供跨页公式任务使用。通用ftfy仍待移植，修复后非ASCII/基本汉字或HTML实体明确unsupported，不能计为完整公式兼容。下一步补齐ftfy与真实公式文档对照，再接表格及正式多页API；具体证据见rust-migration-status.md顶部。

2026-10-24 接续：PP-FormulaNet Plus输入预处理已移植，88实际旧参考RGB逐字节一致；张量与旧IPP存在≤1e-6的FMA舍入差异，不能宣称网络输出等价。下一步tokenizer/LaTeX/ftfy解码与真实公式ONNX推理，再接表格和正式多页API；详见rust-migration-status.md顶部。

2026-10-24 接续：跨页公式任务执行/非空结果回填已接通，100组实际旧固定响应验证策略。内置PP-FormulaNet预处理/tokenizer解码/实际ONNX推理仍待实现，之后接表格执行与正式多页API；不能将适配器合同计为公式网络完成。详细证据见rust-migration-status.md顶部。enhanced/XY-cut及其他语言SDK继续延期。

**Goal:** 构建不依赖 Python/PyTorch 的 Rust 解析核心，本期仅交付 CPU/可选 CUDA 的 Python SDK，供用户测试。其他语言和通用 C ABI 等用户测试确认并明确要求后再实施。

**Architecture:** Rust workspace 内分离格式抽取、规则、ONNX 推理、文档 IR 和输出。保留 PDFium/ORT 原生运行库，通过兼容 exporter 和小步路由切换迁移当前 Python API；不覆盖本地 DocVortex 修复。

**Tech Stack:** Rust、serde、PDFium（firecrawl-pdfium 与 pdfium-render 在 P01 对照选型）、ort、zip/quick-xml/cfb、image/imageproc、PyO3/maturin；优先评估本地 DocVortex Rust 内核。候选版本须经 P01 实测统一锁定。

---

## 执行规则与交付定义

- 2026-10-24 用户明确要求以现有 Python 解析质量为基准。先完成兼容性审查并按旧调用链逐项移植；新算法只能作为实验候选，作者合同不能替代旧输出差分，不能凭 Rust 或内核微基准宣称整文档质量/速度/内存更优。审查及准入条件见 `rust-compatibility-audit.md`。段落/融合实验须显式开启，原生表格优先默认关闭。

- 当前基线 `d0e98fb`。开始执行时记录新的 HEAD、工作区差异和模型哈希，保护用户已有改动。当前 `todo.md` 有用户的暂存/未暂存变更，不重写它。
- 本期不实现 Node.js/Java/Go、通用 C ABI 或相关 CI；仅保持 Rust 核心无 Python 依赖。不把其他 SDK 作为 Python 发布的前置条件。
- P00 只建立基线工具与语料，不修改生产解析器。后续开发在 `codex/rust-migration-*` 分支按阶段提交；不要把整个计划当作一次改写任务。
- 每个任务按“最小回归用例 → 确认能抓住缺失行为 → 实现 → 对比 → 提交”循环，复杂任务按子项拆成独立 PR。先处理一个格式/规则，不并行改同一套 IR。
- 下文新增路径是规划，不应被误读为已有文件；Python 源路径是迁移依据。每个 crate 同时创建其 `Cargo.toml` 和 `src/lib.rs`，可执行 crate 创建 `src/main.rs`。
- 所有源路径以仓库根目录为基准；命令也在根目录执行。Windows 使用 `.venv\Scripts\python.exe`，其他平台替换为环境内 Python。不依赖可能被忽略的 `tools/`，新开发脚本统一放 `scripts/`。
- 每个公开方法及非平凡内部方法写中文文档注释，阐明单位、前提、错误、回退与资源释放；跨语言 API 参数名使用一致英文。
- 不以 `unwrap()`、吞异常、空结果或全局 skip 掩盖输入/依赖错误。明确区分“不支持、加密、损坏、资源超限、模型缺失、设备不可用”。
- 每个阶段记录：实现范围、测试命令/结果、实际 EP/库版本、未覆盖样本、基准、下一任务。CPU 成功不算 GPU 验收，编译成功不算安装成功。

## 阶段依赖与发布顺序

```text
P00 基线/语料 → P01 技术探针 → P02 IR/错误/坐标
P02 → P03 PDF事实抽取 → P04 原生规则 → P05 输出/CLI → P06 Python CPU预览
P02 → P07 ORT/缓存 → P08 OCR → P09 布局/表格/公式 → P10 CUDA与图优化
P02/P05 → P11 OOXML → P12 OLE旧格式 → P13 公式/矢量图完整化
P06/P09/P11/P12/P13 → P15 Python发布与用户测试
P15 + 用户测试确认 + 用户明确要求 → P14（延期）其他SDK规划与实现
```

依赖图仅表示先决条件，不要求自动创建多个 agent。P10 可在 CPU 版发布后继续；完整六格式承诺须等 P11–P13 完成。

## P00：恢复可复现基线与范围清单

**新增：**

- `scripts/migration/inventory.py`
- `scripts/migration/export_baseline.py`
- `scripts/migration/compare_results.py`
- `scripts/bench/parse_bench.py`
- `tests/migration/corpus_manifest.json`
- `tests/migration/golden/`（每个样本独立版本化 JSON）
- `docs/plans/rust-migration-status.md`

**依据：** `rapid_doc/main.py`、`pyproject.toml`、`rapid_doc/_vendor/docvortex_core/UPSTREAM.md`、`tests/native_pdf/`、`tests/native_office/`、`tests/test_office_native_integration.py`。

步骤：

1. 列出公开 API/配置、MakeMode、输出字段、模型与资源；将其他语言 SDK/C ABI 标为 deferred（等待用户明确启动）；给每项标记 `parity / excluded / deferred`。明确排除 VLM、UniTable torch、独立 torch/OpenVINO/Paddle 后端；CANN/DirectML 作为延期 EP，不静默改配置。
2. 检查 fixtures 与 UPSTREAM 引用文件。恢复有权分发的六格式样本；无现成样本时用 `tests/native_office/_legacy_*_test_utils.py` 等生成最小文件。生成文件也登记 SHA256/来源/生成器版本。
3. 样本覆盖：中文/英文、多栏、原生/扫描/混合 PDF、CID 异常、旋转、稀疏文字、图表、公式、印章、页眉目录、跨页表格；六种 Office 含样式、链接、嵌入图片/公式、合并格、异常容器与资源限制。
4. 抽取当前 Python 输出：markdown、middle_json、content_list_json、asset 摘要和完整字节。另存 PDF 字符/绘图事实、模型前后处理张量以定位回归。
5. manifest 至少含 `id/input/sha256/format/license/tags/config/model_manifest/expected`。缺文件直接失败；禁止通过空 glob 获得“零项通过”。
6. 比较器仅规范化随机 ID、绝对缓存路径、时间戳等无语义字段；不可折叠中英文空格、排序块或删除未知字段来隐藏差异。
7. 基准脚本增加 warmup、repeat、分阶段耗时、RSS、输出哈希及环境记录。CPU 与 CUDA 分别建基线；当前 Python 标准图与自定义 CUDA 图分别记录。

**验收命令（脚本创建后）：**

```powershell
.venv\Scripts\python.exe scripts/migration/inventory.py --check
.venv\Scripts\python.exe scripts/migration/export_baseline.py --manifest tests/migration/corpus_manifest.json --output tests/migration/golden
.venv\Scripts\python.exe -m pytest tests/native_pdf tests/native_office tests/test_office_native_integration.py tests/test_native_pdf_integration.py -q
.venv\Scripts\python.exe scripts/bench/parse_bench.py --engine python --manifest tests/migration/corpus_manifest.json --warmup 1 --repeat 5 --output build/bench/python.json
```

期望：manifest 无缺失项，合成/真实语料均能产生结果。预先存在的失败单列并修复或解释，不在这份计划中假定全量测试已通过。提交建议：`test: establish reproducible migration corpus`。

## P01：验证 Rust 复用与 PDFium/推理/分发边界

**新增：** `experiments/rust-probe/Cargo.toml`、`experiments/rust-probe/src/main.rs`、`docs/plans/rust-runtime-matrix.md`、`docs/plans/rust-reuse-matrix.md`。

**参考：** `D:/CodeProjects/doc/DocVortex/rust` 的三个 crate，及 firecrawl/pdfium-rs 的 library/text/sys 实现；具体评审见可行性文档 2.1、4.1。

步骤：

1. Windows MSVC 编译 hello-world，确认 linker、目标架构和运行库可用；不是只检查 rustc 版本。
2. 在相同 PDFium 版本、独立进程下对照 firecrawl-pdfium 与 pdfium-render：字符、advance/ink bbox、字体/角度/颜色、paths、嵌套 Form、图片、注释/链接、页选择及渲染。先列函数表缺口，不能只测渲染即宣布满足。扩展方案须复用唯一运行库和同一把锁；若扩展面过大，选择 pdfium-render 并仅借鉴前者部分设计。
3. ort 探针加载当前 OCR、UNET/SLANet 和 FormulaNet 代表模型，验证动态尺寸、字典元数据、输出 dtype；用相同输入张量对比 Python 标准 ORT。
4. oar-ocr 仅选择小模型模块，在自有适配接口内跑同一模型；检查是否能保留本项目 preprocess 与 postprocess。不能等价则仅借鉴局部算法。
5. 制作最小 wheel，独立虚拟环境加载 PDFium/ORT 并运行；动态库定位失败即修正包结构。
6. 可获得 NVIDIA 环境时独立测 CUDA EP 与现有 custom op 注册；不可获得时记录 pending，不伪造通过。
7. 记录本地 DocVortex 来源提交/文件哈希/许可，运行 core 契约测试，与 RapidDoc 的 Python 规则逐模块差分。建立映射：core 的 geometry/text/tables 等 → native/core；pdfium 的批量抽取 → PDF 适配层；python 的转换/错误处理仅作参考。禁止正式依赖本机绝对路径。
8. 对 DocVortex 借用函数地址、句柄、Python guard 的入口重设计为 Rust 所有权；验证旧 pypdfium2 与新 binding 共存的锁和初始化策略，不能传递跨库句柄。先用独立进程差分。
9. 固定 Rust MSRV、ort crate/ORT API/runtime、PDFium API/runtime、oar feature、模型版本；统一 Cargo.lock，禁用不必要默认下载与 VLM feature。
10. 在复用矩阵标明“可直接复用/需适配/需重写”及验证证据，再调整排期；本地 PyO3 abi3-py311 不得直接覆盖 RapidDoc 的 Python 3.10 支持。

**命令：** `cargo run --manifest-path experiments/rust-probe/Cargo.toml --release -- --manifest tests/migration/corpus_manifest.json`。

**验收：** 产出 Rust 内核复用矩阵、PDFium 能力/缺口矩阵、至少一个 PDF 事实差异报告、一个模型张量差异报告和 wheel 安装报告。另运行 `cargo test --manifest-path D:/CodeProjects/doc/DocVortex/Cargo.toml -p docvortex-core --offline --locked --target-dir build/docvortex-core-probe`，作为本地内核参考测试；依赖未缓存时记录环境缺口，不修改相邻仓库来绕过失败。若 PDFium API 缺失，列精确函数并小范围封装；若模型不兼容，先解决版本/算子再进入模型大规模迁移。提交建议：`chore: validate Rust native runtime boundaries`。

## P02：建立核心类型、坐标、错误与资源契约

**新增：** 根 `Cargo.toml`、`Cargo.lock`、`rust-toolchain.toml`；`rust/rapiddoc-core/src/{lib,document,geometry,config,error,assets,limits}.rs`；`rust/rapiddoc-core/tests/contracts.rs`；`docs/api/document.schema.json`。

**依据：** vendor `schema.py`、`foundation/_coordinates.py`、`foundation/_geometry.py`、`content/spans.py`、Office `limits.py`、现有两路 middle_json 转换器。

步骤：

1. 创建只含 core 的 workspace；配置 fmt/clippy，依赖只用 P01 已验证组合。
2. 测试 PDF point/像素/归一化坐标往返、90/180/270 度旋转、空框、NaN/Infinity 拒绝与容差边界。
3. 定义 Block/Span/Table/Cell/Asset，保留公式、链接、列表、单元格跨度、来源和阅读顺序。Office 逻辑页使用可选几何与来源单位。
4. 定义错误码、每页诊断、资源预算、取消检查点；限制文档字节、解压总量、XML 深度、记录数、图片像素、资源累计量与模型缓存。
5. JSON schema 加版本；采用稳定枚举序列化名。内部不依赖 Python、PDFium、ORT；禁止核心初始化时读环境或联网。

**验收：** `cargo test -p rapiddoc-core`；契约测试覆盖无损序列化、无效输入与资产引用完整性。提交建议：`feat: define document and runtime contracts`。

## P03：迁移 PDF 事实抽取，不同时迁移布局规则

**新增：** `rust/rapiddoc-pdf/src/{lib,document,page,text,paths,images,transform}.rs`；`rust/rapiddoc-pdf/tests/extraction.rs`。

**依据：** `rapid_doc/utils/PyPDFium2Parser.py`、`pdf_text_tool.py`、`pdf_reader.py`、`pdfium_guard.py`、vendor `analyzers/native/pdf/char_geometry.py`、`native_text.py`。

步骤：

1. 使用 P01 选定的唯一 PDFium binding；将 DocVortex read_characters/fonts/paths/objects/path_evidence 等按复用矩阵改接类型化句柄，移除 Python ctypes 地址协议。以 P00 事实数据为 golden，先覆盖字符内容、font/advance/ink 边界、空白、图元、cropbox、旋转、Form 变换。
2. 用 RAII 管理 document/page/textpage/bitmap；父对象存活直到子对象关闭。FFI 调用和销毁都遵守同一锁。
3. 一个文档上下文复用句柄；提取为独立拥有的 Rust 数据后再释放锁。禁止裸句柄通过 Send/Sync 绕过 PDFium 限制。
4. 实现页选择和坏页诊断，保留原始 page index；不得因为跳过坏页重编号导致图片与页错配。
5. 最后加入按页窗口渲染，并用长文档验证峰值内存不会随已消费页面线性增加。

**验收：** `cargo test -p rapiddoc-pdf --test extraction`；覆盖关闭顺序、多次打开/释放、空文档、损坏页。像素差异单列；文本/几何差异定位到字符而非最终 Markdown。提交建议：`feat: add PDFium extraction adapter`。

## P04：按现有规则族迁移原生 PDF

2026-10-24 公式编号追加检查点：legacy_page/formula.rs/optimize_formula_numbers 按旧 `_optimize_formula_number_blocks` 移植邻块优先、同排几何回退、tag 去重/冲突与孤立编号处理；718 组完整旧 JSON、382 组原模型→编号→段落、639 组原模型→编号→输出精确一致。assemble_model_pages 显式 optimize_formula_numbers=True 时在 para_split 前接入，默认 False 保持既有行为；不运行公式识别模型，不含跨页编号、资产、后置 OCR 或跨页表格合并。

2026-10-24 页面组装追加检查点：legacy_page.rs 与 legacy_page/magic.rs 移植 raw layout_dets→MagicModel 规则→块/span 预处理→模型 original_order→page_info，再串联旧 para_split；286+18 组实际旧单页输出与 382 组串联精确一致。范围为 OCR 已识别、writer=None；不含原生字符填充、资产写出、后置 OCR/公式编号/跨页表格。默认 Markdown/content_list 的独立 stage 亦已接通，详见 P05；enhanced/XY-cut 仍延期。

**2026-10-24 用户范围调整（优先于下方历史安排）：** 本期采用版面模型 original_order，enhanced/XY-cut 阅读顺序及其专属辅助内核延期，其他任务完成后等待用户明确要求才启动。缺少模型顺序时保持明确 unsupported，不使用实验几何排序代替。下一步迁移 model_json_to_middle_json.py 的页面组装及实际依赖：span/块归属、图表 body/caption/footnote 分组、丢弃块与坐标，串联已验证的模型顺序和段落内核；随后按 P05 迁移 Markdown/content_list 与图片资产，完成模型 PDF 路径的端到端差分。

2026-10-24 旧排序追加检查点：`legacy_block_sort.rs`/`sort_model_blocks` 迁移旧 block_sort 的 original_order 分支及虚拟行、全页行号、图表回组、中位数索引。1,206 组完整旧输出及 308 组 sort→para_split 串联（含 8 组实际 CPU OCR）精确一致；独立 JSON 副本入口，无模型顺序明确 unsupported。完整模型/导出集成继续迁移；enhanced/XY-cut 按上述用户范围调整延期，不启用实验几何替代。

2026-10-24 旧段落追加检查点：`legacy_paragraphs.rs`/`split_paragraphs` 按未修改的 `para_split.py` 移植完整 JSON 处理链，850 组逐字段对照一致，含列表/目录标记、倒序跨块/跨页合并、原字段与删除标记。FP64 边界测试定位并修复 JSON 解析 1 ULP 偏差，固定 Unicode 表补 isupper/isalpha；759 个大写边界字符及独立 CPU OCR 输入通过。此为旧 stage 的独立 API，尚未接入完整模型/阅读顺序/导出，实验策略继续隔离。

**新增：** `rust/rapiddoc-native/src/{lib,pipeline,text,layout,reading_order,graphics,margins}.rs`、`rust/rapiddoc-native/src/tables/{mod,grid,recovery,continuation}.rs`；`rust/rapiddoc-native/tests/regressions.rs`。

**依据：** vendor `analyzers/native/pdf/` 全部规则、`rapid_doc/model/docvortex_adapter.py`、`rapid_doc/utils/table_merge.py`、`rapid_doc/model/reading_order/`。

优先对照本地 `docvortex-core/src/{geometry,dedup,text_spacing,text_pipeline,table_candidates,tables,table_merge}.rs` 及相关模块。逐项迁入通过差分的内核，保留对应契约测试和来源，不再从 Python 重写已有等价算法；尚无覆盖的整页/整文档编排继续迁移。

每个子项独立对比/提交：

1. 文本去重、行合并、CJK 空格与字号几何；映射 `test_pdf_text_dedup.py`、`test_native_table_spacing.py`、`test_native_table_rotation.py`。
2. 原生表格：物理网格 → 单元格归属 → partial grid → 无框线回收 → 表格/图形仲裁；映射 `test_partial_grids_and_body_labels.py`、`test_graphic_table_arbitration.py`。
3. 图片与表内媒体归属、链接与注释；映射 `test_table_media_and_toc.py`。
4. 标题、页眉/页脚/目录、正文边缘标签与阅读顺序；映射 `test_reported_layout_regressions.py`、`test_leading_flow_order.py`。
5. 跨页表格、相邻页证据、终端线段和列轨道；映射 `test_trailing_table_grid.py`、`tests/test_table_merge_continuation.py`。
6. 路由与回退：稀疏文字、CID 异常、图片覆盖、规则布局为空、强制 OCR；映射 `test_pdf_classify_sparse_text.py`、`test_pdf_classify_cid_quality.py`、`test_rule_table_routing.py`。

规则移植先保持阈值和顺序；优化索引/排序另提交并做差分。所有测试名默认在 `tests/native_pdf/`，明确以 `tests/` 开头或根测试中的同名文件除外；执行时通过 `rg --files tests` 定位，禁止臆造新 golden。

**验收：** `cargo test -p rapiddoc-native --test regressions`；每个原 Python 回归用例有 Rust 对应 ID，验证文本/资产不丢失、block 顺序与表格拓扑。提交建议按子项命名，不一次搬运 5 万行。

## P05：输出器、路由与 CLI 闭环

2026-10-24 模型OCR执行追加：execute_model_ocr及拆分stage接通真实det1→旧框整理→原始RGB裁图→批rec6→旧模型回填。131框规则、32计划/16裁图、112阈值规则、5真实CPU链精确对照；旧ocr_patch实际固定960、crop旋转门槛2、F32弱标量与处理时序保留，预览默认不变。det批>1/旧逐图失败重试未实现，错误原子传播；下一步公式/表格任务与多页正式编排，enhanced仍延期。

2026-10-24 原生OCR准备追加：derive_native_ocr_boxes/prepare_model_ocr保留旧原生框聚类/合并/公式切分和FP32逆变换、模式/方向回退、遮罩与64像素分组补白。182旧框、54计划和75实际旧检测输入像素精确；固定空检测响应不计为实际检测证据。下一步旧det框整理、真实det/rec执行/回填，再接公式/表格和多页正式API。

2026-10-24 模型准备追加检查点：predict_model_layout/convert_model_layout/filter_model_layout移植旧PPV2/V3类别及BatchAnalyze过滤；prepare_model_crops按旧收集与裁图规则输出任务/图片，prepare_pdf_model_inference自动执行PDF→真实布局模型→计划。258转换/86过滤、60计划/409裁图、3真实CPU布局精确对照。必须显式rect，掩码多边形不以矩形替代；PIL/ndarray与随机UUID投影差异已记录。下一步旧原生检测框与OCR任务执行/回填，再接公式/表格及多页正式API。

2026-10-24 PDF原图追加检查点：extract_original_pdf_images与assemble_pdf_model_page接通PDF→旧原生分组/RGB渲染/原图→旧资产组装，102组实际旧兼容别名提图参考/432图片像素与17完整资产页。get_pos移除导致未适配旧输出全空，Rust使用对应get_bounds恢复；UUID稳定命名与解码诊断列有意差异。原有像素/遍历/过滤/缩放/最后IoU匹配规则保持，正式模型路由/API仍待完成。

2026-10-24 原生PDF桥接追加检查点：read_native_pdf_page按旧pdftext 0.7.1从PDF读取原生字符、去重、span/行/上下标/块，28完整旧PDF页面与268边界参考；render_pdf_page增加RGB模式供现有字符填充使用。5组PDF页面及5条真实CPU后置OCR链改为Rust直接读取/渲染；默认RGBA和模型original_order保持。正式模型推理路由、PDF原图导出仍待完成；enhanced继续延期。

2026-10-24 原生字符追加检查点：fill_model_text_spans/assemble_model_page_with_native_text移植旧txt_spans_extract，接入原资产writer及后置OCR；198组旧span、32完整页、295对比度、15条真实CPU链/45输出，保留NumPy负切片、对比度0.17边界、字符首归属、正负零稳定排序与字段类型。仅RGB页图、旧分组JSON显式输入、无word_box；下一步迁移原生事实去重/分组自动桥接，仍不能视为完整PDF解析器。

2026-10-24 后置 OCR 检查点：新增副本回填 apply_model_ocr_results 与图片ID/PNG映射 post_process_model_ocr，串联 Rust 宽高比排序、共同 padding 批识别。73 组旧回填、12 组实际旧CPU批参考（11支持/1明确歧义拒绝）、11条CPU响应→编号→段落链/33输出。结果保持严格>0.5和三位小数；失败保留原输入。下一步迁移 txt_spans_extract 原生 PDF 字符填充与补识别裁图，再补 PDF 原图提取和正式推理/API；不是完整解析器验收。

2026-10-24 跨页表格追加检查点：在现有 rapiddoc crate 内实现旧 merge_table stage，独立 merge_cross_page_tables 及批组装 merge_tables=True（默认 False）。313 组完整旧页面 JSON、864 输出、48 组实际单页→段落→合并及 144 输出，覆盖续表/脚注/网格/稀疏续行/多页/对象身份。仅支持正确闭合非嵌套单表 HTML，错误 HTML 修复等明确 unsupported；后置 OCR、正式推理/API 和 PDF 原图提取仍待完成，enhanced 继续延期。

2026-10-24 表内 UUID 追加检查点：table_fill_images 描述符接入 MagicModel 过滤后、普通裁图前，57 组旧页面/writer 与 171 组旧输出保留原始证据，明确投影新旧资产名称后比较其他字段/像素/元数据/顺序。旧命名包含函数地址，改为 table_fill/页像素 MD5/页号/取整 bbox 稳定哈希是有意输出差异；计算失败不再吞异常。跨页表格已冻结 99 组实际旧完整页面 JSON 基线（含既有 5 个续表回归），Rust 合并器尚未实现；下一步按旧规则移植，不启用 enhanced。

2026-10-24 资产追加检查点：legacy_page/assets.rs/assemble_model_page_with_assets 按旧页像素 MD5、bbox 取整和 SHA256 生成平铺 PNG，在旧裁图位置回填路径；支持目录与 write(name,bytes) writer、越界补零和显式原图最后匹配。337 组实际旧 writer 页面/调用的 JSON、文件名、顺序和解码像素一致，337 组资产页→段落与 1011 组旧输出通过；PNG 压缩字节不保证相同。追加 HTML 内联图片（含表格 HTML）原字节写出与旧字符串替换，9667 组工具与 12 组完整页面参考；UUID 表内图片替换、PDF 原图提取、其余文档后处理和正式推理路由仍待完成。

2026-10-24 输出追加检查点：legacy_page/output.rs 与 render_model_pages 移植旧 union_make 的默认 $/$$、mm_markdown/nlp_markdown/content_list；726 组完整旧输出与 raw-model→页面→段落→输出链精确一致。171 组 Magika 算法链显式排除，遇到算法文本块报 unsupported；图片 writer、自定义分隔符、content_list_v2 与正式解析路由仍待完成，不能把 stage API 当作本阶段全部验收完成。

**新增：** `rust/rapiddoc-render/src/{lib,markdown,html,legacy_json,assets}.rs`、`rust/rapiddoc/src/{lib,engine,router}.rs`、`rust/rapiddoc-cli/src/main.rs`；对应 tests 下 `golden.rs`、`routing.rs`、`cli.rs`。

**依据：** `rapid_doc/backend/office/mkcontent/`、`backend/pipeline/pipeline_middle_json_mkcontent.py`、`backend/utils/markdown_utils.py`、`rapid_doc/main.py`。

步骤：

1. 先用纯 IR fixture 测试 Markdown escaping、嵌套 span、链接、数学、HTML 合并单元格和图片命名。
2. 保留原有 middle_json/content_list 输出契约，单独实现 exporter；不要求外部用户立即改字段。
3. 引擎统一 bytes/文件输入与 parse_method，确保 native PDF 路线不创建模型 session。
4. CLI 定义 `parse INPUT --output DIR`、`doctor`、`capabilities --json`；stdout JSON 与 stderr 日志分离。此时 capabilities 只列真正已实现的格式。
5. 写性能日志和差分入口；跨页处理保留上下文，资产输出路径校验位于目标目录内。

**验收：** `cargo test -p rapiddoc-render -p rapiddoc -p rapiddoc-cli`；`cargo run -p rapiddoc-cli --release -- doctor`；用 manifest 第一份原生 PDF 运行 parse，得到有资产可解析的完整输出。提交建议：`feat: ship native PDF CLI preview`。

## P06：Python 兼容层与首个 CPU 安装包

**新增：** `bindings/python/Cargo.toml`、`bindings/python/pyproject.toml`、`bindings/python/src/lib.rs`、`bindings/python/python/rapid_doc_rust/__init__.py`、`bindings/python/tests/test_api.py`、`bindings/python/tests/test_install.py`。

**修改（验收后）：** `rapid_doc/main.py` 增加明确的试验性 Rust 路由；不改变默认 backend。

步骤：

1. 先用独立预览包避免覆盖现有 `rapid-doc`；封装同名风格 RapidDoc/RapidDocOutput 与 bytes/path 输入。
2. 按 P00 契约实现单文档/批处理返回、markdown/images 解包、参数默认、页范围与异常；文件流在 Python 层读成 bytes。
3. 计算阶段释放 GIL；writer callback 只在安全宿主边界调用。Rust 内部不保存未经生命周期管理的 Python 对象。
4. wheel 携带或明确依赖匹配的 CPU runtime，模型不嵌入；测试禁网、无 Rust、无 PyTorch 环境。保留 Python 3.10 契约；不能因参考库使用 abi3-py311 就自动抬高最低版本。
5. 原生 PDF 必须能在没有模型缓存的情况下完成；缺模型的扫描 PDF 返回明确错误，不能冒充已支持。

**验收：** `maturin build --release --manifest-path bindings/python/Cargo.toml`；在独立环境安装实际产出的 wheel 后运行 `bindings/python/tests/`。将 P00 的 API 合同用例对两种引擎执行。提交建议：`feat: add Python CPU preview package`。

## P07：统一 ONNX Runtime、模型仓库与缓存

**新增：** `rust/rapiddoc-inference/src/{lib,runtime,session,model_store,manifest,provider}.rs`、`models/manifest.json`、`rust/rapiddoc-inference/tests/runtime.rs`。

**依据：** 三套 `rapid_*_self/inference_engine/onnxruntime/`、`backend/pipeline/model_init.py`、`utils/models_download_utils.py`、`model/ocr/onnx_ops_session.py`。

步骤：

1. 枚举所有目标 ONNX 模型，记录 hash、license、tensor 合同、字典/预处理/后处理版本；本地已存在模型也校验。
2. 统一 session 配置、EP 选择、线程预算、provider 实际启用检查、错误类型；将模型专属图优化保留为独立策略。
3. 缓存键包含模型与规范配置；引擎明确持有 session pool，不复制多个全局 singleton。定义同 session 可否并发，依所选 ort API 验证。
4. model store 提供显式下载、代理/镜像、超时、锁、SHA256、临时文件原子替换和 offline；损坏/半下载缓存可恢复。
5. `models fetch` / `models verify` / `doctor` 使用同一 manifest。没有选择模型路线时不加载 ORT。

**验收：** `cargo test -p rapiddoc-inference --test runtime`；覆盖断网、哈希不符、并发 fetch、无权限、模型缺失、CUDA 请求不满足、缓存配置隔离。提交建议：`feat: centralize ONNX runtime and model storage`。

## P08：OCR、方向与印章

**新增：** `rust/rapiddoc-inference/src/ocr/{mod,detect,recognize,preprocess,postprocess,crop,seal,orientation}.rs`、`rust/rapiddoc-inference/tests/ocr_parity.rs`。

**依据：** `rapid_doc/model/ocr/{rapid_ocr,ocr_patch,seal_crop}.py`、`utils/ocr_utils.py`、`model/orientation/`、当前 rapidocr 锁定版本行为。

步骤：

1. 针对固定输入先导出 Python 前处理张量，验证 RGB/BGR、resize/interpolation、padding、normalize、dtype 和 shape。
2. 完成 DB 检测、框排序/合并、旋转透视裁剪、宽度分桶、CTC 与字典；按功能使用 oar 模块或自有实现，不整体替换默认参数。
3. 迁移印章多边形裁剪和方向分类；空白、超长、低分、极小框都写边界用例。
4. 接通扫描 PDF 与纯图片输入；混合 PDF 只给需要的区域 OCR，保留 native ownership。

**验收：** `cargo test -p rapiddoc-inference --test ocr_parity`；原测试 `tests/test_ocr_detector_preprocess_compat.py`、`test_native_ocr_region_bounds.py`、`test_ocr_model_download_config.py` 的行为有对应回归。先比较张量再比较检测框/字符，误差不能只看平均置信度。提交建议：`feat: port OCR and seal processing`。

## P09：布局、表格与公式模型

原生表格追加检查点（2026-10-23）：`rapiddoc/src/native_table.rs` 已接布局候选的保守原生网格优先选择，可靠时使用原文/物理轨道与安全 HTML，否则回到现有模型路线；同页原生/扫描表分别处理。`grid::plan_tracks/recover_with_tracks` 记录实际轨道，失败查询计入全页预算，原已知区域 API 保持输出契约。旋转、无边线/填充条、漏检补偿和完整全文/跨页融合仍未完成。

段落追加检查点（2026-10-23）：`rapiddoc-native/src/text_boundary.rs` 迁移旧文本换行规则，13,930 组旧 Python 参考一致，固定 Unicode 16.0.0 属性。`rapiddoc/src/paragraphs.rs` 与 Python `organize_page_text` 提供显式块内段落视图，保留源行/块/栏索引，支持保守分栏与间距/缩进/列表边界，表格整体保留、算法保留原换行；7 组几何合同和真实页面验证通过。续增悬挂列表续行：19 组独立合同 × 3 比例、双栏/边界/并发和真实扫描 OCR、72/137 DPI 原生 PDF 通过；不跨块，不推测嵌套层级。几何策略是明确的新预览规则，完整旧段落器、跨块阅读顺序、复杂列表和跨页合并仍待迁移。

行序追加检查点（2026-10-23）：Rust `organize_ordered` 与 Python `line_order="geometry"` 提供可选栏内几何排序；默认 source 保持原行为，所有索引指向原输入行。对同排/重叠、空行、非水平、竖排和算法保留原序并诊断；排序后的前缀扫描量仍受全页累计预算。14 组独立合同 × 3 比例与打乱真实 OCR、72/137 DPI 原生页面结果后的恢复通过。尚未解决跨块/跨页阅读顺序、通栏混排和同排片段组装。

块序追加检查点（2026-10-23）：`reading_order.rs` 与 Python `block_order="geometry"` 提供可选完整块排序，支持清晰并排栏及通栏头尾；默认 source 不增加结果字段，启用时返回原索引计划/原因/工作量。歧义时整页保留原序，预算超限报错，保留表格和未支持块。16 组合同 × 3 比例、真实扫描双栏和 72/137 DPI 原生页、720 种排列与 1000 区域验证通过。完整旧 XY-cut-plus、页中通栏混排、重叠仲裁、标题脚注归属与跨页整合仍未迁移。

追加检查点（2026-10-23）：`native_text.rs` 与 `recognize_pdf_page_hybrid` 已接正文逐检测行原生/OCR 融合，按实际渲染尺寸投影，保留源字符索引和回退原因；原生可靠覆盖行跳过识别模型，图片重叠/旋转/覆盖不足等继续 OCR。当前为保守水平行策略，表格仍使用 OCR/结构模型；原生表格与完整文档融合仍未完成。

当前实现检查点（2026-10-23）：模块暂按能力平铺在 `rust/rapiddoc-inference/src/`，包括 `layout.rs/layout_post.rs`、`table_structure.rs/table_match.rs`、`unet.rs/unet_lines.rs/table_components.rs/table_lines.rs/table_draw.rs/table_regions.rs/table_recover.rs/wired_match.rs/blank_cell.rs` 及 `table_classification.rs/table_classifier_gray.rs`。已接 PP-DocLayoutV3 rect、SLANet 裁图 OCR/HTML、UNET 格框→跨格 OCR 匹配→空格补识别→安全 HTML（`rapiddoc-render/src/wired.rs`），以及 Paddle/Qanything 单/双分类器到两个表格分支的裁图路由；已接低置信度双结构质量仲裁（rapiddoc-render/src/quality.rs）及共享 OCR；已新增 page_tables.rs，接通矩形单页表格、PDF 页渲染及页面像素坐标投影；已接正文区域 OCR 与表格有序页面块（recognize_page_png/recognize_pdf_page），共用整批裁图预算并保留未处理区域诊断；完整文档版面、原生融合、mask 版面和公式仍未完成。具体合同及安装验证以 `rust-migration-status.md` 为准。

**新增：** `rust/rapiddoc-inference/src/{layout,table,formula}/mod.rs` 及各自 `preprocess.rs`、`postprocess.rs`；`rust/rapiddoc-inference/tests/structure_parity.rs`。

**依据：** `model/layout/rapid_layout_self/model_handler/`、`model/table/rapid_table.py`、`table_structure/unet/`、`pp_structure/`、`model/formula/rapid_formula_self/model_handler/pp_formulanet_plus/`、`model/formula/fix_utils.py`。

子任务：

1. 对齐布局类别 ID、坐标、mask、NMS 和排序；PP-DocLayout 不同版本分别登记。
2. 表格分类→UNET 有线→SLANet 无线→OCR cell matcher；验证空表、跨行跨列、无框/旋转、错误分类回退。
3. FormulaNet 的 encoder/decoder 或图内生成逻辑以实际模型合同为准，对齐 EOS、token、长度上限、cache、batch 与公式修复；不能按普通单次分类器封装。
4. 原生规则优先/补充模型/强制 OCR 的当前配置语义完整接通；旧 UniTable/Torch 配置返回带迁移说明的错误。
5. 标准图达到质量门槛后才允许进入 P10，不先移植优化图掩盖基线问题。

**验收：** `cargo test -p rapiddoc-inference --test structure_parity`；CLI 对扫描/混合文档输出完整结果；分别报告 OCR CER、布局匹配、表格拓扑/TEDS、公式 token 或结构匹配。提交建议按 layout/table/formula 拆分。

## P10：CUDA、已有插件与模型图优化

**新增：** `rust/rapiddoc-inference/src/{custom_ops,optimized_models}.rs`、`rust/rapiddoc-inference/tests/cuda_parity.rs`、`packaging/runtime-manifest.json`。

**依据：** `rapid_doc/backend/onnx_ops/{session,rewrite,formula,formula_rewrite,ctc}.py`、`native/onnx_ops/src/ops.cc`、`kernels.cu`、布局/table 的 `optimization.py`。

步骤：

1. 原始图用相同 GPU、EP 和精度与 Python 标准图对照；GPU 请求失败时按明确策略报错或 fallback。
2. 用 ORT custom-op 注册接口加载现有 native 库；无需第一版重写 C++/CUDA。插件能力/版本与模型 custom domain 校验。
3. 优先离线生成优化图并随 manifest 记录来源；自定义用户模型用原始图。若运行时图重写是必要公开能力，再实现 protobuf 变换和结构验证。
4. 覆盖动态 shape 冷启动、TF32 开关、预优化、mask dtype、公式 rewrite、缓存失效、外部权重，禁止原模型被就地覆盖。
5. provider 初始化失败与推理失败分别处理；输出含实际设备和优化状态。性能对比包括“Python 已启用现有优化”的基线。

**验收：** `cargo test -p rapiddoc-inference --features cuda --test cuda_parity -- --ignored` 在 NVIDIA 实机执行。Windows/Linux 分别验收；无 GPU 时只能登记未测，不能发布已验证声明。提交建议：`perf: integrate optional CUDA optimizations`。

## P11：DOCX/PPTX/XLSX 原生迁移

**新增：** `rust/rapiddoc-office/src/{lib,opc,relationships,styles,resources,limits}.rs`、`src/docx/mod.rs`、`src/pptx/mod.rs`、`src/xlsx/mod.rs`、`rust/rapiddoc-office/tests/ooxml.rs`。

**依据：** vendor `analyzers/native/office/{docx,pptx,xlsx,spreadsheet}/`、`opc.py`、`ooxml_chart.py`、`rich_text.py`；`backend/office/model_output_to_middle_json.py`。

步骤：

1. 共享 OPC 容器、content types、relationship 解析与资源限额；不要假设固定部件路径，兼容 Strict/Transitional 命名空间。
2. DOCX 分项迁移段落样式、列表编号、表格/嵌套表格、超链接/域、图片、OMML；取代 mammoth HTML 往返时逐样本对照。
3. PPTX 迁移 shape 顺序、group transform、文本层级、表格、图片、notes、默认 slide size、AlternateContent。
4. XLSX 先用 calamine 探针对比值/公式，再按 OOXML 补格式化显示、日期基准、合并格、富文本、图片/图表、区域发现；不执行公式和宏，明确缓存值缺失行为。`xlsm` 继续按现有 XLSX 路由范围登记。
5. 资源输出沿用统一 AssetStore；P13 未覆盖的公式/矢量内容显式标诊断，阶段能力状态不得写“完全等价”。

**验收：** `cargo test -p rapiddoc-office --test ooxml`；对应 `tests/test_office_structured_spans.py`、`tests/native_office/test_office_robust_fixtures.py`。P00 必须让后者不再因空目录而无样本。每种格式独立提交。

## P12：DOC/PPT/XLS 二进制迁移

**新增：** `rust/rapiddoc-office/src/legacy/{mod,binary,officeart}.rs`、`src/doc/{mod,pieces,sprm,fields}.rs`、`src/ppt/{mod,records}.rs`、`src/xls/{mod,records,strings,number_format}.rs`、`rust/rapiddoc-office/tests/legacy.rs`。

**依据：** vendor Office `legacy/`、`doc/`、`ppt/`、`xls/`、`streams.py` 及 `tests/native_office/_legacy_*_test_utils.py`。

步骤：

1. cfb 仅负责 stream；实现长度检查、checked arithmetic、有界递归和截断诊断，不信任文件里的计数。
2. DOC：FIB、piece table、编码、格式 SPRM、列表、域、图片与表格。
3. PPT：persist/record 链、文本样式、对象/图片引用；保留对象索引稳定。
4. XLS：BIFF、SST/continuation、显示数值格式、合并格、图表/图片与区域投影；复用已验证的 calamine 能力时也必须保留 RapidDoc 输出。
5. 按现有策略处理超大单图：跳过并诊断但不重排后续图片引用；累计资源预算超限整文档失败。

**验收：** `cargo test -p rapiddoc-office --test legacy`；对应 `test_flash_legacy_doc.py`、`test_flash_legacy_ppt.py`、`test_flash_legacy_xls.py`、`test_office_image_resource_limits.py`（位于 tests/native_office）。无需 Office/LibreOffice、不启动转换进程。每种格式独立提交。

## P13：公式编码与 WMF/EMF 覆盖收口

**新增：** `rust/rapiddoc-office/src/equation/{mod,unicode_latex,omml,mtef}.rs`、`src/metafile/{mod,records,state,display_list,render}.rs`、`tests/equations.rs`、`tests/metafiles.rs`（后两个位于该 crate）；`scripts/migration/export_latex_cases.py`。

**依据：** vendor `office/equation/`、`office/image.py`、当前 pylatexenc 2.x 行为以及固定提交的 metafile-render 实现。

步骤：

1. 导出 Unicode 编码映射、保护花括号和未知字符用例；迁移本地 `latex_dict.py` 覆盖优先级。
2. 对 OMML/MTEF 结构迁移分式、上下标、根号、矩阵、组合字符与损坏记录。不要把所有公式降级为图片以通过文本用例。
3. WMF/EMF 按 record 类型实现状态机，统一 display-list，再接 SVG/resvg；原始资源保留，支持/近似/跳过统计可查询。
4. 固定字体和 DPI，测文本/裁剪/变换/位图/EMF+，受限解压和像素预算不得取消。空结果不能被标记为成功图片。
5. 公布覆盖矩阵；未支持项转为具体样本任务。六格式“完整替换”门槛应覆盖原 Python 能正确处理的目标样本。

**验收：** `cargo test -p rapiddoc-office --test equations --test metafiles`；公式结构对照、矢量图可视化检查与资源边界用例均通过。字体导致的容差单列，不能用大阈值吞掉缺文字。提交建议分 Unicode/OMML/MTEF/metafile。

## P14：其他语言 SDK（延期，等待用户明确启动）

本阶段不属于当前执行范围，也不是 P15 的依赖。现在不创建 rapiddoc-ffi、C 头文件、Node.js/Java/Go 绑定、npm/Maven/Go 包或对应 CI。

启动条件必须同时满足：P15 已交付可安装的 Python 包；用户实际测试并确认没问题；用户再次明确要求实现其他语言。自动化测试通过、agent 自评完成或时间经过都不构成启动指令。

满足条件后另写执行计划，按用户指定的语言决定 N-API/JNI/cgo/C ABI、发布平台和验收范围。当前只保持核心使用 Rust 类型、资源/错误契约清晰，不为将来的绑定提前实现额外基础设施。

## P15：Python 多平台发布、性能验收与用户测试

**新增：** `.github/workflows/rust-ci.yml`、`.github/workflows/rust-release.yml`、`packaging/README.md`、`docs/rust-installation.md`、`docs/rust-compatibility.md`、`docs/benchmarks/rust-migration.md`。

**修改：** P06 的 Python 预览路由、根包配置与 README，仅在本阶段门槛满足后调整正式发布策略。

步骤：

1. CPU Windows x64/Linux x64 先发布测试产物，随后 macOS arm64；建立干净机器/容器安装测试，Python 用户无需 Rust 编译器或 PyTorch，仍需受支持的 Python 解释器。
2. runtime manifest 包含平台、文件 SHA256、ORT/PDFium/plugin/API 版本、CUDA/cuDNN 需求和许可证；扫描漏打包依赖及不必要大模型。
3. GPU 制品与 CPU runtime 不覆盖同一路径；验证 GPU 缺失、驱动不匹配、同进程重复初始化/不同 runtime 请求。
4. Cargo feature 组合逐个测试，不能只用 `--all-features`，因为多 EP/打包选项可能互斥；执行 fmt、clippy、单测、golden、Python wheel 真实安装/生命周期和实机 GPU 测试。
5. 用 P00 相同配置复跑性能和质量。保存逐文档报告，CPU/GPU、新旧优化分别比较，热点确实改善后才作性能声明。
6. 旧 API 的每个参数标明兼容/报错/延期；Python callback 与非 ONNX 选项显式说明。Rust route 先 opt-in，提供 wheel、安装说明、支持矩阵、对比结果和问题反馈模板给用户实测。记录用户反馈并修复；达到质量门槛并完成用户验证后再安排默认切换，保留版本回退说明。用户未反馈时状态为“Python 包已交付，待用户测试”，不自动进入 P14。
7. 最后删除不再被使用的 Python 重复实现和运行时依赖；先确认 wheel/sdist、示例、Docker/API 可选应用未断链。未达到质量门槛的格式继续标预览，不用静默 Python fallback 冒充独立 Rust SDK。

**最终验收命令（各模块存在后）：**

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
.venv\Scripts\python.exe scripts/migration/compare_results.py --manifest tests/migration/corpus_manifest.json --expected tests/migration/golden --actual build/rust-results
.venv\Scripts\python.exe scripts/bench/parse_bench.py --engine rust --manifest tests/migration/corpus_manifest.json --warmup 1 --repeat 5 --output build/bench/rust.json
```

GPU 命令按 P10 单独执行；本期仅 Python 安装测试，使用真实 wheel。上述命令不表示现在就能运行，脚本与 crate 将由对应任务创建。

## 统一验收指标

| 范围 | 必须满足 | 暂定量化目标（P00 后冻结） |
| --- | --- | --- |
| 确定性规则/原生文本 | 所有已登记回归通过，无文本/公式/图片静默丢失 | 文本与块顺序逐项一致；坐标容差按实际单位登记，初始建议 0.1 pt |
| 表格 | 行列、rowspan/colspan、单元格文本、跨页连接不回退 | 固定样本拓扑完全一致；真实评测集 TEDS 单列 |
| 模型 | 同模型/精度/输入比较，所有差异可追溯 | FP32 张量先用 atol=1e-4/rtol=1e-3 诊断，离散预测另比；FP16/CUDA 单独定阈值 |
| OCR/公式质量 | 不用平均分掩盖特定文档缺失 | 初始建议整体字符错误率退化不超过 0.2 个百分点且关键用例零退化；实际值由基线决定 |
| 资源 | 超大输入可控失败，句柄/显存不持续泄漏 | 固定长文档峰值不超过 Python 同配置 110%，并记录绝对值 |
| 性能 | 不降低功能/分辨率、不含下载混比 | 首轮以同功能无明显退化为门槛；原生路径 1.5x 是探索目标，不是已证实结果 |
| Python SDK | 对照旧版输出/错误语义，实际 wheel 可独立安装 | 支持的 Python 版本通过代表语料和生命周期测试；交付用户实测，记录反馈；不要求其他语言 SDK |
| 分发 | runtime 来源可定位、CPU 不依赖 GPU、离线可运行 | 每个声称支持的 OS/架构/EP 有对应安装记录 |

容差是起始建议，禁止在出现差异后不断放宽以“刷绿”。性能不达目标时根据分阶段 profile 优化；若主要时间仍在 ORT/PDFium，允许发布以分发和维护为收益的版本，但如实陈述。

## 下一个 agent 的启动提示

```text
阅读 docs/plans/2026-10-22-rust-feasibility.md 和本计划。
先执行 P00，只建立可复现语料、行为清单和基准，不重写解析器。
检查 HEAD、用户改动、缺失 fixtures 和测试运行环境。
保留本地 DocVortex 修复，以现有 RapidDoc 输出为 oracle。
P01 检查相邻 DocVortex Rust 内核与 pdfium-rs 的复用/缺口；不把它们当作已验证替代。
本期只实现 Python SDK；P14 等用户实际测试确认并明确要求后才启动。
优先阅读 rust-migration-status.md 顶部最新安排：enhanced/XY-cut 延期至用户明确要求。旧模型结果→页面→段落→默认 Markdown/content_list 的无 writer、已识别 OCR JSON stage 已实现；下一步是资产 writer、文档后处理及模型推理/正式 API 集成，不重复启动已完成的内核。
完成后在 docs/plans/rust-migration-status.md 记录证据与 P01 的输入。
没有样本、模型或实机验证的项目标为待验证，不报告为兼容通过。
```

## 已确定的目录与命名

Rust 源码统一放在 `rust/rapiddoc-*`，根 `Cargo.toml` 管理 workspace，Python 绑定放在 `bindings/python/`。Rust 主 crate 为 `rapiddoc`。Python 预览发行包名为 `rapid-doc-rust`，导入名为 `rapid_doc_rust`，原生扩展为 `rapid_doc_rust._native`；现有 `rapid_doc` 保留。名称是项目约定，尚未验证注册表可用性。
