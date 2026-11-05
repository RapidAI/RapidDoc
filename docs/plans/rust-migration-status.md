## 2026-11-03 DocVortex 0.6.12 相关修复同步

Python/Rust 已选择性同步少线跨行表格、Office 超限图片占位、明确 NUL 字体恢复，并适配大写开头的跨页/跨栏续段。最终 wheel 已安装项目 `.venv`；206 项 Python 回归、58 项安装差分/公开调用和 5 项 Rust Office 检查通过。未跑全量/CUDA，未同步服务器。范围、限制和来源见 [同步记录](2026-11-03-docvortex-0612-update.md)。仍为保留本地优化的选择性分支，非完整上游包替换。

# Rust 迁移接续记录

**UNET 真实倾斜格框兼容修复（2026-11-02）：用户报错页 Python 实际成功，输出 2 表、6,274 个 Markdown 字符，第一组 225 格恢复成功；两个合法倾斜框的对角点 Y 反序被 Rust 自加检查误拒绝。** 生产恢复及 OCR 匹配改回 Python 的顺序顶点/有符号面积规则，不删格或修形；完整单页结构、Markdown 和图片内容与原 Python 一致。原生异常模块名及评测文件日志修复，pickle/异步日志回归通过。5 项恢复、4 项匹配和 1 项日志测试通过；Linux 原评测环境已安装修复包，可重跑失败文件，未启动全套评测。详见 [真实表格与异常日志](2026-11-02-unet-tilted-cells-and-native-errors.md)。

**真实评测算法块缺口已修复（2026-11-02）：移除旧 algorithm unsupported，按原 Python standard_v3_3 模型/字节特征/高置信规则接入原生 Rust ORT 分类；不调用 Python Magika、不用固定 txt 代替识别。** 服务器 45 组标签、405 组完整输出一致，实际报错 PPT_PresentationSpecification_page_010.pdf 完整解析并与原 Python 渲染一致；本地 46 组特征/标签、414 组输出和 23 文档安装检查通过。服务器手动评测环境及本地 `.venv` 已更新；评测脚本新增完整/错误日志和失败 JSONL 清单，原命令可断点重跑。未启动整套评测。详见 [算法块与日志记录](2026-11-02-magika-algorithm-and-eval-logs.md)。此前小语料通过不代表当时已覆盖算法块。

**干净环境发布检查完成（2026-11-02）：Windows CPU 和 Linux CUDA 各六种 Office、四个 PDF 场景通过，pip check 正常，未安装原 Python 解析框架。** 修正 Linux wheel 缺失的 PDFium/GEOS 资产及 GEOS 相邻库 RUNPATH，携带原许可和匹配源归档；GPU 预览构建单独声明 GPU ORT/NVIDIA 依赖，避免 CPU/GPU ORT 混装。Rust 和 Python 解析代码逐字节未改，包内/原系统 GEOS 的四场景结构、Markdown、资产对照一致。实际原模型下载/复用在 Windows 通过；Linux 网络取消后复用已校验权重，不记作下载通过。15 项受影响检查及语法检查通过，未跑全量；产物已保存，尚未对外发布。详见 [发布验收](2026-11-02-clean-release-acceptance.md)。

**升级后完整回归及连续复用完成（2026-11-02）：pdftext 0.7.1 环境下 8/8 组、183 页完整输出全部一致；两端各连续解析 35 份、270 页，Markdown 和页数全部一致。** 129 页 Python/Rust 复用中位数 64.208/64.243 秒，基本持平，峰值 RSS 6.636/3.787 GiB、进程显存 8.045/6.699 GiB。连续五轮 Rust 结果释放后 RSS 在 2.323～2.339 GiB 波动，显存第 2～5 轮 5.684 GiB 不变，本轮未见持续增长；不推导长期无泄漏。生产包、模型和算法未改，4 项受影响脚本检查通过，原始结果已归档。详见 [升级后完整与连续验收](2026-11-02-upgraded-final-and-continuous.md)。剩余干净环境发布验收。

**服务器依赖对齐（2026-11-02）：按授权升级为 pdftext 0.7.1 / pypdfium2 5.10.1，ORT GPU 保持 1.23.0。** Python/Rust 重跑 demo1 完整 13 页，规范化完整快照与 Markdown 原始字节一致，此前第 5 页差异闭合；未改 Rust 算法，未重跑升级后整套性能。详见 [验收记录追加](2026-11-02-fixed-final-acceptance.md)。

**固定版本完整对照（2026-11-02）：183 页全部解析，7/8 组完整输出一致，demo1 第 5 页保留严格不通过。** 根因是服务器 Python 的 pdftext 0.6.3 与 Rust 移植基准 0.7.1 的旋转文字分行差异，同页 3,178 个字符字典一致，本地 0.7.1 原生 blocks 与 Rust 一致；未修改生产算法来抹平差异。129 页复用 Python/Rust 64.184/63.312 秒，RSS 峰值 6.589/3.822 GiB、进程显存 8.041/6.705 GiB；速度接近持平。Office 六种格式冒烟一致，4 项受影响脚本检查通过。详见 [固定验收记录](2026-11-02-fixed-final-acceptance.md)，此前选页一致不代表本轮所有完整文件一致。

**最新 CUDA 内存优化交付（2026-11-02）：取消优化模型磁盘缓存及旧清单兼容，原模型复用 Python 位置，Rust 初始化时内存改图并复用会话。** 142 页完整输出、10 组模型原始输出一致；129 页首次/复用 71.422/63.239 秒，RSS/进程显存峰值 3.800/6.682 GiB。最终 wheel 已安装项目 `.venv`，受影响及安装检查通过，未跑全量。详见 [记录](2026-11-02-native-memory-models.md)。下述历史阶段的未完成项和磁盘部署方式以最新专项记录为准。

**最新 PPT 二进制上下文（2026-10-31）：记录流/UserEdit 保存链、母版/三态文字样式、UTF-16 链接范围、OfficeArt 组合形状/坐标、PPT9 编号、几何表格证明和备注绑定已移植。** 320 组记录、1,630 样式、400 段落、301 解码、12 保存布局、191 形状/表格检查点、320 FOPT 和 576 编号输入通过；本地文档上下文及记录计数一致。8 项受影响测试通过，最后补充复查形状 2 项通过。仍待 PPT 图片/嵌入对象与整页资源编排、统一输出和公开安装验收；`.venv` 保持已交付 PDF/DOCX/XLSX/XLS/PPTX。详见 [PPT 二进制上下文](2026-10-31-ppt-binary-context.md)。

**最新 PPTX 公开交付（2026-10-31）：包兼容重试和公开 `RapidDoc` 已接通，最终 wheel 已安装项目 `.venv`，可直接运行 `demo/rust_pptx_demo.py`。** 280 份完整参考（279 成功一致、1 原异常保留）、26 组包修复及原始包重试通过。79 项 Office 阶段测试通过；整数精度修复后 5 项共享输出/包重试检查通过。390 个安装文件匹配 wheel；公开输入/writer/导出、禁用 Python Office/模型及 8 组 PDFium 共存通过。9 份 PPTX 两轮 36 个独立进程输出全部一致：两份本地完整示例热运行约快 2.10/2.72 倍，进程峰值更低（含导入基线）；小样例约快 5.0～9.4 倍，不推广为全部文档。详见 [PPTX 公开交付](2026-10-31-pptx-public-delivery.md)。后续 PPT → DOC；未跑全量模型 PDF/CUDA。下述“PPTX 未公开/未安装”为历史阶段。

**最新 PPTX 内部分页/导出（2026-10-31）：表格、图表、图片/OLE 公式编排、第二套公式回退、备注、整页排序和统一导出已接通。** 275 份完整对照中 274 成功的 pages/中间结构/四种导出/图片内容一致，1 份原异常保留；另有 1,151 组公式回退、240 几何顺序、2,077 字体与 800 标题查询。6 项受影响 PPTX 测试及 1 项共享 Office 输出通过。仍待原始包兼容重试、更多资源异常边界、公开安装和性能验收；其中 3 份参考仍为原 Python 归一化输入。`.venv` 未更新。见 [PPTX 资源与完整导出](2026-10-31-pptx-resources-output.md)。下述公式回退待接为前阶段状态。

**最新 PPTX 上下文与文字阶段（2026-10-31）：组合变换、占位符尺寸/字体继承、原 XYCut++ 阅读顺序、图片过滤、富文本、嵌套列表/编号及标题判定已移植。** 206 份包级/逐形状文字块、240 组几何顺序、2,077 组字体与 800 组标题参考通过（4 项受影响测试）。其中 3 包使用原 Python 归一化后的输入；第二套公式回退明确待接，没有用拼接文字代替。仍待表格/图片/图表/OLE资源编排、备注/整页输出、包兼容及公开安装验收；`.venv` 仍是已交付 PDF/DOCX/XLSX/XLS。见 [PPTX 上下文与文字](2026-10-31-pptx-context-text.md)。

**最新 XLS 公开交付（2026-10-31）：OfficeArt 图片、文本框/控件、原生与图片公式、图表/预览、表格吸收与排序、统一导出和 Python `RapidDoc` 已接通，并安装原 `.venv`。** 161 份完整对照（150 成功一致、11 原异常保留）、28 组 OfficeArt、229 组位图通过；修复原 Pillow 可读但 Rust 丢弃的 PNG 数据/尾部兼容。71 项 Office 阶段回归与最终位图边界检查通过；389 安装文件匹配 wheel，公开 IO/writer/导出、禁用 Python Office/模型及 8 组 PDFium 共存通过。两份本地 XLS 的两轮 8 个独立进程输出一致，热运行约快 2.80 / 3.45 倍，进程峰值更低（含导入基线差异）；不推断所有 XLS。可直接运行 `demo/rust_xls_demo.py`，详见 [XLS 完整交付与性能](2026-10-31-xls-resources-and-delivery.md)。下一步 PPTX → PPT → DOC；未跑全量模型 PDF/CUDA。

**最新 XLS 工作簿/单元格阶段（2026-10-31）：原生 BIFF5/8、数字显示、跨段富文本、缓存公式、链接、合并与坏目录恢复已接共享表格投影，独立图表页的 BRAI 引用选择已接通。** 101 份工作簿/105 个工作表检查点、89 组完整无待接资源分页及 11 组原异常对照通过；基础参考含 5,044 组数字、310 文本格式、1,063 跨段字符串、25,246 代码页输入、64 TXO 与 720 图表引用，另穷举 1,048,576 双字节输入核对。6 个受影响测试目标共 11 项通过，最后 Unicode 条件补充复查 2 项通过。仍待 OfficeArt 图片/文本框/控件、嵌入图表/公式资源绑定、公开 XLS 与安装验收；cell-only 入口对未接资源明确拒绝。`.venv` 保持已交付 PDF/DOCX/XLSX，未跑全量 PDF/CUDA 或 XLS 性能基准。详见 [XLS 读取与投影](2026-10-31-xls-native-cells.md)。

**最新 XLSX 公开交付（2026-10-31）：包规范化/失败重试、统一输出与 Python `RapidDoc` 接口已完成并安装原 `.venv`。** 231 组完整导出（200 原成功一致、30 原错误保留、1 原 OLE 故障恢复）、86 组包边界通过；65 项 Office 阶段回归及优化后 3 项受影响检查通过。定位并消除长图片 URI 的正则捕获开销，3,574 组清理语义对照不变。最终 8 份 XLSX 两轮 32 个独立进程全部输出一致，Rust 热运行约快 1.68～9.76 倍，进程峰值更低（含导入基线差异）。389 个安装文件匹配 wheel，可直接运行 `demo/rust_xlsx_demo.py`；详见 [XLSX 交付与性能](2026-10-31-xlsx-public-delivery.md)。下一步 XLS → PPTX → PPT → DOC；未跑全量 PDF/CUDA。

**最新 XLSX 资源与分页接入（2026-10-31）：图片、图表、OMML/MTEF/OLE、DISPIMG 已接入 Rust 内部分页主链。** 表内资源参与共享投影，独立资源保留原排序/预览抑制；图表按原引用选区投影。95 份工作簿/190 组完整结果对照和 8 个受影响测试目标（9 项测试）通过，图片按 MIME/尺寸/完整像素核对。仍待非规范包/备用图片来源/损坏边界兼容、统一导出与公开 XLSX API、安装及效果/速度/内存验收。`.venv` 未更新，无全量 PDF/CUDA 或性能结论。详见 [XLSX 资源编排](2026-10-31-xlsx-resource-integration.md)。

**最新 XLSX 工作簿/单元格接入（2026-10-31）：原生 XLSX 读取已接共享表格投影。** 缓存值、日期/持续时间、共享/内联富文本、样式、合并格链接和默认格规则已接通；50 份工作簿（46 成功、4 预期错误）/70 个工作表与 123 组完整 cell-only 文件到 pages 对比通过，受影响 3 项及最终 XLSX 复查通过。仍待图片/cell-image/图表/OMML/MTEF/OLE、包兼容重试及公开 XLSX API；内部 cell-only 入口对未接资源明确拒绝，避免静默丢失。未重装 `.venv`，无全量 PDF/CUDA 或 XLSX 性能基准。详见 [XLSX 工作簿与单元格](2026-10-31-xlsx-cells-and-workbook.md)。

**最新 DOCX 验收与 XLSX 启动（2026-10-31）：DOCX 两轮 36 个独立进程的效果/速度/内存对比通过，9 份输入全部输出内容一致。** 非空本地样例 Rust 约快 9.6～19.1 倍；25 次重复正文压力样例约 49.1 倍。进程峰值更低，但包含 Python/Rust 导入基线差异，详见 [DOCX 性能验收](2026-10-31-docx-performance-acceptance.md)。已进入 XLSX：共用表格发现/自动 gap/合并格/物化/语义子集过滤/选区/HTML 已移植，83 sheet、415 组查询和 80 组选区通过；工作簿加载、资源与公开 XLSX 接口仍待，见 [电子表格投影](2026-10-31-spreadsheet-projection.md)。当前 `.venv` 继续使用已交付 DOCX/PDF 包；没有为这个内部阶段重装。后续顺序 XLSX → XLS → PPTX → PPT → DOC。

**最新 DOCX 公开接入（2026-10-31）：原生解析已接通 Office 中间结构、四种导出、图片资产和 `rapid_doc_rust.RapidDoc`。** 353 份原 Python 参考（352 成功、1 预期损坏）完成中间结构/四种导出/图片内容对比；PNG 压缩字节及其哈希文件名差异采用已验证图片内容映射，不宣称资产字节完全相同。Office 阶段 60 项测试通过，release wheel 已安装项目 `.venv`，389 个安装文件与 wheel 一致。23 份安装参考及输入/writer/data_uri/无模型调用、Office 后 PDF 读取和 8 组 PDFium 共存通过。可直接运行 `demo/rust_docx_demo.py`。其他五种 Office 格式仍未接入；未做全量 PDF/CUDA 或性能/内存验收。详见 [DOCX 原生导出与 Python 接入](2026-10-31-docx-public-output.md)。

**最新 DOCX 附注/嵌入样式闭环（2026-10-31）：脚注/尾注/批注引用、嵌入 Mammoth 样式映射、整篇表格预解析/候选对齐和逐表回退已接通。** 新增 1,120 组样式匹配查询和 81 份完整 DOCX 对比，保留跨表编号、隐藏资源延迟处理、附注本地关系及失败表格诊断/继续正文。连同此前 272 份正文参考，阶段 59 项测试通过；最后附注容器补充仅复查受影响 3 项，均通过。共 353 份文档参考/362 逻辑页，记录见 `build/office-docx-resources-receipt.json`。DOCX 主要剩统一导出和公开 Office/Python API；少见图片编码及更广泛异常文档仍待验收。未替换 `.venv`，未跑全量 PDF/CUDA 或性能基准。详见 [附注与嵌入样式](2026-10-31-docx-notes-and-embedded-styles.md)。

**最新 DOCX 图表闭环（2026-10-31）：16 种原 OOXML plot 的数据表、嵌入 XLSX 公式缓存/日期/持续时间、缓存及普通工作表回退已接入正文。** 97 组图表查询及 208 组日期转换通过；完整正文扩展为 272 份文档（271 成功、1 损坏）、281 逻辑页/28 图表块/157 图片块。修复真实文档暴露的表格空标签序列化差异，保留坏单元格异常到图表边界及后续正文。阶段 Office crate **57 组测试通过**，记录见 `build/office-docx-charts-receipt.json`。DOCX 仍剩附注/嵌入样式映射等资源、统一导出和公开 Office/Python API；不是完整 XLSX 转换器。未替换 `.venv`，无全量 PDF/CUDA 或性能基准。详见 [图表与工作簿](2026-10-31-docx-charts-and-workbooks.md)。

**最新 DOCX 图片闭环（2026-10-31）：正文图片加载、公式预览排除、引用去重、位图转码、SVG 和浮动/行内输出顺序已接通。** 200 组位图像素与 62 组同版本 resvg SVG 参考通过；完整正文扩展为 254 份文档（253 成功、1 损坏）、261 逻辑页/148 图片块。修复 JPEG TIFF、灰度透明/调色板、方向与 ICO 载体模式差异，普通照片不继承元文件画布限制。阶段 Office crate 56 组回归通过，最后补充仅复查受影响测试。DOCX 主要剩图表、附注等资源、统一导出和公开 Python 接入；少见图片编码仍需验收。未替换 `.venv`，无全量 PDF/CUDA 或性能基准。详见 [正文图片与 SVG](2026-10-31-docx-body-images-and-svg.md)。

**最新 DOCX 正文闭环（2026-10-31）：正文段落/标题/题注、列表、目录/锚点、公式/表格、分节、文本框和页眉页脚已接成 Rust 实际输出流程。** 原 Python 完整 convert 的 97 份文档（96 成功、1 损坏输入）、104 个逻辑页与 450 组富文本参考通过；覆盖本地公式、分节、文本框示例及 4 份 MTEF 文档。任意精度编号与原样式/链接/分页状态保留。Office crate 阶段54组回归通过。正文图片转码、图表、剩余附注资源及公开 Office/Python API 仍待；`.venv` 未替换，也未跑全量 PDF/CUDA 或性能基准。详见 [DOCX 正文、目录和页眉页脚](2026-10-31-docx-body-lists-toc-margins.md)。

**最新 Office 容错闭环（2026-10-31）：损坏矢量图已按原 Python 生成标准占位并接通 DOCX 表格输出；Source::open 已接入 CRC/可达关系图、坏媒体降级、失效关系移除和内容类型修复。** 使用共享 ZIP 覆盖层及流式 CRC 验证，未增加完整包副本。35 组占位像素、259 组回退策略、32 份包级修复参考和 298 份 DOCX/596 次图片放置通过，阶段 Office crate 52 组回归通过。完整 DOCX 正文编排、普通正文位图/SVG转码、其他格式转换器及公开 Office API 仍待；未替换 `.venv` 中的 rapid_doc_rust。见 [图片回退与包级修复](2026-10-31-office-image-fallback-and-docx-repair.md)。

**最新 Office EMF+ 闭环（2026-10-31）：EMF+ Only 已接通对象/跨记录重组、状态/GetDC、实际路径/图片/文字绘制、PNG/SVG 与 DOCX 表格图片。** 255 份原 Python 文件参考（212 成功、43 异常）、424 组最终输出，以及 251 份 DOCX/502 次图片放置通过；阶段 Office crate 49 组回归通过，新增样例后仅复查受影响测试。JPEG 改用发布版 libjpeg-turbo，消除纯 Rust 解码器已测色阶差异，本机构建启用 SIMD。完整 DOCX/其他 Office 转换器、公开 API、损坏图占位和独立真实矢量验收仍待；未替换 `.venv` 的 rapid_doc_rust。PNG 压缩字节及极近预算阈值的差异已明确记录。详见 [EMF+ 与 DOCX 渲染](2026-10-31-emfplus-docx-rendering.md)。

**最新 Office 文字闭环（2026-10-30）：普通GDI文字已从WMF/EMF接到实际绘制、SVG及DOCX表格图片。** 本轮连续补齐字体选择/度量/字形、字符集、锚点、旋转、多行、背景/装饰、裁剪和替代诊断。432组字体查询、72组文字层、262,144组DBCS输入、380组最终图像及39份DOCX共78次放置参考通过，Office crate的48组测试通过。当前验收对应本机Windows/BASIC布局；FreeType版本差异与跨平台/RAQM风险已记录。EMF+、损坏图策略、独立真实矢量验收及完整Office公开入口仍待完成，`.venv`未更新。详见[文字与DOCX渲染链路](2026-10-30-metafile-text-and-docx-rendering.md)。

**最新 Office 输出与接线进展（2026-10-30）：无文字/EMF+的GDI图像已贯通DOCX part→Rust渲染→SVG/PNG→表格HTML。** 接入自适应超采样、Lanczos/锐化、原SVG结构/复杂合成fallback、同part复用与诊断；233组最终图像/SVG和15份DOCX共30次图片放置参考通过，Office crate的43组测试通过。文字/字体、EMF+、损坏图占位策略与完整DOCX公开入口仍未完成，`.venv`未更新。PNG压缩字节不同但解码像素/元数据一致。详见[输出与DOCX资源接入](2026-10-30-metafile-output-and-docx-resources.md)。后续以完整矢量资源阶段连续推进，不按内部小函数停下。

**最新 Office 混合绘制进展（2026-10-30）：路径填充、描边、hatch与最终裁剪已接入单次栅格化。** 采用原Clipper 6.4.2内核与Pillow扫描行规则；492组文件渲染查询中466组像素和26组异常行为一致，另有305组多边形与填充/描边/裁剪内核参考通过，Office crate的40组测试通过。自适应超采样后的缩小/锐化、文字、EMF+和SVG/资源输出尚待完成，DOCX公开入口与 `.venv` 未更新。详见[混合绘制记录](2026-10-30-metafile-path-raster.md)。

**最新 Office 位图绘制进展（2026-10-30）：DIB解码、最近邻/双三次采样、source-over和按文档缓存已实现。** 834组解码、428组变换、100组放置链路、65,536对alpha和43份构造纯位图WMF/EMF的像素参考通过；Office crate的36组测试通过，最后边界调整仅重跑受影响8组。混合路径/裁剪、文字、EMF+与SVG/资源输出仍待接通，DOCX公开入口和 `.venv` 未更新。详见[位图绘制记录](2026-10-30-metafile-bitmap-rendering.md)。

**最新 Office 位图进展（2026-10-30）：WMF/EMF 的六种位图记录已接入回放。** DIB 校验、预乘透明度、源裁剪/镜像和14种逻辑合成内核已移植；227份文件896个逐记录状态、75组载荷、9组裁剪与逐像素合成参考通过，Office crate 的31组测试通过。通用解码、仿射采样、最终裁剪/绘制、文字和EMF+仍待完成，尚不能交付完整矢量图或默认DOCX入口，`.venv` 未更新。详见[位图阶段记录](2026-10-30-metafile-bitmap-replay.md)。下文为之前阶段记录。

**最新 Office 矢量进展（2026-10-30）：WMF/EMF 文件校验、几何及 GDI 路径回放层已移植。** 166份构造文件、760个逐记录状态、40组几何与7组画布参考通过，Office crate的28组测试通过。字体/文字、位图合成、EMF+绘图和SVG/栅格后端仍待接入，尚未完成渲染像素验收；DOCX默认入口和 `.venv` 未切换。详见[矢量回放记录](2026-10-30-metafile-geometry-replay.md)。

**最新 Office MTEF 进展（2026-10-30）：v3/v5、Equation Native/OLE、WMF/GIF 内公式解码及 DOCX 按 part 的资源接入已实现。** 2,952组MTEF/封装、254组图片、11包16表格和10次预览查询通过，crate的25组测试通过。成功二进制公式语料主要来自构造器，真实MathType OLE仍需后续语料验证。普通WMF/EMF图像渲染、完整正文与公开DOCX入口尚未完成，未更新 `.venv`。详见[MTEF与资源接入记录](2026-10-30-office-mtef-and-docx-embedded.md)。

**最新 Office 公式进展（2026-10-30）：原生 OMML、Equation XML 和含公式单元格重建已接入内部表格流程。** 192组OMML（含46个真实公式）、2,654字符映射、45组公式提取/渲染、13组Equation XML及49组链接参考通过，受影响crate的22组测试通过。明确修复公式原文直接拼HTML导致`<`被当标签的问题；MTEF、矢量、完整正文及公开入口尚未完成，未更新 `.venv`。详见[公式阶段记录](2026-10-30-office-omml-and-docx-formulas.md)。

**最新 Office 进展（2026-10-30）：共享底座、DOCX 上下文及普通表格内容路径已推进。** 普通单元格 HTML、列表/嵌套表格、图片原字节读取和 colspan 校正已移植；79组 HTML 与 Mammoth 字节一致，另检查异常/待接公式与12组规范化边界，受影响 crate 的17组测试通过。完整公式/矢量/正文及公开 Office API 尚未贯通，原 `.venv` 未更新。详见[表格内容实施记录](2026-10-30-docx-table-content-html.md)；下文“Office尚未开始”等为历史状态。

**Office Python 准备（2026-10-30）：已选择性同步 DocVortex 0.6.8 的 XLSX #37、缓存、DOCX 复用及 XLS 视图优化。** 保留本地 #36、PDF 与共享数值核，未覆盖整个 vendor 包或升级运行库。Office 全目录166项测试通过，六格式20份同步前后完整输出/资产字节严格一致；新增图片回归覆盖 #37 修复。电子表格 Rust 原生批处理保留到 Rust Office 阶段直接整合，当前 Python 仍使用原参考算法。用户可用原 `.venv` 和 `from rapid_doc import RapidDoc` 测试本步；Rust Office尚未集成。详见[本步交付记录](2026-10-30-python-office-068-update.md)。

**最新结构整理（2026-10-30）：轻量整理已完成并安装原 `.venv`。** 256个原路径按领域归组，规则规范入口为 `processing`，页面组装为 `page_assembly`；保留原Rust路径/API。推理层按领域存放，Python绑定根文件由2570行降为173行；446个Rust文件、104个绑定项和20份JSON资源核对无算法/常量/规则变更。12份101页与原Python及整理前Rust输出严格一致，PNG编码字节也保持一致；13项Rust、18项Python单测及pip check通过，148个安装文件匹配、17个模型SHA未变，119模块兼容清单已更新。速度/内存未重新做基准，未运行CUDA，Office尚未开始。详见[整理与验证](2026-10-30-pdf-structure-cleanup.md)和[当前目录导航](../../rust/README.md)。

本轮最终8组PDFium实际同进程共存验证也全部通过；阶段汇总为`build/pdf-structure-final-receipt-20261030.json`。

**最新复验（2026-10-30）：已关闭首轮 PDF 的执行失败、标点/空格、空文本 JSON 及置信度差异。** 正式入口复用共享 Rust，修正正文/表格 OCR 默认阈值区别、跨页分辨率分组、NumPy 置信度归约及多边形重叠过滤。最终 wheel 已安装原 `.venv`，12 份去重 PDF 共 101 页的 Markdown、两个 JSON、图片名称/像素全部与冻结 Python 基线严格一致。6 项 Rust、18 项安装包 Python 单测、8 组 PDFium 共存及 pip check 通过；148 个安装文件匹配、17 个模型 SHA 未变。效果回归已通过，速度/峰值内存未重测，未运行 CUDA。下一步先轻量整理 PDF 领域/契约及排版，再集成 Office；不更换算法或重划全部 crate。详见[修复和完整效果复验](2026-10-30-pdf-parity-fixes.md)。下文保留各历史阶段结论，以本段为当前状态。

**最新修复（2026-10-30）：正式PDF入口默认取消迁移时额外增加的工作量、字符/对象/区域、像素、JSON及表格数量预算；显式限额与格式/溢出检查保留。** 同时按原Python后写覆盖规则修复UNET重叠逻辑格，demo2现可完整解析6页。最终包已安装原`.venv`，3份11页与原Python的Markdown、两个JSON、图片名称/像素严格一致；7项Rust、5项Python容量政策、18项印章/PDFium回归及pip check通过，148个安装文件核对一致、17个模型SHA未变。原验收的标点/空格、空文本JSON及置信度差异仍待处理，不把本轮记为全部PDF验收通过；未跑全量101页或CUDA。详见[资源政策修复](2026-10-30-pdf-resource-policy.md)。

**最新验收（2026-10-30）：第5项PDF效果/速度/峰值内存验收执行结束，结论为未通过，尚不能开始结构调整。** 12份去重PDF共101页全部尝试：原Python成功12份，Rust成功11份；成功案例中9份Markdown/内容列表/图片完全一致，8份中间JSON也一致。发现公式固定累计预处理预算导致demo2失败、05的9个标点/空格文本字段差异、国标JSON少两个空文本条目及small_ocr两处置信度差异。4份反序复测输出稳定但性能波动明显，不能承诺Rust稳定更快或更省内存。18项印章/PDFium回归及pip check通过，148个安装文件和模型SHA核对通过，未改生产算法/模型、未重新构建。下一步先修复本轮问题，仅重测受影响案例，再调整结构及接入Office。详见[验收记录](2026-10-30-pdf-delivery-acceptance.md)及[逐文件结果](2026-10-30-pdf-delivery-results.md)。下文保留历史交付记录。

**最新交付（2026-10-30）：第 4 项印章识别已完成共享 Rust 实现、验收并安装到原 `.venv`，按约定停下供用户测试。** 默认开启，按版面印章候选惰性加载专用检测器，保留多边形/曲线裁剪和原 PP-OCRv6 识别回填。105 组差分通过，含 12 组逐像素裁剪及真实模型；4 页 `含印章.pdf` 的印章文字与原 Python 一致，开关和安装后默认 API 验证通过。148 个安装包文件逐项核对，安装后 8 组 PDFium 共存及 pip check 通过。未改模型、未跑全量或 CUDA；第 5 项完整输出、速度、峰值内存验收仍待开始。详见 [印章识别交付](2026-10-30-seal-recognition.md)。下文为各轮历史记录，以本段安装状态为准。

**最新交付（2026-10-29）：第3项 PDFium 与 pypdfium2 同进程共存已完成本机验证并安装到原 `.venv`，按约定待用户测试。** 去掉过早导入限制，默认保留独立 DLL，显式/环境变量外部库使用私有哈希副本。八组导入顺序、raw、路径、并发、销毁重建及中文渲染验证通过；论文页和扫描页的正式 Python/Rust 混合运行各自与独立输出一致，Python活动句柄不受影响。145个安装文件核对、pip check、6项隔离单测及进度专项通过。未改算法、模型、第三方PDFium或Rust二进制，无Cargo构建。当前wheel SHA256为 `e15a83c33d3ffc99b129c8e9e977e86ba6ea3fc402736bed34c2aca7853d4b0c`，旧包备份已保存。详见 [第3项交付](2026-10-29-pdfium-coexistence.md)。第4项印章、第5项最终对比仍待开始；下方为历史节点。

**最新资源诊断与安装（2026-10-29）：已同步 `.venv` / build/wheels，速度目标仍未达到。** 新增只读CPU/内存/页面读入采样；实测系统提交比例90%～97%、最低可用物理内存1.31GiB，说明资源压力存在，未唯一解释全部慢例。原生共享CPU arena已实现并验证并发、缓存隔离、最后模型释放及重建；三篇反向复测私有提交3.48→2.28/2.29GiB，但平均耗时慢约2.6%，因此默认关闭共享池、批量仍1。共享实验七篇60页/92图输出参考一致，299.225秒、峰值2.593GiB，不用不同状态旧速度作收益声明。安装后144文件一致、pip check、共享池生命周期/并发和43组规则合同通过。当前wheel SHA256为2865f56ec85f359001cb16b0e7b51c80d9204b5916457e6998b88e67a04e57c3，直接运行原demo，无需调参；旧包已备份。见[完整证据与边界](2026-10-29-resource-telemetry-and-shared-arena.md)。

**最新安装交付（2026-10-29）：已按用户要求将候选新版安装进项目 `.venv`，并同步 `build/wheels`。** 直接在IDE使用现有项目解释器运行demo即可，不再需要PYTHONPATH或开发目录启动命令。默认公式批量仍1：批量2未证明稳定最优，不强制用户手动配置，也不擅自设为默认。新版含35处固定正则缓存、显式批量>1时两页分组及原序恢复。144个非pyc包文件与wheel一致，pip check、安装后43组规则合同与公式分组/重复输入/回填测试通过。wheel SHA256为c5b85c94260d30d9643bfc7989cb2e801566b41b72f0c68f85ab4c5b2442c204；原包保留在build/pre-click-run-install-*，安装状态见build/click-run-install-20261029.json。没有声称默认整篇已稳定比Python快。

**最新速度候选（2026-10-29）：已实现固定正则缓存和相邻两页公式分组，未安装。** 35处固定正则复用；显式batch_num>1时按裁图比例分组、恢复原输出顺序，聚合触发预算回退逐页，默认仍1。74真实公式裁图像素一致，FP32末位差异没有改变任何原token或长度；43组规则合同及分组顺序/重复输入测试通过。实际PDF ABBA12次及Python/候选补充6次输出参考一致。最后三篇Python140.958秒、候选101.161秒，但先前存在分组慢例，不能声称稳定提升；安装保留原包。详见[修改、全部正反结果及证据](2026-10-29-regex-and-bounded-formula-grouping.md)。

**最新原生调度复测（2026-10-29）：目标仍未达到，不安装实验包。** 已实现可选 CPU 自旋/动态块/推理结束停止自旋，并将全部策略纳入原生及 Python 会话缓存身份；默认不变，源码公式批量已恢复 1。停止自旋 ABBA 三篇合计 110.004→86.889→91.750→85.460 秒，末次原策略更快，不能认定稳定收益。12 次整篇/80 页输出及像素参考一致，原生缓存隔离、真实公式、named、OCR/CTC 受影响测试通过。公式输入 130 次仅 1 次重复，未加入低收益缓存。当前安装仍为 borrowed-layout-masks 稳定包。见 [完整原生实验与不利结果](2026-10-29-native-cpu-session-experiments.md)。

**最新瓶颈分析（2026-10-29）：安装不变，未声称已修复。** 公式父阶段48.5%、版面23.3%、表格10.4%；优先级转为真实多模型推理/线程调度诊断。固定张量试验关闭自旋明显降低CPU时间，但有墙钟慢例，动态负载亦不稳定，不改默认；单公式算子跟踪已保存。附件与上次相同，无整套PDF重复跑。见 [耗时拆分、全部调度对照与可执行优化路线](2026-10-29-runtime-bottleneck-analysis.md)。

**最新诊断（2026-10-29，13:44）：性能问题仍未解决，安装保持同一回退包。** 用户七篇343.374秒，较历史155.352秒约2.21倍；144安装文件一致，7篇/92图参考一致。绕过Rust与PDF，Python直接执行固定三张量也由此前中位1.091秒变为当前2.024秒，自动线程重复耗时显著波动，固定4线程反而更慢。仅确认原生推理执行性能也在变化，具体根因未定，未再改参数或归咎用户机器。磁盘当前未满。见 [本次诊断与证据边界](2026-10-29-reported-1344-inference-variability.md)。

**当前安装（2026-10-29，13:32实测后回退）：恢复版面掩码包，默认公式逐张1。** 用户七篇203.609秒，相对此前155.352秒慢31.1%；输出七篇/92图核对仍一致。当前同条件三篇旧包115.257秒、批量2包117.928秒，未复现稳定收益，撤回本轮默认2交付，144安装文件校验/pip check通过。旧包当前也明显变慢，具体根因未确认，回退不保证恢复历史速度。源码/dev包仍是实验，当前安装不是实验包。见 [本轮完整证据与未解决问题](2026-10-29-reported-1332-slowdown.md)。

**已撤回的安装（2026-10-29，公式批量2与表格借用）：已同步 .venv / build/wheels，旧包保留。** 同一FP32模型用小批量执行，显式batch_num=1/4可配置，固定单张模型保持1；表格只读缓存共享、借用反序列化并减少候选复制。两轮三篇平均92.510→89.003秒（少3.8%），03纯公式推理少8.6%；表格组件两轮方向不一致，仍不能认定已稳定提速。峰值工作集约增加0.14GiB，默认未采用内存更高的4。六组原参考/28批量图、最终七篇60页92图片、119模块审计、144安装文件校验及pip check通过，保留small_ocr既有两处分数差异。未跑全量/CUDA。详见 [本轮结果及全部不利数据](2026-10-29-formula-batching-table-borrowing.md)。

**上一阶段安装（2026-10-29，版面掩码借用优化）：已同步 .venv / build/wheels，旧包保留。** 去掉版面输出整数深拷贝及整块FP32转换，保留原算法、精度和完整验证；CPU arena关闭实验变慢，已撤回。三篇同条件新Rust两轮平均79.490秒、旧Rust79.861秒、Python本轮94.460秒：对Python耗时少15.85%，对旧Rust整体基本持平（0.46%），不把后者夸大。03两轮均值仍比旧包慢，内存峰值也未明显降低。24组版面、原named/布局参考、七篇60页及92图片通过，119模块审计、144个安装文件校验及pip check通过；没有全量/CUDA。见 [完整结果及限制](2026-10-29-borrowed-layout-masks.md)、[arena不利实验](2026-10-29-runtime-cpu-arena.md)。

**历史回退（2026-10-29 10:56）：性能回退，已恢复用户 09:31 对应的旧正确 wheel。** 用户新日志七篇 248.523 秒，相比 143.291 秒增加约 73.4%；上一轮财报整篇未通过提速验证时仍交付，判定不稳妥。`.venv` 和 `build/wheels` 已恢复 `build/pre-layout-render-owned-wheels-20261029/`，新包隔离于 `build/quarantined-native-table-allocation-20261029/`。源码、`build/dev-wheels` 和开发 SDK 仍含后续实验，**不是当前稳定安装包**。回退后 small_ocr 23.148 秒、实验包 21.809 秒，均未恢复历史约 17 秒，根因未确认；先定位整篇回退，暂停继续覆盖安装，不以组件结果代替整篇验收。见 [回退记录](2026-10-29-performance-regression-rollback.md)。

2026-10-29 原生表格分配提速：候选数组临时移交并在所有退出路径恢复，生成结果数组移出；网格/文字借用 Value 反序列化；JSON 字节预算计数避免临时 String，原校验/Work/候选顺序不变。四组候选/引擎/准备/完整 HTML 参考及四份 23 页/29 图片输出通过，保留 small_ocr 两处旧差异。固定输入交替组件中位恢复 -5.4%、渲染 -20.0%、候选引擎 -9.3%，存在较大波动；四份单次整篇并未一致加速，额外交替财报中位 15.994→16.841 秒，不能宣称财报整体变快。已安装现有 `.venv`；无全量/CUDA/峰值内存测试。完整有利与不利记录见 [本轮记录](2026-10-29-native-table-allocation.md)。

2026-10-29 渲染像素/表格恢复提速：当前页版面直接复用渲染 RGB，处理后立即释放；表格恢复使用拥有的请求、移出结果，减少整树复制，原校验/Work/错误行为保留。初轮多余 B/R 交换被整篇坐标比较检出，未安装；已修正并加入彩色模型对照。最终原像素、模型输出及七份 60 页/92 图片严格通过，保留 small_ocr 两处既有分数差异，已安装现有 `.venv`。本轮 139.841 秒，前用户日志 143.291 秒（单次少约 2.4%，非受控普遍结论，密集表格仍有回升）；无全量/CUDA/峰值内存基准。见 [本轮记录](2026-10-29-layout-render-owned-tables.md)。

2026-10-29 表格图片与 OCR 输出提速：表格尺寸查询不再重新编码 PNG，无正交旋转时借用原 PNG；CTC 在保留全部输入/输出检查后直接解码 ORT 概率切片，去掉整块复制及重复有限值扫描。原像素/原批次张量及 token/分数、六组安装测试通过；三份 14 页/17 图片输出与修改前一致，原有 small_ocr 两处分数差异保留。相邻单次 old→new：small_ocr 18.398→17.203 秒、密集表格 3.693→3.625 秒、研究报告 14.312→10.261 秒；存在未修改阶段的运行波动，不承诺固定比例。已安装现有 `.venv`，无全量/CUDA。整页 PNG/表格遮罩及原生几何 JSON 复制仍待，见 [本轮记录](2026-10-29-borrowed-outputs.md)。

2026-10-29 识别裁图提速：正文/表格行改用拥有BGR像素的共享Rust识别入口，省掉内部PNG往返和副本；透视裁图公共坐标计算复用；裁图与V3浮点缩放增加CPU/OS运行时FMA分派，保留通用回退。64裁图/50版面硬件-通用像素一致，原张量/CTC/布局/实际CPU链及七篇60页输出通过，92图片与原Python像素一致，small_ocr两处旧score差异仍保留。组件FMA内核约2.15倍速度、相邻small_ocr裁图2.915→1.460秒，整篇19.965→18.487秒；03相邻整篇55.642→53.704秒。首轮七篇179.706秒受03公式阶段异常影响，不用后续单篇替换，也不宣称整体快很多。完整数据及边界见 [本轮记录](2026-10-29-recognition-pixels.md)。

2026-10-29 检测/表格通用提速：修掉每单元格筛选前的整页几何映射深拷贝；正式检测改用内存RGB，去掉遮罩补白后的PNG往返及来源字节复制。75组像素/张量一致，表格及实际CPU链通过，七篇60页与修改前Rust完整输出相同，原Python文本/92图片一致，small_ocr两处旧score差异显式保留。交替组件基准恢复加渲染耗时下降34.7%、单独渲染44.1%；单次密集表格整篇3.463秒，不承诺固定整篇比例。增加检测和表格细分计时，剩余识别裁图PNG及结构/JSON热点见 [本轮记录](2026-10-29-detector-pixels-and-table-copy.md)。

2026-10-29 性能回退修正：原 Python 公式默认为自动线程，之前 Rust 误设为 4，历史“已对齐”声明有误。已更正，并将共享 Rust 管线改成窗口内版面→公式→OCR 检测的阶段顺序。七篇 60 页本轮合计 156.300 秒，用户旧 Rust 日志 251.827 秒、Python 168.335 秒；不构成反复同条件正式基准，密集表格等仍慢于 Python。七篇完整输出与修改前 Rust 一致；原 Python 文本和 92 图片一致，small_ocr 两处旧置信度差异显式保留。详见 [定位及验证](2026-10-29-formula-threads-and-phase-order.md)。

2026-10-29 通用开销优化：正式公式改为内存 RGB 传递、公开 PNG 入口单次解码，表格单元格渲染借用整页映射，共用缩放改善访问顺序，增加公式细分计时。38 页三类输出与原 Python 一致，64 PNG 逐字节一致；交替组件测试公式预处理/表格渲染耗时中位数下降约 17%/13%，不宣称整篇普遍提速。详见 [优化验证](2026-10-29-native-overhead-optimization.md)。

2026-10-28 用户验收修复：03 整篇 OCR 等宽高比跨批排序报错已用原 NumPy native 排序后端修复，正式默认线程按原 Python 对齐，正文裁图重复解码消除，模型/页/裁图/阶段计时日志补齐。四文件共 38 页的 Markdown、middle JSON、content list 与用户原 Python 完全一致，64 图片像素一致；新 Rust 合计 127.214 s，用户 Python 日志 126.357 s，尚不宣称整体性能优势。最终包已安装现有 .venv，详情见 [用户实测与修复](2026-10-28-reported-speed-and-ocr.md)。第 3–5 项仍按原顺序，未启动 CUDA 或增强阅读序。

2026-10-28 最新交付：第 2 项的正式 PDF API、自动模型加载及共享 Rust 核心已构建安装到现有 `.venv`，等待用户测试验收。分类、多边形后处理、公式/跨页正文 OCR/表格双模型/资产/后处理编排均在 Rust，Python 正式入口仅做资源与 API 适配；低层预览接口保留。正式解析固定字符计数门槛已取消，独立资源预算可配置。demo1 完整 13 页/51,140 字符/18 资产通过，全文 Markdown 与此前 Rust 安装版本一致。最终安装包 5 默认案例（6 页）、4 原生案例（5 页）与原 Python 严格对照通过；36 分类 PDF、100,058 字符采样、原生宿主及 119 模块审计/进度回归通过。详见 [Rust 核心交付记录](2026-10-28-native-api-core.md)。第 3–5 项仍为 PDFium 共存、印章、质量/速度/峰值内存对比，不提前启动增强阅读顺序/CUDA。下方为历史检查点。

2026-10-28 当前节点：原 PDF 宿主路由已接入 Rust 原生全文，`parse_pdf_models` 支持 `off/rule/rule_plus_s`、`native_only` 和内置原生区域表格恢复，贯通模型回退、OCR、公式、段落、跨页表格、Markdown、content_list 和图片资产。已完成 release 构建安装（8 分 51 秒）；本阶段安装对照、输出位置和限制统一见[宿主接入验收](2026-10-28-native-pdf-host-integration.md)。跨页 OCR 批次、BGR 渲染和边界坐标类型均按原 Python 保留，未改冻结参考或降低算法精度。`docvortex_mode` 默认仍为 `off`，原生模式显式传 `rule`；增强阅读顺序、CUDA 与同条件性能测量后置，不能据局部样本宣称全部 PDF 等价或普遍更快/更省内存。下方“宿主/Markdown 待接”为历史检查点。

2026-10-28 当前节点：PDF 文件级采集和完整 PdfModel 入口已实现，已统一 release 构建（8 分 41 秒）安装并通过验收：4 文件/20 页、两个示例完整 23 页/13 表格，以及受影响回归均通过。新增 `analyze_native_pdf` 直接接收 PDF；保留原可见文字/签名/空格/字形粗体及跨页采集、完整全文和末尾清洗顺序。76 个实际可见页面、108 字形、418 清洗、4 文件/20 页采集分析，以及实际财报的 5 个整页分数和 3 个大候选集合已建立对照；文件预算独立于旧诊断预算。剩余明确为 RapidDoc 默认模型路由/适配器接入、Markdown 整体验收及性能测量，详见[文件级节点](2026-10-28-native-pdf-file-integration.md)。以下为历史记录。

2026-10-28 最新节点：全文 spacing、链接/样式/上下标、诊断和表格媒体接线已构建安装通过。915 空格规则、1,645 行内编排、35 原始页源链、6 媒体链/3 解码 PNG 与 3 页实时 PDFium 通过；`finalize_native_document` 可从已采集原始页源/几何输出最终逐页块。剩余文件级证据采集/默认 PDF 路由与用户整篇质量、耗时和峰值内存验收。见[完整接续](2026-10-28-document-inline-integration.md)。下方“全文行内/媒体待接”为历史检查点；无全量/CUDA或整篇/性能提升声明。

2026-10-28 最新安装节点：完整最终页 release/安装通过，211 页级用例、35 原始页源链、1,458 视觉/排序合同及 3 页实时 PDFium 完整输出。`finalize_output=True` 可返回完整排序、归一化的 `model_list`；源码另通过 298 XYCut 和 7 项相关测试。见[完整最终页](2026-10-28-final-page-output.md)。剩余为外层 spacing/行内证据及媒体接线、默认 PDF 和整篇验收；下方进行中记录已由本条覆盖。无全量/CUDA及全文/性能提升声明。

2026-10-28：完整最终页（图注带/图例、视觉区域绑定、全部分组排序、tight 与归一化）已接通，211 页/35 原始页源链源码通过；统一安装验收进行中。新增 `finalize_output=True` 返回 `model_list`。精确剩余为外层全文行内收尾、默认 PDF 文件入口及整篇质量/耗时/内存验收。见[完整对应表](2026-10-28-final-page-output.md)和主链最新记录；以下为历史检查点。

2026-10-28 最终输出内核安装验收：release 构建 7 分 46 秒，已安装至 build/rust-wheel-env。1,248 输出/紧框/行框与文献/窄图形栏序守护对照通过，严格类型和输入所有权保留；118 模块来源 hash 与默认门禁通过。日志 build/native-final-output-wheel-build.txt、build/native-final-output-install.txt、build/native-final-output-compatibility-install.txt。成员/图注/列表新入口再次回归状态见 build/native-final-output-page-blocks-install.txt。仍未在最终页提前接入输出内核，下一步为完整视觉注释与分组排序/行内证据，然后默认 PDF 和整篇质量/耗时/内存验收。无全量或 CUDA，无全文精度/性能提升声明。

2026-10-28 后续输出内核：聚合后 tight、归一化块/行及文献/窄图形栏序守护已按现有 Python 完整迁移，1,116 输出和 132 栏序参考通过（45 个正向排序变化）。主链未跳过中间的视觉注释/完整分组排序，当前仅诊断内核；正在统一构建安装并复核来源审计。默认 PDF、完整最终排序/视觉注释和用户整篇验收仍待。详见[接续对应表](2026-10-28-text-block-integration.md)。

2026-10-28 最新：成员块重建、标题/首页列/页边、完整图注及环绕正文、来源附注、矢量/字母局部列表均已接入 `refine_blocks=True`。642+558 规则、10 空白合同、211 连续页和 35 原始页源链通过。当前构建安装见[主链记录](2026-10-27-form-runtime-integration.md)。下一句为 `absolute_blocks`，余视觉注释/tight、排序/归一化/行内、默认 PDF 和整篇验收。下方为历史记录。

2026-10-28 最新：完整 `_build_text_blocks` 已完成，包含全部段界、悬挂/局部栏、分组脚注和全部块级补合并，204 连续页与 35 原始页源整链及定向对照通过。下一句已推进至 `merge_overlapping_member_blocks`，剩余后续成员/图注/局部列表、注释/排序/归一化/行内、默认 PDF 和整篇验收；上述文字块构建不再待办。构建安装见[主链记录](2026-10-27-form-runtime-integration.md)，[对应表](2026-10-28-text-block-integration.md)。下方为历史记录。

2026-10-28 最新：最终页面标题/语义和行准备已完整贯通至 `_build_text_blocks` 前，新增最终页/全文行状态入口，404+378+416 规则对照、204 连续页、35 原始页源整链及 15 Unicode/标签合同通过，统一安装状态见[主链记录](2026-10-27-form-runtime-integration.md)。下一步明确为完整文字块构建及后续块重建/排序/归一化/行内、默认 PDF 和整篇验收；最终标题/行状态不再待办。未重构目录，新增[Python→Rust 对应表](2026-10-28-final-text-line-integration.md)。无全量/CUDA/提升声明。下方为历史记录。

2026-10-28 最新安装：最终公式节点 release 构建并安装完成，243 页面前缀、6 定向正例、35 原始页源整链和两页实时 PDFium 通过；受影响旧公式、旧目录/Form/容器、118 模块兼容性套件全部通过。无全量/CUDA。下一步为最终标题/文字/行内聚合、文献块重建排序，再接默认 PDF 和整篇验收；公式接线不再待办。详见[主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-28 后续：最终页面公式接线已完成，`prepare_native_document_formulas` 连通原始页源/全文文献及语义/目录→公式恢复→canonical 重放/库存替换→墨迹→样式恢复→回退目录检查点。1037 原语句、6 正例、35 原始页源整链通过；最终安装状态见[主链记录](2026-10-27-form-runtime-integration.md)。下一步从 `title_container_bboxes` 继续最终标题/文字/行内及文献块重建排序，然后默认 PDF 和整篇验收。公式内核、canonical 和目录/墨迹接线不再待办。下方为历史记录。

2026-10-28 最新：完整公式认领/替换仲裁和密集行→同基线→公式输入链已实现，新增 `recover_native_formula_blocks`、`prepare_native_formula_input`。787 前缀、1059 完整恢复/分量、1023 输入序列及受影响旧合并/完整页验证通过；统一安装验收见[主链记录](2026-10-27-form-runtime-integration.md)。待接最终页面的目录/语义分流、canonical 重放、未映射墨迹和样式恢复，后续文字/行内/文献块重建排序、默认 PDF 与整篇验收；公式识别内核不再待办。无全量/CUDA或性能/效果提升结论。下方为历史记录。

2026-10-28 当前：文献区域/条目分组和前置小字号强调标题已迁移，`prepare_native_document_references` 连通原始页源→全文文字/标题→文献/紧凑标题，抵达最终页面编排前。637 文献序列、24 分组、22 文字合同、35 原始页源整链通过，含三页 PDFium 的跨页文献和附录停止正例；已统一 release 构建安装，3 组相关安装套件通过（新文献、旧标题、兼容性审计），含三页实时 PDFium、严格类型、所有权与并发。剩余最终公式/文字/行内编排、文献块重建排序、默认 PDF 和整篇验收；文献区域/分组内核不再待办。无全量/CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 当前：完整页面标题分类和全文标题原型/校准结构仲裁/未知字重重复标题已实现，`prepare_native_document_text` 连通原 `_classify_document_text` 全阶段。验证为 769 完整页、521 原型、590 全文标题、34 原始页源整链及 43 边界/4 集合轨迹（另 744 前缀）；已统一 release 构建安装到 build/rust-wheel-env，3 组相关安装套件通过（新全文/标题、旧辅助入口、兼容性审计），包含实时 PDFium、严格类型、所有权和并发。剩余文献、最终公式/文字/行内编排、默认 PDF 和整篇验收；标题推断阶段不再待办，最终收尾的标题调用归最终编排。无全量/CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 当前：全文辅助文本/正文画像节点已贯通原始页源、完整逐页准备、顺序末表继承、矢量重复装饰和全部辅助分类/双正文画像。546 组连续旧语句、11 组定向证据、34 组完整文档链通过；新入口 `prepare_native_document_auxiliary`，已统一 release 构建并安装到 build/rust-wheel-env，新文档链、旧完整页和兼容性审计 3 组安装套件通过，含实际 PDFium 新入口和并发/所有权验证。剩余明确为全文标题原型/结构/未知字重标题、文献区与条目、其余公式和最终文字/行内编排、默认 PDF 及用户整篇验收。无全量/CUDA，无性能/质量提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 最新：本轮已完整贯通逐页准备节点：同基线/二维簇/同基线合并→上/下标 sidecar→字符释放前证据冻结→PreparedPage 构建→末尾页辅助复核和末表查询；并接在已有全文几何、容器和脚注链之后。`prepare_native_document_pages(..., complete_page=True)` 可调用，旧阶段默认保持。976 组合并、499 组完整页（含 6 旧异常）、12 组身份/快照、3 组字体旧异常和实时 PDF 完整页通过。已在整个节点统一 release 构建并安装到 build/rust-wheel-env，5 组相关安装套件通过（含新入口/严格类型/并发/所有权/实时 PDF 和旧入口兼容）。剩余全文/跨页语义、其余公式恢复和最终文字编排、默认 PDF 及用户整篇效果/性能验收；不再把完整逐页准备列为待办。无全量或 CUDA，无效果/性能提升声明。见 [主链记录](2026-10-27-form-runtime-integration.md)。下方为历史记录。

2026-10-27 最新：容器前页眉/页脚/页码及局部正文/标题保护、重复页眉分隔线检测已完成，形成几何→页边保护→Form→分隔线完整文档前缀。207 组连续前缀通过；三页实际 PDF 的 6 页眉/3 页脚/3 页码/3 分隔线排除通过。页内通讯脚注/标题和容器后的辅助文字/公式/最终编排仍待，默认 PDF 标志不提升。当前仍为源码开发，未重新安装。见 [主链记录](2026-10-27-form-runtime-integration.md)，下方待办为历史。

2026-10-27 最新：全文几何计划生成器及“原始页面→生成计划→容器前修复→loose 侧表释放”已实现。通过完整生成器及阈值/局部字体/转换合同对照，146 组文档前缀、10 页实际 PDFium 完整行和几何侧表对照。局部字体有 7 组文档正例；4 组旧字体异常保留错误。全文几何生成不再待办，剩余页边/公式/文字分类、最终编排、默认 PDF 和整文档验收。源码 SDK 接口已补，未重新构建安装包；下文“计划生成仍待”为历史快照。见 [主链记录](2026-10-27-form-runtime-integration.md)。

2026-10-27 后续进展：几何计划的应用、旋转跨栏视觉重拆、样式/链接按字符身份分片已实现，并连通已有图像回收/分式恢复阶段。通过 66 组计划应用、27 组重拆、113 组连续原语句对照及 5 组来源缓存副作用对照。全文几何风险判断和计划**生成器**仍待，不能把计划消费完成写成完整几何完成；本阶段继续不打包或跑全量。详见 [主链记录](2026-10-27-form-runtime-integration.md)。

2026-10-27 最新源码进展：实际 Form 结构及全文角色归一化、Form/一般图形/栅格/图题图像和矢量公式已贯通容器主链；几何后的并排标题拆行、图内标签补认领和行首分式恢复也已连成阶段。矢量公式 139 场景、134 组连续文档链（含 4 个旧异常）、1,591 组文字特征通过；后几何阶段 230 组连续页通过。各阶段复用语料，不累加为独立文档数量。详情及验证边界见 [当前主链记录](2026-10-27-form-runtime-integration.md)。

剩余已收敛为：全文几何修复/视觉重拆、其余公式及文字分类编排、默认 PDF 接入和整文档验收。本阶段尚未构建安装包，已安装版本仍是上一阶段；源码审查清单 118 模块。下文目录/Form 选择及更早剩余列表均为历史快照。

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

新增 `prepare_native_bar_geometry(path_infos, drawing_lines, *, em, table_bboxes, axis_queries=())`，迁移复合矩形共同基线、柱体准入/方向和零轴扩边；沿旧规则，返回原数组身份及原序独立证据。table_bboxes 必须显式提供，完整填充表格检测、柱组聚类、标签/图体扩展及默认 PDF 编排仍待。170 组旧参考和 8 页实际 PDF 实时路径/绘图→准入→重建零轴查询链通过，详见 `2026-10-26-native-bar-geometry.md`。

栅格连接图阶段验收：144旧参考（63连接图/46文字框正例）、20页实际PDF全实时链（5连接图/1文字框，另1网格排除正例）通过；两个定向Rust单元测试和一个PDF集成测试、一次release wheel与五组相关安装套件通过。fmt、93模块审查及固定36＋3＋1哈希保持。无全量/CUDA，完整native diagram/bar产生器、跨页准备和默认PDF仍待。日志build/native-raster-diagrams-install-test.txt、build/native-raster-diagrams-regression.txt。

## 栅格连接图与普通文字框（2026-10-26）

新增 `detect_native_raster_diagrams(source, *, em, frame_bboxes=())`，复用已迁移绘图组件和真实有字网格，沿旧4图/12线/横纵数量/组件尺度面积及图片95%覆盖，先网格保护再检查3条独立内部短横线；同时返回普通4线文字框的双向覆盖/内部图像复杂路径拒绝/文字证明。输入不修改，共享调用传递同一Work，不保留跨调用缓存。

144旧参考（63连接图/46文字框正例）；20页实际PDF全实时字符/路径/绘图/图片→视觉行→组件查询框→检测链，5连接图/1文字框正例，另有1实际有字网格排除正例。180度默认视觉行没有文字，按旧支持域处理，不能据此宣称任意旋转方向的表格保护完成。

完整原生diagram/bar产生器、跨页准备及默认PDF编排继续待办；增强阅读顺序和CUDA延后。

同心轮廓阶段验收：196旧参考（145正例）、8页实际PDF候选→网格仲裁链（1正例），两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel及四组相关安装套件通过。fmt、92模块审查和固定36＋3＋1哈希保持。无全量/CUDA，完整强图形产生器集合、跨页准备和默认PDF仍待。日志build/native-concentric-graphics-install-test.txt、build/native-concentric-graphics-regression.txt。

## 图题与同心轮廓候选（2026-10-26）

新增 `detect_native_concentric_graphics(source, *, em)`：沿旧英文编号图题、百分比标签、同心二维描边轮廓和短引线证据生成候选，保留原path身份/顺序、中心计算顺序和包含去重。Unicode数字/空白及Python IGNORECASE的I/İ/ı行为保持，未扩展中文图题策略。

196组旧参考（145正例），8页专用四方向PDF实时字符/路径→视觉行→同心图候选→有字网格仲裁链（1正例），没有路径/文字/候选结果注入。em仍显式输入，图题只消费angle0，默认180度视觉行范围沿旧版本。

完整强图形候选集合、跨页准备和默认PDF编排继续待办；增强阅读顺序和CUDA延后。

图形/有字网格仲裁阶段验收：178旧参考（120网格排除正例）、8页实际PDF候选生成→仲裁链（1表格排除正例），两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel及四组相关安装套件通过。fmt、91模块审查和固定36＋3＋1哈希保持。无全量/CUDA；完整强图形产生器集合、跨页准备和默认PDF仍待。日志build/native-graphic-arbitration-install-test.txt、build/native-graphic-arbitration-regression.txt。

## 有字网格保护与图形候选仲裁（2026-10-26）

新增 `arbitrate_native_graphic_candidates(source, candidate_bboxes, *, median_height)` 和 `inspect_native_graphic_grids(source, regions, *, median_height)`。实时从绘图/文字构造长横线连通网格、贯穿竖轨、物理列占用和候选内行带，Form保护及多表并集覆盖达到70%才排除图形；随后沿旧稳定面积序做包含拒绝/重叠合并和空间排序。不是只传入网格结论的框合并。

178组独立旧参考（164基础＋14数值边界），120网格排除正例；最终裁决参考直接执行原强图形函数尾段AST，网格证明仍调用原旧实现。8页实际PDF文字/绘图→组件图形候选→网格保护/仲裁链全实时，8候选正例、1页表格排除正例，四方向保留旧angle0文字网格支持域。

候选产生器集合和median_height仍是明确阶段输入，完整强图形入口/跨页准备/默认PDF编排继续待办；增强阅读顺序和CUDA延后。

独立图形阶段验收：229旧参考、41既有实际PDF输入和8页专用实时链（3矢量/1小栅格正例）、两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel和四组相关安装套件通过。fmt、90模块审查及固定36＋3＋1哈希保持。无全量/CUDA；完整强图形/表格仲裁、跨页准备和默认PDF仍待。日志build/native-isolated-graphics-install-test.txt、build/native-isolated-graphics-regression.txt。

## 独立矢量与小栅格证据（2026-10-26）

新增 `detect_native_isolated_graphics(source)`：迁移旧独立矢量图体和有意义小栅格判定，保留路径增量合并（新路径在组首/旧命中组按原序追加）、样式修复字号/中位数、正文和栅格覆盖拒绝及多色页角例外。返回em、原位置path_groups、vector_bboxes及meaningful_raster_indices，不修改页源，无跨调用缓存。

229组旧参考含137矢量/7小栅格正例，41既有实际PDF字符/路径/图片/视觉行实时输入；另8页专用四方向PDF链含3矢量/1小栅格正例。未注入路径/图片/行/分组结果；180度文字沿旧默认视觉行支持域，未自行扩展覆盖识别。

完整强图形/表格仲裁、跨页准备、候选检测与默认PDF编排仍待，增强阅读顺序和CUDA延后。

绘图组件阶段验收：239旧事实、41既有实际PDF输入和8页专用实时链（2正例）通过；两个定向Rust单元测试和一个PDF集成测试、一次release wheel和四组相关安装测试通过。fmt、89模块审查及固定36＋3＋1哈希保持。无全量/CUDA；完整强图形/表格仲裁、跨页准备及默认PDF仍待。日志build/native-drawing-graphics-install-test.txt、build/native-drawing-graphics-regression.txt。

## 绘图组件与复杂路径图形（2026-10-26）

新增 `detect_native_drawing_graphics(drawing_lines, path_infos, *, page_size, median_height)`：按旧左缘稳定扫描及并查集连接绘图线，保留原行首次出现的组件顺序、原数组位置及bbox类型，统计横纵线后按组件尺寸/复杂路径重叠生成图形候选。复用已有CPython hypot，算术不变，仅扩大可见性供JSON适配层调用。

239组独立旧参考含134图形正例；41页既有实际PDF实时线/路径输入，另8页专用四方向PDF链含2图形正例。所有几何/候选都由实时读取生成，不注入组件或路径结果。接口无跨调用缓存，median_height仍显式传入。

完整强图形/表格仲裁、跨页准备、候选检测和默认PDF编排仍待，增强阅读顺序和CUDA延后。

复杂路径阶段验收：263旧参考、41既有实际PDF输入和8页专用实时链通过（4容器/2轴正例）；两个定向Rust单元测试和一个PDF集成测试、一次release wheel及四组相关安装套件通过。fmt、88模块审查及固定36＋3＋1哈希保持。无全量/CUDA；完整强图形/表格仲裁、跨页准备和默认PDF仍待。日志build/native-path-graphics-install-test.txt、build/native-path-graphics-regression.txt。

## 复杂路径容器与坐标轴候选（2026-10-26）

新增 `detect_native_path_graphics(path_infos, *, page_size, median_height)`：保留旧二维复杂路径判定、填充容器内部路径证明、相交坐标轴和内部轮廓扩框。返回complex_path_indices（原数组位置）、container_bboxes和axis_bboxes，顺序/重复候选保留；union逐坐标保留原数值赢家和JSON类型。

263组独立旧事实（60容器/73轴正例），41实际PDF实时路径输入；另8页四方向专用PDF链，4容器/2轴正例，无路径或候选结果注入。中位图形高度显式传入，尚不执行完整图形/表格/重叠仲裁。

继续待办：完整强图形检测、跨页页源准备、候选检测及默认PDF编排；增强阅读顺序和CUDA仍延后。

堆叠柱候选阶段验收：304旧事实、41页真实PDF负例读取、两个定向Rust测试通过；一次release wheel及四组相关安装套件通过。fmt、87模块来源审查和固定36＋3＋1哈希保持。真实PDF无异常CJK/柱图正例，此限制保留；无全量/CUDA，完整图形/表格仲裁及默认编排仍待。日志build/native-stacked-graphics-install-test.txt和build/native-stacked-graphics-regression.txt。

## CJK图形尺度与堆叠柱候选（2026-10-26）

新增 `detect_native_stacked_graphics(source, *, em, fallback_em)`，沿旧20个CJK松紧框锚点、中位数字宽/墨迹证据校准图形尺度，保留柱宽/共同基线/间隔、同列叠层、外侧刻度和轴/路径标签扩框。接口返回graphic_em、inflated_cjk_metrics和stacked_column_bboxes，source不修改；em和fallback明确传入，尚未串完整强图形尺度编排及图形/表格仲裁。

304组独立旧参考，含162柱图正例、210尺度校准。41页实际PDF读取字符/路径/视觉行是负例（没有inflated或柱图正例），不能用它们声称已验证实际PDF柱图正例效果；正例来自独立合成页源。输出候选坐标为float PDF point，保留path顺序与重复候选。

下一步继续完整强图形检测/跨页页源准备、候选检测及默认PDF编排。增强阅读顺序与CUDA仍延后。

栅格图归组阶段验收：272组旧参考、16页全实时PDF链、两个定向Rust单元测试和一个PDF集成测试通过；一次release wheel构建及五组相关安装套件通过。fmt、86模块来源审查、固定36＋3＋1哈希保持。无全量/CUDA，完整强图形/跨页准备/候选检测/默认PDF编排仍待。日志build/native-raster-charts-install-test.txt、build/native-raster-charts-regression.txt。

## 栅格图说明续行归组（2026-10-26）

新增 `group_native_raster_chart_descriptions(source)`：沿旧数字语法和固定刻度分组内核，以等差刻度/相邻栅格图证明图体，按旧标签/图例窗口扩框，再归组图上方同字体长说明及短续行。返回独立source、graphic_bboxes和axis_groups（原数组位置），仅修改paragraph_group。保留稳定顺序、图片面积首项平局（含精确整数/浮点比较）、负/已有分组及同一core相邻链覆盖。

272组旧完整事实（269基础＋3面积边界），223图体/151归组正例。16页实际PDF完整字符/图片→视觉行/局部页源→背景/水印/白边过滤→噪声→图体/说明归组链，10图体/2归组正例，覆盖四方向及字体/列表/句末/宽尾拒绝，无图片/刻度/颜色/图体结果注入。

完整强图形检测、跨页页源准备、候选检测和默认PDF编排仍待；增强阅读顺序和CUDA延后。

噪声阶段验收完成：236组旧决策、32页实际PDF全实时链（4页删除正例）、两个定向Rust单元测试和一个PDF集成测试通过；一次wheel构建，五组相关安装套件通过。fmt、85模块审查和固定36＋3＋1哈希保持。无全量/CUDA，图注归组/强图形框自动检测/完整默认编排仍待。日志build/native-text-noise-install-test.txt及build/native-text-noise-regression.txt。

## 原生文字噪声与按需填色（2026-10-26）

新增 `prepare_native_text_noise`、`apply_native_text_noise`、`read_pdf_text_paint`、`filter_pdf_native_text_noise`。先按旧短数字/正文/目录/图缝规则判候选，再读取纯填色RGBA，以浅灰和深色/白色正文比例证明删除；无候选不访问PDF。固定rule_text几何内核不改，字体签名/Unicode特征和来源号/字符索引清理沿旧语义。返回独立source，保留输入字段和未删除行原始表示。

236组独立旧决策（208基础＋28边界，199候选/178删除正例），32页真实PDF完整实时字符→视觉行→局部源→候选→填色→清理链含4删除正例，描边/裁剪、范围外灰度等保留。颜色请求精确、重复来源号/缺失字符索引、80%阈值、所有权/限额纳入验证。

强图形框仍是显式阶段输入，当前真实PDF链不需要图缝图形框；完整强图形检测、栅格图注归组、跨页准备/候选检测及默认PDF编排继续待办。增强阅读顺序和CUDA仍延后。

验证完成：一个定向Rust实际PDF测试、一次release wheel构建及六组相关安装套件通过；10页全实时链与85页图片事实均匹配旧参考，公式88组回归保持。fmt、84模块审查及固定36＋3＋1哈希通过。无全量/CUDA。日志build/pdf-image-infos-install-test.txt和build/pdf-image-infos-regression.txt。

## PDF 图片证据自动读取（2026-10-26）

新增 `read_pdf_image_infos`，读取页面尺寸/框/旋转、图片视觉框、原始流SHA-256指纹、平滑背景和白边证据。沿用固定对象树/Form/累积裁剪，所有PDFium句柄和bitmap在唯一锁内释放，仅返回自有JSON。共享已有Pillow RGB缩放算术至core，公式预处理保留原调用和运算。

已冻结85页旧PDF图片结果，包含57图、13处平滑背景、20处白边；专用PDF覆盖RGB/L/alpha、嵌套Form、裁剪/翻转、四方向和重复图片。另冻结10页完整实时页源链的旧参考，验证字符/绘图/图片→视觉行/局部页源→全文图片裁决，不注入图片结果。尺寸和累计解码/原始流/缩放预算为显式支持边界。

下一步仍是噪声/图文归组、完整页源和跨页准备、候选检测/默认流程集成；增强阅读顺序与CUDA延后。

## 图片区域裁决（2026-10-26）

新增 `prepare_native_image_regions(pages)`：先全文统计跨至少三页、面积至少8%的重复栅格指纹，再逐页过滤正文充分时的平滑背景，最后按旧标题和图片白边证据裁剪。保持空指纹有效、不同页计数、原始 bbox/类型/顺序、普通中位数和重复 bbox 最后一份证据覆盖规则。

已冻结328组旧决策，含33组水印正例、30个含平滑图删除的页和78处白边裁剪输出；64组实际PDF页源/文档用例包含11个图片证据输入。图片证据仍需传入，安装链仅实时读取字符并运行视觉行/局部行准备，不能称为实时图片读取或完整PDF解析。

验证：两个定向Rust测试、一次wheel构建和四组相关安装测试通过；fmt、83模块来源审查及固定36＋3＋1哈希通过。未跑全量或CUDA。日志见build/native-image-regions-*.txt。

下一步：实现PDF图片指纹/平滑背景/白边像素证据读取，再处理噪声、完整页源/候选检测及默认流程集成。增强阅读顺序和CUDA仍延后。

## 当前检查点：脚本证据生成、投影与物化诊断（2026-10-26）

验证完成：五项受影响Rust测试（四单元＋一实际PDF集成）、249脚本输出/诊断、516 sidecar构造与五组相关安装套件通过。57实时PDF负例链及四方向专用PDF实际检测产生7处物化脚本，字符/角色/sidecar未注入；字段/类型/诊断顺序、所有权/并发/限额/错误恢复及旧apply默认兼容通过。阶段末一次wheel构建；fmt、82模块审查及固定36＋3＋1哈希保持，无全workspace/全部100组runner/未受影响渲染/全量clippy/CUDA。日志build/native-script-output-install-test.txt、build/native-script-output-regression.txt及build/native-script-output-*-rust.txt。

新增prepare_native_script_lines，把已有视觉行/紧框/原点/原始字符和绘图线转为完整script_lines，复用旧角色分类与分式保护；新增trace_native_inline_evidence返回blocks和materialized_ranges。apply_native_inline_evidence增加可选script_lines：省略或None保持原默认，显式数组才运行脚本阶段（空数组也按旧行为清理_inline_math_regions）。

脚本阶段接通整行主归属、逐范围紧框备用归属、短脚本独立匹配、晚期公式区域保护、稳定原对象位置身份和精确物化诊断。诊断保留原始框数值类型、来源/角色/文本、原字符串和证据码点区间、角度/公式/稳定body计数及旧顺序，不用可能重复的source_index替代对象身份。

独立旧参考249完整输出/诊断（80正例）和516 sidecar构造事实；44个预分类分式成员输入留在原560行精炼参考，生成接口从原始字符/规则计算成员。57实际PDF链为无脚本范围负例，另专用实际PDF四方向读取→视觉行/局部准备→检测/证据→Span与诊断产生7处真实脚本。五项受影响Rust测试通过，含原297样式/链接输出和实际PDF集成。

投影/诊断沿外层10M Work和8MB结果上限，证据准备自身10M保守计费并沿已有叶子域（每行1500字、side-map20000、分式原语1500字/5000规则）；普通JSON和良构几何，资源错误传播。完整页源、像素字重、图片/噪声、跨页规则、候选检测和默认PDF集成仍待；完成标志False，继续定向/阶段相关回归，无全量/CUDA。

## 历史检查点：样式/链接投影到InlineSpan（2026-10-26）

验证完成：两个定向Rust测试、297旧完整输出、57实时字符＋注解链及五组相关安装套件通过（新增inline_output、原pdf_link_annotations、native_inline_evidence、compatibility、smoke）。核对全部字段/类型/码点偏移、现有脚本标记、输入不变/并发、限额/扫描预算和错误后恢复；块仍为明确视觉行派生输入。阶段末一次wheel构建，fmt、81模块审查和固定36＋3＋1哈希保持。未跑全workspace/全部99组runner/未受影响渲染/全量clippy/CUDA；日志build/native-inline-output-install-test.txt和build/native-inline-output-regression.txt。

新增apply_native_inline_evidence(blocks, *, page_size, style_lines=(), link_lines=())，把明确的finalized块与原生证据投影为独立InlineSpan块。接通单位框/point框归属、类型样式过滤、Unicode码点文本对齐、公式空洞、严格断词和唯一上下文回退、链接冲突/跨行合并、原字符串区间及文字/公式/链接Span。已有detector-owned平坦sup/sub标记沿旧逻辑物化；新script_lines的匹配/诊断暂未接入。

独立旧Python冻结297完整输出，含57实际PDF字符/注解→视觉行/局部准备/检测→输出链。块为明确视觉行派生输入，不冒充整页块检测。修正旧规范Span中链接子文字保留styles空数组而顶层省略该字段的差异。两个定向Rust测试通过，已有Work保持；共享10M预算覆盖几何、查找、公式扫描/DP、链接冲突和Span边界，未闭合公式扫描同样计费。

支持普通JSON、4000块/各4096证据行/每行5000良构范围、坐标abs≤1e9、8MB输出及既有JSON限制。链接目标按PDF来源的四种安全外链协议；不接受任意Python私有区间对象。原字符/块字段保留，返回新块；未开启默认PDF，完整script_lines投影、像素字重、图片/噪声/跨页页源准备及候选检测仍待。

## 历史检查点：PDF链接注解自动读取（2026-10-26）

验证完成：两个定向Rust测试（207 URI、36实际PDF页/26注解）、57实时PDF字符＋实时注解链和五组相关安装套件通过。该链不再注入旧注解输出；路径/bytes/流所有权、并发、加密/NUL/错误后恢复和作者URI/Quad资源限额PDF通过。阶段末仅构建一次wheel；fmt、80模块审查及固定36＋3＋1哈希保持。未跑全workspace/全部98组runner/未受影响渲染/全量clippy/CUDA。日志build/pdf-link-annotations-install-test.txt、build/pdf-link-annotations-regression.txt及build/pdf-link-*-rust.txt。

新增read_pdf_link_annotations(source, *, page_index=0, pdfium_library=None, password=None)，Rust直接读取外部URI Link，返回独立target/bboxes/source_index。沿旧可见性标志/动作/UTF8/URI策略跳过普通坏链接，QuadPoints优先、去重与排序，全部无效时回退Rect；按原CropBox和四方向旋转转换并裁页。原枚举索引包含跳过的链接，不重新编号。复用唯一PDFium运行库锁、RAII文档/页和即时关闭注解句柄。

独立旧参考为207 URI策略事实、36实际PDF页/26有效注解，两个定向Rust测试已通过。URI保留旧http/https/mailto/tel及Unicode16 trim/NFKC、urllib括号/IPv6/IPvFuture和非规范端口行为，不做浏览器URL规范化/网络访问。枚举20000/有效5000、单URI1MB/累计URI8MB、单链接5000 Quad/累计20000区域，资源异常传播而非漏链；输入/页数沿Limits。另两份作者PDF专测URI/Quad资源限制，不计作旧Python等价证据。

该阶段只完成注解读取及与既有检测的串接；像素字重补证、样式/链接最终输出投影、完整图片/噪声/跨页页源准备、候选检测和默认PDF集成仍待。两个legacy完成标志False，继续定向测试/阶段末构建相关回归，不运行全量/CUDA。

## 历史检查点：原生样式与链接证据检测（2026-10-26）

验证完成：两个定向Rust测试、285旧事实、57实时PDF字符链、五组相关安装套件通过（新增inline_evidence、原source_lines、native_table_materialization、compatibility、smoke）。覆盖码点半开区间/完整字段类型顺序、所有权/并发、限额/累计扫描预算和错误后恢复。最终wheel隔离安装，fmt、79模块审查及固定36＋3＋1哈希保持；未跑全workspace/全部97组runner/未受影响渲染/全量clippy/CUDA。日志build/native-inline-evidence-rust.txt、build/native-inline-evidence-install-test.txt和build/native-inline-evidence-regression.txt。

新增detect_native_inline_evidence(lines, *, drawing_lines=(), annotations=())，返回style_lines/link_lines的Unicode码点半开区间。接通字体粗体、短粗体/分离项目符号过滤、下划线/删除线最佳归属、紧字框/分数线保护，以及链接中心/面积命中、异目标冲突和连续同目标区间。只有全整数char_idx时稳定重排；粗体元数据复用已有函数，几何复用固定vendor inline_styles.rs，原样上游未改。

独立旧参考285组：275基础页含150样式与116链接正例，另10组多字符项目符号边界。57实际PDF字符→视觉行→局部准备→检测链，其中2样式和4链接正例；链接注解为旧PDF读取的冻结阶段输入，尚不代表SDK实时读取注解。补查复现并修正按原片段而非规范化片段判断项目符号的差异，两项定向Rust测试通过；因此在首次构建后追加了一次必要的修正构建。

统一10M Work接受已有预算；最多4096行/每行20000字符/单片段256字节、各5000绘图/注解、20000链接区域、8MB输出和既有JSON限制，有限坐标abs≤1e9、来源i64/布尔；资源异常明确传播。后续继续SDK链接注解读取、像素字重补证、证据向输出投影、图片/噪声/跨页准备与候选检测；默认完成标志False，CUDA/enhanced继续延期。

## 历史检查点：页源局部段落与分隔线准备（2026-10-26）

验证完成：两个定向Rust测试、290旧完整事实、57实时PDF读取→视觉行→局部准备链及五组相关安装套件通过；含原视觉行、完整表格块事务、compatibility和smoke。保留字段/类型/行序、输入不变/并发、输入限额/共享扫描预算及错误后恢复通过。阶段末只构建一次wheel；fmt、78模块审查和固定36＋3＋1哈希通过。未跑全workspace/全部96组runner/未受影响渲染/全量clippy/CUDA。日志build/native-source-lines-rust.txt、build/native-source-lines-install-test.txt和build/native-source-lines-regression.txt。

新增prepare_native_source_lines(lines, *, page_size, drawing_lines=(), image_infos=())，按旧页源顺序接通下沉首字母正文、悬挂字母编号段、极薄水平栅格图片转线、页顶重复私用区装饰文字转线。局部分组保留原字符/bbox/行序/来源号，仅修改旧规则对应的text、semantic_type、paragraph_group、title_suppressed；图线先规范化原绘图，再追加薄图线和装饰线。薄图em取正em_height的上中位数，保留样式修复时canonical em优先的有效行高规则。

独立旧Python冻结290完整事实，包括22装饰线和16实际薄图线输出及57真实PDF字符→视觉行→本阶段链；提供的图片/绘图区域是阶段输入，尚未自动读取整页图像。两项定向Rust测试通过，包含调用方已有工作预算不重置；共享10M扫描预算、既有JSON限制、最多4096行/每行20000字符、各5000输入规则/图片、10000输出线/8MB输出，资源异常明确传播。

下一步继续样式/链接、图片水印与噪声、跨页收集准备及完整表格候选检测；高层默认PDF流程和整文档效果/性能验收仍待。两个legacy完成标志False。遵循开发定向测试、阶段末统一构建相关回归，CUDA与enhanced阅读顺序继续延期。

## 历史检查点：原生字符到视觉行（2026-10-26）

已完成验证：新增定向Rust测试覆盖317旧事实/977行；阶段结束构建一次release wheel，隔离安装后57组实时PDF读取→视觉行、全部字段/类型、输入不变/并发、限额和组行＋配对共享Work通过。另五组相关回归通过，共六组安装套件；77模块来源审查、fmt与固定36＋3＋1哈希通过。日志build/native-visual-lines-rust.txt、build/native-visual-lines-install-test.txt和build/native-visual-lines-regression.txt。未跑全workspace/全95组runner/未受影响渲染/全量clippy/CUDA。

新增build_native_visual_lines(chars, *, page_size, page_rotation=0, supported_angles=(0.,90.,270.))，接通基础片段/行组装、旋转和公式方向筛选、视觉间隙分run、文字规范化、排版统计、互为最近的上下标和链式合并。复用原样固定vendor text_pipeline/inline_pairs及现有字体度量，Unicode16属性和Sm范围固定。保留全部公开LineItem字段、原字符字典、合并前source_bbox/标记、normal/formula分别编号再按row/run稳定排序。

独立旧Python冻结317事实/977视觉行，包含25代理对恢复后的实际PDF字符和32去重后的实际PDF字符。开发只运行新增定向Rust测试；本阶段结束统一构建wheel及相关回归。按用户要求暂停CUDA编译/GPU验证，不重复全workspace/全安装套件/未受影响渲染测试。最多4096输入字符、单字符256字节、单行排版1500字符，既有JSON和共享10M工作预算明确传播。

这只是完整页源中的视觉行基础链。首字母/局部段落、装饰规则、样式/链接/图片/噪声和跨页校准等收集准备、完整表格候选检测、高层默认PDF集成与整文档质量/性能验收仍待。两个legacy完成标志继续False，enhanced阅读顺序与GPU实机依既定安排延期。

## 历史检查点：表内媒体 PNG 与页级文字回退（2026-10-26）

相关安装回归另10组全部通过（块事务、注释、准备、整表物化、页面/内联/表格资产、model_document、compatibility、smoke），连新增媒体PNG共11组；独立36组渲染像素对照亦通过。未全跑94组runner；日志build/native-table-media-png-regression.txt。model_document仍分别保留冻结提供器与真实CPU模型烟测的证据边界。

验证：232项Rust workspace测试、fmt、严格clippy、CUDA feature编译及76模块审查通过；release wheel已隔离安装。182 PNG页/502引用、45真实PDF/125引用全部通过像素/模式/尺寸/ICC/tRNS及其他JSON/HTML对照，18 RGBA＋18 RGB组通用PDF渲染像素重新核对一致。固定36 core＋3 paths＋1 scripts哈希不变。日志build/native-table-media-png-install-test.txt、build/native-table-media-png-render-regression.txt、build/native-table-media-png-wheel.txt与build/rust-current-*.txt。GPU仅编译检查，尚无整文档性能或质量结论。

新增 `materialize_native_table_media_page(blocks, page_png=None, *, page_size=None)` 和 `materialize_native_table_media(pages, source, *, page_sizes=None, pdfium_library=None, password=None, max_pixels=16_000_000)`。复用自有Pixels裁剪，保模式90/180/270逆时针旋转和ICC/tRNS；按旧floor/ceil像素边界生成PNG data URI。页内一张图失败时，已生成图片保留，其余当前及后续表占位符回退html.escape文字，最终清理非空_table_images；空媒体字段保持原值。

PDF入口对每个有媒体的原页只渲染一次RGB，沿旧scale2和长边3500上限；无媒体不读PDF/PNG。pages按原PDF页号对齐，未参与页用[]占位。默认接收旧finalized单位框；显式page_size(s)仅将媒体point框按旧千分刻度/ties-even/最小刻度转换，保留其余块字段，衔接上一阶段事务。源PNG支持8bit L/LA/RGB/RGBA，沿64MB/16MP解码和4MP单裁图；新增累计64MP裁图/64MB PNG/64MB输出及10M文字扫描预算，资源和不支持模式异常明确传播。一般render_pdf_page的原表达式不变。

旧Python独立冻结182单页/502图片引用，含四模式四旋转、32页内部分成功后失败、14 ICC/tRNS引用及point刻度；45真实PDF/角度/越界页/块事务链产生125引用，含长边3500及4组旧事务占位→实际PDF裁剪链。PNG由Rust编码，压缩字节、data URI和由URI哈希派生的文件名为明确输出差异；解码mode/尺寸/原像素/ICC/tRNS及其他HTML/JSON对照。区域来自明确阶段输入，尚不证明页级图像/候选检测或整文档质量提升。

接下来为完整页源/候选检测及高层默认集成；两个legacy完成标志仍False。GPU实机和enhanced阅读顺序继续延期；下方为历史检查点。

## 历史检查点：完整表格块计算事务（2026-10-26）

新增 `materialize_native_table_blocks(source, candidates, *, tight_bboxes=None, origins=None, calibrated_char_bboxes=None)`，接通旧_materialize_table_blocks的稳定score排序、原完整bbox去重、注释准备、原生恢复/语义HTML、首页弱表撤销、原字符空间投影和表体/注释共同发布及行认领；返回table_blocks、annotation_blocks、claimed_line_indices、独立更新后的lines/candidates。原始字符负责投影，校准字符仅用于恢复；页级校准缓存供候选复用。权威inferred_grid继续替换源规则/矩形，校准证据触发字体样式保留。候选逆向框按旧行为原样保留，由引擎/投影各自归一化；字符/校准框的校验域保持，数值char_idx原值转交引擎。

媒体已移植选择、嵌套去重、原语遮罩、唯一物理单元格归属和临时URI插入；失败时走原文字恢复。返回_table_images保存区域和图内文字，后续PNG编码与失败时转义文字替换尚待接入。坏origin/tight映射在旧行级入口会发生解包/旋转异常，本轮修正被容错成缺几何后误接受HTML的差异；低层脚本特征容错保持。新的准备/媒体/引擎/语义/弱表/投影共享10M外层Work，资源异常明确传播，普通恢复异常仍回退、普通投影异常整组不发布不认领。

独立旧Python冻结781完整事务：192原生接受、512投影、102首页撤销、109注释整组回滚、24媒体、4真实恢复异常。输入为明确_PageSource/_TableCandidate阶段数据；原始引擎抽样中2个char非str记录不属于PageSource域，排除原因独立记录，原724引擎参考保留。另1组八候选/2000字体组案例验证全页缓存，96旧媒体事实为40附着/56拒绝；16实时PDF/角度以实际字符/线/矩形及重建glyph视觉行进入完整计算，媒体区域正例为提供区域的阶段参考，尚不含实际图像检测/PNG证明。

支持≤128候选、每类媒体集合5000，沿既有源行/字符/原语/JSON上限，最多4000输出块/8MB块序列化及4MB HTML。后续转到实际PNG编码和页级来源/候选检测接入；两个legacy默认完成标志仍False，增强阅读顺序和GPU实机依既定安排延期。下方为历史检查点。

验证完成：231项Rust workspace测试、fmt、严格clippy、CUDA feature编译、75模块来源审查及固定36 core＋3 paths＋1 scripts哈希通过。最终wheel已安装，781旧完整事务＋八候选缓存、16实时PDF/角度串联的值/类型/候选与行副作用、所有权/并发、输入限额、跨准备/恢复/渲染累计预算、异常后恢复及许可证检查通过。96媒体选择/遮罩/附着事实由Rust直接核对（40附着、56拒绝），安装包通过完整事务中的24媒体案例。资源异常直接传播，未伪装成投影或空表。日志：build/native-table-block-commit-rust.txt、build/native-table-media-rust.txt、build/native-table-block-commit-install-test.txt、build/native-table-block-commit-wheel.txt及build/rust-current-*.txt。

相关安装回归共12组通过：新完整块事务、注释、块准备、整表物化、二维簇、脚本准备、多行/空间投影、完整引擎、字符去重、model_document、compatibility、smoke。model_document仍含冻结旧版面/公式/提供器与真实CPU V3烟测，不替代完整原生PDF质量证明。全量runner为93组，本轮未全跑或重跑36组像素对照，GPU仅编译检查。日志build/native-table-block-commit-regression.txt。

## 历史检查点：注释块物化与首页弱表撤销（2026-10-26）

新增 `prepare_native_table_annotations(source, candidate, *, calibrated_char_bboxes=None)`：复用块级准备，在同一Work中实际物化注释，返回 annotation_blocks、externalized_line_indices、failed_annotations 和已有准备字段；按真实失败集合回并表体，并从投影成员扣除成功外置行。保留按来源排序、字体族/字号视觉重启、原控制字符/Zs/软断词清理及物理行连接、tight+1pt裁页、局部框/行高/字体集合元数据。bbox及height保留int/float类型，字体集合转为排序唯一列表。旧 `prepare_native_table_block` 的显式失败输入契约不改。

新增 `restore_native_front_matter_panel(source, bbox)`，仅供结构恢复失败后显式调用。保留两类元数据字段、侧栏长正文/摘要证据、真网格保护，返回 restore 和独立lines；成功撤销时只对原规则匹配的metadata/abstract行添加paragraph_title与explicit_section_title。不主动覆盖已验证表格。NFKC固定为既有Unicode16版本unicode-normalization0.1.24，casefold/word/space复用固定表；元数据正则按旧ICASE和词边界语义移植，不添加文档名称规则。

独立旧Python冻结904组完整注释事务（327有输出、141分组正例、183tight框案例、98含失败注释），另24坐标/高度类型事实；240首页面板中102撤销，32角色/元数据事实。16实时PDF/角度输入为无注释负例，非空视觉行由实时字符生成glyph后重建，不能作为实际PDF注释正例或完整行检测器证据。输入沿用4096行/20000字符与成员/5000规则/2000注释/有序有限坐标abs≤1e9及原JSON限制；新增最多4000输出块、序列化累计8MB，行连接保留1000行/1MB文本/32M扫描上限。准备、选择、文本和字体扫描、输出序列化共用10M外层预算。

下一步接投影降级、媒体归属与最终表体/注释原子提交；两个legacy完成标志仍False，增强阅读顺序及GPU实机仍延期。下方为历史检查点。

验证记录：229项Rust workspace测试、fmt、严格clippy、CUDA feature编译、74模块来源审查及固定36 core＋3 paths＋1 scripts哈希通过。最终wheel已安装，928注释/类型事实、240首页面板和16实时PDF负例逐值/类型/元数据、输入不变、并发、异常后恢复、许可证检查通过。4000个字体组的原准备和1600个注释各自可运行，组合后因共享10M预算明确失败，验证没有按阶段重置。日志：build/native-table-annotations-rust.txt、build/native-table-annotations-install-test.txt、build/native-table-annotations-wheel.txt与build/rust-current-*.txt。GPU只作编译验证。

本轮10组安装套件通过：新注释、块级准备、整表物化、二维簇、脚本准备、完整表格引擎、字符去重、model_document、兼容性及smoke。model_document仍含冻结旧版面/公式/提供器与真实CPU V3烟测，不替代完整原生PDF质量证明。安装runner共92组，本轮未全跑或重跑36组像素对照；回归日志build/native-table-annotations-regression.txt。

## 历史检查点：PDF 表格块准备（2026-10-26）

新增 `prepare_native_table_block(source, candidate, *, failed_annotations=(), calibrated_char_bboxes=None)`，移植旧 `table_materialization` 的同字体汉字宽松框修复、嵌入脚注分离、投影成员／同基线闭包及失败注释表体框回并。输出 candidate、projection_line_indices、body_bbox、calibrated_char_bboxes、native_chars；显式校准覆盖自动校准，候选和字符均为独立 JSON，字符除被校准 bbox 外的字段保留。仍按原字段/阈值决定，不新增解析规则。

独立旧 Python 冻结612组准备（53脚注分离、207认领扩展、46校准案例）及12整数／浮点边界类型事实。保留同基线分轮 selected_lines 快照、严格字体签名、旋转表体核心成员不筛行角度、表头5%交叠、脚注候选先缩core再算投影成员和 Python min/max 获胜坐标值的类型。来源集合以排序去重列表表示。另冻结16组实际PDF／角度的准备→完整恢复→语义HTML链，12非空；其中行是已有glyph视觉行的显式投影，不将其称为完整PDF行检测器结果。

输入≤4096唯一来源行、20000字符／成员／注释框映射、5000规则和2000注释，有序有限坐标abs≤1e9，沿用JSON节点及2MB文本限额；字体组比较、脚注分离和认领闭包共享10M工作预算。该接口只准备候选，`failed_annotations` 是注释物化阶段的显式结果输入；尚不输出注释文本、提交表格块或最终认领。下一步实现注释块物化和首页弱表撤销，再接投影降级、图片归属与原子提交。两个legacy默认开关仍False，GPU与enhanced依既定安排延期。

验证记录：228项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；最终wheel已安装。624组旧准备/类型事实、16组实时PDF/角度的准备→恢复→语义HTML串联（12非空）通过。非空PDF的行由SDK实时字符重新生成glyph视觉行，4组空行阶段输入单列为负例，不等同完整PDF行检测。字符字段保留、独立输出/输入不变、并发、输入限额、字体分组与基线比较各自成功但累计超10M失败、异常后恢复及随包许可证检查通过。73模块来源审查、36 core＋3 paths＋1 scripts固定哈希通过。日志：build/native-table-block-rust.txt、build/native-table-block-install-test.txt、build/native-table-block-wheel.txt、build/rust-current-*.txt。

本轮共9组安装套件通过：块级准备、整表物化、二维簇、脚本准备、完整表格引擎、字符去重、model_document、兼容审查、smoke。model_document 仍含冻结旧版面/公式/原生提供器及真实CPU V3烟测，与新块准备的实时原语串联证据分开记录。全量runner共91组，本轮未全跑，也未重跑36组像素对照。相关日志：build/native-table-block-regression.txt。

## 历史检查点：整表语义 HTML 与结果恢复（2026-10-26）

新增 `render_native_table_html_with_scripts(result, table_input, *, tight_bboxes=None, origins=None, preserve_font_styles=False)` 和 `recover_native_table_result(table_input, *, tight_bboxes=None, origins=None, preserve_font_styles=False)`。后者返回 `None` 或 `(content, result)`，result 为独立中立 JSON，保留原始 `result.html`。已串联完整恢复引擎、单元格准备、二维合并、脚本角色、缺空格检测和整表渲染；复用已有字符与绘图接口。缺侧映射、无有效文字契约及无语义变化时按旧条件保留原 HTML；跨度、转义、字体样式和文本不匹配回退均沿用旧实现。

结构恢复错误使用 `NativeTableRecoveryError(RapidDocError)`，含 `code` 和 `stage="recovery"`；HTML 阶段异常不包成恢复错误。新编排使用共享 10M 外层 Work，既有内核预算仍保留；叶子 parts 不再重置编排预算，侧映射克隆按条目预收费，整表追加片段前检查 4MB。保留既有 JSON/准备/字形/规则限额。内部接入兼容旧非直角整数 angle 的原坐标几何回退，独立二维公共接口仍限四直角。

独立旧 Python 冻结 526 组完整整表 HTML（168 组自动上下标正例，输入为中立 NativeTableResult 阶段事实）、724 组原始字符/绘图恢复（547 非空）和 32 组实际 PDF/旋转/字体样式组合。实际 PDF 中 24 组恢复非空，4 组语义 HTML 改变原 HTML；不将阶段正例称作 PDF 效果提升。

待办转到 PDF 块级物化：表格接受/回退、字符认领、注释/图片归属、跨表修复及最终输出接入；visible-only 遮挡和 owned 批量加速未移植。`legacy_native_table_recovery=False`、`legacy_table_execution=False` 保持，enhanced 顺序与 GPU 实机测试仍延期。下方检查点为历史记录。

验证记录：227 项 Rust workspace 测试、fmt、严格 clippy、CUDA feature 编译通过。最终 wheel 已安装；新增526整表／724恢复／32实时PDF组合逐值类型、所有权、并发、错误阶段、异常恢复及4MB转义输出限额检查通过。1500字符样本的引擎和语义渲染分别成功，累计串联因10M预算失败，确认不会按阶段重置。72模块来源审查和36 core＋3 paths＋1 scripts固定哈希通过。日志：build/native-table-materialization-rust.txt、build/native-table-materialization-install-test.txt、build/native-table-materialization-wheel.txt及build/rust-current-*.txt。

相关安装包回归共13组通过：新整表＋二维簇＋栏带＋准备＋精炼＋基础脚本＋完整引擎＋候选＋文字＋去重＋model_document＋兼容性＋smoke。model_document 的旧版面/公式与部分原生提供器使用冻结输入，另有真实CPU V3 PDF烟测；新整表套件的32组PDF原语则全部实时读取，不混淆两类证据。全量runner现有90组，本轮未重跑全部90组或36组像素对照；GPU未实机测试。回归日志：build/native-table-materialization-regression.txt。

## 历史检查点：二维簇与多行cell角色（2026-10-26）

新增`merge_native_inline_clusters(lines, *, page_size, table_bboxes=())`与`classify_native_table_script_cell(lines, *, page_size, tight_bboxes=None, origins=None, fraction_members=())`。完整移植旧二维inline簇的栏内连接、表格/图题/尺度跨栏保护、行内/紧凑公式簇判断、跨run成对角标补证、宿主评分和物化；宿主支持量复用CPython补偿sum，保留float(source_index)的平局和按源序拼接、结构归属、公式区域及排版重填。

合并返回`lines`与`groups`：lines是36个消费字段的普通JSON投影（32默认来自原_LineItem），groups保存每个输出簇的原输入行位置；不按source_index或字符ID解释。投影不包含私有缓存及其他LineItem字段，native_math_words用排序去重列表表示。单元格接口实际执行二维合并→脚本判定→词首抑制→连续同源角色冲突聚合，最终输出char_idx来源角色；跨行冲突不能先逐行删除body状态再合并。

冻结910组旧完整二维合并（262实际合并、18成对补证、155旧区间索引分支）、750组完整cell角色（428多行输入）及480叶子HTML链（160多行，无脚本标签）；另88组带标签正例由显式视觉行阶段输入构成，不当作实际PDF正例。48组原始字符→真实旧glyph生成→准备→合并/角色→HTML调用链均无脚本，用作误标负例。16空/控制字符簇与完整行边界、9分式尺度及48完整分段参考定位并修正了空字符串被错误缓存为不可打印的问题；保留旧Python空文本对邻接、可见列表和统计的影响。

空或非空批次≤512行，source_index为唯一i64、角度0/90/180/270；合计chars≤1500、表格屏障≤5000，输入及合并后单行公式区域≤2000，有限坐标abs≤1e9。序列化输入成本8MB及既有JSON节点/2MB文本上限保持；栏推断、连接、宿主、配对和脚本规模预收费共享10M Work，行内精炼保留自身预算。通过503行密集布局和转义文本膨胀验证超限明确报错、后续调用恢复，不降级成“普通未接纳”。

验证记录：225项Rust workspace测试、fmt、严格clippy和CUDA feature编译通过；新wheel已安装，新增合并/角色/HTML及空字符事实的逐值/类型、原输入成员完整性、所有权/并发、10M累计预算、转义文本序列化8MB限制、异常后恢复和许可证检查通过。9组相关安装包回归通过（新二维簇＋栏带＋表格准备＋脚本精炼＋基础脚本/叶子HTML＋完整表格引擎＋model_document＋来源审查＋smoke）。model_document仍为旧固定版面/公式和真实CPU OCR/SLANet的PDF多页编排对照；不据此承诺整文档效果改善。71模块来源审查、36 core＋3 paths＋1 scripts原样源码哈希通过；本轮未重跑全部89组安装包和36像素对照，历史全量记录保留。日志：build/native-inline-clusters-rust.txt、build/native-inline-clusters-install-test.txt、build/native-inline-clusters-regression.txt、build/native-inline-clusters-regression-run.txt、build/native-inline-clusters-wheel.txt和build/rust-current-*.txt。GPU未作实机测试。

下一步：接整表render_native_table_html_with_scripts、recover_table_result及PDF级原生/模型回退编排，再做整文档效果和性能验收。二维簇与多行cell角色阶段现已接通；完整整表HTML编排和默认替换尚未完成，legacy_native_table_recovery/legacy_table_execution仍False。visible-only、enhanced阅读顺序模型及GPU实机运行继续延后。

## 栏带检查点：完整原生栏带推断（2026-10-26）

新增`infer_native_text_lanes(line_geometry, *, local_page_width, median_height, recalculate_intervals=True)`。完整移植旧`_infer_text_lanes`编排，保留锚点资格、贪心中位数聚类、局部双栏候选/稳定平局、成员分配和可选栏宽扩张、跨栏续行、重复缩进及同高短尾批量结算。复用固定vendor原样rule_text.rs的三个数值内核，Rust适配入口另校验空集合、非有限数值和非法栏索引；没有新写栏位启发式替换旧规则。

冻结1128组旧完整调用参考，含182组局部双栏命中、24组重复缩进状态变化、47组跨栏短尾迁移；语料包含随机字体/尺度/语义、正文/通栏/局部双栏和已有旧表格视觉行。分支记录通过观察旧函数调用/返回获得，不替换分类器或注入栏位；每条输出的成员位置、顺序、栏边界、is_span和类型逐项对照。

输入为非空≤512条`(line JSON, 已转正bbox)`，source_index为唯一i64、同一行只出现一次；几何有限正向且abs≤1e9，字体名≤4096字节，字体签名使用普通二元JSON。字重只要求有限，不套坐标上限。返回`left/right/is_span/line_indices`，line_indices是原输入列表位置，不能按source_index或字符来源ID解释；保留旧空栏及成员顺序。内核规模预收费与外层簇比较、前序/并行行扫描、迁移共享10M Work预算，沿宿主JSON节点/2MB文本限制。

验证记录：222项Rust workspace测试、fmt、严格clippy和CUDA feature编译通过；新wheel已安装，新增1128组完整栏带逐值/类型对照、成员完整性、重复调用、所有权/并发、输入限制、503行有效密集布局累计10M预算拒绝与错误恢复通过。8组相关安装包回归通过（新栏带＋表格准备＋脚本精炼＋基础脚本/叶子HTML＋完整表格引擎＋model_document＋来源审查＋smoke）；model_document为旧固定版面/公式与真实CPU OCR/SLANet的PDF多页编排对照。70模块来源审查、随包许可证及36 core＋3 paths＋1 scripts原样源码哈希通过。本轮运行定向安装包回归，未重跑全部88组；历史87组全量及36组像素对照记录保留于上一检查点。日志：build/native-text-lanes-rust.txt、build/native-text-lanes-install-test.txt、build/native-text-lanes-regression.txt、build/native-text-lanes-regression-run.txt、build/native-text-lanes-wheel.txt和build/rust-current-*.txt。GPU未作实机测试，未承诺整文档质量/速度/内存提升。

下一步直接接旧二维重叠簇的连接、宿主选择与合并，再完成多行cell角色聚合、整表语义HTML与recover_table_result。当前栏带推断已完成，二维行合并仍未完成；默认PDF模型路径和legacy_native_table_recovery/legacy_table_execution=False不变。visible-only、enhanced阅读顺序模型及GPU实机运行继续延后。

## 表格准备检查点：表格视觉行和排版准备（2026-10-26）

本轮新增`measure_native_line_typography`、`prepare_native_table_script_cells`和`classify_native_table_script_line`。按旧Python计算裁页/旋转后的字高、字宽、主字体及覆盖率、主字重、行首强调/异族宽度和缩小引用句尾；字体平局保留首项，Unicode16全部1557个casefold映射固定随包。保留旧版空文本isprintable为真、合法字框仍参与统计的细节。

表格准备保持char来源首项/glyph来源末项、visual_row/x/y/glyph_id稳定顺序、多行才填排版、cell边界横线排除和分式来源保护；单行专用分类继续调用实际脚本流程，再抑制偏移词首并聚合同源冲突。准备接口返回fraction_rules和cells，每格包含glyphs/chars/visual_lines/fraction_members；visual_lines尚未二维合并。fraction_members与角色字典均用char_idx来源ID，不是列表位置。

988组旧排版度量（836常规＋112边界＋40既有em字段值/类型保留）、171组完整准备，以及3978组实际准备→单行分类→叶子HTML旧参考已冻结；其中168组上下标输出、43个分式保护cell、159组规则筛选场景（其中138组实际排除cell边界网格线）。单行路径不需要二维合并，不把这些对照当作多行整表自动语义完成。普通NativeTableInput/Result JSON要求cell(row,col)唯一；单行/单格≤1500字符/字形，页面≤20000字符/3000字形/2000cell/5000线，累计选择字形≤10000，有限坐标abs≤1e9。字体flags支持i64/u64及可转换文本，超范围报资源错误。网格比较与跨cell分式乘积共享10M预算，重复字形/字符累计序列化成本8MB在克隆前限制；宿主JSON预算不变，MIT/Unicode归属随包。

验证记录：220项Rust workspace测试、fmt、严格clippy和CUDA feature编译通过；最终wheel已安装到build/rust-wheel-env，87组安装包回归及18 RGBA＋18 RGB旧PDF精确像素对照全部通过。新增5137组事实（988排版＋171准备＋3978单行串接）逐值/类型、所有权、并发、共享10M工作预算及重复克隆8MB上限、异常恢复和随包许可证核对通过。69模块来源审查与36 core＋3 paths＋1 scripts原样源码哈希通过。日志：build/table-script-preparation-rust.txt、build/table-script-preparation-install-test.txt、build/table-script-preparation-wheel.txt、build/rust-current-*.txt和build/wheel-all-tests.txt。GPU仅编译检查，未作实机运行或性能/内存改善承诺。

下一步：移植旧text-lane推断与二维簇合并，接通多行cell角色聚合、整表render_native_table_html_with_scripts和recover_table_result，再接PDF级回退编排。默认PDF模型路径不变；legacy_native_table_recovery/legacy_table_execution仍False。visible-only、enhanced阅读顺序、GPU实机运行继续延后。

## 单行语义检查点：单行语义与分式保护（2026-10-26）

2026-10-26 单行语义与分式保护：新增`refine_native_script_tokens`、`classify_native_script_runs`、`classify_native_script_line`和`native_fraction_member_sources`。按旧Python移植引用/数学索引精炼、公式区域连续分段、首字母纠正和结构补证，单行路径实际调用基础分类后再精炼，不注入预分类角色。分式几何复用原样vendor的rule_text.rs；最终分子/分母恢复body，body_counts/formula_flags保持旧版最终覆盖前值。

冻结1838组精炼、2757组完整分段、560组完整单行、843组分式（255正例）以及20组真实分式检测→单行保护链；20组均实际改变角色，覆盖0/90/180/270度、来源ID不等于列表位置、旧空间索引分支及prepared零高分数线边界。数学标识政策按旧函数固定Unicode16的232区间。来源SHA随参考数据记录，独立wheel不导入旧Python。

接口约定：roles输出按chars位置；单行输出chars按旧规则排序后与roles对应；fraction输出及fraction_members输入均为char_idx来源ID。单行普通JSON支持chars、angle、inline_math_regions、compact_formula_cluster/restored_inline_cluster；side-map键是来源ID字符串。最多1500字符、5000规则、20000 side-map/保护ID、单字符文本256字节；坐标有限且abs≤1e9。精炼比较10M预算，分式规则×字符10M上限，沿既有JSON节点/2MB文本预算；超限显式错误。MIT/Unicode归属随wheel分发。

验证记录：218项Rust workspace测试、fmt、严格clippy与CUDA feature编译通过；新wheel已安装到build/rust-wheel-env，全部86组安装包回归及18 RGBA＋18 RGB旧PDF精确像素对照通过。新增6018组旧版事实（1838精炼＋2757分段＋560单行＋843分式＋20联动）逐值/类型、所有权、并发、输入限制和许可证检查通过。68模块来源审查及36 core＋3 paths＋1 scripts原样源码哈希通过。日志：build/script-refinement-rust.txt、build/script-refinement-install-test.txt、build/script-refinement-wheel.txt、build/rust-current-*.txt和build/wheel-all-tests.txt。CUDA仅编译检查，GPU运行仍待服务器验收。

剩余优先项：二维行合并、单元格视觉行/字体构建、表格网格线过滤/角色聚合/偏移词前缀、整表`render_native_table_html_with_scripts`和`recover_table_result`编排，再接原生/模型回退。当前完成单行语义，尚未自动生成整表语义HTML；默认PDF模型流程不变，legacy_native_table_recovery与legacy_table_execution仍False。visible-only、enhanced阅读顺序和GPU实机测试继续延后。

## 基础组件检查点：语义HTML基础组件（2026-10-25）

新增`classify_native_script_roles`、`native_numeric_script_indices`与`render_native_table_styled_cell`。前两者移植旧_script_geometry基础几何分类及数值上标补充，按loose/tight/origin、保护位置和原始字体身份判断；第三个是给定来源角色的旧单元格片段渲染，保留转义、紧框补空格、None来源样式继承和文字不匹配回退。渲染只生成固定sup/sub/b/i标签，preserve_font_styles默认False。

基础分类复用本地DocVortex b2ef8d0的一份原样scripts.rs，补齐CPython整数set槽位等距平局；原36份core/3份paths不变。1536基础分类参考含64实际PDF/模式几何、72等距平局，另302显式已有角色的数值补充（42正例）；142975非零Unicode16标志与全部1112064标量的零/非零标志逐项核对。1288叶子渲染参考含1174上下标输出、626字体标签输出；另390字体元数据转换参考，修正了大整数标志低位丢失、浮点字符串误作int及大字重受无关坐标上限限制造成的118条差异。叶子中的角色是该阶段显式输入，不把这些结果作为整表自动分类已经完成的证据。

接口约定：分类输出与输入chars的位置一一对应；protected_body_indices和数值补充输出均为列表位置，不是char_idx。渲染roles与chars_by_source的键则是glyph.source_index来源ID（普通JSON键为字符串）。基础几何接口限1500字符、单文本≤256字节、字符/字体≤1MB、side-map≤20000；渲染叶子限1500字形，沿既有JSON节点/2MB键文本及单元格parts工作预算，最终HTML≤4MB，回退分支也执行输出预算。

本轮最终验证：215项Rust workspace测试、严格clippy及CUDA feature编译通过；最终wheel已安装到build/rust-wheel-env，全部85组安装包回归与18 RGBA＋18 RGB旧PDF精确像素对照通过。1536基础分类/72集合平局、302显式角色数值补充、1288单元格HTML及390字体转换、所有权/并发/输入预算/HTML回退超限/许可证核对通过；全Unicode标志在Rust逐码点验证。67模块来源审查及原36 core＋3 paths、新增1 scripts原样源码哈希通过。日志：build/script-geometry-rust.txt、build/table-styled-cell-rust.txt、build/table-style-metadata-rust.txt、build/native-table-script-primitives-*.txt、build/rust-current-*.txt和build/wheel-all-tests.txt。

剩余：inline token精炼、分式保护、二维行合并、按cell自动生成角色、整表render_native_table_html_with_scripts和recover_table_result编排。visible-only与enhanced阅读顺序仍延后；默认PDF模型流程不变，完整PDF级原生恢复标志仍False。


## 字符去重检查点：DocVortex完整字符去重（2026-10-25）

新增`deduplicate_native_characters(chars)`和`read_pdf_native_text_geometry(source, *, page_index=0, pdfium_library=None, password=None, include_geometry=True)`。前者接收SDK已恢复代理对的普通字符JSON；后者直接读取PDF并完成代理恢复→同字形映射保护→重复绘制/平移副本来源归并→隐藏OCR副本抑制，返回字符与`loose_bboxes/tight_bboxes/origins`索引（JSON索引键为字符串）。核心复用原样vendor的text_snapshot.rs/dedup.rs，字体结构相等与signed-zero、固定Unicode16空白/NFKC/Han、Unicode17部首政策保持；仅同字形的异常Han映射规范化，不改写其他正文。

冻结840条旧完整去重参考（包含64组实际PDF/扩展模式及6字体/角度相等平局），4964 NFKC非恒等码点与1010角度cos/sin/round事实。64组实际PDF整链几何参考另行从旧函数独立读取，不从Rust回推。PDFium自身在实际PDF读取时已过滤部分重复绘制，真实Han多码值合并有正例；其他删除路线以独立旧完整字符输入核对，不把这些输入冒充实际PDF删除正例。

去重边界为5000条记录、总来源≤20000、单映射≤256字节、文字/字体合计≤8MB、绘制候选访问≤200万及隐藏配对保守上界≤2000万；索引唯一、来源非空、几何有限且abs≤1e15。超限/输入错误传播，原raw接口20000字符上限不变。`docvortex_character_dedup/docvortex_pdf_native_text_geometry=True`；旧raw API与默认PDF模型路径不变。visible-only裁剪/遮挡及适配器语义上下标HTML仍待接入，PDF级完整恢复标志仍False。

本轮最终验证：213项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；最终wheel已安装到build/rust-wheel-env，全部84组安装包回归与18 RGBA＋18 RGB旧PDF精确像素对照通过。840去重输入逐值/类型、64实际PDF整链几何、16组实时读取字符和绘图→完整引擎评估/恢复/诊断（不注入冻结字符）、并发/所有权/输入预算/错误恢复及随包许可证核对通过。66模块来源审查、36份core与3份paths原样源码哈希通过。日志：build/character-dedup-*.txt、build/rust-current-*.txt、build/wheel-all-tests.txt及build/wheel-all-run-output.txt。

下一步：移植table_materialization.recover_table_result及table_text_styles.render_native_table_html_with_scripts，复用已就绪的字符/紧框/原点和NativeTableResult，补齐语义上下标HTML后再接默认模型回退编排；visible-only裁剪/遮挡另行验收，enhanced阅读顺序继续延后。


## 字符原语检查点：DocVortex字符原语（2026-10-25）

新增`read_pdf_character_primitives(source, *, page_index=0, pdfium_library=None, password=None, include_geometry=True)`。Rust直接读取旧DocVortex去重前的`chars`，并分别返回代理对恢复后的`restored_chars`；原始码值、来源索引、字体、页内对象编号、渲染模式、生成标记、原点、书写方向和可选loose/tight字框完整保留。基础布局沿旧整数ceil页面尺寸，扩展几何沿浮点CropBox；含隐藏文字时才按旧规则读取透明度可见标记，页面上没有隐藏模式时不新增该字段。

58组真实PDF/扩展模式、7650原始记录及恢复结果已独立冻结，包括50组既有PDF与8组补充ToUnicode/旋转/Form/隐藏透明文字；补充平面代理对、一个字形多码值、孤立高低代理均由实际PDFium读取。上限20k字符/1MB单字体名/8MB累计字体名/32MB宿主JSON，text/page/document按逆序在全局锁内释放，不输出原生地址。该接口不执行DocVortex重复绘制/隐藏副本去重，也不启用visible-only页面裁剪与遮挡过滤；不能拿它替代完整`get_chars_with_geometry`或默认PDF表格恢复。下一步接通这些旧规则，再接适配器语义HTML；完整恢复标志仍False。

本轮最终验证：211项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel已安装到build/rust-wheel-env，全部83组安装包回归与18 RGBA＋18 RGB旧PDF精确像素对照通过。新增58组字符/7650原始记录及代理恢复逐值/类型/可选字段、并发/所有权/密码和错误恢复通过；绘图41页及16组引擎串接再次通过。65模块来源审查、原36份core及新增3份paths源码哈希通过。日志：build/pdf-character-primitives-*.txt、build/rust-current-*.txt、build/wheel-all-tests.txt与build/wheel-all-run-output.txt。


后续接续：优先复用固定vendor/docvortex-core的text_snapshot.rs与dedup.rs，补齐字体等价类、固定Unicode16空白/NFKC/Han与Unicode17部首等价数据，再执行mapping_groups→collapse_paints→hidden_pairs/suppress_hidden；保留来源归并、平移证据、桶候选上限和旧隐藏OCR容错门。先把去重后的实际PDF字符与现有绘图读取接入完整表格引擎，再完成适配器语义上下标HTML；visible-only裁剪/遮挡作为另一个明确边界验收。

## 绘图检查点：DocVortex PDF绘图原语（2026-10-25）

Rust端DocVortex绘图原语读取已接通。`read_pdf_vector_geometry`直接读取PDF页，返回视觉坐标`drawing_lines/path_infos`以及`page_size/page_bbox/rotation`；路径保留来源索引、Form层次、填充颜色和内部矩形。复用固定上游三份原样Rust源码，在现有PDFium全局锁内借用同库20个符号，不再次初始化引擎；对象/段计数先受限，再读取几何。旧深度15、裁剪/旋转、非均匀矩阵、零面积框过滤和共线合并保持。独立基线覆盖41页实际PDF、428条合并线、3694路径；505 hypot位值、507 round3和180合并事实。16组实时绘图＋冻结旧原始字符已接完整表格引擎对照。

验证：210项Rust workspace测试、最终PDF定向复测、fmt、严格clippy及CUDA feature编译通过；新版wheel已安装到build/rust-wheel-env，82组完整安装包回归和18 RGBA＋18 RGB精确像素对照通过。新增41页逐值/类型、16组实时绘图引擎串接、所有权/并发/密码与错误恢复及随包许可证核对通过。64模块来源审查、原36份core与新增3份paths原样源码哈希通过。日志：build/pdf-vector-primitives-*.txt、build/rust-current-*.txt、build/wheel-all-tests.txt和build/wheel-all-run-output.txt。

剩余：旧DocVortex字符多码值恢复/几何/去重、适配器语义上下标HTML和PDF级恢复；现有pdftext字符接口不能替代。`docvortex_pdf_vector_geometry=True`，`legacy_native_table_recovery/legacy_table_execution=False`，默认PDF流程未切换。


2026-10-25 补充真实PDF预览：新增`scripts/migration/preview_pdf.py`，使用独立已安装wheel、本地真实V3/公式/OCR/UNET-SLANet双模型CPU推理，不固定版面/公式响应、不调用旧Python解析器。`demo/pdfs/示例1-论文模板.pdf`全部14页输出8403字符Markdown、1张HTML表格、6份资产（解析33.828秒）；`demo/pdfs/比亚迪财报.pdf`全部9页输出38671字符Markdown、9张HTML表格、12份资产（解析29.401秒）。均为144DPI、CPU单线程、原生文字优先、公式开启；耗时不含共享模型加载，不是与旧版同条件性能对比。退出码0，页数和Markdown图片引用存在性检查通过。结果位于`build/rust-pdf-preview/two-real-pdfs-20261025/`，不代表表格/公式准确率、完整默认配置或整文档质量已验收。

## 上一检查点：给定原始字符/页面原语的完整表格引擎（2026-10-25）

- 验证：208项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel在独立环境完成全部81组安装包回归和18 RGBA＋18 RGB旧PDF像素精确对照。724完整引擎评估/恢复/诊断逐值及类型对照、候选身份保留、所有权/并发与共享预算（失败后可再次恢复）通过；63模块来源审查和36份vendor哈希通过。日志：build/rust-current-*.txt、build/native-table-engine-rust.txt、build/native-table-engine-budget.txt、build/wheel-all-tests.txt及build/wheel-all-run-output.txt。

- 接通旧engine.py核心_evaluate、recover_native_pdf_table与diagnose_native_pdf_table；原始字符在Rust内构建NativeTableText，各候选实际执行，不注入预制文字/候选/选中结果。公开evaluate_native_table_primitives、recover_native_table_primitives、diagnose_native_table_primitives。
- 保留原执行顺序、数值重分行及撤销欠分割矢量、alias空白行/物理漏行/叶子列下界、line/rect尝试冲突和唯一rect漏行解围、rule_band先于hybrid、文本守卫、多行及banded条件回退、表头欠分割数值覆盖、各来源阈值和稳定平局。生成/保留/选中及移除诊断使用内部候选身份索引，不用结构相等替代对象身份；选中aligned时文字也切换到aligned_text。
- 724条未经mock的完整原始输入参考，547恢复覆盖六类来源，9物理拓扑冲突，3含移除记录、2 aligned接纳；16实际旧DocVortex PDF/角度原语本轮包含真实原始字符、线、矩形，Rust仍使用冻结JSON而不读取这些PDF。完整恢复字段/普通HTML及所有路线重放诊断逐项对照。
- 各生成器及文字构建新增内部with_work入口，共用一次调用的实际1000万保守工作预算；原公开阶段调用仍各自独立，未改变其策略/返回合同。新增独立阶段均能完成、整链诊断累计超限的回归；错误后后续恢复仍可正常执行。旧文档表格阶段误写2000万，已按现有Work实现修正为1000万，运行时上限未改；公式ftfy独立2000万不变。
- 输入为NativeTableInput普通JSON，不实现任意Python对象属性coerce适配；原始字符≤20000/字形≤3000/物化格≤2000、有限几何abs≤1e9。早期原语计数最多读20000、实际相交>5000返回primitive_limit；进入生成后沿各路线原语上限。资源/输入错误传播，不当作普通未接纳。
- native_table_engine=True；PDF级legacy_native_table_recovery和legacy_table_execution仍False。下一步Rust DocVortex PDF字符/绘图原语提取与适配器语义上下标HTML，再把原语级引擎接进PDF路径并做整文档新旧差分；默认模型路径、enhanced/其他语言/GPU实机延期不变。

## 已完成检查点：banded_numeric分带数值候选（2026-10-25）

- 验证：206项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel在独立环境完成分带数值、多行稀疏、融合、三线、数值对齐、文本三路、引擎守卫、完整矢量、拓扑、物理证据、候选、文字、兼容性与多页PDF十四组定向安装回归。1751候选逐值/类型/诊断/重投影普通HTML、所有权/并发/预算及后段仲裁通过，另1843数字规则/12净空/164有符号set事实在Rust核对。62模块来源审查、36份vendor哈希通过；最终wheel包含更新PSF说明及同步来源哈希。全量runner现80组，本轮未重复上一检查点79组全量及PDF像素回归。日志：build/rust-current-*.txt、build/native-table-banded-numeric-*-rust.txt/build/native-table-banded-numeric-rust.txt及build/native-table-banded-numeric-*-final.txt。

- 移植旧banded_numeric：数值/单位/状态正则语义、重复行众数及平局首遇优先（3/4列两条旧正例）、完整外线或同高相邻填充表头、所有相关字形的共同净空、正文整字唯一归属与数值校验、独立局部横线证明的三层表头rowspan/colspan，原始反向框/边界行为保留。
- 保留100%捕获、零歧义和score≥0.98门；不擅自添加100%顺序门，原95/96顺序/score候选仍接纳。全部单元格从原始字符重投影，正文去空白必须一致；表头沿旧行为可重建或变空，不用新的文字修补策略代替旧语义。
- 新增build_native_table_banded_numeric_candidate，返回candidate/diagnostics/html。1728主＋21边界＋2众数平局参考共1751条（262非空、16旧实际PDF/角度原语）；1843旧numeric正则、12共同净空、164有符号来源set事实及3旧非法来源ID异常。补充覆盖builder接受但不完全捕获的最终拒绝、正文/表头跨字、缺列和重投影文本拒绝；原参考未覆写。
- 原始来源字典保持重复键首次插入位置，源ID无效时沿本分支严格报错。复用已有空间投影，扩展已许可CPython3.14整数set适配至abs≤1e12有符号来源/重复输入（-1 hash为-2），既有物理线索引set入口算法不改。资源为每类5000drawing/rectangle、3000字形、20叶子列/2000物化格、20000原始字符和共享JSON/10M保守预算。
- native_table_banded_numeric=True；完整原生恢复和legacy_table_execution仍False。下一步串完整_evaluate的调用顺序、冲突/覆盖和最终接纳，再接Rust DocVortex PDF原语提取及语义HTML；默认PDF模型路径、enhanced/其他语言/GPU实机延期不变。

## 已完成检查点：完整多行稀疏候选与整行分组标题回退（2026-10-25）

- 验证：203项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel在独立环境完成全部79组安装包回归及18 RGBA＋18 RGB旧PDF像素精确对照。1136候选/诊断/普通HTML及356空间投影逐值/类型对照，Rust另核114紧框间距事实；所有权/并发/资源门及后段仲裁通过。61模块来源审查与36份vendor哈希通过。日志：build/rust-current-*.txt、build/native-table-multiline-rust.txt、build/native-table-projection-rust.txt、build/wheel-all-tests.txt及build/wheel-all-run-output.txt。

- 接通旧sparse_multiline主执行链：原列证据、表头边界、物理行带与关键列runs、keyed短runs/对称居中续行、filled首列rowspan、两层表头、空白走廊和完整性发布；只有主候选失败才调用grouped_rule_band，诊断保留原先后顺序。
- 回退沿独立横线带证明整行组标题colspan；只认领同格左右可见来源之间的显式空白，保留原始来源重复键的首次插入顺序。标题由原始字符空间投影重新复核，紧框/font/字号/书写角度/基线等可靠证据才补空格，剔除空白后不一致则拒绝。
- 新增generate_native_table_multiline_candidates与project_native_table_spatial_text；1031主＋105独立补充实际旧参考（548非空、54成功分组回退）、356空间投影、114紧字框间距，含16旧实际PDF/角度原语。主参考最初未触发成功回退，保留后另冻独立居中/跨列样本，未覆写期望；不是Rust PDF原语提取或整文档效果验收。
- 每类5000drawing/rectangle、3000字形、20推断列/2000物化格、20000原始字符、投影≤100万列/800万字符、有限几何/连续row_index/有效字形引用和共享JSON/10M保守预算；Unicode alnum/alpha/space/decimal复用已固定Unicode16数据，不改既有文字路线。
- native_table_multiline/native_table_spatial_projection=True；完整原生恢复和legacy_table_execution仍False。下一步banded_numeric，再串完整_evaluate、Rust DocVortex PDF原语提取及语义HTML；默认PDF模型路线、enhanced/其他语言/GPU实机延期不变。

## 已完成检查点：sparse_hybrid物理/文字融合候选（2026-10-25）

- 验证：200项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；新wheel在独立环境完成全部78组安装包回归及18 RGBA＋18 RGB旧PDF像素精确对照。799条hybrid参考逐值/类型/诊断/普通HTML、所有权/并发/资源门及后段仲裁通过；60模块来源审查与36份vendor哈希通过。日志：build/rust-current-*.txt、build/native-table-sparse-hybrid-rust.txt、build/wheel-all-tests.txt及build/wheel-all-run-output.txt。

- 移植旧sparse_hybrid完整候选路线：最长连续稠密行选择/后段平局优先、分段竖线75%覆盖、矩形端点、物理假设优先与ties-even去重、文字median轨道、逐列60%/总体80%锚点支持（复用精确statistics.mean）、相邻行间最长横线优先吸附、两层表头rowspan/colspan与20%/80%局部分隔证据。
- 粗token只有跨越强物理边且不横切字形墨迹才可拆分；保留两层表头原token列zip配对及nearest首项平局，完整100%捕获/顺序、零歧义和0.98发布门，多个成功拓扑冲突时不发布，不把低分候选升级。
- 新增generate_native_table_sparse_hybrid_candidates；返回候选/诊断/普通HTML。740主＋59独立补充实际旧版参考（571非空、16旧实际PDF/角度输入），覆盖行轨拒绝、分段竖线、物理额外列、局部分隔、缺列与最终完整性。原参考未覆写，Rust此处仍使用给定PDF原语和NativeTableText。
- 每类5000drawing/rectangle、3000字形、推断最多200稠密列/物理可多一列、2000物化格、有限几何/连续row_index/有效字形引用及共享JSON/10M保守工作预算；公开入口不缓存跨调用状态。
- native_table_sparse_hybrid=True；完整原生恢复与legacy_table_execution仍False。下一步完整sparse_multiline及grouped_rule_band回退、banded_numeric，再串完整_evaluate和Rust DocVortex PDF原语提取/语义HTML；默认模型路径、enhanced/其他语言/GPU实机延期不变。

## 已完成检查点：rule_band三线表与共享稀疏列证据（2026-10-25）

- 验证：199项Rust workspace测试、fmt、严格clippy及CUDA feature编译通过；独立安装wheel完成三线表、数值对齐、文本三路、引擎守卫、完整矢量、拓扑、物理证据、候选、文字、兼容性、多页PDF十一组定向回归。59模块来源审查与36份vendor哈希通过。全量runner现77组，本轮未重复此前70组全量与PDF像素回归。日志：build/rust-current-*.txt及build/native-table-rule-band-*-final.txt/build/native-table-rule-band-final.txt。

- 移植旧rule_band：三条完整横线及内部竖线硬门、1..3条视觉表头与≥3条表体、普通单行表头留原路线、换行表头折叠、至少两描述列同步重启的rowspan、表头占用和token不得跨列、墨迹空白走廊、逻辑边界及完整性发布门。新增generate_native_table_rule_band_candidates，候选/诊断/普通HTML与旧版逐项对照。
- 共享移植sparse_multiline所需的重复最大2..20列、全局净空/median回退、全token细化、重复矩形端点优先、填充行带/evidence分类、按字形中心首项inclusive占用、空白走廊（≥3可比行/80%稳定）与高公式行拒绝。物理边界只跳过支持计数，不绕过墨迹跨界门；新增inspect_native_table_sparse_columns诊断入口，尚不执行完整sparse_multiline。
- 从aligned_numeric抽出未改逻辑的_local_rules到table_native_sparse，数值路线继续复用相同原语判向/裁剪。median聚类保持与当前簇median比较，不能换成相邻点链或均值聚类。
- 386条主三线表（97条非空）＋2条严格发布补充、386条共享列/矩形/占用/空白走廊参考（16旧实际PDF/角度输入）、195条旧median聚类事实。两个补充在builder已接纳且candidate_rejection_gate=None时，分别因未100%捕获或少量歧义而不发布；不放宽0.98/100%顺序/零歧义门。既有参考未覆写。
- 资源：每类≤5000 drawing/rectangle、3000字形、20叶子列/2000物化格，有限几何/尺度abs≤1e9，连续row_index及有效visual_row/字形引用、共享JSON/10M保守工作预算。规则候选一条调用只发布0或1个，不自动进入生产PDF路径。
- native_table_rule_band/native_table_sparse_columns=True；legacy_native_table_recovery和legacy_table_execution仍False。下一步继续sparse_hybrid、完整sparse_multiline及grouped_rule_band回退、banded_numeric，再串完整_evaluate；PDF DocVortex原语提取与语义HTML仍待接入。默认模型路线、enhanced/其他语言/GPU实机延期不变。

以下为历史检查点。

补充金额/状态PDF参考：另8条由两份实际PDF经旧DocVortex提取的角度输入，2条非空候选；与主218条分开冻结，独立wheel逐项通过。合计226条候选/重分行、125非空、16旧PDF/角度事实；Rust此处未执行PDF提取。

## 已完成检查点：aligned_numeric数值对齐候选（2026-10-25）

- 验证：196项Rust workspace测试、fmt、严格clippy、CUDA feature编译通过；独立安装wheel完成数值对齐、文本三路、引擎守卫、完整矢量、拓扑、物理证据、候选、文字、兼容性、多页PDF十组定向回归。主218条/532 regex在Rust逐项对照，wheel含补充8实际金额/状态PDF角度输入共226条（125非空、16旧PDF/角度事实）；所有权/并发/资源限制/后段仲裁通过。58模块来源审查和36份vendor哈希通过；审查范围补充后重建同一二进制wheel，数值/兼容性最终复测通过。全量runner现76组，本轮未重复此前70组全量与PDF像素回归。日志：build/rust-current-*.txt及build/native-table-aligned-numeric-*-final.txt/build/native-table-aligned-numeric-final.txt。

- 移植旧aligned_numeric实际执行链：按字形中心median重分行、0.35倍原字宽重新token化、数字右缘分组、金额共同净空/表头扩列、章节/单位/标签逻辑表体、分组表头colspan/左栏rowspan和额外状态列跨行，以及最终表头来源跨格拒绝。保留原Glyph ID稳定排序、source_index/spacing和视觉行更新，不重新估算字号。
- 数值表达式沿旧regex的Unicode decimal、正负号/缺省横线、金额小数/分隔符和百分比语义；左右括号独立可选的原quirk保留，不按新语法“修正”。isdigit上标门与regex decimal区分；时间、完整物理格线、超宽数值列等拒绝规则不改。
- 新增regroup_native_table_numeric_text与build_native_table_aligned_numeric_candidate，后者默认先regroup，返回重分行text、numeric_groups、candidate与普通html；regroup=False要求row_index等于数组位置，资源/输入错误明确传播。此处是给定页面原语/字形，不读取PDF，也未接完整引擎的aligned覆盖决策。
- 218条实际旧重分行/分组/候选参考（123条非空候选，8旧实际PDF/角度原语），532条实际旧numeric_token regex逐项对照；包含左/右括号quirk、Unicode数字、表头净空/状态列、章节单位/多行标签、高数字框、重复ID与显式空白。已有参考未覆写；未被旧执行链调用的numeric_region_bounds辅助函数未移植，不宣称整个文件每个辅助函数完成。
- 资源：≤5000 drawing、3000字形、2000原子物化格，坐标/spacing/字号尺度abs≤1e9，调用内共享JSON/10M保守工作预算；regroup单独调用有独立预算。普通HTML继续复用已对照候选输出器，不含语义上下标。
- native_table_aligned_numeric=True；legacy_native_table_recovery和legacy_table_execution仍False。下一步rule_band/sparse_hybrid/sparse_multiline/banded_numeric与完整_evaluate（含数值覆盖/表头欠分割决策）；随后合流PDF DocVortex原语提取/语义HTML。默认模型路径、enhanced/其他语言/GPU实机延期不变。

以下为历史检查点。

## 已完成检查点：少线/无线/key-value候选与引擎诊断守卫（2026-10-25）

- 验证：194项Rust workspace测试、fmt、严格clippy、CUDA feature编译通过；独立安装wheel完成文本三路、引擎事实守卫、完整矢量、拓扑、物理证据、候选、文字、兼容性和多页PDF九组定向回归。375候选/351守卫逐值对照及输入所有权、并发、资源限制、后段仲裁集成通过；57模块来源审查与36份vendor哈希通过。全量wheel runner现75组，本轮未重复此前70组全量与PDF像素回归。日志：build/rust-current-*.txt、build/native-table-text-grid-*-final.txt/build/native-table-text-grid-final.txt及build/native-table-engine-policy-final.txt。

- 移植旧text_grid.py完整三路：sparse_grid少线/底纹、≥3列无线text_grid、两列key_value；新增generate_native_table_text_candidates，返回候选、诊断和普通HTML。不以缺失分支的空列表假装完整引擎。
- 沿旧重复最大token列数、相邻净空median、60%锚点/净空支持、首项inclusive单调落列、子集续行和稠密行歧义规则。续行只取严格占用子集且无贯穿横线，保留原group_occupancy；逻辑行稳定排序合并但落格仍使用原NativeTableText及原字形来源。
- 移植单/多分组表头colspan、下层叶子列归属/最先等距、表头全列覆盖和需rowspan时拒绝；稀疏物理支持按原长横线和可见填充底纹计数。评分/atomic-token门复用已对照候选构建器，不提升低分结果。
- 375条实际旧三路候选逐值/类型/诊断/HTML参考，含8条旧实际PDF/角度原语及字形事实、60%门、表头/非单调、续行/稠密歧义与真横线。运行时不调用生产Python；这些仍不等于Rust已提取DocVortex PDF原语或整文档效果完成验收。
- 新增inspect_native_table_engine_policy：输入几何/严格正面积相交原语门，alias影响空白行、嵌套漏行诊断、仅最终规范轨道的列下界、line/rect尝试网格冲突、唯一verified rect逐行完整性解围。返回事实与原候选索引，不执行完整_evaluate。
- 引擎守卫351条实际旧函数参考通过。输入早期门在旧_evaluate的text阶段固定None隔离，其他守卫直接调用旧函数。首版参考33/34号记录因脚本浅拷贝诊断在后续被改动，原native-table-engine-policy.json保留；新stable文件深拷贝后重新冻结351条，不改Rust或覆写旧期望来通过测试。
- 资源：文本候选5000原语/3000字形、最多200列/2000物化格，三路共享JSON/10M保守工作量预算；引擎事实检查允许最多20000输入原语以复现“只数相交原语”的5000门，1000候选/attempt及累计20000候选格，显式null诊断集合报错。均只支持有限普通JSON字段。
- native_table_text_candidates/native_table_engine_policy=True；legacy_native_table_recovery和legacy_table_execution仍False。下一步优先移植无条件参与旧_evaluate的aligned_numeric（含regroup），随后rule_band/sparse_hybrid/sparse_multiline/banded_numeric并串完整引擎；PDF原语提取和语义HTML仍待接入。默认模型路径、enhanced/其他语言/GPU实机延期不变。

以下为历史检查点。

补充9条完整路线边界：部分续行在真纵隔断/无隔断时的旧区别、alias冲突和残留窄带拒绝，已在独立wheel逐项对照；完整矢量样本共268条，轨道阶段204条。主参考未覆写。

## 已完成检查点：给定页面原语的完整矢量候选生成（2026-10-25）

- 验证：192项Rust workspace测试、fmt、严格clippy、CUDA feature编译通过；独立安装wheel完成完整矢量生成、拓扑、物理证据、候选、文字、兼容性和多页PDF七组定向回归。268条完整矢量＋204轨道阶段（含8旧实际PDF/角度原语）逐值/类型对照、完整888 isdigit属性与所有权/并发/共享预算/后段仲裁集成通过。55模块来源和36份vendor哈希通过。本轮未重复此前70组wheel全量与PDF像素回归，全量runner现73组。日志：build/rust-current-*.txt及build/native-table-vector-pipeline-*-final.txt/build/native-table-vector-pipeline-final.txt。

- 接通旧_build_vector_tracks与build_vector_candidates：给定drawing/矩形原语和NativeTableText→原始证据→受条件约束halo→矩形晶格→规范轨道→稠密基线/尺寸硬门→拓扑/物理证据→文字/评分/普通HTML。新增generate_native_table_vector_candidates与轨道阶段inspect_native_table_vector_tracks。
- 保留drawing/rect分路、raw→supported切换条件（含高分但非零隔断歧义）、supported失败时原候选/诊断保留的旧组合、track_hypotheses快照顺序、候选输出line先rect后。不以新网格策略替代旧逻辑。
- 保留单行/单列halo提示，80%矩形晶格/双轴重复、原正线宽median、窄空带/alias冲突/≤200轨道/≤10000原子格的原拒绝顺序；短竖装饰只在旧空白外带或横切正文字形条件下删除。装饰阶段仍用raw表框，后续证据才normalize，反向框quirk不统一改写。
- 稠密基线沿旧最大列占用、部分续行/真实纵隔断和答题列表头例外；isdigit使用固定Unicode16 decimal及128额外码点，完整888码点与旧运行时核对，不依赖宿主Python或Rust Unicode版本。
- 248条旧完整矢量生成＋204轨道阶段，另11条补充（8实际PDF/角度原语、201轨道/超万格/5000展开片段边界），候选/诊断/普通HTML逐项对照；既有参考未覆写。8条实际PDF在旧进程提取几何后冻结，Rust此阶段使用给定原语，不能当作Rust已读取DocVortex几何或整文档效果已验证。
- raw/supported/rect通过内部with_work共享一次调用10M保守工作预算；缓存仅保存当前调用按来源/halo区分的原始证据，阶段不再重置预算。原独立prepare/canonicalize/topology入口合同与边界保留；内部允许规范化前超过200轨道，再按旧track_count拒绝，不能由公开单轴helper的输入上限提前改变拒绝门。
- native_table_vector_generation=True；legacy_native_table_recovery和legacy_table_execution仍False。矢量路线已串联，但完整引擎还缺其他文本/稀疏候选、完整_evaluate及语义HTML，Rust DocVortex PDF字符/绘图原语提取也待接入。默认模型路径不变；enhanced/其他语言/GPU实机继续延期。
- 下一步接完整引擎的早期决策/拒绝门及其他旧候选路线，再与PDF原语提取合流；不能把矢量单路线直接提升为完整原生恢复默认后端。

以下为历史检查点。

## 已完成检查点：规范轨道到矢量拓扑、物理证据与候选（2026-10-25）

- 验证：188项Rust workspace测试、fmt、严格clippy、CUDA feature编译通过；独立安装wheel完成拓扑/候选、物理证据/轨道、原生候选、文字、兼容性与多页PDF六组定向回归，421条旧拓扑/候选逐值/类型对照、104精确均值及所有权/并发/资源限制/最终仲裁集成通过。54模块来源审查、36份vendor哈希通过。全量wheel runner已扩到72组，本轮未重复此前70组全量与PDF像素回归。日志：build/rust-current-*.txt及build/native-table-vector-grid-*-final.txt/build/native-table-vector-grid-final.txt。

- 新增build_native_table_vector_candidate：给定已验证规范轨道/合并线与NativeTableText，接旧_build_vector_topology及_materialize_vector_candidate；返回topology/candidate/diagnostics/html，不从PDF读取或自动挑选轨道假设。
- 隔断按strict坐标与alias覆盖分别计算；保留缺边≤1-0.8连接、0.2二进制边界、严格>5%歧义拒绝、alias恢复计数与影响行。复用未改vendor并查集/连通分量，L形整体拒绝、退化网格门顺序不改。
- 接单行/单列外框与全部隔断物理证据、0.8门与0.95..1分数校准、字形穿隔断/穿行判定；按旧开放外缘端点支持计算物理行。视觉行/token中心inclusive首项落带，独立验证空行可补行稳定度；alias影响空行仍不能发布。
- 结构分取隔断决断度精确均值、物理线数量、歧义与单行/列置信的min；后续复用旧文字落格/来源/普通HTML，并执行单行/单列100%文字完整性和空行发布门。statistics.mean的[0.5,1]子域使用固定vendor精确二进制和/整数分母一次ties-to-even舍入，未以普通浮点平均替代。
- 409条实际旧规范轨道→拓扑/候选主参考＋12条补充边界，104条旧statistics.mean参考逐值对照。内容/来源、质量、诊断、拒绝顺序、普通HTML与数值类型均覆盖；全为合成/边界输入，不作为真实PDF整文档效果证明，已有参考未覆写。
- 输入上限：10000原子格、物化最多2000格；行列≤199、各轴≤200轨道、每轨道≤200递增alias、5000物理线/3000字形；有限坐标abs≤1e9、非负容差，结构/坐标一致性预检与共享JSON/保守10M工作量上界。非空文字行全部无token显式报错，对应旧ZeroDivisionError，不伪造稳定度。
- native_table_vector_topology=True；legacy_native_table_recovery和legacy_table_execution仍False。下一步接旧_build_vector_tracks（halo/矩形晶格/稠密基线等拒绝门）与drawing/rect多假设选择，把已完成前后段串成矢量候选生成；完整_evaluate、文本/稀疏候选、语义HTML及Rust DocVortex PDF原语提取仍待接入。默认模型路径不变，enhanced/其他语言/GPU实机延期不变。

以下为历史检查点。

追加等距边界：14条实际旧最近物理线参考表明，等距时Python整数索引set的遍历顺序会影响外缘折叠。已按CPython3.14的有界整数插入/扩容/遍历语义保留该行为，204条集合遍历参考通过；PSF许可和来源随wheel分发。旧370条轨道参考未覆写。

## 已完成检查点：物理线段证据与规范轨道（2026-10-25）

- 验证：185项Rust workspace测试、fmt、严格clippy、CUDA feature编译全部通过；独立安装wheel完成物理证据/轨道、原生候选、原生文字、兼容性及多页PDF五组定向回归（720条新旧物理/轨道对照，另204整数set参考）。53模块来源审查和36份vendor哈希通过。本轮未重复此前70组wheel全量及像素回归；已将新套件加入全量runner（现71组）。日志：build/rust-current-*.txt、build/native-table-vector-*-final.txt及build/native-table-vector-final.txt。

- 沿旧vector.py移植drawing/矩形裁剪与四方向局部坐标，保留halo仅在原表框完全无交集时扩大裁剪、矩形segment_count=5/可见性/85%面积门、线长与snap门。不使用drawing标注方向直接替代几何方向。
- 按旧当前簇均值吸附，不改为相邻点链；调用固定vendor合并内核连接共线间隙。保留重复长线端点/外缘门推断开放外框、raw/supported两套轨道及装饰短横线过滤，零halo的5000展开片段硬门不改。
- 移植轨道规范化：外缘halo吸附、坐标alias合并、空窄带折叠、alias跨度冲突、独立物理双边界保留与唯一性；同分/最先命中顺序沿旧版。坐标分类不依赖宿主Python。
- 新增prepare_native_table_rule_evidence与canonicalize_native_table_tracks，给定JSON页面原语/浮点轨道输入。336旧物理证据＋370规范轨道参考逐项对照，既有参考未覆写；包含旋转、反向/退化框、halo、85%边界、合并间隙、重复/嵌套线、随机几何、缺字窄带和多物理边界。该批为合成/边界参考，不冒充真实整文档对照。
- 资源合同：≤5000输入线/矩形、矩形展开最多20000片段，规范化≤200轨道/3000字形中心、有限坐标abs≤1e9、非负容差和共享JSON/10M工作预算；超限显式失败，不返回部分轨道。旧5000片段门仅用于原算法的零halo分支，额外资源上限独立明确。
- native_table_vector_evidence=True；**完整物理候选仍未接通**。下一步接拓扑合并与物理行/单行/单列证据评分，再串联多假设选择、完整_evaluate和语义HTML；Rust DocVortex PDF字符/绘图原语提取仍需对照。legacy_native_table_recovery和legacy_table_execution仍False，默认模型路线不变。enhanced/其他语言/GPU实机延期不变。

以下为历史检查点。

## 已完成检查点：给定网格的原生表格候选构建（2026-10-25）

- 验证：181项Rust workspace测试、fmt、严格clippy、CUDA feature编译；70组独立安装wheel全量回归、18 RGBA＋18 RGB精确像素对照；52模块来源审查和36份vendor哈希全部通过。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt及build/wheel-all-tests.txt。
- 移植旧candidate.py：逻辑占用/跨度验证、原spec顺序平均轨道与索引、索引缺轨道回穷举；复用未改动vendor的字形落格。保留旧重叠同分取后项、中心点回退和歧义门槛，索引/穷举在不一致物理规格上的旧差异不强行统一。
- 候选包含单元格文字/原字符来源索引、字形分配、capture/token split、顺序一致性、结构质量夹取与min评分，以及拒绝诊断。重复字形ID映射后项覆盖但单元格累加、整数/浮点bbox类型和未知diagnostics字段保留。三个结构质量指标须由上游显式给出；构建器不提前应用最终来源门槛，后续仍交已有候选仲裁。
- 新增inspect_native_table_candidate、build_native_table_candidate、render_native_table_candidate；普通HTML保留原跨度属性顺序、按列稳定排序、空行和quote=False转义，不猜表头。尚无自定义语义上下标渲染。
- 327实际旧主候选＋22边界参考、4 HTML，逐项对照网格、轨道索引、每字归属、拒绝原因、内容/来源、分数与JSON类型；参考不覆写。额外覆盖100字重复ID、零面积/相切中心、合并格缺轨道、补偿求和抵消和低分候选交最终仲裁。独立wheel另测输入所有权、并发、种子诊断和资源错误。
- 范围是**给定物理网格**；native_table_candidate_builder=True，legacy_native_table_recovery及legacy_table_execution仍False。缺向量线/矩形/文本物理网格发现、多候选生成、完整_evaluate、语义HTML与Rust DocVortex PDF字符几何提取；默认解析路线不变。最多2000规格/10000逻辑格/3000字形，逻辑坐标/跨度abs≤1000、物理坐标abs≤1e9，共享JSON/输出/工作量预算继续生效。
- 下一步沿旧版实现物理网格候选，之后串联完整恢复。enhanced、其他语言、GPU实机继续按用户要求延期。

以下为历史检查点。

更新：2026-10-25。阶段：**P00 基线持续扩充；P01 CPU 运行库与 wheel 探针已通过；P02/P03/P07 已有实现，P04 正在接入数值内核。尚未完成完整解析器迁移。**

## 已完成检查点：原生表格字形、视觉行/token与单元格文字（2026-10-25）

- 验证：178项Rust workspace测试、fmt、严格clippy、CUDA feature编译；69组独立wheel全量回归及18 RGBA＋18 RGB像素对照通过。最后补充反向表框分支后，完整Rust及文字/兼容性wheel定向复测通过；51模块来源和36份vendor哈希通过。日志：build/rust-current-*.txt、build/wheel-all-tests.txt、build/native-table-text-final.txt、build/native-table-text-compatibility-final.txt。
- 新增`prepare_native_table_glyphs`与`build_native_table_text`：按旧字符中心选择/原索引稳定排序、显式空格与控制换行、ligature/排版标点归一化、局部裁剪与四方向旋转、0.5高度/CJK守卫，生成带来源索引的字形、视觉行与token。复用固定vendor的视觉分行内核，不新增几何排序规则。
- 保留visible bbox与advance/spacing bbox分离：重复来源编号按原输入最后有效loose bbox赋值（局部裁剪None也覆盖）；低标点只在分行时借用字体高度，最终字形仍用可见框；单行间隔及token切分沿用旧阈值。
- 新增`build_native_table_cell_text`：按旧全表视觉组做有限几何细分，保留水印/上下标组；返回text、逐片段parts和分组索引。算法插入空格的来源为None，原字符保留source_index；Latin/NFKC Unicode16边界、日期、紧凑单位、URL和连字符规则逐项对照，不删除原连字符。
- 362条实际旧pending/完整文字构建参考（含16组真实DocVortex PDF字符几何×角度输入）、1934条实际旧单元格文字/来源片段、841条跨行边界对照通过。另12条反向表框参考保留旧quirk：字形选择用归一化表框，advance裁剪仍用原表框，不能统一坐标“修复”后改变空格输出。既有参考未覆写。已安装wheel验证JSON数值和类型、输入所有权、并发、资源错误；Unicode16 printable/decimal与Latin/NFKC属性冻结，运行时不依赖宿主Python Unicode/NumPy。
- 仅接受给定JSON字符/字形；**尚未对齐Rust提取DocVortex字符几何、字符归属到单元格、物理候选/完整候选生成、打分及HTML/上下标物化**。不把pdftext现有读取自动当作DocVortex几何，也不改变默认模型路线。native_table_text_preparation/native_table_cell_text=True，legacy_native_table_recovery仍False。
- 输入char字段为str/null，合法Unicode/有限坐标；≤2万输入字符/3000输出字形或单元格字形、坐标abs≤1e9、来源索引abs≤1e12、单元格输出≤1MB及共享JSON/工作量预算。char_idx支持有界JSON标量和Unicode十进制/下划线整数字符串，解析失败回到原枚举索引；不承诺Python任意大整数或定制对象协议。
- 下一步沿旧candidate.py移植网格校验、字符归属、文本落格和候选评分，再进入物理网格候选生成；Rust PDF字符几何需另做旧DocVortex对照后接入。enhanced、其他语言、GPU实机延期不变。

## 已完成检查点：原生表格捷径策略与已有候选仲裁（2026-10-25）

- 验证完成：174项Rust workspace测试、fmt、严格clippy、CUDA feature编译；68组独立安装wheel测试、18 RGBA＋18 RGB精确像素对照通过；49模块来源记录、36份vendor哈希通过。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt及build/wheel-all-tests.txt。
- 移植旧`_process_single_table`的DocVortex捷径守卫：use_rule_table/engine/force_ocr、页面OCR和字符串旋转标签、源PDF、公式/checkbox中心落框（含相切）与原图正面积相交。坐标保留公式页面像素、图片PDF point、表框除scale的旧约定；不是实验native_table优先规则。
- 移植结果接纳：HTML按旧Python空白strip后非空且confidence≥0.90；保留原HTML，不额外要求<table>，移除layout_image_list并写rule_table_score/table_parse_method。原始数值类型和未知表字段保留，返回副本。
- 移植已有原生候选的两段helper：欠分割向量过滤、来源0.95/0.98门槛、verified line唯一优先、line/rect异构冲突、来源稳定优先与严格<0.05分差拒绝。返回原索引；**未移植完整_evaluate_native_pdf_table、候选生成和HTML/上下标物化**。
- 547条实际旧捷径守卫/结果接纳、425条实际旧候选helper对照通过；另9条旧引擎配置参考区分缺省、显式null和非字符串，保留仅缺省取默认值的语义，未覆写已有参考。多页入口新增显式native_table_recoverer适配点；采用独立冻结的实际旧PDF原生结果，另两条五页组合对照JSON/Markdown/content_list/资产像素，验证接纳后不需表格模型，空结果或普通异常回模型、ImportError传播。该证据只验证编排，未执行Rust原生生成后端；既有6条模型组合继续保留。
- 未提供适配器时默认仍走原模型路径；没有把实验网格恢复器当成DocVortex恢复。legacy_native_table_policy/native_table_candidate_arbitration=True，**legacy_native_table_recovery=False**，完整legacy_table_execution仍False。SDK仍不依赖生产Python，显式恢复器由调用方提供，接口同时供后续Rust后端接入。
- 新代码限制：JSON深度64/20万节点/字符串2MB；≤1000公式/复选框/图片/候选，候选格累计≤2万、逻辑坐标≤1000、有限坐标绝对值≤1e9、正scale≤1000，沿用共享工作量预算。恢复输入错误先抛出；恢复器内部普通异常保留error_type诊断后回模型，ImportError保持抛出。
- 下一步移植旧原生表格字符归一化、分行/token与物理候选生成，再接完整候选诊断和HTML物化；此前不能宣布原生表格恢复完成。enhanced、GPU实机、其他语言延期不变。

## 已完成检查点：旧UNET与默认双模型表格路由（2026-10-25）

- 验证完成：170项Rust workspace测试、fmt、严格clippy、CUDA feature编译；67组独立安装wheel测试、18 RGBA＋18 RGB精确像素对照通过；47模块来源记录、36份vendor哈希通过。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 新增`match_legacy_unet_table`/`recognize_legacy_unet_table_png`，复用已对照的UNET格框、逻辑格、跨格拆分/同行归并和噪声边缘裁剪；为旧已转义OCR/公式/图片占位独立保留raw HTML，不再转义。原安全预览输出器不改。无格框返回空串，无OCR时跳过结构；错误显式传播，对应旧`enable_blank_cell_rec=False`默认配置。
- 新增`select_legacy_table_candidate`/`recognize_legacy_table_auto_png`：只准备一次OCR/提示输入，按旧分类命中的候选先执行，分数严格低于0.9或显式compare时才运行另一候选；复用Paddle/Qanything共识。保留原HTML的<td/<th字面前缀计数（含thead quirks）、BeautifulSoup文本空格判断、已转义OCR的原文命中和旧决策优先级，最终选中后归一化。
- 证据：28旧UNET原HTML合同、5给定OCR真实UNET及5实际全图OCR→UNET；237旧候选选优/指标（含富文本/实体/注释/脚本/template/CData）、12固定响应阈值/调用顺序；5正常与5强制比较的真实CPU双分类器/UNET/SLANet链。所有最终HTML与实际旧函数对照，未覆写旧参考。
- `parse_pdf_models`新增`unet`、`unet-slanet-plus`和`unet-slanet-1m`配置，保留原`slanet-plus/1m`默认参数。双模型传`wired_model`为UNET、`table_model`为SLANet、`classifier`为Paddle、`secondary_classifier`为Qanything。SLANet、UNET、双模型三配置×原生优先/强制OCR共6条实际旧五页组合，页面JSON、Markdown/content_list及18份RGB资产像素精确一致；版面/公式响应仍固定，不能据此宣称任意文档质量等价。
- 尚缺原生表格结构捷径（旧默认use_rule_table）、可选空格补识别、word-box/批失败重试、自动文档方向/分类、印章/checkbox和Office；`legacy_unet_inference`/`legacy_table_model_routing=True`，完整`legacy_table_execution`仍False。UNET/选优资源及有限HTML树约束继续生效，复杂容错HTML可能明确unsupported。GPU实机/enhanced/其他语言仍按用户要求延期。
- 下一步以实际`recover_docvortex_table`为基线接原生结构捷径及模型回退，保留源PDF、页面旋转、公式/原图相交和0.90置信门槛；不能把已有实验native_table仲裁接成旧默认。随后扩大真实整文档语料/性能对照。

## 已完成检查点：多页PDF模型预览入口（2026-10-25）

- 验证完成：168项Rust workspace测试、fmt、严格clippy、CUDA feature编译、65组独立安装wheel测试、18组RGBA＋18组RGB精确像素对照通过；45模块来源记录、36份vendor哈希通过。日志仍为build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt及build/wheel-all-tests.txt。
- 新增Python `parse_pdf_models`：PDF→RGB→V3矩形版面→旧公式任务/原生文字或OCR→旧SLANet→页面组装/资产→文档后置OCR→公式编号→段落→跨页表格→Markdown/content_list。返回原始页号、页面JSON和`images/`资产bytes，无需生产Python/NumPy/Pillow/ORT Python包。仅在内存返回完整结果，不自动写目录。
- 新增最终`normalize_legacy_table_html`和`recognize_legacy_slanet_table_png`。12936旧CJK字符串、2270实际旧BeautifulSoup默认HTML、6真实SLANet最终HTML通过；不改变td/th嵌套标签内文字，没有直接文本变化则原串返回。注释、实体、属性及变化后的规范序列化抽样对照，有限HTML树并非任意HTMLParser等价。
- 独立安装wheel与实际旧BatchAnalyze→result_to_middle_json对照：2条五页组合（原生优先/强制OCR），包含正文、表格、图片、公式和中间/末尾空页；页面JSON、Markdown、content_list、6份资产RGB像素精确一致。OCR/SLANet实际CPU执行，版面与公式响应固定；另真正V3模型完成PDF冒烟，不能把固定响应证据当成全模型整文档等价。
- 参考生成初版误用表格1.6扩框给正文，原参考`model-document.json`保留；新增按旧正文1.8/表格1.6配置的`model-document-default.json`及独立图片前缀参考。未为通过测试改写既有期望或Rust默认参数。
- 预览须显式选择SLANet-plus/1m和native_text，不自动分类。缺必要模型/识别失败/资源超限直接报错；正文OCR逐页执行，跨页识别批组成与旧全批可能不同，尚未证明任意文档数值等价。后置OCR按整文档执行。默认最多128页（上限256）、单页1600万/总1.28亿像素、资产/后置OCR各128MB，模型裁图仍限400万像素；共享JSON预算继续生效。
- 尚未完成：默认UNET旧matcher和分类/双模型仲裁、原生表格结构捷径、word-box/批失败重试、自动文档方向/分类、印章/checkbox、Office及生产API全面兼容。`model_pdf_preview=True`，`legacy_table_execution`及完整解析器兼容仍False。enhanced、其他语言、GPU实机按用户要求延期。
- 下一步优先接旧UNET匹配与默认表格路由，使多页入口覆盖现有Python默认表格配置，再扩大真实PDF整文档差分与性能验证。

## 已完成检查点：旧SLANet matcher与raw HTML（2026-10-25）

- 验证完成：165项Rust workspace测试、fmt、严格clippy、CUDA feature编译、63组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照通过；43模块来源检查、36份DocVortex vendor哈希通过。6个真实结构样本的坐标逐值一致，raw HTML/匹配/逻辑/诊断类型严格一致；6 CPU OCR串联中2无字跳过结构。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 新增独立match_legacy_slanet_table与recognize_legacy_slanet_raw_png，复用现有plus/1m结构前处理/网络/解码及旧空OCR/提示输入。接通已定方向输入→可选全图OCR→结构网络→旧matcher raw HTML；仍无OCR时不要求或调用结构模型。现有安全预览输出器及生产Python不改。
- 保留四点OCR min/max归并、仅下界clip min/仅上界clip max、位于所有单元格之上的OCR剔除；按1-IoU、distance和首项同分规则匹配，不直接以最大IoU替代。匹配索引保留过滤后的OCR顺序。全部整数OCR坐标时保留整数dt_boxes诊断类型，含float时按旧NumPy数组提升。
- raw HTML保留原token和已转义文本，不再次转义、重排或强行校验网格。多行文本按旧空白/粗体合并；过滤完整thead/tbody chunk，包含恰好等于这些标签的内容。逻辑点按旧跨度/占用规则，保留独立<td>跳过后续token、零/负跨度等原解析quirks；资源超限明确失败，不冒充语法修复。
- 冻结54旧几何、156 HTML/逻辑、96格式→过滤→匹配及6实际结构样本；plus/1m各3种图，含旧模型在空白图上的非规范结构，未接安全预览规则替换。另6 CPU OCR串联（其中2条无字跳过结构），结构网络在非空分支实际执行。真实样本的给定OCR和真实OCR分开列证据，不代表任意文档质量。
- 增加OnnxSession.table_structure_legacy_png：F32坐标完整提升为F64再序列化，不沿用预览JSON短小数回读的约1e-5像素偏差；原table_structure_png接口不改。6个结构样本坐标逐值一致。置信度仍沿用F64均值，与旧F32均值实测差≤1e-5，仅元数据，不参与本matcher选择；raw HTML/匹配/诊断类型严格对照。
- 输出字段明确为raw_html，尚未执行RapidTableModel最终normalize_table_html_cell_text/BeautifulSoup+CJK归一化，不能当成完整表格最终HTML。UNET、分类器/双模型仲裁、原生结构捷径、OCR word-box/异常重试及正式多页API仍待接入。legacy_slanet_matcher/legacy_slanet_raw_inference=True，legacy_table_execution仍False。
- JSON输入深度64/20万节点/2MB字符串；≤1000 OCR/10000 cell、20000 token、1000万匹配/逻辑工作量、10万逻辑占用格、输出预检≤4MB；PNG源限400万像素/64MB。有限坐标abs≤1e9、ASCII整数跨度abs≤1000；空cell或三空列进入matcher明确失败，列数不符新增严格报错而非旧zip截断。结构/匹配异常显式传播，输入副本和流所有权沿用原合同。

## 已完成检查点：旧表格空OCR全图回退（2026-10-24）

- 验证完成：161项Rust workspace测试、fmt、严格clippy、CUDA feature编译、62组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照通过；42模块来源检查、36份DocVortex vendor哈希通过。8实际CPU全图OCR和16模型输入回退的框/文本/RGB精确，最大分数误差0；结构网络未执行。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 新增recognize_table_full_ocr_png与prepare_table_model_input_png，复用既有DB/CTC、框整理、裁图及提示payload；已定方向/skip_table_orientation=True上下文。回退在原RGB图检测，不先涂公式遮罩；只排序/按公式切框，不merge/宽3过滤/薄图白边重试。真实直接TextRecognizer的默认ch配置与batch6单列，和前轮只测行级rec适配器区分。
- 保留RapidOcrModel.__call__默认drop_score=0.5：先过滤分数，再按旧表格文字规则转义，空文本只要分数足够仍保留。不把此过滤加到旧行级OCR或既有三列结果。检测本身为空则不要求识别器；检测非空但全部被公式切掉时按旧空列表识别失败显式报错。
- prepare_table_model_input_png只在给定OCR为假时调用全图回退；已有三列包括[[],[],[]]直接复用，无需det/rec session。回退后仍空返回ocr_result=None、needs_ocr=False、can_infer=False，跳过全部图片/公式；否则接既有提示阶段并返回can_infer=True。低级prepare_table_model_payload_png的needs_ocr待处理合同保留，不能把can_infer当作已执行结构网络。
- 冻结243条实际旧全图流程策略、81条0.5阈值/转义策略及识别前裁图像素；另8条真实CPU直接det+rec参考（4图×两种公式提示）和16条实际旧predict模型输入回退（4图×None/[]/三空列/既有低分结果）。空白/12px薄图不重试，结构网络仅捕获输入，未执行。
- 默认ch/non-seal、有限RGB8 PNG64MB/400万像素、1000检测框/识别裁图、累计3200万裁图像素/128MB PNG、共享JSON及工作预算不变。数量不符新增严格报错，不复现旧zip截断；无RTL、word-box、逐图失败重试或direct predict额外方向猜测。输入不修改，流只读一次且不关闭，tRNS保留原RGB。
- legacy_table_full_ocr=True，legacy_table_execution仍False。下一步结构网络执行、旧matcher/HTML兼容编排及候选比较；原生结构捷径、OCR word-box/失败重试、正式多页API继续接入。enhanced、其他语言SDK和GPU实机验证延期不变。

## 已完成检查点：旧表格公式/图片提示与模型输入（2026-10-24）

- 验证完成：157项Rust workspace测试、fmt、严格clippy、CUDA feature编译、61组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照通过；41模块来源检查、36份DocVortex vendor哈希通过。4条实际PDF/CPU模型输入链的框/文本/RGB精确、最大分数误差0；结构网络未执行。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 新增prepare_table_hints与prepare_table_model_payload_png，复用已有表格OCR/原生文字/方向与normalize内核，不改生产Python。公式/checkbox按旧useful_list变换、完全在裁图外才剔除，保留相切与未夹紧坐标；latex和checkbox各自保留真值，插入时latex优先。公式框转F32后进入三列OCR，公式文本只按旧table规则转义，可显式传inline_delimiters（默认两侧均$）。
- 图片元数据按原生包含且IoU<0.9优先，只要选到任意原图即不补布局图；无原图才全部使用layout_image_list，不新增区域筛选。页面rotate_label为90/180/270时禁止原图。保留ori_bbox/ocr_bbox及page.table_fill_image_list累计列表；返回page/table JSON副本，PIL/PNG像素由调用方按uuid另持有，不承诺跨调用Python对象别名。
- 模型输入保留points_to_bbox取p1.x/p1.y/p2.x/p3.y（不是min/max）、int向零取整且右下包含的OpenCV白矩形；逐张图插入uuid并删除其包含的此前OCR/图片占位，之后插公式。已转义OCR不重复转义，uuid原文保留。空[]/None返回needs_ocr=True，表示尚需旧predict全图OCR分支；[[],[],[]]按旧truthy允许插提示，不能混同无OCR结果。
- 冻结241条实际旧提示选择/坐标参考、343条实际RapidTableModel.predict结构输入/精确像素参考；结构网络是捕获器。另4条真实PDF→DB/方向/原生或CTC→提示→模型输入链（2页×强制/非强制OCR），公式和布局图片元数据为给定测试输入，不代表公式/图片检测或提取，也未执行结构网络。
- PNG限RGB8/64MB/400万像素，元数据输入深度64/20万节点/2MB字符串；1000图片/公式列表、2万OCR总行、1000万包含比较工作量、重复矩形面积6400万像素、累计图片元数据2000项，分隔符每侧≤64字节。无效结构/列数不符/超限显式失败，输入不修改，流仅读一次且不关闭，tRNS不合成白底。
- legacy_table_hints和legacy_table_model_payload=True，legacy_table_execution仍False。旧空OCR的全图OCR分支、OCR word-box/批失败重试、原生表格结构捷径、结构模型执行/匹配与HTML兼容编排及正式多页API仍待完成；enhanced/其他语言/GPU实机延期不变。

## 已完成检查点：旧表格原生文字优先（2026-10-24）

- 验证完成：154项Rust workspace测试、fmt、严格clippy、CUDA feature编译、60组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照通过；40模块来源检查、36份DocVortex vendor哈希通过。新原生/CPU专项另确认无需识别session即可使用原生结果、强制OCR缺识别器时报错、输入不修改及并发/流所有权。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 新增extract_table_native_text_png及recognize_table_text_stage_png：接通初次检测/方向/旋转后重检测→PDF文字优先→整体结果为空时行级OCR。保留force_ocr、ocr_enable、表格角度必须为0及页面rotate_label排除90/180/270守卫；只提取到部分原生行也不补OCR，不能以新融合策略替代旧逻辑。返回最终RGB PNG/检测框/转义后三列ocr_result及text_source。
- 原生入口支持word_boxes=True，输出逐字符框，不把它误称为OCR模型词框。保留旧宽3过滤/斜框纠正、除scale向零取整、char_idx稳定顺序、Unicode/连字替换及间距策略；字符框文本沿用旧raw字符后table归一化，和整行content的连字处理不同。旧稀疏字符/对比度裁图仍用全局bbox作用于表裁图，不擅自改偏移。
- 复用现有字符填充内核；表格将每个JSON字符出现位置投影为独立pdftext.Bbox对象，因此重复字典的list.index按对象语义处理。现有普通JSON正文填充仍保留原值相等语义，未改生产Python。262旧坐标映射、262原生填充（含字符框）、10实际PDF填充逐字段/类型对照；另28条实际_process_single_table固定检测/识别响应隔离整表回退策略。
- 实际PDF填充参考为5页×2种行/字符框模式；10条真实旧PDF→DB→方向→原生文字/CTC参考为5页×强制/非强制OCR（2原生、8 OCR），实测框/文字/分数均精确，最大分数误差0。结构模型只捕获输入，不执行，不代表完整表格或整文档质量。最早固定检测响应使用整数坐标；另存typed-detector参考，按真实DB返回的F32列表浮点类型重做28条策略，旧参考保留，未按Rust结果改写期望。
- 原生入口要求RGB8 PNG≤64MB/400万像素、1000检测框；共享JSON深度64/20万节点/2MB字符串与1000万工作量，原生补识别裁图≤3200万像素/128MB PNG，逐字符输出≤20000框。坏参数/资源异常显式抛出而非旧except吞掉后回退OCR；页/表JSON不修改，流仅读一次且不关闭。综合入口目前仅行级OCR，旧OCR word-box及批失败逐图回退仍待迁移。
- legacy_table_native_text=True，legacy_table_execution仍False。原生表格结构捷径、公式图片填充、实际结构模型与正式多页API继续接入；enhanced/XY-cut、其他语言SDK与GPU实机测试延期范围不变。

## 已完成检查点：旧表格方向准备（2026-10-24）

- 验证完成：151项Rust workspace测试、fmt、严格clippy、CUDA feature编译、59组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照通过；39模块来源检查、36份DocVortex vendor哈希通过。方向专项另验证模型异常直接传播、调用前页面预检、共享session并发及流只读一次。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 新增plan_table_direction与prepare_oriented_table_text_png。按实际txt_most_angle_extract_table投票：页面/文字角度叠加、Python ties-even取整、标点边界包含及同票首项；页面rotate_label为90/180/270时跳过原生文字。无角度时才用旧长宽比>1.2、竖框比例>=0.28且数量>=3门槛，空框直接允许分类。
- 保留_process_single_table给分类模型传入BGR ndarray的通道顺序，现有显式RGB分类API不改。90逆时针、270顺时针并重新检测；180及其他原生角度不转图。旋转后沿用原公式遮罩坐标，不擅自修正旧行为。返回最终RGB PNG与框，可供后续原生文字/OCR及结构模型使用。
- 冻结320组旧角度/门槛、18组旋转/通道像素、76条实际_process_single_table固定响应方向链（含4条旋转公式遮罩），以及28条实际CPU检测/方向准备链。其中2条确实运行方向网络，其他按旧门槛跳过；后两项未执行OCR识别与结构模型，不能作为完整表格质量证据。早先非竖版参考保留，额外竖版/空白及彩色参考另存，未覆盖冻结期望；相同320组计划在各文件重复保存，不累计为960组。
- JSON适配将pdftext span.bbox.bbox投影为普通四坐标；角度接受JSON number或ASCII无下划线float字符串，其他字符串格式可先转JSON number。共享深度64/20万节点/2MB字符串、PNG64MB/400万像素/1000检测框及合成文字角度绝对值1e12度预算；失败不修改page/table，流仅读一次且不关闭。
- legacy_table_direction=True；legacy_table_execution仍False。原生表格路线、原生文字优先/word-box、公式图片填充、实际结构模型和正式多页API仍待接入。enhanced/XY-cut、其他语言SDK与GPU实机验证按原范围延期。

## 已完成检查点：旧表格行级OCR与模型结果回填（2026-10-24）

- 验证完成：148项Rust workspace测试、fmt、严格clippy、CUDA feature编译、58组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照通过；38模块来源检查、36份DocVortex vendor哈希通过。8条真实表格CPU链的框/文本精确，识别分数实测最大误差0。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 在现有legacy_model新增table_text，复用inference DB/CTC/crop内核。新增recognize_table_text_png、organize_table_detection_boxes、apply_table_ocr_results、apply_table_prediction与OnnxSession.detect_table_png。只覆盖已选方向之后的OCR阶段和模型HTML返回后的末段；原生表格捷径、方向选择/旋转、原生表格文字优先、word-box、公式图片插入及结构模型编排仍待接入，legacy_table_execution=False。未把现有实验表格策略接成旧默认，也未修改生产Python。
- 保留项目ocr_patch固定960、表格unclip1.6/box_thresh0.5/dilation，不进行正文merge；先用局部公式/checkbox遮罩检测，初次为空且短边<32时8px白边重试，重试不再切公式框，NumPy2 F32减padding/clip后不重排。薄图对齐成0时旧检测返回None，表格入口按空检测处理以继续重试；正文/通用预览原有错误合同不改。
- 识别裁图来自未遮罩原图，pipeline旋转门槛2；不应用正文宽3过滤或0.5/0.8低分策略，保留空文字和未三位舍入的分数。source只解码一次生成批裁图。normalize_table_ocr_text按旧Python空白/香→否/哦樂→哦/单ASCII数字號及html.escape处理；返回旧HTML转义的三列ocr_result，不能当成现有实验matcher的原文line字典。
- 模型HTML回填保留首<table>至末</table>、精确大小写/标签限制、反序标记生成空串、失败保留已有html/box字段、无论成功与否删除layout_image_list；成功时保存整页公式/checkbox bbox及有bbox键的fill image ori_bbox，按scale除后向零取整，不自行筛选表内框。64组实际_process_single_table固定模型响应仅隔离末段，不冒充实际结构模型证据。
- 131旧表格框整理、75旧薄图恢复、209行级OCR策略、64HTML回填及遮罩/白边/4裁图像素精确对照；8真实CPU DB→CTC参考另列，含12×320薄裁图成功恢复的两条链。框/文本精确，真实识别分数误差限1e-5；None空检测投影为[]。结果数量不符显式报错而非旧zip截断，旧批失败逐图重试/word-box回退未实现，错误直接传播。
- RGB8 PNG≤64MB/400万源及重试像素，最多1000框/遮罩、重复遮罩面积≤6400万、识别裁图累计3200万像素/128MB PNG；重试配置padding0..1024/短边0..65536，只在实际触发后分配白边。返回模型副本，流仅读一次且不关闭；表格RGB tRNS不合成白底。
- 下一步：原生表格路线/方向及原生文字优先、公式图片填充、实际结构模型执行，再接正式多页API。enhanced/XY-cut、其他语言SDK继续延期，GPU实测留服务器。

## 已完成检查点：公式默认ftfy修复（2026-10-24）

- 验证完成：144项Rust workspace测试、fmt、严格clippy、CUDA feature编译、57组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照通过；37模块来源哈希、36份DocVortex vendor哈希检查通过。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 在现有inference crate新增unicode_fix内核，按旧ftfy6.3.1默认fix_text迁移：10种单字节编码恢复、坏字节/空格恢复、Java NUL/CESU8、嵌入乱码、C1、连字/字宽、引号/换行、ANSI/控制字符、HTML实体及Unicode16 NFC。规则/实际编码映射/3501实体和原库4模块SHA冻结；NFC锁定unicode-normalization0.1.24/Unicode16，不使用主机Python版本的Unicode表，不导入Python ftfy。
- 29517组实际旧ftfy输出逐字符串精确；覆盖全部有效Unicode标量、每个规范NFD（含Hangul）、全部实体/数字边界/多行HTML auto状态及乱码样本。另29517组实际旧公式post_process验证LaTeX→ftfy完整顺序；1352旧tokenizer用例全部精确，包括此前18组拒绝分支，原参考未覆写。LaTeX的normalize仍保持注释状态；ftfy本来启用的默认NFC独立保留。
- 新增fix_formula_text独立完整后处理入口，repair_formula_latex仍只执行ftfy之前的旧修复。FormulaDecoder和真实M会话自动使用默认ftfy，移除ASCII/基本汉字恒等子集限制。特别保留HTML auto跨行禁用状态、实体生成的<不会回头禁用当段修复、数字分号实体仍按旧规则保留、ftfy产生的新反斜杠/引号不再回到LaTeX阶段等细节。
- 新增希腊/全角/中文3张合成图的2批真实旧M CPU参考及跨页回填检查；加前轮共6批/6张图，不代表真实文档公式质量、S/L/GPU已验证。capabilities中formula_full_ftfy=True、formula_ftfy_version=6.3.1；完整解析器仍未完成，生产rapid_doc未切换。
- 仅迁移公式实际使用的默认选项和合法Unicode字符串；不提供ftfy自定义配置/说明步骤，也不接收孤立Python surrogate。每输入/输出64KB、内部临时文本256KB、修复/编码128轮、递归64、共享工作量20M；解码批和模型跨batch共享文本预算，模型累计输出2MB，超限明确报错且不返回部分结果。Apache-2.0/PSF及Unicode归一化许可随wheel分发。
- 下一步：旧表格执行与回填、正式多页API，并继续扩充真实文档对照；enhanced/XY-cut、其他语言SDK及GPU实机验证按原范围延期。

## 已完成检查点：公式token解码与M模型CPU链（2026-10-24）

- 验证完成：141项Rust workspace测试、fmt、严格clippy、CUDA feature编译、56组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照通过；36模块来源哈希、36份DocVortex vendor哈希检查通过。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 在已有inference crate内增加formula模块，复用已移植预处理。新增FormulaDecoder、repair_formula_latex及OnnxSession.recognize_formula_batch_png；execute_model_formulas可直接使用该原生会话，按旧页序完成真实模型识别/回填。tokenizers固定0.22.2，与旧Python依赖一致。SDK需要Python宿主，Rust内核不调用Pillow/NumPy/torch或Python tokenizers。
- 冻结673实际旧修复、1352 tokenizer用例及4组额外token配置。保留首EOS=2截断、特殊token/未知ID、中文text包装和双引号移除、旧括号移动时Python混合list索引、各环境补齐、up/不支持命令移除；align*在旧正则中未转义的量词行为也保留，不自行修复。normalize仍不启用。
- 额外配置参考使用formula-token-configs-v2.json：旧构造器会pop传入配置的added_tokens_decoder，初版误保存调用后的输入；保留初版，v2在调用前复制输入，未修改旧期望来迁就Rust。旧tokenizer的Fixed4096 padding也保留在冻结IDs中，安装对照按16行分组满足token总量预算。1352组中18组需要未迁移ftfy，测试明确验证unsupported而非跳过。
- M模型来自旧配置固定URL，593908338字节/SHA256=52afa60ff51ad66cd7840a74188a058fe99081e617d3bd780117a2d67d94f50d；迁移脚本下载并校验，SDK不自动下载。4条真实旧RapidFormula CPU链覆盖3张合成图（2张简单公式、1张空白），batch1/2；旧CPU不改写ArgMax图。空白旧模型输出\_照原保留，不增加去空白启发式。不是实际文档公式语料质量证明。
- 完整ftfy尚未迁移：只准入修复后ASCII/基本汉字、无HTML实体且ftfy为恒等变换的子集；希腊Unicode、全角、乱码/组合字符/控制符及HTML实体等明确unsupported，不能把这些分支计为完整公式兼容。formula_model_inference=True只表示已有原生推理路径，formula_full_ftfy=False；S/L仅有形状合同，当前真实模型证据为M。生产rapid_doc仍未切换。
- 元数据≤8MB、词表≤100000+额外token≤1000、每行≤8192/累计≤128000token、≤1000行、单文本64KB/总输出2MB；ByteLevel解码前按词长预检，括号扫描10M。推理batch1..64、源PNG合计128MB/32M像素、累计张量32M/预处理256M工作量，沿用单图预算，释放GIL并持有会话锁。通用ftfy和旧CUDA专用ArgMax优化仍待迁移；GPU实测留服务器。
- 下一步：补齐通用ftfy并扩充真实公式文档对照，再接表格执行和正式多页API；enhanced/XY-cut及其他语言SDK继续延期。

## 已完成检查点：PP-FormulaNet Plus 输入预处理（2026-10-24）

- 验证完成：138项Rust workspace测试、fmt、严格clippy、CUDA feature编译、55组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照通过；35模块来源哈希、36份vendor哈希检查通过。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 新增inference/formula_preprocess及内部pillow_resize，Python入口prepare_formula_input。移植实际旧crop_margin、Pillow双线性resize、thumbnail bicubic/整数reduce、居中黑边、float32归一化后BGR灰度、16倍数补齐。S/M默认384×384；L显式768×768。返回RGB原始字节、小端FP32 NCHW字节和裁边/缩放元数据，不需要Pillow/NumPy/OpenCV运行库。
- 88组实际旧预处理参考，覆盖默认384/768、随机尺寸、非16倍数、常色/白边/彩色、窄高及长公式、reduce余边和Pillow极窄高图的纵向优先缩放。最终RGB逐字节一致；张量绝对误差上限1e-6，独立wheel实测最大2.38418579e-7。差异来源为公开OpenCV FMA表达式与旧5.0 IPP的运算次序，不能宣称逐位一致或未经真实网络验证就宣称输出等价。冻结文件及SHA未按Rust结果改写。
- RGB PNG≤64MB/400万源像素，中间图≤1600万像素且每轴≤65536，目标每轴2..2048，工作量≤2.56亿；尺寸/中间放大图/工作量在分配或循环前检查，超限明确失败，不把先放大小边改成直接缩小。目标边1不在支持范围，旧squeeze/format路径存在维度异常。流不关闭、并发/损坏图/尺寸合同均验证。
- Pillow MIT-CMU与OpenCV许可/来源随wheel分发。未修改生产rapid_doc，也未增加Python依赖。capabilities公式预处理为True，formula_model_inference仍为False；tokenizer解码、LaTeX修复/ftfy及实际公式ONNX推理为下一步，之后接表格执行和正式多页API。enhanced/XY-cut、其他语言SDK继续延期，GPU留服务器。

## 已完成检查点：跨页公式任务执行与回填（2026-10-24）

- 验证完成：136项Rust workspace测试、fmt、严格clippy、CUDA feature编译、54组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照通过；34模块来源哈希、36份vendor哈希检查通过。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- 新增legacy_model/formulas及Python入口execute_model_formulas/apply_model_formula_results，按旧页序和页内公式顺序统一调用适配器，再只用非空字符串更新latex；None/空串保留已有字段，空白、中文、数学Unicode和LaTeX命令不改写。保留未知字段与大整数，返回计划副本。
- 100组实际旧BatchAnalyze._run_formula_recognition固定响应差分；仅证明任务顺序/回填策略，未运行公式网络。安装验证另覆盖跨页顺序、损坏PNG/缺图在推理前失败、共享流跨页仅读一次且不关闭、识别失败/结果数量异常不修改原模型。
- 明确差异：旧zip会静默截断结果，新入口要求结果等长；只接收准备阶段完整有序的8/13/14类别公式任务。单RGB PNG≤64MB/400万像素，重复任务输入累计≤128MB/3200万像素；1000任务/1000页/20000布局，JSON共享原树预算。batch_size默认1、范围1..64。
- 当前识别器须实现recognize_formula_batch_png(PNG列表,batch_size=...)，返回str/None序列；内置PP-FormulaNet预处理、tokenizer/LaTeX解码及真实公式推理尚未迁移，capabilities明确formula_model_inference=False。未改生产Python解析器或接入实验规则。
- 下一步：迁移并对照旧PP-FormulaNet数值预处理/解码与ONNX推理，再完成表格执行和正式多页API；enhanced/XY-cut、其他语言SDK继续延期，GPU实测留服务器。
- 已核对的下一阶段细节：旧公式输入先crop_margin，再Pillow双线性resize及thumbnail、居中黑色扩边；float32归一化后按BGR灰度系数转换，16倍数补齐为单通道张量。输出为token ID，遇首个ID=2截断，经tokenizer decode、中文text包装移除、LaTeX修复及ftfy；normalize调用在旧源码中已注释，不能额外启用。须分别冻结像素/张量/解码证据，不用CTC或近似缩放替代。

## 已完成检查点：模型OCR检测、批识别与回填（2026-10-24）

- 验证完成：134项Rust workspace测试、fmt、严格clippy、CUDA feature编译、53组独立wheel验证及18组RGBA＋18组RGB精确像素对照全部通过；33模块来源哈希、36份vendor哈希检查通过。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。

- 新增legacy_model/ocr_results，复用已有inference数值内核。execute_model_ocr执行已准备任务的真实det→框整理→识别裁图→批rec→旧模型回填；独立organize_model_detection_boxes/prepare_model_recognition/apply_model_ocr_recognition可拆分编排。不执行公式或表格模型，正式多页API仍未完成。
- 保留两段排序、NumPy2 FP32弱标量、0.6行合并/0.8遮罩及宽高4倍严格门槛、角框追加；crop在宽3过滤前，模型框矫正在crop之后；pipeline竖行旋转门槛2，已有通用预览1.5保持。score三位、原始<0.5和窄竖向特定文字<0.8标为LowScoreText，和page后置OCR阈值分开。
- 131旧框规则、32识别计划/16原RGB裁图、112旧回填策略；5条真实CPU det批1→rec批6→回填链含1280/2048大图。对照定位并修正最初尺寸分档假设：本项目ocr_patch覆盖依赖方法，实际固定limit_side_len=960；原参考及精确断言未放宽，加入补丁来源哈希。检测unclip=1.8，已有detect_png预览默认参数保持。
- 识别使用原始未遮罩crop_source；RGB PNG的tRNS不合成白底，元数据不改变原RGB。共享流跨检测/裁图仅读一次、调用方模型不变。det_batch_size>1明确拒绝，rec同分跨批padding歧义仍拒绝；旧批失败逐张重试未复现，错误显式传播，不返回部分回填。
- 预算：JSON2MB字符/键、200000节点/64层、共享工作1000万；1000框/任务、1000裁图尝试，含宽度过滤掉的尝试累计计费，单图400万/累计3200万像素；PNG输入/输出各128MB。框仅有限数值/凸非退化四边形，不支持的透视输入明确失败。
- 下一步：串接公式/表格任务与模型结果回写，按旧顺序整合页面/后置OCR/段落/跨页输出，补齐多页正式API。enhanced/XY-cut和其他语言SDK继续延期，GPU留服务器实测。

## 已完成检查点：原生OCR检测框与检测输入准备（2026-10-24）

- 验证完成：131项Rust workspace测试、fmt、严格clippy、CUDA feature编译、52组独立wheel验证及18组RGBA＋18组RGB精确像素对照全部通过；32模块来源哈希、36份vendor哈希检查通过。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。

- 在现有legacy_model内新增native_ocr及detection内部模块。derive_native_ocr_boxes移植旧txt_spans_bbox_extract，保留正交容差、中心/标点、稳定排序/行聚类、NumPy pairwise均值、首span字号相邻合并、Python round2、公式严格重叠及±1切框。切框FP32及其逆变换保留，不将所有坐标统一FP64；宽3门槛与score/text字段类型一致。
- prepare_model_ocr从已有计划/原生事实生成旧空文本OCR span，保留need_ocr_det、auto/txt/ocr与rotate_label回退。检测输入按实际旧函数遮罩、64像素分辨率分组和白边补齐，任务保留原始crop_source用于后续识别，返回副本与独立RGB PNG；不回写旧merge临时span字段。
- 182组实际旧原生框/数值边界、54组native模型/检测计划、75张实际送入旧检测器的输入像素精确对照。旧检测器使用固定空响应只隔离输入准备，不宣称实际检测推理或质量；另有真实PDF原生/扫描页的布局→此阶段安装验证。
- 预算：JSON2MB字符/键、200000节点/64层；1000任务/遮罩每任务≤1000，单检测PNG≤400万/累计≤3200万像素，PNG输入/输出各≤128MB；重复覆盖遮罩面积累计≤6400万像素；原生共享工作量1000万、每页扫描10万span、输出2万框。
- 下一步：旧sorted_boxes/merge_det_boxes及det结果到rec裁图，执行真实det/批rec并按旧规则回填，然后接公式/表格与正式多页API。当前只准备检测输入；enhanced/XY-cut、其他语言SDK继续延期，GPU留服务器。

## 已完成检查点：旧模型布局转换与推理区域准备（2026-10-24）

- 验证完成：129项Rust workspace测试、fmt、严格clippy、CUDA feature编译、51组独立wheel验证及18组RGBA＋18组RGB精确像素对照全部通过；31模块来源哈希、36份vendor哈希检查通过。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。

- 在现有rapiddoc crate新增legacy_model及crops内部模块。predict_model_layout执行已有真实PP-DocLayoutV3 ONNX，再按旧RapidLayoutModel/BatchAnalyze输出layout_dets；convert_model_layout/filter_model_layout可独立使用。保留PPV2/V3类别及忽略标签、三位score、模型order、reference/6像素/inline重叠/image等例外、formula开关和等级；不排序或生成缺失order。
- prepare_model_crops按实际旧_collect_detection_regions/crop_img收集任务：表内图80%包含及图片外层/表格内层重复顺序、表格5像素网格/NumPy切片、公式round/扩框/邻框限制、OCR白边50，RGB PNG像素精确核对。model保留公式bbox和表内图片引用，UUID原值冻结并投影稳定table-image-page-image-table，PIL/ndarray换ID和独立PNG，不把表示差异隐去。
- 258组实际旧类别转换、86组保留未知字段/大整数的旧过滤、60组任务计划/409裁图参考通过；3组实际CPU布局模型输出精确一致，另有真实PDF准备入口。固定regions仅隔离转换，实际模型证据另列，不把固定响应当作推理效果。
- prepare_pdf_model_inference自动完成PDF→原生事实/RGB→真实布局推理→旧任务计划；仍未执行后续OCR/公式/表格任务及回填，不是完整自动解析入口。必须显式layout_shape_mode='rect'；auto/poly/quad及原生/VL预填布局明确拒绝，不改旧默认形状或接入实验几何排序。
- 资源：1000布局/裁图，页图64MB/1600万像素，单裁图400万/累计3200万像素及128MB PNG；JSON2MB字符/键、200000节点/64层；公式扩框0..1024、等级0/1/2；空图和crop_img越界明确失败。
- 下一步：迁移旧原生文字检测框生成和OCR det/rec任务执行/回填，再串接公式、表格执行与多页正式API。enhanced/XY-cut、其他语言SDK继续延期，GPU留服务器。

## 已完成检查点：PDF原图提取与资产集成（2026-10-24）

- 验证完成：127项Rust workspace测试、fmt、严格clippy、CUDA feature编译、50组独立安装wheel验证及18组RGBA＋18组RGB精确像素对照全部通过；29模块来源哈希与36份vendor哈希检查通过。日志为build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。

- 新增 `extract_original_pdf_images` 和 `assemble_pdf_model_page`：在现有rapiddoc-pdf crate内实现有界图片遍历、5单位小图过滤、原始分辨率缩放、mask/矩阵渲染、L/RGB/RGBA PNG；独立图片字节可直接进入原图替换及表内UUID接口。组合入口自动读取旧原生分组、渲染RGB、可选提取原图并复用旧资产组装，仍由调用方提供模型结果、按需另行后置OCR。
- 102组实际旧提图参考/432图片像素、17组完整旧页面/资产写出对照，覆盖四方向页旋转、嵌套Form、非均匀缩放/镜像/剪切、mask、重叠图和5单位边界；另有坏JPEG原始解码诊断、巨大原图预分配拒绝、257图片上限合同。保留旧局部bbox/页高翻转，不用新视觉几何替换。
- 明确兼容差异：本机pypdfium2 5.10.1移除了get_pos，未适配旧get_ori_image吞异常返回空；参考先记录空结果，再仅加get_pos=get_bounds API别名运行原函数。Rust使用对应原生API恢复提图。旧随机uuid冻结raw_uuid并投影稳定pdf-image-page-index；对象解码失败跳过并诊断，资源超限/不支持模式显式失败。不能宣称与旧空列表完全等价。
- 资源：最多256候选/32层、原始和渲染单图≤1600万像素、输出累计≤3200万像素、PNG单图64MB/合计128MB，枚举对象受原有Limits约束；调用内持唯一PDFium锁，位图/页/文档按序析构。pypdfium2缩放来源与Apache许可随wheel分发。
- 下一步：正式模型推理路由/API及多页串联；完整自动解析迁移仍未完成。enhanced/XY-cut与其他语言SDK继续延期，GPU留服务器实测。

## 已完成检查点：PDFium 到旧原生分组自动桥接（2026-10-24）

- 验证完成：126项Rust workspace测试、fmt、严格clippy、CUDA feature编译、49组独立安装wheel验证与18组RGBA＋18组RGB精确像素对照全部通过。28个模块来源哈希检查通过，36份vendor哈希未变。日志：build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。

- 新增 `read_native_pdf_page`，Rust直接按旧pdftext 0.7.1读取双精度字体、按角度选择loose/tight字框、处理裁剪/旋转、单词及相邻偏移去重、span/行/上下标/块分组。Unicode16 digit/alnum/Sm属性范围固定，不使用新的阅读顺序算法。bbox对象换等坐标JSON数组，保留整数/浮点类型。
- `render_pdf_page(...,color_mode='RGB')` 提供旧原生填充需要的RGB页图；默认RGBA行为保留。28组真实旧get_page完整JSON、268组旧去重/分组边界，另将5组原生页面及5条真实CPU串联的输入改为Rust直接读取PDF和RGB渲染，比较原页面、裁图像素及最终输出。pdftext Apache许可和来源说明随wheel分发。
- 资源边界：每页10万字符、单字体名1MB/累计32MB、页框尺寸≤10亿；原始句柄在全局PDFium锁内逆序销毁，不向Python暴露。仅旧默认分组阈值；不宣称任意PDF/其他pdftext版本等价。
- 下一步：PDF原图自动提取及与现有资产接口集成，随后继续正式模型推理路由/API。当前模型结果仍由调用方提供，完整RapidDoc迁移未完成；enhanced/XY-cut、其他语言SDK继续延期，GPU实测留服务器。

## 已完成检查点：原生字符填充与补识别裁图（2026-10-24）

- 最新验证：125项Rust workspace测试、fmt、严格clippy、CUDA feature编译、48组独立wheel验证与18组PDF RGBA精确对照全部通过；27个模块兼容审查、36份vendor哈希不变。日志为build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt和build/wheel-all-tests.txt；正式解析入口未切换。
- 在现有 rapiddoc crate 内移植旧 txt_spans_extract/fill_char_in_spans：保留正交行方向（弧度，严格小于0.1度容差）、模型 span 高度中位数、正文/丢弃块归属、纵排按原生行追加、横排网格首个候选归属、标点边缘机会、char_idx 稳定顺序、重复字典 list.index、显式空白抑制插空格、连字/控制字符替换与旧长度回退条件。保留未归属/纵排 span 的 height/width 类型，正负零相同 top 保留原顺序。
- 新增 `fill_model_text_spans` 独立 stage 和 `assemble_model_page_with_native_text` 完整旧页面集成，后者在 MagicModel/过滤后、归块前执行原生填充，仍复用原资产 writer。返回 `{page,ocr_images}`，np_img 换为 `ocr-{page_index}-{N}` 引用，OCR 图像在内存中保存为不带 ICC/tRNS/EXIF 的 PNG，解码 RGB 与旧 NumPy BGR 反转后逐字节一致，直接交给 post_process_model_ocr。
- 已冻结并对照198组旧 span（含3组正负零归属）、32组完整旧页面（含5组真实PDF字符事实）、295组OpenCV/NumPy对比度，以及15条原生填充→真实CPU后置OCR→编号→段落链/45组旧输出，包含图片/表格/公式普通资产。只将 bbox 对象换等坐标 JSON、np_img 像素负载换引用并另比像素，不归一化文字/字段/浮点类型差异。
- 裁图按旧 NumPy 切片：bbox×scale 向零取整、负下标相对图像末尾、边界截断、空裁图删除；不使用普通资产的 PIL 越界补零。RGB2GRAY 整数路径、NumPy FP64 pairwise 标准差与 NumPy 两位小数 round 后严格 `<0.17` 才删除，0.17保留。NumPy来源及BSD许可随源码和wheel携带。
- 当前页图仅8-bit RGB PNG（64MB/1600万像素），native_page须为旧分组事实JSON，line.bbox/char.bbox为页面单位数组、rotation为弧度、char_idx为绝对值≤10亿的整数。每次最多1000 span、10000有效原生行/100000字符、1000补识别裁图；单裁图≤400万像素、含低对比度尝试的裁图合计≤3200万像素、OCR PNG累计≤128MB；网格区间和全局工作量有界，极小高度/索引溢出明确resource_limit。仅旧 return_word_box=False 路径。
- 下一步：迁移 Rust PDFium 原始字符到旧 pdftext 去重/行/span/块分组的自动适配，接上真实PDF输入；当前仍须显式传 native_page，不能把新实验分行输出冒充旧分组。PDF原图自动提取、正式推理路由/API、WMF/EMF与算法分类等仍待完成；enhanced和其他语言SDK继续延期。

## 已完成检查点：后置 OCR 与批识别（2026-10-24）

- 最新验证：122 项 Rust workspace 测试、fmt、严格 clippy、CUDA feature 编译、47 组独立 wheel 验证与 18 组 PDF RGBA 精确对照全部通过；26 个模块兼容审查、36 份 vendor 哈希不变。日志为 build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt；原生产 Python 和正式 PDF 入口未切换。
- 在现有 rapiddoc/rapiddoc-inference crate 内新增 legacy_post_ocr/recognition_batch。`apply_model_ocr_results(pages,results)` 按旧页面、主块/图表 caption/footnote、丢弃块、行/span 顺序回填；只按 np_img 键是否存在收集，不全树递归。73 组真实旧函数参考含固定模型响应、严格 >0.5、FP64 三位小数/相邻浮点边界、重复引用和非文本 span，全部精确对照。
- `post_process_model_ocr(pages,recognizer=...,images=...,batch_size=6)` 接收 np_img 字符串图片 ID 与独立 PNG 映射，先快照页面，再识别及回填副本；同一 ID 的流读取一次，重复 span 仍重复识别，不做结果去重。缺图、输入/推理/数量错误不返回部分回填；与旧函数先 pop 图片键再推理不同，调用方原输入保留。应在公式编号、段落和跨页表格前调用。
- `OnnxSession.recognize_batch_png` 沿用旧宽高比排序、批内共同 max_wh_ratio、3×48 BGR/线性 resize/最小320宽零 padding、CTC 与原输入顺序还原。12 组实际旧 CPU 批识别参考：11 组受支持，逐原图张量 FP32 位模式一致，真实 PP-OCRv6 文字精确、置信度绝对误差≤1e-5；1 组同分跨批改变 padding 明确 unsupported，不使用稳定排序悄悄替换 NumPy 非稳定排序。未声称任意模型/设备批处理等价。
- 另有 11 组实际旧页面→真实 CPU 响应→后置回填→编号→段落及 33 组旧输出精确对照。纯回填参考用固定模型响应隔离逻辑，真实模型参考另行运行 CPU ONNX；两者证据范围分开，不将模拟响应当作 OCR 质量验证。
- 回填 JSON≤2MB 字符/键、200000节点/64层、1000页、1000有效文本块/10000行/20000 span/1000裁图。批识别 batch_size=1..64、每张 PNG≤64MB/1600万像素、输入累计≤128MB/3200万像素；重复引用按展开后的输入收费。宽度≤4096，张量受 OnnxSession.max_elements（默认1600万）约束，不自动下载、切换设备或拆成单张掩盖不兼容。
- 下一步：按旧 txt_spans_extract 迁移 ocr_enable=False 时的原生 PDF 字符填充和补识别裁图，接上此后置 OCR 图片来源；PDF 原图自动提取、正式模型推理路由/API、WMF/EMF 与算法分类等仍未完成。enhanced/XY-cut 与其他语言 SDK 继续延期，GPU 留服务器实测。

## 已完成检查点：旧跨页表格合并（2026-10-24）

- 最新验证：118 项 Rust workspace 测试、fmt、严格 clippy、CUDA feature 编译、46 组独立 wheel 验证与 18 组 PDF RGBA 精确对照全部通过；24 个模块兼容审查、36 份 vendor 哈希不变。日志为 build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。原生产 Python 与正式解析入口未切换。
- 已在现有 rapiddoc crate 内移植 `legacy_table_merge`：从后向前处理相邻页首尾表，沿用续表标题/脚注/宽度差≥10% 的拒绝条件、逻辑列与渲染边界匹配、重复表头与 rowspan 扩展、整表列网格映射和旧回退、稀疏续行 cell_merge、空白 rowspan 下沉及跨页脚注替换。
- 新增 `merge_cross_page_tables(pages)` 副本接口，以及 `assemble_model_pages(..., merge_tables=True)` 显式串联；默认 False，不改变已有预览调用。公式编号在段落前，跨页合并在段落后。独立 stage 不读取 MINERU_TABLE_MERGE_ENABLE，正式解析路由仍未接入。
- 313 组完整旧页面 JSON 已对照（288 组包含原 99 组，另有 25 空表/无行边界），含三页以上连续合并、caption 恢复导致块下标变化、跨度/图片/全角字/HTML 实体与空白。另冻结 864 组旧输出与 48 组实际旧单页→编号→段落→跨页合并、144 组输出，拒绝归一化 HTML 差异。
- HTML 支持显式正确闭合、单表、非嵌套 td/th、标准 void 标签、合法分号结尾 HTML5/数值实体；按旧 BeautifulSoup 排序属性、输出 void 标签、纯 ASCII 空白压缩及图片/数学内容语义。嵌套/多表、需补闭合的错误 HTML、script/style/template、非法/控制字符实体和未闭合实体明确 unsupported，不仿造 HTML 修复。合法 rowspan/colspan 为 ASCII 正整数 1..1000。
- 原 JSON≤2MB 字符/键、200000 节点/64 层，最多1000页、每页初始1000块；单表合并后≤1000行/1000逻辑列、HTML节点≤20000/深度≤64，每次跨度占位扫描≤200000项，全流程工作预算1000万。搬移后的旧状态及时释放，复制行/单元格内容计入预算；不把旧含共享对象引用的状态缓存改成无限累积副本。
- 下一步：后置 OCR 裁图/批识别结果回填，再推进 PDF 原图自动提取、真实模型推理与正式 API 集成。WMF/EMF 转换、算法文本 Magika 分类等边界仍待完成；enhanced/XY-cut 与其他语言 SDK 继续延期。

## 已完成检查点：表内 UUID 图片资产（2026-10-24）

- 最新验证：114 项 Rust workspace 测试、fmt、严格 clippy、CUDA feature 编译、45 组独立 wheel 验证、18 组 PDF RGBA 精确对照全部通过；23 个模块兼容审查和 36 份 vendor 哈希检查通过。表内图片名称另以三个独立进程验证稳定；原生产 Python 未切换。日志为 build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。
- `assemble_model_page_with_assets(..., table_fill_images=[{"uuid":..., "bbox":[...], "source":PNG}])` 已接入旧页面流程：data URI 内联写出→MagicModel 候选过滤→表内 UUID 原图写出/HTML 替换→普通裁图→归块排序。仅 category_id=5 且 HTML 含 UUID 的保留候选写出，按检测项和图片列表顺序处理；字符串全量替换，不限于 img 标签，不重复消费已替换 UUID。
- 新增 57 组实际旧页面/writer 及 171 组旧输出参考，覆盖四种像素模式、缩放、负数/小数坐标、多个/重复/重叠 UUID、同坐标不同像素的顺序覆盖、候选过滤、Unicode/空目录名、data URI 先写、ICC/透明色。冻结文件保留原始旧页面、调用和含地址哈希种子，并另列 stable-name 投影；不宣称旧文件名精确等价。
- 有意修复旧函数对象地址参与命名：新种子为 `table_fill/{PAGE_RAW_MD5_UPPER}_{page_index}_{int_bbox0}_{int_bbox1}_{int_bbox2}_{int_bbox3}`，SHA256 后为平铺 PNG。不同于普通 table 裁图命名空间；UUID/图片像素不加入种子，保留旧同页同坐标顺序覆盖语义。像素、HTML 替换、调用顺序和名称投影后的页面/串联输出按旧参考检查。
- 仅接收 8-bit L/LA/RGB/RGBA PNG，原图保持 ICC/tRNS、不继承 EXIF；UUID 为非空≤256 字节，最多 256 张/128MB 输入、单张≤64MB，命中后懒解码，缓存累计≤6400 万像素。与普通裁图共享 1000 项/3200 万输出像素/128MB 输出预算、页/原图/表内图 ICC 合计≤8MB，UUID HTML 扫描额外限制 1000 万字节工作量、展开后单 HTML≤2MB。
- 计算或编码失败在 writer I/O 前抛出，修复旧函数吞异常后可能返回缺图页面的行为；I/O 异常原样传播，已写文件仍不承诺回滚。表内 UUID 图片不套用 data URI writer 的生命周期去重。
- 下一步：按旧 table_merge.py 移植跨页表格，已冻结 99 组实际旧完整页面 JSON，覆盖既有 5 个续表回归 HTML、续表/后置标题、脚注、宽度边界、三页链及显式 cell_merge；原有 5 个续表测试通过。此处仅是旧基线准备，Rust 跨页合并器尚未实现。后置 OCR、PDF 原图自动提取、正式推理/API 集成仍待完成。enhanced/XY-cut 与其他语言 SDK 继续延期。

## 已完成检查点：HTML 内联图片资产（2026-10-24）

- 最新验证：112 项 Rust workspace 测试、fmt、严格 clippy、CUDA feature 编译、44 组独立 wheel 验证和 18 组 PDF RGBA 精确对照通过。兼容审查 22 个模块，36 份 vendor 哈希不变；日志为 build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt、build/wheel-all-tests.txt。原生产 Python 未切换。
- 已按旧 Python 工具实现 `legacy_page/inline.rs`，并接入 `assemble_model_page_with_assets`：先处理所有 raw layout_dets 的 HTML，再执行 MagicModel 和普通裁图。新增独立 `replace_inline_base64_img_src`，精确保留双引号/大小写/空白匹配、前缀直接拼接、原字节、完整 data URI SHA256、jpeg→jpg 和无效编码保留原文。
- 9667 组未修改旧工具参考、12 组完整旧页面/writer 与 36 组旧输出串联参考已冻结，验证包括重复 src、过滤前写出、普通裁图顺序、Unicode 前缀、空/畸形编码及原始非图片字节。自定义 writer 沿用 `_mineru_written_image_paths` 跨调用去重；写入失败不标记，无法设置属性的 writer 不去重。
- 补齐原始 JSON 的前置资源检查，再执行内联替换；替换缩短 HTML 不能绕过原始 2MB 字符/键预算。无效 model 类型返回 invalid_input，不触发 serde_json 可变索引 panic。失败均在 writer I/O 前返回。
- 畸形 base64 padding 对齐本机旧 Python 3.14.5，不宣称兼容旧 Python 小版本的不同畸形输入行为；正常 base64 与 Python 无关。数据 URI 格式扩展名限制为 1–32 个小写 ASCII 字母、数字、短横线或下划线；WMF/EMF 转换明确 unsupported。HTML 单次≤2MB、前缀≤4096 字节、最多 1000 项；整页仍受旧资源预算约束。
- 下一步继续 table_fill_image_list 的 UUID 图片替换，再推进跨页表格和后置 OCR。旧 save_table_fill_image 使用函数对象字符串参与文件名哈希，包含进程地址；新增只读 probe 实际运行三个子进程，相同 PNG 字节、页号、bbox 和页面 hash 得到三个不同名称，已冻结原始哈希种子及输出于 build/migration/legacy-table-fill-naming.json。后续须明确审查命名差异，不能归一化地址后宣称精确等价。PDF 原图提取、正式 API/推理集成仍待完成；enhanced 继续延期。

## 已完成检查点：旧图片裁图与资产 writer（2026-10-24）

- 最新验证：109 项 Rust workspace 测试、fmt、严格 clippy、CUDA feature 编译、43 组独立安装 wheel 验证与 18 组 PDF RGBA 精确对照全部通过；兼容审查为 21 个模块，36 份 vendor 哈希不变。日志见 build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt 与 build/wheel-all-tests.txt。

- 新增 `assemble_model_page_with_assets`：传入页 PNG 和已识别的旧模型 JSON，将 Rust 图片裁图接入旧页面组装中与 cut_image_and_table 相同的位置，回填真实 image_path 后再归块/排序。支持输出目录或 write(name, bytes) writer；旧无 writer API 保持不变。
- 文件名沿用“页原始像素 MD5 大写＋span 类型＋页号＋未缩放 bbox 向零取整”的 SHA256 平铺 PNG 规则。裁图 bbox×scale 向零取整，越界按原模式补零，透明通道不合成白底。可显式提供原图 PNG，仅 image 类型按 IoU≥阈值选取最后一个匹配项，表格/公式继续页图裁切。
- 337 组实际未修改旧单页函数与 writer 调用的完整页面 JSON、文件名、调用顺序和 PNG 解码像素一致；覆盖 8-bit L/LA/RGB/RGBA、负数/小数/越界框、原图优先、空页、未归块仍写出、同名不同裁图的顺序覆盖。另有 337 组资产页→编号→段落及 1011 组旧 Markdown/content_list 输出精确一致。PNG 压缩字节不要求相同，不把编码器差异当作像素差异。
- 追加 11 组实际旧 writer 的 ICC/EXIF/透明色合同：ICC 字节和 RGB/L 透明色按旧 PNG save 保留，EXIF 按旧默认行为不继承。RGB/L 的透明色解码即使展开 alpha，也恢复原始模式计算 MD5 并保存 tRNS，避免文件名、透明背景或显示色彩改变。单 ICC≤1MB、页与原图 ICC 合计≤8MB，独立标准 PNG 检查同时比较元数据与像素。
- 所有原生计算完成后才调用 Python writer；计算失败不写出部分文件。目录模式单文件临时写入后原子替换，文件名只允许平铺 SHA256 PNG。保留重复写入顺序；writer/I/O 失败直接抛出，已成功写出的文件不承诺回滚。目录与自定义 writer、独立 PNG 滤波解码、并发及失败/预算验证均已覆盖。
- 页 PNG 及单个原图输入≤64MB、单张解码≤1600 万像素；原图最多 256 张、合计 6400 万解码像素/128MB PNG 输入。单资产≤400 万像素，最多 1000 次写出，累计≤3200 万资产像素/128MB 编码字节；沿用页面 JSON/几何预算。P/16-bit PNG 明确 unsupported，不静默改模式或哈希。
- 仍缺 HTML 内联/表内图片资产、PDF 原图自动提取、后置 OCR、跨页表格、实际推理与正式 API 完整集成；原生产 Python 未切换，整文档质量/速度/内存未验收。enhanced/XY-cut 与其他语言 SDK 继续延期，GPU 留到服务器实测。

## 已完成检查点：旧公式编号后处理（2026-10-24）

- 最新验证：105 项 Rust workspace 测试、fmt、严格 clippy、CUDA feature 编译、42 组独立安装 wheel 验证与 18 组 PDF RGBA 精确对照全部通过。审查清单为 20 个模块，36 份 vendor 哈希不变；原生产 Python 未切换。完整日志为 build/rust-current-tests.txt、build/rust-current-clippy.txt、build/rust-current-cuda.txt 与 build/wheel-all-tests.txt。

- 新增 legacy_page/formula.rs 与 `optimize_formula_numbers(pages)`，按未修改的 `_optimize_formula_number_blocks` 移植编号关联：优先前一公式，再按旧相邻条件尝试后一公式，最后按同排重叠≥0.5、右侧关系和间距≤0.4 页宽查找最近中心候选；中心距离相同保留首个候选，最近候选冲突后不擅自尝试其他公式。
- 保留旧全角标点/字母/数字转换与括号剥离、既有同名 tag 的去重、不同 tag 冲突时退回文本、空编号删除、孤立编号转 text；保留后续编号修改此前已纳入结果的公式这一对象引用行为。API 返回副本，仅修改 preproc_blocks，不重算已有 para_blocks；应在段落处理之前执行。
- `assemble_model_pages(..., optimize_formula_numbers=True)` 显式串联页面组装→编号优化→旧 para_split，默认 False 保留已有预览接口行为。该处理不运行公式识别模型，不进行跨页编号配对、后置 OCR、资产写出或跨页表格合并。
- 718 组完整旧公式编号 JSON、382 组原模型→页面→公式编号→段落、639 组原模型→编号→旧输出逐字段/类型精确一致。输出对照按处理后的实际块保守排除 169 组 Magika 算法链；独立旧输出检查点原有的 171 组排除记录保持不变。基线导出脚本拒绝覆盖旧文件，不用新规则生成旧参考。
- 单独编号入口最多 1000 页、总计 1000 块/10000 行、200000 JSON 节点、64 层、2MB 字符串及 32MB native JSON；坐标绝对值≤100 万，几何/文字扫描共用 1000 万工作预算。保留未知字段和 JSON 数值类型；非法结构、无效 Unicode 或超限明确报错，不返回部分处理结果。
- 下一步继续补齐图片资产 writer、后置 OCR/跨页表格与实际推理/正式 API 集成。enhanced/XY-cut 及其他语言 SDK 继续延期，GPU 实测留到服务器；完整解析器、质量与整文档性能仍未验收。

## 已完成检查点：旧模型页面组装与输出（2026-10-24）

- 该检查点已完成验证：102 项 Rust 测试、fmt、严格 clippy、CUDA feature 编译、41 组安装包验证和 18 组 PDF RGBA 对照全部通过；36 份 vendor 哈希不变。日志见 build/rust-current-tests.txt 与 build/wheel-all-tests.txt。后续新增后处理需在此基础上重新验证。

- 新增 `assemble_model_page`：接收旧 raw layout_dets，不需要 Python MagicModel。移植 poly/scale 坐标转换、polygon_points 舍入、置信度/重叠块过滤、图表标题和脚注关联、native foreground 所属区域约束、span 去重/过滤/归块、横排/竖排行、丢弃块和模型 original_order 排序。保留旧 image_footnote 被标记为 image_caption 的行为，不自行“修正”差异。
- 新增 `assemble_model_pages` 串联单页组装与旧 para_split，空页按旧包装生成；新增 `render_model_pages` 提供默认 $/$$ 的 mm_markdown、nlp_markdown 和 content_list。输出沿用旧转义、全角字母/数字转换、断词、列表换行、标题、公式、表格 HTML 和图片路径规则；这些是显式 JSON stage 接口，未替换正式 PDF/RapidDoc API。
- 286 组实际未修改 page_model_info_to_page_info 的完整输出、18 组数值边界、382 组页面→段落串联、726 组完整旧输出逐字段/类型一致；覆盖 8 组实际 CPU OCR 输入。数值补测复现并修复大整数被转成浮点后误判相等、改变保留 span 的问题；整数精确比较，整数与浮点只有精确数值相等才视为重复，断言未放宽。
- 当前合同为 OCR 已识别、ocr_enable=True、image_writer=None；非空主块需要有效模型 original_order。不执行模型推理、原生 PDF 字符填充、图片/表内内联资产写出、后置 OCR、公式编号后处理、跨页表格合并。算法 Magika 分类尚未移植，输出参考明确排除 171 组算法链，运行遇到算法文本块明确 unsupported；不使用固定 txt 标签冒充旧分类。
- 单页最多 1000 个原始模型项、200000 JSON 节点、64 层、2MB 字符串及 32MB native JSON；坐标绝对值≤100 万，scale 为 (0,1000]，页面尺寸 [1,100 万]。几何操作预算 1000 万，多页共享，不逐页重置；段落层继续执行其总块/行/文本预算。输出最多 1000 页并有 3200 万累计文字扫描预算；不返回截断或部分成功。
- 下一步补齐图片资产 writer、公式编号/后置 OCR/跨页表格等文档级处理，再接入现有模型推理入口与正式 API；继续按旧 Python 冻结输出并差分。enhanced/XY-cut 与其他语言 SDK 仍按用户要求延期，GPU 实测留到服务器。

## 已完成检查点：旧模型顺序排序与段落串联（2026-10-24）

### 用户确定的范围调整与下一步（2026-10-24）

- **enhanced/XY-cut 阅读顺序延期**：现阶段采用版面模型自带的 original_order。无模型顺序的 enhanced 调用链及专属辅助内核不继续实现；待其他任务完成后，由用户明确要求再启动，不自动恢复。
- 已实现的模型顺序排序和段落模块继续使用。缺少有效 original_order 时仍明确 unsupported，不静默启用实验几何排序。此前对 enhanced 源码仅完成阅读，本次范围调整前未新增其 Rust 实现。
- 原定页面组装及默认 Markdown/content_list 的 JSON stage 已在本轮实现，详见顶部当前检查点。图片资产与文档级后处理尚未完成，不把局部 stage 差分当成完整 PDF API 等价。
- 后续继续完成资产与模型推理集成，以及其余原生 PDF 规则、方向/公式、Office、正式 Python API 和验收。GPU 实测留到服务器，其他语言 SDK 仍延期。

- `legacy_block_sort.rs` 按未修改的 `rapid_doc/utils/block_sort.py` 移植有 original_order 的排序分支，Python 暴露 `sort_model_blocks`。保留块顺序优先、span 最小顺序回退、缺失顺序索引等于块数、并列稳定性、全页行号、虚拟行恢复和图片/表格回组；不推测图表归属或修改原输入。
- 1,206 组完整旧排序输出精确一致，覆盖模型顺序 100 与缺失顺序的先后、bool/int/float 同序、负坐标/零类型、虚拟行全部条件、首 body 框、多组图表及中位数索引。另有 308 组旧 sort_blocks_by_bbox→para_split 串联输出精确一致，包含 8 组实际 CPU OCR 行；不是整文档质量验收。
- 接口支持有效 Unicode/JSON、正宽高块/行框和正页面尺寸；坐标绝对值≤100 万、顺序/数值 group_id 绝对值≤2^53−1。预算为 1000 块、10000 输入行、20000 输入与虚拟行合计、200000 节点、64 层、2MB 字符串与 32MB JSON；空模型顺序明确 unsupported，无效结构或超限报错。
- 无 original_order 的 enhanced/XY-cut 尚未迁移；完整模型/导出/原生 PDF 规则集成、Office 与服务器 GPU 验证仍待完成。实验几何/原生仲裁继续默认关闭，原生产 Python 未切换；兼容审查清单扩为 17 个模块。
- 本轮 97 项 Rust workspace 测试、fmt、严格 clippy、CUDA feature 编译、40 组独立安装 wheel 验证和 18 组 PDF RGBA 精确对照全部通过；36 份 vendor 哈希与原 Python 解析器未变。日志见 build/rust-current-tests.txt 和 build/wheel-all-tests.txt；未进行整文档性能/内存或 GPU 实机验收。

## 已完成检查点：旧 Python 段落规则移植（2026-10-24）

- 新增 `legacy_paragraphs.rs` 与 `split_paragraphs(pages)`，按 `backend/pipeline/para_split.py` 的实际规则移植 text 分组、bbox_fs 重算、text/list/index 分类、列表行首尾标记、倒序跨块合并和 cross_page/lines_deleted 标记。输入为已排序的旧 page_info_list JSON，输出为副本；保留 preproc_blocks、未知元数据、原数值类型及非文字块，不调用实验几何段落器。
- 冻结旧函数的 636 组完整输出，再补 214 组浮点/Unicode/CPU OCR 合同，总计 850 组逐字段一致；覆盖 759 个 Unicode 大写边界字符、中文/日文/韩文混合阈值、所有结束符、空行/空页、标题/表格屏障和跨页合并。比较 JSON 类型、字段、数组顺序和数值，不折叠或删去差异；基线导出脚本拒绝覆盖已有文件。
- 边界测试定位到 serde_json 默认解析的 1 ULP 偏差会改变旧列表标记；旧 Python 在同一舍入输入上也产生该差异。启用 float_roundtrip 后 850 组完全一致，已有内核同时接受全量回归。Unicode 16.0.0 固定表补充 isupper/isalpha，避免运行时字符库版本变化。
- 合同限制：有效 Unicode、正文字块/行几何、唯一整数页号、JSON 数据；最多 1000 页/块、10000 行、200000 节点、2MB 字符串及 32MB JSON。非文字块整体保留，无效几何/结构或资源不足报错。不是旧 Python 对任意对象/非法输入的完整异常行为兼容。
- 该独立入口尚未接入正式 RapidDoc API、旧阅读顺序、模型图像对象与导出。已有几何段落/原生融合实验仍默认关闭。审查清单扩为 16 个模块。

## 已完成：兼容性审查与实验隔离（2026-10-24）

- 回到现有 Python 行为优先的迁移路线。15 个模块的来源、证据和偏差见 `rust-compatibility-audit.md` 与 `packaging/compatibility-audit.json`；抽样内核差分、作者几何合同、部分流程和完整文档质量分别记录，不互相替代。
- 对旧 `para_split.py` 和 `block_sort.py` 实际执行后，35 组适配段落合同有 33 组结构不同；15 组模型顺序优先合同有 6 组不同，5 组不适配输入明确跳过。这不是文档准确率或优劣评分，足以说明新策略不能冒充等价移植。完整差异见 `build/migration/compatibility-differences.json`。
- `organize_page_text` 与 `recognize_pdf_page_hybrid` 现在默认拒绝运行，必须显式传入 `experimental=True`；原生表格选择另需 `prefer_native_tables=True`，默认已改 False。wheel 携带审查元数据并暴露 `compatibility_report()`；原生产 Python 未切换。这是预览接口的有意兼容变更，旧预览调用须补实验参数。
- 两个独立进程对 13,930 组换行核的输出完全一致；1 次预热/5 次重复中位耗时约 Python 0.0695s、Rust wheel 0.0113s（约 6.14 倍批次收益）。进程峰值 RSS 约 42,024,960/42,094,592 字节，含导入和输入，未证明内存下降；不据此作整文档性能结论。复测入口 `scripts/bench/compare_text_join.py`。
- 此处原定的 enhanced 排序调用链现按用户要求延期，优先页面组装与输出闭环，详见顶部最新范围调整。未经证明的候选策略仅保留为实验。下方 2026-10-23 的“接通”记录属于开发预览，不能当成完整 Python 等价验收。

## 历史检查点：显式页面块阅读顺序

- `organize_page_text` 新增 `block_order="source"/"geometry"`。默认 source 保持字段和顺序兼容；geometry 只重排完整布局块，段落的 `block_index/layout_index/line_indices` 仍指向原输入。启用后额外返回 `block_order={indices,reason,work_used}`，表格/unsupported 作为整体保留，不跨块合并或去重。
- `reading_order.rs` 使用区间投影留白，优先有垂直重叠的并排栏；剥离通栏头尾后再组织中间栏区，避免标题/页脚令双栏正文逐行交错。留白阈值为页面短边 0.5%，通栏宽度为当前总宽的 80%；迭代处理不使用递归调用栈。仅处理清晰几何，不宣称旧 XY-cut-plus 全规则等价。
- 任意块重叠、空/越界区域、不支持的方向或无法明确切分时，整页保留原块序，原因分别为 `overlapping_regions/invalid_region_geometry/unsupported_direction/ambiguous_geometry`；不保留部分重排。最多 1000 块，默认/上限 `max_block_order_work=2_000_000`，验证、成对重叠与排序成本估计累计收费，超限报错，不降级成普通歧义。
- 16 组作者合同 × 3 比例、默认输出兼容、表格/unsupported 保留、并发、非法参数与预算拒绝通过。新增独立六区域 OCR/PDF 事实语料，真实扫描双栏及 72/137 DPI 无识别会话的原生页恢复预期顺序；6 项 Rust 测试含同一六块页的全部 720 种输入排列和 1000 个区域预算验证。
- 页中通栏混排、轻微重叠仲裁、图表标题/脚注归属、RTL/竖排、跨页阅读顺序和内容合并仍待迁移。

## 已接通：显式栏内几何行序

- `organize_page_text` 新增 `line_order="source"/"geometry"`，默认 source 保持既有输出。geometry 在已有保守分栏后按栏内行框顶部排序，保留输入的原始 `line_indices`，不重排布局块、不修改输入；标题可按行恢复顺序，算法与竖排标签保留原序。
- 只有整栏都是非空、近水平且上下顺序可判断的行才排序。同排片段或行框垂直重叠超过较小行高的 20% 时整栏保留原序；空行可能是人为段界，也不移动。诊断包括 `lines_reordered` 与 `line_order_preserved_overlap/blank/non_horizontal/algorithm/vertical_text`，按块去重。关闭分栏不会改用横向顺序拼接同排文字。
- 新增 14 组独立作者合同 × 3 比例：单/双栏乱序、同排/重叠、部分栏歧义、空行/方向/标题/算法、列表续行；默认兼容、非法选项、并发、跨块隔离和排序后全页累计扫描预算通过。对真实扫描 OCR 和 72/137 DPI 原生页面结果打乱行序后，可恢复独立旧连接器的参考文字；纯原生页仍不需要识别会话。
- 这只是显式栏内行序策略。保守块序见当前检查点；复杂跨块/跨页阅读顺序、页中通栏混排、同排片段组装、重叠去重和完整旧段落规则仍未完成。

## 已接通：列表悬挂续行

- `organize_page_text` 的列表段落现在可合并同块、同栏内具有稳定悬挂缩进的续行，条目保留 `kind="list_item"`、原始标记及全部源行索引。修正旧预览中首行孤立、续行被并入后续普通正文的问题；新编号/圆点、空行、正文左缘、短尾、过大间距和缩进漂移均截断条目。标题和算法规则不变，不推测嵌套层级或跨块列表。
- 行高近似 em：续行相对首行左缘缩进 0.5～2 em，后续续行与首个续行左缘差最多 0.25 em，行高比不超过 1.15，行间隙为 -0.3～0.7 em；仍受水平/源顺序/重叠及短尾约束。借鉴旧 `local_lists.py` 的稳定几何约束，但当前输入没有字体签名/矢量圆点，不能宣称完整旧列表规则等价。
- 新增 19 组独立作者合同，覆盖数字/字母/括号/CJK/符号标记、十进制正文、边界拒绝、双栏、标题和算法；三种比例与并发、源行完整性验证通过。另新增独立旧 CPU OCR 与 PDF 字符事实语料，真实扫描列表及 72/137 DPI 原生 PDF 验证通过；原生路线不提供识别会话，确认无需 OCR 回退。

## 已接通：换行规则与块内段落预览

- 新增 `merge_text_lines`：迁移旧 Python 的 CJK/标点/样式边界、英文断词及跨多行 URL 连接规则；使用固定 Unicode 16.0.0 属性表，不依赖宿主 Python。13,930 组独立旧实现参考逐项一致，输入仅接受有效 Unicode；设置行数、字节和累计扫描预算，不静默截断。
- 新增显式后处理 `organize_page_text(page, split_columns=True)`，适用于现有 PNG/PDF/混合页面块结果，原识别接口输出不变。仅在同一布局块内按几何间隔、缩进、短尾、字号和列表边界组织段落；保守分栏要求每栏至少两行且垂直范围重叠。每段保留原块、布局、栏和行索引及像素框，表格/unsupported 整体保留，算法块保留换行和缩进。
- 7 组独立几何合同覆盖分栏/保持源顺序、空行/缩进/列表、脚本/间距、布局隔离、算法和短尾；另验证实际 OCR、纯原生正文与原生表格页面、并发、非法四边形及累计资源限制。该检查点新增 5 项 Rust 测试，当时 workspace 共 79 项通过；完整 wheel 回归结果见下方验证记录。
- 这是保守的显式段落视图，不宣称完整旧段落器等价。跨块阅读顺序、复杂列表/嵌套、图表标题/脚注归属、重叠去重和跨页合并仍待迁移；悬挂列表续行见上方对应章节。

## 已接通：单页原生表格优先与模型回退

- `recognize_pdf_page_hybrid` 提供 `prefer_native_tables`（审查后默认 False）、`native_table_margin_pt=6.0` 与累计原生表格预算，原生候选在任意 OCR/分类/结构调用前统一规划。纯原生表格可不提供推理会话（布局模型除外）；同页混合表格分别选择原生/模型，并保留 `table_selection/native_table_routes` 原因。显式关闭原生选择仍走原模型流程；坏模型错误不被屏蔽。
- 新增 `rapiddoc/src/native_table.rs`，复用物理网格与安全 HTML 内核；`grid::plan_tracks/recover_with_tracks` 保留实际不均匀轨道，原 `recover_pdf_table` 输出契约不变。原生结果保留合并格、字符索引、point 轨道与页面像素格框/矩阵。策略为无旋转/图片重叠、可靠且非空的原生文字、90% 闭边覆盖与矩形合并、布局/网格 IoU 至少 80%、所有所选非空字符有归属。空原生表、缺边或范围不符继续模型，不把它们当成功空表。
- 每页原生表格工作量上限 1000 万，单网格最多 2000 原子格/分隔线；失败的覆盖查询也计入累计预算。3 项 Rust 测试覆盖不均匀合并格/投影、转义与来源、回退原因及失败候选预算。8 页/10 组页与 DPI 对照（含 137 DPI）已通过独立 PDF 事实/像素及作者已知网格合同；4 组/6 张纯原生表不调用识别模型，同页原生/扫描表并存、扫描模型与独立裁图等价、真实布局模型→原生表均通过。
- 仍只处理已检测候选的保守描边网格；无边线/填充条规则、旋转原生表、漏检补偿、重叠布局去重、完整段落和跨页整合待实现。

## 已接通：单页正文的原生/OCR 行融合

- 新增 `native_text.rs` 与 `recognize_pdf_page_hybrid`，复用已有页面块流程，保留纯模型 `recognize_pdf_page` 的行为。正文继续布局/DB 检测，只把可靠覆盖的水平行替换为原生文字，其余行继续识别；表格选择见原生表格优先章节。原生视觉 point 按实际渲染宽高映射为像素；原始 `native_page`、含空格的字符索引及 `native_routes` 原因均保留。原生行 confidence 为 null，不伪造识别评分；OCR 阈值不作用于原生行。
- 当前保守预览策略：未旋转、检测行接近水平、字符 Unicode/源顺序可靠、宽/高覆盖至少 80%/50%、无图片重叠。混合在同一行的图片扫描部分会触发整行 OCR；旋转、覆盖不足、不连续或无效 Unicode 也回退。每页累计匹配工作量默认/上限 1000 万，资源错误或实际模型错误不伪装为普通回退。
- 六页原生/扫描同页、同一行、扫描页、旋转和空白语料、7 组页/DPI 对照（含非整数 DPI 137）已通过独立旧 Python CPU DB/CTC 与 PDF 字符事实验证：8 行复用原生字符，恰好 4 行调用识别模型。另通过裁图非零偏移、无 rec 会话的全原生行、流所有权、阈值仅过滤 OCR 和跨区域累计工作预算；3 项 Rust 单测覆盖唯一字符分配、空格源索引、无效 Unicode/几何/索引及保守回退原因。原生表格选择见原生表格优先章节；仍依赖布局/检测，未补回漏检原生字符，也未完成任意字体/隐藏文字层质量评估、完整段落重排或跨页整合。

## 已接通：正文 OCR 与表格有序页面块

- 新增 `recognize_page_png/recognize_pdf_page`：一次矩形布局后，文字区域运行 DB/CTC，表格复用既有分类/选优流程，按模型 `order` 稳定组织 `blocks`（同 order 保持布局数组顺序）。保留原 `layout/layout_index`、局部 OCR、页面像素 OCR、表格 HTML 和完整表格识别结果；页面 text 按块连接逐行文字，表格取检测 OCR 行，补识别后的单元格正文以 HTML 为准。
- Rust `PageImage.plan_blocks/project_ocr` 共用自有像素/网格裁图及几何映射。正文和表格合计预检数量/像素预算，默认 256 区域/3200 万累计裁图像素，单裁图 400 万像素；全部规划完成后才开始识别。图片、公式及当前未处理的页眉/页脚等保留 unsupported 块和诊断；空布局不回退整页 OCR，模型漏检仍可能漏文。当前不做重叠文本去重、复杂段落重排、图表标题/脚注归属、原生融合或跨页合并，不能视为完整解析器。
- 基线新增 `export_page_blocks.py`：独立旧 Python CPU DB/CTC 的 3 个正文裁图参考，含真实模型正文区域和指定的标题/正文布局；另外验证模型顺序/同序稳定性、unsupported 保留、全页预算预检、表格接口等价、置信度筛选和 PDF 页/DPI。裁图为明确定义的网格矩形预览，不宣称已迁移旧正文裁图的全部 padding/公式遮盖规则。

## 已接通：单页模型表格与页面坐标投影

- 新增 `recognize_page_tables_png`：PP-DocLayoutV3 rect→筛选 table→旧 5 像素网格裁图→现有分类/双结构选优；保留所有布局区域和原数组索引。`page_tables.rs` 持有解码像素，裁图不借用 Python/FFI 句柄；整批候选先校验数量/累计预算，再逐一识别，无表时不调用表格模型。13 组旧 RGB 裁图合同、真实单表/双表/空白三页（3 表）与独立裁图识别等价、预算和裁图并发通过。
- 每表同时返回局部 `recognition` 与全页 `page_geometry`：格框、OCR 行、有线匹配片段及平移矩阵；保留 OCR/格索引和模型角点顺序。修复了空匹配数组的无标签反序列化歧义，改为按已知 wired/wireless 分支解析。四/八值无线框、空匹配及无效索引有回归验证。
- 新增 `recognize_pdf_page_tables`，按原页号渲染后复用 PNG 流程。两个 PDF 页/DPI 场景通过，含 page_index=1 的逐像素一致输入及 DPI 144 空白页。输出为渲染图像素，未输出 PDF point；仍缺自动方向、标题/脚注归属、原生表格规则融合、完整段落阅读顺序和跨页整合。正文模型块接口见有序页面块章节；此处是单页模型路线，不是完整文档解析器。

- `recognize_table_auto_png` 接入 `use_compare_table=False/cls_conf_threshold=0.9`。评分严格小于阈值或显式比较时运行两个结构分支，共用一次检测/区域 OCR，按旧规则选优；最终 `route` 可不同于分类标签，`comparison` 保留计数和原因。197 组决策/计数合同覆盖六种原因，4 个实际双模型对照、默认双分类器低分比较、一次 OCR 共享及错误/模型缺失通过。坏结构或资源错误不伪装成成功候选；关闭评分触发可设置阈值 0。
- 新增 `rapiddoc-render/src/quality.rs` 和 `select_table_candidate`，移植旧函数规则优先级、Python ties-even 舍入、字面 OCR 命中、重复/空文字、Unicode 空白和旧 `<th` 前缀（含 thead）计数。只接受本输出器的规范闭合 HTML，不用通用 HTML 修复库；标签/属性/实体受限，16MB 候选输入、1000 OCR/1MB 文字及 50MB 检索工作预算，错误和并发通过。
- 修复分类评分的 JSON 边界：FP32 先精确转成 f64 再序列化，与 Python `astype(float)` 一致。已复现旧短序列化将 `0.8999999761581421` 变为 `0.9`，从而改变 `<0.9` 分支；概率输出精确值、等于阈值不比较和 FP32 0.9 触发比较均有回归验证。

- 新增 `OnnxSession.classify_table_png`（`paddle/qanything`）及 CLI `classify-table-png`。Paddle 使用 BGR/Lanczos4、短边 256、224 中心裁图；Qanything 保留旧 COLOR_RGB2GRAY 通道语义，移植 Pillow 12.2.0 的灰度 BICUBIC 双遍/22 位定点缩放。各 13 组 FP32 张量逐位一致，6 组解码、6 个实际 CPU PNG、4 组共识策略、CLI、预算/错误恢复通过。Paddle 默认概率、Qanything 默认 logits，避免重复 softmax；自定义导出模型须明确指定输出类型，不自动检查图猜测。固定 Pillow 源码哈希与 MIT-CMU 许可已记录/分发。
- `recognize_table_auto_png` 对显式裁图按分类调用 UNET 或 SLANet，支持可选 Qanything 第二分类器。复用旧共识规则：两分类器不一致选 wireless、评分取较小值，保留成员预测。单/双分类器到 UNET/SLANet-plus/SLANet-1m 的真实路由通过；未触发比较时只调用选中分支，坏空白结构错误继续传出。低置信度比较见本节顶部，尚不代表完整旧解析器等价。

- UNET 新增 `recognize_wired_table_png` / `match_wired_table`：有线格框→OCR→投影匹配/跨格拆分→同行合并→空格补识别→噪声边缘裁剪→安全 HTML。59 组匹配合同、10 组 HTML/裁边合同、6 组彩色补识别裁图/对比度和 5 张 PNG 全流程已与 Python 参考对照通过。数字/日期/CJK 拆分使用固定 Unicode 16.0.0 属性表，不依赖 Python 正则或第三方图像库；补识别保留旧通道交换、对比度及伪字过滤规则。保留 OCR 源索引和未匹配文字，补识别片段源索引为 null；无格框时也不丢弃 OCR 诊断。重叠逻辑格报错，文字统一按纯文本转义，不承诺旧 HTML 字节等价。已接矩形单页模型路线，完整版面集成仍未完成。

- UNET `OnnxSession.table_cells_png` 已串接神经推理、源尺度线几何、2px LINE_AA、最大外轮廓角度、nearest/BORDER_REPLICATE 旋转、闭合格发现、原图逆变换、同行排序及逻辑格。24 组绘线与 36 组旋转灰度图逐像素一致，24 组外轮廓角度、12 组倾斜网格合同（包括坏轨道拒绝）、5 张 PNG（wired/wireless/blank/±3° wired）的区域与逻辑格通过；并发和失败后恢复通过。像素 400 万、边长 4096、线段/连通域 2000，绘图/角度工作量有预算。保留 OpenCV drawing 的 Intel BSD 声明及 5.0.0 warp 宏来源哈希，不依赖 OpenCV 动态库。

- UNET 已新增 `table_line_geometry`，从原始 0/1/2 mask 回缩到源图、分离横竖线、矩形闭运算、8 连通域、中心线、补连及端点延长。20 组回缩/闭运算图与连接线输出逐位一致；84 组独立补连/延长端点、48 组连通域/BBDT 块顺序/旋转外接框逐位一致。采用单调队列闭运算，线性像素工作量；限制 400 万源像素、4096 边长和 2000 连通域，连接数量及工作量有预算。
- UNET `recover_table_polygons` 实现基准行列、轨道插入、跨行跨列及源索引逻辑格恢复，30 组合同通过（含实际 UNET 区域框）。保留相邻 y 聚类、首个最长行、距离首索引及固定 10 合并容差；Python 列表入口使用 FP64。旧规则在一组样本产生 [col_start=4,col_end=3]，Rust 明确报 invalid_input，不伪造有效跨度。已接有线裁图 OCR/HTML，分类路由另见本节顶部，已接单页模型流程，尚未接完整文档版面。

- UNET 新增 `OnnxSession.table_line_mask_png`，长边 1024 等比 cubic/area、BGR→RGB、逐步 FP32 写回归一化和 int64 线类别图。14 组张量逐位一致，包含极小图、整数/非整数 area 缩小；3 PNG 原始类别图精确一致。通过差分修正通用 cubic 的非 FMA 运算及 IPP 边界路径横向积累顺序，已有 15 组布局张量仍全部通过。当前返回模型缩放图的 0/1/2 类别图，不宣称已完成 UNET 线修复和格拓扑。

- P09 新增 `match_table` / `recognize_table_png` / CLI `recognize-table-png`。Rust 完成连续面积 IoU→距离→首索引平局匹配，保留原 OCR 索引及未匹配文字，校验结构 token/格数、跨度、覆盖和闭合；复用安全 HTML 输出。8 组几何合同、4 组真实 SLANet/plus wired/wireless 完整裁图流程、并发和转义通过；两模型的空白坏结构按合同拒绝，不修补成成功表格。文字视为纯文本，保留原空白，HTML 是规范化片段，不保证与旧 HTML 字节等价。单页候选已接通，完整版面集成仍待迁移；UNET 裁图路线见本节顶部。

- P09 新增 `table_structure_png` / CLI `table-structure-png`：SLANet-plus 与 SLANet-1m 的 488 长边 BGR/FP64 normalize/FP32 zero pad、字典转换、非 CTC 序列解码、EOS、重复 token、bbox 过滤及 plus 回缩。7 个前处理用例（有效张量逐位一致）、2 个序列合同、两模型 × wired/wireless/blank 的 6 个 PNG 原始结构基线通过。完整裁图识别另行验证结构有效性。

- P09 已新增 `OnnxSession.layout_png` 和 CLI `layout-png` 的 PP-DocLayoutV3 rect 模式。15 组 RGB/cubic/CHW 张量逐位一致（包含实际 800x800 图、随机上下采样、边界和极小图）；15 组矩形后处理规则和 3 张 PNG 输出与当前 Python 基线匹配。已覆盖阈值、同类/跨类 NMS、大背景 image 过滤、large 类包含淘汰、模型顺序、unclip=1 的 FP32 运算与边界裁剪。mask 多边形、自定义类别阈值和其他 PP 版本尚未实现。
- cubic 通过独立脉冲响应与图像差分校准当前 OpenCV 5.0.0 / IPP 2026 AVX2 的 phase/系数与积累顺序；极小图保留整数回退。代码不链接 IPP/OpenCV；其他后端/平台的逐位等价仍 pending。NMS 相同评分使用明确的后出现索引优先，源 NumPy 不稳定排序的偶然排列不作为跨平台保证。

- P09 运行层支持 `OnnxSession.run_named`：命名多输入 FP32，输出保留 float32/int32/int64，输入/输出分别按合计元素预算检查。合成双输入模型（含超过 2^53 的 int64）及真实 PP-DocLayoutV3 三输入/三输出差分通过，整型逐项一致、FP32 容差 1e-5。缺输入、坏形状/多余字段、预算、旧 run 的多输入拒绝和错误恢复通过。

- `parse_pdf_ocr` 已接入 Rust 页面路由：原生文字页保留原文、空白页不调用模型、需要 OCR 的无原生文字页才推理。纯原生范围可不提供模型，需要模型时明确报错。OCR 四角转换为视觉左上 point；原生 Document facts.text 保留，处理成功诊断改为 ocr_completed。
- 新增混合 PDF 原生/扫描/空白三页的 wheel 安装测试，验证页范围保留原始页号、point 坐标、流所有权、缺模型、每页/累计像素和 OCR 页数预算、并发与失败后恢复。此路由尚不补识别部分原生文字页面中的图片块，不等同于完整版面混合 OCR。

- 新增 Rust DB 检测前处理、概率图后处理、透视裁图和区域识别。Python 可调用 `OnnxSession.detect_png/recognize_regions_png`、`ocr_png`、`ocr_pdf_page`；CLI 新增 `ocr-png` / `ocr-pdf-page`。检测、同行排序、裁图、CTC 均在 Rust 内执行，不依赖 Python NumPy/OpenCV/ORT。
- 18 组检测尺寸/模式用例：有效 FP32 张量逐位匹配当前 RapidOCR，2 组 CPU 概率图匹配。24 组 DB 样本覆盖旋转、触边、孔洞、噪声及真实模型输出，文本框坐标完全一致、评分容差 1e-5。固定阈值/膨胀/扩框默认来自 RapidDoc 配置；seal poly/slow 模式尚未迁移。
- 6 组透视裁图（轴对齐、竖行、斜切、透视、越界复制、极小图）与当前 OpenCV 5.0.0 逐像素一致。修正了 FP32 calipers/量化后角点重排、LINE_8 mask 边界和反向透视矩阵不再归一化的舍入差异。保留 OpenCV/Clipper/imageproc 许可；未增加 OpenCV 或 Clipper 动态库。
- 3 张真实文本/15°旋转/空白 PNG、4 个检测区域识别和对应 3 页扫描 PDF 在隔离 wheel 安装后通过；识别文字一致、置信度容差 1e-5。路径/bytes、并发、非法四边形、错误恢复及 PDF OCR CLI 均通过。
- DB 图像最多 400 万像素、候选最多 1000、几何/评分工作量有预算；区域识别最多 1000 个区域、累计裁图最多 3200 万像素。错误明确返回，不将预算不足或坏输入伪装为空白页。
- `parse_pdf_text` 仍保留原来的扫描页错误/allow_partial 行为；新接口负责页面级路由。页面内混合图文 OCR/去重合并、方向自动策略、版面/段落、Office 与正式兼容 API 仍待开发，GPU 实机验证继续留到服务器。
- 方向模型新增 `OnnxSession.classify_orientation_png`：匹配 RapidOrientationModel 的 RGB 路径（256 短边、224 中心裁、FP32 ImageNet 归一化）；5 组张量逐位一致，4 张 PNG 的角度/评分及并发通过。只实现显式分类，原解析器宽高比/竖行启发式、自动旋转和坐标还原尚未接通。
- 上一检查点指定原 Python 回归为 417 passed in 29.95s，本轮未重跑、未改生产 Python 解析器。本轮 Rust workspace 97 项测试、fmt、clippy 与 Python binding 的 CUDA feature 编译全部通过；36 份 vendored 源码/测试哈希未变。新版 wheel 隔离安装后的 40 组安装验证和 18 组 PDF RGBA 精确对照全部通过；新增旧 para_split 的 850 组完整 JSON 差分及并发/预算验证；兼容性入口默认关闭、元数据分发/哈希、实验偏差和模型默认路线验证仍通过。新增 16 组块序合同 × 3 比例、完整块保留/整页歧义回退/并发/预算及真实 OCR/原生双栏页验证，含 720 种排列和 1000 区域；14 组行序合同 × 3 比例、歧义保留/默认兼容/重排累计预算及打乱真实 OCR/原生页后的恢复验证；19 组列表续行合同 × 3 比例、双栏/边界/并发及真实扫描和 72/137 DPI 原生列表页验证；13,930 组旧换行差分、7 组段落几何合同及真实 OCR/原生正文/表格页面验证；既有 8 页/10 组原生表格合同，含像素对照、6 张无识别模型的纯原生表、同页原生/扫描表共存、真实模型裁图等价和真实布局→原生表格；原有正文融合、表格双结构选优与评分阈值仍通过。复测入口为 `scripts/migration/check_wheel.py`，全量日志为 `build/wheel-all-tests.txt`。使用 `-j 1` 控制本机编译内存。来源声明随 wheel 分发，源码 SHA256 见 `packaging/inference-provenance.json`；GPU 实机推理始终 pending。

## 已完成的基础能力

- wheel 实际安装验证已接通上述原生/OCR/布局矩形/表格裁图接口，提供 `python -m rapid_doc_rust` / `rapid-doc-rust` 命令。已新增 SDK 单页模型表格入口，完整 RapidDoc API 仍在开发。
- 页面渲染通过 18 组逐 RGBA 像素完全一致的对照，覆盖 72/144 DPI、扫描/混合页、四种 CropBox/Form/旋转组合；像素预算、DPI、页号和错误恢复均验证。渲染复用唯一运行库锁，复制自有 RGBA 后释放锁再编码 PNG。通过差分修正 fixed_size 没有缩放内容的问题，当前使用 target_size。
- 区域表格恢复已经通过真实 wheel 的 3x2 格、合并表头、单元格文字、HTML 和无效区域拒绝验证；尚不进行整页候选发现/仲裁。表格前后处理识别并保留真实空格，仅剔除不同单元格之间的 PDFium 生成分隔空白。同一格内生成空格保留。
- `rapiddoc-native` 已包含物理轨道组装、完整外边界验证、矩形合并格、碎片覆盖、轴线提取/裁剪/非等比线宽/共线合并及稳定排序；覆盖查询复用一次内核缓存，有计算预算。`rapiddoc-render` 已实现验证跨度完整性的安全 HTML 表格输出。
- 7 份 PDF 几何差分还新增逐项验证已合并轴线的顺序、起止点、线宽和框；描边线通过，填充细条和复杂任意裁剪仍需补齐。
- OCR CTC 后处理通过 5 组中英文/空白/batch 样本；文字、token、时间步精确匹配，置信度固定 1e-5 容差。模型字典从 ONNX 元数据读取并缓存，blank 隔开的重复字符保留。
- OCR 识别前处理（BGR、uint8 双线性、NCHW、归一化、有效宽度和零 padding）已移植。13 组随机/极小/单行/整数缩小/非整数缩放/真实文本张量与当前 RapidOCR/OpenCV 5.0.0 的 FP32 位模式完全一致；4 张真实 PNG 行的端到端识别与 Python 一致，含中文、英文与空白。
- PNG 的检测、透视裁剪和识别全部计算在 Rust 内完成，包含白底透明合成；当前只接受 PNG，word boxes、RTL 和复杂版面融合尚未完成。
- OpenCV 定点插值参考与许可已保存在 Rust crate 和 Python 包 notices；模型、识别字典不另行下载。通用 ONNX 接口仍需显式 ORT runtime 和模型哈希。
- 原 Python 回归最近一次为 450 项通过。近期 Rust tests/clippy 均通过，后续新增模块仍需对应复测。版本与报告脚本见 rust-runtime-matrix.md；当前 wheel 仍是本地开发产物，不是完整功能发布包。

下一步继续完整文档版面/原生阅读顺序、复杂原生表格规则与跨页融合、原生/OCR 漏检补偿与复杂字体、公式及 Office。用户已明确无需阶段性确认，GPU 实测不阻塞。

## 历史接续：PDF 原生文字、几何和可复用 ONNX 会话（最新能力以上节为准）

- 新增 `rust/rapiddoc-pdf`、`rust/rapiddoc`、`rust/rapiddoc-cli`、`rust/rapiddoc-inference`。Python `parse_pdf_text()` 已可处理 bytes/路径/流、原始页范围及密码。PDFium 在 Rust 全进程单一所有者与整次调用锁内运行，计算释放 GIL，返回自有 JSON；预览暂要求与 pypdfium2 在不同进程运行。
- 默认遇到扫描页返回 `model_missing`；`allow_partial=True` 才返回部分文字和 `ocr_required` 诊断。空输入、坏文件、页范围、输入/字符/页数超限有稳定异常 code，错误后合法调用可恢复。
- wheel 已包含独立命名的 PDFium DLL、版本/SHA256 清单及来源许可文件。隔离环境无需指定 DLL 路径即可通过三份 PDF golden、24 次并发、流所有权、密码、页选择、错误恢复验证。运行库 staging 脚本和清单在 scripts/migration 与 packaging；不把暂存 DLL 本体加入 Git。
- 可复用 ONNX 会话提供模型 SHA256 校验、显式运行库/设备、线程和张量预算。`OnnxSession` Python 接口已接通；它是单输入 FP32 底层接口，尚未接模型前后处理。隔离环境没有 NumPy 或 Python ORT，安装 wheel 后六组已导出张量、会话复用/并发及错误测试通过。
- ORT 固定 `2.0.0-rc.15`、`api-26`、`load-dynamic`，关闭默认下载，以本地 ORT 1.26.0 测试。q_cls、paddle_cls、rapid_orientation 的 batch 1/2 六组与 Python CPU 输出在 atol/rtol=1e-5 内一致。模型不打包、不下载；ORT runtime 当前需显式提供。
- CUDA feature 已通过 `cargo check -p rapiddoc-inference --features cuda`；设备注册使用 error_on_failure，不静默改为 CPU。GPU 实机推理仍 pending，按用户指示留到服务器。
- 新增独立 Affine/页面坐标契约，0/90/180/270、CropBox、嵌套矩阵、空裁剪测试通过。P03 物化路径控制点、闭合标记、颜色、描边、Form 路径/累积 clip、图片框和 URI 链接，不持有 FFI 句柄；对象、片段、深度和链接均有预算。
- 七份 PDF 的字符原点/部分 loose 框、路径证据框/数量/颜色可见性/Form 深度、图片框和链接在固定 0.001 point 容差内匹配 Python。含四份新生成的两层 Form/非等比缩放/CropBox/页旋转组合；脚本为 `check_pdf_geometry.py`。任意曲线的精确裁剪、全部字体语义尚未验收。
- 差分抓住并修正 `text.all()` 在裁剪旋转页漏字的问题，当前从完整字符索引构建文字；未知 Unicode 保留原码并诊断替代符。
- 本地 DocVortex core 作为固定快照存于 `rust/vendor/docvortex-core`，36 份源码/测试 SHA256 与来源一致。`rapiddoc-native` 的边界校验已接入矩形单元格连通及区间覆盖内核，3 个适配契约测试通过；还没有整页表格识别器。
- 当前按模块重新构建并安装开发 wheel；交付前继续执行安装回归。不代表完整 RapidDoc API 或六格式 Rust 支持已完成。

继续项：原生规则层的文本/表格/阅读顺序与输出器、CPU 模型前后处理/OCR、Office 各格式、Python 正式兼容 API、运行库多平台打包。GPU 不作为本机开发停点。

## 首个 Rust 骨架记录（历史，以上节为准）

- 用户明确本机无 GPU，GPU 实机验证留待项目完成后到服务器测试。此项不再阻塞开发；后续按模块补模型张量基线，不等全量 P00 语料完备才启动 Rust。
- 根 Cargo workspace、Cargo.lock 已创建；`rust/rapiddoc-core` 实现稳定错误分类、有限有序矩形及反序列化校验、页范围、输入资源限制。3 个 Rust 契约测试通过。
- `experiments/rust-probe` 使用 pdfium-render 0.9.4，在独立 Rust 进程加载 PDFium。原生/扫描/混合 PDF 三份文字、页数、字符数与 Python golden 完全一致；还可导出字体名、旋转角、原点和字框，几何差分待补。
- 相邻 DocVortex core 离线契约测试 20 项通过，未修改其源码或 Cargo.lock。
- `bindings/python` 已构建 `rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`，新建隔离环境安装后通过 smoke；环境没有 rapid_doc、torch。当前只有基础契约，尚不支持文档解析，不是可替换旧包的成品。
- rustfmt/clippy 已安装；workspace 测试和 clippy 通过。详细版本、能力与剩余验证见 `rust-runtime-matrix.md`。
- 下一步：完整 IR/坐标契约、PDF 字符/路径差分与资源生命周期、CPU ORT 探针，再逐步接通解析和 Python API。GPU 代码仍须实现，实机通过状态始终保持 pending，直到服务器测试。

## 本次续作结果（以此节为准）

- 从相邻 DocVortex 恢复 15 份缺失文件：六格式历史 Office、8 份特殊 OOXML、字符去重 JSON。保留原始字节、根许可声明、来源提交及 SHA256。未修改相邻仓库，也未修改 `todo.md`。
- 旧 PDF 测试改为验证每页都保留表格前导信息，补齐 drawing_lines 和正文归属断言；依据是当前生产实现与 UPSTREAM 已记载的规则。没有改生产解析器。
- 特殊 OOXML 检测测试使用明确的 8 个样本路径，缺样本会失败，不再允许空 glob 跳过。
- 文档清单扩展至 **23 份**，保留原 6 份 golden 不变，新增 17 份。三份 PDF 包含原生表格/多栏/旋转/链接、扫描图片页、混合三页；导出自有字符/图元事实，未接通 OCR 或 PDF 最终 Markdown 基线。
- 扩展清单重复生成一致。仓库模型清单现包含 default_models.yaml 声明与范围标记；尚未加载模型，也未核实各模型许可。
- 现有回归加扩展 golden：**449 passed in 47.18s**，无失败/跳过。随后新增的恢复样本来源哈希测试单独运行：**1 passed in 0.07s**。之前 33 个失败均已处理。
- 23 份文档的 warmup=1/repeat=5 基准成功，重复输出哈希稳定；最初基准与测试同时运行，不用于性能结论，另运行无并发测试的 `build/bench/python-expanded-isolated.json`。
- inventory 仍报告 11 份缺失的历史迁移文档/源码清单，均列在 inventory_snapshot.json；已没有缺失的历史测试输入目录。不要以此宣称原始上游来源审计已完整。

此处记录的是进入 Rust 开发前的基线进展，最新状态见上节；模型张量和特殊语料按对应模块继续补充。

## 固定决策

- Rust 代码使用 `rust/rapiddoc-*`，根 Cargo workspace；绑定在 `bindings/python/`。
- Python 预览发行包 `rapid-doc-rust`；导入 `rapid_doc_rust`；扩展 `rapid_doc_rust._native`。主 Rust crate 为 `rapiddoc`。
- 本期只交付 Python；其他语言和 C ABI 必须等用户实际测试确认并明确要求。
- 现有 `rapid_doc` 保留；不覆盖 DocVortex 本地修复。包名可用性尚未查询。
- HEAD：`d0e98fb550f741a3b979bf79217f7f8e180f8812`；用户原有 `AM todo.md` 未修改。未改旧 Python 生产解析器、未提交；Rust workspace 已创建并持续开发。

## 已实现

- inventory：AST 导出 RapidDoc 参数、输出字段、MakeMode；记录仓库内 ONNX/YAML 哈希、环境、范围和缺失引用。
- 可重复生成 DOC/DOCX/PPT/PPTX/XLS/XLSX 六个最小输入；manifest 保存 SHA256、来源、生成脚本及辅助函数文件哈希。
- exporter 保存 Markdown、middle_json、content_list_json 和完整资产；拒绝覆盖已有结果。
- comparator 严格比较 JSON 的类型、字段、列表顺序、文本空格与资产。
- benchmark 支持 warmup/repeat、读取/构造/解析耗时、输出哈希、进程生命周期峰值 RSS。
- 15 个工具及 golden 回归测试，含缺失输入、空清单、哈希错误、重复 ID 等负例。

## 首轮验证证据（历史记录，失败已由本次续作处理）

- `python -m pytest tests/migration -q`：**15 passed in 3.38s**。
- 生成器重复执行：manifest 与输入哈希完全一致。
- 导出到独立目录再比较：6 份输出完全一致，包含图片字节。
- inventory `--check` 通过；报告 12 项缺失历史引用（并非全语料完成）。完整报告保存在 `tests/migration/inventory_snapshot.json`。
- 原计划回归命令：`python -m pytest tests/native_pdf tests/native_office tests/test_office_native_integration.py tests/test_native_pdf_integration.py -q --tb=line`：**376 passed, 33 failed, 1 skipped in 11.93s**。
- 32 个失败源于缺失 `tests/fixtures/pdf_char_dedup.json`、`tests/fixtures/docx/handmade-*`、`tests/fixtures/pptx/handmade-*`、`tests/fixtures/docvortex_migration/*`；不通过 skip 隐藏。
- 另一个失败：`test_table_preamble_keeps_first_copy_and_changed_values` 的 SimpleNamespace 没有 `drawing_lines`，在 pipeline.py:782 报 AttributeError。旧测试还期望重复页前导文字被删除，需对照当前“每页保留”规则澄清后修订，不能仅补字段刷绿。
- 唯一 skip 来自既有测试；P00 未新增 skip。完整日志在本地 `build/migration-test-results.txt`。

基准：Windows 11、Python 3.14.5；warmup=1，repeat=5；仅当前 Python 原生 Office，不涉及模型或 GPU。

| 样本 | parse 中位数 ms |
| --- | ---: |
| DOC | 1.41 |
| DOCX | 209.84 |
| PPT | 3.87 |
| PPTX | 8.10 |
| XLS | 2.81 |
| XLSX | 5.13 |

同进程生命周期峰值 RSS 最终约 366.6 MiB；不可解释为各格式独立峰值。完整结果 `build/bench/python.json`。没有 Rust 性能结论。

## 接续 P00

1. 历史 fixtures 已恢复并通过原断言；补充外部内容许可来源核验，不将根许可声明当作第三方内容授权证明。
2. 原生/扫描/混合 PDF 事实已覆盖；继续扩展 CID、跨页表格、公式/印章等语料及最终输出。
3. 补六格式异常容器、资源上限、链接、公式等登记样本；已有单测覆盖不能自动视为跨引擎 golden 覆盖。
4. 深入配置字典逐项分类；登记外部模型缓存、模型许可/哈希和前后处理张量合同。当前 inventory 只扫描仓库，不扫描外部缓存，也未实现模型下载。
5. 补充实际 EP、CPU/CUDA 和标准/自定义图的独立基准及内部阶段 profile；当前只有外层 read/init/parse。
6. 旧 PDF 测试差异已处理；剩余门槛完成后进入 P01。

P01 需参考相邻 `D:/CodeProjects/doc/DocVortex/rust`（只读，其工作区有用户改动），对照 firecrawl-pdfium 与 pdfium-render 能力矩阵。不能把已有文档评审当作编译、FFI 生命周期或运行库验证。

上下文压缩由宿主系统管理；此文件和两份计划是接续依据，不声称已手动触发压缩。

## PPT 旧格式阶段验收（2026-10-31）

- Rust `.ppt` 已接入 Python `RapidDoc`，覆盖文字、链接、列表、备注、隐藏页、表格、图片、公式及图表资源回退。
- `synthetic.ppt`：2 页成功；`demo/ms_office_docs/pptx_01.ppt`：6 页成功、4 个资源。
- `.ppt` 性能样本 `build/ppt-perf-20261031.json`：Rust 33.45 ms，Python 64.75 ms，约 1.94x。
- `.ppt` 内存样本 `build/ppt-memory-worker-20261031/`：Rust 峰值工作集 83.1 MiB，Python 317.3 MiB；Rust 峰值私有提交 84.8 MiB，Python 1024 MiB。
- PPTX 全量安装验收仍为 279 组成功、1 组原始错误；Office 兼容冒烟通过。
- 当前 PPT 阶段可进入交付整理；图表复杂样本仍建议后续补充，不阻塞基础 `.ppt` 使用。

## 2026-11-01 DOC 格式属性迁移

- 新增 DOC CHPX/PAPX FKP 页面读取、BTE 页索引、按 FC 二分查询和有界 SPRM 遍历。
- 参考当前 Python formatting.py 的截断页跳过及可恢复尾部行为，尚未将原始 SPRM 应用到公开 DOC 输出。
- 固定两个原始 DOC 的 Python 参考数据：doc_formatting.json；真实样本包含 1995 个字符 run、582 个段落 run。
- DOC 表格、图片、字段、样式继承和完整分页仍未交付；当前公开 DOC 为基础正文预览，不能视为与 Python 等价。

## 2026-11-01 DOC 公开预览修复

- FIB 补齐原 Python 固定槽位恢复；无 CLX 时按 fcMin/fcMac/ccpText 构造单 piece；有 CLX 时必须解析，缺 Table 不再静默输出空正文。
- 使用 PlcfSed section 范围归属段落，停止将分页符直接解释成逻辑页。真实 docx_01.doc 的原生转换已恢复 3 个 section，与 Python 页数一致。
- 字段栈保留嵌套缓存结果、跨段 TOC 和 HYPERLINK；控制字符不进入公开文本。软换行、制表符及不换行连字符按原 Python 转换。
- 本次仍未实现完整样式继承、表格、图片、公式等语义，公开入口输出明确预览诊断；不得将纯文本性能作为等价转换性能。
- DOC 相关测试 9 项通过（新增输出/字段测试 4 项，原 FIB/文本/FKP 测试 5 项）。

## 2026-11-01 DOC 样式继承接入正文

- STSH/STD 样式表读取、迭代 based-on 继承和循环链截断已实现；真实 DOC 的 59 个样式名称、字符格式及标题/目录/代码语义与 Python 对照一致。
- 正文字符按段落/字符样式、CHPX、piece Prm 依次解析，保留相对 toggle，过滤 hidden/deleted，输出粗体、斜体、下划线、强调、删除线、上下标。
- 标题/文档标题从样式继承接入正文块。段落 PAPX 的表格/列表/outline 覆盖尚待语义接入；目前保存了样式链 paragraph grpprl。
- 受影响 DOC 测试 11 项通过。本轮未构建 wheel，安装版本仍是上一轮 DOC 基础修复。

## 2026-11-01 DOC 段落属性与表格定义

- 新增 Paragraph SPRM 语义层：表格/行/内层单元格标志、outline、列表引用、层级绝对/相对调整、huge PAPX，以及 TDefTable 单元格边界和横纵向合并标记。
- 段落属性按样式继承、PAPX 默认增量合并、piece Prm 顺序应用；huge PAPX 读取 Data，递归时禁用二次 Data 查找，与 Python 一致。
- 真实 DOC 的 582 个 PAPX 增量属性（含表格定义）与 Python 完全一致，8 项受影响测试通过。
- 正文标题已应用 PAPX outline 覆盖；表格重建、列表编号和对象资源尚未接入输出。未重新构建 wheel。

## 2026-11-01 DOC 表格输出链路

- 段落终止后保留表格标志和空单元格，连续表格段落按 depth 重建行、单元格和嵌套表格，使用行 twip 边界统一网格；支持横向续格、纵向起始/续格、表头以及缺失 TAP 的伪边界回退。
- 表格接入正文输出，消费原段落避免重复。已修正正文标题 level 为 Python 的文档标题 1、段落标题 heading+1（2..6）。
- 表格单元格列表、对象图片/公式和完整富文本 HTML 差分仍待补齐，不声称整表与 Python 完全等价。本轮未重新安装 wheel。

## 2026-11-01 DOC 表格富文本与安装验证

- 单元格 HTML 对齐 Python 样式嵌套顺序、上下标优先级、换行和转义规则，链接按文本 run 安全输出；384 组样式/链接参考全部通过。
- 10 项受影响测试通过。构建 build/office-doc-tables-20261101 wheel 并安装至项目 .venv，包含前几轮样式、段落属性和表格重建改动。
- 真实 docx_01.doc 为 3 个逻辑页；8 张表的 table/tr/td/th 与 rowspan/colspan 结构全部匹配 Python。Python 另有 1 个 chart block，不混入普通表格统计。
- 结构一致不代表资源和内容完全一致：单元格图片、公式、列表和嵌入图表仍缺，详见 build/doc-tables-comparison-20261101.json。

## 2026-11-01 DOC 行内图片资源

- PICF/Data 偏移读取、OfficeArt/BSE/BLIP、旧图片签名回退已接入，缓存 Data 偏移及内容去重计量，复用 Rust 图片序列化和 MathType 图片公式解码。
- CHPX sprmCPicLocation 将行内对象绑定到段落，正文输出图片块，表格单元格输出内联 img/eq。无效位图采用图片级诊断回退，避免中断整份 DOC。
- 尚未接入 floating drawing、ObjectPool 原生公式/图表及字段对象仲裁。当前样本 PICF 参考未包含图片公式，不能据此宣称公式接入验收完成。
- 7 项受影响测试通过；157 个 PICF 候选的可输出/回退结果与 Python 对齐（不等同逐像素验收）。参考导出脚本保存为 scripts/migration/export_doc_images_reference.py。
- 已构建 build/office-doc-images-20261101 wheel 并安装 .venv；真实 DOC 3 页、46 个资产、20587 字符 Markdown，控制字符不泄漏。Python 当前样本 48 个资产，仍需补齐浮动图及对象语义，不能以数量接近视为完整等价。

## 2026-11-01 DOC 浮动图片与对象池

- 接入 PlcfSpaMom/OfficeArt drawing 浮动图片锚点，按 shape id 与正文 CP 绑定；复用 BStore/BLIP 解码、图片去重和矢量序列化。
- 接入 ObjectPool Equation Native 扫描，加载数量写入诊断；复杂字段对象仍需把 storage id 绑定到对应字段结果。
- 真实 DOC 重新构建并安装 wheel 后：synthetic.doc 1 页/0 资源，docx_01.doc 3 页/48 资源，原始控制符无泄漏；与当前 Python 样本资源数量一致。
- 受影响 DOC 测试通过；复杂对象公式、图表和列表语义仍需专项差分。

## 2026-11-01 DOC ObjectPool 公式与浮动对象闭合

- ObjectPool Equation Native 已按 CHPX 图片位置绑定到正文/表格，输出 Equation；浮动 OfficeArt 图片按 PlcfSpaMom CP 锚点绑定并输出。
- 受影响 DOC 测试全部通过：正文、图片、段落、表格、样式和字段。
- 新 wheel build/office-doc-objects-20261101 已安装至 .venv；两个真实 DOC 验证保持 1/3 页、0/48 资源、无控制字符泄漏。
- 列表编号、ObjectPool 图表及复杂浮动对象仍需专项语义接入；图片与公式已具备基础资源链路。

## 2026-11-01 DOC 列表试验回退

- 试接 `ilfo/ilvl` 到 List 块后，真实 docx_01.doc 产生 47 组列表，而 Python 只有 11 组，说明缺少 PlfLst/PlfLfo 定义解析会误判普通段落。
- 已撤回该过宽输出条件，保留 PAPX 列表属性读取；`.venv` 恢复安装上一版 office-doc-objects wheel，避免结果精度下降。
- ObjectPool 图表读取已接入源码并通过 Rust 编译；列表编号仍需先完成 PlfLst/PlfLfo，再重新接入公开输出。

## 2026-11-01 DOC PlfLst/PlfLfo 列表定义

- 新增 PlfLst LSTF/LVL 和 PlfLfo override 解析，按九级列表定义维护 lsid 计数、起始值、bullet/none marker 和 ilfo/ilvl 映射。
- 公开输出只对成功命中定义表的段落生成 List 块；未知 ilfo 不再误判为列表。
- 基础 DOC body/paragraph 回归通过。新列表代码尚未重新构建 wheel，真实 DOC 列表数量对照待安装验收。

## 2026-11-01 DOC 列表完整链路验收

- PlfLst/PlfLfo 解析、wheel 构建、安装和真实 DOC 对照已完成。
- 连续列表合并后，docx_01.doc 的 Rust 列表块从 65 降到 15，Python 为 11；页数 3、资源 48 保持一致。
- 剩余 4 个列表差异来自列表容器嵌套和 restart/identity 合并规则，仍需继续修正，当前不能宣称列表完全等价。

## 2026-11-01 DOC 列表重构（覆盖此前列表数量试验结论）

- 删除无条件相邻 List 合并和 List/普通块/List 吞并补丁，新增独立 list_tree::Builder。按 identity、类型和层级维护路径栈；普通块结束列表，列表附属视觉块保持列表状态并独立输出。
- 重写 PlfLst/PlfLfo：u32 lsid、描述符/覆盖区分离、克隆复用列表定义、格式覆盖、编号模板、Roman/alpha、restart 与一次性 override。
- 504 组真实定义上的编号序列与 Python 对照一致；真实 DOC 的 11 个列表容器按完整树骨架（类型、层级、start、嵌套和子项）与 Python 相同，非仅比较数量。
- 增加普通段落不被吞并、图片保持列表栈和类型切换回归；本轮 15 项相关测试通过。
- 仍未完成 DOC 全语义等价：编号 label 尚未贯穿公开叶节点、表内列表、TOC、辅助 story 和嵌入图表需要继续迁移。之前“图表完成”“只剩四个差异”等结论作废。
- 安装验收：build/office-doc-list-refactor-20261101 wheel 已安装 .venv，真实 DOC 为 3 页、48 资产、11 列表、8 table_body；image_body 仍为 61，图片归属尚未等价。公开结果保存 build/doc-list-refactor-installed-20261101.json。

## 2026-11-01 DOC 页眉页脚重构

- 用 PlcfHdd 的分节 story 范围解析独立段落，移除字符串拼接及跨 section 的全局 Footer 去重。按 Python 的第一段继承规则处理页眉，过滤页码并在当前 section/kind 内去重。
- 正文与辅助 story 共用字符样式解析；页眉页脚保留字段链接、字符样式和软换行。正文未终止末段只取正文范围的 PAPX/PRM，避免误用后续辅助 story 的样式。
- 真实 DOC 的全部 4 个 Header、1 个 Footer 按逐字符内容、样式、链接与 Python 对照通过；另覆盖本地去重和 Unicode 页码过滤。504 组编号序列及完整列表树对照保持通过。本轮运行 4 项相关测试，未跑全量回归或 CUDA。
- 参考导出脚本：scripts/migration/export_doc_auxiliary_reference.py。页码判定范围来自 Python str.isdigit，避免 Rust isnumeric 把罗马数字当页码过滤。
- build/office-doc-aux-refactor-20261101 wheel 已安装项目 .venv；公开 API 对真实及 synthetic DOC 验证通过。真实文件保持 3 页、48 资源、11 List、8 table_body，报告 build/doc-aux-refactor-installed-20261101.json。
- 仍待处理：图片归属（当前 image_body 61，Python 44）、ObjectPool 图表和复杂字段绑定、TOC、注释/脚注/文本框、表内列表与公开编号 label；辅助 story 内对象公式未完整迁移，当前样本验证不能覆盖这些情况。运行时通用 DOC preview 警告仍为早期文案，不能据此判断单项是否完成。

## 2026-11-01 DOC 图片、嵌入图表与字段链路重构

- 根因修复：按图片原始内容摘要与 render_size_emu 识别同一载荷，在段落内去重；跨段落仍保留重复引用。缓存图片转换结果，复用 Arc，避免同图不同 Data 偏移重复解码/序列化。段落属性从每次重复解析三遍改为解析一次。
- 删除把二进制 Workbook/Book 交给 OOXML XML 图表解析器的错误路径。新增 xls::embedded_chart，复用 BIFF 工作簿、选区与 HTML 投影，并迁移 MS-OGRAPH datasheet、自定义数字格式、行列选区及异常回退；保留 Python 原有资源限制，无新增人为阈值。
- 原生对象改在 Word 字段分隔符处绑定 ObjectPool id。公式通过富文本 span 返回父字段并抑制预览图片；图表仅绑定 Excel.Chart/MSGraph.Chart EMBED，保留预览，在缺少预览时于字段结束输出一次。正文纯公式段落投影为独立 Equation。辅助 story 对象仍待统一迁移。
- 真实 DOC 的全部正文视觉块顺序及内容与 Python 对齐：44 个图片、1 个图表。45 个图片引用（含图表预览）的 RGBA 像素、SVG XML 与嵌入回退 PNG 像素全部对齐；编码压缩字节不同不视为像素差异。此验证不包含表格 HTML 内的所有资源和完整文档语义。
- 1 个真实 Workbook 与 5 组 MSGraph（完整/选区/坏选区/空选区/截断文本）HTML 参考通过；同时覆盖公式嵌套链接、对象类型仲裁、有/无图表预览和缓存身份。共 21 项受影响测试通过，包含 504 组编号、157 个 PICF 候选和 384 组单元格样式参考；未跑全量/CUDA。
- 参考导出 scripts/migration/export_doc_objects_reference.py，图片对照 scripts/migration/compare_doc_object_images.py；结果 build/doc-objects-image-comparison-20261101.json。
- wheel build/office-doc-objects-refactor-20261101 已安装 .venv，公开 API 验证真实与 synthetic DOC。真实文件保持 3 页、48 资源、11 List、8 table_body、4 Header、1 Footer；image_body 从 61 修正为 44，新增 1 chart_body。报告 build/doc-objects-refactor-installed-20261101.json。单次约 0.40 秒仅为安装烟测，不作正式性能结论。
- 后续：TOC/书签/代码段/caption 完整语义、脚注/尾注/文本框、表内列表和公开编号 label、辅助 story 资源及更多复杂对象语料。当前真实 DOC 没有覆盖原生 ObjectPool 公式的完整二进制链路，字段单元测试不能代替该验收。运行时 preview 文案已更新，不再声称全部对象和文档列表均未接入。

## 2026-11-01 DOC 编号、表内列表与目录完整输出链路

- 保留 ListTables::Info 到段落投影，不再只传四元组并丢弃 label。编号标题补上完整标签；普通列表 raw 叶节点携带 list_label，公开 middle 映射为 prefix，Markdown/content_list/content_list_v2 按原 Python 使用精确前缀并维护后续计数。
- 表格单元格支持嵌套 ordered/unordered 列表、层级变化、类型切换、精确编号及普通段落边界。重构表格中间模型：先保留段落/嵌套表格片段并完成横向合并，再渲染列表，避免合并单元格内的同一列表被拆开；未采用字符串标签拼接修补。
- 新增 STTB/PlcfBkf/PlcfBkl 书签解析，过滤 _GoBack/无效终点/反向范围；标题优先采用 _Toc 书签。TOC 样式和跨段字段状态驱动独立 Index 树，按 run 清理排版页码并恢复目标 anchor，同目标链接重新分组以保留富文本样式。
- 列表/目录的图片公式转为行内公式且只输出一次。空目录条目不生成空树叶；目录内视觉块不打断目录层级。表格外正文及浮动图不再额外生成未使用的 HTML/base64 字符串副本；本轮未做性能/峰值内存定量验收。
- 真实 DOC 的全部标题、目录/列表树按逐字符内容、样式、链接、anchor、label 及层级对照 Python 通过；8 张表的完整 HTML 内容也对齐（此项替换图片 src 后比较，图片像素验收见上阶段）。公开 middle 的标题/目录/列表亦完全对齐，仅排除全局数字 block index；所选导航块的四种导出格式逐项对照通过。
- 8 组二进制书签、8 组目录清理、4 组单元格列表及横向合并格列表参考通过。本阶段共 22 项受影响测试通过，含 DOC 列表/对象/表格和共享 DOCX/Office 输出回归；未跑全量/CUDA。复查中发现并修正了目录链接碎片化和公开 prefix 丢失，未将这些差异留作已通过结果。
- 导出脚本 scripts/migration/export_doc_navigation_reference.py；公开 API 对照 scripts/migration/compare_doc_navigation_public.py，报告 build/doc-navigation-public-comparison-20261101.json。wheel build/office-doc-navigation-20261101 安装到项目 .venv；保持 3 页、48 资源、44 image_body、1 chart_body、8 table_body、11 List。
- 尚未完整迁移：脚注/尾注/文本框及其引用、辅助 story 内的对象资源、code/caption 显式语义与更多复杂 DOC 语料。不得据本轮导航/表格通过宣称 DOC 已完全等价；当前运行时 preview 是部分未完成事项提示，不是穷尽清单。

## 2026-11-01 DOC 共享 story、脚注尾注与完整公开结果对照

- 用 story::Reader 替换正文和 auxiliary 中两套字符循环。统一字段栈、CHPX/PRM 字符属性、段落属性、ObjectPool 公式/图表、图片去重、段落结束及未终止末段行为；每个 story 重置字段状态，复用文档资源缓存。空非表格段落不再错误推进列表计数。
- 迁移 PlcffndRef/PlcffndTxt、PlcfendRef/PlcfendTxt：正文引用按 UTF-16 CP 输出上标编号，脚注绑定引用所在 section，尾注追加最后 section，首段标签、跨段落和辅助文本去重遵循 Python；损坏 PLC、空注释及重复引用保留相同回退。
- 接入 SEQ 字段题注语义。原生公式现在贯穿正文、页眉、脚注和尾注；对照发现的公式前空白差异及公式错误继承超链接已按 DocConverter._rich_text 的分段 flush 规则修复。代码样式生成原始 Code 块，公开输出仍按当前 Python MagicModel 的跳过行为处理。
- 删除浮动图片的空 Text/None 占位块；它们此前没有可见内容，却消耗了公开 block index，导致真实文件索引偏移。
- 7 组合成 Word97 二进制流覆盖跨节/Unicode 引用、空注释、未闭合字段、重复引用、损坏 note PLC、代码样式和真实 ObjectPool Equation Native + CHPX 绑定。全部逐项对照 raw pages、完整 middle（含数字索引）和四种导出通过；原生公式闭环不再仅依赖字段栈单元测试。
- 本轮相关回归共 29 项通过（DOC 正文/辅助文本/列表/导航/对象/表格、共享 Office 输出及库测试），未跑全量或 CUDA；源码生成脚本 scripts/migration/export_doc_stories_reference.py，安装验收 scripts/migration/accept_doc_stories.py，报告 build/doc-stories-installed-20261101.json。
- build/office-doc-stories-20261101 wheel 已安装项目 .venv。真实 docx_01.doc 的完整公开 pdf_info（含全局数字 block index）、Markdown 与全部 48 个图片资源内容对齐 Python；仅对编码不同但 RGBA/SVG/回退图像相同的资产文件名做别名。对照脚本 scripts/migration/compare_doc_full_public.py，报告 build/doc-full-public-comparison-20261101.json。此结论限定当前真实样本，不代表所有 DOC 已验收。
- 复核共同缺口：当前 Python DOC parser 本身未组装 textbox/annotation stories；原始 Code 在公开 MagicModel 被忽略。Rust 暂保持这些行为，不能标记为“已支持文本框/批注/公开代码块”，后续若补齐应同步制定两端输出规则。仍需扩大复杂 OLE/浮动对象、辅助区域资源及非规范 DOC 的真实语料验收。本轮未重做性能/峰值内存对比，也未修改 Python 生产解析代码。

## 2026-11-01 DOC 编码、损坏输入与对象池兼容收尾

- 补齐 Far East 标志下的 lidFE/rgw[13] 选择。DOC 使用 Windows CP932/CP949/CP950，不能直接替代为 XLS 的 Shift-JIS/EUC-KR/Big5 或 WHATWG 编码；新增 DOC 专属冻结状态表，共约 208 KiB 压缩数据。ANSI/CP936 等复用相同语义的原生解释器，未修改 XLS 的编码规则，也不调用 Python 运行时。
- 165,465 个字符的解码结果、CP、FC 和 piece_index 与 Python 逐项一致；覆盖单字节完整字节范围、四种东亚 DBCS 首/尾字节组合和异常 UTF-16 代理项。解码热路径改为直接写入 UTF-16/ASCII 字符，去掉之前逐字符创建 Vec/String 的分配；本轮未宣称端到端速度或峰值内存提升倍数。
- 修复 complex DOC 缺失 CLX 时错误进入 legacy 单分片恢复的问题；无可恢复 piece 且声明有文本时返回原 Python 同类损坏错误。保留合法空文件和非 complex legacy 恢复。FIB story 长度采用地址空间内饱和累加，避免异常计数在 debug 中溢出或 release 中回绕。这些是结构兼容处理，不是新增字符/页数资源预算。
- ObjectPool 图表路径按完整 ObjectPool/_数字/(Workbook|Book) 匹配，忽略无下划线、多下划线和嵌套伪路径；数值别名按原 Python 保留第一个成功解析结果，首个损坏别名不阻止后续有效对象。避免了错误对象替换和越范围识别。
- 15 组 Word 二进制兼容用例（含 5 组预期损坏错误）逐项对照 raw、中间结果和四种导出通过；安装后的公共 API 同样通过，并检查 corrupt_document 错误码。参考 scripts/migration/export_doc_compat_reference.py，安装验证 scripts/migration/accept_doc_compat.py，报告 build/doc-compat-installed-20261101.json。
- 22 项受影响测试通过，包括 DOC 编码/正文/导航/对象/脚注、CFB 基础和 XLS 字符规则回归；未跑全量/CUDA。专属表生成脚本 scripts/migration/export_doc_codec_tables.py，输入来源哈希及许可写入 packaging/office-native-provenance.json 和 wheel notices。
- build/office-doc-compat-20261101 wheel 已安装 .venv。7 组之前的 story 安装验证保持通过；真实样本完整 pdf_info（含索引）、Markdown 和 48 个资源内容再次与 Python 对齐。对照脚本先读取一次输入快照，避免 Word 打开文件时两端读到不同版本。
- 真实语料限制：RapidDoc demo、migration fixture 和本地 DocVortex demo 中的 DOC 为同一 SHA256（6aae60b01f0e020c59d2299f217dbd5904df08182a31384db40953937f322fd1），仍仅 1 份独立真实文件；新增二进制用例不计作新增真实语料。文本框、批注、公开代码块仍是已记录的两端共同缺口。

## 2026-11-01 DOC 速度与峰值内存验收

- 新增 scripts/migration/accept_doc_performance.py。顺序独立进程交替执行两轮；每轮冷调用一次、热调用三次，计时/内存/验证分别起进程。同一输入快照，完整默认公开解析，无模型预加载和精度参数改动。
- 真实 DOC：Python 热中位数 2.079 秒，Rust 0.353 秒，5.89 倍；Windows 进程峰值工作集 332.5 MiB 对 79.4 MiB，Rust 约低 76.1%。完整中间结果、Markdown、content list 和资源规范化快照哈希一致。
- 生成的 6000 段纯文本 DOC：1.376 秒对 0.0646 秒，21.30 倍；进程峰值 348.0 MiB 对 114.3 MiB。此项只作重复文本压力验证，不是新增真实语料或其他格式的性能承诺。
- 内存限制必须一并报告：两端导入基线约 300.7/35.8 MiB，真实解析采样新增工作集约 21.3/42.5 MiB。Rust 总进程内存低，但新增解析内存仍高；未宣称算法自身新增内存已优于 Python。父进程外部采样避免 GIL 干扰，但会漏短峰，主要结论采用 Windows 保留的峰值。
- 结果 build/doc-performance-20261101/summary.json，详细方法和复现 docs/plans/2026-11-01-doc-performance-acceptance.md。本轮没有生产代码改动、wheel 重建或全量/CUDA 回归，项目 .venv 仍为已验收的 office-doc-compat 版本。

## 2026-11-01 DOC 功能收尾与内存优化

- 完成 Python/Rust 的正文及页眉页脚文本框、批注引用/正文和公开代码块。文本框按 MS-DOC shape 索引绑定，链式 story 只读一次，保留正文框内表格/列表；代码块的缩进、空白行和动态 Markdown 围栏贯穿四种输出。批注作者/日期等元数据仍未输出，页眉绘图和精确浮动排版不在完整保真承诺内。
- 内存归因后删除全篇段落暂存和绑定 Value 的整树复制，紧凑保存字符映射。6000 段压力样本正文逻辑存活峰值约降 31.3%，绑定阶段约降 55.0%；这是分配器计数，不等于 RSS 降幅，无新字符/页数阈值。
- 独立真实 DOC 从 1 份扩到 12 份，新增 Apache POI 固定提交语料及完整许可/来源哈希。全部真实输入完整 middle、Markdown、content list 和资源内容对照通过；另固定四种导出参考。新功能包含独立明确预期，不仅做两端同源比较。
- Python 64 项相关测试、Rust Office 阶段 127 项测试通过。最终小改后 DOC 4 项受影响测试再通过。安装后的 32 组二进制用例（含 5 组预期错误）与 12 份真实 DOC 验收通过；未跑全 workspace 或 CUDA。
- wheel build/office-doc-completion-20261101 已安装项目 .venv，可直接运行 demo/rust_doc_demo.py。公开调用方式不变。安装报告 build/doc-completion-installed-20261101.json；来源去重和哈希检查 build/doc-completion-provenance-20261101.json。
- 无并发构建干扰的两轮性能：真实 DOC Python/Rust 热中位数 2.130/0.353 秒，6.04 倍；Windows 进程峰值工作集 333.5/73.5 MiB。Rust 耗时与此前基本相同，峰值由 79.4 MiB 减少约 7.4%。6000 段生成文本 1.429/0.0623 秒，进程峰值 348.9/99.0 MiB；生成输入不能算新增真实语料。
- 解析阶段采样新增工作集仍为 Python/Rust 22.2/36.6 MiB，Rust 虽有改善但仍较高；不得把较低进程总内存解释为全部解析增量指标均更低。12 份真实文件也不能证明所有未知复杂 DOC 等价。
- 完整功能、边界、性能方法与复现：docs/plans/2026-11-01-doc-completion-acceptance.md。上述文本框、批注正文和公开代码块已完成，替代之前条目中的对应“共同缺口”状态；之前的性能记录作为基线保留。

## 2026-11-01 七种格式公开接口统一验收

- 新增 scripts/migration/accept_public_delivery.py：以已安装 RapidDoc() 默认配置对照 PDF/DOC/DOCX/XLS/XLSX/PPT/PPTX。Office 检查完整文件、path/bytes/stream、批量/混合批量、资源前缀/data URI、writer/磁盘及四种导出；PDF 抽查扫描/论文/印章的选页和单页窗口，不声称本轮重跑全部 PDF。
- 首轮发现 PPT 入口外部文本拆散、逐段与列表/标题/表格输出不完整、排序未接入，以及 ExObjRefAtom 字段偏移和 sync-flush OLE 解压兼容问题。按现有 Python 格式规则修复，复用原生 XYCut++、BIFF/MSGraph、富文本和图片公式组件，无按文档名特判或精度阈值变更。
- PPT 完整参考扩到 21 组二进制输入（17 组预期错误），另有 9 组段落/表格投影；压缩流与原生/图片块公式类型有独立断言。当前真实 PPT 仍只有 1 份，不能把构造用例当作广泛真实覆盖。
- 修复空文件兼容：带扩展名提示的空 DOC/XLS/PPT 拒绝，空 PPTX 无页，空 DOCX/XLSX 一空逻辑页，保留无提示低层空字节调用的既有行为。
- 安装后七格式 73 组正向对照和 36 组异常/空文件检查通过；三个公开方法的原参数/default 对齐，额外 CPU 控制均为可选。Office 阶段 130 项回归通过，后续 21 项受影响检查通过。PPT 最后公式类型修正单独构建 R2，仅重跑受影响项，未再跑完整 PDF。
- 统一示例 demo/rust_document_demo.py；范围、原始失败证据和最终安装检查见 docs/plans/2026-11-01-public-delivery-acceptance.md。后续是独立性能/大型文档内存趋势验收，当前 API 正确性运行不作新性能结论。
- 最终 office-public-delivery-20261101-r2 wheel 已安装项目 .venv；新增公式修正后的 13 项原生受影响检查、PPT 安装正向 11 组及异常 4 组通过。R1/R2 Python 运行时代码字节一致，最终打包来源哈希与已安装二进制指纹已核对，统一记录 build/public-delivery-receipt-20261101.json。

## 2026-11-01 七格式性能、规模递增与连续释放检查

- 新增 accept_office_performance.py、accept_document_memory.py、profile_pdf_pipeline.py；沿用 DOC 外部 Windows 内存采样与完整输出像素校验，默认完整解析的时间/内存/验证分进程，顺序交替两端执行。运行包未修改，仍为已安装的 public-delivery R2。
- 六份完整 Office 样本两轮、六次热调用全部输出对齐：DOC/DOCX/PPT/PPTX/XLS/XLSX 热加速比分别 6.44/17.10/2.22/2.88/2.66/1.75，Rust 总进程峰值均更低。DOC/PPT/XLSX 新增工作集仍更高，不混淆两种内存口径。
- 两份完整 PDF 默认模型测量对齐：密集表格 Python/Rust 热耗时 2.740/2.576 s，扫描件 14.147/15.126 s；峰值分别 1911/1343 MiB、2422/1712 MiB。扫描件 Rust 热耗时约高 6.9%，没有宣称所有 PDF 已更快。冷调用含模型加载，另计导入初始化。
- 18 份生成的递增规模负载加 1 份真实 PPT，各规模及最大输入六次连续解析/释放，完整结果均对齐；19 份输入不作 19 份真实语料。Rust 本轮各负载峰值均更低，PDF/XLS 释放值有下降及波动，不能据六次调用保证不存在长期泄漏。PDF 压力为显式原生文本规则配置，跨 64 页窗口，不代表默认 OCR 内存。
- 129 页原生规则发现明显速度差距：诊断 Python/Rust 12.55/39.48 s，Rust native_analysis 28.335 s、render 7.535 s、assemble 2.715 s；内部页面编排/几何与表格候选为主要方向，尚未修复。扫描件模型 layout 两端约 4.3 s，额外图像/组装开销仍待优化。
- 三类页面无损 PNG 滤波实验像素一致，但编码仅省约 1–4 ms/页且体积增加，未采用；没有把微小局部实验冒充端到端提速。测量与风险详见 docs/plans/2026-11-01-document-performance-and-memory.md，统一收据 build/document-performance-receipt-20261101.json。未跑 CUDA、全 workspace 或全部 PDF。

## 2026-11-01 原生 PDF 行所有权与借用优化

- 在文字角色、脚注、辅助文本、行簇和上下标脚本阶段传递已校验行的所有权或借用字符/几何数据，去除内部 JSON 请求及结果转移中的整树复制。原公开/诊断请求入口保留，分类/几何/表格规则不变。
- 无大小预算的生产兼容作用域不再序列化每行/每字只为计数；有限预算仍保留原计数/错误。计数作用域恢复和输出有限预算检查通过，没有新增资源限制。
- 新增固定压缩 Python 参考，271 个阶段用例通过；其中 92 组文字角色、143 组脚注、32 组行簇、4 个实际生成 PDF 角度的脚本。另重建并通过原文字角色与早期脚注的相关参考。固定新参考不依赖可清理的 build 缓存。
- 原生独立例程三轮交替的 16 页中位数 2.493→2.344 s，129 页单轮 20.167→18.927 s，完整输出相同；该诊断边界不含正式流水线渲染/组装，不能当作整体速度。论文两页未见明确收益。
- 最终公开接口同一 129 页规则场景首次/热调用：修改前 38.408/39.489 s，修改后 38.901/38.914 s；热原生分析 28.356→27.343 s。单次热总耗时仅约下降 1.5%，冷调用略慢，尚不能宣称稳定端到端收益，也没有追平 Python。未重新量化峰值内存，不用“少复制”推算内存下降比例。
- 构建 build/native-owned-lines-20261101 并安装项目 .venv；16/65/129 页完整公开输出与上一版本严格一致。最终选页回归、安装指纹和源码记录见 build/native-owned-delivery-receipt-20261101.json；方法与范围见 docs/plans/2026-11-01-native-owned-lines.md。未重复无关 Office、全 workspace 或 CUDA；原生规则剩余表格/几何/编排开销仍需后续定位。

## 2026-11-01 原生 PDF 热点 JSON 传递优化

- 函数级采样确认主要热点为 Value 重建/数字文本处理及树复制；保留 Office/PDF 高精度配置，调整原生事实收集、几何、表格候选、容器和行内文字准备的内部对象构造，已有数组/页面大字段直接移交，不跳过识别规则。
- 相同 129 页公开规则场景：首次 38.201→29.772 s，复用 38.202→31.069 s，总耗时下降 22.1%/18.7%；原生分析分别 26.949→18.996 s、26.891→19.905 s。每版本两次顺序调用，不是多轮交替基准；尚未追平此前 Python，不外推扫描或其它文档，也未重新测内存。
- 16/65/129 页与密集长表格完整快照严格一致；默认模型扫描/论文/印章 6 组公开对照、271 组固定 Python 参考、数字边界测试和 5 项资源策略测试通过。
- 新 wheel build/value-transfer-20261101 已安装项目 .venv，安装指纹核对通过。记录 docs/plans/2026-11-01-native-value-transfer.md，收据 build/value-transfer-delivery-receipt-20261101.json。未跑无关 Office、全 workspace、全 PDF 或 CUDA；剩余深层复制、几何上下文与渲染仍需定位。

## 2026-11-01 原生 PDF 页面所有权、脚本上下文与 RGB 渲染

- 基于上一安装版再次采样，去除图形/Form/目录/噪声/图表阶段整页树的多次往返复制，表格事务直接更新自身 source，生产上下标路径借用完整几何映射。保留诊断接口和有限预算校验、全部候选/恢复规则、数字精度。
- RGB 直接整理 PDFium 已渲染的 RGB 行缓冲，避免 RGB→RGBA→RGB；30 组四方向、奇偶宽、长边上限和真实 PDF 的 RGB/RGBA/模型渲染尺寸、像素与 PNG 字节完全一致，另通过 288 组行 padding 组合与错误边界。
- 构建/回归停止后顺序比较相同 129 页原生规则输入：上版首次/复用 31.006/31.063 秒，新版 18.895/18.787 秒，耗时下降 39.1%/39.5%；同期 Python 11.894/10.318 秒。热原生分析 19.898→13.660 秒，渲染 7.546→1.513 秒，组装约 2.72 秒。仍慢于 Python，未完成追平，不外推扫描件。首轮与编译重叠的计时已排除。
- 16/65/129 页及密集长表格完整快照严格一致；默认模型扫描/论文/印章 6 组公开对照通过。4 项借用入口检查、32 组脚本角色/错误对照、271 组固定 Python 参考和 5 项安装资源策略检查通过。
- build/borrowed-native-r2-20261101 wheel 已安装并保留于项目 .venv；wheel/安装/回归指纹匹配。记录 docs/plans/2026-11-01-native-borrowed-context-and-rgb.md；收据 build/borrowed-native-delivery-receipt-20261101.json。未跑无关 Office、全 workspace、全 PDF、CUDA 或新内存基准。

## 2026-11-01 原生 PDF 参考行、最终整理与资产哈希

- 再次采样上轮最终包，优化表格画线过滤中的整页复制、角色/脚注参考行 options、几何重拆输出，以及连续标题整理私有请求和中间结果。保留所有识别/恢复步骤、别名覆盖、数字精度、公开适配与有限预算行为。
- 页面 MD5 仅在首个资产需要命名时计算，同页复用；PNG 读取及错误校验仍执行，不按特定文件绕过算法。36 组 L/LA/RGB/RGBA 的裁图、印章、表内填图、原图与坏 PNG 对照保持命名、顺序、像素/字节及错误一致。
- 最终 129 页公开原生规则的顺序计时：上版首次/复用 19.380/18.884 秒，新版 15.149/15.014 秒，耗时下降 21.8%/20.5%；同期 Python 12.060/10.375 秒。热原生分析 13.710→11.849 秒，组装 2.724→0.725 秒，渲染约 1.52 秒。仍未追平 Python；不是多轮交替基准，不外推扫描件或新内存收益。
- 最终版 16/65/129 页及密集长表格完整快照严格一致，默认模型扫描/论文/印章 6 组公开对照和 5 项资源策略通过。235 组参考行场景在有限预算/生产作用域共 470 组输出、错误和 Work 计数对照通过，另通过 271 组原固定参考和 3 组最终页面参考的多入口检查。
- build/native-reference-r2-20261101 wheel 已安装项目 .venv，指纹与最终回归匹配。记录 docs/plans/2026-11-01-native-reference-and-hash.md；收据 build/native-reference-delivery-receipt-20261101.json。未跑全 workspace、无关 Office、全部 PDF、CUDA 或新内存基准。

## Python 性能后续待办（2026-11-01 用户明确要求）

- Rust 提速继续优先，保留 Python 为当前稳定对照；后续也优化 Python，逐项检查此前模型/运行配置等共享收益是否已启用。不能把 Rust 特有的深复制和 RGB 往返转换问题归给 Python。
- 首项已确认候选为整页 MD5 按需计算；仍需 Python 独立效果、命名、速度/内存验证，收益不足不合入。完整记录见 [Python 性能后续工作](python-performance-followup.md)。

## 2026-11-01 原生 PDF 只读候选页和文字行准备

- 基于上版最终包重新采样，只读候选页改为借用原行（Cow），需要跨候选缓存时显式转为自有数据；角度准备/页面策略/浅色表头接收源页与小选项，避免完整 source 包装。文字行准备按原 defaults/marker/未知字段顺序接收所有权，原生事实收集移交已有数组，保留算法、精度和预算。
- 顺序同条件 129 页公开规则场景：上版首次/复用 14.676/14.321 秒，新版 12.961/13.153 秒，下降 11.7%/8.2%；同期 Python 11.993/10.173 秒。热原生分析 11.240→9.971 秒；仍未追平，不外推全部 PDF、模型扫描或内存收益。
- 4 份真实表格候选参考、12 类缓存预算对照、15 组来源行参考在两类作用域、固定最终页参考通过；16/65/129 页及密集表格完整快照、6 组默认模型公开对照、36 组资产/错误和 5 项资源策略检查通过。
- build/native-page-borrow-20261101 wheel 已安装并保留 .venv，安装/回归指纹匹配；记录 docs/plans/2026-11-01-native-borrowed-table-pages.md，收据 build/native-page-borrow-delivery-receipt-20261101.json。Python 后续共享模型/按需 MD5/既有低收益缓存的记录已保存，本轮未改 Python 生产解析代码。未跑无关 Office、全 workspace、全 PDF、CUDA 或新内存基准。

## 2026-11-01 原生 PDF 图内文字与分式恢复移交

- 重采上版最终安装包后，改造图内文字拆行/补认领及分式恢复的行流转。未改行按原始身份移交，实际恢复才生成新行；正式链只传递实际使用的来源号/字宽摘要，公开诊断仍返回完整来源行。保留所有算法步骤、归属/排除/顺序和有限预算；非对象输入保持错误而非 panic。
- 新增 27 组原 Python 分式参考（8 个恢复正例）和 29 组图内文字拆行/补认领参考，双作用域核对输出/更新/身份/Work，加上既有引用共 6 项受影响 Rust 检查通过。最终 16/65/129 页、密集表格完整快照、默认模型扫描/论文/印章 6 组公开对照、36 组资产/错误和 5 项资源策略通过。
- 同条件顺序 129 页计时：上版首次/复用 12.959/12.939 秒，新版 11.865/11.981 秒，下降 8.4%/7.4%；同期 Python 11.798/10.325 秒。热原生分析 9.772→8.831 秒。首次接近 Python，热调用仍慢约 16%，没有追平，不外推扫描、全部 PDF 或内存。
- build/native-restore-owned-r2-20261101 wheel 已安装保留项目 .venv，指纹与公开回归匹配。详见 docs/plans/2026-11-01-native-restoration-owned.md，收据 build/native-restore-delivery-receipt-20261101.json。未跑无关 Office、全 workspace、全 PDF、CUDA 或新内存基准。

## 2026-11-01 原生表格事务与补充候选页面借用

- 表格事务移交自有文字行、候选和结果，补充候选检测器及其缓存借用不可变页面；数值候选、矢量公式接收来源与小选项，避免整页 JSON 包装。保持候选/保护/归属/修改顺序、数值精度和预算路径。
- 新增 13 组原 Python 表格事务及 3 组矢量公式参考；连同已有真实候选、有限预算和自有行检查共 10 项受影响 Rust 测试通过。16/65/129 页与密集长表格完整快照严格一致，默认扫描/论文/印章 6 组公开对照通过。22 项历史测试缺少 build/migration 语料，未伪称通过。
- 独立顺序计时、各一次首次及两次复用：复用中位数上版 12.109→新版 10.922 秒，耗时下降 9.8%；同期 Python 10.564 秒，仍慢约 3.4%。这是同一 129 页原生规则输入，不能外推所有 PDF、扫描或内存。
- build/native-table-borrow-20261101 wheel 已安装并保留 .venv，指纹与公开回归及计时一致。详见 docs/plans/2026-11-01-native-table-transaction-borrow.md、build/native-table-borrow-delivery-receipt-20261101.json。未跑全 workspace、全 PDF、CUDA 或新增内存基准。

## 2026-11-01 原生文字收尾、阅读列与脚注借用

- 文字簇和脚注列推断直接借用原行/几何，避免把完整字符树装入临时 JSON；脚注行按 Cow 借用，实际合并才复制；角色参考保留紧凑的初始判断字段及动态别名，文字簇规范化直接移交字段。规则、身份和 Work 路径不变。
- 新增 3 组原 Python 别名/分离对象/稳定排序参考；阅读列借用/JSON、角色完整/紧凑快照、脚注引用及完整收尾/表格参考合计 10 项受影响测试通过。16/65/129 页与密集表格完整快照严格一致，默认模型扫描/论文/印章 6 组公开对照通过。
- 同一 129 页规则 PDF 独立顺序计时，复用中位数上版 11.502→新版 10.072 秒，耗时下降 12.4%；同期 Python 10.230 秒。本次略快约 1.5%，只说明该场景基本追平，不能证明稳定领先或外推所有 PDF。
- build/native-finish-borrow-20261101 wheel 已安装并保留 .venv，指纹与公开回归、计时匹配。详见 docs/plans/2026-11-01-native-finish-borrow.md、build/native-finish-borrow-delivery-receipt-20261101.json。未跑全 workspace、全 PDF、CUDA 或新内存基准。

## 2026-11-01 原生 PDF 收集字段借用和数组移交

- 字符建行、间距准备、样式/链接证据与字重分组接收借用字段，不再构造完整字符树请求；已有字符、行、Form、路径及坐标数组直接移交。保留字重条件渲染/像素判断、数值和规则；显式有限树预算的建行请求继续原完整验证。
- 新增自包含 390 组原 Python 参考：16 字符建行、48 间距、218 样式/链接、108 像素字重（9 个加粗正例），双作用域核对输出/Work；异常输入及既有完整收尾合计 8 项受影响测试通过。16/65/129 页与密集表格完整快照严格一致，默认扫描/论文/印章 6 组公开对照通过。
- 同条件 129 页规则场景，复用中位数上版 10.009→新版 9.006 秒，耗时下降 10.0%；同期 Python 10.592 秒，本轮 Rust 耗时少约 15.0%。原生分析减少约 1 秒，渲染/组装基本不变；不外推全部 PDF、模型扫描或内存。
- build/native-collect-borrow-20261101 wheel 已安装保留 .venv，指纹与公开回归、计时匹配。详见 docs/plans/2026-11-01-native-collection-borrow.md、build/native-collect-borrow-delivery-receipt-20261101.json。PDFium 多次开文档/页面尚未改为句柄复用；未跑全 workspace、全 PDF、CUDA 或新内存基准。
