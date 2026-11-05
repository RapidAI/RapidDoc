# XLS BIFF 读取、数字显示与原生表格投影

## 本阶段完成

`rust/rapiddoc-office/src/xls/` 已加入原生 BIFF5/BIFF8 工作簿读取：复用已有 OLE/非规范目录兼容，读取 Workbook/Book 流，解析 BOF/目录/SST/FONT/XF/FORMAT/CODEPAGE/DATEMODE/WINDOW1/SUPBOOK/EXTERNSHEET，以及单元格、缓存公式、合并和链接。

- 保留截断记录已完成内容、跨 CONTINUE 压缩/Unicode 模式切换、SST 字符与富文本的不同跨段规则、UTF-16 字体偏移到 Unicode 字符区间的转换、RSTRING 与 TXO 格式区间。
- 保留 NUMBER/RK/MULRK、BOOLERR 和缓存 FORMULA→STRING、错误字面量、无缓存/无效项及旧 DISPIMG 缓存回退；不计算公式。
- 数字显示独立移植自 `xls/number_format.py`，包括 General 15 位有效数、条件/颜色/地区/文本 section、half-up 十进制、科学计数、工程指数、固定/近似分数、分组/百分号/缩放及两种日期系统。不套用 XLSX 的普通值字符串化。
- 用 BigInt 实现原 Decimal/Fraction 所需的整数运算，避免加入运行时 Python；Decimal 的 28 位上下文和指数异常回退沿原行为。工作簿记录/合并/物化预算沿原 Python 定义，没有另加字符数量报错。
- XLS 原有 openpyxl 适配对普通字符串的 32767 字符裁剪语义保留；这是原显示适配规则，不是新增 Rust 解析输入限额，富文本另循原规则。
- 安全 HLink、内部 fragment、URL/字符串目标和本地 File Moniker 排除沿已有 Office 链接策略；隐藏行列仍参与数据重建，隐藏工作表默认不输出，可见空页保留。
- 坏 BoundSheet offset、重复 offset、目录缺失与无法恢复时的空页保留。记录 payload 和 CONTINUE 借用同一 Arc 流，资源记录保存偏移，避免复制整个剩余工作簿；候选 BOF 扫描以一次尾部非零索引维持原零填充停止规则。
- 单元格直接投影到共用 `spreadsheet::Sheet`，复用现有 gap、合并、HTML 与 singleton 规则，不构造 openpyxl 对象。保留原工作表重名后的源名称查询和分页次序。
- 原生 BRAI PtgRef/PtgArea/3D 引用选择已移植，独立 chart sheet 可按原始目录顺序输出源单元格选区表格；不把图表当图片猜测。

## 旧代码页

Python codec 与 WHATWG `encoding_rs` 在未定义单字节、Shift-JIS/GBK/Big5 和 EUC-KR 组合/非法尾部处理上并不完全相同。本阶段不直接替换后声称等价，而是由 `export_xls_codec_tables.py` 从当前 CPython 编解码器生成固定原生表：

- CP874、Shift-JIS、GBK、Big5、CP1250～1258：字节状态转移与替换文本；未知 CODEPAGE 按原版回落 CP1252。
- EUC-KR 保留全部 11172 种八字节韩文组合拼写，包括同时有较短标准编码的音节；短尾部/非法组合按原解码消费范围处理。
- UTF-16LE 保留孤立 surrogate 和奇数字节尾部的替换语义。

生成器只在开发阶段使用。Rust 运行时直接解码固定 gzip 表，不启动或链接 Python。来源、Python 版本、表 SHA256 与 PSF 许可沿现有 Office 来源记录保存。

## 验证

- `xls_cells.json.gz`：**101 份工作簿、105 个工作表检查点**；**89 组完整文件到分页**与原 Python 一致；保留 **11 组原异常**，另 1 份资源文档只验读取检查点并明确拒绝不完整输出。
- 对比每格原始富文本区间、链接、合并、可见/恢复状态、活动页、独立图表元数据，及中立 Cell IR、完整表格 HTML/分页。覆盖真实 `.xls`、合成 BIFF5/8、重复/非法标题、截断、坏目录、数字/日期/公式缓存、RSTRING/SST/CONTINUE、链接、RK/MULRK、合并格边界与原合并预算。
- 基础参考 `xls_primitives.json.gz`：数字显示/文本格式、跨段字符串、TXO、记录边界、旧代码页和 BRAI 引用。精确数量以收据 `build/xls-native-cells-receipt.json` 为准。
- 对 **16 组代码页/默认映射的全部 65536 种双字节输入**逐一核对输出 UTF-8 的长度与聚合 SHA256，共 1048576 组；另核对所有韩文音节和八字节组合、随机坏字节和截断尾部。
- 开发仅运行 XLS 基础/单元格、共用表格投影、XLSX 单元格、共用图表与 OLE 基础测试。日志 `build/xls-cells-affected.log`；数字条件 Unicode 边界补充后仅复查基础测试 `build/xls-primitives-final-tests.log`。未运行全量 PDF/CUDA 或 XLS 性能基准。

原 Python 资源绑定会将文本框/控件文字追加到单元格，所以文件读取检查点明确在 `_bind_objects` 之前对照；真实最终资源后文字也单独冻结，留给下一阶段核对。没有把资源之前的检查点当成完整 XLS 效果验收。

## 尚未完成

1. OfficeArt BStore/形状与 OBJ/TXO 绑定，图片、文本框/控件、嵌入式图表及其预览、OLE Equation Native/图片公式优先级。
2. 完整 XLS 结果统一导出与公开 `RapidDoc`/Rust 路由、构建安装、最终效果/速度/内存验收。

`Workbook::cell_pages()` 是本阶段诊断入口：资源未绑定的工作簿返回 Unsupported，避免图片、公式或文本框被静默遗漏。公开 Python 仍为已交付 PDF/DOCX/XLSX，**本阶段未替换 `.venv`**。后续顺序仍为 XLS → PPTX → PPT → DOC。
