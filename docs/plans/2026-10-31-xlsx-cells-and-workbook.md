# XLSX 工作簿、单元格与表格投影接入

本阶段把 XLSX ZIP/XML 原生读取接到前一阶段的共享电子表格投影。Rust 可以从真实工作簿读取缓存值、富文本和样式，并对不含待接资源的工作簿输出原 model-list；**尚未开放完整 XLSX 的 Python 默认入口**。

## 已接通

- 原 OPC 包解析器定位 workbook/worksheet，按原顺序读取工作表和局部关系，不解压到目录，不构造 openpyxl 对象或调用 Python。
- 共享字符串按 Content Types 定位，样式沿原 openpyxl 固定 `xl/styles.xml` 入口。保留缺 manifest 项、缺共享字符串关系时原读取行为的区别，不把“存在 XML 文件”当作等价于已声明部件。
- 单元格支持普通/共享/内联字符串、富文本、布尔、任意精度整数、浮点、错误值、ISO 日期/时间/持续时间和数值日期格式。复用图表阶段已验证的日期/持续时间规则及 1900/1904 epoch，不进行公式重算，只读取缓存值；普通单元格使用 Python 字符串化，不能套用图表中会把 `1.0` 化为 `1` 的规则。
- 富文本保留原字体标签嵌套次序、下划线/删除线空格 `&nbsp;`、上下标、换行和 XML escape 标识处理，排除 phonetic 内容；空富文本与普通空字符串保持原类型差别。
- 样式保留直接字体、RGB 颜色、实色填充和水平/垂直对齐；theme/indexed 颜色不误当 RGB，渐变填充不伪造成纯色背景。新生成的空格沿工作簿默认字体，显式单元格则沿其 cellXf。
- 合并格覆盖非锚点旧值和样式；单格链接若落在合并格内，回到原锚点；区域链接跳过合并隐藏格。原无值链接单元格可用目标/location 填值，链接显示使用既有安全策略。
- 读取后忽略声明 dimension 的空洞膨胀，以原实际数据/合并格边界发现表格。保留可见/隐藏表开关、空表页面、多非空表标题、显式 gap 与 singleton 配置。
- 保留空名称默认值、重复工作表名递增和非法名称错误。未额外增加文件大小、单元格数量或字符配额。

## 防止提前丢失资源

内部 `Workbook::convert_cells()` 是当前单元格阶段入口：遇到绘图、VML、OLE、OMML 或 cell-image 资源时明确返回未接入，而不是输出缺图片/公式的“成功结果”。单元格读取器仍保留 worksheet part 和局部关系供后续接线。

本阶段的参考生成器调用 openpyxl/原 Python 作为基线，**不是运行时方案**。完整 XLSX 尚需图片/cell-image、图表、OMML/MTEF/OLE、包兼容重试与公共接口验收。

## 验证

- `export_xlsx_cells_reference.py` 冻结 **50 份工作簿**：46 份读取成功、4 份保留原损坏/非法输入错误；成功样例含 **70 个工作表**。
- 对比每格有值/语义状态、完整 Cell IR、默认样式、选定 gap、表格 IR 和最终 HTML；覆盖已有真实 XLSX、稀疏/膨胀 dimension、共享/内联富文本、公式缓存/无缓存、日期格式和 ISO、数值边界、合并格链接、链接填空、默认字体及重复/非法工作表名。
- 对没有绘图/OLE 的工作簿另做 **123 组完整文件到 pages 对比**：默认、包含隐藏表、显式 gap=0 且关闭 singleton 降级三种配置，比较原 `XlsxConverter.convert()` 输出。
- 运行 XLSX 单元格、共享投影、共享图表三个受影响测试；最终追加边界后只重跑 XLSX 测试。未全量回归元文件/PDF/CUDA，未做 XLSX 性能基准。
- 原始参考 SHA256 和测试收据：`build/xlsx-cell-receipt.json`；测试日志：`build/xlsx-cell-tests.log`、`build/xlsx-cell-final-tests.log`。

未增加外部依赖。openpyxl/DocVortex 行为来源沿现有许可证保留，来源哈希记在 `packaging/office-native-provenance.json`。当前 `.venv` 继续使用已交付 DOCX/PDF 包，完整 XLSX 接入后再构建安装。
