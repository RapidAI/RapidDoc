# DOCX 图表与嵌入工作簿

原 Python 的 OOXML 图表数据提取已接入 Rust DOCX 正文：图表输出为数据 HTML 表格，不重新绘制图表图片。继续使用原转换规则，没有按特定文件增加替换逻辑，也没有改变模型或 PDF 流程。

## 已接通

- `chart/mod.rs`：原 16 种 plot 标签、分类/日期/散点/气泡数据、兼容组合图、标题、系列名、多级分类、稀疏缓存、共享/独立 X 列及 HTML 转义。
- `chart/workbook.rs`：使用现有 ZIP/关系解析器读取嵌入 XLSX，按 `read_only=True, data_only=True` 语义读取缓存值，不计算公式。支持共享/内联字符串、布尔、错误、整数/浮点、日期与持续时间格式、ISO 日期和持续时间、1900/1904 epoch、带转义工作表名的单行/单列公式及首个非空工作表兜底。
- 保留原顺序：工作簿公式数据 → 图表缓存 → 普通工作表表格。工作簿容器不能打开时允许缓存回退；已打开工作簿读取坏单元格产生的异常继续传到 DOCX 图表边界，记录诊断并保留空图表块，不误用正常缓存掩盖错误。未读取的其他工作表或目标范围之后的坏单元格不污染此前有效查询。
- 正文按原 `_handle_drawingml` 时机先处理图表，再处理文本框、表格、图片和段落；同一图表多次放置仍多次输出，提取结果按部件路径复用。缺 ID、错误关系、无效 XML、坏嵌入工作簿与后续正文保留规则均覆盖。
- 真实 DOCX 发现的 `<br />` / `<img ... />` 空格差异已在表格 BeautifulSoup 等效序列化边界修复；原 Mammoth 直接输出序列化保持自己的格式，不做最终字符串替换。

运行阶段不依赖 openpyxl、python-docx 或 Python 回调；直接传递 Rust 类型和共享部件字节。此工作簿读取层只服务图表，不代表完整 XLSX 转换器已完成。

## 验证

- `export_office_chart_reference.py` 冻结原 Python 的 **97 组查询**，逐项比较图表完整描述、缓存 HTML、工作簿图表 HTML、普通工作簿兜底 HTML、最终优先级及异常结果；另有 **208 组日期转换**。
- `export_docx_body_reference.py` 扩展为 **272 份完整 DOCX**：271 成功、1 损坏输入；281 个逻辑页，包含 28 个图表块、157 个正文图片块。包括现有 `demo1-docx.docx`、`test.docx`、`drawingml.docx` 中的真实图表，以及坏工作簿、缓存公式和 ISO 日期接入样例。
- 测试比较完整 pages，正文图片比较解码像素/SVG 规范化内容；不降低像素容差，也不删掉输出字段来通过验证。
- 阶段 Office crate **57 组测试通过**，`cargo fmt --check` 通过。结果及固定参考 SHA256 见 `build/office-docx-charts-receipt.json`；日志为 `build/office-docx-charts-tests.log`。未跑全量 PDF/CUDA 或性能/峰值内存基准。

## 许可与后续

原行为来源包括 DocVortex `ooxml_chart.py`、openpyxl 3.1.5 的单元格/日期/格式处理；保留 `licenses/OPENPYXL.txt`。日期运算采用已发布 `chrono 0.4.45`，不引入时间区数据库或系统时钟调用；依赖声明记录于 `packaging/office-chart-dependency-notices.json`，由现有通知收集脚本的参数化入口生成。

DOCX 仍需附注/嵌入样式映射等资源、统一导出及公开 Office/Python API；少见图片编码和更广泛异常 XLSX 仍需扩大兼容验收。当前测试不证明所有客户端文档或跨平台完全等价，不宣称尚未测量的速度收益。没有重建生产 wheel，也没有替换 `.venv` 的 `rapid_doc_rust`。
