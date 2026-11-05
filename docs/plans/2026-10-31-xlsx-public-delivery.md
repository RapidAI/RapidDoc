# XLSX 包兼容、统一输出与 Python 交付

## 实现范围

- `xlsx::convert()` 保持原先失败后才规范化的策略；共享字符串空下划线属性、空 fill、纯行范围 autoFilter 使用 Package overlay 修复，只有确实有修改才重试一次。正常路径复用已分类的 ZIP，不重新压缩包。
- 保留无可修复修改时的原错误；失败重试读遍 ZIP 验证成员，避免把损坏包误当成功。纯行筛选器按 dimension、真实单元格、A:XFD 的原优先级推导列界。
- 原生读取直接持有 worksheet part 与局部关系，不依赖 openpyxl 的工作表名称映射和剥离 OLE 副本。空工作表名与 OLE 的一个原 Python KeyError 可恢复，公式结果与同内容正常命名版本一致。这是原生读取能力，不是为某份文档写特例。
- XLSX 页结果以所有权移动方式转共享 Office 块，再走已有 middle、图片落盘、Markdown/NLP Markdown/content_list/content_list_v2；没有中间 JSON 字符串往返，也没有 Python Office 解析回调。
- 新原生绑定 `parse_office_bytes()` 自动分发 DOCX/XLSX；原 `parse_docx_bytes()` 继续保留。公开 `RapidDoc()(source)` 自动识别 XLSX，参数和输出沿已有 Python API。空 `.xlsx`、批量、bytes/流、图片 data URI、文件/自定义 writer 均沿共用入口。

示例：项目 `.venv/Scripts/python.exe demo/rust_xlsx_demo.py`。输出目录为 `demo/output/rust-xlsx/<文件名>/office`。不需要 OCR/版面模型，也无需配置 PDFium。

## 对照与验证

- 包修复参考：86 组边界，直接比较原规范化后 XML 的命名空间、属性、文本与结构；覆盖空/有名空间下划线属性、空 fill、行范围过滤器及 dimension 回退、损坏数值/日期/共享字符串、空表名、坏可选图片。
- 完整导出参考：231 组输入，200 组原 Python 成功结果完整一致；30 组保留原错误；另 1 组原 OLE 空表名故障由 Rust 恢复。比较完整 middle、四种输出和所有图片像素/SVG 内容，再建立文件名别名，不放宽文字、结构或输出顺序。
- 测试参考用 fixture + source_index 标识输入，避免不同目录下的 `xlsx_01.xlsx` 被混为同一个文件。
- 既有 DOCX 353 组完整 Office 输出也通过共享验证器；本阶段结束运行 Office crate 回归，未跑全量 PDF/CUDA。
- 参考与测试：`xlsx_compat.json.gz`、`xlsx_output.json.gz`；日志 `build/xlsx-compat-tests.log`、`build/xlsx-output-final-tests.log`、`build/xlsx-office-stage-regression.log`。

## 安装与性能验收

构建目录 `build/office-xlsx-wheels`；安装后执行 `scripts/migration/check_xlsx_install.py`，产物 `build/xlsx-installed-checks.json`。该检查禁止 Python Office 包导入和识别模型加载。

性能脚本 `scripts/migration/accept_xlsx_delivery.py` 复用已验收 DOCX 的独立进程测量器，逐个测本地 XLSX、交替 Python/Rust 顺序。默认两轮/每轮 3 次热运行，另一次采样内存；对比结果先验证完整文字/结构及图片像素。构建、输出落盘、像素核对不计入解析耗时；进程峰值包含导入基线，2ms 采样可能遗漏短峰值。具体安装与性能结论以最终收据为准。

初次端到端测量记录保留在 `build/xlsx-acceptance-20261031`，原 wheel 在 `build/office-xlsx-baseline-wheels`。7 份本地文件约快 3～10 倍，但含 1806×2500 JPEG 的工作簿为 Python 55.38ms / Rust 72.94ms，因此没有按最初数字宣称全部提速。

原生分段诊断 `examples/profile_xlsx.rs` 定位到表格长 data URI 的 HTML 清理，约 40ms 耗在正则捕获；在不改变 tag/attribute 匹配表达式的前提下，改为 `find_iter` 扫描相同匹配范围，只提取短名称、从已匹配的赋值文本读取属性值。没有跳过图片转码、降低 JPEG 质量、按文件名处理或修改文档内容。该样例清理阶段降至约 2.4ms，中间结果组装约 45ms→5ms；这只是原生分段耗时，端到端以重新安装后的基准为准。

3,574 组原捕获版 cleaner 对照通过，覆盖普通/非法/Unicode/缺失引号/长 base64 属性；其后 DOCX 353 组与 XLSX 231 组完整导出参考再次通过。日志 `build/xlsx-html-optimization-tests.log`、`build/xlsx-stage-profile-detail.log`、`build/xlsx-stage-profile-optimized.log`。优化后的正式基准独立保存在 `build/xlsx-acceptance-optimized-20261031`，不覆盖首次测量。

## 边界

未做跨平台或所有未知坏 DrawingML/图表的穷尽验证；保留既有 Office 编解码器的少见格式边界。XLS/PPTX/PPT/DOC 后续移植，增强阅读顺序、CUDA 继续按用户要求延后。

## 最终安装与端到端验收

优化版 wheel 已安装到项目原 `.venv`；389 个安装包文件逐字节匹配，公共入口、8 份本地文件、4 类 IO、batch/data URI/writer/四类导出、空 XLSX、DOCX 旧入口与禁用 Python Office/识别模型检查通过。安装 SHA256：`3c727fde07849c6135f8a9eaff8e47348bbd8a936ee0c73ee5dd8c206909afaf`。完整收据 `build/xlsx-delivery-receipt-20261031.json`。

两轮 32 个独立进程全部成功，8 份输入完整输出一致。下表为两轮 6 次热运行的中位数；峰值为整个进程的操作系统峰值，含导入与此前运行，不能把全部差距归因于解析算法。

| 输入 | Python ms | Rust ms | 提速倍数 | Python 峰值 MiB | Rust 峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| demo/xlsx/xlsx_01.xlsx | 27.34 | 8.35 | 3.27 | 310.9 | 44.0 |
| demo/xlsx/xlsx_03_chartsheet.xlsx | 8.96 | 0.92 | 9.76 | 306.7 | 39.8 |
| demo/xlsx/xlsx_04_inflated.xlsx | 27.38 | 8.22 | 3.33 | 310.8 | 43.4 |
| demo/xlsx/xlsx_05_table_with_title.xlsx | 5.71 | 0.79 | 7.26 | 306.3 | 39.3 |
| demo/xlsx/xlsx_06_edge_cases_.xlsx | 8.89 | 1.20 | 7.42 | 306.9 | 39.6 |
| demo/xlsx/xlsx_07_gap_tolerance_.xlsx | 16.17 | 1.73 | 9.35 | 307.1 | 41.3 |
| demo/xlsx/xlsx_08_one_cell_anchor.xlsx | 4.79 | 0.76 | 6.28 | 307.1 | 39.1 |
| demo/ms_office_docs/xlsx_01.xlsx | 54.02 | 32.12 | 1.68 | 328.9 | 80.9 |

最初大图样例约 72.94ms，优化后的实际端到端耗时以本表为准；文档内容、编解码质量和模型均未改变。XLSX 可直接运行示例测试，后续进入 XLS。

最终优化版安装后的 8 组 PDFium 同进程共存、`pip check` 与 Rust 格式检查也通过；未运行模型 PDF 全量回归或 CUDA。
