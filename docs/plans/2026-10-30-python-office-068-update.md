# Python Office 选择性同步 DocVortex 0.6.8

状态：Python 本步已完成，供用户测试；Rust Office 尚未集成。公开目标范围为 PDF、DOC、DOCX、XLS、XLSX、PPT、PPTX，不为其他格式新增入口。

## 更新内容

精确目标提交：DocVortex `2de5f42b1884495313d9df63868e6a2f3d85fea0`（v0.6.8）。同步前先验证四个文件与记录的原来源 AST 一致，防止覆盖本地修改，再同步以下上游实现：

- `office/xlsx/xlsx_converter.py`：修复 #37，不信任 openpyxl 写出用的 `Image.path`，有界读取实际 `ref` 并恢复游标，根据实际图片字节判断类型；保留资源错误传播。同步工作簿内样式及成功媒体/公式结果缓存，返回独立样式字典，成功/失败退出均清理，失败或矢量降级仍保留逐次诊断。
- `office/docx/docx_converter.py`、`office/docx/tables.py`：正文与表格复用本次文档解析，没有顶层表格时跳过额外预解析，复用候选签名；独立调用和异常路径保持契约。
- `office/xls/records.py`：使用共享 memoryview 检查剩余 BIFF 数据，避免逐记录复制完整尾部；保持截断、EOF 和预算规则。

本地 #36 的专属图片异常以及 DOC/BStore/PPT 捕获边界保留。PDF、共享 Python 数值核、Rust 源码、模型和运行库依赖版本保持原状态，没有用整份上游目录覆盖。来源清单为 `rapid_doc/_vendor/docvortex_core/OFFICE_UPSTREAM_068.json`，包含每个同步文件的上游/本地 SHA；该清单已加入 Python 包数据。

电子表格原生稀疏快照/三档候选批处理依赖新版 Rust 内核，此轮保留原 Python 参考算法；后续 Rust Office 阶段直接复用/适配该内核，不先重新实现一份 Python 算法，也不新增外部 DocVortex 原生扩展依赖。因此这是选定 Office 模块的同步，不将整个裁剪包虚标为完整 0.6.8。

## 验证

- 导入上游 0.6.8 的图片来源、XLSX 缓存、DOCX 复用、XLS 记录视图测试，保留原断言；只调整包命名空间和公开导出调用，后者使用实际 RapidDoc API。测试及辅助夹具来源 SHA 见 `tests/native_office/upstream_068_tests.json`。
- 修改前 #37 原始流测试明确失败；修改后图片主来源/备用来源均返回两张正确位置和尺寸的图片，公开 RapidDoc PNG/BMP 图片导出也通过。
- 新版定向测试与现有 #36 测试共 80 项通过；随后整个 `tests/native_office` 共 **166 项通过**，覆盖六种 Office 格式的实际转换及错误行为。
- 同步前/后重新运行登记的 **20 份 Office 文档**，六种格式均覆盖。Markdown、内容 JSON、中间 JSON、资产名称/字节/Base64 全部严格一致，不忽略空白、字段或顺序。新增 #37 用例独立验证修复后的正确图片数，未将旧错误结果写回基线。
- PDF、共享核和本地 #36 代码 SHA 核对一致；`pip check` 通过。现有 `.venv` 从项目 `rapid_doc/__init__.py` 导入，现有 Python 调用方式不变，不需重编译 Rust 或切换解释器。

证据：

- `build/python-office-068-before-20261030/`、`build/python-office-068-after-20261030/`：20 份完整快照及逐项比较。
- `build/python-office-068-source-before-20261030/`：升级前裁剪源码及 SHA。
- `build/python-office-068-targeted.log`、`build/python-office-068-regression.log`：80 项/166 项测试日志。
- `build/docvortex-068-analysis/issue37-local-repro.json`、`issue37-after.json`：修复前后的最小复现。

此轮不进行新性能承诺，不重复全部 PDF 模型推理或 CUDA。结构源码 SHA 保持一致，不代表其他平台或所有自定义文档组合已经穷尽验证。原 Rust PDF 包继续保持已交付状态；测试本步请使用 `from rapid_doc import RapidDoc`，并用实际 DOC/DOCX/XLS/XLSX/PPT/PPTX 文件调用。


## 后续 #36 决定（2026-10-30）

用户确认后，Office `errors.py`、`doc/images.py`、`legacy/officeart.py`、`ppt/parser.py`
已直接采用上游 v0.6.8 实现，取代本地同类分支；本地回归断言保留，并补充 DIB 加 BMP
文件头后超限的测试。单张图片拒绝与整文档资源错误的边界不变，异常名对齐上游。
本段取代前文“本地 #36 保持不动”的阶段记录。24 项 #36 测试通过，20 份 Office
文档与第一步前完整基线一致。PDF 后续行为同步另见同日 PDF 补丁交付文档。
