# DocVortex 0.6.8 与 Office 迁移顺序分析

状态：分析完成，建议待实施；本轮没有升级生产代码或依赖。范围仅 PDF、DOC、DOCX、XLS、XLSX、PPT、PPTX。

## 决策建议

先锁定 0.6.8 的目标与参考，对当前 Python Office 选择性同步有效修复并验证，再按该基线移植到共享 Rust。新版已经提供且适合直接使用的纯 Rust 内核直接复用，不先改写成 Python 再重写一次 Rust。已经验收的 PDF 保持冻结，上游 PDF 新修复单独审查和回归，不随 Office 更新覆盖。

不推荐完整照搬新版 DocVortex Python 包，也不推荐先按旧版完整移植 Office 后再追赶 0.6.8：前者会覆盖本地 PDF/Office 补丁并引入新的后端依赖，后者会让重复图片、容器读取和区域发现等旧问题进入新 Rust 实现，再付一次修复与验收成本。

## 实际版本与来源

- 已检查本机 `D:/CodeProjects/doc/DocVortex` 的提交对象，不依赖其未提交的 demo 和样本改动。
- `v0.5.16` 实际提交为 `fb59f89dcb4f76dd1adbb3c84bd8dfd51a982c16`；`v0.6.8` 实际提交为 `2de5f42b1884495313d9df63868e6a2f3d85fea0`。两者之间 50 个提交、368 个变更文件；包含大量测试、语料和不需要的格式，不能当作七种格式的必要移植工作量。
- RapidDoc Python 裁剪源的 `UPSTREAM.md` 标注 0.5.16，记录的实际源提交是 `f5da4e69022f5b0a935fb64579d7f0f7ec7d08b6`，另有本地修改。它不是纯净的 v0.5.16 tag。
- Rust `vendor/docvortex-core/UPSTREAM.json` 另固定在 `2c1ec66ee52bb37135599af92aa175bd77fc04d7`。Python 参考层、上游 Rust 数值核和 RapidDoc 自有规则是不同来源边界，不能靠统一修改版本号表示全部升级。
- Python `UPSTREAM.md` 引用的 `docs/migrations/docvortex-source-manifest.json` 在当前工作区未找到；实际升级前需要补齐逐文件来源和本地补丁台账，不能依靠旧说明覆盖目录。

## 与七种格式有关的变化

| 范围 | 主要变化 | 推荐处理 |
| --- | --- | --- |
| DOC、PPT、共享旧 Office 图片 | #36：单张超限图片使用专属异常，在图片入口跳过，文档总预算仍报错；有界解压保留 | 对照本地已有修复，保留更完整的错误边界及现有测试，不重复套补丁 |
| XLSX | #37：不再把 openpyxl 的写出 `Image.path` 当作源 ZIP part，改读原始流并恢复游标；媒体类型来自实际字节 | 优先同步并建立回归，不能只凭本地已有 ZIP 图片主路径判断已修好 |
| XLSX | 工作簿内样式/媒体/公式结果复用，结束或失败时清理，保持结果对象独立 | 作为新版 Office 基线的一部分验证，不引入跨文档状态污染 |
| XLS | BIFF 遍历用共享 memoryview 避免逐条复制剩余尾部；原 EOF、截断和预算语义不变 | 直接采用等价所有权/切片实现，先保留异常契约 |
| DOCX | 正文与表格共享一次文档解析，无顶层表格时跳过预解析，复用表格签名 | 同步有效优化，再迁移；保持表格对齐和缺关系处理 |
| XLS、XLSX 共享电子表格 | 区域发现、投影快照等批量原生入口；新增纯 Rust `spreadsheet_regions.rs` | 同步 Python 调用契约；适用的 Rust 核直接整合，不重写成 Python 参考核 |
| PPTX | 本次比较未见其格式专属目录直接变化；共享图片、数值、文本后处理仍可能影响它 | 纳入真实样本回归，不能按“专属文件没变”跳过 |
| PDF | 上下标冲突、连字空格、跨页列网格、旋转表头、稀疏层级参数表、损坏文字表格拒绝、完整私用区支持等修复 | 单列补丁清单，和本地版面/原生表格优化三方对照，不重置已完成 PDF 实现 |

OFD、ODF、RTF、HTML、EPUB 等新增格式能力不作为公开格式迁移目标；共享工具若被七种格式的嵌入对象或内部处理实际引用，则保留相应依赖能力，不按目录名机械删除。

## #36、#37 的本地验证

上游提交分别是：

- `c44336dfda41f00e7195247faee16fcb60e626fc`：`fix(office): skip oversized legacy pictures (#36)`。
- `6c1f11e61ded9cdb3718094b8ca4268def8db707`：`fix(xlsx): read original image streams to avoid duplicates (#37)`。

本地 `test_office_image_resource_limits.py` **23 项通过**，已有 `LegacyOfficeImageResourceLimitError` 和 DOC/BStore/PPT 图片边界处理。这证明本地已覆盖 #36 类型的资源错误隔离，不意味着任意真实 DOC 都已完成验收。日志：`build/docvortex-issue36-local-test.log`。

进一步逐行比较：两者主要策略相同，异常命名和日志并不构成功能优劣。本地先检查元文件声明大小再切取 payload，减少无用复制；上游在裸 DIB 补 BMP 头后再次检查单图预算，边界更完整，本地目前缺少此复检。将测试限额设为128字节、构造128字节合法DIB时，本地返回142字节BMP，上游拒绝；仅在独立进程对齐异常名称做比较，未修改生产实现。证据为 `build/docvortex-068-analysis/issue36-bmp-boundary-comparison.json`。合理后续是保留本地提前检查和已验证的捕获边界，补入上游BMP复检，而非认定本地修复整体优于上游；此合并尚未实施。

#37 在当前 Python 代码中仍复现。内存构造两张不同尺寸/颜色 PNG，放在同一个 XLSX 的 C3/C20，调用实际 `XlsxConverter._collect_sheet_images()`：

- ZIP 主来源启用：返回 3 张，第二个 anchor 多出第一张图片。
- 仅备用来源：返回 2 张，但第二张内容/尺寸错误地等于第一张。

未修改输入特征或按文件名打补丁。复现记录：`build/docvortex-068-analysis/issue37-local-repro.json`。

按记录的 Python 来源提交、当前本地文件及 0.6.8 做包前缀归一化后的 AST 三方比较：当前 Office 子树有 6 个“本地仍等于原来源、上游已变化”的文件，以及 4 个双方均改动、必须审查的文件：`errors.py`、`doc/images.py`、`legacy/officeart.py`、`ppt/parser.py`。双方都改动不等于已经证明冲突。PDF 另有 7 个双边改动文件，包含 graphics/pipeline/table_detection/table_rules 及表格恢复 text/vector/contracts；这也是不全目录覆盖的直接原因。清单：`build/docvortex-068-analysis/source-overlap.json`。

## 新版本的后端边界

0.6 系列移除了多份重复的 Python 数值参考核，必需加载原生扩展；当前 RapidDoc 裁剪源明确采用 Python 回退策略。因此不能只替换若干 Python 文件便视为升级完成，也不能恢复一整份被删 Python 计算后端来模拟升级。

建立 0.6.8 参考时可使用隔离的上游原生后端做输出核对，或为选择同步的模块接入经过验证的内核适配；生产共享 Rust 侧优先直接复用纯 Rust core。`docvortex-python` 中依赖 PyO3/解释器行为的适配不能直接当作跨语言核心。源和私有协议应一起固定，不把未来语言绑定绑到 DocVortex Python 扩展。

新版依赖、Python 最低版本、pypdfium2/ONNX Runtime 平台选择也发生变化；不在本轮分析中升级现有 `.venv`、改写 RapidDoc 包元数据或替换其 PDFium 共存机制。

上游性能报告中有 XLS/XLSX/DOCX 优化证据，但测试平台是 macOS ARM64、固定样本和固定统计口径，不外推为本机 Windows 或所有 Office 文件的提速承诺。

## 建议实施顺序

1. 固定 0.6.8 提交、七格式白名单及本地补丁清单，保留现有 Python 和 Rust PDF 的完整快照。
2. 选择性补齐 Python Office：优先 #37；审查 #36 是否已等价覆盖；同步 DOCX/XLS/XLSX 的有效改动，接齐所需内核契约。用真实与针对性夹具验证各格式输出和异常。
3. 冻结升级后的 Office 参考输出，保留原 Python 外部调用/中间结构作为兼容目标；不要求新上游识别修复仍等于旧版错误结果。
4. 对照该参考一次性完成 Rust Office 迁移，直接复用适用的新版 Rust 内核；按格式验收文本、表格、图片、公式、顺序、错误和受影响性能。
5. PDF 的上游修复独立安排，不与 Office 阶段混成一次大范围基线替换。后续追踪以精确提交和每格式补丁台账为单位，避免每次重新整包比较。
