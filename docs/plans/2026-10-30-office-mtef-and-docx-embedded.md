# MTEF v3/v5、图片公式及 DOCX 资源接入

继续移植本地 Python 的 `equation/mtef.py`、`mtef_v5.py`、`image.py`、`ooxml.py` 和 DOCX 资源规则。运行链路均为 Rust；新增 Python 脚本只冻结参考或生成静态字符表，不参与发布包解析。

## 已实现

- `rust/rapiddoc-office/src/equation/mtef.rs`：版本分派、Equation Native 头、独立 CFB 公式对象、DOC ObjectPool 公式读取和 OOXML 公式缓存。保留 v3 接受尾随字节、v5 必须完整消费、空结果及损坏对象回退的差别。
- `equation/mtef/v3.rs` 和 `v5.rs`：分别读取记录格式、偏移、标尺、字号、矩阵、嵌套列表、字体/编码定义、颜色和首选项，不把 v3 tag/flags 套到 v5。v5保留 MTCode、PUA、Symbol/MTExtra、cp1252、粗斜体、文本/函数分组与 selector/variation/options 校验。
- `equation/mtef/render.rs`：共同的强类型公式节点和 LaTeX 输出，保留原有分式、根式、脚标、定界符、大运算、上下限、矩阵、纵列、箭头、重音、划线等规则；不猜测原 Python 不支持的模板或 PUA。只渲染一次，避免原 reader 可见性检查和最终输出的重复渲染。
- `equation/image_mtef.rs`：WMF pre-6、AppsMFCC 单段/分段、GIF MathType/001 解包；保留 baseline 忽略、签名/版本、拼段重启、冲突候选、完整 EOF/trailer、填充尾字节及失败告警规则。普通 GIF 帧只遍历 sub-block 边界，不收集压缩像素；WMF 单段候选借用源字节，仅拼段时申请缓冲区。
- `docx/embedded.rs`：按当前 part 的关系定位 OLE/图片公式，保留 ProgID、Embed/Link、图标、外部/缺失关系回退；连接公式载体优先级和预览图判断，并接入已完成的表格公式注入路径。
- 表格图片输出和公式探测共用一个 ZIP 读取器与 Arc 载荷缓存，不重复构建包对象。公式结果按关系缓存，内容摘要仍负责不同关系间的原去重计费；没有文档 JSON 中转或 PNG 编解码。
- 深度64、记录数、矩阵4096格、资产/候选128MiB等来自现有 Python Office 限制，没有新增 PDF 字符或公式工作量预算。保留资源错误与普通损坏回退的分类。

## 验证

`cargo test --locked -p rapiddoc-office`：25组测试通过；开发阶段仅跑受影响测试，阶段完成时统一跑这个 crate，没有 PDF 全量或 CUDA 回归。

- 2,952组 MTEF/Native/OLE/ObjectPool参考：来自现有构造器及原解码器，覆盖 v3/v5模板、变体、字符编码、修饰、缺失槽位、截断、尾随字节、深度边界和数字别名。包含63个构造的独立 OLE 公式对象；当前既有验收文档未提供真实 Equation Native/OLE 公式流，不将构造语料称作真实 MathType OLE 验收。
- 254组图片公式参考：202组解出公式，其余检查普通图片、baseline、损坏/冲突/不支持以及缓存、告警去重和候选计费。包含19张现有真实文档图片，这19张未解出 MTEF，验证的是普通图片结构和回退；成功 MTEF 案例来自确定性构造器。
- 11份 DOCX 包、16个表格、10次预览图查询：覆盖 v3/v5 OLE、GIF公式、图标、链接、损坏对象和原四份 DOCX 文档。比较完整公式 token、预览排除、表格 DOM，以及累计 OLE/图片候选计费。
- 既有 OMML、Equation XML、格式/编号/字段、表格结构/符号/匹配和 OLE/OPC 测试保留。公式 HTML 边界转义仍按上一阶段明确记录的修复比较，未改变 LaTeX。
- Rust格式、静态数据重生成和新参考脚本语法检查通过。没有整文档 Office 速度/峰值内存结论，未构建或安装 wheel。

持久参考：`tests/fixtures/office_rust_reference_068/` 下的 `mtef.json.gz`、`image_mtef.json.gz`、`docx_embedded.json.gz` 及 `mtef_packages/` 构造输入。生成器位于 `scripts/migration/`，来源/SHA与许可在 `packaging/office-native-provenance.json`。

## 尚未完成

- **图片内公式解码不等于矢量图像渲染。** 普通 WMF/EMF 栅格化仍待移植；表格的前置 Mammoth 图片路径若需要此能力，仍会明确未接入，不提前删除预览图来绕过原注入保护和回退。
- DOCX 完整正文、分页/目录/页眉页脚、图表、脚注/嵌入 style-map、资产与最终 schema/Markdown，以及完整包容错。
- 公开 Python Office API和原 `.venv` 更新；之后进行完整文档效果、时间和峰值内存验收。当前包内入口仅用于 Rust 组件接续，不代表完整 DOCX 已可交付。

上一阶段：[OMML 与含公式表格](2026-10-30-office-omml-and-docx-formulas.md)。本轮未修改 Python 生产解析器、模型或 `.venv` 安装包。

后续更新：普通WMF/EMF的文件/几何/GDI回放层见 [矢量回放记录](2026-10-30-metafile-geometry-replay.md)。完整字体/位图/绘制后端仍未完成，不能把本次组件进展视为矢量图像已经可输出。
