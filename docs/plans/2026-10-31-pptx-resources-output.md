# PPTX 资源、公式回退及完整分页导出

本阶段完成 Rust 内部 PPTX 从包到 pages，再到 Office 中间结构、四种导出和图片资产的主链。公开 Python 路由尚未开启，`.venv` 不变；后续仍需包兼容重试及安装/性能验收。

## 实现

- `pptx/resources.rs`：逐形状编排表格、图表、图片与公式，保持 OMML → OLE/MTEF → 公式预览图片的优先级和原先的提前返回规则。普通图片仍执行小图/背景图过滤；占位图片保持不同处理。SVG 失败后寻找伴随位图，替代文本作为图片内容保留。资源按包内路径共享字节，复用已验证的 Office 图片、图表和公式引擎。
- 表格保持首行 th、rowspan/colspan、逐段 br、软换行、样式、链接和 eq 标签规则。覆盖区域用矩形记录检查实际单元格，避免原 Python 对超出表格的合并跨度逐格分配；输出和错误不以特定文件判断。
- `pptx/math_fallback.rs`：移植项目 `pypptx-with-oxml 1.0.3` 的第二套公式转换，保留其与主 OMML 引擎不同的默认值、函数/括号空格、上下标、根式、矩阵、n 元运算、重音及异常规则。主 OMML 失败后才调用，不修改 DOCX 的 OMML 行为。显式工作栈按原方法实际访问的子节点求值，忽略的坏节点不提前报错。
- 冻结 233 项字符映射、28 项元素映射和 Python 字母判定区间，避免 Rust Unicode Alphabetic 与 Python isalpha 的差异改变函数名输出。随附原 MIT 许可证 `licenses/PYPPTX-WITH-OXML.txt`，生成数据保存来源 SHA256。
- `pptx/body.rs`：连接完整逐页生命周期。只对有输出块的形状执行 XYCut++；多块形状保持内部顺序；无坐标内容稳定置尾。备注保留 XML/组合形状顺序，排除幻灯片图像、页号、日期和页脚占位符，并支持备注内公式。
- 首个有可见内容的幻灯片状态在追加备注前确定；保留空白页及备注独占页。连接标题提升和副标题规则。
- Office 类型增加 `PageFootnote`，统一输出按原 Python 放入 discarded_blocks；原渲染器已有该类型支持。
- 上阶段公式回退的 Unsupported/None 检查点已移除：对应真实数学样例现在输出完整公式，不再豁免比较。

## 验证

生成脚本及固定参考：

- `scripts/migration/export_pptx_context_reference.py` → `pptx_context.json.gz`：275 份包级上下文/逐形状资源参考，240 组几何顺序、2,077 字体、800 标题查询。
- `scripts/migration/export_pptx_math_reference.py` → `pptx_math.json.gz`：1,151 组第二套公式参考，含 Unicode、空/损坏节点、隐藏上下限、实际访问子树及默认值。
- `scripts/migration/export_pptx_output_reference.py` → `pptx_output.json.gz`：275 份完整 convert 对照；274 成功的原 pages、中间结构、四种导出和资产内容一致，1 份无效单元格跨度保留原异常。图片按类型、尺寸及完整像素比较，资产名称差异按已验证内容映射。

覆盖本地 PPTX，分组/布局/母版/AlternateContent，表格合并、样式/链接/换行，8 类图表，9 组图片模式/编码，原生 OLE 公式/坏对象/图标和非公式对象，备注占位符、备注独占首张和尾部空白页，混合嵌套列表及任意精度编号。

受影响测试：`pptx_context`（4 项）、`pptx_math`（1 项）、`pptx_output`（1 项）通过。共享 `office_output`（1 项）通过；最终 Unicode 判定补充后仅复查公式及完整 PPTX 输出，两项通过。没有全量构建/Office 阶段回归、模型 PDF 或 CUDA 测试。

参考生成曾误设巨大合并跨度，触发原 Python 的占格循环；该次生成已中止，最终参考改为有限的表外跨度。最终验证没有使用该无界任务的结果。

## 仍待

1. 原始 Strict、坏媒体、缺命名空间/内容类型及 contentPart 的包规范化与失败重试。当前 275 份中有 3 份使用原 Python 归一化后的输入，不能算作原始包兼容已完成。
2. 更广泛的 SVG/伴随位图、损坏资源及备注异常边界参考；共享图像引擎已复用，但本阶段没有覆盖所有 PPTX 绑定变体。
3. 接公开 `RapidDoc`、构建/安装 wheel、安装输入与 writer 对照，再做统一 Office 阶段回归及速度/峰值内存验收。

当前完整原生内部入口：`rapiddoc_office::pptx::body::convert(&mut Package)`。不应在原始包兼容和公开验收前宣称 PPTX 已正式交付。
