# PPTX 上下文、文字与阅读顺序

本阶段按项目当前 Python PPTX 转换器移植，没有使用新的简化提取算法；尚未开启公开 PPTX 路由。项目 `.venv` 继续使用已交付的 PDF/DOCX/XLSX/XLS wheel。

## 已实现

- `rust/rapiddoc-office/src/pptx/context.rs`：按 presentation 关系顺序读页，定位 layout/master/notes；占位符位置按 idx 继承布局，再按原占位符类型映射继承母版。字体继承额外保留类型回退，与尺寸查找区分。
- `geometry.rs`：嵌套组合形状缩放/平移、缺失及退化变换回退、普通及占位图片区分、原始文本检测、小图/背景图过滤。沿用 Python 对旋转、翻转及 EMU 坐标的处理，没有另改包围盒规则。保留项目 `pypptx-with-oxml` 对 AlternateContent 的首个 Choice/首个形状选择行为。
- `reading_order.rs`：现有 XYCut++，PPTX 使用 beta=2.0、density=0.9；跨栏回插、窄元素重试、无效坐标稳定置尾。显式任务栈实现分割及合并，没有添加形状数、字符数或递归深度预算。
- `styles.rs`：段落、文本框、布局、母版的字体继承；字号、粗体/斜体/下划线/删除线及段落样式轮廓；列表标记及显式重启来源。保留母版标记覆盖布局标记的既有规则，编号起点和段落层级使用任意精度整数。字号按原 Python 浮点十进制舍入核对。
- `text.rs`：普通 run、字段、软换行、主 OMML 路径、样式及安全链接，复用原生 Office Span；按 shape 生命周期编排列表、缺级嵌套、连续列表首层归零、编号续接和显式重启。直接构建类型化 Block，没有 JSON 字符串往返。
- `titles.rs`：正文主字号、二三级标题提升、标题/副标题占位符及首个有可见正文的居中标题规则；标题元信息与最终 Block 分开保存。

## 验证

参考生成：`scripts/migration/export_pptx_context_reference.py`。

固定参考：`tests/fixtures/office_rust_reference_068/pptx_context.json.gz`；原 Python 生成，Rust 测试不运行 Python。

- 206 份包级上下文及逐形状文字块参考，包括本地示例、继承/链接/数学手工样例、非标准路径、分组变换、AlternateContent、字体及列表继承、60 份混合列表序列。
- 240 组 XYCut++、矩形并集与图片过滤查询。
- 2,077 组字号舍入及样式属性查询。
- 800 组标题提升/首个可见页查询。
- `./scripts/migration/test_office_native.ps1 --test pptx_context`：4 项通过。初次接模块另运行 foundation 的 5 项通过。未跑全量 Office、模型 PDF 或 CUDA；未重新构建/安装 wheel；没有 PPTX 性能结论。

注意验证边界：其中 3 份原始包先由原 Python normalizer 修复，再冻结其归一化输入进行几何核对，不能视为 Rust 已完成包级重试。python-pptx 访问 slides 会重命名 part，参考生成器在此之前保留真实 OPC 名称，以核对原生资源查找路径。

## 后续未完成

1. `pypptx-with-oxml` 的第二套公式转换回退。真实手工数学样例中主 OMML 因缺 `dPr` 失败，Python 回退后有有效结果；不能用拼接字符代替。当前 `text::paragraph/shape` 对此明确返回 Unsupported；仅上下文检查点用 `None` 标记对应段落/shape，其他文字、样式和几何仍核对。没有把这个未完成路径投入公开转换。
2. 表格 HTML、合并格，图表、图片、OLE/MTEF/公式预览的优先级与资源编排。
3. 备注正文/公式与特殊占位符排除、按“有输出块的形状”执行整页排序、分页及可见首页状态，接通标题处理及统一导出。
4. Strict、缺命名空间、contentPart fallback、坏媒体及内容类型的包规范化/失败重试；未归一化 Strict 的检查点明确拒绝，避免静默输出空页。
5. 公开 RapidDoc/Python 路由、安装校验及效果/速度/峰值内存验收。

当前入口 `pptx::context::inspect_package` 是内部完整上下文检查点，不是已经交付的 PPTX 文档转换器。后续复用同一组类型化底层规则接整页与资源流程。
