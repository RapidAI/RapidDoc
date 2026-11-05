# DOCX 原生导出与 Python 接入

DOCX 现已接通 `rapid_doc_rust.RapidDoc`：原生 Rust 完成文档解析、Office 中间结构、Markdown、content_list、content_list_v2 和图片资产提取。Python 负责输入读取、配置、日志和文件写出；不调用 python-docx、Mammoth、openpyxl 或原 Python Office 转换器。

## 使用

直接使用项目当前 `.venv`，运行 `demo/rust_docx_demo.py`，或：

```python
from rapid_doc_rust import RapidDoc

engine = RapidDoc(output_dir="demo/output/rust-docx")
result = engine("demo/docx/demo1-docx.docx")
print(result.markdown)
```

文件输出在 `<output_dir>/<文件名>/office/`，包含 Markdown、middle/content_list JSON 和 images。保留原 `RapidDocOutput` 的四个字段及二元迭代行为；支持路径、bytes、bytearray、可读流、URL 和批量输入，流按原 API 尝试回到起点。支持自定义 writer、图片目录、data_uri 模式和两个 JSON 导出开关。命名为空的原始 bytes 不被当作合法 DOCX；有 `.docx` 名称的零字节文件按原入口生成空结果。

Office 路由按原 Python 忽略 PDF 的页码裁剪、识别模型开关和画框选项；DOCX 的“页”仍是原转换器的分节/逻辑页，不是 Word 实际分页。默认不加载识别模型或 PDFium；显式 `preload_model=True` 仍遵循原初始化预加载选项。

## 共用 Rust 输出层

- `output/mod.rs`：直接消费已完成的 typed DOCX blocks，不先转 JSON 再解析。保留原题注分类、按有效索引距离绑定、连续题注选择、两层媒体块、标题自动编号/编号同步、目录有效锚点和 discarded 内容。
- `output/inline.rs`：保留原块级 Markdown/HTML 选择、复杂样式、连续 wrapper 折叠、可见下划线/删除线空格、公式间距、超链接 children、行内代码、标点边界和正文 Markdown 前缀转义。
- `output/render.rs`：保留原 mm/nlp Markdown、content_list、content_list_v2 的列表起点/缩进、目录页码尾部、题注、图表、图片路径、HTML 内公式与标题/正文锚点。
- `output/html.rs`：保留 HTML 属性清理与链接策略、内联图片提取、同名资产去重和 SHA256(data URI) 命名。图片数据通过 Arc 复用，在 Python 边界只输出结果 JSON 和资产字节。
- 公式定界符沿原 `MINERU_TOOLS_CONFIG_JSON` / `~/magic.json` 配置；未配置时使用 `$` 和 `$$`。
- Python 输入层取消原 Rust 兼容层额外加的 256 MiB 输入限制，保留容器自身已记录的格式检查与原 Office 资源策略。

## 验证与限制

原 Python 冻结参考覆盖 **353 份文档**，其中 352 份完整中间结构、四种导出及图片内容对比通过，1 份保留预期损坏输入。图片比较尺寸与解码 RGBA，SVG 比较结构及嵌入像素；不放宽文本/结构差异。

**PNG 压缩字节可能不同，导致 data URI 哈希文件名不同**：验收先核对图片内容，再按已核对的资产映射比较引用。当前不是与 Python 的 PNG 压缩字节/哈希文件名逐字节相同，不应把此验收表述为所有文件字节相同。JPEG 常用路径继续复用 libjpeg-turbo。

Office crate 阶段 **60 项测试通过**，`cargo fmt --check` 和 `pip check` 通过。安装与详细测试记录见 `build/office-docx-public-receipt.json`；安装测试覆盖 23 份真实/资源参考、输入变体、batch/writer/data_uri、空文件、公式定界符、无 Python Office 导入和无模型加载，并在 Office 后实际读取 PDF。另有 8 组实际 PDFium 同进程共存检查通过。没有全量 PDF/CUDA 或同条件性能/峰值内存基准。

构建与许可入口为 `scripts/migration/build_office_wheel.ps1`：复制 canonical Office 许可证/来源声明进 wheel，再调用 maturin release 构建。生成 Unicode/HTML 表的脚本为 `generate_office_output_unicode.py`；原输出算法来源的哈希记入 `packaging/office-native-provenance.json`。

本次交付 **DOCX**。DOC、XLS、XLSX、PPT、PPTX 默认接口继续明确报未接入，不能把图表内的 XLSX 数据读取称为完整 XLSX 解析。少见图片编码、跨平台字体/渲染及更广泛客户异常文档仍需扩大验收；当前不承诺 Rust 已普遍更快或更省内存。

旧 `api_compat_install.py` 仍要求构造函数与原 Python 参数完全相等，未计入已有 Rust CPU 配置扩展，且当前缺其 `build/migration/api-document.json` 基线，本轮未以该旧脚本验收完整 PDF 模型输出；以实际 PDF 文字读取和 PDFium 共存检查验证本次边界。
