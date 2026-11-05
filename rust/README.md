# Rust 实现导航

工作区由项目根目录 `Cargo.toml` 管理。Python 发行名为 `rapid-doc-rust`，导入名为 `rapid_doc_rust`；PyO3 适配位于 `bindings/python`，解析算法由这些共享 Rust 包实现。

| 包 | 职责 |
| --- | --- |
| `rapiddoc-core` | 错误、几何、资源政策、进度和基础事实契约 |
| `rapiddoc-pdf` | PDFium 读取、字体、字符、路径、图片、分类和渲染 |
| `rapiddoc-native` | 基于原生事实的文字、线条、表格规则及 DocVortex 适配 |
| `rapiddoc-inference` | ONNX 会话与版面、OCR、公式、表格、印章的前后处理 |
| `rapiddoc-render` | 已有表格结构的 HTML 输出与候选质量度量 |
| `rapiddoc-office` | Office 容器、图像与公式、共享导出；DOCX/XLSX/XLS/PPTX 已接通，其余格式继续移植 |
| `rapiddoc` | 正式 PDF 流水线、规则处理、页面组装和全文后处理 |
| `rapiddoc-cli` | 当前原生文字预览及能力查询；不是完整模型 PDF 入口 |

`vendor/` 是第三方来源边界；来源、许可证和本地改动按已有说明管理，不与业务模块归组。

## 业务代码从哪里找

```text
rapiddoc/src/
  pipeline/                  正式 Engine、模型调度、OCR/表格/印章流程
  processing/
    mod.rs                   类别转换、过滤及既有模块导出
    document/                文档编排、容器、内联物化和最终输出
    layout/                  版面、阅读顺序、原生/模型适配
    text/                    原生行、标题、参考文献、脚注和文字组装
    table/                   表格候选、网格、文字、仲裁和页面提交
    graphics/                图形、图表、diagram 和图片区域
    formula/                 公式规则、文字与矢量公式恢复
    ocr/                     检测/识别任务、裁图规划及结果回填
  page_assembly/             页面结构、文字/公式/表格组装、资产和导出
  operations.rs              兼容诊断协议的操作分发
```

新的规范 Rust 路径是 `rapiddoc::processing` 和 `rapiddoc::page_assembly`。旧 `rapiddoc::legacy_model`、`rapiddoc::legacy_page` 是同一模块的兼容重导出，并无第二份算法。为保留 Rust 调用者和诊断协议，`processing` 继续导出原模块名；`#[path]` 将其定位到领域目录。含子模块的组件使用 `组件/mod.rs`，相关 JSON 规则与实现放在一起。

推理层的 `src/` 按 `runtime/`、`image/`、`ocr/`、`layout/`、`formula/`、`table/`、`seal/` 存放代码，原 `rapiddoc_inference::{ctc, layout_post, ...}` 路径不变。`lib.rs` 保留共享会话与模型方法；两个 C++ 内核仍由 `build.rs` 在原路径构建，算法及编译参数未改。

## Python 绑定的职责

`bindings/python/src/lib.rs` 只保留统一注册、异常转换、序列化及共享类型别名。按职责查找 `pdf.rs`、`assets.rs`、`tables.rs`、`ocr.rs`、`formula.rs`、`models.rs`、`diagnostics.rs`；正式入口仍在 `pipeline.rs`，进度显示在 `progress.rs`。Python 导出名、类型、参数和注册顺序保持兼容。

底层预览/差分接口不作为其他语言必须重写的业务流程。后续语言绑定调用共享 `rapiddoc::pipeline::Engine`，按各语言习惯适配模型描述符、进度和输出即可。

## 数据与后续 Office 接入

`core::document::Document/Page` 当前描述 PDF 事实，包括 PDF 字符/路径；不能将其视为已经完成的通用 Office 输出模型。Office 按原 Python 各格式路径独立读取文档结构，只复用已具备等价语义的基础类型、资产、规则和输出能力；不强制转 PDF 或 OCR，不预建空 crate。

`rapiddoc-office/src/docx/`、`xlsx/`、`xls/`、`pptx/` 分别负责原生格式读取，`spreadsheet/` 共享单元格与表格投影，`officeart.rs`、`equation/`、`image/`、`metafile/` 为 Office 资源，`output/` 统一中间结果和导出。Python Office 薄绑定为 `bindings/python/src/office.rs`，公开 `RapidDoc` 自动分发 DOCX/XLSX/XLS/PPTX；其他语言可直接复用这些 Rust 转换与导出入口。

本次保留正式流水线的 JSON 适配边界、FP32/FP64 往返规则、模型参数和资源政策。强类型化及进一步跨格式抽象应按真实 Office 契约单独实施，并继续使用冻结效果基线验证。

## 验证和维护

整理前源码与 API 快照、移动清单及实现核对位于 `build/pdf-structure-*-20261030*`。`scripts/migration/audit_pdf_structure.py` 核对实现/常量/字符串与 JSON 资源，允许模块路径、可见性等价路径及 rustfmt 的常规排版变化；冻结快照位于本机 build 目录，不随 wheel 发布。

生成器和 `audit_compatibility.py` 已改为新的实际文件位置；兼容性清单记录拆分后的绑定文件。开发只运行受影响测试，阶段结束构建安装并运行完整 PDF 效果基线。结构移动不代表已经提速或降低内存。
