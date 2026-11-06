# RapidDoc Rust SDK

[English](README_EN.md) | [中文](README.md)

直接使用共享 `rapiddoc` crate。Python、Java、Go、Node.js 绑定均调用这套 SDK，共享模型策略和解析算法。

发布状态（2026-11-06）：11 个 crate 已全部发布到 crates.io，包括主包 `rapiddoc` 和 `rapiddoc-cli`。可直接使用下方的版本依赖。

Windows/Linux x64、macOS 14+ Intel/ARM64 的 CPU 运行库 ZIP 已在 [SDK Release](https://github.com/RapidAI/RapidDoc/releases/tag/sdk-v0.1.0) 提供。各渠道最新发布状态见[发布说明](../../docs/publishing.md)。

## 安装 ONNX Runtime

Rust 应用可开启 `download-runtime` 功能，在 Cargo 构建时下载并校验官方 ORT 1.23.2 归档，覆盖 Windows MSVC x64、Linux glibc x64、macOS Intel 和 Apple Silicon CPU：

```toml
[dependencies]
rapiddoc = { version = "0.1.0", features = ["download-runtime"] }
# For GPU use features = ["download-runtime", "cuda"] instead.
```

也可执行 `cargo add rapiddoc@0.1.0 --features download-runtime`。源码开发时可将版本依赖换成 `path = "../RapidDoc/rust/rapiddoc"`。该功能需要显式启用；各语言绑定自行提供运行库，不会触发此下载。运行时不需要 Python。

启用 `download-runtime` 和 `cuda` 时会下载 GPU ORT，但机器仍需提前安装 NVIDIA 驱动及匹配的 CUDA/cuDNN。默认构建使用 CPU，明确请求 CUDA 时不会静默回退为 CPU。

下载文件缓存在 Cargo 构建输出目录，按固定的上游 SHA-256 校验。离线构建可将 `RAPIDDOC_ORT_ARCHIVE` 指向相同的官方归档，仍会校验其哈希。显式指定的外部运行库路径优先于自动发现。

构建后会将 ORT、provider 库及许可说明复制到 `target/<profile>/runtime`，并放到示例和测试程序附近。部署时需将整个 `runtime` 目录随可执行文件一起分发，不能依赖构建机上的绝对路径。清理 Cargo 输出会删除这份缓存。

`download-runtime` 仅安装 ORT。PDFium、GEOS，以及可选的 CUDA 自定义算子仍需从 RapidDoc 对应平台运行库包提供。

## 快速开始

```rust
use rapiddoc::{RapidDoc, RapidDocConfig, RuntimeResources, ParseOptions};

let config = RapidDocConfig::default()
    .layout_model("pp_doclayoutv3")
    .formula_model("pp_formulanet_plus_m")
    .table_model("unet_slanet_plus");
let resources = RuntimeResources {
    bundle_dirs: vec!["runtime".into()],
    ..Default::default()
};
let mut engine = RapidDoc::new(config, resources);
let result = engine.parse("document.pdf", &ParseOptions::default())?;
println!("{}", result.markdown.as_str().unwrap_or(""));
```

默认配置与 Python 一致：自动解析，每批 64 页、200 DPI，开启公式和表格，使用 V3 版面、M 公式、UNET+SLANet 表格模型，不配置额外兜底模型。

原有配置映射仍可使用 `layout_config`、`ocr_config`、`formula_config`、`table_config` 和 `image_config`。模型别名、参数校验与会话设置由共享 SDK 处理，各语言无需重复实现。

## 输入、输出与运行库

`parse` 接收本地文件路径，也支持 HTTP/HTTPS 字符串，例如 `engine.parse("https://example.com/report.pdf", &options)`。共享加载器跟随 HTTP 重定向、发送下载进度并返回 HTTP 错误；连接、响应和读取超时均为 60 秒。URL 内容读入内存，不建立磁盘下载缓存，也不额外设置默认大小限制。下载结果包含 `metadata.source_name`，供绑定确定输出名称。

`parse_bytes` 用于传入文档内容。DOC、DOCX、XLS、XLSX、PPT、PPTX 使用相同的 `parse` / `parse_bytes` 方法，不初始化 PDF 或模型会话。PDF 解析需要对应平台的 PDFium、ORT 和 GEOS；CUDA 模式还需算子运行库。

`RuntimeResources` 支持运行库包目录、显式库路径、模型目录和下载开关。不依赖 Python 环境，也不搜索开发源码目录。模型下载、哈希及缓存目录由已有的共享模型存储处理。

`ParseOptions.end_page` 不包含结束页；Python 绑定会将包含结束页的 `end_page_id` 转换为该语义。`image_dir_name` 和 `draw_layout` / `draw_span` / `draw_line_sort` 控制资源引用及可选的矢量标注 PDF。

结果包含 Markdown、中间数据、内容列表、图片和绘图，SDK 本身不强制写入文件。`Middle` 在序列化前保留类型化的 Office 或 PDF 结构；图片保留 Vec/Arc 所有权，不额外复制。

## 模型预加载与配置更新

`warmup` 与解析共用原生会话注册表，只预加载当前配置需要的模型；单独使用 TATR 不会同时加载 UNET 或分类模型。`loaded_model_roles` 可查看已经加载的模型角色。

`configure` 保留未变化的模型绑定，仅使模型或会话设置发生变化的角色失效。资源和模型错误直接返回调用方。`parse_with_host` / `warmup_with_host` 提供进度与可选的宿主设备准备回调；普通 Rust 调用方使用 `parse` / `parse_bytes` / `warmup` 即可。

原生 CUDA 构建需启用 `rapiddoc/cuda`，并提供 GPU ORT 和 CUDA 运行库。

## 并发与实例复用

`parse` / `parse_bytes` / `warmup` 接收 `&self`，因此可以通过 `Arc<RapidDoc>` 服务并发请求。Office 处理使用请求独立的数据；PDFium 保留并发保护，每个共享模型会话分别串行推理。OCR 正文预处理和解码在模型会话锁之外执行，不复制完整 logits。

修改配置需要 `&mut self`；克隆引擎保留有效的配置快照。不要在进度回调中同步重入解析；并发写出时使用不同的输出路径。

## 源码构建与验证

工作区示例：

```powershell
cargo run --release -p rapiddoc --example parse_document -- tests/migration/inputs/synthetic.docx
cargo run --release -p rapiddoc --example parse_document -- document.pdf path/to/release/runtime
```

本地可运行 `python scripts/release/crates.py --allow-dirty` 检查源码包及解包编译，该命令默认不发布。

原生构建需要 C/C++ 工具链、CMake 和 NASM：Windows 使用 MSVC，Linux 使用 GCC/Clang，macOS 使用 Xcode Command Line Tools。PDFium、GEOS、ORT 仍是独立运行库资源，仅完成 Cargo 编译不等于已准备全部运行库。手动发布流程见[发布说明](../../docs/publishing.md)。
