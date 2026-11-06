# rapid-doc-rust

RapidDoc 的 Python 绑定，调用共享 Rust 引擎，在本地将文档解析为 Markdown。支持 PDF（电子版和扫描版）、DOC、DOCX、XLS、XLSX、PPT、PPTX。

包名为 `rapid-doc-rust`，导入名为 `rapid_doc_rust`。要求 Python 3.10 或以上，无须安装原 Python RapidDoc 或 PyTorch。

正常解析及已公开的底层 API 均包含在默认包中。迁移差分检查点属于源码开发工具，使用方式见[绑定整理说明](https://github.com/RapidAI/RapidDoc/blob/main/docs/plans/2026-11-04-python-binding-cleanup.md)。

## 安装

本页对应 SDK `0.1.1`。各渠道上架状态见[发布说明](https://github.com/RapidAI/RapidDoc/blob/main/docs/publishing.md)；CPU 环境安装：

```bash
python -m pip install "rapid-doc-rust[cpu]==0.1.1"
```

仅解析 Office 或自行提供 ONNX Runtime 时，可安装 `rapid-doc-rust==0.1.1`。离线环境也可下载对应平台的 wheel 后本地安装。

平台包覆盖 Windows/Linux x64 与 ARM64，以及 macOS 14+ 的 Intel x64 / Apple Silicon ARM64（CPU）。其他平台使用对应文件名；CUDA 环境须选择匹配的 GPU 构建及运行库。

基础包不强制安装 ORT。安装 `[cpu]` 使用 `onnxruntime`；GPU 构建安装 `[gpu]` 使用 `onnxruntime-gpu`（也能执行 CPU 推理），两者不要同时安装。已有合适运行库时直接安装 wheel 即可；Rust 自动定位其动态库，也支持显式设置 `RAPIDDOC_ONNXRUNTIME_LIBRARY`。Office 解析不需要 ORT。

本次构建覆盖六个平台的 CPU wheel。`[gpu]` 仅提供 Python GPU ORT 依赖，不代表已提供完整的 CUDA SDK 包；GPU 部署仍需匹配的 CUDA 构建、自定义算子、NVIDIA 驱动及 CUDA/cuDNN。extras 不会把 CPU Rust 构建变成 CUDA 构建。

### ORT 版本与原 Python 包的区别

| 使用方式 | ORT 来源 | 版本声明 |
| --- | --- | --- |
| 原 Python `rapid-doc[cpu]` | pip `onnxruntime` | `>=1.18.0` |
| 原 Python `rapid-doc[gpu]` | pip `onnxruntime-gpu` | `<=1.23.0` |
| `rapid-doc-rust[cpu]` | pip `onnxruntime` 的动态库 | `>=1.23.0` |
| `rapid-doc-rust[gpu]` | pip `onnxruntime-gpu` 的动态库 | `>=1.23.0` |
| Rust `download-runtime` / Java 默认平台包 | 构建时下载并打包官方原生运行库 | 当前基线 `1.23.2` |

Rust 引擎编译使用 ORT C API 23，因此最低要求 ORT 1.23；不能直接沿用原 Python 的 1.18 下限。Rust crate `ort = 2.0.0-rc.15` 是 Rust 绑定自身的版本，不是运行库版本。`>=1.23.0` 表示最低要求，不代表所有后续版本都已验证，也不会固定 pip 最终安装的版本。需要可复现环境时，在应用自己的依赖锁文件中固定已验证的 CPU/GPU 版本。

原 Python 的 GPU 上限属于原实现的依赖策略，不代表 Rust 也必须采用相同上限。本次未改变原 Python 的依赖范围；两套实现共用环境时，GPU 约束交集目前只有 `1.23.0`。不满足交集时应分别使用环境，避免强制覆盖依赖。

## 快速开始

缺少模型时自动记录下载开始、保存路径、大小进度、SHA-256 校验及完成/失败日志。默认目录为 `~/.cache/rapid_doc/models`，可用 `RAPID_MODELS_DIR` / `RAPIDDOC_MODEL_DIR` 修改；模型校验通过后直接复用。

```python
from rapid_doc_rust import RapidDoc

engine = RapidDoc(output_dir="output")
result = engine("input.pdf")
print(result.markdown)
```

设置 `output_dir` 后，自动保存 Markdown、图片、中间 JSON 和内容列表。省略它则只返回结果。

Office 使用相同入口；复用同一个实例可避免重复加载模型：

```python
result = engine("report.docx")
results = engine(["input.pdf", "report.docx", "table.xlsx"])
```

输入也支持 HTTP/HTTPS URL 字符串、文件内容 `bytes` 或二进制文件流。URL 下载由共享 Rust 层执行，直接传 URL，不要拼接项目根目录。列表输入返回结果列表。

## 实例复用与资源释放

短脚本可以使用上下文管理器，在正常结束或发生异常时释放原生模型会话：

```python
with RapidDoc(output_dir="output") as engine:
    for source in ["input.pdf", "report.docx"]:
        result = engine(source)
        print(result.markdown)
```

服务中在启动时创建实例，处理多个请求，在应用关闭时调用 `engine.close()`。
`close()` 可重复调用，`engine.closed` 返回关闭状态。关闭后再次解析或预加载会报错，
已有结果可继续读取和保存。解析期间调用 `close()` 会提示实例正在使用，待本次解析
结束后再关闭。同一实例支持多线程并发解析；Office 请求独立执行，PDF 保留 PDFium 和各模型会话锁，共享模型权重。不要在进度回调中同步重入解析；并发写出时使用不同输出路径。

## 返回结果

| 字段 | 内容 |
| --- | --- |
| `result.markdown` | Markdown 字符串 |
| `result.images` | 图片文件名到 `bytes` 的字典 |
| `result.middle_json` | 中间结构，页面位于 `pdf_info` |
| `result.content_list_json` | 内容列表 |

也可以使用 `markdown, images = result` 解包。

## 常用参数

```python
engine = RapidDoc(
    parse_method="auto",
    formula_enable=True,
    table_enable=True,
    pdf_pages_batch=64,
    output_dir="output",
)

result = engine(
    "input.pdf",
    start_page_id=0,
    end_page_id=9,
    f_draw_layout_bbox=True,
    f_draw_span_bbox=True,
)
```

页码从 0 开始，`end_page_id` **包含该页**；省略它表示解析到末页。上例解析前 10 页，并保存版面框、文字框 PDF。

- `parse_method`：默认 `auto`，也可选择 `txt` 或 `ocr`，用于 PDF。
- `formula_enable`、`table_enable`：默认开启。
- `pdf_pages_batch`：PDF 分页处理窗口，默认 64 页。
- `preload_model=True`：提前加载当前配置需要的模型，后续解析复用相同会话。
- `image_output_mode="data_uri"`：将 Markdown 图片引用改为内嵌数据；默认是相对路径。

### 模型切换

```python
engine = RapidDoc(
    layout_config={"model_type": "pp_doclayoutv3"},
    formula_config={"model_type": "pp_formulanet_plus_m"},
    table_config={
        "model_type": "unet_slanet_plus",
        "rule_table_engine": "docvortex_ocr",
        "fallback_model_type": "table_transformer",
        "fallback_to_text": True,
    },
)
```

| 配置 | 默认模型 | 可选模型 |
| --- | --- | --- |
| `layout_config.model_type` | `pp_doclayoutv3` | `pp_doclayoutv2`、`docvortex_rule` |
| `formula_config.model_type` | `pp_formulanet_plus_m` | `pp_formulanet_plus_s`、`pp_formulanet_plus_l` |
| `table_config.model_type` | `unet_slanet_plus` | `slanet_plus`、`unet`、`table_transformer` |

上例显式开启了 OCR 规则表格、备用模型和文字回退。默认规则引擎为 `docvortex`；备用模型与文字回退默认关闭，可分别用 `fallback_model_type=None`、`fallback_to_text=False` 关闭。

## 模型与设备

PDF 模型按需加载、下载并校验；默认模型目录为 `~/.cache/rapid_doc/models`。可在创建实例前用环境变量 `RAPID_MODELS_DIR` 指定已有模型目录。Office 解析不加载 PDF 识别模型。

默认使用 CPU。使用 GPU 构建时，可在创建实例前设置 `os.environ["MINERU_DEVICE_MODE"] = "cuda"`。

整页方向矫正默认关闭，需要时启用：

```python
import os
os.environ["USE_DOC_ORIENTATION_CLASSIFY"] = "true"
```

## 开发文档

- [共享 Rust SDK 与原生调用示例](https://github.com/RapidAI/RapidDoc/blob/main/rust/rapiddoc/README.md)
- [Python 绑定历史迁移记录](https://github.com/RapidAI/RapidDoc/blob/main/docs/archive/python-binding-migration.md)（仅供追溯，不作为当前用法）
