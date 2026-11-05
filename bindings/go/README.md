# RapidDoc Go preview

Go 调用共享 Rust SDK，支持 PDF、DOC、DOCX、XLS、XLSX、PPT、PPTX，不需要 Python、Java 或 PyTorch。Windows/Linux 统一使用 **C ABI + cgo**，模块没有第三方 Go 依赖。

当前为源码预览，尚未发布 Go 模块标签和原生运行库。Linux 调用层已提供，实际 Linux 原生包及 GPU 推理仍需单独验收。

## 编译准备

Go 1.20+、`CGO_ENABLED=1` 和可用 C 编译器。Windows 使用 MinGW-w64 GCC（如 MSYS2 UCRT64），Linux 使用 GCC。只有编译 Go 应用的机器需要编译器；部署编译好的应用不需要 Go、GCC 或 Rust，但需要携带原生运行库包。

本机已准备便携 GCC，不改系统 PATH，在项目根目录的 PowerShell 中可设置：

```powershell
$env:CC = "$PWD/build/tools/w64devkit/bin/gcc.exe"
$env:CGO_ENABLED = "1"
```

发布者从根目录构建原生包：

```powershell
.venv/Scripts/python.exe scripts/migration/build_go_sdk.py --output build/go-sdk/windows-x86_64-cgo
```

输出目录必须是新目录；工具不自动删除旧构建。Python 仅用于源码构建脚本，Go 应用运行时不需要 Python。

当前本机已生成 `build/go-sdk/windows-x86_64-cgo.zip`。解压整个目录并保留子目录结构，用 `Config.LibraryPath` 或 `RAPIDDOC_NATIVE_LIBRARY` 指定其中 `rapid_doc_c.dll`（Linux 为 `librapid_doc_c.so`）。ORT、PDFium、GEOS 和私有 CRT 从同一运行库目录自动发现；显式 `Resources` 路径仍优先。不会自动下载原生运行库。

## 独立项目本地引入

在调用项目的 `go.mod` 中添加（Windows 路径使用正斜杠）：

```go
require github.com/RapidAI/RapidDoc/bindings/go v0.0.0

replace github.com/RapidAI/RapidDoc/bindings/go => D:/CodeProjects/doc/RapidAI/RapidDoc/bindings/go
```

```go
package main

import (
    "fmt"
    "log"
    rapiddoc "github.com/RapidAI/RapidDoc/bindings/go"
)

func main() {
    engine, err := rapiddoc.New(rapiddoc.Config{
        LibraryPath: "D:/CodeProjects/doc/RapidAI/RapidDoc/build/go-sdk/windows-x86_64-cgo/rapid_doc_c.dll",
    })
    if err != nil { log.Fatal(err) }
    defer engine.Close()

    result, err := engine.Parse("input.pdf", nil, nil)
    if err != nil { log.Print(err); return }
    fmt.Println(result.Markdown)
    if err := result.WriteTo("output"); err != nil { log.Print(err) }
}
```

默认模型、参数和自动下载策略来自 Rust SDK，模型缓存为 `~/.cache/rapid_doc/models`。使用 `Config.Resources.ModelDirectory`、`RAPID_MODELS_DIR` 或 `RAPIDDOC_MODEL_DIR` 改变目录；`AllowDownload` 为 `*bool`，`nil` 保持默认，指向 `false` 则禁止下载。

## 输入和配置

- `Parse(path, options, progress)`：本地文件路径。
- `ParseBytes(data, options, progress)`：文档内容。
- `ParseReader(reader, options, progress)`：从当前读取位置读取，不关闭调用方的 reader；当前完整读入内存，大文档优先传路径。
- `ParseAll(paths, options, progress)`：顺序处理并复用会话；结果保留输入顺序，遇错停止。大量文件建议逐个解析、保存和释放。
- `Warmup(options, progress)` / `LoadedModels()`：预加载、查询会话；Office 不需要模型。

`Config.Options` 与单次调用的 `Options` 都使用共享 Rust 配置名，例如：

```go
config := rapiddoc.Config{Options: rapiddoc.Options{
    "layout_config": map[string]any{"model_type": "pp_doclayoutv3"},
    "formula_enable": true,
    "table_enable": true,
    "device": "cpu",
}}
options := rapiddoc.Options{
    "start_page": 0,
    "end_page": 2, // 不包含结束页：只处理第 1、2 页
    "draw_layout": true,
    "draw_span": true,
    "image_dir_name": "images",
}
```

未知字段会报错，未指定的字段保持 Rust 默认值。`Parse`/`ParseAll` 的字符串支持本地路径及 HTTP/HTTPS URL；下载、重定向和错误处理由共享 Rust 层完成。图片文件不属于直接输入范围。

## 结果、生命周期和回调

`Result` 包含 `Markdown`、`MarkdownValue`、`MiddleJSON`、`ContentListJSON`、`Metadata`、`Diagnostics`、`RawJSON`、`SourcePageCount`、`Images`、`Drawings`。非字符串 Markdown 模式读取 `MarkdownValue`。图片和绘图是二进制数据，不经 Base64。

原生响应会在返回 Go 结果前释放，结果拥有独立 Go 数据，关闭引擎后仍可读取或 `WriteTo`。显式 `Close()` 可及时释放模型会话，重复关闭安全；全局 ORT/动态库会保留至进程退出。返回的原生错误类型为 `*NativeError`，提供稳定的 `Code` 和 `Message`。

`WriteTo(outputRoot)` 统一保存到 `outputRoot/源文件名/auto|txt|ocr|office/`，输出带源文件名前缀，例如 `示例1.md`、`示例1_middle.json`、`示例1_content_list.json` 和 `示例1_layout.pdf`。图片保持原目录及相对链接。`OutputDirectory(outputRoot)` 返回最终目录；字节/流输入使用 `document` 作为名称。结构化 Markdown 保存为 `源文件名_markdown.json`。

进度是同步回调：

```go
progress := func(event rapiddoc.Event) error {
    if event.Completed != nil && event.Total != nil {
        log.Printf("%s %d/%d", event.Stage, *event.Completed, *event.Total)
    }
    return nil
}
```

未知计数为 `nil`；下载事件与 Java/Python 共用同一套 Rust 事件。**传 `nil` 默认显示解析进度**，不需要在应用里再写进度打印。阶段名称、20 格进度条、`processing` 耗时、后处理和完成信息与 Java 一致。重复的累计完成事件会过滤，新批次重新计数。开始/完成信息写 stdout，进度条写 stderr；IDE 的红黑通常由输出流决定，VS Code 终端可能对两者使用同一颜色。

交互终端按行刷新，重定向时使用逐行格式。`Config.ProgressMode` 可选 `auto` / `bar` / `plain`，未填写时读取 `RAPIDDOC_PROGRESS_MODE`，否则默认 `auto`。例如 IDE 的非终端 Run 面板不能自动识别时，可显式选 `bar`。`processing` 是原生阶段实际处理时间，最终完成时间为真实经过时间。

`Config.ProgressLogging` 是 `*bool`，`nil` 默认开启、指向 `false` 关闭默认解析展示。提供自定义回调后由调用方控制解析展示，下载开始、大小进度、校验、完成和失败日志仍默认保留，与 Java 的行为一致。

回调返回错误会取消当前调用并原样返回；回调 panic 在原生调用返回后于 Go 侧重新抛出，不跨 C 栈展开。不要在回调中使用 `runtime.Goexit` / `t.Fatal`。回调同步重入同一引擎解析会立即报错，不死锁。同一个引擎支持并发解析：Office 独立执行，PDF 只在 PDFium 和共享模型会话处串行保护。并发写出时使用不同输出路径。`Close()` 不等待正在执行的回调，已在运行的原生调用持有独立生命周期引用。

## 手动示例

在根目录设置编译环境后：

```powershell
$env:RAPIDDOC_NATIVE_LIBRARY = "$PWD/build/go-sdk/windows-x86_64-cgo/rapid_doc_c.dll"
cd bindings/go
go run ./cmd/rapiddoc-go -output ../../demo/output/go ../../demo/ms_office_docs/docx_01.doc
```

也可传 PDF。默认 PDF 会加载所需模型，缺少模型则按 Rust 策略下载。CUDA 使用配套 GPU ORT 包，发布者构建时加 `--cuda --ort-dir <GPU-ORT目录>`，调用配置 `"device":"cuda"`；仍需要匹配的 NVIDIA 驱动、CUDA/cuDNN。Windows 真 GPU、自定义算子及 CRT 共存待验收。
