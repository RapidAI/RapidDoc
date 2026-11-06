# RapidDoc Node.js SDK

[English](README_EN.md) | [中文](README.md)

Node.js / TypeScript 绑定调用共享 Rust 解析引擎，支持 PDF、DOC、DOCX、XLS、XLSX、PPT、PPTX，不支持浏览器运行。

## 安装与发布状态

平台包包含原生扩展和 CPU 运行库，使用者不需要安装 Python、Java、Rust 或 C 编译器。模型由共享 Rust SDK 自动定位或下载。

`0.1.0` 发布状态（2026-11-06）：Linux x64、macOS Intel/ARM64 的 npm 平台包已发布；Windows 包 `rapid-doc-win32-x64` 被 npm 包名反垃圾检测拦截，正在等待审核。`rapid-doc` 主包暂未发布，因此目前不能直接使用 `npm install rapid-doc` 安装该版本。

下一次构建将 Windows x64 平台包改为 `@hzkitty/rapid-doc-win32-x64`，对应 CUDA 包为 `@hzkitty/rapid-doc-win32-x64-cuda`。
主包继续使用 `rapid-doc`，其他平台包保持原名；Windows 平台包由主包自动选择，用户调用方式不变。
scoped 包尚未发布，发布 Token 需具有对应包的写入权限；使用 scope 不保证免于 npm 反垃圾审核。

当前可按下方说明构建 standalone 包，再通过 `npm install /path/package.tgz` 安装。安装后统一使用 `require('rapid-doc')`，调用方式如下。

要求 Node.js 20+。CPU 平台包覆盖 Windows/Linux x64，以及 macOS 14+ Intel x64 / Apple Silicon ARM64。加载器根据运行 Node.js 的系统和架构选择平台包。Windows/Linux CUDA 仍需单独实机验收。

## 快速开始

```js
const { RapidDoc } = require('rapid-doc');

async function main() {
  const engine = new RapidDoc();
  try {
    const result = await engine.parse('input.pdf');
    console.log(result.markdown);
    await result.writeTo('output');
  } finally {
    await engine.close();
  }
}
main().catch(console.error);
```

## 输入与结果

`parse` 支持本地路径、HTTP/HTTPS 字符串或 URL 对象、file URL、Buffer、Uint8Array，以及输入数组。HTTP/HTTPS 输入由共享 Rust SDK 下载；直接传入 URL，不要使用 `path.resolve()` 转换。

数组输入和 `parseAll()` 按顺序处理文档并复用模型。`parseReader()` 接收 Node 二进制可读流或异步可迭代对象，当前会将输入完整缓存在内存中。

```js
const results = await engine.parse(['a.pdf', 'b.docx']);
const result = await engine.parse(bytes, {
  start_page: 0, end_page: 2, // End page is exclusive, matching the Rust SDK.
  draw_layout: true, draw_span: true,
});
```

结果包含 `markdown`、`middleJSON`、`contentListJSON`、`metadata`、`diagnostics`、`sourcePageCount`、`images` 和 `drawings`。图片和绘图使用 Node Buffer，关闭引擎后结果仍然有效。

`writeTo(outputRoot)` 将结果保存到 `outputRoot/<sourceName>/<auto|txt|ocr|office>/`，文件包括 `input.md`、`input_middle.json`、`input_content_list.json`、`input_layout.pdf` 和 `input_span.pdf`。图片保存在配置的目录中，保持 Markdown 引用有效；`outputDirectory(outputRoot)` 返回最终输出目录。

二进制或流输入默认使用 `document` 作为名称。结构化 Markdown 输出使用 `<sourceName>_markdown.json`，而不是 Markdown 文件。

## 配置、进度与取消

配置沿用共享 Rust SDK 的字段名和默认值：

```js
const engine = new RapidDoc({
  config: {
    device: 'cpu',
    table_config: { model_type: 'table_transformer' },
  },
  // resources: { model_directory: '/models', allow_download: false },
  progressMode: 'auto', // auto / bar / plain
});
const controller = new AbortController();
const result = await engine.parse('input.pdf', {}, null, controller.signal);
```

默认进度使用与 Java/Go SDK 相同的阶段名称和有效处理耗时。终端中进度条在 stderr 刷新，重定向输出时改为普通文本行；状态和下载消息输出到 stdout。

设置 `progressLogging: false` 可关闭解析进度，也可在 `parse` 的第三个参数传入异步进度回调。下载消息仍会显示。环境变量 `RAPIDDOC_PROGRESS_MODE=bar` 可在 IDE 控制台强制使用回车刷新进度条。

每个引擎先启动一个工作线程，并按并发需求创建、复用更多线程。原生模型会话通过进程缓存共享，解析和推理不会阻塞 Node 主事件循环。Office 请求独立执行；PDF 保留 PDFium 和各模型会话的锁。`parseAll()` 保持顺序执行，不会默认并行展开大量文档。

调用 `await engine.close()` 会取消活动请求并关闭该实例的全部工作线程。进度回调抛出的异常原样返回调用方；不允许在回调中同步重入同一引擎。并发写出时应使用不同输出路径。

`AbortSignal` 在下一个共享 SDK 进度事件处协作取消；已经执行中的模型推理不会在算子运行中途被强行中断。

高级部署可设置 `libraryPath`（或 `RAPIDDOC_NODE_LIBRARY`）、`runtimeDirectory`，也可通过 `resources` 显式指定运行库路径。使用完整平台包时通常无须设置。CUDA 还需匹配的 GPU 运行库包及 CUDA/cuDNN，已发布的 CPU 包不提供这些依赖。

## 本地构建与验证

先使用 `scripts/migration/build_go_sdk.py` 构建共享运行库目录，各绑定复用这套运行库，然后执行：

```powershell
.venv/Scripts/python.exe scripts/migration/build_node_sdk.py --runtime-dir build/go-sdk/windows-x86_64-cgo --output build/node-sdk/windows-x86_64
```

原生验收测试使用 `RAPIDDOC_TEST_ROOT`、`RAPIDDOC_NODE_LIBRARY` 和 `RAPIDDOC_NODE_RUNTIME`；运行 `node --test bindings/node/test/*.test.js` 执行测试。
