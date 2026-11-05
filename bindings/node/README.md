# RapidDoc Node.js

Node.js / TypeScript bindings for the shared Rust parser. Supports PDF, DOC,
DOCX, XLS, XLSX, PPT and PPTX. Browser execution is not supported.

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

The platform package contains the native addon and CPU runtime libraries.
Users do not need Python, Java, Rust or a C compiler. Models are automatically
resolved/downloaded by the shared Rust SDK. This source preview is not published
to npm yet; install the locally built `.tgz` with `npm install /path/package.tgz`.
Node.js 20+ is required; Windows x64 is currently tested with Node.js 24.12.
Linux and CUDA distribution still require package acceptance testing.

## Inputs and results

`parse` accepts a local path, HTTP/HTTPS string or URL object, file URL, Buffer, Uint8Array or array of inputs. HTTP input is downloaded by the shared Rust SDK; pass the URL directly without `path.resolve()`.
Arrays / `parseAll` process files sequentially, reusing models. `parseReader`
accepts a binary Node readable or async iterable; it buffers the full input.
No HTTP fetch is performed for input documents.

```js
const results = await engine.parse(['a.pdf', 'b.docx']);
const result = await engine.parse(bytes, {
  start_page: 0, end_page: 2, // End page is exclusive, matching the Rust SDK.
  draw_layout: true, draw_span: true,
});
```

Results expose `markdown`, `middleJSON`, `contentListJSON`, `metadata`,
`diagnostics`, `sourcePageCount`, `images` and `drawings`. Assets are Node Buffers
and remain valid after the engine closes. `writeTo(outputRoot)` saves under
`outputRoot/<sourceName>/<auto|txt|ocr|office>/`, with names such as
`input.md`, `input_middle.json`, `input_content_list.json`, `input_layout.pdf`
and `input_span.pdf`. Images remain in their configured directory so Markdown
links stay valid. `outputDirectory(outputRoot)` returns the final directory.
Binary/stream inputs use `document` as their source name. Structured Markdown
uses `<sourceName>_markdown.json` instead of a Markdown file.

## Configuration, progress and cancellation

Shared Rust configuration uses its existing field names and defaults:

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

Default progress uses the same stage labels and active processing seconds as
the Java/Go SDK. Bars refresh on stderr in terminals; redirected output uses
plain lines. Status and download messages go to stdout. Set
`progressLogging: false` to hide parsing progress, or supply an optional async
progress callback as the third parse argument. Download messages remain visible.
`RAPIDDOC_PROGRESS_MODE=bar` can force carriage-return bars in IDE consoles.

Each engine starts one worker and reuses additional workers on demand for
concurrent calls. Native model sessions are shared through the process cache;
parsing and inference do not block the main Node event loop. Await `close()` to
cancel active calls and stop every owned worker. Callback exceptions are returned
to the caller unchanged; callback reentry into the same engine is rejected. An
AbortSignal cancels cooperatively at the next shared SDK progress event; a
running model inference is not interrupted in the middle of a kernel.

Advanced deployments can set `libraryPath` (or `RAPIDDOC_NODE_LIBRARY`) and
`runtimeDirectory`, or explicit paths under `resources`. These are unnecessary
with the prepared platform package. CUDA requires a matching GPU runtime bundle
and its CUDA/cuDNN prerequisites; the CPU preview does not install them.

## Build and verify locally

Build a shared runtime first using `scripts/migration/build_go_sdk.py` (the
runtime libraries are shared across bindings), then:

```powershell
.venv/Scripts/python.exe scripts/migration/build_node_sdk.py --runtime-dir build/go-sdk/windows-x86_64-cgo --output build/node-sdk/windows-x86_64
```

Native acceptance tests use `RAPIDDOC_TEST_ROOT`, `RAPIDDOC_NODE_LIBRARY`, and
`RAPIDDOC_NODE_RUNTIME`; run `node --test bindings/node/test/*.test.js`.

同一实例支持并发 `parse()`：按需创建并复用工作线程，原生模型会话通过进程内缓存共享。Office 独立执行；PDF 保留 PDFium 与模型会话锁。`parseAll()` 仍按输入顺序逐个执行，避免默认展开大量文档。进度回调中禁止同步重入同一实例；并发写出请使用不同输出路径。`close()` 会取消并关闭该实例的全部工作线程。
