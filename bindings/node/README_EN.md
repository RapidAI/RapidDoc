# RapidDoc Node.js

[English](README_EN.md) | [中文](README.md)

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
resolved/downloaded by the shared Rust SDK.

This guide covers SDK `0.1.1`. See the [publishing guide](../../docs/publishing.md) for registry availability. The entry point remains `rapid-doc`; before publication, build and install the standalone package as described below.

The Windows x64 platform package is `@hzkitty/rapid-doc-win32-x64`, with `@hzkitty/rapid-doc-win32-x64-cuda` for CUDA. Other platform names stay the same, and the loader selects the package automatically. The publishing token needs write access to the scoped packages; using a scope does not guarantee exemption from npm spam detection.

Node.js 20+ is required. CPU platform packages cover Windows/Linux x64 and ARM64, and
macOS 14+ on Intel x64 / Apple Silicon ARM64. The loader selects the package
matching the running Node.js architecture. CUDA requires separate Windows/Linux acceptance testing.

## Inputs and results

`parse` accepts a local path, HTTP/HTTPS string or URL object, file URL, Buffer, Uint8Array or array of inputs. HTTP input is downloaded by the shared Rust SDK; pass the URL directly without `path.resolve()`.
Arrays / `parseAll` process files sequentially, reusing models. `parseReader`
accepts a binary Node readable or async iterable; it buffers the full input.
Input document URLs are downloaded by the shared Rust loader.

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
and its CUDA/cuDNN prerequisites; the published CPU runtime packages do not install them.

## Build and verify locally

Build a shared runtime first using `scripts/migration/build_go_sdk.py` (the
runtime libraries are shared across bindings), then:

```powershell
.venv/Scripts/python.exe scripts/migration/build_node_sdk.py --runtime-dir build/go-sdk/windows-x86_64-cgo --output build/node-sdk/windows-x86_64
```

Native acceptance tests use `RAPIDDOC_TEST_ROOT`, `RAPIDDOC_NODE_LIBRARY`, and
`RAPIDDOC_NODE_RUNTIME`; run `node --test bindings/node/test/*.test.js`.

Concurrent `parse()` calls share an engine and lazily reused workers. Office requests run independently; PDFium and individual model sessions retain their locks. `parseAll()` remains sequential. Use distinct output paths for concurrent writers and do not synchronously reenter parsing from a progress callback. `close()` cancels calls and stops all owned workers.
