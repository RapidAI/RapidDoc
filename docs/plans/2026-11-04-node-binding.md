# Node.js binding

Implemented `bindings/node`, npm preview name `rapid-doc`, reusing the
shared seven-format Rust SDK through napi-rs. No parsing algorithms or model
scheduling were added to JavaScript. Node.js 20+; validated on Windows x64,
Node.js 24.12.0. Browser support is outside this binding.

Public calls return Promises. Each engine owns a worker isolate; native parsing
and inference execute there, preserving the main event loop. Paths, file URLs,
binary buffers, binary readable streams, lists, drawing PDFs, result export,
warmup/model reuse, async progress callbacks and AbortSignal are supported.
Readers materialize the complete input. Lists process sequentially. Cancellation
is cooperative at SDK events. A single engine rejects overlapping calls.

Progress defaults match Java/Go stage names, active processing seconds,
stdout status / stderr progress, completion deduplication, terminal refresh and
plain redirected output. Download notifications remain enabled with custom
callbacks or disabled parsing progress.

Initial lifecycle acceptance exposed an access violation after closing a worker
that used native rendering. The SDK's dependencies own process-global native
threads; unloading the addon when that isolate exits leaves their executable
code unmapped. The adapter now uses napi-rs's process-lifetime module retention
before constructing an engine. Engine close still releases sessions, documents
and the worker. The pinned code/global dependency threads remain until process
exit, consistent with long-lived native bindings. The failing multi-instance
sequence is retained in tests.

Linux installation acceptance found a second, independent lifecycle issue:
after the first worker exited, the next worker crashed in mimalloc's
`_mi_subproc -> mi_heap_main -> mi_thread_init` during N-API registration.
Changing its TLS model did not resolve the failure. Node now uses Rust's host
allocator rather than installing mimalloc as its global allocator. Module
retention remains necessary for process-global native dependency threads.
The regression creates, parses and closes eight successive engines; Linux
CPU and CUDA packages pass this alongside the other lifecycle tests. This
change does not alter Python's allocator or document parsing algorithms.

The runtime/macro/build crates are pinned together: napi 3.16.1,
napi-derive 3.7.0, backend 7.0.0 and napi-build 2.6.0. Older mixed
macro/runtime versions were rejected during compilation and are not shipped.

Packaging: `scripts/migration/build_node_sdk.py` verifies a prepared shared
runtime manifest, bundles addon, CPU ORT/private CRT, PDFium, GEOS and notices,
then creates a local npm tarball. Users do not need Rust, Python, Java or GCC.
Node-specific license texts are verified against crate source commits and merged
into the existing shared notice archive without adding a license file per crate.

Validation:

- Six affected test groups: seven formats match Markdown/middle/content/assets;
  drawings/export; buffer/stream/URL/list; main-loop heartbeat; native errors and
  recovery; original callback exceptions/reentry/cancellation/close; concurrent
  engines; startup errors; CPU OCR warmup and two PDF reference comparisons;
  progress completion dedup and format.
- TypeScript strict declaration and consumer compilation.
- Local npm tarball installed into `build/node-consumer`; native library override
  variables removed and PATH limited to Node/System32. Seven formats and scan
  OCR pass without development DLL paths. CommonJS and ESM imports pass.
- Evidence: `build/node-acceptance/tests.log`, `build/node-consumer/receipt.json`,
  `build/node-consumer/check.log`. No full project regression or GPU test run.

Not released to npm. Linux/CUDA package acceptance, Node 20/22 compatibility
runs, final platform package distribution and signing remain release work.
