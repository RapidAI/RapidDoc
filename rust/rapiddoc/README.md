# RapidDoc Rust SDK

Use the shared `rapiddoc` crate directly. The same SDK backs the Python binding; Java can wrap this API without copying model policy or parsing algorithms.

## ORT installation

For a Rust application, enable `download-runtime` to download and verify the official ORT 1.23.2 archive during Cargo build (Windows MSVC x64 / Linux glibc x64 / macOS Intel and Apple Silicon CPU):

```toml
[dependencies]
rapiddoc = { path = "../RapidDoc/rust/rapiddoc", features = ["download-runtime"] }
# For GPU use features = ["download-runtime", "cuda"] instead.
```

After crates.io publication the same features can be used with a version dependency (`cargo add rapiddoc --features download-runtime`). The feature remains opt-in: language bindings provide their own runtime distribution and do not trigger this download. No Python installation is needed. GPU downloads contain GPU ORT, but NVIDIA drivers and compatible CUDA/cuDNN must already be available.

Downloads are cached in Cargo build output and verified against pinned upstream SHA-256 digests. For offline builds, set `RAPIDDOC_ORT_ARCHIVE` to the same official release archive; its checksum is still verified. Explicit external library paths override bundled discovery. The installed bundle, including provider libraries and required notices, is copied to `target/<profile>/runtime` (also next to examples/tests). Ship that `runtime` directory alongside the executable; a build-machine absolute path is not a deployment substitute. Cleaning Cargo output removes this cache. This feature installs ORT only; PDFium/GEOS and optional CUDA custom operators still come from RapidDoc's platform runtime bundle.

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

`Config` defaults match Python: auto parsing, 64 pages per window, 200 DPI, formula/table enabled, V3 layout, M formulas, UNET+SLANet tables and no fallback model. Legacy configuration maps remain available as `layout_config`, `ocr_config`, `formula_config`, `table_config`, `image_config`; model aliases, validation and session settings are shared rather than reimplemented per language.

`ParseOptions.end_page` is exclusive. Python's inclusive `end_page_id` is converted at its adapter. `image_dir_name` and `draw_layout`/`draw_span`/`draw_line_sort` control output references and optional vector PDFs. Result fields include Markdown, middle data, content list, images and drawings; the SDK returns data without imposing a filesystem writer. `Middle` retains the typed Office tree or PDF tree until serialization; images retain their Vec/Arc ownership without an additional copy.

DOC/DOCX/XLS/XLSX/PPT/PPTX use the same `parse`/`parse_bytes` methods and initialize no PDF or model sessions. PDF inputs require the platform's release PDFium/ORT/GEOS libraries, plus CUDA operators when enabled. `RuntimeResources` accepts package directories or explicit library paths, model directory and a download switch. It does not depend on a Python installation or search a development checkout. Original model downloads, hashes and cache paths use the existing shared model store.

`warmup` and parsing use the same native registry. Only relevant configured models are warmed; TATR alone does not warm UNET/classifiers. `loaded_model_roles` supports verification. `configure` retains unchanged model bindings and invalidates only roles whose model/session settings changed. Resource and model failures propagate; requested CUDA is not silently replaced by CPU. Progress and optional host device preparation hooks are available through `parse_with_host`/`warmup_with_host`; ordinary Rust callers use `parse`/`parse_bytes`/`warmup`.

For native CUDA builds, enable the `rapiddoc/cuda` feature and provide GPU ORT/CUDA runtime libraries. Default builds are CPU; requested CUDA never silently falls back.

Local workspace example:

```powershell
cargo run --release -p rapiddoc --example parse_document -- tests/migration/inputs/synthetic.docx
cargo run --release -p rapiddoc --example parse_document -- document.pdf path/to/release/runtime
```

The crates.io package is `rapiddoc`; local first-release packaging is checked with `python scripts/release/crates.py --allow-dirty`. This command verifies the source archives without publishing. Native build prerequisites are a C/C++ toolchain, CMake and NASM (MSVC on Windows; GCC/Clang on Linux; Xcode Command Line Tools on macOS). Platform PDFium/GEOS/ORT libraries remain separate runtime resources; a successful Cargo build does not install them. See the repository publishing guide for the manual release workflow. No crates.io upload has been performed by this change.

`parse`/`parse_bytes`/`warmup` take `&self`: an `Arc<RapidDoc>` can serve concurrent requests. Office parsing is request-local. PDFium remains protected, and each shared model session serializes its own inference. OCR body preparation and decoding run outside that session lock without copying full logits. Configuration changes require `&mut self`; cloned engines preserve active configuration snapshots. Do not synchronously reenter parsing from a progress callback. Concurrent writers must use distinct output paths.

`engine.parse("https://example.com/report.pdf", &options)` also accepts HTTP/HTTPS input. The shared loader follows HTTP redirects, reports download events, propagates HTTP errors and uses 60-second connect/response/read timeouts. It reads into memory without a disk cache or a separate default size limit. Downloaded results include `metadata.source_name` for binding output names.
