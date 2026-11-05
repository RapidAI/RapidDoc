# Native process-memory model optimization (replaces disk CUDA bundles)

## Final direction

The user explicitly chose no optimized-model disk cache and no legacy-cache
compatibility. Original model files are reused. Rust reads the original ONNX,
performs the same validated graph edits in memory during session creation,
creates native ORT sessions, then releases temporary graphs. The live engine
reuses its sessions across documents. Multiple live engines share matching
native sessions via the existing registry; dropping all owners releases them.
This is not a permanent process-global model leak. A new process or a new
engine after all previous owners are gone may initialize again.

No cache install/activation API, active pointer, optimized-model manifest,
legacy `~/.cache/rapid_doc_rust/models` lookup or disk-decoder descriptor remains
in the production path. The five old `prepare_cuda*.py` bundle/deployment scripts
are removed. Previous build artifacts are not used for parsing and are not
silently deleted. The `cuda` module now reports the memory strategy/runtime
operator asset only; `install` is no longer a command.

## Native implementation

- A lossless ONNX protobuf wire editor keeps tensor payloads in shared Arc-backed
  slices. Unknown fields are retained; malformed lengths/varints are rejected.
  No Python ONNX package or original Python graph utility is imported at parse
  time. No new crate/network dependency is added.
- OCR: original FP32 CTC output contract is compacted where selected, ORT applies
  the original graph optimizations, and supported convolutions / reductions are
  rewritten to the existing CachedConv2D / MaxWithIndex implementations.
- Formula: the validated Loop decoder last-axis FP32 ArgMax rewrite is performed
  in Rust. Metadata and output token contract remain intact.
- V3: only proven binary mask chains gain uint8 output. The public output name
  expected by the native layout pipeline remains unchanged.
- SLANet: the original Loop input boundary is extracted in Rust; the existing
  1x3x488x488 crop contract is explicit. ORT folds constant boundary outputs in
  memory; they move into CPU decoder initializers. CUDA encoder and CPU decoder
  sessions are created from memory and retain borrowed OrtValue transfer.
- UNET: the original ORT preoptimization and classic CachedConv2D rewrite occur
  in memory, with the original tensor and FP32 execution policy.

ORT's Compile API can return its graph to an allocated memory buffer. A key
compatibility detail is that Compile API graph optimization defaults to disabled
independently of the ordinary SessionOptions setting. Explicitly applying
ORT_ENABLE_ALL is necessary to preserve the original Python fusions and
constant folding. The installed `ort-sys` exposes the stable v22 Compile API
prefix; a narrow native wrapper calls the appended v23 optimization-level slot.
The workspace already requires ORT API 23. Its ABI layout is checked, the
options lifetime is owned by ModelCompiler, and the compiled memory is freed
by its allocator. Source reference: [ORT v1.23 Compile API implementation](https://github.com/microsoft/onnxruntime/blob/v1.23.0/onnxruntime/core/session/compile_api.cc)

The first diagnostic run found tiny OCR/印章 differences when this independent
Compile API setting was omitted, and SLANet constants did not fold. Those
results were rejected. After the setting is corrected, ten captured production
model cases (including two SLANet captures and both classifiers) all match
reference raw outputs byte for byte. OCR indices **and scores**, detector maps,
layout masks/boxes, formula tokens, UNET labels and SLANet bboxes/probabilities
are checked. No tolerance is introduced. Model initialization measurements from
that standalone probe are approximately 0.30–0.44 s for OCR/UNET/SLANet,
1.10 s for layout and 3.55 s for the original large formula model. These are
model-load probes, not whole-parser startup or cross-platform guarantees.

## Original model paths

Existing explicit model paths and RAPID_MODELS_DIR/RAPIDDOC_MODEL_DIR remain
supported. Existing Python resources and per-family `rapid_layout_self/models`,
`rapid_formula_self/models`, `rapid_table_self/models` directories are searched.
Downloads use the configured directory when present. With a writable installed
original Python package, they use its matching resource/per-family location.
Standalone/read-only installations use `~/.cache/rapid_doc/models`. The old
Rust-specific cache is not searched or migrated. All downloaded source files
still require the expected SHA; explicit custom model paths retain their
original graph, rather than being silently transformed as known built-in models.

## Native operator assets

Custom operator binaries are still necessary to execute optimized nodes. They
are shipped as runtime assets under `rapid_doc_rust/runtime/onnx_ops` with their
third-party notices; they are not generated model cache files. An explicit
RAPIDDOC_OPERATOR_LIBRARY/ONNXRUNTIME_OPS_LIBRARY override is supported. Normal
loading does not require the original Python parser installation. CPU requests,
unsupported custom graph hashes and explicit per-model disable flags keep the
original graph. Raw OnnxSession APIs retain their original full-model contract;
production pipeline descriptors request the native memory transformation.

Local affected Python tests and native cache/protobuf tests are run; complete
GPU/PDF and package verification results follow below. No full workspace
regression is claimed.


The initial complete memory-only implementation passed all 142 checked pages,
but its initialization-inclusive RSS peak was 4.927 GiB. The large formula graph
was repeatedly materialized while replacing nested protobuf messages. This
intermediate result is not the delivered final memory figure. The editor was
then changed to Arc<Vec<u8>> ownership with shared raw tensor slices, deferred
nested message trees, and one pre-sized streaming serialization. Tensor bytes
are not re-copied into every parent message. A nested-edit regression asserts
shared storage identity, byte preservation, exact encoding length and round-trip
of the changed graph. Final memory measurements follow below.

## Final delivery verification (2026-11-02)

The final reduced-copy implementation passed the complete 129-page corpus and
8 groups / 13 additional selected pages: full snapshots, Markdown and embedded
asset contents match the accepted original Python references exactly. The long
snapshot SHA256 remains
`0b485e45c01b515e12bdf0e40301383f1299a3cbf3861114afe6f7c0791bb48d`.
All 10 captured model cases also match raw output bytes exactly, including OCR
scores, layout masks and both SLANet captures. Native session reuse is asserted.
A separate installed-package parse forbids importing Python `onnx` and
`rapid_doc`: Markdown remains exact and no optimized ONNX file is generated
under a fresh cache root. Original model files are not changed.

The final 129-page CUDA run took 71.422 s for the first call and 63.239 s for
the reused engine. Initialization-inclusive sampled RSS peaked at 3.800 GiB
(4,080,721,920 bytes), down from the rejected intermediate memory-only
implementation's 4.927 GiB; process GPU memory peaked at 6.682 GiB
(7,174,356,992 bytes). Sampling interval is 50 ms, includes imports, engine
initialization and parsing, and excludes snapshot serialization. The earlier
accepted prepared-file implementation had a 3.128 GiB RSS peak: memory-only
initialization still has a transient graph-processing cost. Its retained warm
sessions avoid repeating this work for every document. The earlier Python warm
median was 63.766 s; this new single warm observation is consistent with parity,
not evidence of a significant speedup or a newly paired benchmark.

Affected checks: 11 Python configuration/download tests, 2 protobuf tests,
5 model-validation checks and 1 obsolete-descriptor/cache check passed;
`cargo check -p rapiddoc-python` passed. Final Windows wheel is installed into
the project `.venv`: attribution validation and 23 DOCX reference documents,
public IO/writers, no-model/no-Python-Office checks and PDF text smoke passed.
The wheel contains the two platform operator libraries and their four original
notices, and no ONNX model. Installed runtime configuration matches source.
No full workspace regression is claimed.

Server final SDK: `sdk-memory-owned-final/`. Local Windows wheel:
`build/cuda-memory-final-wheels/`. Final reports, raw model outputs, logs, source
hashes, reproduction scripts and Linux wheel are archived as
`build/cuda-validation-20261101/cuda-memory-results.tar.gz`, with a SHA256
receipt beside it. Selected readable reports are retained under
`build/cuda-validation-20261101/memory-final-records/`.

The former CUDA bundle install/activation API, manifest discovery, Rust-specific
model cache lookup, graph preparation scripts and disk decoder descriptor are
removed. Old files already present on disk are not automatically deleted or
migrated. Original-model downloads still use shared Python-compatible locations;
custom operator libraries remain required runtime assets.
