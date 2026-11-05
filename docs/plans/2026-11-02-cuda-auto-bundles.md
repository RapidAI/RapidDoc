# Automatic prepared CUDA bundle discovery and deployment

> 当前实现已改为 [Rust 初始化时内存优化](2026-11-02-native-memory-models.md)。本文的磁盘模型准备/清单/安装流程属于历史方案，不再提供；性能与输出对照记录保留。

## Scope

Make already validated optimization graphs and native operator libraries usable
without `RAPIDDOC_CUDA_MODEL_MANIFEST` on every parse. Normal `RapidDoc(...)`
parameters and model/session computation remain unchanged. This stage does not
add ONNX rewriting to page parsing or a dependency on the original Python
parser at runtime. `rapiddoc-render` is unchanged.

## Selection

CPU never searches CUDA bundles. CUDA uses the explicit environment manifest
when supplied; that selection remains strict. Otherwise fixed candidate paths
are checked in order:

1. `cuda/manifest.json` beside the selected original model.
2. `cuda/manifest.json` under the existing configured/default model roots.
3. The installed Python package's `cuda/manifest.json`.
4. The user's installed active cache bundle, selected by `active.json`.

There is no recursive search through experiment output directories. Automatic
selection checks source-model SHA, version, optional platform/architecture,
ORT major/minor, prepared graph SHA, decoder SHA and operator-library SHA.
Custom source models are not replaced by mismatched built-in graphs. Missing,
invalid or incompatible discovered candidates are skipped with a once-per-
process reason; the original graph remains available. Explicit selections keep
strict failure behavior. Graph or runtime inference failures after session
creation are not swallowed and retried with altered models.

Operator libraries normally use manifest-relative paths. For a missing original
absolute library path, only an adjacent `lib/<basename>` or the native package
runtime library location is considered, and the same SHA must match. No
arbitrary library scan or unsigned replacement is used. Native session loading
still registers the selected operator libraries in Rust.

OCR `onnx_ops=False`, formula `engine_cfg.onnx_ops=False`, UNET
`engine_cfg.unet_conv_cache=False`, and layout `engine_cfg.mask_uint8=False`
retain CUDA device selection while bypassing prepared graphs for those roles.
SLANet raw-session and partition-disable contracts remain unchanged.

## One-time deployment, ordinary parse thereafter

To install an existing prepared release bundle in the current user's cache:

```sh
python -m rapid_doc_rust.cuda install /path/to/manifest.json
python -m rapid_doc_rust.cuda status
```

`install_cuda_bundle()` in `rapid_doc_rust.cuda` copies all graphs, the SLANet
decoder and the operator libraries into a self-contained immutable cache bundle.
The source directories may then be removed/moved. It verifies copied bytes,
rewrites only the deployment paths, and atomically switches `active.json` only
after success. A failed disk copy or corrupt source leaves the previous active
bundle intact. Reinstall is idempotent; corrupt cache entries are repaired by
publishing a new immutable directory, without mutating one used by parsers.
Original model files remain separate and are resolved by the usual model loader.
Native CUDA/cuDNN/ORT dependencies still come from the installed runtime.

Default user cache is `$XDG_CACHE_HOME/rapid_doc_rust/cuda` when XDG_CACHE_HOME
is set, otherwise `~/.cache/rapid_doc_rust/cuda`. It is user-specific. A bundled
`models/cuda/manifest.json` is another deployment option requiring no activation.
An explicit manifest environment variable overrides automatic discovery.

To prepare all validated graphs from original sources in a development or
release environment and install them in one operation:

```sh
python scripts/migration/prepare_cuda_bundle.py --output /new/output/directory
```

This combines the existing OCR/layout/SLANet preparation, UNET convolution
cache and formula MaxWithIndex preparation. Add `--library PATH` for an explicit
built custom-op library, or `--no-install` to prepare release files only.
Preparation requires ONNX, CUDA-capable ORT, the original Python graph utilities
and a built custom-op library. These development dependencies are not imported
by normal parsing or by installing a prepared deployment bundle.

A fresh installation with no prepared bundle falls back to original models,
with a diagnostic message. It does not silently promise that every CUDA
optimization is enabled, automatically compile CUDA code, download executable
operator libraries, or rewrite models on the first parse. The one-time release
preparation/import step is explicit; per-parse manifest configuration is removed.

## Verification

23 affected Python tests cover automatic/model-adjacent selection, complete
copy/relocation, decoder/library hashes, explicit strictness, source/platform
mismatch, malformed manifests, broken pointers, corruption repair, interrupted
installation, override/CPU and disable behavior. They pass. Windows wheel is
installed in the project `.venv`; installed attribution and 23-DOCX public API
checks including PDF text smoke and five IO variants pass. No full workspace
regression is claimed. GPU verification uses an isolated server cache and
unsets RAPIDDOC_CUDA_MODEL_MANIFEST; results are recorded below.


## Real deployment acceptance

On the isolated server, RAPIDDOC_CUDA_MODEL_MANIFEST is unset. A copied bundle
under `auto-cache/rapid_doc_rust/cuda/bundles/` automatically supplies all eight
production roles (including shared OCR/seal recognition); classifier roles keep
their original models. Both SLANet graphs and all required libraries resolve
inside this copied bundle. The original experiment paths are not required by
these installed manifest entries. Actual native parsing—not only descriptor
inspection—loads the graphs and custom operators successfully.

Default-auto 129 pages, two calls, complete snapshot and Markdown exact versus
the preceding Python reference; eight selected-page groups (13 additional pages)
also match exactly. Recorded time is 69.129 s first / 64.663 s reused, sampled
RSS 3.131 GiB and process GPU memory 6.717 GiB. This is deployment verification,
not a fresh paired speed claim. The two-call timing does not establish a new
performance regression or gain versus earlier separately sampled runs.

The final schema-hardening wheel adds one further scan check, also exact.
The complete `prepare_cuda_bundle.py` command is run in a second isolated cache
and installs automatically. Every prepared graph, SLANet decoder and custom
operator-library SHA matches the previously accepted bundle. `status` validates
the installed files. The one-command preparation is therefore tested with the
actual CUDA toolchain, not only mocked commands. Original models/business
Python environment and services remain unchanged.

Final server SDK/wheel: `sdk-auto-final/`, `auto-final-wheels/`. The activated
isolated cache is `auto-cache/rapid_doc_rust/cuda/`; independently reproduced
cache is `auto-built-cache/rapid_doc_rust/cuda/`. These are isolated validation
caches, not edits to the business environment. Local `.venv` has the matching
Windows wheel in `build/cuda-auto-final-wheels/`; local status correctly reports
no installed CUDA bundle on this CPU-only machine. It does not try to install
a Linux native library on Windows.

All three changed deployment source hashes match the server files. No Rust
model computation is modified in this stage. Full raw reports/snapshots,
descriptor routing, activation/deployment receipts, logs, run scripts and Linux
wheel are archived below; already archived model weights are not duplicated.


Local archive: `build/cuda-validation-20261101/cuda-auto-results.tar.gz`,
28,738,394 bytes, SHA256
`a6a3ea6534b705218e44173e4bec92e192407a90955fda7d46845eec4e799c27`.
It intentionally stores deployment metadata and verification outputs rather
than another copy of all unchanged model weights. Full standalone installed
bundles remain on the isolated server; their referenced source graphs were
archived in the preceding CUDA stages. Reports are extracted locally in
`build/cuda-validation-20261101/auto-final-records/`.
