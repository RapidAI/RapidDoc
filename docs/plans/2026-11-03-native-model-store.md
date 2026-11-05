# Native model catalog/store

Goal: Rust owns model IDs, paths, downloads and SHA256 checks; bindings pass parameters only.
Preserve existing default models, URL/hash pins, inference config, graph optimizations and parse output.
Canonical schema-versioned catalog is rust/rapiddoc-inference/models.json, with named models and separate default role mapping. No Python package/dev-path scanning or optimized disk model cache. Explicit model directory shares files across languages.

1. Rust catalog validation and directory/override/nested-file discovery; streaming HTTPS, unique partial file, hash verification, atomic publication and failure cleanup.
2. Thin Python native methods; remove Python downloading/path policy and obsolete manifest. Preserve model selection guards for unvalidated parsers.
3. Rust tests (catalog/default pins, cache reuse, nested directories, overrides, mismatched/truncated/failed downloads, concurrent publication) and affected Python runtime tests.
4. Build release wheel, install in current .venv, check a real download and existing PDF output with identical models. No full regression/CUDA run.
5. Record local directory configuration, manifest schema, unsupported pipeline status and new dependency notices.

Completed 2026-11-03: 7 native store tests and 13 affected Python tests passed.
Actual ModelScope orientation download verified its pinned SHA256; the second
resolution reused the file offline. All 10 existing models were resolved from an
explicit shared directory. The same PDF first page produced byte-identical
Markdown before/after. All 10 original URL/filename/hash pins were compared and
preserved. Release wheel built and installed into the project .venv; embedded
catalog and dependency notices checked in the wheel. No full regression or CUDA
run. Validation details: build/model-store-validation.json and model-store-*.log.
Usage and migration: docs/rust-models.md. Existing source-package weights require
an explicit shared directory or per-model path; no implicit Python scanning.

Catalog simplified: each model retains only filename, URL and SHA256. The ID
distinguishes variants; parser support is tracked in implementation/documentation
rather than duplicated metadata flags. Default mapping and all 14 artifact pins
remain unchanged.
