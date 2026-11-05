# Python binding cleanup

The shared Rust SDK already owns production parsing, inference scheduling, models and drawings. The Python extension still registered 75 private migration checkpoints: intermediate tensors/geometry, legacy page assembly, duplicate Office entry points and the superseded PdfEngine adapter. Source-tree tests still use them, so deleting their implementations would discard useful regressions. They now live in `scripts/migration/python_native` and compile/register only with the explicit `rapiddoc-python/migration-tools` feature. Production bindings retain the existing public Python API and required environment adaptation.

Normal wheel builds do not enable that feature. To run historical migration checkpoints from a source checkout:

```powershell
scripts/migration/build_office_wheel.ps1 -MigrationTools -OutputDirectory build/migration-tools-wheels
# Or add --features migration-tools to the normal maturin build command.
```

Install the resulting development wheel into a test environment before running the checkpoint tests. `preview_compat.activate()` explains this requirement if a normal wheel is installed. Pytest's migration conftest activates previews only for an explicit development build. Default wheels support production regression tests without loading preview tools. Development builds are not release artifacts; the migration feature requires a source checkout, rather than a standalone binding crate.

Kept: RapidDoc constructor/call/warmup signatures and defaults; all 26 public names, including OnnxSession, PDF text/rendering, table matching and formula helpers; runtime discovery, optional CUDA preload, pypdfium2 isolation, progress and native drawing. These interfaces account for the remaining binding code. Merging them into one file would not reduce runtime work.

Removed: the unused `_compat._render_document_pages` wrapper and its import. Rendering has already moved into the shared SDK. Its historical capacity regression now exercises the actual diagnostic host scope, while callback tests exercise DocumentEngine instead of PdfEngine. The installed API acceptance script now observes the real DocumentEngine through a forwarding adapter.

No parsing algorithms, model weights/defaults, budgets, thread settings, or Java/native SDK behavior changed. Source movement alone is not a claim of improved parsing speed or accuracy. Acceptance compares public signatures and seven-format Markdown/middle/content/image hashes, checks optional checkpoint registration, exercises production OCR/drawing/progress tests and checks the opt-in extension against the original native interface snapshot. CPU/local validation only; no full dataset or CUDA run.

## Local validation

- Binding Rust source: 3,604 → 1,347 lines (14 → 11 files); development adapters moved to source tools, not discarded from the project.
- 75 private checkpoints absent from the default extension; the opt-in extension preserves all 104 original native names.
- All 26 public Python names and RapidDoc signatures/defaults preserved.
- 52 affected production tests and 33 development regressions passed.
- Seven-format reference outputs plus two default-model scanned/mixed PDFs matched Markdown, middle JSON, content lists and image hashes exactly (nine inputs total).
- Final release wheel installed into the project .venv; installed Python/native files checked against the wheel. The wheel contains no development source modules.
- No full dataset, CUDA tests, or performance benchmark. Migration tests that call private checkpoints require the explicit development build; ordinary parsing uses the normal wheel.
