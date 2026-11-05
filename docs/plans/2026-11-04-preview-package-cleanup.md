# Preview package cleanup (2026-11-04)

## Release boundary

Normal RapidDoc calls, results, drawing flags and existing CLI commands remain supported. The release package exposes 26 production/useful diagnostic names rather than 241 migration checkpoints. `_diagnostics.py` retains 15 definitions (session wrappers, page rendering/text/table/OCR helpers and their dependencies). Runtime warmup uses the same retained OnnxSession class.

`_preview.py` and `_native_pdf.py` are removed from the installed package. Historical algorithms/checkpoints now live in `scripts/migration/rapid_doc_rust_preview.py` and `native_pdf_preview.py`, excluded from the wheel. This is a release-size reduction, not deletion of all source-tree migration tools. Six wrappers with no callers or dependent helpers were deleted: `apply_native_geometry_plan`, `build_native_document_geometry_plan`, `prepare_native_document_containers`, `prepare_native_document_geometry`, `prepare_native_document_sources`, `prepare_native_page_marginals`.

Historical standalone tools/install tests explicitly activate `scripts/migration/preview_compat.py`. Migration pytest tests opt in through their local conftest. That developer adapter supplies old aliases only in the process explicitly invoking it; the installed package has no fallback or filesystem lookup for development sources. Fresh subprocess tests verify the real release interface independently. Remaining declarations/classes are shared to preserve type identity.

The old `parse_pdf_models` and other migration-only APIs are no longer release imports. Use the existing RapidDoc API for applications. Developer tools can opt in through preview_compat; Java does not need these migration APIs.

The obsolete large stage-compatibility report is removed from the wheel; its canonical copy stays at `packaging/compatibility-audit.json`. Source provenance moves to `packaging/compatibility-sources.json`. The old Python reference generator now writes under `build/migration/python-compat-reference` rather than overwriting the Rust-backed drawing adapter. CLI capabilities reports the production formats instead of the obsolete native-text-only checkpoint.

## Verification

- Installed wheel: `build/preview-cleanup/wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`; the project `.venv` is updated.
- Packaged Python code: 3,759 -> 1,172 lines (about 69% less). Additional copied compatibility report/provenance data is removed from the wheel.
- Native extension bytes match the prior native-drawing wheel; no Rust algorithms or model weights changed.
- 87 affected tests passed plus 3 subtests, including release exports/lazy loading, explicit isolated developer activation, shared drawing, model configuration, host scopes and table helpers. Follow-up release/developer boundary checks passed after removing the copied compatibility data.
- Seven-format Markdown/middle/content-list/image hashes equal the pre-cleanup baseline. CLI capabilities and isolated preview tool help work. Modified legacy consumers compile.
- Two historical standalone test attempts could not run because generated fixtures `native-table-text.json` and `geometry/forms-rot0.pdf` are absent. They are not counted as passes; their developer imports loaded successfully. No full regression/CUDA/server sync.
- Receipt/inventory under `build/preview-cleanup/` records removed/retained wrappers, updated consumers and wheel identity.
