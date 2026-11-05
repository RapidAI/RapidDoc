# Python binding boundary cleanup (2026-11-04)

> Follow-up: migration-only exports and `_preview.py` were subsequently removed from the release package. See [package cleanup](2026-11-04-preview-package-cleanup.md); the 241-export preservation below records the earlier intermediate step.

## Decision

Keep the existing Python RapidDoc API and explicit diagnostic APIs. The shared Rust engine owns PDF final export and Office format dispatch. Python owns host input adaptation, logging, output writers and optional debug drawings. Other language bindings can reuse the shared operations without copying Python scheduling or Office dispatch.

## Changes

- The default package entry imports only native identity/error and the small PDF input/text adapter. The original preview helpers live in `_preview.py` and load only when requested. All 241 existing public exports remain available, including `from rapid_doc_rust import ...` and wildcard imports. `dir()` lists public exports without loading diagnostics. This is mainly separation, not removal of all those lines.
- Production `PdfEngine.parse_document` runs parsing, custom image-prefix rewriting, Markdown and content-list export within Rust. The owned page tree is moved into one rendering request and reused by both exporters; no clone is needed for constructing that request. Existing renderer rules and Magika behavior are retained.
- Removes two complete Python page JSON encodings and two Rust page JSON decodings per PDF document. Progress is passed as Python dictionaries, avoiding one encode/decode pair for each progress callback. Final metadata encoding runs with the GIL released.
- `PdfEngine.parse` keeps its previous raw-page result and JSON-progress contract for diagnostic scripts and tensor capture.
- Six-format Office dispatch and export now live in `rapiddoc_office::convert::parse_document`. Python wrappers only decode options, release the GIL, translate errors and expose bytes. Empty-input and legacy entry format gates are preserved.
- Production export explicitly uses the host resource scope, retaining caller-supplied limits without imposing diagnostic page/node limits.

## Remaining boundaries

One final metadata JSON response, the small parse configuration and model descriptor callbacks still use JSON. Runtime discovery and original Python configuration mapping remain in Python. Drawings already load on demand; dependencies are unchanged in this iteration. No speedup percentage is claimed from this structural cleanup.

## Validation

- Built and installed `build/binding-refactor/wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl` into the project `.venv`.
- 31 affected Python tests and 3 targeted Rust integration tests passed (including export equivalence, error restoration and production capacity behavior). Old fake-engine fixtures were updated for the production entry and existing fallback-table config field.
- Nine document outputs matched their pre-change baselines exactly: six Office formats, native PDF, scanned PDF and mixed PDF. Compared Markdown, middle JSON, content list and image SHA256 hashes.
- Eight Office empty-input policies/errors matched; layout/span PDF drawings and output writing passed.
- Production PDF/Office calls do not load `_preview` or `_drawing` unless requested. All 241 public diagnostic/compatibility exports remain accessible.
- Receipt and before/after artifacts: `build/binding-refactor/receipt.json` and sibling JSON files. Sample timings are not a controlled benchmark and do not establish a speedup percentage.
- No full regression or CUDA inference run; no server synchronization.
