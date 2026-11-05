# Shared Rust SDK and Python adoption (2026-11-04)

## Decision

Add a single public `rapiddoc::sdk::RapidDoc` facade, re-exported at the crate root, over the existing PDF pipeline and six-format Office converter. Keep parsing kernels, model weights, thresholds and selection rules. The facade owns model selection, aliases, validation, resource discovery, optimized descriptors, warmup, output export and vector drawings. This is the entry future Java adapters should use.

`Config`, `ParseOptions`, `Resources`, `Document`, `ModelPlan` and typed ownership wrappers are shared Rust types. Configuration defaults match the current Python entry. Python retains host input/URL/image conversion, original argument/result compatibility, log presentation and writer handling. Python-specific facts remain in a small bridge: installed ORT package location, current Python environment, independent PDFium copy for pypdfium2 coexistence, NVIDIA package preload and installed-reference crop profile. These are not duplicate parsing/model selection policies.

## Changes and trade-offs

- Model plan generation and CUDA optimization/hash gating move from Python to Rust. Low-level diagnostic LazySession adapters use that same plan; production does not call Python model descriptors.
- Runtime search/override selection moves to Resources. Hosts pass bundle directories rather than implementing platform globs. Model resolution/downloads continue using the existing Catalog and verification.
- Warmup loads into the production registry. Relevant main/fallback models, orientation and seals follow configuration; TATR-only warmup excludes unrelated UNET/classifiers. Session identity includes optimization and CPU options. Reconfiguration keeps matching roles; changed roles are invalidated without discarding every session.
- Python production calls DocumentEngine for both PDF and Office. It no longer opens each PDF once merely to count pages. The PDF engine reports the source count with a document_opened event; the shared facade preserves Python's clamped end-page policy, with Python converting inclusive end_page_id once.
- Common output preserves original Office structs and PDF Vec image ownership until serialization, preventing an extra Office JSON tree and image copy merely for a unified result envelope. The final Python response still uses one JSON boundary; config/call options are small JSON messages. No model tensors or intermediate page processing cross Python.
- Existing raw PdfEngine/Office/diagnostic APIs remain for source-only migration tools. Enhanced order, Plus-L full parsing and other previously deferred functionality remain deferred. No simplified algorithms are added.
- Office remains lazy with respect to unused model settings and runtime assets. Callback exceptions are preserved. Explicit host capacity settings remain scoped; no new document size/token caps are added.
- Rust is usable in the local workspace. Publishing crates and packaging runtime bundles are distinct release work; no crates.io or Java publication is performed here.

## Validation

- Built/installed the Windows wheel in the project `.venv`; installed native and Python files match the final wheel/source.
- 66 affected Python tests and 5 direct Rust SDK integration tests passed, plus 3 subtests. Rust example compiled and executed; its DOCX Markdown matches Python.
- 108 old-runtime configuration snapshots match exactly: 72 model combinations and 36 CPU/session settings, including explicit-null overrides of global settings.
- Nine pre-change document outputs match: six Office formats, native PDF, scanned PDF and mixed model PDF. Compared Markdown, middle JSON, content list and image hashes. Shared vector drawing and output-writer tests passed.
- Verified preload and parse use the same DocumentEngine/loaded roles; repeat warmup emits no additional model resolutions. Threshold changes retain sessions; changed session options invalidate affected roles. TATR warmup excludes unrelated UNET/classifier roles.
- Reused the existing release-notice policy generator after the Cargo dependency/feature changes; target package set/selected notice counts remain 261 decisions and 53 files. No new external dependency was added.
- Final receipt and comparison data: `build/shared-sdk/receipt.json`. Sample times are not a controlled speed/memory benchmark and no speedup percentage is claimed.
- No full regression, CUDA execution, server synchronization, crates.io publication or Java binding in this change.
