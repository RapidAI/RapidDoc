# DocVortex parsing core

Source: https://github.com/myhloli/DocVortex, 0.5.16, commit `f5da4e69022f5b0a935fb64579d7f0f7ec7d08b6`.

See `docs/migrations/docvortex-source-manifest.json` for source paths and original SHA-256 hashes. File-level copyright notices remain in place. The upstream project declares MIT, Apache-2.0 and BSD-3-Clause; its root MIT notice is in `licenses/`. Bundled DroidSansFallback font has its own `resources/fonts/NOTICE`; Unicode data retains its original notice.

Local changes: package imports/resource ownership are relocated; only PDF and six Office models are exported; computation uses the Python fallback; language/file classification delegates lazily to RapidDoc's bundled Magika. No external DocVortex, Magika distribution, fasttext or Rust extension is required.

Do not re-import over local fixes. Compare the pinned source hashes and review upstream changes explicitly.

Selective Office update (2026-10-30): four unmodified-from-base Python files
(`docx/docx_converter.py`, `docx/tables.py`, `xls/records.py`, and
`xlsx/xlsx_converter.py`) now follow DocVortex 0.6.8 commit
`2de5f42b1884495313d9df63868e6a2f3d85fea0`. This fixes #37 image source
identity and adds the upstream DOCX reuse, XLS record-view and XLSX
conversion-cache optimizations. Exact upstream/local hashes and retained
components are recorded in `OFFICE_UPSTREAM_068.json`. This is not a whole-package
upgrade. This paragraph describes the first Office step; subsequent #36 and PDF updates are recorded below. Sparse spreadsheet native batching is reserved for
direct Rust integration; its original Python reference algorithm is retained,
without approximations or an external DocVortex runtime dependency.

Local native table spacing fix: retain PDF advance bounds separately from visible glyph bounds when inferring spaces, using local font height rather than the whole-table scale. Low punctuation uses font bounds for initial row grouping while cell ownership and semantic geometry retain ink bounds. Explicit whitespace and inputs without advance geometry keep their existing behavior. See `tests/test_native_table_spacing.py` and `tests/test_native_table_rotation.py`.

Local PDF fix: strong vector graphic candidates yield to verified, text-occupied connected table grids. Grid verification uses the complete connected component for repeated data columns while requiring the candidate itself to span populated cells; merged headers and partial graphic crops no longer hide the table. Single-cell graphics, thin bands, missing borders/columns and Form containers remain outside this override. See `docs/migrations/docvortex-table-graphic-fix.md` and `tests/native_pdf/test_graphic_table_arbitration.py`.

Local legacy Office fix: per-image BLIP resource limits use a dedicated exception caught at DOC inline-image, shared DOC/XLS BStore and PPT image-map boundaries. Oversize images are skipped with warnings without renumbering later references; document record and aggregate-asset budgets still fail hard. Metafile decompression remains bounded without an unbounded flush. See `docs/migrations/docvortex-image-limit-fix.md` and `tests/native_office/test_office_image_resource_limits.py`.

Local native PDF fix: closed grids distinguish enclosing Forms from small cell icons. Embedded media primitives are masked only when a verified physical grid places every image in one cell; icons are rendered once per page and preserved in HTML. Leader-and-page-number TOC entries are excluded from repeated marginal matching before directory recovery. See `docs/migrations/docvortex-table-media-toc-fix.md` and `tests/native_pdf/test_table_media_and_toc.py`.

Local native PDF fix: short marginal lines adjoining local body text are protected using spacing, alignment, text scale and surrounding content geometry, without a label vocabulary or mandatory background fill. Partial-width row separators and a missing bottom segment no longer split a table whose continuous outer borders and repeated column tracks prove the complete region. Only the common track endpoint is closed; internal rowspans remain intact, and pure background rectangles cannot override the confirmed drawing grid. See `docs/migrations/docvortex-partial-grid-body-label-fix.md` and `tests/native_pdf/test_partial_grids_and_body_labels.py`.

Local native PDF fix: fragmented column strokes and multiple populated grids participate in table/graphic arbitration using union coverage. Weak rule candidates cannot expand confirmed grids over prose. Page-level leader evidence protects TOCs independently of heading language; connected raster-label diagrams survive fragment filtering, and plain text frames stay text. Sparse foreground text remains separate from large backgrounds, including unique marginal text, while grouped table preambles retain their first/changed values. RapidDoc obtains rule layout independently of OCR and carries native text ownership through final image-overlap filtering. See `docs/migrations/docvortex-layout-optimization.md` and `tests/native_pdf/test_reported_layout_regressions.py`.

Local native PDF fix: leading centered title groups establish body ownership before repeated-margin matching, using hierarchy, spacing, body scale and separator evidence rather than text vocabulary. Confirmed ownership survives split-row and rule-delimited header passes. Connected multi-line table preambles are retained on every page without deduplication or a page-top table cutoff; independent separators and whitespace prevent running headers from borrowing preamble evidence. See `docs/migrations/pdf-body-marginal-fix.md` and `docs/migrations/test_pdf_body_marginal_roles.py`.

Local native table fix: leading outer horizontal tracks can collapse endpoint/stroke duplicates under the existing width, glyph occupancy and physical coverage checks. Distinct nearest physical lines remain separate even when snap tolerance matches both tracks; uniqueness and alias-span rejection remain enabled. RapidDoc resolves native table defaults consistently during page preparation, model initialization and table recovery, including explicit rule disablement and forced OCR. See `tests/test_native_table_defaults_and_edges.py` and `docs/migrations/pdf-table-edge-defaults.md`.

Local native table continuation fix: export column tracks from the final full-width physical rule band rather than requiring every separator to cross merged headers. Keep the existing union-coverage threshold, require terminal contact, reject decoration-sized bands, and retain adjacent-page, complete-column alignment and intervening-content checks. Fragmented vertical rules contribute jointly. See `tests/native_pdf/test_trailing_table_grid.py` and `docs/migrations/pdf-table-terminal-continuation.md`.

Local native reading-order fix: a leading single-line centered band may precede its nearby section even when classified as ordinary text, provided a shared wide multiline body proves the reading flow. Reject competing bands, sidebars, earlier content, rotations and weak geometry; move only across the matching section and its supporting paragraph. Text, geometry and semantic classes remain unchanged. See `tests/native_pdf/test_leading_flow_order.py` and `docs/migrations/pdf-leading-flow-order.md`.

Selective follow-up (2026-10-30): #36 now uses the upstream 0.6.8 implementation
in Office errors, DOC image extraction, shared OfficeArt parsing and PPT images.
The local regression suite remains, with an additional BMP-header budget case.
`OFFICE_UPSTREAM_068.json` records all eight adopted Office files.

`PDF_UPSTREAM_068.json` records selected table text-quality fallback, hierarchical
parameter tables, rotated cell reading and script-conflict fixes. These are also
ported to the shared RapidDoc Rust processing layer. Local horizontal advance
spacing and grid/layout policies remain; original text is never deleted by the
quality check. All three private-use ranges are already supported in the active
span path, and the retained vendor text-content kernel is now aligned as well.
The initial caption/grid mismatch on catalog page 54 is fixed by retaining enclosed external annotations while preserving the complete-grid growth guard. The 328-page Python regression passes; installed RapidDoc range tests cover all 31 targeted pages with neighboring context. The legacy Rust whole-document diagnostic has a separate unresolved memory issue. Cross-page merging and diagonal watermark filtering remain under
separate review. No model or runtime dependency was replaced.

Optional raster table adapter (2026-11-03): `recover_native_pdf_table` accepts an
explicit prepared-text argument for the OCR adapter's atomic word boxes and
character-scale estimate. The original native path, text-quality guards,
hierarchical-cell safeguards and candidate thresholds are retained. Raster-only
acceptance/outer-frame and word-ownership checks live outside the vendor engine.

Selective 0.6.12 follow-up (2026-11-03): `UPSTREAM_0612.json` records the
0.6.9 banded-rowspan table recovery, 0.6.10 Office image-limit placeholders
and dependency boundaries, and 0.6.11 explicit NUL ToUnicode recovery.
The 0.6.12 uppercase continuation policy is adapted in RapidDoc's own
`backend/pipeline/para_split.py`, because the upstream raw model-list
postprocessor is not used here. The matching Rust processing paths are
updated as well; no upstream Rust binding or external DocVortex runtime
is required. Local layout/grid/spacing, model selection and fallback defaults
remain. RTF and upstream review tools are outside the supported scope.
This remains a selective fork, not a complete DocVortex package upgrade.

Host PDFium concurrency integration (2026-11-04): the vendored PDFium/font
runtime reuses `rapid_doc.utils.PyPDFium2Parser.lock` instead of constructing
another RLock. Both wrappers load the same pypdfium2 library, so separate locks
allowed overlapping native calls and reproducible heap corruption. Only native
critical sections are serialized; model inference and detached image processing
are not wrapped in a document-wide lock. Keep this host integration when
updating the vendored code. Regression: `test_shared_pdfium_lock.py` and the
recorded five-thread stress cases under `build/pdfium-concurrency`.

Selective 0.6.13–0.6.15 follow-up (2026-11-05): the PDF table text path now
uses the upstream ink-geometry calibration when inflated layout glyph boxes
merge independent text bands. The table merge path applies the upstream text
boundary rule across page cells while preserving inline HTML. Rust normalizes
out-of-range PDFium font weights before layout heuristics. The complete
DocVortex package is still not vendored wholesale, and the Rust font path does
not read embedded OS/2 tables yet.
