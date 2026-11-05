# Shared Rust PDF debug drawing (2026-11-04)

## Decision

Move layout/span/line-order PDF overlays into `rapiddoc-pdf::drawing`, using the existing lopdf dependency. Other language bindings call the same Rust implementation. Python retains only the output/writer adapter. This avoids separate Java PDF libraries and duplicated selection/coordinate logic.

## Preserved behavior

- Vector source PDF contents, annotations, crop boxes and rotation; selected-page ordering.
- Original layout layer colors/order, alpha 0.3 fills, Helvetica 10-point numbering, table caption/body/footnote order, cross-page exclusions and inner image/formula outlines.
- Original polygon preference and coordinate rules for rotations 0/90/180/270, including nonzero crop origins. Span drawings use preproc blocks; layout drawings use para blocks.
- Line-order drawings select virtual lines and sort by index per page. The original Python multi-page generator capture bug is not copied: each page uses its own lines.
- Form XObjects isolate fonts/transparency/resources. Existing page resource names and sibling inherited resources are not overwritten.

## Integration

Production PDF parsing returns debug PDF blobs separately from image assets. Drawing occurs while page facts are still owned in Rust, without a Python page JSON round trip. Python writes the existing `_layout.pdf`/`_span.pdf` files through paths or the caller's writer. Standalone `_drawing` wrappers and line-sort output remain available.

The Python wheel no longer requires `pypdf` or `reportlab`. Original Python parser dependencies and development visual-reference tests remain unchanged. No additional Rust dependency is introduced; `rapiddoc-render` is unchanged.

## Validation

- Built and installed `build/native-drawing/wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl` in the project `.venv`.
- 45 affected tests passed (plus 3 existing subtests). No full/CUDA regression.
- 24 pixel-render comparisons against the original Python implementation passed: three drawing modes × four rotations × two crop-box variants. Tests retain vector text and check resource-name collisions. Comparisons permit only small PDF numeric/antialiasing differences (mean RGB error < 0.12/255 and fraction of channel errors > 10 below 0.001).
- Selected-page/inherited-resource comparison, distinct line orders across pages, preserved link annotations and empty-password encryption cases passed.
- Native drawing handles a malformed unused xref entry without invoking pypdf or changing its log level. Invalid ranges/coordinates remain explicit errors.
- Production directory and custom-writer tests block all imports of pypdf/reportlab and still generate both debug PDFs, without importing `_preview` or `_drawing`.
- All seven-format output baselines (Markdown, middle JSON, content lists and asset hashes) remain identical to `build/binding-refactor/before.json`.
- Removed the two dependencies from installed wheel requirements. They remain installed in the project environment for the original Python parser and reference tests.
- Build/validation receipt: `build/native-drawing/receipt.json`. No server synchronization.
