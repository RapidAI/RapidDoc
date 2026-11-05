# Rust model switching implementation plan

Goal: add Plus-L/V2 layout, FormulaNet Plus S/L, TATR recognition and optional docvortex_ocr rule tables without changing existing defaults.
Architecture: original model registry selects artifacts; Rust owns preprocessing, inference, postprocessing and rule fallback. Python only forwards model IDs/config. Match current Python adapters and retain conservative fallback guards.
Tech: Rust, ORT, existing image/Pillow/OpenCV-equivalent routines, native table recovery.

1. Inspect actual input/output metadata and Python pre/postprocessing; capture shared-input reference fixtures.
2. Generalize layout six/seven-column decoding, model-specific containment and ordering; preserve V3 masks and defaults.
3. Select formula S/M/L by concrete ID, retaining metadata-driven sizes/dictionaries and verified optimization guards.
4. Port TATR structure refinement/merged cells/OCR assignment, retain official thresholds, escaping and rowspans. Integrate into native production table dispatch.
5. Port docvortex_ocr raster rule extraction/deskew and atomic OCR text preparation to native Rust, reuse native recovery with physical-frame and unique-word-ownership guards; native PDF rules first, chosen model fallback.
6. Add affected reference comparisons and real model/PDF smoke checks, release wheel build/install .venv, document parameters and any validation limits. No full corpus or CUDA runs.

## Scope decision and validation

User confirmed: defer Plus-L complete PDF parsing because it requires the previously deferred XY-cut reading order. Keep original model download and independent detector support; reject complete PDF use clearly rather than inventing detector-order reading order. V2/V3 complete PDFs remain supported. Plus-L converter/XY-cut/formula-category handling is not claimed complete.

Implemented native FormulaNet Plus S/L selection, six/eight-column layout handling for Plus-L/V2, TATR structure recognition and docvortex_ocr recovery/fallback. No existing defaults or model URL/hash pins changed; registry adds original Plus-L pin.

Validation: 40 Python TATR structure references; 7 synthetic raster cases (gray/Canny bytes, deskew angles and extracted rules); native OCR-rule recovery/HTML/low-confidence fallback; 5 captured real raster tables (one accepted, four declined, matching Python including accepted HTML); 19 affected Python tests plus 3 subtests. Actual Python/Rust model comparisons: Plus-L 12 regions, V2 10, V3 9; S/L both recognized x+y=2 identically; real TATR crop 62 cells with matching logical spans/text. V2 and scanned small_ocr PDF smoke checks passed. The real notes_..._81 PDF used native TATR production dispatch and its table HTML exactly matched the previous Python TATR output. No full corpus or CUDA run; no performance claims from these checks. New variants use original ONNX graphs rather than unvalidated CUDA graph rewrites.

Artifacts: build/model-switch-acceptance/report.json, model-switch-native-references.log, model-switch-rules.log, model-switch-real-rules.log, model-switch-table-pdf.log. Acceptance scripts remain in scripts/migration/export_model_switch_references.py and accept_model_switching.py.

Final release wheel installed into the project .venv. Native Plus-L PDF guard and embedded 15-model catalog/notices verified after installation.
