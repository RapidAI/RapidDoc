# Optional raster/OCR table rules

Goal: opt-in docvortex_ocr routing in the current Python parser, retaining defaults and selected-model fallback.
Reuse the reviewed adapter and current native table recovery core; no replacement table algorithm, new model, or Rust change.

1. Reproduce source tests and inspect real OCR/line/cell diagnostics.
2. Native PDF rules first; reuse existing OCR/orientation; delay table model initialization until fallback.
3. Accept only physically verified raster line grids, not text-only candidates. Preserve all OCR tokens, require unique cell ownership, and keep formula/image cases on the model route.
4. Test merged/empty cells, broken/decorative lines, skew, low-confidence/malformed OCR, routing, failure, lazy loading and debug IO failure. Preserve the original global use_word_box=False default and explicit overrides.
5. Run affected table tests plus fresh-OCR crop tests and public scan/native PDF calls; document observed limits, without lowering thresholds or changing defaults.

Completed: OpenCV 4/5 Hough compatibility fixed; prepared-text argument added
without replacing the current vendor engine or its quality/nesting checks.
Reject text-only grids, unobserved outer frames, and omitted/cross-cell OCR words.
171 affected tests passed. Five fresh-OCR crops accepted table_04 only (32x5,
144 cells); three public calls verified native-first, raster acceptance and
selected-model fallback. No default switch, weights change, CUDA or corpus run.
