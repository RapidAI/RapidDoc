# CUDA image compatibility and long-document auto routing

This stage keeps the existing model weights, thresholds and selection rules.
`rapiddoc-render` remains unchanged.

## Proven causes and changes

1. The Windows reference uses OpenCV 5.0.0 and the CUDA server uses 4.13.0.
   Their perspective cubic sampling differs. Added a native OpenCV 4 kernel
   with the original 1/32 coordinate quantization, Q15 coefficient tables,
   normalization and replicated borders. The existing OpenCV 5 path is retained.
   A real skew table crop previously differed in 4062 RGB channel values;
   the new kernel matches every pixel. All 262 additional independently generated
   border, skew and rotation cases match OpenCV 4.13 exactly.
2. Exact integer translations now copy pixel rows rather than interpolating.
   Borders still replicate the original edge pixels. The full 129-page output
   before/after this optimization is identical within the same sampling profile.
3. For one real DB probability map, Python and Rust had identical detector
   inputs and raw model outputs, but boxes differed by one pixel. OpenCV 4
   reflects the last two RotatedRect corners; OpenCV 5 evaluates them separately.
   A vertex 66.0 versus 65.999984 crossed Pyclipper's integer truncation boundary,
   changing recognition crop width 273 to 274 and subsequent sorted batches.
   Both corner construction modes are retained. Replaced the handwritten DB
   offset approximation with the already bundled original Clipper 6 core and
   compute polygon area/perimeter from all rounded vertices as Python does.
4. Python's public `auto` route classifies each processing window. Rust used
   to classify the entire selected range once. On the mixed 129-page corpus,
   the last single electronic page must switch to text. Added native per-window
   classification using the existing classifier; no new Python page loop.

`pipeline::Request.crop_sampling` explicitly selects `opencv4`/`opencv5` for
native callers (existing native default remains `opencv5`). The Python
compatibility wrapper selects the installed reference version without importing
the OpenCV extension; `ocr_config.crop_sampling` can override it. The same
thread-scoped mode controls crop sampling and RotatedRect rounding and restores
on errors/panics. `auto_classify` defaults false for existing low-level requests;
the high-level auto route enables it. Explicit txt/ocr routes remain explicit.

OpenCV implementation provenance:
https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/imgwarp.cpp
Full source notice is retained in `licenses/OPENCV-WARP.txt` and distributed
with the inference crate and wheel.

## Validation and measurements

- Native binding check passed; 12 affected integration checks across crop,
  DB detection, detection preprocessing and layout passed, plus integer-copy
  boundary equivalence and panic scope restoration. Python runtime/worker/notice
  checks: 15 passed.
- Native OpenCV 4 crop golds are under `tests/fixtures/opencv4_crop/`.
  The live DB probability map and Python intermediate geometry are under
  `tests/fixtures/detector_boundary/`.
- Before per-window auto classification, all 129 pages of explicit-OCR Python
  and Rust have identical complete snapshots. Against default-auto Python,
  Markdown and assets already match, with only the final page differing.
  Final auto-route regression is now complete: all 129 pages match the Python
  default-auto complete snapshot exactly, including every coordinate,
  confidence, text/formula/table field, Markdown and image asset snapshot.
- Integer-copy A/B run (same OpenCV 4 profile, one cold + one reused call):
  84.711/81.031 seconds before, 81.668/73.881 seconds after. Complete snapshots
  are identical. These single measurements are not a statistical speed claim.
  Earlier Python reused call was 64.318 seconds: an end-to-end gap remains.

## Final stage result

The final build preserves both OpenCV corner modes and classifies every native
processing window. Logs confirm OCR/OCR/text for the 64/64/1 windows. This fixes
the original nine differing pages without tolerances or output normalization.
Eight additional groups (13 selected pages) also match their complete Python
snapshots exactly: paper, formula, scan, dense table, mixed, seal, finance, report.
That is 142 checked pages including the synthetic long-document pages, not a
claim about arbitrary documents or every model/hardware combination.

| 129-page default-auto pipeline | Python baseline | Final Rust |
| --- | ---: | ---: |
| First call | 76.898 s | 80.408 s |
| Reused call | 64.318 s | 73.704 s |
| Sampled peak process RSS | 6.503 GiB | 3.181 GiB |
| Sampled peak process GPU memory | 8.043 GiB | 6.705 GiB |

Rust remains 14.6% slower on the reused long call, while RSS is about 51.1%
lower and GPU memory about 16.6% lower. Memory values are taken from the same
two-call reports as the timing values, not the earlier single-call baseline.
Versus the preceding Rust reused
measurement of 77.214 s, final time is about 4.5% lower. Single long calls are
not statistical medians. The six short speed groups have three reused calls
per group; their final Rust medians sum to 4.221 s versus the earlier Python
5.229 s (19.3% less time), with full snapshots unchanged. The Python short
baselines were captured earlier, not interleaved repeats in this final stage.
These short-page results do not establish long-document speed superiority.

Final Linux wheel/SDK: `batch-final-wheels/`, `sdk-batch-final/` in the isolated
server workspace. Local reports, full snapshots and Markdown are in
`build/cuda-validation-20261101/batch-final-records/`; intermediate records
(`cubic4-records`, `cubic4-fast-records`, `boundary-records`) retain the separate
sampling, copy-path and DB-boundary experiments. The real DB map, original
crop inputs and reference-generation probes are retained for reproduction.

The Windows release wheel is under `build/cuda-compat-wheels/` and the project
`.venv` is updated. This machine selects OpenCV 5 compatibility automatically;
the server selects OpenCV 4. No model weights were changed. Next performance
work should profile the long pipeline's CPU preprocessing/postprocessing rather
than infer speed from language choice. Candidate redundancy includes repeated
horizontal rows in cubic resize and repeated rows in nearest mask expansion;
these are inspection findings, not implemented or measured gains in this stage.

The isolated server SDKs/wheels, reports and logs are kept under
`/web/hzkitty/rapiddoc-cuda-validation/20261101`. Business environments, original
models and the two existing GPU service processes are not modified. Diagnostic
crop captures are excluded from performance claims.
