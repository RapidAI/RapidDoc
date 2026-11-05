# CUDA table pixels and layout mask reuse

Continue the measured performance work without changing models, thresholds,
region order, table arbitration, text or output formatting. Keep `rapiddoc-render`
and server business environments unchanged. The implementation removes proven
lossless serialization/repeated pixel work rather than adding document rules.

## Baseline phase diagnosis

`profile_pdf_pipeline.py --python-phases` now wraps the production Python
layout/formula/OCR/table methods and table detection/recognition/predict paths.
The standalone diagnostic uses the same 129-page input and previous accepted
CUDA bundle. Cold/warm times: 73.623/64.058 s. Warm spans:

| Python stage | Seconds |
| --- | ---: |
| Layout | 5.375 |
| Formula | 17.630 |
| OCR detection/crops | 12.471 |
| Table, including text and prediction | 9.611 |
| Table text detection (nested) | 1.417 |
| Table text recognition (nested) | 3.617 |
| Table predict (nested) | 4.530 |
| OCR recognition | 7.262 |
| Page assembly | 4.062 |

The previous Rust accepted warm spans were layout 8.119, formula 17.277,
OCR detection/crops 10.147, table 17.515, table text 6.596 and table models
9.271 s. These separate runs locate candidates, not statistically proven
per-stage overheads. In particular, Python materializes some region crops
lazily inside OCR/table paths; Rust reports separate eager region-crop time.
Do not subtract every similarly named span as an identical boundary or add
nested rows twice. Profiling is excluded from final acceptance speed ratios.

## Implementation

- Table masking returns the original validated RGB pixel buffer to the native
  detector, removing PNG compression followed immediately by PNG decompression.
  The RGB-to-BGR conversion, detector preprocessing, DB policy, formula cutting,
  detection ordering, thin-image white-padding retry and restoration remain.
  Public PNG diagnostic adapters retain their outputs and input validation.
- Nearest-neighbor layout mask expansion precomputes source columns and copies
  a completed row when the next output row samples the same source row. It
  preserves the exact reciprocal FP64 coordinate arithmetic and truncation /
  integer modulo conversion, including nonbinary mask tensor types.
- Layout contour padding copies rows into the same zero-bordered GrayImage;
  contour selection and simplification are unchanged.
- Existing native stage summaries gain table payload, classification, UNET
  cells and SLANet structure spans. They are nested under table_models.

Local checks cover independently expected clipped-mask RGB bytes, white retry
borders, malformed inputs and the preexisting padding-error priority. Fifteen
resize/crop/target combinations cover up/downsampling, singleton regions,
nonzero tensor offsets and nonbinary float values; compact uint8 versus float
mask equivalence is also checked. Full CUDA acceptance is recorded below.


## Full acceptance

New Rust and fresh Python each run sequentially in an isolated process with one
first call and two reused calls. Same model bundle, source and default-auto
129-page pipeline; source SHA remains
`7417012be5b90cbe79c13c88a8d3d21e2626f9e15af202ead127cf31823fea01`.

| Three-call report | New Rust | Fresh Python |
| --- | ---: | ---: |
| First call | 71.285 s | 77.176 s |
| Reused call 1 | 67.956 s | 63.849 s |
| Reused call 2 | 68.336 s | 65.198 s |
| Reused median | 68.146 s | 64.523 s |
| Sampled peak RSS | 3.155 GiB | 6.571 GiB |
| Sampled peak process GPU memory | 6.688 GiB | 8.049 GiB |

Rust is still 5.6% slower on these reused medians. RSS is approximately 52.0%
lower and GPU memory 16.9% lower. Two reused calls are limited evidence on a
shared server, not a broad statistical distribution or a universal speed claim.
The previous single reused measurement was 69.280 s; a same-environment old-SDK
control is also measured before interpreting the small improvement.

Full 129-page snapshots and Markdown are exact versus fresh Python and the
preceding accepted snapshot. Eight additional selected-page groups (13 pages)
also match the preceding complete Python snapshots and Markdown exactly. No
normalization or output tolerance is introduced. These are 142 checked pages,
including a synthetic long document, not 142 independent client documents.

Local compile checks, the three affected pixel/mask unit tests and installed
attribution passed. The installed 23-DOCX public API/IO test (including PDF text
smoke) also passed. Old generated `build/migration/table-text.json` and
`layout-polygons.json` fixtures are absent locally; their old tests are not
claimed as passed. No full workspace test was run.

Windows release wheel: `build/cuda-pixel-wheels/`; installed into the project
`.venv`. Isolated server wheel/SDK: `pixel-fast-wheels/`, `sdk-pixel-fast/`.
All six changed server source hashes match local files. The previous table
partition/weights/CUDA bundle remain unchanged. Original models and server
business environments are untouched.


## Old-SDK control and conclusion

An additional sequential process uses the preceding accepted `sdk-table-final`
with the same input, model manifest and three-call protocol. First call 71.880 s;
reused calls 67.945/69.076 s, median **68.510 s**. New median **68.146 s** is
only **0.53% lower**. This is a small change comparable to observed variation;
it is not evidence of a substantial end-to-end speedup. The older 69.280 s
measurement alone would overstate the improvement. The accepted optimization
removes repeated lossless work, but the remaining bottlenecks dominate.
The control's full snapshot matches the same accepted SHA as both new runs.

New Rust final reused nested table spans: classification 1.124 s, payload
0.267 s, SLANet structure 1.874 s, UNET cells 5.645 s for 62 items. The UNET
span includes preprocessing, inference and line/topology recovery; it is not
pure model inference. Further work should split that span and measure matching
Python UNET stages before choosing an implementation change. Do not attribute
this span to ORT or change weights/precision based on the aggregate time.

The complete final records are archived at
`build/cuda-validation-20261101/cuda-pixel-results.tar.gz` (30,813,911 bytes,
SHA256 `d351cf609bc3708804647ec9d58faf6c37d67146c5d6ed6af57352c5e0f3f6ff`).
Reports/snapshots/Markdown and source hashes are also extracted under
`build/cuda-validation-20261101/pixel-final-records/`. The archive contains the
Linux wheel, raw diagnostic phase report, run scripts and logs, and previous
SDK control reports. GPU processes after all checks remain the two original
services (675258 / 2531609); validation processes exited.
