# UNET production execution parity and CPU preprocessing

> 当前实现已改为 [Rust 初始化时内存优化](2026-11-02-native-memory-models.md)。本文的磁盘模型准备/清单/安装流程属于历史方案，不再提供；性能与输出对照记录保留。

## Root cause and corrected reference

The previous Rust table-model span included 5.645 s for 62 UNET cell calls.
A diagnostic captures the actual Python production table pixels, preprocessed
NCHW tensors and UNET class masks; 31 unique cases are found in the 129-page
corpus. All 31 native preprocessing tensors match Python byte for byte.

Python's UNET defaults to CachedConv CUDA execution, while Rust was using the
standard full ONNX graph. The earlier single-input CachedConv experiment was
rejected because nine class-mask values differed from the standard CUDA graph.
That was not the correct production reference: the original Python production
parser already uses CachedConv. The recorded difference is real, but it does
not demonstrate an accuracy loss versus the Python parser.

Actual native same-tensor inference, over all 31 varying production shapes:

| Pass through 31 cases | Standard graph | Python-production cached graph |
| --- | ---: | ---: |
| First | 1.590 s | 0.669 s |
| Reused 1 | 1.430 s | 0.377 s |
| Reused 2 | 1.407 s | 0.242 s |
| Reused 3 | 1.359 s | 0.245 s |
| Different class-mask values versus Python, each pass | 381 | 0 |

Each cached pass matches every raw uint8 class value of Python's production
reference. Model loading, file IO, input clone and CPU preprocessing are outside
the timed inference span. This is a microbenchmark, not the PDF speedup.

`prepare_cuda_unet.py` uses the existing Python production ORT preoptimization
and classic CachedConv rewrite, offline. Original weights and source files are
preserved. The graph and required operator-library hashes are checked by the
native session descriptor. No Python graph rewriting or per-table callback is
introduced at parse time. CPU and explicit `unet_conv_cache=False` keep the
original graph; a prepared bundle for the built-in UNET does not override a
custom UNET model. The prepared bundle is explicit via the existing
`RAPIDDOC_CUDA_MODEL_MANIFEST` mechanism.

## Exact CPU work reuse

Paddle table classification now caches eight horizontal Lanczos rows using
source-row keys. Integer coefficient accumulation, vertical accumulation,
center crop and float32 normalization are unchanged. Five boundary/up/downscale
cases are checked against the old uncached accumulation, plus every real
classification tensor below.

UNET morphology uses block prefix/suffix min/max instead of a deque per scan
line. The rectangular kernel, default anchor, neutral borders, closing order,
uint8 values and even/oversized-kernel behavior are unchanged. A brute-force
reference checks 216 combinations of image shapes, kernel widths, axes and
min/max operations, including all uint8 intensities in deterministic inputs.
This is the same morphological operation, not an approximate table algorithm.

Same 31 captured real cases; sum of per-case medians from five timed repeats:

| CPU span | Before | After |
| --- | ---: | ---: |
| Paddle classification preprocessing | 188.669 ms | 66.644 ms |
| UNET class-map resize and morphology | 488.976 ms | 403.299 ms |

All 31 classification tensors, UNET preprocessing tensors and full recovered
cell/topology JSON files remain byte-identical to the frozen native baseline.
The original Python reference is also retained. These CPU benchmark stages
exclude graph execution and are not full PDF timing ratios.

## Reproduction

```sh
python scripts/migration/prepare_cuda_unet.py \
  --base-manifest /absolute/path/table-bundle/manifest.json \
  --output /absolute/path/new-unet-bundle
export RAPIDDOC_CUDA_MODEL_MANIFEST=/absolute/path/new-unet-bundle/manifest.json
```

The isolated server bundle is `cuda-models-unet/manifest.json`. Diagnostic cases
are under `unet-real-cases/`; probes are `unet_post_benchmark` and
`unet_model_benchmark` in `experiments/onnx-probe/src/bin/`. Capture is opt-in:
`profile_pdf_pipeline.py --engine python --python-phases --capture-unet PATH`.
It adds diagnostic IO and must not be used for final speed measurements.

Local affected checks: 12 Python CUDA routing/configuration tests, 3 existing
classification tests, the independent integer row-cache oracle and the
brute-force morphology oracle passed. Full-pipeline acceptance follows below.


The initial full-pipeline Rust run succeeds: first 69.191 s, reused
65.327/64.534 s. Final full snapshot SHA remains
`0b485e45c01b515e12bdf0e40301383f1299a3cbf3861114afe6f7c0791bb48d`.
Sampled peak RSS 3.131 GiB, GPU 6.707 GiB. Final paired/control and selected-page
checks are recorded below. Windows wheel is already built and
installed in the project `.venv`; installed attribution and 23-DOCX public API
checks (including PDF text smoke and five IO variants) passed. CPU defaults
retain original graph loading. The exported CUDA graph uses the same FP32 ORT
fusions and TF32 execution policy as Python production; it adds no quantization
or reduced-precision weight storage.


## Fresh Python comparison

Both engines run sequentially in separate processes on the same shared RTX 4090
with one first call and two reused calls, default-auto, same 129-page source.

| Full pipeline | Rust | Fresh Python |
| --- | ---: | ---: |
| First call | 69.191 s | 72.597 s |
| Reused call 1 | 65.327 s | 63.510 s |
| Reused call 2 | 64.534 s | 63.737 s |
| Reused median | 64.930 s | 63.623 s |
| Sampled peak RSS | 3.131 GiB | 6.560 GiB |
| Sampled peak process GPU memory | 6.707 GiB | 8.055 GiB |

Full 129-page snapshots match exactly. Rust remains **2.1% slower** on the two
reused-call medians, with approximately 52.3% lower RSS and 16.7% lower GPU
memory. These are limited sequential measurements, not a universal performance
guarantee or a statistical distribution. Raw model microbenchmark gains must
not be substituted for overall PDF gains.


## Previous-SDK control, delivery and limits

Same-environment prior `sdk-pixel-fast` control with the preceding manifest:
first call 71.562 s; reused calls 68.178/67.768 s, median **67.973 s**. The new
**64.930 s** median reduces elapsed time **4.48%**. This control confirms a
larger improvement than the preceding pixel-only stage, but does not establish
universal performance across documents/hardware. The remaining observed
Python/Rust reused difference is approximately 1.307 s (2.1%).

Eight selected-page groups (13 pages: scan, seal, paper, dense table, mixed,
formula, finance, report) match the preceding full Python snapshots and
Markdown exactly. The 129-page new Rust/fresh Python/control snapshots also
match the accepted hash. Total unique test coverage is 142 pages including the
synthetic stress document, not extra independent documents for each repetition.

Final CUDA artifacts: isolated server `unet-final-wheels/`, `sdk-unet-final/`,
`cuda-models-unet/manifest.json`. Original models, business environments and the
two existing GPU services are unchanged. Windows release wheel is in
`build/cuda-unet-wheels/`, installed to the project `.venv`. No full workspace
regression or local GPU test is claimed. All seven relevant changed server
source hashes match local files.

Local archive: `build/cuda-validation-20261101/cuda-unet-results.tar.gz`,
88,914,229 bytes, SHA256
`cdf7b2ec11af86e2003a59cca8e071b6b3a883cf0a2446be7a9e51a373a00614`.
It contains the Linux wheel, prepared UNET graph/manifest, raw real-case pixels
and tensors, model outputs, old/new CPU outputs, reports, complete full-PDF
snapshots/Markdown, source hashes, run scripts and logs. Reports/snapshots are
also extracted in `build/cuda-validation-20261101/unet-final-records/`.
The bundle extends the previous prepared model/library locations; it is not a
self-contained distribution of all original model/runtime dependencies.
