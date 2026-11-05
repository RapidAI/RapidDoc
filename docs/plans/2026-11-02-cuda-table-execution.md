# CUDA table execution and cubic row reuse

> 当前实现已改为 [Rust 初始化时内存优化](2026-11-02-native-memory-models.md)。本文的磁盘模型准备/清单/安装流程属于历史方案，不再提供；性能与输出对照记录保留。

## Change and evidence

The Python production SLANet adapter partitions the original model into a CUDA
encoder and CPU recurrent decoder. Rust previously ran the entire graph on CUDA.
The earlier full-graph same-input benchmark therefore measured matching ORT
calls, but did not represent the Python production table execution plan.

A captured real table tensor gives 290.612 ms for full CUDA and 31.729 ms for
Python split execution (10 measured calls after 3 warmups). The native split
probe gives 38.817 ms. Its output names, shapes and float32 bytes match the
original Python production split exactly. These measurements are from separate
runs, not an interleaved speed guarantee. Full-CUDA versus split raw floats are
slightly different; the acceptance target is the original Python production
split, not the old full-CUDA control graph.

`prepare_cuda_tables.py` prepares portable ONNX files offline. Learned weights
and original source files are preserved. The existing single-crop 1x3x488x488
contract is explicit in the encoder; six constant cut tensors become decoder
initializers. This also avoids an ORT 1.23 output-memory-info alias failure for
constant encoder outputs. It does not export ORT's machine-specific optimized
graph. Both graph hashes and the original source hash are recorded and checked.

Rust owns both sessions and passes borrowed encoder OrtValues into the CPU
decoder. There is no per-crop Python callback, graph rewriting, JSON tensor
round-trip, or intermediate float-vector clone. Decoder ownership and hash are
part of the shared model cache key. Raw OnnxSession/session() calls keep the
original complete-model contract. CPU and custom table model overrides keep
their original execution path; an explicit `slanet_partition=False` disables
the prepared plan.

Cubic float resize now caches four horizontal rows, keyed by source row and
boundary accumulation order. Floating-point operation order is preserved.
Three standalone tensors match the preceding native implementation byte for
byte, including boundary rows:

| Source to 800x800 | Before | Cached rows |
| --- | ---: | ---: |
| 1600x2200 | 21.347 ms | 15.723 ms |
| 800x800 | 21.250 ms | 8.250 ms |
| 400x600 | 21.245 ms | 7.112 ms |

The 18-case OpenCV fixture records 15 exact reference cases and three
**pre-existing** tiny-image enlargement discrepancies (cases 2, 5, 8). Those
three retain both the OpenCV reference hash and the unchanged old native hash;
they are not silently accepted with a tolerance or claimed as OpenCV matches.
This stage does not fix those prior discrepancies.

An UNET CachedConv experiment was rejected: 5.491 to 5.238 ms was a small gain
and nine mask values changed by one. Production UNET remains unchanged at this stage.

Follow-up correction: this experiment compared against the standard full-CUDA
control graph. The Python production parser already uses UNET CachedConv.
[The subsequent production-reference audit](2026-11-02-cuda-unet-production.md)
finds that 31 actual cached graph outputs match Python exactly, whereas the
standard graph differs at 381 class-mask values. Therefore the earlier control
difference alone is not a valid reason to reject the production cache path.

## Reproduction

Using the isolated CUDA validation SDK/environment, extend an existing bundle:

```sh
python scripts/migration/prepare_cuda_tables.py \
  --base-manifest /absolute/path/previous/manifest.json \
  --output /absolute/path/new-table-bundle
export RAPIDDOC_CUDA_MODEL_MANIFEST=/absolute/path/new-table-bundle/manifest.json
```

Alternatively add `--partition-tables` to `prepare_cuda_models.py` when creating
a new full bundle, alongside the existing detector/CTC/layout preparation flags.
Preparation needs ONNX and the original Python partition utility; parsing uses
the prepared manifest and native Rust sessions. Originals are not overwritten.
The tested server bundle is `cuda-models-tables-v5/manifest.json`, under the
isolated `/web/hzkitty/rapiddoc-cuda-validation/20261101` workspace.

## Verification

Local affected checks: 11 Python runtime configuration tests; 3 layout tests
(including all 18 tensor fixtures); 5 model validation tests; 1 native decoder
cache/hash/output test. All passed. The hardware-FMA/scalar layout unit test also passed. The native real-tensor probe additionally
checks raw outputs against the original Python production split.

The paired 129-page default-auto run succeeds on both engines. Full snapshots
and Markdown are exact (snapshot SHA256
`0b485e45c01b515e12bdf0e40301383f1299a3cbf3861114afe6f7c0791bb48d`),
including the original nine previously differing pages. No output tolerance,
text normalization, or document-specific rule is introduced.

| 129 pages, two calls per process | Rust | Python |
| --- | ---: | ---: |
| First call | 74.054 s | 72.898 s |
| Reused call | 69.280 s | 63.507 s |
| Sampled peak process RSS | 3.161 GiB | 6.526 GiB |
| Sampled peak process GPU memory | 6.684 GiB | 8.035 GiB |

Compared with the preceding Rust 73.704 s reused call, runtime drops 6.0%.
Against the freshly measured Python reused call, Rust is still 9.1% slower.
RSS is 51.6% lower and GPU memory 16.8% lower. These are sequential two-call
measurements on the shared RTX 4090 server, not statistical long-run medians.
The original background GPU services continue to run. No claim of universal
Rust speed superiority or arbitrary-document precision is made.

The Rust reused-stage log still reports formula 17.277 s (inference 15.975 s),
table 17.515 s (model portion 9.271 s, table text 6.596 s), layout 8.119 s,
OCR detection/crops 10.147 s and recognition 6.352 s. Sub-stages overlap their
parent categories and must not be added twice. They identify remaining work,
not direct Python/Rust stage comparisons; the Python log lacks matching spans.


Final CUDA SDK/wheel: `sdk-table-final/`, `table-final-wheels/`. Eight groups
(13 selected pages: paper, formula, scan, dense table, mixed, seal, finance,
report) match the preceding complete Python snapshots and Markdown exactly.
Combined with the synthetic long test, 142 pages are checked. Selected-page
Python references are from the preceding stage, not fresh paired speed runs;
only the long document has a fresh paired Python timing in this stage.

Relevant 11 server source hashes match the local files. Windows release wheel
`build/cuda-table-wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl` is installed
in the project `.venv`. Installed attribution and the 23-document DOCX public
API checks (including PDF text smoke and five IO variants) passed. No full
workspace regression or local CUDA test is claimed. `rapiddoc-render` is
unchanged. Server business environments and original models are untouched.

Reports, complete snapshots/Markdown, preparation/probe evidence,
prepared table graphs, relevant source hashes and the Linux wheel are archived
locally at `build/cuda-validation-20261101/cuda-table-results.tar.gz`.
