# Remaining migration overhead: formula graph and cubic SIMD

> 当前实现已改为 [Rust 初始化时内存优化](2026-11-02-native-memory-models.md)。本文的磁盘模型准备/清单/安装流程属于历史方案，不再提供；性能与输出对照记录保留。

Continue the original Python behavior/precision contract; no new document rules,
thresholds, quantization or model weight changes. `rapiddoc-render` is unchanged.

## Confirmed differences

Python production `create_formula_session` uses a MaxWithIndex replacement for
the decoder FP32 ArgMax. The Rust production descriptor still selected the
original graph. The old same-model benchmark tested a prepared captured graph
in Rust and therefore did not prove the production descriptor was using it.
`prepare_cuda_formula.py` now extends the prepared CUDA bundle offline with the
same original Python rewrite and required operator library. The prepared graph
SHA equals the captured Python production graph SHA exactly. No runtime graph
rewriting is introduced. Explicit `onnx_ops=False`, CPU execution and custom
source models retain the original graph; source/graph/library hashes are checked.

Same captured formula input, ten timed calls after three warmups: original
44.911 ms, original-Python MaxWithIndex graph 41.178 ms. Both outputs match the
captured Python reference byte for byte. This is one shape/input, not a blanket
formula accuracy or overall speed guarantee; full pipeline checks are below.

Layout preprocessing was functionally matching Python on the three measured
inputs but substantially slower: Rust scalar cubic 15.814/8.194/7.135 ms versus
OpenCV Python 3.771/2.904/3.665 ms for source sizes 1600x2200, 800x800, 400x600,
all resized to 800x800. All three NCHW tensors match Python exactly. This is a
scalar implementation cost, not a change of preprocessing semantics.

The existing FP32 cubic kernel now uses runtime-dispatched AVX2/FMA/SSE4.1.
Horizontal RGB channels and vertical rows use explicit SIMD with the original
coefficient formulas, FP32 FMA grouping, boundary grouping and ties-to-even
rounding. Source indices are precomputed once; bounded final RGB reads and
scalar tail handling preserve short rows. Unsupported CPUs retain the existing
scalar/FMA fallback. No global compiler architecture flag is required.

New timings: 9.415/4.406/3.693 ms, 40.5%/46.2%/48.2% lower. All full tensors
are byte-identical to the previous native baseline, including unchanged prior
small-image limitations described in the earlier cubic fixture record. The
layout resize primitive is also used by UNET, so the captured real UNET tensors
are checked separately before delivery.

## Other inspected differences

Python layout uses per-call CUDA IO Binding and copies results to CPU. Rust
currently uses ordinary session run with borrowed CPU output views. Both paths
still transfer the outputs for CPU geometry, but this does not establish equal
transfer overhead. IO Binding is a remaining experiment, not a proven bug or
omitted performance gain. No unmeasured change is made to that path this stage.

Existing OCR prepared convolution/compact CTC, layout uint8 binary masks,
SLANet CUDA-encoder/CPU-decoder and UNET cached convolution remain selected by
the extended manifest. This audit is not proof that every remaining migration
path is free of extra work.

## Local verification

13 Python CUDA configuration tests pass, including disable/custom formula model
behavior and corruption checks. Three layout integration tests (18 tensor cases)
pass. Existing scalar/FMA comparison passes; a new SIMD versus scalar check
covers 196 tiny/boundary/short-row combinations, including non-vector tails.
No output tolerance is introduced. Full acceptance results follow below.


Windows final release wheel `build/cuda-formula-vector-wheels/` is installed in
the project `.venv`; installed attribution and the 23-DOCX public API check
(including PDF text smoke/five IO variants) passed. No full workspace regression
or local CUDA run is claimed. Five relevant changed source hashes match the
actual server build. Rust's initial 129-page run gives 67.849 s first and
63.714/63.496 s reused; full final snapshot remains exactly the accepted hash
`0b485e45c01b515e12bdf0e40301383f1299a3cbf3861114afe6f7c0791bb48d`.
Fresh Python/control/selected-page results are recorded below.


## Fresh full-pipeline comparison

129-page default-auto, same prepared models except the explicit formula graph
extension, independent sequential processes, one first/two reused calls:

| Full pipeline | Rust | Fresh Python |
| --- | ---: | ---: |
| First call | 67.849 s | 73.062 s |
| Reused call 1 | 63.714 s | 63.847 s |
| Reused call 2 | 63.496 s | 63.685 s |
| Reused median | 63.605 s | 63.766 s |

The full snapshots match exactly. The observed reused difference is only
0.25% in Rust's favor, comparable to measurement variation on this shared
server. This supports near parity on this case, not meaningful broad speed
superiority. The previous-SDK control and selected-page results follow below.


## Previous version control and precision closure

Same-environment prior `sdk-unet-final` control: first call 70.184 s, reused
66.791/65.834 s, median **66.312 s**. New reused median **63.605 s** reduces
elapsed time **4.08%** in this run. The control's complete snapshot also matches
the accepted hash. These sequential measurements on a shared GPU do not isolate
all environmental variability or establish broad cross-document gains.

New Rust peak RSS **3.128 GiB**, Python **6.594 GiB**; process GPU memory
**6.701/8.057 GiB** (Rust/Python), approximately 52.6%/16.8% lower. New median
is only **0.161 s** below Python; it is accurate to call this near parity,
not significant superiority. First-call values include engine initialization
and should not be substituted for reused throughput.

Eight groups / 13 selected pages match their complete preceding Python
snapshots and Markdown. The 129-page default-auto Rust/fresh Python/control
snapshots also match exactly. All 31 captured real UNET preprocessing tensors,
classification tensors and recovered cells/topology JSON remain byte-identical
after the shared cubic SIMD change. No new tolerance, normalization or rule
patch is used. Existing tiny-source cubic limitations remain documented and
are not claimed fixed by vectorization. This is 142 checked pages including
synthetic stress pages, not 142 independent client documents.

Final Linux artifacts: `formula-vector-final-wheels/`,
`sdk-formula-vector-final/`, `cuda-models-formula/manifest.json` in the isolated
server workspace. Original models, business environment and existing GPU
services are preserved. Windows `.venv` contains the new CPU/native layer;
CUDA model graphs are explicitly prepared on the GPU server and selected via
the manifest. An environment using an older manifest will not automatically
receive the new formula graph optimization just by reinstalling the wheel.


Final reports, full snapshots/Markdown, prepared formula graph, same-input
microbenchmarks, real UNET outputs, source hashes, scripts/logs and Linux wheel
are archived locally at
`build/cuda-validation-20261101/cuda-formula-vector-results.tar.gz`
(460,067,088 bytes, SHA256
`b6323a3ff74b7e037a3e37fbc3c2869fc1af91ae98607c4b019b62898eeb9927`).
Reports are also extracted under `formula-vector-final-records/` beside it.
Archive integrity was checked after transfer. Validation processes exited;
only original GPU service PIDs 675258/2531609 remain. The bundle references
prior prepared graph/operator-library locations and does not package every
external runtime/model dependency.
