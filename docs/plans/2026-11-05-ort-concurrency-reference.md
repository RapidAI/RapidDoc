# ORT concurrency work and oar-ocr reference

## Delivered locally

The user authorized concurrent Office parsing and PDF request-local processing,
keeping PDFium locking and one shared, locked session per model. No raw ORT C API
bypass or default duplicate sessions were introduced. Office has no image OCR.

- `RapidDoc` parse/warmup now use `&self`. Request-local pipeline clones share a
  short-locked model cache and independent timing. Config/resource updates detach
  cache mappings while active requests retain their snapshots.
- JNI/C handles retain `Arc<RapidDoc>` for each call. Python locks only lifecycle
  and configuration state; an explicit active counter also protects close after
  configuration snapshots change. Office calls no longer overwrite PDF resources.
- Body OCR resize/padding, progress callbacks and CTC decoding run outside the
  shared model lock. Full ORT outputs retain their own handles without copying
  logits; compact CTC copies only its small indices/scores. Existing thresholds,
  batch order, padding and decoders remain unchanged. Other model operations still
  retain their model-level locks; global ORT thread-pool policy is unchanged.
- Node reuses additional workers for overlapping requests; native session caching
  shares weights within the process. `parseAll` remains sequential. Close cancels
  all owned workers. Async callback reentry is rejected through AsyncLocalStorage.
- Synchronous parse reentry is rejected through a thread-local SDK guard; reading
  loaded-model roles is safe in native Rust/JNI/C/Python callbacks.

Validation (Windows CPU, no full regression or CUDA run):

- Rust SDK: 7 tests, including concurrent parsing of all six Office formats.
- Installed Python: 9 fixtures (six Office formats, native/scanned/mixed PDF),
  exact Markdown/middle/content/image hashes against pre-change installed output
  at 1/2/4 workers; forced overlapping PDF and OCR preparation barriers; lifecycle
  tests covering close, callback errors and result ownership.
- Java: all 10 NativeIntegrationTest cases passed, including forced same-engine
  PDF overlap, parallel Office, close and CPU OCR. An initial test used raw pip ORT
  and bypassed private CRT isolation; normal bundle discovery fixed test setup.
- Go: binding tests passed including same-engine overlap, cancellation, reentry,
  reference output, CPU OCR and owned results.
- Node: all 10 tests passed including same-instance PDF/Office overlap, CPU OCR,
  callback exceptions/cancellation/close, output and progress behavior.

Warm-model 9-document local sample (not a general speedup claim):

| workers | seconds | sampled peak process RSS bytes |
| --- | --- | --- |
| 1 | 2.705 | 548458496 |
| 2 | 2.221 | 574533632 |
| 4 | 3.353 | 585056256 |

Model initialization is excluded. RSS was sampled every 10 ms in one process;
these are small-fixture observations, not isolated-process capacity benchmarks.
More CPU concurrency was not always faster. Caller controls concurrency; no
fixed Office limit or automatic parallel parseAll expansion was added.

Artifacts/reports are under build/release/concurrency-*. Python wheel installed
into project .venv; Java API/native JARs installed into local Maven. Go/Node
preview bundles were generated under build/*-sdk/windows-x86_64-concurrency;
existing external Go/Node demos point to them and passed smoke calls.

Remaining validation: CUDA model/custom-operator concurrency and representative
production throughput/memory workloads. No code was committed or published.

## Reference inspected at the user's request

Repository: https://github.com/GreatV/oar-ocr
Commit: d59a42e484b28cf2ff2d6b31d22698df15bf9843
Local read-only checkout: build/reference/oar-ocr (ignored build directory).

Findings from source, without running performance benchmarks:

1. `oar-ocr-core/src/core/inference/mod.rs`: `Vec<Mutex<Session>>`;
   `ort_infer_builders.rs` constructs one session in both inspected constructors.
   `ort_infer_execution.rs` uses public `run`/`run_with_options` under the lock.
   This is not same-session parallel Run or a raw C API bypass.
2. `infer_first_output_f32` consumes SessionOutputs into an owned DynValue,
   drops the Session guard, then performs borrowed tensor postprocessing.
   CRNN preprocessing runs before calling it. This is a useful narrower lock
   boundary than our current whole-model-operation mutex.
3. `thread_pool.rs` offers a process-wide ORT thread pool committed before any
   session. Worth benchmarking for CPU oversubscription; do not silently alter
   process-global runtime policy in a library hosted by other applications.
4. CUDA CRNN supports device-bound outputs plus CTC reduction. Our ONNX compact
   CTC optimization already addresses large logits transfers; compare actual
   implementations rather than treating this as a missing feature.
5. Formula workaround sets process-global `CUDA_LAUNCH_BLOCKING=1` when unset.
   Its comments report correctness problems in that project. Do not copy this
   policy or conclude our model has the same issue without reproducing it.
6. Generic infer copies output buffers with `to_vec`; the specialized CRNN path
   avoids it. It is not evidence the entire project is faster or more efficient.

Recommendation: keep the current concurrency design; next consider splitting
pure preprocessing/owned output postprocessing from the session lock, using only
safe public ort ownership APIs. Global CPU thread pools are a measured optional
follow-up, not a default change. No oar-ocr source has been copied into production.
