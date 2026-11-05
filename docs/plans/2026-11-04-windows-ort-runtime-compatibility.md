# Windows ORT / JVM runtime investigation

## Confirmed cause

The observed Oracle Java 8u421 and Oracle Java 17.0.12 failures are caused by the older `msvcp140.dll` loaded from their own runtime directories before ORT is loaded. The same official ORT 1.23.2 binary loads when that dependency is updated in a private runtime copy. This is not a Java language-level, JNI protocol, model or Rust parsing failure.

The installed system MSVC runtime is 14.51.36247.0. Having it installed does not ensure a JVM will use it. Windows considers already-loaded modules when resolving dependent DLL names, even when the main DLL is opened using an absolute path. See [Windows DLL search order](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order) and [MSVC binary compatibility restrictions](https://learn.microsoft.com/en-us/cpp/porting/binary-compat-2015-2017).

## Process module observations

The probe pauses each process before and after loading the same ORT DLL, allowing its actual loaded module paths and file versions to be inspected.

| Environment | MSVCP loaded before ORT | ORT load |
| --- | --- | --- |
| Oracle Java 8u421 | JDK `jre/bin/msvcp140.dll`, 14.36.32532.0 | Failed |
| Corretto Java 8u422 | JDK `jre/bin/msvcp140.dll`, 14.16.27033.0 | Failed |
| Oracle Java 17.0.12 | JDK `bin/msvcp140.dll`, 14.36.32532.0 | Failed |
| JetBrains Java 17.0.14 | JDK `bin/msvcp140.dll`, 14.29.30153.0 | Failed |
| Microsoft Java 11.0.26 | JDK `bin/msvcp140.dll`, 14.38.33135.0 | Passed |
| Microsoft Java 21.0.7 | JDK `bin/msvcp140.dll`, 14.38.33135.0 | Passed |
| Python 3.12.10 | No MSVCP140 before loading ORT; VCRUNTIME140 14.42 loaded by Python | Passed |
| Python 3.14.5 | Probe process did not preload an old MSVCP140 | Passed |

These are verified combinations, not a universal minimum-version rule. Corretto/JetBrains were observed to preload old libraries; the controlled replacement experiment below was performed on the two Oracle runtimes.

## Controlled experiments

All changes in these replacement experiments were made to new runtime copies under `build/runtime-compatibility`. Original JDK directories and system files were not modified; source CRT file hashes were checked afterwards. The official ORT DLL and model files were unchanged in these experiments. The later SxS prototype below adjusts only a private ORT copy.

| Private Oracle Java 8u421 runtime | Result |
| --- | --- |
| Unchanged copy | ORT still fails to initialize |
| Replace only MSVCP140 14.36 with the installed system 14.51 file | ORT loads successfully |
| Update the matching runtime DLL set to 14.51 | ORT loads successfully |

A separate Oracle Java 17 copy with the matching CRT set updated to 14.51 also passed real inference. These copied DLLs are a local diagnostic control, not a new distributable runtime package.

## Real inference and output verification

The packaged Java SDK was exercised on Microsoft Java 21, the updated Oracle Java 8 copy and the updated Oracle Java 17 copy. Each initialized real OCR detector and recognizer sessions and parsed `pdf_scan.pdf` and `pdf_mixed.pdf` using identical rule-layout/OCR settings. Markdown, middle JSON, content-list JSON and image SHA-256 values were identical across all three JVMs.

This covers CPU model loading and OCR inference, not CUDA or every layout/formula/table model. No performance comparison was made.

## What does not fix an already-running incompatible JVM

Explicitly loading `C:/Windows/System32/msvcp140.dll` from Java before subsequently loading ORT still failed in the original Oracle Java 8 and 17 processes. The JVM had already loaded its old library. Merely placing another DLL beside ORT or changing PATH is therefore not a reliable SDK-level remedy. Unloading/replacing a runtime library already used by the JVM is not an acceptable automatic fix.

## Deployment conclusion

Recompiling ORT or changing Java's major version is not required to solve these reproduced failures. A library embedded in a third-party JVM must not rewrite that JVM's installation. Package-private SxS isolation now succeeds without updating the host JVM and has been integrated into the shared Rust loader and Windows Java packaging, as described below.

Do not automatically copy arbitrary system DLLs into users' JDK directories. Any runtime bundle intended for distribution must use a supported, redistributable dependency set with its required notices, rather than treating these local control copies as release artifacts.

Initial evidence is retained in `build/runtime-compatibility/loaded-modules.json`, `jre8-controlled-results.json`, `system-preload-results.json`, and `model-control-results.json`. Original JDK installations and upstream ORT files remain unchanged; the integration builds a separately derived ORT copy.

## Package-private CRT isolation prototype

Windows side-by-side assembly redirection can resolve dependencies before the loaded-module lookup. A private assembly containing CRT 14.51 and ORT was activated with `CreateActCtxW` / `ActivateActCtx`. The original ORT still failed: its embedded manifest (RT_MANIFEST resource 2, language 1033) established its own dependency context. Removing that resource from a private ORT copy allowed it to use the isolated CRT. The source DLL is untouched and the `.text` section SHA-256 is unchanged; ORT was not recompiled and models were not changed.

With this prototype, all six original JVM installations in the table above passed real CPU OCR on both PDFs. Markdown, middle JSON, content-list JSON and image hashes matched the working Microsoft Java 21 baseline. Module inspection in the original Oracle Java 8 process confirmed that its own MSVCP 14.36 and the private MSVCP 14.51 were loaded simultaneously. The old runtime was neither replaced nor unloaded.

Reproducible research sources are saved in [scripts/migration/windows_ort_sxs](../../scripts/migration/windows_ort_sxs/README.md). The saved preparer and pared-down probe were independently rerun with Oracle Java 8; output again matched the baseline exactly. Evidence: `build/runtime-compatibility/sxs-model-results.json`, `sxs-coexisting-msvcp.json`, and `build/ort-sxs-repro/output.json`.

### Prototype limitations (superseded where covered by integration below)

- Updating the resource removes the original signature guarantee. The original DLL reports `Valid`; derived copies reported `HashMismatch` or `NotSigned`. A release needs an explicit signing and packaging solution.
- CRT files copied from System32 are diagnostic inputs, not a redistributable release bundle. A supported redistributable source and its required notices must be selected.
- The JNI probe is deliberately single-threaded. Production code needs scoped activation guards, lifecycle and concurrency handling in the shared Rust loader, without leaking the context into unrelated host callbacks.
- CUDA provider loading, custom operators, concurrent engines, class-loader lifecycle, restrictive signature policies and complete parsing regressions have not been verified with this isolation mechanism.

## Shared loader and platform-package integration

`rapiddoc-inference/src/runtime/private_crt.rs` uses a thread-bound RAII guard rather than the diagnostic JNI probe. Each ORT initialization/session creation scope activates its own context and restores the caller's context on success or error. It does not span progress callbacks. Non-Windows and ordinary external ORT paths retain their existing behavior. SDK discovery prefers a prepared `RapidDoc.Runtime.PrivateCRT` subdirectory; explicit ORT paths still take precedence.

`scripts/migration/windows_ort_bundle.py` obtains matching x64 CRT 14.51 files from the Visual Studio REDIST directory, rejects System32 as the distribution source, verifies actual ORT API version 1.23.2 and unchanged `.text`, and records original/derived hashes and provenance. Runtime CRT downloads and host modifications are unnecessary. `build_java_sdk.py` includes the private bundle by default on Windows. `--external-ort` packages an unmodified external runtime without isolation. CRT additions total 1,646,032 bytes before compression.

The derived ORT does not retain Microsoft's signature. Publishers can supply `--ort-sign-command-json` to sign the temporary derived DLL before the JAR's hashes are generated; no trusted publisher certificate was available or used for this local build. Signature-enforcing deployments therefore remain a release limitation. The CRT DLLs are not modified.

### Integrated-package validation

- Built and installed the actual Java SDK and Windows CPU native JAR into local Maven; not published externally.
- Six original JVMs (Oracle/Corretto 8, Oracle/JetBrains 17, Microsoft 11/21) parsed both OCR fixtures through the ordinary `ModelCompatProbe`/Java SDK, with no activation wrapper, custom runtime path, JDK update or startup script. A fresh extraction cache was used. All outputs matched the earlier working baseline exactly. Evidence: `build/private-crt-acceptance/results.json`.
- Nine Java tests passed, covering seven-format reference output, OCR, callbacks, writing/closing and concurrent session creation. The concurrent OCR test also passed separately on the original Oracle 8u421.
- The Python native wheel was rebuilt and installed in `.venv`. Four recognition sessions were created on two worker threads; actual Windows activation contexts matched their original values after both successful creation and a deliberate invalid-model error. Evidence: `build/private-crt-acceptance/python-context-checks.json`. The reusable regression is `tests/migration/test_private_crt.py` (set `RAPIDDOC_TEST_PRIVATE_ORT`).
- Six affected Python tests passed (`test_private_crt.py` and `test_shared_sdk.py`), covering private context cleanup as well as ordinary external/pip runtime operation.
- All 30 packaged file hashes were checked, and the original ORT hash remained unchanged. Evidence: `build/private-crt-acceptance/bundle-checks.json`.

Windows CUDA/custom-operator execution, unusual class-loader isolation and enterprise signature policies are not yet accepted. Python's normal pip runtime remains external and unmodified; private CRT binaries are currently shipped in the Java Windows platform package, not automatically added to Python wheels or Cargo downloads. The shared loader accepts prepared bundles from any host. No parsing models or algorithms were changed.

## Windows CUDA package checks without a GPU

The local machine has neither an NVIDIA GPU/driver utility nor a CUDA toolkit installation. The official Windows GPU ORT 1.23.2 archive was downloaded and verified against the pinned SHA-256 `e77afdbbc2b8cb6da4e5a50d89841b48c44f3e47dce4fb87b15a2743786d0bb9`. The JNI engine was built with `cuda`; the actual CUDA-classified platform JAR was built and installed into local Maven alongside the existing CPU-classified JAR.

- All 35 packaged file hashes passed verification. The private CRT, main GPU ORT, shared/CUDA/TensorRT provider DLLs, CUDA custom-operator DLL and notices are present.
- All six original JVMs passed CPU OCR using the GPU platform JAR. Both PDF outputs exactly matched the CPU baseline.
- All six explicitly requested CUDA runs failed with `device_unavailable`, naming the missing `cublasLt64_12.dll`. They did not silently fall back to CPU. This validates missing-dependency reporting, not the behavior of a fully installed CUDA environment without a GPU.
- Oracle Java 8 and Microsoft Java 21 also passed a same-process recovery check: after CUDA initialization failed, an explicitly selected CPU engine parsed both PDFs with unchanged output.
- Evidence is in `build/windows-cuda-smoke/archive-receipt.json`, `bundle-checks.json`, `results.json`, `recovery-results.json` and raw stdout/stderr logs. The probe now accepts an optional `cpu`/`cuda` argument and reports native error codes. No parsing or production loader changes were required in this check.

Actual Windows CUDA provider initialization, custom-operator loading/execution and inference still require an NVIDIA GPU and matching CUDA/cuDNN. Static inspection found provider/custom-operator DLLs have their own embedded manifests; their interaction with private CRT selection must also be checked with actual dependencies installed. Only the main ORT manifest is adjusted in the current package. These checks do not constitute GPU accuracy, performance or memory acceptance.

## Independent runtime packaging update

ORT and its private CRT now ship in the versioned `rapid-doc-ort` artifact, separate from `rapid-doc-native`. The shared Windows isolation mechanism is unchanged. Private CRT handling was additionally verified with ORT 1.23.0 on original Oracle 8/17 and Microsoft 21. See [Java ORT distribution](2026-11-04-java-ort-distribution.md) for the tested Maven selection rules and current package-validation scope; earlier monolithic-package measurements above are historical evidence.
