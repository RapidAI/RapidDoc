# Windows private CRT compatibility prototype

Research tools only: Windows x64, official ORT 1.23.2 CPU. This code is not linked into release bindings. The JNI probe maintains one activation context on one calling thread; it is not a production concurrent context manager and its abort-on-panic build must not be reused as a release adapter.

`prepare.py` copies the supplied ORT and local CRT files into a new output directory. It removes RT_MANIFEST/2, language 1033, from that private ORT copy so its original embedded context does not defeat the private assembly's dependency selection. It verifies the executable `.text` section is unchanged. The source DLL and JDK files are not edited. Updating PE resources invalidates the original Authenticode signature: source and derived-file hashes, redistribution notices, and signing must be addressed before any release. Locally copied system DLLs are diagnostic inputs, not a ready-to-publish redistributable.

Example preparation from the repository root:

```powershell
.venv/Scripts/python.exe scripts/migration/windows_ort_sxs/prepare.py --ort target/release/runtime/onnxruntime.dll --output build/ort-sxs-repro
rustc --edition 2021 --crate-type cdylib -C opt-level=1 -C panic=abort scripts/migration/windows_ort_sxs/probe.rs -o build/ort-sxs-repro/sxs_probe.dll
```

Compile the Java probes with Java 8 bytecode, supplying the Java SDK JAR and its Jackson dependencies on the classpath; `ModelCompatProbe` also uses the platform native JAR. Set `RAPIDDOC_SXS_TEST_MANIFEST` to the output `app.manifest`, and `RAPIDDOC_ONNXRUNTIME_LIBRARY` to its `RapidDoc.Runtime.PrivateCRT/onnxruntime.dll`. `OrtSxsProbe` accepts the JNI probe DLL path, ORT DLL path and mode (`native`, `java`, `model`, `modelhold`). Model modes additionally accept the project root and output JSON path. `modelhold` pauses for one input line after parsing, allowing loaded DLL versions to be inspected.

Six original JVMs passed real CPU OCR using the existing SDK through this wrapper: Oracle 8u421, Corretto 8u422, Oracle 17.0.12, JetBrains 17.0.14, Microsoft 11.0.26 and Microsoft 21.0.7. Two scanned/mixed PDF outputs matched the working Microsoft 21 baseline exactly. Process inspection confirmed old JDK MSVCP140 14.36 and private 14.51 loaded simultaneously. This does not yet validate CUDA, concurrent activation, classloader lifecycle, resource cleanup stress or corporate signature policies. No production integration is claimed.

Full evidence and the failed intermediate experiments are described in `docs/plans/2026-11-04-windows-ort-runtime-compatibility.md`.

The shared Rust loader and Windows Java packaging now contain the production
integration, independently of this diagnostic JNI wrapper. Its validation and
remaining signing/CUDA limitations are recorded in the report. Normal users do
not execute these probe scripts. Production packaging uses
`scripts/migration/windows_ort_bundle.py` and a Visual Studio REDIST source.

`ModelCompatProbe` can also be run directly (without `OrtSxsProbe`) against the
normal engine JAR plus the independently selected CPU or CUDA `rapid-doc-ort`
JAR (and the API/Jackson dependencies). Its arguments are project root, output JSON and
optional `cpu`/`cuda` device. Native failures print `NATIVE_ERROR_CODE|...` before
propagating the exception, so a missing-CUDA check can distinguish an expected
device error from packaging, model or JNI failures.
