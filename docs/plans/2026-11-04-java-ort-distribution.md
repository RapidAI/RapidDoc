# Java independent ORT distribution

## Result

The Java SDK selects ORT independently of the Rust/JNI engine. Default installation still supplies CPU ORT 1.23.2. Runtime versions must exist in the configured Maven repository; this does not convert arbitrary upstream downloads at application startup.

| Artifact | Version | Payload |
| --- | --- | --- |
| `rapid-doc` | SDK version, currently 0.1.0 | Java 8 API |
| `rapid-doc-native` | SDK version | Unified CPU/CUDA-capable JNI engine, PDFium, GEOS and notices |
| `rapid-doc-ort` | Actual ORT version | CPU/CUDA ORT, Windows private CRT where enabled, CUDA custom operators and notices |

Platform classifiers remain `windows-x86_64`, `windows-x86_64-cuda`, `linux-x86_64` and `linux-x86_64-cuda`. The CUDA native-engine classifier is retained as an alias; both Java engine builds enable the optional CUDA API and load GPU dependencies only when requested. Python's default features have not changed.

## Loading and Maven selection

Engine resources use `META-INF/native/<platform>/`; ORT uses `META-INF/rapiddoc-ort/<platform>/`. Each distribution has its own manifest, file hashes and extraction cache. Java passes the selected ORT directory to the shared Rust SDK ahead of the default engine directory, while preserving explicitly supplied resource paths and user bundle directories. Missing ORT remains allowed for Office-only usage. Multiple ORT manifests on the classpath are rejected rather than selecting one by classpath order.

Maven dependency identity includes the classifier. Version-only overrides can use `dependencyManagement` with the same classifier. Switching CPU to CUDA requires excluding the default CPU ORT dependency and declaring the CUDA dependency. A property in the consuming POM does not automatically replace an internal property of the SDK dependency. The [README](../../bindings/java/README.md) provides a tested direct-dependency example.

## Build and validation

`build_java_sdk.py --ort-only --ort-version <version> --ort-dir <directory>` packages/installs a runtime without invoking Cargo or compiling Java. `--cuda` chooses the GPU classifier and includes the custom-operator library. The packager compares the supplied version to ORT's actual `OrtGetApiBase()->GetVersionString()`, enforces API >=23 and rejects a GPU payload labeled CPU. Private CRT packaging is verified for ORT 1.23.0/1.23.2 CPU; other versions require an explicitly unmodified `--external-ort` distribution and separate host validation.

Local validation:

- 12 Java tests passed, including seven-format reference outputs, OCR/session reuse, concurrent session creation, independent-runtime checksum validation, duplicate-runtime rejection and absent-runtime handling.
- Six independent Maven consumer projects verified default CPU selection, 1.23.0 selection through dependency management, CUDA runtime selection with the same CPU-classified JNI engine, intentional conflicting CPU/CUDA dependencies, property-only non-override behavior, and actual Office parsing without any ORT dependency. Successful OCR outputs matched the existing two-PDF baseline.
- ORT 1.23.0 with private CRT passed real CPU OCR on original Oracle Java 8u421, Oracle Java 17.0.12 and Microsoft Java 21, with unchanged output.
- The separated CUDA runtime returned `device_unavailable` naming missing `cublasLt64_12.dll` on Oracle 8 and Microsoft 21, rather than the JNI engine reporting CUDA support was not compiled.
- Three invalid packaging configurations were rejected: ORT <1.23, version label differing from the actual DLL, and GPU libraries under a CPU classifier.
- Independent JAR manifests and every payload checksum were verified. The engine artifact contains no ORT DLL; both selected ORT versions use the same engine artifact.

Evidence: `build/ort-maven-switch/consumer-results.json`, `private-1.23.0-results.json`, `split-cuda-negative.json`, `bundle-checks.json`, `rejected-configurations.json` and the generated standalone consumer POMs/logs. ORT 1.23.0 was obtained from the official GitHub release and checked against its upstream asset SHA-256, recorded in `source-1.23.0.json`.

Only local Maven installation was performed. No artifacts were published. Actual Windows GPU execution, provider/custom-operator CRT coexistence, corporate signing policies and Linux acceptance of the newly split Java packages remain outstanding. Existing CUDA/Linux parser acceptance does not substitute for validation of these new package boundaries.
