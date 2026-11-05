# Java binding

Use JNI as a thin adapter around the shared Rust SDK; no Java ORT wrapper, model scheduler or parser implementation. Java 8 API with Maven coordinates `io.github.rapidai:rapid-doc`; separate `rapid-doc-native` classified artifacts hold platform runtime files. No packages have been published.

JNI maintains opaque numeric handles in an Arc/Mutex registry. Closing an in-flight engine removes its handle while the active call retains ownership. Reentrant/concurrent parsing fails rather than deadlocking. Every JNI boundary catches Rust panics, and Java callback exceptions remain pending and propagate unchanged. Asset bytes transfer directly to Java arrays; document metadata serializes once. Bounded JNI local frames cover progress and asset loops.

The platform loader verifies SHA-256 digests and extracts resources to a content-addressed directory. Explicit native library/directory properties support independent deployment. The builder passes bundle roots into shared Rust resource discovery; Rust owns model defaults, downloads, initialization, inference and optional vector drawing. Java handles typed options/results and output writing only. Result asset maps are unmodifiable, but byte arrays are intentionally not cloned.

Local acceptance: seven-format Markdown/middle/content/image-hash parity; byte/file input and Unicode/emoji paths; callback exception identity and reentry rejection; concurrent/idempotent close; native vector drawing and image writer; scanned PDF CPU OCR/session reuse. Seven integration tests passed on Microsoft JDK 21.0.7 with Java 11 bytecode, plus a standalone JAR launch outside the checkout's runtime environment. Oracle JDK 17.0.12 passed non-model tests but failed direct ORT loading: its bundled MSVC 14.36 runtime conflicts with ORT; the same DLL loads on the Microsoft JDK. No system/JDK DLLs were changed. Linux/CUDA remain separate acceptance work. Full dataset and server benchmarks are not part of this binding change.


## Java 8 language/API compatibility

The Maven release target is now 8. Path.of, local var declarations and InputStream.readAllBytes were replaced with Java 8 equivalents, including usage samples and tests. Cleaner is replaced by a small PhantomReference/ReferenceQueue daemon; atomic cleanup preserves explicit/idempotent close and GC fallback without finalizers or a new dependency. Registration retains the owner through a monitor until the phantom reference is tracked, since Java 8 has no reachabilityFence. Rust/JNI parsing and native runtime binaries are unchanged.

Actual Oracle javac 8u421 compiled every SDK source. Oracle 8u421 and Corretto 8u422 both parsed DOCX through the native engine. This establishes Java 8 API/bytecode and basic JNI compatibility, not ORT compatibility: their separately observed ORT initialization failures remain unresolved. Microsoft JDK 21 runs the seven existing integration regressions plus the cleanup queue/concurrent-close test. This stage does not promise model inference on all JDK 8 builds or change MSVC/ORT packaging.
