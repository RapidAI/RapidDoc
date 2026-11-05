# RapidDoc C ABI v1

`include/rapid_doc.h` is the canonical header. This adapter owns handles, response
memory and a panic boundary; all parsing, model configuration, runtime discovery,
download and postprocessing belong to the existing Rust SDK. Python/PyO3 and
Java/JNI remain unchanged. Node.js has not been added yet.

Build with `cargo build --release -p rapiddoc-c`. `rapid_doc_c.dll` /
`librapid_doc_c.so` exports `rd_abi_version`, `rd_invoke` and
`rd_response_free`. Default builds enable CPU/CUDA API support without linking
the CUDA toolkit; the supplied ORT runtime determines available providers.

Operations, buffer sizes and ownership are documented in the header. Creation
accepts Rust SDK config/resources JSON, parsing accepts path UTF-8 or document
bytes and options JSON. Successful results expose JSON metadata plus separate
borrowed binary assets. Free every response exactly once, even on errors. Copy
any data needed after freeing it. Close engines explicitly. A numeric registry
validates engine handles and protects active calls from concurrent close.

Progress callbacks are synchronous. Event buffers are valid only during the
callback. Return nonzero to cancel; do not throw/longjmp/unwind across the C
boundary. Same-engine reentry/concurrency fails explicitly instead of blocking.
Callers must supply valid readable buffers/function pointers; a C ABI cannot
validate arbitrary stale pointers or double frees.

The Go module carries a distribution copy of the header so it can be installed
without this sibling directory. Keep it identical to the canonical file.
Supported packaged target is currently 64-bit Windows; Linux source loader is
provided but native Linux package acceptance remains outstanding.
