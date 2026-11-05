# Shared model download progress

The model store emits download events before HTTP, while streaming, before hash verification, and after successful publication or failure. Byte counts and optional Content-Length, model ID, source URL, final destination and elapsed time are shared by all SDK hosts. Cached/custom models do not emit download events. Download callback errors cancel the transfer and retain partial-file cleanup; SDK callbacks retain their original host exceptions.

Connection, response-header and individual read waits have explicit 60-second timeouts. There is no fixed total-download deadline for large models. A started event makes the response wait visible; it does not imply any bytes have arrived. Completion is emitted only after SHA-256 verification and atomic publication.

Java logs downloads automatically through `java.util.logging` (`io.github.rapidai.rapiddoc.download`), even without a custom parse listener. Console messages follow the original Python wording: `Initiating download`, destination/size, transfer progress, hash verification, `Successfully saved to` or `Download failed`. Transfer updates are throttled to about five seconds. User listeners still receive the original structured events. Python's compatibility wrapper displays the same lifecycle through Loguru. This does not implement a Java parsing progress bar or change parsing algorithms.

Default models are stored at `~/.cache/rapid_doc/models`, not in the installed SDK or native-library extraction cache. Java can set `modelDirectory(Path)`; all hosts can use `RAPID_MODELS_DIR` or `RAPIDDOC_MODEL_DIR`. Downloads use temporary `.part` files and reuse validated final models on subsequent runs.

Targeted validation covers local HTTP transfer with known/unknown length, cache reuse, refusal, cancellation, checksum failure, Java default logging and callback exception propagation before any network request. Build/test receipts are retained under `build/download-progress`.

Validation completed: 10 Rust model-store tests, 14 Java tests and 5 affected Python SDK tests passed. The Java API and native engine were installed into both local Maven repositories (`D:/mavenlib` and the IDEA-used `C:/Users/huazhen/.m2/repository`). A standalone probe using the user's Spring Boot dependency classpath and Oracle JDK 17 confirmed default download messages and original callback cancellation. The rebuilt Python wheel was installed into `.venv`. No full-document or CUDA regression was run for this logging change.
