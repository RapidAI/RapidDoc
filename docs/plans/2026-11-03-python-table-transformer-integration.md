# Python Table Transformer integration

Goal: add the reviewed TATR v1.1-all ONNX single-table model as an explicit Python option, preserving current defaults and Rust.
Architecture: existing layout crops and OCR/native PDF text feed the ONNX adapter and pinned Microsoft structure postprocessing. No second OCR model or default model switch.
Tech stack: existing NumPy, Pillow, ONNX Runtime; optional export uses torch/transformers.

1. Review source adapter against pinned Microsoft inference/postprocess; verify ONNX SHA256 and licenses.
2. Add focused tests for routing, batch alignment, empty OCR, spans, escaping, and low-confidence table hypotheses; import only TATR, not POTATR/UI changes.
3. Integrate ModelType/model lookup/RapidTable/RapidTableModel; preserve the original use_word_box=False default and explicit overrides. Keep current native-rule opt-out explicit.
4. Verify actual ONNX on three crops and independent upstream geometry/postprocessing; run affected existing table/routing tests and one public PDF call.
5. Document explicit configuration, local weights/export, findings, and unvalidated CUDA/accuracy scope. No full corpus or Rust rebuild.

Completed: TATR-only adapter/ONNX and independent reference verification imported;
OCR alignment, single escaping, caller immutability, fully spanned HTML rows and
table threshold reviewed/fixed. Three crops matched upstream cells and torch/ORT
within tolerance; one public PDF call and an image with real OCR completed.
The very dense 120-row PDF smoke has substantial structure errors: integration
is available for opt-in evaluation, not a replacement recommendation.

Correction: the conditional word-box default was reverted. All models/rule
engines now retain False unless the caller explicitly supplies True. Regression
coverage includes enum/string models and combined TATR/docvortex_ocr settings.
