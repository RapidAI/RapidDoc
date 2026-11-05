//! ocr adapters; algorithms remain in the shared Rust crates.
use super::*;

/// OCR 图像级输出保留像素四角和 token 诊断。
#[derive(serde::Serialize)]
pub(super) struct OcrLine {
    pub(super) points: [[f32; 2]; 4],
    pub(super) detection_score: f64,
    #[serde(flatten)]
    pub(super) recognition: rapiddoc_inference::ctc::Recognition,
}

/// 两个会话分阶段取锁，保证相反顺序的并发调用也不会死锁。
pub(super) fn run_ocr(
    detector: &Mutex<rapiddoc_inference::Model>,
    recognizer: &Mutex<rapiddoc_inference::Model>,
    data: &[u8],
) -> Result<Vec<OcrLine>, rapiddoc_core::Error> {
    let lock_error = || rapiddoc_core::Error {
        code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
        message: "ONNX 会话锁已损坏".into(),
    };
    let mut boxes = detector
        .lock()
        .map_err(|_| lock_error())?
        .detect_png(data)?;
    rapiddoc_inference::detection::sort_reading_order(&mut boxes);
    if boxes.is_empty() {
        return Ok(Vec::new());
    }
    let points: Vec<_> = boxes.iter().map(|b| b.points).collect();
    let recognitions = recognizer
        .lock()
        .map_err(|_| lock_error())?
        .recognize_regions_png(data, &points)?;
    Ok(boxes
        .into_iter()
        .zip(recognitions)
        .map(|(b, recognition)| OcrLine {
            points: b.points,
            detection_score: b.score,
            recognition,
        })
        .collect())
}

/// PNG 检测、排序、透视裁图、逐行识别全部在释放 GIL 后运行。
#[pyfunction]
pub(super) fn ocr_png_json(
    py: Python<'_>,
    detector: PyRef<'_, OnnxSession>,
    recognizer: PyRef<'_, OnnxSession>,
    data: Vec<u8>,
) -> PyResult<String> {
    let detector = &detector.model;
    let recognizer = &recognizer.model;
    py.detach(move || serialize(&run_ocr(detector, recognizer, &data)?))
        .map_err(|e| python_error(py, e))
}
