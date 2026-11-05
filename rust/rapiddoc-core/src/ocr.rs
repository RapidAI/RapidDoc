//! OCR 文字与四角契约；坐标单位由包含它的输出类型声明。
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TextLine {
    pub points: [[f64; 2]; 4],
    pub text: String,
    pub confidence: f64,
    pub detection_score: f64,
}
