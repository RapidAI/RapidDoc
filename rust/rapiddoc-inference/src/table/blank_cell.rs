//! UNET 未匹配格的轴框裁图、对比度门槛及独立 CTC 补识别。
use crate::{error, preprocess, table_recover::Polygon, Model};
use rapiddoc_core::{Error, ErrorCode};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultCell {
    pub cell_index: usize,
    pub text: String,
    pub confidence: f64,
    pub accepted: bool,
    pub reason: String,
    pub contrast: Option<f64>,
}
#[derive(Serialize)]
pub struct Crop {
    pub bgr: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub contrast: f64,
}

/// 原实现先交换 BGR/RGB，再把该数组作为 BGR 计算灰度和送入 rec；显式保留此合同。
pub fn crop(bytes: &[u8], width: usize, height: usize, p: Polygon) -> Result<Option<Crop>, Error> {
    let count = preprocess::check_size(width, height, 4_000_000)?;
    if count.checked_mul(3) != Some(bytes.len())
        || p.iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > rapiddoc_core::resources::limit(1_000_000.))
    {
        return Err(error(ErrorCode::InvalidInput, "UNET 补识别图像或格框无效"));
    }
    let x1 = p[0][0] as i64 + 1;
    let y1 = p[0][1] as i64 + 1;
    let x2 = p[2][0] as i64 - 1;
    let y2 = p[2][1] as i64 - 1;
    if x1 >= x2 || y1 >= y2 || x1 < 0 || y1 < 0 {
        return Ok(None);
    }
    if (x2 - x1) as f64 / (y2 - y1).max(1) as f64 > 20.
        || (y2 - y1) as f64 / (x2 - x1).max(1) as f64 > 20.
    {
        return Ok(None);
    }
    let x2 = x2.min(width as i64);
    let y2 = y2.min(height as i64);
    if x2 <= x1 || y2 <= y1 {
        return Ok(None);
    }
    let w = (x2 - x1) as usize;
    let h = (y2 - y1) as usize;
    let mut crop = Vec::with_capacity(w * h * 3);
    let mut sum = 0f64;
    let mut sum2 = 0f64;
    for y in y1 as usize..y2 as usize {
        for x in x1 as usize..x2 as usize {
            let i = (y * width + x) * 3;
            let a = bytes[i + 2];
            let b = bytes[i + 1];
            let c = bytes[i];
            crop.extend_from_slice(&[a, b, c]);
            let gray =
                ((a as u32 * 3735 + b as u32 * 19235 + c as u32 * 9798 + 16384) >> 15) as f64;
            sum += gray;
            sum2 += gray * gray;
        }
    }
    let n = (w * h) as f64;
    let mean = sum / n;
    let std = (sum2 / n - mean * mean).max(0.).sqrt();
    let contrast = format!("{:.2}", std / (mean + 1e-6))
        .parse()
        .map_err(|e| error(ErrorCode::InvalidInput, e))?;
    Ok(Some(Crop {
        bgr: crop,
        width: w,
        height: h,
        contrast,
    }))
}

pub fn accepted(text: &str, confidence: f64) -> bool {
    let text =
        text.trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{001c}'..='\u{001f}'));
    confidence.is_finite() && confidence >= 0.6 && !text.is_empty()
}
fn normalize(text: &str) -> String {
    let text =
        text.trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{001c}'..='\u{001f}'));
    if text == "香" {
        return "否".into();
    }
    if text == "哦樂" {
        return "哦".into();
    }
    let chars: Vec<_> = text.chars().collect();
    if chars.len() == 2 && chars[0].is_ascii_digit() && chars[1] == '號' {
        return chars[0].to_string();
    }
    text.into() // 转义推迟到 HTML 输出，避免把实体重复编码。
}
impl Model {
    pub fn recognize_blank_cells_png(
        &mut self,
        data: &[u8],
        polygons: &[Polygon],
        needed: &[bool],
    ) -> Result<Vec<ResultCell>, Error> {
        if polygons.len() != needed.len() {
            return Err(error(
                ErrorCode::InvalidInput,
                "UNET 空格标记数量与格框不匹配",
            ));
        }
        if polygons.len() > rapiddoc_core::resources::limit(2000) {
            return Err(error(
                ErrorCode::ResourceLimit,
                "UNET 空格补识别超过数量预算",
            ));
        }
        let (bytes, width, height) = preprocess::decode_png(data, 4_000_000)?;
        let mut results = Vec::new();
        let mut remaining = rapiddoc_core::resources::limit(32_000_000usize);
        let mut images = Vec::new();
        let mut indices = Vec::new();
        for (cell_index, &p) in polygons.iter().enumerate() {
            if !needed[cell_index] {
                continue;
            }
            let mut result = ResultCell {
                cell_index,
                text: String::new(),
                confidence: 0.,
                accepted: false,
                reason: "invalid_crop".into(),
                contrast: None,
            };
            if let Some(crop) = crop(&bytes, width, height, p)? {
                remaining = remaining
                    .checked_sub(crop.width * crop.height)
                    .ok_or_else(|| error(ErrorCode::ResourceLimit, "UNET 空格累计裁图超过预算"))?;
                result.contrast = Some(crop.contrast);
                if crop.contrast <= 0.17 {
                    result.reason = "low_contrast".into();
                } else {
                    indices.push(results.len());
                    images.push(crate::recognition_batch::Image {
                        bgr: crop.bgr,
                        width: crop.width,
                        height: crop.height,
                    });
                }
            }
            results.push(result);
        }
        if !images.is_empty() {
            let recognized = self.recognize_images(&images, 6)?;
            for (i, recognition) in indices.into_iter().zip(recognized) {
                let result = &mut results[i];
                result.accepted = accepted(&recognition.text, recognition.confidence);
                result.confidence = recognition.confidence;
                result.text = if result.accepted {
                    normalize(&recognition.text)
                } else {
                    recognition.text
                };
                result.reason = if result.accepted {
                    "accepted"
                } else {
                    "filtered"
                }
                .into();
            }
        }
        Ok(results)
    }
}
