//! SLANet / SLANet-plus BGR 前处理和序列解码；尚不匹配单元格 OCR 文字。
use crate::{
    preprocess::{check_size, resize_linear, PreparedTensor},
    TensorOutput,
};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

#[derive(Clone, Copy)]
pub enum Variant {
    SlanetPlus,
    Slanet1m,
}

#[derive(Serialize)]
pub struct Prepared {
    pub tensor: PreparedTensor,
    pub shape_list: [f64; 6],
}

#[derive(Debug, Serialize)]
pub struct Structure {
    pub tokens: Vec<String>,
    pub confidence: f64,
    /// 原图像素坐标，四值轴框或八值四角，不进行裁剪或结构拓扑猜测。
    pub cell_bboxes: Vec<Vec<f32>>,
}

fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}

/// 长边缩放到 488，归一化后在右/下补零；输入为 BGR uint8。
pub fn prepare(
    bytes: &[u8],
    width: usize,
    height: usize,
    max_elements: usize,
) -> Result<Prepared, Error> {
    let pixels = check_size(width, height, 16_000_000)?;
    if pixels.checked_mul(3) != Some(bytes.len()) {
        return Err(error(ErrorCode::InvalidInput, "表格 BGR 字节长度无效"));
    }
    let ratio = 488. / width.max(height) as f64;
    let w = (width as f64 * ratio) as usize;
    let h = (height as f64 * ratio) as usize;
    if w == 0 || h == 0 {
        return Err(error(ErrorCode::InvalidInput, "表格缩放后的边长为零"));
    }
    let elements = 3 * 488 * 488;
    if elements > max_elements {
        return Err(error(ErrorCode::ResourceLimit, "表格输入张量超过预算"));
    }
    let resized = resize_linear(bytes, width, height, w, h);
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    let mut values = vec![0.; elements];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let scaled = resized[(y * w + x) * 3 + c] as f32 * (1f32 / 255.);
                values[c * 488 * 488 + y * 488 + x] = ((scaled as f64 - mean[c]) / std[c]) as f32;
            }
        }
    }
    Ok(Prepared {
        tensor: PreparedTensor {
            shape: vec![1, 3, 488, 488],
            values,
            resized_width: w,
        },
        shape_list: [height as f64, width as f64, ratio, ratio, 488., 488.],
    })
}

/// 字典去掉独立 <td>，补 <td></td> 并加 sos/eos；不能作为 CTC 去重解码。
pub fn alphabet(dictionary: &str) -> Result<Vec<String>, Error> {
    let mut labels: Vec<_> = dictionary.lines().map(str::to_owned).collect();
    if labels.is_empty()
        || labels.len() > rapiddoc_core::resources::limit(1024)
        || labels.iter().any(|s| {
            s.is_empty()
                || s.len() > rapiddoc_core::resources::limit(256)
                || s == "sos"
                || s == "eos"
        })
    {
        return Err(error(ErrorCode::Unsupported, "表格结构字典无效"));
    }
    if !labels.iter().any(|s| s == "<td></td>") {
        labels.push("<td></td>".into());
    }
    labels.retain(|s| s != "<td>");
    labels.insert(0, "sos".into());
    labels.push("eos".into());
    Ok(labels)
}

/// 单图序列按首个最大 token 解码；EOS 截断、空 bbox 过滤与 plus 坐标回缩保留源顺序。
pub fn decode(
    bboxes: &TensorOutput,
    probabilities: &TensorOutput,
    labels: &[String],
    width: usize,
    height: usize,
    variant: Variant,
) -> Result<Structure, Error> {
    if width == 0
        || height == 0
        || width > rapiddoc_core::resources::limit(16_000_000)
        || height > rapiddoc_core::resources::limit(16_000_000)
        || labels.len() < 3
        || labels.len() > rapiddoc_core::resources::limit(1026)
        || labels.first().map(String::as_str) != Some("sos")
        || labels.last().map(String::as_str) != Some("eos")
    {
        return Err(error(ErrorCode::InvalidInput, "结构解码尺寸或字典无效"));
    }
    let d = match variant {
        Variant::SlanetPlus => 8,
        Variant::Slanet1m => 4,
    };
    if bboxes.shape.len() != 3
        || probabilities.shape.len() != 3
        || bboxes.shape[0] != 1
        || probabilities.shape[0] != 1
        || bboxes.shape[2] != d
        || probabilities.shape[2] != labels.len() as i64
        || bboxes.shape[1] != probabilities.shape[1]
        || bboxes.shape[1] <= 0
        || bboxes.shape[1] > rapiddoc_core::resources::limit(10000)
    {
        return Err(error(
            ErrorCode::Unsupported,
            "结构输出必须为单图匹配的 bbox/概率序列",
        ));
    }
    let steps = bboxes.shape[1] as usize;
    let d = d as usize;
    if bboxes.values.len() != steps * d
        || probabilities.values.len() != steps * labels.len()
        || bboxes.values.iter().any(|v| !v.is_finite())
        || probabilities
            .values
            .iter()
            .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
    {
        return Err(error(ErrorCode::InvalidInput, "结构张量形状或数值无效"));
    }
    let ratio = 488. / width.max(height) as f64;
    let sx = (488. / (width as f64 * ratio)) as f32;
    let sy = (488. / (height as f64 * ratio)) as f32;
    let mut tokens = vec!["<html>".into(), "<body>".into(), "<table>".into()];
    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    for step in 0..steps {
        let row = &probabilities.values[step * labels.len()..(step + 1) * labels.len()];
        let mut index = 0;
        for candidate in 1..row.len() {
            if row[candidate] > row[index] {
                index = candidate;
            }
        }
        if step > 0 && index == labels.len() - 1 {
            break;
        }
        if index == 0 || index == labels.len() - 1 {
            continue;
        }
        let token = &labels[index];
        if matches!(token.as_str(), "<td>" | "<td" | "<td></td>") {
            let mut bbox = bboxes.values[step * d..(step + 1) * d].to_vec();
            for (coordinate, v) in bbox.iter_mut().enumerate() {
                *v *= if coordinate % 2 == 0 {
                    width as f32
                } else {
                    height as f32
                };
                if matches!(variant, Variant::SlanetPlus) {
                    *v *= if coordinate % 2 == 0 { sx } else { sy };
                }
                if !v.is_finite() {
                    return Err(error(ErrorCode::InvalidInput, "结构 bbox 缩放后非有限"));
                }
            }
            if bbox.iter().any(|v| *v != 0.) {
                boxes.push(bbox);
            }
        }
        tokens.push(token.clone());
        scores.push(row[index] as f64);
    }
    if scores.is_empty() {
        return Err(error(
            ErrorCode::InvalidInput,
            "模型未产生结构 token，不能返回 NaN 评分",
        ));
    }
    tokens.extend(["</table>".into(), "</body>".into(), "</html>".into()]);
    Ok(Structure {
        tokens,
        confidence: scores.iter().sum::<f64>() / scores.len() as f64,
        cell_bboxes: boxes,
    })
}
