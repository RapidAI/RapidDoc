//! UNET 原始线类别图路径；线段修复/拓扑恢复另行实现。
use crate::{
    error,
    layout::resize_cubic,
    named::{Input, Output},
    preprocess::{check_size, PreparedTensor},
    Model,
};
use rapiddoc_core::{Error, ErrorCode};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub fn prepare(
    bytes: &[u8],
    width: usize,
    height: usize,
    max_elements: usize,
) -> Result<PreparedTensor, Error> {
    let pixels = check_size(width, height, 16_000_000)?;
    if pixels.checked_mul(3) != Some(bytes.len()) {
        return Err(error(ErrorCode::InvalidInput, "UNET BGR 字节长度无效"));
    }
    let ratio = 1024. / width.max(height) as f64;
    let w = (width as f64 * ratio + 0.5) as usize;
    let h = (height as f64 * ratio + 0.5) as usize;
    let count = check_size(w, h, 1_048_576)?;
    if count * 3 > max_elements {
        return Err(error(ErrorCode::ResourceLimit, "UNET 张量超过预算"));
    }
    let image = if width.min(height) > 1024 {
        resize_area(bytes, width, height, w, h)
    } else {
        resize_cubic(bytes, width, height, w, h)
    };
    let mean = [123.675f32, 116.28, 103.53];
    let std = [58.395f32, 57.12, 57.375];
    let mut values = vec![0.; count * 3];
    for i in 0..count {
        for c in 0..3 {
            // OpenCV 的 subtract/multiply 是两个独立 FP32 写回，stdinv 先用 FP64 求值。
            let shifted = (image[i * 3 + 2 - c] as f64 - mean[c] as f64) as f32;
            values[c * count + i] = (shifted as f64 * (1. / std[c] as f64)) as f32;
        }
    }
    Ok(PreparedTensor {
        shape: vec![1, 3, h, w],
        values,
        resized_width: w,
    })
}

fn area_coefficients(source: usize, target: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = source as f64 / target as f64;
    (0..target)
        .map(|i| {
            let start = i as f64 * scale;
            let end = start + scale;
            let span = scale.min(source as f64 - start);
            let first = start.ceil() as usize;
            let last = (end.floor() as usize).min(source - 1);
            let mut result = Vec::new();
            if first as f64 - start > 1e-3 {
                result.push((first - 1, ((first as f64 - start) / span) as f32));
            }
            for j in first..last {
                result.push((j, (1. / span) as f32));
            }
            if end - last as f64 > 1e-3 {
                result.push((last, ((end - last as f64).min(1.).min(span) / span) as f32));
            }
            result
        })
        .collect()
}

fn resize_area(bytes: &[u8], width: usize, height: usize, w: usize, h: usize) -> Vec<u8> {
    let mut output = vec![0; w * h * 3];
    if width == w * 2 && height == h * 2 {
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let a = (y * 2 * width + x * 2) * 3 + c;
                    let sum = bytes[a] as u16
                        + bytes[a + 3] as u16
                        + bytes[a + width * 3] as u16
                        + bytes[a + width * 3 + 3] as u16;
                    output[(y * w + x) * 3 + c] = ((sum + 2) >> 2) as u8;
                }
            }
        }
        return output;
    }
    let xs = area_coefficients(width, w);
    let ys = area_coefficients(height, h);
    for (y, weights) in ys.iter().enumerate() {
        let mut row = vec![0f32; w * 3];
        for &(sy, beta) in weights {
            for (x, weights) in xs.iter().enumerate() {
                for c in 0..3 {
                    let mut value = 0f32;
                    for &(sx, alpha) in weights {
                        value += bytes[(sy * width + sx) * 3 + c] as f32 * alpha;
                    }
                    row[x * 3 + c] += value * beta;
                }
            }
        }
        for (i, v) in row.into_iter().enumerate() {
            output[y * w * 3 + i] = v.round_ties_even().clamp(0., 255.) as u8;
        }
    }
    output
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mask {
    pub width: usize,
    pub height: usize,
    /// 0 背景、1 横线、2 竖线；在模型缩放图像素坐标内，未回缩到原图。
    pub classes: Vec<u8>,
}

impl Model {
    pub fn table_line_mask_png(&mut self, bytes: &[u8]) -> Result<Mask, Error> {
        let (bgr, width, height) = crate::preprocess::decode_png(bytes, 16_000_000)?;
        self.table_line_mask_bgr(&bgr, width, height)
    }

    pub(crate) fn table_line_mask_bgr(
        &mut self,
        bgr: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Mask, Error> {
        let tensor = prepare(bgr, width, height, self.max_elements)?;
        let (h, w) = (tensor.shape[2], tensor.shape[3]);
        let outputs = self.run_named(BTreeMap::from([(
            "input".into(),
            Input {
                shape: tensor.shape,
                values: tensor.values,
            },
        )]))?;
        let Some(Output::Int64 { shape, values }) = outputs.get("output") else {
            return Err(error(
                ErrorCode::Unsupported,
                "UNET 输出必须为 output int64 张量",
            ));
        };
        if shape != &[1, 1, h as i64, w as i64] || values.iter().any(|v| !(0..=2).contains(v)) {
            return Err(error(ErrorCode::Unsupported, "UNET 类别图形状或标签无效"));
        }
        Ok(Mask {
            width: w,
            height: h,
            classes: values.iter().map(|v| *v as u8).collect(),
        })
    }
}
