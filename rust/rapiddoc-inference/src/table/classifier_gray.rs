//! Qanything 灰度前处理；Pillow 12.2.0 L/BICUBIC 双遍、22 位定点权重。
//! 来源和许可见 packaging/inference-provenance.json 与 licenses/PILLOW.txt。
use crate::{
    error,
    preprocess::{check_size, PreparedTensor},
};
use rapiddoc_core::{Error, ErrorCode};

fn cubic(mut x: f64) -> f64 {
    x = x.abs();
    if x < 1. {
        (1.5 * x - 2.5) * x * x + 1.
    } else if x < 2. {
        (((x - 5.) * x + 8.) * x - 4.) * (-0.5)
    } else {
        0.
    }
}
fn coefficients(size: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = size as f64 / 224.;
    let filter_scale = scale.max(1.);
    let support = 2. * filter_scale;
    (0..224)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(size);
            let mut weights: Vec<_> = (start..end)
                .map(|x| cubic((x as f64 - center + 0.5) * (1. / filter_scale)))
                .collect();
            let sum: f64 = weights.iter().sum();
            if sum != 0. {
                for w in &mut weights {
                    *w /= sum;
                }
            }
            (
                start,
                weights
                    .into_iter()
                    .map(|v| (v * ((1 << 22) as f64)).round() as i32)
                    .collect(),
            )
        })
        .collect()
}
pub fn prepare_bgr(
    bytes: &[u8],
    width: usize,
    height: usize,
    max_elements: usize,
) -> Result<PreparedTensor, Error> {
    let count = check_size(width, height, 4_000_000)?;
    if count.checked_mul(3) != Some(bytes.len()) {
        return Err(error(ErrorCode::InvalidInput, "Qanything BGR 字节长度无效"));
    }
    check_size(224, height, 4_000_000)?;
    if 3 * 224 * 224 > max_elements {
        return Err(error(ErrorCode::ResourceLimit, "Qanything 张量超过预算"));
    }
    // 源 BGR 数组按 RGB 灰度解释，保留历史 COLOR_RGB2GRAY 语义。
    let gray: Vec<u8> = bytes
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            ((p[0] as u32 * 9798 + p[1] as u32 * 19235 + p[2] as u32 * 3735 + 16384) >> 15) as u8
        })
        .collect();
    let x = coefficients(width);
    let y = coefficients(height);
    let mut horizontal = vec![0u8; 224 * height];
    for row in 0..height {
        for (col, (start, weights)) in x.iter().enumerate() {
            let mut value = 1i64 << 21;
            for (i, &weight) in weights.iter().enumerate() {
                value += gray[row * width + start + i] as i64 * weight as i64;
            }
            horizontal[row * 224 + col] = (value >> 22).clamp(0, 255) as u8;
        }
    }
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut values = vec![0f32; 3 * 224 * 224];
    for (row, (start, weights)) in y.iter().enumerate() {
        for col in 0..224 {
            let mut value = 1i64 << 21;
            for (i, &weight) in weights.iter().enumerate() {
                value += horizontal[(start + i) * 224 + col] as i64 * weight as i64;
            }
            let pixel = (value >> 22).clamp(0, 255) as u8;
            let normalized = pixel as f32 / 255.;
            for c in 0..3 {
                values[c * 224 * 224 + row * 224 + col] = (normalized - mean[c]) / std[c];
            }
        }
    }
    Ok(PreparedTensor {
        shape: vec![1, 3, 224, 224],
        values,
        resized_width: 224,
    })
}
