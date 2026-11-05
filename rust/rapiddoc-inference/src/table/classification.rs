//! Paddle 表格分类：BGR、短边 256 Lanczos4、中心裁 224、FP32 ImageNet。
//! Lanczos4 系数/定点运算参考 OpenCV resize.cpp；许可见 OPENCV-RESIZE.txt。
use crate::{
    error,
    preprocess::{check_size, PreparedTensor},
    Model,
};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

#[derive(Clone, Copy)]
pub enum Variant {
    Paddle,
    Qanything,
}
impl Variant {
    pub fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "paddle" => Ok(Self::Paddle),
            "qanything" => Ok(Self::Qanything),
            _ => Err(error(
                ErrorCode::InvalidInput,
                "表格分类变体须为 paddle/qanything",
            )),
        }
    }
    pub fn output_type(self) -> OutputType {
        match self {
            Self::Paddle => OutputType::Probabilities,
            Self::Qanything => OutputType::Logits,
        }
    }
    pub fn prepare(
        self,
        bytes: &[u8],
        w: usize,
        h: usize,
        max: usize,
    ) -> Result<PreparedTensor, Error> {
        match self {
            Self::Paddle => prepare_bgr(bytes, w, h, max),
            Self::Qanything => crate::table_classifier_gray::prepare_bgr(bytes, w, h, max),
        }
    }
}

#[derive(Clone, Copy)]
pub enum OutputType {
    Probabilities,
    Logits,
}
impl OutputType {
    pub fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "probabilities" => Ok(Self::Probabilities),
            "logits" => Ok(Self::Logits),
            _ => Err(error(
                ErrorCode::InvalidInput,
                "表格分类输出类型须为 probabilities/logits",
            )),
        }
    }
}
#[derive(Serialize)]
pub struct Classification {
    pub label: &'static str,
    pub score: f64,
}
pub fn decode(values: &[f32], output_type: OutputType) -> Result<Classification, Error> {
    if values.len() != 2 || values.iter().any(|v| !v.is_finite()) {
        return Err(error(ErrorCode::InvalidInput, "表格分类需要两个有限评分"));
    }
    let mut scores = [values[0], values[1]];
    match output_type {
        OutputType::Probabilities => {
            if scores.iter().any(|v| !(0. ..=1.).contains(v))
                || (scores[0] + scores[1] - 1.).abs() > 1e-4
            {
                return Err(error(
                    ErrorCode::InvalidInput,
                    "表格分类概率无效；logits 模型须明确指定输出类型",
                ));
            }
        }
        OutputType::Logits => {
            let max = scores[0].max(scores[1]);
            scores = scores.map(|v| (v - max).exp());
            let sum = scores[0] + scores[1];
            scores = scores.map(|v| v / sum);
        }
    }
    let index = usize::from(scores[1] > scores[0]);
    Ok(Classification {
        label: ["wired", "wireless"][index],
        // Python 基线先将 FP32 转为 Python float；保留其精确数值以免改变阈值判断。
        score: scores[index] as f64,
    })
}

fn coefficients(source: usize, target: usize, start: usize) -> Vec<(isize, [i32; 8])> {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let cs = [
        [1., 0.],
        [-s, -s],
        [0., 1.],
        [s, -s],
        [-1., 0.],
        [s, s],
        [0., -1.],
        [-s, s],
    ];
    let scale = 1. / (target as f64 / source as f64);
    (start..start + 224)
        .map(|i| {
            let pos = ((i as f64 + 0.5) * scale - 0.5) as f32;
            let base = pos.floor() as isize;
            let x = pos - base as f32;
            let a = -((x + 3.) as f64) * std::f64::consts::PI * 0.25;
            let (sin, cos) = a.sin_cos();
            let mut sum = 0f32;
            let mut kernel = [0f32; 8];
            for k in 0..8 {
                let z = x + 3. - k as f32;
                kernel[k] = if z.abs() >= 1e-6 {
                    let y = -(z as f64) * std::f64::consts::PI * 0.25;
                    ((cs[k][0] * sin + cs[k][1] * cos) / (y * y)) as f32
                } else {
                    1e30
                };
                sum += kernel[k];
            }
            let inv = 1f32 / sum;
            (
                base,
                kernel.map(|v| ((v * inv) * 2048.).round_ties_even() as i32),
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
    let pixels = check_size(width, height, 4_000_000)?;
    if pixels.checked_mul(3) != Some(bytes.len()) {
        return Err(error(ErrorCode::InvalidInput, "表格分类 BGR 字节长度无效"));
    }
    let ratio = 256. / width.min(height) as f64;
    let w = (width as f64 * ratio).round_ties_even() as usize;
    let h = (height as f64 * ratio).round_ties_even() as usize;
    check_size(w, h, 4_000_000)?;
    let elements = 3 * 224 * 224;
    if elements > max_elements {
        return Err(error(ErrorCode::ResourceLimit, "表格分类张量超过预算"));
    }
    let x = coefficients(width, w, (w - 224) / 2);
    let y = coefficients(height, h, (h - 224) / 2);
    let mut values = vec![0f32; elements];
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut rows: [Vec<i32>; 8] = std::array::from_fn(|_| vec![0; 224 * 3]);
    let mut cached = [None; 8];
    for (dy, (sy, beta)) in y.into_iter().enumerate() {
        let mut slots = [0; 8];
        for (k, slot) in slots.iter_mut().enumerate() {
            let yy = (sy + k as isize - 3).clamp(0, height as isize - 1) as usize;
            let index = yy % 8;
            *slot = index;
            if cached[index] == Some(yy) {
                continue;
            }
            let row = &mut rows[index];
            for (dx, (sx, alpha)) in x.iter().enumerate() {
                for c in 0..3 {
                    let mut value = 0;
                    for (j, &a) in alpha.iter().enumerate() {
                        let xx = (sx + j as isize - 3).clamp(0, width as isize - 1) as usize;
                        value += bytes[(yy * width + xx) * 3 + c] as i32 * a;
                    }
                    row[dx * 3 + c] = value;
                }
            }
            cached[index] = Some(yy);
        }
        for dx in 0..224 {
            for c in 0..3 {
                let value: i64 = (0..8)
                    .map(|k| rows[slots[k]][dx * 3 + c] as i64 * beta[k] as i64)
                    .sum();
                let pixel = ((value + (1 << 21)) >> 22).clamp(0, 255) as u8;
                let normalized = pixel as f32 / 255.;
                values[c * 224 * 224 + dy * 224 + dx] = (normalized - mean[c]) / std[c];
            }
        }
    }
    Ok(PreparedTensor {
        shape: vec![1, 3, 224, 224],
        values,
        resized_width: w,
    })
}

impl Model {
    pub fn classify_table_png(
        &mut self,
        bytes: &[u8],
        variant: Variant,
        output_type: OutputType,
    ) -> Result<Classification, Error> {
        let (bgr, w, h) = crate::preprocess::decode_png(bytes, 4_000_000)?;
        let tensor = variant.prepare(&bgr, w, h, self.max_elements)?;
        let outputs = self.run(tensor.shape, tensor.values)?;
        if outputs.len() != 1 {
            return Err(error(ErrorCode::Unsupported, "表格分类需要单输出"));
        }
        let output = outputs
            .into_values()
            .next()
            .ok_or_else(|| error(ErrorCode::InvalidInput, "表格分类输出为空"))?;
        if output.shape != [1, 2] {
            return Err(error(ErrorCode::Unsupported, "表格分类输出须为 1x2"));
        }
        decode(&output.values, output_type)
    }
}

#[cfg(test)]
mod row_cache_tests {
    use super::*;
    #[test]
    fn cached_lanczos_rows_match_uncached_integer_accumulation() {
        for (width, height) in [(1, 1), (4, 7), (17, 19), (256, 256), (1024, 512)] {
            let bytes: Vec<u8> = (0..width * height * 3)
                .map(|i| ((i * 37 + i / 13) % 256) as u8)
                .collect();
            let actual = prepare_bgr(&bytes, width, height, 16_000_000).unwrap();
            let ratio = 256. / width.min(height) as f64;
            let w = (width as f64 * ratio).round_ties_even() as usize;
            let h = (height as f64 * ratio).round_ties_even() as usize;
            let xs = coefficients(width, w, (w - 224) / 2);
            let ys = coefficients(height, h, (h - 224) / 2);
            let mut expected = vec![0f32; 3 * 224 * 224];
            for (dy, (sy, beta)) in ys.into_iter().enumerate() {
                for (dx, (sx, alpha)) in xs.iter().enumerate() {
                    for c in 0..3 {
                        let mut value = 0i64;
                        for k in 0..8 {
                            let yy = (sy + k as isize - 3).clamp(0, height as isize - 1) as usize;
                            let mut row = 0i32;
                            for j in 0..8 {
                                let xx =
                                    (sx + j as isize - 3).clamp(0, width as isize - 1) as usize;
                                row += bytes[(yy * width + xx) * 3 + c] as i32 * alpha[j];
                            }
                            value += row as i64 * beta[k] as i64;
                        }
                        let pixel = ((value + (1 << 21)) >> 22).clamp(0, 255) as u8;
                        expected[c * 224 * 224 + dy * 224 + dx] = (pixel as f32 / 255.
                            - [0.485, 0.456, 0.406][c])
                            / [0.229, 0.224, 0.225][c];
                    }
                }
            }
            assert_eq!(actual.values, expected, "{width}x{height}");
        }
    }
}
