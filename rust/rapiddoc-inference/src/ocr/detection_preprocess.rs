//! 文字检测前处理；保留 RapidDoc 的 BGR/ImageNet 配置和 32 倍数 ties-even 对齐。

use crate::preprocess::{check_size, resize_linear, PreparedTensor};
use rapiddoc_core::{Error, ErrorCode};

#[derive(Clone, Copy)]
pub enum LimitType {
    Minimum,
    Maximum,
}

/// 原始默认来自 RapidDoc 的 detector 配置，不使用库内另一个 mean/std 默认值。
pub struct DetectionConfig {
    pub side: usize,
    pub mode: LimitType,
    pub mean: [f64; 3],
    pub std: [f64; 3],
    pub maximum_side: usize,
    pub max_pixels: usize,
    pub max_elements: usize,
}
impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            side: 960,
            mode: LimitType::Maximum,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            maximum_side: 4096,
            max_pixels: 16_000_000,
            max_elements: 16_000_000,
        }
    }
}

/// 将 BGR uint8 转为检测 NCHW。先在 FP32 中乘 scale，再以 FP64 mean/std 计算后转回 FP32。
pub fn prepare(
    bytes: &[u8],
    width: usize,
    height: usize,
    config: &DetectionConfig,
) -> Result<PreparedTensor, Error> {
    let pixels = check_size(
        width,
        height,
        rapiddoc_core::resources::limit(config.max_pixels),
    )?;
    if pixels.checked_mul(3) != Some(bytes.len()) {
        return Err(error(ErrorCode::InvalidInput, "检测 BGR 字节长度不匹配"));
    }
    if config.side == 0
        || config.maximum_side < 32
        || config.mean.iter().any(|v| !v.is_finite())
        || config.std.iter().any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err(error(ErrorCode::InvalidInput, "检测配置无效"));
    }
    let edge = match config.mode {
        LimitType::Maximum => width.max(height),
        LimitType::Minimum => width.min(height),
    };
    let ratio = match config.mode {
        LimitType::Maximum if edge > config.side => config.side as f64 / edge as f64,
        LimitType::Minimum if edge < config.side => config.side as f64 / edge as f64,
        _ => 1.,
    };
    let target_width = aligned(width, ratio, config.maximum_side)?;
    let target_height = aligned(height, ratio, config.maximum_side)?;
    let elements = target_width
        .checked_mul(target_height)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "检测尺寸溢出"))?;
    if elements > config.max_elements {
        return Err(error(ErrorCode::ResourceLimit, "检测张量超过预算"));
    }
    let resized = resize_linear(bytes, width, height, target_width, target_height);
    let mut values = vec![0.; elements];
    for y in 0..target_height {
        for x in 0..target_width {
            for channel in 0..3 {
                let scaled = resized[(y * target_width + x) * 3 + channel] as f32 * (1f32 / 255.);
                let value = ((scaled as f64 - config.mean[channel]) / config.std[channel]) as f32;
                if !value.is_finite() {
                    return Err(error(ErrorCode::InvalidInput, "检测归一化产生非有限值"));
                }
                values[channel * target_height * target_width + y * target_width + x] = value;
            }
        }
    }
    Ok(PreparedTensor {
        shape: vec![1, 3, target_height, target_width],
        values,
        resized_width: target_width,
    })
}

/// 先截断缩放尺寸再以 ties-even 取最近的 32 倍数，与当前 Python 计算顺序一致。
fn aligned(size: usize, ratio: f64, maximum: usize) -> Result<usize, Error> {
    let value = ((size as f64 * ratio).trunc() / 32.).round_ties_even() * 32.;
    if !value.is_finite() || value > maximum as f64 {
        return Err(error(ErrorCode::ResourceLimit, "检测边长超过预算"));
    }
    if value < 32. {
        return Err(error(ErrorCode::InvalidInput, "检测对齐后的边长为零"));
    }
    Ok(value as usize)
}

/// 检测参数错误不能伪装为无文字页面。
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}
