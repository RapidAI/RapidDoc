//! 识别模型 BGR/CHW 前处理。线性插值的整数路径参考 OpenCV resize.cpp，见 licenses/OPENCV-RESIZE.txt。

use image::{ImageFormat, ImageReader};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;
use std::io::Cursor;

/// 当前 PP-OCR 识别合同；最大宽度与像素/张量预算在分配前检查。
pub struct RecognitionConfig {
    pub height: usize,
    pub minimum_width: usize,
    pub maximum_width: usize,
    pub max_pixels: usize,
    pub max_elements: usize,
}
impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            height: 48,
            minimum_width: 320,
            maximum_width: 4096,
            max_pixels: 16_000_000,
            max_elements: 16_000_000,
        }
    }
}

/// 自有 FP32 NCHW 张量；padding 保持归一化后的零值。
#[derive(Serialize)]
pub struct PreparedTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub resized_width: usize,
}

/// 解码 PNG；透明像素以白底合成，尺寸在解码前检查，不猜测 JPEG 等其他编码。
pub fn from_png(bytes: &[u8], config: &RecognitionConfig) -> Result<PreparedTensor, Error> {
    let (bgr, width, height) =
        decode_png(bytes, rapiddoc_core::resources::limit(config.max_pixels))?;
    recognition_bgr(&bgr, width, height, config)
}

/// 解码白底 BGR；调用方可在解码后选择检测或识别前处理。
pub(crate) fn decode_png(
    bytes: &[u8],
    max_pixels: usize,
) -> Result<(Vec<u8>, usize, usize), Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    if bytes.len() > rapiddoc_core::resources::limit(256 * 1024 * 1024) {
        return Err(error(ErrorCode::ResourceLimit, "PNG 输入超过字节预算"));
    }
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(error(ErrorCode::Unsupported, "当前图像接口只接受 PNG"));
    }
    let (width, height) = ImageReader::with_format(Cursor::new(bytes), ImageFormat::Png)
        .into_dimensions()
        .map_err(|e| error(ErrorCode::InvalidInput, e))?;
    check_size(width as usize, height as usize, max_pixels)?;
    let mut reader = ImageReader::with_format(Cursor::new(bytes), ImageFormat::Png);
    if rapiddoc_core::resources::is_python_compatible() {
        reader.no_limits();
    }
    let image = reader
        .decode()
        .map_err(|e| error(ErrorCode::InvalidInput, e))?
        .to_rgba8();
    let mut bgr = Vec::with_capacity(width as usize * height as usize * 3);
    for pixel in image.pixels() {
        let alpha = pixel[3] as u32;
        for channel in [2, 1, 0] {
            bgr.push(((pixel[channel] as u32 * alpha + 255 * (255 - alpha) + 127) / 255) as u8);
        }
    }
    Ok((bgr, width as usize, height as usize))
}

/// pipeline页图/裁图是PIL RGB→ndarray语义，透明色不合成白底，元数据不改原RGB。
pub(crate) fn decode_pipeline_rgb(
    bytes: &[u8],
    max_pixels: usize,
) -> Result<(Vec<u8>, usize, usize), Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    if bytes.len() > rapiddoc_core::resources::limit(64_000_000) {
        return Err(error(ErrorCode::ResourceLimit, "pipeline PNG超过预算"));
    }
    if bytes.len() < 33 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(error(ErrorCode::InvalidInput, "无效pipeline PNG"));
    }
    if bytes[24] != 8 || bytes[25] != 2 {
        return Err(error(ErrorCode::Unsupported, "pipeline仅支持8-bit RGB PNG"));
    }
    let (w, h) = ImageReader::with_format(Cursor::new(bytes), ImageFormat::Png)
        .into_dimensions()
        .map_err(|e| error(ErrorCode::InvalidInput, e))?;
    check_size(w as usize, h as usize, max_pixels)?;
    let mut reader = ImageReader::with_format(Cursor::new(bytes), ImageFormat::Png);
    if rapiddoc_core::resources::is_python_compatible() {
        reader.no_limits();
    }
    let mut rgb = reader
        .decode()
        .map_err(|e| error(ErrorCode::InvalidInput, e))?
        .to_rgb8()
        .into_raw();
    for p in rgb.as_chunks_mut::<3>().0 {
        p.swap(0, 2);
    }
    Ok((rgb, w as usize, h as usize))
}

/// 单图规则与当前 Python 一致：H 固定、W=max(基础宽度,H*宽高比)，向上取整有效宽度。
pub fn recognition_bgr(
    bytes: &[u8],
    width: usize,
    height: usize,
    config: &RecognitionConfig,
) -> Result<PreparedTensor, Error> {
    let pixels = check_size(
        width,
        height,
        rapiddoc_core::resources::limit(config.max_pixels),
    )?;
    if pixels.checked_mul(3) != Some(bytes.len()) {
        return Err(error(ErrorCode::InvalidInput, "BGR 字节长度不匹配"));
    }
    if config.height == 0
        || config.minimum_width == 0
        || rapiddoc_core::resources::limit(config.maximum_width) < config.minimum_width
    {
        return Err(error(ErrorCode::InvalidInput, "识别尺寸配置无效"));
    }
    let ratio = width as f64 / height as f64;
    let max_ratio = ratio.max(config.minimum_width as f64 / config.height as f64);
    let padded = (config.height as f64 * max_ratio) as usize;
    recognition_bgr_padded(bytes, width, height, config, padded)
}

/// 整批宽度已按旧 max_wh_ratio 取整，避免重复乘除造成二次取整。
pub(crate) fn recognition_bgr_padded(
    bytes: &[u8],
    width: usize,
    height: usize,
    config: &RecognitionConfig,
    padded: usize,
) -> Result<PreparedTensor, Error> {
    let pixels = check_size(
        width,
        height,
        rapiddoc_core::resources::limit(config.max_pixels),
    )?;
    if pixels.checked_mul(3) != Some(bytes.len()) || config.height == 0 || padded == 0 {
        return Err(error(ErrorCode::InvalidInput, "识别尺寸配置无效"));
    }
    let ratio = width as f64 / height as f64;
    let resized = (config.height as f64 * ratio).ceil() as usize;
    let resized = resized.min(padded);
    if padded > rapiddoc_core::resources::limit(config.maximum_width) || resized == 0 {
        return Err(error(ErrorCode::ResourceLimit, "识别宽度超过限制"));
    }
    let elements = config
        .height
        .checked_mul(padded)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "张量尺寸溢出"))?;
    if elements > config.max_elements {
        return Err(error(ErrorCode::ResourceLimit, "识别张量超过限制"));
    }
    let data = resize_linear(bytes, width, height, resized, config.height);
    let mut values = vec![0f32; elements];
    for y in 0..config.height {
        for x in 0..resized {
            for channel in 0..3 {
                values[channel * config.height * padded + y * padded + x] =
                    (data[(y * resized + x) * 3 + channel] as f32 / 255. - 0.5) / 0.5;
            }
        }
    }
    Ok(PreparedTensor {
        shape: vec![1, 3, config.height, padded],
        values,
        resized_width: resized,
    })
}

/// 在任何像素缓冲分配前校验输入尺寸与乘法溢出。
pub(crate) fn check_size(width: usize, height: usize, limit: usize) -> Result<usize, Error> {
    let limit = rapiddoc_core::resources::limit(limit);
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "图像尺寸溢出"))?;
    if width == 0 || height == 0 {
        return Err(error(ErrorCode::InvalidInput, "图像尺寸为空"));
    }
    if pixels > limit {
        return Err(error(ErrorCode::ResourceLimit, "图像超过像素预算"));
    }
    if pixels
        .checked_mul(4)
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(error(
            ErrorCode::ResourceLimit,
            "Image buffer size is not representable",
        ));
    }
    Ok(pixels)
}

/// 准备半像素坐标和 11 位定点权重；水平边缘固定到边界像素，纵向保留重复行权重。
fn weights(source: usize, target: usize, horizontal: bool) -> Vec<(usize, usize, i32, i32)> {
    (0..target)
        .map(|index| {
            let position = ((index as f64 + 0.5) * (source as f64 / target as f64) - 0.5) as f32;
            let base = position.floor() as isize;
            let mut fraction = position - base as f32;
            if horizontal && (base < 0 || base >= source as isize - 1) {
                fraction = 0.;
            }
            let first = base.clamp(0, source as isize - 1) as usize;
            let last = (base + 1).clamp(0, source as isize - 1) as usize;
            (
                first,
                last,
                ((1. - fraction) * 2048.).round_ties_even() as i32,
                (fraction * 2048.).round_ties_even() as i32,
            )
        })
        .collect()
}

/// 保留 OpenCV uint8 双线性路径的分阶段定点舍入，避免用通用图像滤镜改变前处理像素。
pub(crate) fn resize_linear(
    bytes: &[u8],
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Vec<u8> {
    if width == target_width && height == target_height {
        return bytes.to_vec();
    }
    let xs = weights(width, target_width, true);
    let ys = weights(height, target_height, false);
    let mut result = vec![0; target_width * target_height * 3];
    for (y, &(y0, y1, b0, b1)) in ys.iter().enumerate() {
        for (x, &(x0, x1, a0, a1)) in xs.iter().enumerate() {
            for channel in 0..3 {
                let top = bytes[(y0 * width + x0) * 3 + channel] as i32 * a0
                    + bytes[(y0 * width + x1) * 3 + channel] as i32 * a1;
                let bottom = bytes[(y1 * width + x0) * 3 + channel] as i32 * a0
                    + bytes[(y1 * width + x1) * 3 + channel] as i32 * a1;
                result[(y * target_width + x) * 3 + channel] =
                    ((((b0 * (top >> 4)) >> 16) + ((b1 * (bottom >> 4)) >> 16) + 2) >> 2) as u8;
            }
        }
    }
    result
}

/// 无效输入与资源预算保持不同错误码。
fn error(code: ErrorCode, message: impl std::fmt::Display) -> Error {
    Error {
        code,
        message: message.to_string(),
    }
}
