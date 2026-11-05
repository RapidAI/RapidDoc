//! 页面方向分类：RGB、短边 256、中心裁 224、FP32 ImageNet 归一化。
use crate::preprocess::{check_size, resize_linear, PreparedTensor};
use rapiddoc_core::{Error, ErrorCode};

/// 按 RGB 输入生成 1x3x224x224；缩放后尺寸和张量在分配前受预算限制。
pub fn prepare_rgb(
    bytes: &[u8],
    width: usize,
    height: usize,
    max_pixels: usize,
    max_elements: usize,
) -> Result<PreparedTensor, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    let pixels = check_size(width, height, max_pixels)?;
    if pixels.checked_mul(3) != Some(bytes.len()) {
        return Err(Error {
            code: ErrorCode::InvalidInput,
            message: "方向分类 RGB 字节长度无效".into(),
        });
    }
    let ratio = 256. / width.min(height) as f64;
    let w = (width as f64 * ratio).round_ties_even() as usize;
    let h = (height as f64 * ratio).round_ties_even() as usize;
    check_size(w, h, max_pixels)?;
    let elements = 3 * 224 * 224;
    if elements > max_elements {
        return Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "方向分类张量超过预算".into(),
        });
    }
    let resized = resize_linear(bytes, width, height, w, h);
    let (x0, y0) = ((w - 224) / 2, (h - 224) / 2);
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut values = vec![0.; elements];
    for y in 0..224 {
        for x in 0..224 {
            for channel in 0..3 {
                let value = resized[((y + y0) * w + x + x0) * 3 + channel] as f32 * (1f32 / 255.);
                values[channel * 224 * 224 + y * 224 + x] = (value - mean[channel]) / std[channel];
            }
        }
    }
    Ok(PreparedTensor {
        shape: vec![1, 3, 224, 224],
        values,
        resized_width: w,
    })
}
