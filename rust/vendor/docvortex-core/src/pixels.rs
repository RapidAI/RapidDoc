//! 对独立位图字节执行裁剪、通道排列及直角旋转，不持有 PDFium 指针。

/// 在连续灰度图上批量取最近邻字形，越界区域沿用 Pillow crop 的黑色填充。
pub fn glyph_masks(
    data: &[u8],
    width: usize,
    height: usize,
    boxes: &[[i64; 4]],
) -> Vec<Option<(Vec<u8>, f64)>> {
    boxes
        .iter()
        .map(|b| {
            let w = b[2] - b[0];
            let h = b[3] - b[1];
            if w < 6 || h < 9 {
                return None;
            }
            let mut mask = vec![0u8; 128];
            let mut count = 0;
            for y in 0..32 {
                for x in 0..32 {
                    let sx = b[0] + (((x as f64 + 0.5) * w as f64) / 32.0) as i64;
                    let sy = b[1] + (((y as f64 + 0.5) * h as f64) / 32.0) as i64;
                    let value = if sx < 0 || sy < 0 || sx >= width as i64 || sy >= height as i64 {
                        0
                    } else {
                        data[sy as usize * width + sx as usize]
                    };
                    if value < 128 {
                        let i = y * 32 + x;
                        mask[i / 8] |= 1 << (i % 8);
                        count += 1;
                    }
                }
            }
            if !(0.08..=0.85).contains(&(count as f64 / 1024.0)) {
                return None;
            }
            Some((mask, w as f64 / h as f64))
        })
        .collect()
}

/// 白色合成后的 RGB 图仅需首个有墨迹行，等价于差分掩码的完整 getbbox。
pub fn blank_top_rgb(data: &[u8], width: usize, height: usize) -> f64 {
    blank_top_channels(data, width, height, 3)
}

/// 无透明元数据的灰度图与 RGB 图使用相同近白阈值，不必先扩展三通道或合成白底。
pub fn blank_top_gray(data: &[u8], width: usize, height: usize) -> f64 {
    blank_top_channels(data, width, height, 1)
}

/// 已验证的独立像素缓冲逐行读取，保留全宽、面积限制和百分比边界。
fn blank_top_channels(data: &[u8], width: usize, height: usize, channels: usize) -> f64 {
    if width == 0 || height == 0 || width.saturating_mul(height) > 4_000_000 {
        return 0.0;
    }
    let top = data
        .chunks_exact(width * channels)
        .position(|row| row.iter().any(|v| *v < 250));
    let fraction = top.unwrap_or(0) as f64 / height as f64;
    if (0.03..=0.12).contains(&fraction) {
        fraction
    } else {
        0.0
    }
}

/// 校验步长和边界，按原方向分类语义返回连续 BGR 像素，颜色编码由调用方完成。
pub fn crop_bgr(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    mode: &str,
    bbox: (usize, usize, usize, usize),
    angle: u16,
) -> Result<(Vec<u8>, usize, usize), &'static str> {
    let (channels, rgb) = match mode {
        "BGR" => (3, false),
        "BGRX" | "BGRA" => (4, false),
        "RGB" => (3, true),
        "RGBX" | "RGBA" => (4, true),
        "L" => (1, false),
        _ => return Err("unsupported bitmap mode"),
    };
    if width.checked_mul(channels).is_none_or(|n| n > stride)
        || height.checked_mul(stride).is_none_or(|n| n > data.len())
    {
        return Err("invalid bitmap buffer or stride");
    }
    let (x0, y0, x1, y1) = bbox;
    if x0 >= x1 || y0 >= y1 || x1 > width || y1 > height {
        return Err("invalid crop bounds");
    }
    if ![0, 90, 180, 270].contains(&angle) {
        return Err("invalid crop angle");
    }
    let (w, h) = (x1 - x0, y1 - y0);
    let (ow, oh) = if angle == 90 || angle == 270 {
        (h, w)
    } else {
        (w, h)
    };
    let len = ow
        .checked_mul(oh)
        .and_then(|n| n.checked_mul(3))
        .ok_or("crop size overflow")?;
    let mut output = vec![0; len];
    if mode == "BGR" && angle == 0 {
        for y in 0..h {
            let start = (y0 + y) * stride + x0 * 3;
            output[y * w * 3..(y + 1) * w * 3].copy_from_slice(&data[start..start + w * 3]);
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                let source = (y0 + y) * stride + (x0 + x) * channels;
                let (dx, dy) = match angle {
                    90 => (y, w - 1 - x),
                    180 => (w - 1 - x, h - 1 - y),
                    270 => (h - 1 - y, x),
                    _ => (x, y),
                };
                let target = (dy * ow + dx) * 3;
                let pixel = if channels == 1 {
                    [data[source]; 3]
                } else if rgb {
                    [data[source + 2], data[source + 1], data[source]]
                } else {
                    [data[source], data[source + 1], data[source + 2]]
                };
                output[target..target + 3].copy_from_slice(&pixel);
            }
        }
    }
    Ok((output, ow, oh))
}

/// 白边检查直接扫描独立位图行，忽略填充通道及行尾；含透明度的格式完整交回 Pillow 合成。
pub fn blank_top_bitmap(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    mode: &str,
) -> Option<f64> {
    let (channels, colors) = match mode {
        "RGB" | "BGR" => (3usize, 3usize),
        "RGBX" | "BGRX" => (4, 3),
        "L" => (1, 1),
        _ => return None,
    };
    if width.checked_mul(channels).is_none_or(|v| v > stride)
        || height.checked_mul(stride).is_none_or(|v| v > data.len())
    {
        return None;
    }
    if width == 0 || height == 0 || width.saturating_mul(height) > 4_000_000 {
        return Some(0.0);
    }
    let top = (0..height).find(|&y| {
        (0..width).any(|x| {
            data[y * stride + x * channels..y * stride + x * channels + colors]
                .iter()
                .any(|&v| v < 250)
        })
    });
    let fraction = top.unwrap_or(0) as f64 / height as f64;
    Some(if (0.03..=0.12).contains(&fraction) {
        fraction
    } else {
        0.0
    })
}
