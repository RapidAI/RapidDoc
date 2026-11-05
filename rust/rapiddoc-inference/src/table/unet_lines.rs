//! UNET 类别图→原图尺度闭运算→连通域中心线→补连/延长。
use crate::{
    error,
    preprocess::{check_size, resize_linear},
    table_components, table_lines,
    unet::Mask,
};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

#[derive(Serialize)]
pub struct LineMasks {
    pub width: usize,
    pub height: usize,
    pub horizontal: Vec<u8>,
    pub vertical: Vec<u8>,
}

pub fn prepare(
    mask: &Mask,
    width: usize,
    height: usize,
    morph_close: bool,
) -> Result<LineMasks, Error> {
    let count = check_size(mask.width, mask.height, 4_000_000)?;
    check_size(width, height, 4_000_000)?;
    if mask.classes.len() != count || mask.classes.iter().any(|v| *v > 2) {
        return Err(error(ErrorCode::InvalidInput, "UNET 类别图必须为 0/1/2"));
    }
    if width > rapiddoc_core::resources::limit(4096)
        || height > rapiddoc_core::resources::limit(4096)
        || mask.width > rapiddoc_core::resources::limit(4096)
        || mask.height > rapiddoc_core::resources::limit(4096)
    {
        return Err(error(ErrorCode::ResourceLimit, "UNET 线图边长超过限制"));
    }
    // 复用 uint8 整数插值；三个通道分别存横线、竖线和零，避免两次临时分配。
    let mut channels = Vec::with_capacity(count * 3);
    for &label in &mask.classes {
        channels.extend_from_slice(&[u8::from(label == 1), if label == 2 { 2 } else { 0 }, 0]);
    }
    let resized = resize_linear(&channels, mask.width, mask.height, width, height);
    let horizontal = resized.iter().step_by(3).copied().collect::<Vec<_>>();
    let vertical = resized
        .iter()
        .skip(1)
        .step_by(3)
        .copied()
        .collect::<Vec<_>>();
    let hk = (mask.width as f64).sqrt() * 1.2;
    let vk = (mask.height as f64).sqrt() * 1.2;
    let horizontal = if morph_close {
        close(&horizontal, width, height, hk as usize, false)
    } else {
        horizontal
    };
    let vertical = close(&vertical, width, height, vk as usize, true);
    Ok(LineMasks {
        width,
        height,
        horizontal,
        vertical,
    })
}

pub fn geometry(
    mask: &Mask,
    width: usize,
    height: usize,
    morph_close: bool,
    config: &table_lines::Config,
) -> Result<table_lines::Lines, Error> {
    let masks = prepare(mask, width, height, morph_close)?;
    let rows = table_components::analyze(&masks.horizontal, width, height, Some(0), 50)?.lines;
    let columns = table_components::analyze(&masks.vertical, width, height, Some(1), 30)?.lines;
    table_lines::connect(rows, columns, config)
}

fn close(bytes: &[u8], width: usize, height: usize, kernel: usize, vertical: bool) -> Vec<u8> {
    let dilated = extrema(bytes, width, height, kernel, vertical, true);
    extrema(&dilated, width, height, kernel, vertical, false)
}

/// Exact rectangular min/max with OpenCV's default anchor and neutral borders.
/// Block prefix/suffix extrema avoid per-pixel deque operations and strided
/// comparisons while keeping even/oversized kernels and all uint8 values.
fn extrema(
    bytes: &[u8],
    width: usize,
    height: usize,
    kernel: usize,
    vertical: bool,
    maximum: bool,
) -> Vec<u8> {
    let kernel = kernel.max(1);
    let anchor = kernel / 2;
    let (length, lines) = if vertical {
        (height, width)
    } else {
        (width, height)
    };
    let padded = length + kernel - 1;
    let neutral = if maximum { 0 } else { 255 };
    let choose = |a: u8, b: u8| if maximum { a.max(b) } else { a.min(b) };
    let mut prefix = vec![neutral; padded];
    let mut suffix = vec![neutral; padded];
    let mut output = vec![0; bytes.len()];
    for line in 0..lines {
        let offset = |position: usize| {
            if vertical {
                position * width + line
            } else {
                line * width + position
            }
        };
        let value = |position: usize| {
            if position >= anchor && position - anchor < length {
                bytes[offset(position - anchor)]
            } else {
                neutral
            }
        };
        for start in (0..padded).step_by(kernel) {
            let end = (start + kernel).min(padded);
            let mut current = neutral;
            for i in start..end {
                current = choose(current, value(i));
                prefix[i] = current;
            }
            current = neutral;
            for i in (start..end).rev() {
                current = choose(current, value(i));
                suffix[i] = current;
            }
        }
        for i in 0..length {
            output[offset(i)] = choose(suffix[i], prefix[i + kernel - 1]);
        }
    }
    output
}

#[cfg(test)]
mod extrema_tests {
    use super::*;
    #[test]
    fn block_extrema_preserve_all_values_borders_and_anchors() {
        for (w, h) in [
            (1usize, 1usize),
            (1, 71),
            (71, 1),
            (3, 2),
            (17, 23),
            (53, 31),
        ] {
            let bytes: Vec<u8> = (0..w * h)
                .map(|i| ((i * 73 + i / 17) % 256) as u8)
                .collect();
            for kernel in [0, 1, 2, 3, 4, 7, 18, 64, 99] {
                for vertical in [false, true] {
                    for maximum in [false, true] {
                        let mut expected = vec![0; bytes.len()];
                        let k = kernel.max(1);
                        let anchor = k / 2;
                        for y in 0..h {
                            for x in 0..w {
                                let p = if vertical { y } else { x };
                                let n = if vertical { h } else { w };
                                let mut v = if maximum { 0u8 } else { 255 };
                                for s in p.saturating_sub(anchor)..(p + k - anchor).min(n) {
                                    let b = bytes[if vertical { s * w + x } else { y * w + s }];
                                    v = if maximum { v.max(b) } else { v.min(b) };
                                }
                                expected[y * w + x] = v;
                            }
                        }
                        assert_eq!(
                            extrema(&bytes, w, h, kernel, vertical, maximum),
                            expected,
                            "{w}x{h}, kernel {kernel}, vertical {vertical}, maximum {maximum}"
                        );
                    }
                }
            }
        }
    }
}
