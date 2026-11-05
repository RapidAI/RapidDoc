//! PP-DocLayoutV3 RGB / cubic resize / NCHW 前处理；尚未包含布局框仲裁。
use crate::preprocess::{check_size, PreparedTensor};
use rapiddoc_core::{Error, ErrorCode};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod cubic_avx;

/// 当前 PP-DocLayoutV3 固定目标 800x800；内部差分允许 32 倍数测试尺寸。
pub fn prepare_rgb(
    bytes: &[u8],
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
    max_elements: usize,
) -> Result<PreparedTensor, Error> {
    let pixels = check_size(width, height, 16_000_000)?;
    if pixels.checked_mul(3) != Some(bytes.len())
        || target_width == 0
        || target_height == 0
        || !target_width.is_multiple_of(32)
        || !target_height.is_multiple_of(32)
    {
        return Err(Error {
            code: ErrorCode::InvalidInput,
            message: "版面 RGB 字节或目标尺寸无效".into(),
        });
    }
    let target_pixels = check_size(target_width, target_height, 16_000_000)?;
    let elements = target_pixels.checked_mul(3).ok_or_else(|| Error {
        code: ErrorCode::ResourceLimit,
        message: "版面张量尺寸溢出".into(),
    })?;
    if elements > max_elements
        || target_width > rapiddoc_core::resources::limit(4096)
        || target_height > rapiddoc_core::resources::limit(4096)
    {
        return Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "版面张量超过预算".into(),
        });
    }
    let image = resize_cubic(bytes, width, height, target_width, target_height);
    let mut values = vec![0.; elements];
    for y in 0..target_height {
        for x in 0..target_width {
            for c in 0..3 {
                values[c * target_pixels + y * target_width + x] =
                    image[(y * target_width + x) * 3 + c] as f32 * (1f32 / 255.);
            }
        }
    }
    Ok(PreparedTensor {
        shape: vec![1, 3, target_height, target_width],
        values,
        resized_width: target_width,
    })
}

/// 匹配已验证的 OpenCV 5/IPP 2026 CPU 路径；极小源图用通用整数路径。
pub(crate) fn resize_cubic(
    bytes: &[u8],
    width: usize,
    height: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    if width >= 4 && height >= 4 {
        return resize_cubic_float(bytes, width, height, w, h);
    }
    let x = coefficients(width, w);
    let y = coefficients(height, h);
    let mut output = vec![0; w * h * 3];
    let mut rows: [Vec<i32>; 4] = std::array::from_fn(|_| vec![0; w * 3]);
    for (dy, (sy, beta)) in y.into_iter().enumerate() {
        for (k, row) in rows.iter_mut().enumerate() {
            let yy = (sy + k as isize - 1).clamp(0, height as isize - 1) as usize;
            for (dx, (sx, alpha)) in x.iter().enumerate() {
                for c in 0..3 {
                    let mut v = 0;
                    for (k, &coefficient) in alpha.iter().enumerate() {
                        let xx = (sx + k as isize - 1).clamp(0, width as isize - 1) as usize;
                        v += bytes[(yy * width + xx) * 3 + c] as i32 * coefficient;
                    }
                    row[dx * 3 + c] = v;
                }
            }
        }
        let beta = beta.map(|v| v as f32 / (2048. * 2048.));
        for i in 0..w * 3 {
            // 当前通用 uint8 路径的 SIMD v_muladd 来自非 FMA 基线编译单元。
            let value = rows[0][i] as f32 * beta[0]
                + (rows[1][i] as f32 * beta[1]
                    + (rows[2][i] as f32 * beta[2] + rows[3][i] as f32 * beta[3]));
            output[dy * w * 3 + i] = value.round_ties_even().clamp(0., 255.) as u8;
        }
    }
    output
}

/// 通过独立脉冲响应与原图差分校准：phase 先转 FP32 再加一，然后 FP64 求系数。
/// 此行为仅作为当前 IPP 后端的兼容合同，不宣称适用于所有 OpenCV 后端。
fn float_coefficients(source: usize, target: usize) -> Vec<(isize, [f32; 4])> {
    (0..target)
        .map(|i| {
            let position = (i as f64 + 0.5) * source as f64 / target as f64 - 0.5;
            let base = position.floor() as isize;
            let x = (((position - base as f64) as f32) + 1.) as f64 - 1.;
            let a = -0.75;
            let c0 = a * x * (1. - x) * (1. - x);
            let c3 = a * x * x * (1. - x);
            let c1 = ((a + 2.) * x - (a + 3.)) * x * x + 1.;
            let c2 = ((a + 2.) * (1. - x) - (a + 3.)) * (1. - x) * (1. - x) + 1.;
            (base, [c0, c1, c2, c3].map(|v| v as f32))
        })
        .collect()
}

fn resize_cubic_float(bytes: &[u8], width: usize, height: usize, w: usize, h: usize) -> Vec<u8> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("sse4.1")
    {
        // SAFETY: runtime detection covers both CPU instructions and OS support.
        return unsafe { resize_cubic_float_avx2(bytes, width, height, w, h) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: only execute this target-feature function after CPU/OS detection.
        return unsafe { resize_cubic_float_fma(bytes, width, height, w, h) };
    }
    resize_cubic_float_impl(bytes, width, height, w, h)
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn resize_cubic_float_fma(
    bytes: &[u8],
    width: usize,
    height: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    resize_cubic_float_impl(bytes, width, height, w, h)
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,sse4.1")]
unsafe fn resize_cubic_float_avx2(
    bytes: &[u8],
    width: usize,
    height: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    resize_cubic_float_kernel::<true>(bytes, width, height, w, h)
}
#[inline(always)]
fn resize_cubic_float_impl(
    bytes: &[u8],
    width: usize,
    height: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    resize_cubic_float_kernel::<false>(bytes, width, height, w, h)
}
#[inline(always)]
fn resize_cubic_float_kernel<const SIMD: bool>(
    bytes: &[u8],
    width: usize,
    height: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    let x = float_coefficients(width, w);
    let positions: Vec<[usize; 4]> = if SIMD {
        x.iter()
            .map(|(sx, _)| {
                std::array::from_fn(|k| {
                    (sx + k as isize - 1).clamp(0, width as isize - 1) as usize * 3
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    let y = float_coefficients(height, h);
    let mut output = vec![0; w * h * 3];
    let mut rows: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.; w * 3]);
    let mut cached = [None; 4];
    for (dy, (sy, beta)) in y.into_iter().enumerate() {
        let edge_order = sy >= height as isize - 1 || sy < 0;
        let mut slots = [0; 4];
        for (k, slot) in slots.iter_mut().enumerate() {
            let yy = (sy + k as isize - 1).clamp(0, height as isize - 1) as usize;
            // Four consecutive source rows cannot collide modulo four. Include
            // the boundary accumulation order in the key: it changes FP32 bits.
            let index = yy % 4;
            *slot = index;
            if cached[index] == Some((yy, edge_order)) {
                continue;
            }
            let row = &mut rows[index];
            if SIMD {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                // SAFETY: instantiated only by the runtime-checked AVX2/FMA wrapper.
                unsafe {
                    cubic_avx::horizontal(bytes, yy * width * 3, &x, &positions, row, edge_order);
                }
            } else {
                for (dx, (sx, alpha)) in x.iter().enumerate() {
                    for c in 0..3 {
                        let mut pixels = [0.; 4];
                        for (k, pixel) in pixels.iter_mut().enumerate() {
                            let xx = (sx + k as isize - 1).clamp(0, width as isize - 1) as usize;
                            *pixel = bytes[(yy * width + xx) * 3 + c] as f32;
                        }
                        // IPP 在复制三条边界行时使用另一种横向积累顺序。
                        row[dx * 3 + c] = if edge_order {
                            pixels[0].mul_add(
                                alpha[0],
                                pixels[1].mul_add(
                                    alpha[1],
                                    pixels[2].mul_add(alpha[2], pixels[3] * alpha[3]),
                                ),
                            )
                        } else {
                            pixels[0].mul_add(alpha[0], pixels[3] * alpha[3])
                                + pixels[1].mul_add(alpha[1], pixels[2] * alpha[2])
                        };
                    }
                }
            }
            cached[index] = Some((yy, edge_order));
        }
        let [r0, r1, r2, r3] = slots.map(|slot| &rows[slot]);
        if SIMD {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: feature check above; four cached rows match the output length.
            unsafe {
                cubic_avx::vertical(
                    [r0.as_slice(), r1.as_slice(), r2.as_slice(), r3.as_slice()],
                    beta,
                    if sy < 0 {
                        0
                    } else if sy >= height as isize - 1 {
                        1
                    } else {
                        2
                    },
                    &mut output[dy * w * 3..(dy + 1) * w * 3],
                );
            }
        } else {
            for i in 0..w * 3 {
                // IPP 的边界复制会先合并落在同一源行的系数。
                let value = if sy < 0 {
                    r0[i].mul_add((beta[0] + beta[1]) + beta[2], r3[i] * beta[3])
                } else if sy >= height as isize - 1 {
                    r0[i].mul_add(beta[0], r1[i] * ((beta[1] + beta[2]) + beta[3]))
                } else {
                    r0[i].mul_add(beta[0], r1[i] * beta[1])
                        + r2[i].mul_add(beta[2], r3[i] * beta[3])
                };
                output[dy * w * 3 + i] = value.round_ties_even().clamp(0., 255.) as u8;
            }
        }
    }
    output
}

fn coefficients(source: usize, target: usize) -> Vec<(isize, [i32; 4])> {
    (0..target)
        .map(|index| {
            let position = ((index as f64 + 0.5) * source as f64 / target as f64 - 0.5) as f32;
            let base = position.floor() as isize;
            let x = position - base as f32;
            let a = -0.75;
            let c0 = ((a * (x + 1.) - 5. * a) * (x + 1.) + 8. * a) * (x + 1.) - 4. * a;
            let c1 = ((a + 2.) * x - (a + 3.)) * x * x + 1.;
            let c2 = ((a + 2.) * (1. - x) - (a + 3.)) * (1. - x) * (1. - x) + 1.;
            (
                base,
                [c0, c1, c2, 1. - c0 - c1 - c2].map(|v| (v * 2048.).round_ties_even() as i32),
            )
        })
        .collect()
}

#[cfg(test)]
mod hardware_tests {
    use super::*;
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn hardware_fma_matches_scalar_layout_pixels() {
        if !std::is_x86_feature_detected!("fma") {
            println!("Hardware FMA unavailable; generic path retained");
            return;
        }
        let mut cases = 0;
        for width in [1, 2, 17, 63, 300] {
            for height in [1, 2, 19, 77, 230] {
                let pixels: Vec<_> = (0..width * height * 3)
                    .map(|i| ((i * 31 + i / 19) % 256) as u8)
                    .collect();
                for (w, h) in [(32, 32), (64, 96)] {
                    let scalar = resize_cubic_float_impl(&pixels, width, height, w, h);
                    let hardware = unsafe { resize_cubic_float_fma(&pixels, width, height, w, h) };
                    assert_eq!(scalar, hardware, "{width}x{height} -> {w}x{h}");
                    assert_eq!(hardware, resize_cubic_float(&pixels, width, height, w, h));
                    cases += 1;
                }
            }
        }
        println!("Hardware FMA layout: {cases} exact boundary/resize cases");
    }
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod avx_boundary_tests {
    use super::*;
    #[test]
    fn vector_cubic_preserves_short_rgb_rows_tails_and_boundaries() {
        if !(std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("sse4.1"))
        {
            return;
        }
        for width in [1, 2, 3, 4, 7, 31, 73] {
            for height in [1, 2, 5, 19] {
                let bytes: Vec<u8> = (0..width * height * 3)
                    .map(|i| ((i * 79 + i / 17) % 256) as u8)
                    .collect();
                for w in [1, 2, 3, 7, 19, 32, 33] {
                    let h = 37;
                    let expected = resize_cubic_float_impl(&bytes, width, height, w, h);
                    let actual = unsafe { resize_cubic_float_avx2(&bytes, width, height, w, h) };
                    assert_eq!(actual, expected, "{width}x{height} -> {w}x{h}");
                }
            }
        }
    }
}
