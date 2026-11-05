//! OpenCV 四点透视裁图，BGR uint8、白底解码后的像素所有权保留在 Rust。
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;
#[path = "crop/opencv4.rs"]
mod opencv4;
#[path = "crop/perspective.rs"]
mod perspective;
pub(crate) fn affine_white(
    bytes: &[u8],
    width: usize,
    height: usize,
    m: [f64; 6],
    w: usize,
    h: usize,
) -> Vec<u8> {
    opencv4::affine(bytes, width, height, m, w, h)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum Sampling {
    #[serde(rename = "opencv4")]
    OpenCv4,
    #[default]
    #[serde(rename = "opencv5")]
    OpenCv5,
}
thread_local! { static SAMPLING: std::cell::Cell<Sampling> = const { std::cell::Cell::new(Sampling::OpenCv5) }; }
pub(crate) fn sampling() -> Sampling {
    SAMPLING.get()
}
/// Scope numeric compatibility per calling thread, restoring it even on panic.
pub fn with_sampling<T>(sampling: Sampling, call: impl FnOnce() -> T) -> T {
    struct Restore(Sampling);
    impl Drop for Restore {
        fn drop(&mut self) {
            SAMPLING.set(self.0);
        }
    }
    let _restore = Restore(SAMPLING.replace(sampling));
    call()
}

/// 模型pipeline的独立RGB PNG识别裁图，旋转门槛为2。
pub fn pipeline_png(
    bytes: &[u8],
    points: [[f32; 2]; 4],
    max_pixels: usize,
) -> Result<(usize, usize, Vec<u8>), Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    pipeline_png_with_limits(bytes, points, max_pixels, max_pixels)
}

/// A full rendered page may contain small recognition crops. Keep the input
/// decode budget separate from the output crop budget, without resizing either.
pub fn pipeline_png_with_limits(
    bytes: &[u8],
    points: [[f32; 2]; 4],
    max_source_pixels: usize,
    max_crop_pixels: usize,
) -> Result<(usize, usize, Vec<u8>), Error> {
    PipelineImage::decode(bytes, max_source_pixels)?.crop(points, max_crop_pixels)
}
/// One source decode for all recognition crops, with the original pixel rules.
pub struct PipelineImage {
    bgr: Vec<u8>,
    width: usize,
    height: usize,
    max_source_pixels: usize,
}
impl PipelineImage {
    pub fn decode(bytes: &[u8], max_source_pixels: usize) -> Result<Self, Error> {
        let (bgr, width, height) =
            super::preprocess::decode_pipeline_rgb(bytes, max_source_pixels)?;
        Ok(Self {
            bgr,
            width,
            height,
            max_source_pixels,
        })
    }
    pub fn crop(
        &self,
        points: [[f32; 2]; 4],
        max_crop_pixels: usize,
    ) -> Result<(usize, usize, Vec<u8>), Error> {
        use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
        let crop = self.crop_pixels(points, max_crop_pixels)?;
        let mut rgb = crop.bgr;
        for p in rgb.as_chunks_mut::<3>().0 {
            p.swap(0, 2);
        }
        let mut png = Vec::new();
        PngEncoder::new(&mut png)
            .write_image(
                &rgb,
                crop.width as u32,
                crop.height as u32,
                ExtendedColorType::Rgb8,
            )
            .map_err(|e| error(ErrorCode::InvalidInput, &e.to_string()))?;
        Ok((crop.width, crop.height, png))
    }
    /// Keep the original perspective transform and rotation, without encoding
    /// an internal crop that the recognizer would immediately decode again.
    pub fn crop_pixels(
        &self,
        points: [[f32; 2]; 4],
        max_crop_pixels: usize,
    ) -> Result<Crop, Error> {
        with_rotation_limits(
            &self.bgr,
            self.width,
            self.height,
            points,
            self.max_source_pixels,
            max_crop_pixels,
            2.,
        )
    }
}

#[derive(Serialize)]
pub struct Crop {
    pub width: usize,
    pub height: usize,
    pub bgr: Vec<u8>,
}
pub struct PngCrop {
    pub width: usize,
    pub height: usize,
    pub png: Vec<u8>,
}
/// Same table crop geometry and source budget as pipeline_png_batch, returning
/// owned BGR buffers for the language-independent model scheduler.
pub fn pipeline_pixels_batch(
    bytes: &[u8],
    points: &[[[f32; 2]; 4]],
) -> Result<Vec<crate::recognition_batch::Image>, Error> {
    if points.len() > rapiddoc_core::resources::limit(1000) {
        return Err(error(ErrorCode::ResourceLimit, "表格识别裁图超过1000"));
    }
    let source = PipelineImage::decode(bytes, 4_000_000)?;
    let mut output = Vec::with_capacity(points.len());
    let mut pixels = 0usize;
    for &q in points {
        let crop = source.crop_pixels(q, 4_000_000)?;
        pixels += crop.width * crop.height;
        if pixels > rapiddoc_core::resources::limit(32_000_000) {
            return Err(error(
                ErrorCode::ResourceLimit,
                "表格识别裁图累计像素超过预算",
            ));
        }
        output.push(crate::recognition_batch::Image {
            bgr: crop.bgr,
            width: crop.width,
            height: crop.height,
        });
    }
    Ok(output)
}
/// 同一表图只解码一次，识别裁图取原RGB图而不是检测遮罩/重试白边图。
pub fn pipeline_png_batch(bytes: &[u8], points: &[[[f32; 2]; 4]]) -> Result<Vec<PngCrop>, Error> {
    use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
    if points.len() > rapiddoc_core::resources::limit(1000) {
        return Err(error(ErrorCode::ResourceLimit, "表格识别裁图超过1000"));
    }
    let (bgr, w, h) = super::preprocess::decode_pipeline_rgb(bytes, 4_000_000)?;
    let mut out = Vec::new();
    let mut pixels = 0usize;
    let mut total = 0usize;
    for &q in points {
        if q.iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > rapiddoc_core::resources::limit(1_000_000.))
        {
            return Err(error(ErrorCode::InvalidInput, "无效表格检测框"));
        }
        let crop = pipeline_quadrilateral(&bgr, w, h, q, 4_000_000)?;
        pixels += crop.width * crop.height;
        if pixels > rapiddoc_core::resources::limit(32_000_000) {
            return Err(error(
                ErrorCode::ResourceLimit,
                "表格识别裁图累计像素超过预算",
            ));
        }
        let mut rgb = crop.bgr;
        for p in rgb.as_chunks_mut::<3>().0 {
            p.swap(0, 2);
        }
        let mut png = Vec::new();
        PngEncoder::new(&mut png)
            .write_image(
                &rgb,
                crop.width as u32,
                crop.height as u32,
                ExtendedColorType::Rgb8,
            )
            .map_err(|e| error(ErrorCode::InvalidInput, e.to_string().as_str()))?;
        total += png.len();
        if total > rapiddoc_core::resources::limit(128_000_000) {
            return Err(error(ErrorCode::ResourceLimit, "表格识别PNG累计超过预算"));
        }
        out.push(PngCrop {
            width: crop.width,
            height: crop.height,
            png,
        });
    }
    Ok(out)
}
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}
fn norm(a: [f32; 2], b: [f32; 2]) -> f32 {
    let x = a[0] - b[0];
    let y = a[1] - b[1];
    (x * x + y * y).sqrt()
}

/// 目标边长为对边距离的截断最大值；高宽比至少 1.5 时逆时针旋转 90°。
pub fn quadrilateral(
    bytes: &[u8],
    width: usize,
    height: usize,
    points: [[f32; 2]; 4],
    max_pixels: usize,
) -> Result<Crop, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    with_rotation_ratio(bytes, width, height, points, max_pixels, 1.5)
}

/// RapidDoc pipeline 使用高宽比2旋转，区别于RapidOCR通用辅助函数的1.5。
pub fn pipeline_quadrilateral(
    bytes: &[u8],
    width: usize,
    height: usize,
    points: [[f32; 2]; 4],
    max_pixels: usize,
) -> Result<Crop, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    with_rotation_ratio(bytes, width, height, points, max_pixels, 2.)
}

fn with_rotation_ratio(
    bytes: &[u8],
    width: usize,
    height: usize,
    points: [[f32; 2]; 4],
    max_pixels: usize,
    rotation_ratio: f64,
) -> Result<Crop, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    with_rotation_limits(
        bytes,
        width,
        height,
        points,
        max_pixels,
        max_pixels,
        rotation_ratio,
    )
}
fn with_rotation_limits(
    bytes: &[u8],
    width: usize,
    height: usize,
    points: [[f32; 2]; 4],
    max_source_pixels: usize,
    max_pixels: usize,
    rotation_ratio: f64,
) -> Result<Crop, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    let pixels = super::preprocess::check_size(width, height, max_source_pixels)?;
    if pixels.checked_mul(3) != Some(bytes.len())
        || points
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > rapiddoc_core::resources::limit(1_000_000.))
    {
        return Err(error(ErrorCode::InvalidInput, "裁图输入无效"));
    }
    let w = norm(points[0], points[1]).max(norm(points[2], points[3])) as usize;
    let h = norm(points[0], points[3]).max(norm(points[1], points[2])) as usize;
    // warpPerspective treats an empty dsize (either side zero) as src.size().
    // Do not reject these detection boxes: Python passes them to OpenCV too.
    let crosses: [f64; 4] = std::array::from_fn(|i| {
        let a = points[i].map(f64::from);
        let b = points[(i + 1) % 4].map(f64::from);
        let c = points[(i + 2) % 4].map(f64::from);
        (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    });
    let convex = crosses.iter().all(|&v| v > 0.) || crosses.iter().all(|&v| v < 0.);
    let (target_w, target_h) = (w, h);
    let empty_size = w == 0 || h == 0;
    let (w, h) = if empty_size { (width, height) } else { (w, h) };
    super::preprocess::check_size(w, h, max_pixels)?;
    let p = points.map(|v| v.map(f64::from));
    let dx1 = p[1][0] - p[2][0];
    let dx2 = p[3][0] - p[2][0];
    let dx3 = p[0][0] - p[1][0] + p[2][0] - p[3][0];
    let dy1 = p[1][1] - p[2][1];
    let dy2 = p[3][1] - p[2][1];
    let dy3 = p[0][1] - p[1][1] + p[2][1] - p[3][1];
    let (g, j) = if dx3 == 0. && dy3 == 0. {
        (0., 0.)
    } else {
        let det = dx1 * dy2 - dx2 * dy1;
        ((dx3 * dy2 - dx2 * dy3) / det, (dx1 * dy3 - dx3 * dy1) / det)
    };
    let matrix = [
        (p[1][0] - p[0][0] + g * p[1][0]) / w as f64,
        (p[3][0] - p[0][0] + j * p[3][0]) / h as f64,
        p[0][0],
        (p[1][1] - p[0][1] + g * p[1][1]) / w as f64,
        (p[3][1] - p[0][1] + j * p[3][1]) / h as f64,
        p[0][1],
        g / w as f64,
        j / h as f64,
        1.,
    ];
    // getPerspectiveTransform 将正向矩阵最后一项定为 1；warpPerspective
    // 反求逆矩阵后直接转 FP32，不再归一化。保留这个比例才能匹配舍入。
    let det = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    let scale = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) / det;
    let matrix = if !convex || empty_size {
        perspective::inverse(p, target_w, target_h)
    } else if scale.is_finite() && matrix.iter().all(|v| v.is_finite()) && scale != 0. {
        matrix.map(|v| v * scale)
    } else {
        perspective::inverse(p, w, h)
    };
    let result = if matrix == [0.; 9] {
        bytes[..3].repeat(w * h)
    } else if let Some(pixels) = translated_pixels(bytes, width, height, matrix, w, h) {
        pixels
    } else {
        match SAMPLING.get() {
            Sampling::OpenCv4 => opencv4::resample(bytes, width, height, matrix, w, h)?,
            Sampling::OpenCv5 => resample(bytes, width, height, matrix.map(|v| v as f32), w, h)?,
        }
    };
    if h as f64 / w as f64 >= rotation_ratio {
        let mut rotated = vec![0; result.len()];
        for y in 0..h {
            for x in 0..w {
                let to = ((w - 1 - x) * h + y) * 3;
                rotated[to..to + 3].copy_from_slice(&result[(y * w + x) * 3..(y * w + x) * 3 + 3]);
            }
        }
        return Ok(Crop {
            width: h,
            height: w,
            bgr: rotated,
        });
    }
    Ok(Crop {
        width: w,
        height: h,
        bgr: result,
    })
}

/// An exact integer translation needs no interpolation in either reference.
/// Retain BORDER_REPLICATE even when a rectangle extends outside the source.
fn translated_pixels(
    bytes: &[u8],
    width: usize,
    height: usize,
    m: [f64; 9],
    w: usize,
    h: usize,
) -> Option<Vec<u8>> {
    if m[0] != 1.
        || m[4] != 1.
        || m[8] != 1.
        || [m[1], m[3], m[6], m[7]].iter().any(|v| *v != 0.)
        || m[2].fract() != 0.
        || m[5].fract() != 0.
    {
        return None;
    }
    let (left, top) = (m[2] as isize, m[5] as isize);
    let mut out = vec![0; w * h * 3];
    for y in 0..h {
        let sy = (top + y as isize).clamp(0, height as isize - 1) as usize;
        let row = &mut out[y * w * 3..(y + 1) * w * 3];
        if left >= 0 && left as usize + w <= width {
            let start = (sy * width + left as usize) * 3;
            row.copy_from_slice(&bytes[start..start + w * 3]);
        } else {
            for x in 0..w {
                let sx = (left + x as isize).clamp(0, width as isize - 1) as usize;
                let start = (sy * width + sx) * 3;
                row[x * 3..x * 3 + 3].copy_from_slice(&bytes[start..start + 3]);
            }
        }
    }
    Some(out)
}

fn resample(
    bytes: &[u8],
    width: usize,
    height: usize,
    matrix: [f32; 9],
    w: usize,
    h: usize,
) -> Result<Vec<u8>, Error> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: the runtime check includes OS support for the required AVX state.
        if [matrix[1], matrix[3], matrix[6], matrix[7]]
            .iter()
            .all(|v| *v == 0.)
        {
            return unsafe { resample_axis_fma(bytes, width, height, matrix, w, h) };
        }
        return unsafe { resample_fma(bytes, width, height, matrix, w, h) };
    }
    resample_impl(bytes, width, height, matrix, w, h)
}

/// Axis-aligned cubic transforms are separable. Retain at most the four source
/// rows needed by the next output row. Each cached FP32 horizontal value uses
/// exactly the original tap/FMA order, as does the final vertical reduction.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn resample_axis_fma(
    bytes: &[u8],
    width: usize,
    height: usize,
    matrix: [f32; 9],
    w: usize,
    h: usize,
) -> Result<Vec<u8>, Error> {
    let denominator = (0f32 * matrix[7] + matrix[8]) as f64;
    if denominator == 0. {
        return Ok(bytes[..3].repeat(w * h));
    }
    let mx = 0f32 * matrix[1] + matrix[2];
    let mut result = vec![0; w * h * 3];
    // Fixed-width tiles bound scratch storage to about 24 KiB on 64-bit hosts.
    for first in (0..w).step_by(256) {
        let tile = (w - first).min(256);
        let columns = (first..first + tile)
            .map(|x| {
                let sx = ((mx as f64 + matrix[0] as f64 * x as f64) / denominator) as f32;
                let ix = sx.floor() as i32;
                let indices: [usize; 4] = std::array::from_fn(|c| {
                    (ix + c as i32 - 1).clamp(0, (width - 1) as i32) as usize * 3
                });
                (indices, weights(sx - ix as f32))
            })
            .collect::<Vec<_>>();
        let mut cache = Vec::<(usize, Vec<[f32; 3]>)>::with_capacity(4);
        for y in 0..h {
            let my = y as f32 * matrix[4] + matrix[5];
            let sy = ((my as f64 + matrix[3] as f64 * 0f64) / denominator) as f32;
            let iy = sy.floor() as i32;
            let wy = weights(sy - iy as f32);
            let rows: [usize; 4] =
                std::array::from_fn(|r| (iy + r as i32 - 1).clamp(0, (height - 1) as i32) as usize);
            cache.retain(|(row, _)| rows.contains(row));
            for &row in &rows {
                if cache.iter().any(|(key, _)| *key == row) {
                    continue;
                }
                let mut values = vec![[0.; 3]; tile];
                for (x, (indices, wx)) in columns.iter().enumerate() {
                    for channel in 0..3 {
                        let mut line = bytes[row * width * 3 + indices[0] + channel] as f32 * wx[0];
                        for col in 1..4 {
                            line = (bytes[row * width * 3 + indices[col] + channel] as f32)
                                .mul_add(wx[col], line);
                        }
                        values[x][channel] = line;
                    }
                }
                cache.push((row, values));
            }
            let values: [&[[f32; 3]]; 4] = std::array::from_fn(|r| {
                cache
                    .iter()
                    .find(|(key, _)| *key == rows[r])
                    .unwrap()
                    .1
                    .as_slice()
            });
            for x in 0..tile {
                for channel in 0..3 {
                    let mut value = 0.;
                    for row in 0..4 {
                        value = values[row][x][channel].mul_add(wy[row], value);
                    }
                    result[(y * w + first + x) * 3 + channel] =
                        value.round_ties_even().clamp(0., 255.) as u8;
                }
            }
        }
    }
    Ok(result)
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn resample_fma(
    bytes: &[u8],
    width: usize,
    height: usize,
    matrix: [f32; 9],
    w: usize,
    h: usize,
) -> Result<Vec<u8>, Error> {
    resample_impl(bytes, width, height, matrix, w, h)
}
// Inline the same arithmetic into the generic and hardware-FMA callers. This
// changes instruction selection, not tap order, contraction rules or precision.
#[inline(always)]
fn resample_impl(
    bytes: &[u8],
    width: usize,
    height: usize,
    matrix: [f32; 9],
    w: usize,
    h: usize,
) -> Result<Vec<u8>, Error> {
    let mut result = vec![0; w * h * 3];
    for y in 0..h {
        let mx = y as f32 * matrix[1] + matrix[2];
        let my = y as f32 * matrix[4] + matrix[5];
        let mz = y as f32 * matrix[7] + matrix[8];
        for x in 0..w {
            let denominator = mz as f64 + matrix[6] as f64 * x as f64;
            let (sx, sy) = if denominator == 0. {
                (0., 0.)
            } else {
                (
                    ((mx as f64 + matrix[0] as f64 * x as f64) / denominator) as f32,
                    ((my as f64 + matrix[3] as f64 * x as f64) / denominator) as f32,
                )
            };
            // BORDER_REPLICATE has a constant footprint beyond the image.
            // Resolve coordinates outside the integer range before flooring;
            // otherwise a near-pole projection overflows tap indices/weights.
            let bounded = |value: f32, size: usize| {
                if value >= i32::MAX as f32 {
                    size as f32 + 1.
                } else if value <= i32::MIN as f32 {
                    -2.
                } else {
                    value
                }
            };
            let (sx, sy) = (bounded(sx, width), bounded(sy, height));
            let ix = sx.floor() as i32;
            let iy = sy.floor() as i32;
            let wx = weights(sx - ix as f32);
            let wy = weights(sy - iy as f32);
            // Coordinates are shared by all channels. Keep the exact FP32
            // tap/FMA order below, but clamp/index each row/column only once.
            let rows: [usize; 4] = std::array::from_fn(|row| {
                iy.wrapping_add(row as i32 - 1)
                    .clamp(0, (height - 1) as i32) as usize
                    * width
                    * 3
            });
            let columns: [usize; 4] = std::array::from_fn(|col| {
                ix.wrapping_add(col as i32 - 1).clamp(0, (width - 1) as i32) as usize * 3
            });
            for channel in 0..3 {
                let mut value = 0.;
                for (row, &weight) in wy.iter().enumerate() {
                    let mut line = 0.;
                    for (col, &weight) in wx.iter().enumerate() {
                        let pixel = bytes[rows[row] + columns[col] + channel] as f32;
                        line = if col == 0 {
                            pixel * weight
                        } else {
                            pixel.mul_add(weight, line)
                        };
                    }
                    value = line.mul_add(weight, value);
                }
                result[(y * w + x) * 3 + channel] = value.round_ties_even().clamp(0., 255.) as u8;
            }
        }
    }
    Ok(result)
}

/// 当前 OpenCV 5 cubic 标量公式，不引入 OpenCV 动态库。
#[inline(always)]
fn weights(alpha: f32) -> [f32; 4] {
    let a = -0.75;
    let a2 = alpha * alpha;
    let b = 1. - alpha;
    let b2 = b * b;
    let w0 = a * (alpha * b2);
    let w3 = a * (a2 * b);
    let w1 = a2.mul_add((a + 2.).mul_add(alpha, -(a + 3.)), 1.);
    [w0, w1, 1. - w0 - w1 - w3, w3]
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    #[test]
    fn integer_translation_matches_both_general_samplers_at_borders() {
        let (width, height) = (11, 9);
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 37 + i / 13) % 256) as u8)
            .collect();
        for left in [-12., -1., 0., 4., 12.] {
            for top in [-11., -1., 0., 8., 13.] {
                for [w, h] in [[1, 1], [17, 13]] {
                    let m = [1., 0., left, 0., 1., top, 0., 0., 1.];
                    let fast = translated_pixels(&pixels, width, height, m, w, h).unwrap();
                    assert_eq!(
                        fast,
                        opencv4::resample(&pixels, width, height, m, w, h).unwrap()
                    );
                    assert_eq!(
                        fast,
                        resample(&pixels, width, height, m.map(|v| v as f32), w, h).unwrap()
                    );
                }
            }
        }
        assert!(translated_pixels(
            &pixels,
            width,
            height,
            [1., 0., 0.5, 0., 1., 0., 0., 0., 1.],
            4,
            4
        )
        .is_none());
    }
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn separable_axis_preserves_all_pixels_and_bounds() {
        if !std::is_x86_feature_detected!("fma") {
            return;
        }
        let mut cases = 0;
        for (width, height) in [(1, 1), (2, 7), (17, 11), (193, 117)] {
            let pixels = (0..width * height * 3)
                .map(|i| ((i * 31 + i / 19) % 256) as u8)
                .collect::<Vec<_>>();
            for i in 0..64 {
                let matrix = [
                    0.1 + i as f32 * 0.017,
                    0.,
                    -4.75 + i as f32 * 0.13,
                    0.,
                    if i % 2 == 0 { 0.7 } else { -0.4 },
                    -3.1 + i as f32 * 0.11,
                    0.,
                    0.,
                    if i % 3 == 0 { 0.8 } else { 1. },
                ];
                let old = unsafe { resample_fma(&pixels, width, height, matrix, 67, 39) }.unwrap();
                let new =
                    unsafe { resample_axis_fma(&pixels, width, height, matrix, 67, 39) }.unwrap();
                assert_eq!(old, new, "{width}x{height} matrix {i}");
                assert_eq!(
                    old,
                    resample(&pixels, width, height, matrix, 67, 39).unwrap()
                );
                cases += 1;
            }
        }
        assert!(unsafe { resample_axis_fma(&[0, 1, 2], 1, 1, [0.; 9], 1, 1) }.is_err());
        let (width, height) = (193, 117);
        let pixels = (0..width * height * 3)
            .map(|i| (i * 31 % 256) as u8)
            .collect::<Vec<_>>();
        let matrix = [0.7, 0., -4.75, 0., 0.7, -3.1, 0., 0., 1.];
        for (w, h) in [(255, 9), (256, 9), (257, 9), (1025, 11)] {
            assert_eq!(
                unsafe { resample_fma(&pixels, width, height, matrix, w, h) }.unwrap(),
                resample(&pixels, width, height, matrix, w, h).unwrap(),
                "tile boundary {w}x{h}"
            );
        }
        let mut timing = [Vec::new(), Vec::new()];
        for i in 0..8 {
            for fast in if i % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let started = std::time::Instant::now();
                let result = if fast {
                    unsafe { resample_axis_fma(&pixels, width, height, matrix, 512, 128) }
                } else {
                    unsafe { resample_fma(&pixels, width, height, matrix, 512, 128) }
                }
                .unwrap();
                std::hint::black_box(result);
                timing[fast as usize].push(started.elapsed().as_secs_f64());
            }
        }
        for t in &mut timing {
            t.sort_by(f64::total_cmp);
        }
        println!("Axis cubic: {cases} exact cases; original median {:.6}s; cached rows median {:.6}s; kernel only",
            (timing[0][3]+timing[0][4])/2.,(timing[1][3]+timing[1][4])/2.);
    }
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn hardware_fma_matches_scalar_crop_pixels() {
        if !std::is_x86_feature_detected!("fma") {
            println!("Hardware FMA unavailable; generic path retained");
            return;
        }
        let (width, height) = (193, 117);
        let pixels: Vec<_> = (0..width * height * 3)
            .map(|i| ((i * 31 + i / 19) % 256) as u8)
            .collect();
        for i in 0..64 {
            let matrix = [
                0.3 + i as f32 * 0.017,
                0.03,
                -4.75 + i as f32 * 0.13,
                -0.01,
                0.7,
                -2.5 + i as f32 * 0.07,
                0.0001,
                0.0003,
                1.,
            ];
            let scalar = resample_impl(&pixels, width, height, matrix, 113, 57).unwrap();
            let hardware =
                unsafe { resample_fma(&pixels, width, height, matrix, 113, 57) }.unwrap();
            assert_eq!(scalar, hardware, "matrix {i}");
            assert_eq!(
                hardware,
                resample(&pixels, width, height, matrix, 113, 57).unwrap()
            );
        }
        let matrix = [0.63, 0.02, -2.75, 0.01, 0.8, 1.5, 0.0001, 0.0002, 1.];
        let (mut scalar, mut hardware) = (vec![], vec![]);
        for i in 0..6 {
            for accelerated in if i % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let started = std::time::Instant::now();
                let output = if accelerated {
                    unsafe { resample_fma(&pixels, width, height, matrix, 256, 128) }
                } else {
                    resample_impl(&pixels, width, height, matrix, 256, 128)
                }
                .unwrap();
                let elapsed = started.elapsed().as_secs_f64();
                std::hint::black_box(output);
                if accelerated {
                    hardware.push(elapsed)
                } else {
                    scalar.push(elapsed)
                }
            }
        }
        scalar.sort_by(f64::total_cmp);
        hardware.sort_by(f64::total_cmp);
        let record = serde_json::json!({"scalar_seconds":scalar,"hardware_seconds":hardware,"scalar_median_seconds":(scalar[2]+scalar[3])/2.,"hardware_median_seconds":(hardware[2]+hardware[3])/2.,"scope":"Same cubic transform arithmetic, 64 exact image comparisons; only runtime FMA instruction selection differs"});
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/fma-crop-kernel-timing.json");
        assert!(!path.exists());
        std::fs::write(path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
        println!("Hardware FMA crop: {record}");
    }

    #[test]
    fn source_and_recognition_crop_have_independent_budgets() {
        let bytes = vec![255; 12 * 12 * 3];
        let small = [[1., 1.], [4., 1.], [4., 3.], [1., 3.]];
        assert!(pipeline_quadrilateral(&bytes, 12, 12, small, 16).is_err());
        let crop = with_rotation_limits(&bytes, 12, 12, small, 144, 16, 2.).unwrap();
        assert_eq!((crop.width, crop.height), (3, 2));
        assert_eq!(crop.bgr, vec![255; 18]);
        let large = [[0., 0.], [10., 0.], [10., 10.], [0., 10.]];
        assert!(with_rotation_limits(&bytes, 12, 12, large, 144, 16, 2.).is_err());
        assert!(with_rotation_limits(&bytes, 12, 12, small, 143, 16, 2.).is_err());
    }
}
