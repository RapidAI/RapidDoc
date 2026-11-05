// Pillow 12.2.0 Resample.c, BoxBlur.c and UnsharpMask.c semantics.
// MIT-CMU; see licenses/PILLOW.txt. Kevin Cazabon, copyright 2003.
use super::dib_decode::RgbaImage;
use crate::Result;
fn coeffs(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.);
    let support = 3. * filter_scale;
    (0..output)
        .map(|x| {
            let center = (x as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let mut weights: Vec<_> = (start..end)
                .map(|i| {
                    let x = (i as f64 - center + 0.5) / filter_scale;
                    let sinc = |x: f64| {
                        if x == 0. {
                            1.
                        } else {
                            let x = x * std::f64::consts::PI;
                            x.sin() / x
                        }
                    };
                    if (-3. ..3.).contains(&x) {
                        sinc(x) * sinc(x / 3.)
                    } else {
                        0.
                    }
                })
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
                    .map(|w| (w * (1 << 22) as f64 + if w < 0. { -0.5 } else { 0.5 }) as i32)
                    .collect(),
            )
        })
        .collect()
}
pub fn lanczos(image: &RgbaImage, size: [usize; 2]) -> Result<RgbaImage> {
    if image.size() == size {
        return Ok(image.clone());
    }
    let mut input = image.clone();
    for p in input.pixels.chunks_exact_mut(4) {
        for c in 0..3 {
            let v = u32::from(p[c]) * u32::from(p[3]) + 128;
            p[c] = (((v >> 8) + v) >> 8) as u8;
        }
    }
    let horizontal = if size[0] == input.width {
        input
    } else {
        let coefficients = coeffs(input.width, size[0]);
        let mut out = RgbaImage::blank([size[0], input.height])?;
        for y in 0..input.height {
            for (x, (start, weights)) in coefficients.iter().enumerate() {
                for c in 0..4 {
                    let mut sum = 1i64 << 21;
                    for (i, &w) in weights.iter().enumerate() {
                        sum += i64::from(input.pixels[(y * input.width + start + i) * 4 + c])
                            * i64::from(w);
                    }
                    out.pixels[(y * size[0] + x) * 4 + c] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
        out
    };
    let mut out = if size[1] == horizontal.height {
        horizontal
    } else {
        let coefficients = coeffs(horizontal.height, size[1]);
        let mut out = RgbaImage::blank(size)?;
        for (y, (start, weights)) in coefficients.iter().enumerate() {
            for x in 0..size[0] {
                for c in 0..4 {
                    let mut sum = 1i64 << 21;
                    for (i, &w) in weights.iter().enumerate() {
                        sum += i64::from(horizontal.pixels[((start + i) * size[0] + x) * 4 + c])
                            * i64::from(w);
                    }
                    out.pixels[(y * size[0] + x) * 4 + c] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
        out
    };
    for p in out.pixels.chunks_exact_mut(4) {
        if p[3] != 0 && p[3] != 255 {
            for c in 0..3 {
                p[c] = (u32::from(p[c]) * 255 / u32::from(p[3])).min(255) as u8;
            }
        }
    }
    Ok(out)
}
// The Office backend uses radius=0.6, whose three-pass box radius is <1.
// Retain the same fixed-point edge extension and rounding at each pass.
pub fn unsharp(image: &RgbaImage) -> RgbaImage {
    let radius = 0.6f32;
    let sigma2 = radius * radius / 3.;
    let length = (12. * f64::from(sigma2) + 1.).sqrt() as f32;
    let l = ((f64::from(length) - 1.) / 2.).floor() as f32;
    let mut a = (2. * l + 1.) * (l * (l + 1.) - 3. * sigma2);
    a /= 6. * (sigma2 - (l + 1.) * (l + 1.));
    let radius = l + a;
    debug_assert!(radius < 1.);
    let ww = ((1u32 << 24) as f32 / (radius * 2. + 1.)) as u32;
    let fw = ((1u32 << 24) - ww) / 2;
    let mut blur = image.clone();
    for vertical in [false, true] {
        for _ in 0..3 {
            let mut out = blur.clone();
            for y in 0..blur.height {
                for x in 0..blur.width {
                    for c in 0..4 {
                        let before = if vertical {
                            (y.saturating_sub(1) * blur.width + x) * 4 + c
                        } else {
                            (y * blur.width + x.saturating_sub(1)) * 4 + c
                        };
                        let after = if vertical {
                            ((y + 1).min(blur.height - 1) * blur.width + x) * 4 + c
                        } else {
                            (y * blur.width + (x + 1).min(blur.width - 1)) * 4 + c
                        };
                        let i = (y * blur.width + x) * 4 + c;
                        let bulk = u32::from(blur.pixels[i]) * ww
                            + (u32::from(blur.pixels[before]) + u32::from(blur.pixels[after])) * fw;
                        out.pixels[i] = ((bulk + (1 << 23)) >> 24) as u8;
                    }
                }
            }
            blur = out;
        }
    }
    for (out, &original) in blur.pixels.iter_mut().zip(&image.pixels) {
        let diff = i32::from(original) - i32::from(*out);
        *out = if diff.abs() > 2 {
            (i32::from(original) + diff).clamp(0, 255) as u8
        } else {
            original
        };
    }
    blur
}
