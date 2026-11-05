//! Pillow 12.2 RGB8 resampling/reduce arithmetic. MIT-CMU attribution: licenses/PILLOW.txt.
use crate::{Error, ErrorCode};

pub fn fail(code: ErrorCode) -> Error {
    let message = if code == ErrorCode::InvalidInput {
        "公式图像字节长度或尺寸无效"
    } else {
        "公式图像尺寸或工作量超过预算"
    };
    Error {
        code,
        message: message.into(),
    }
}
pub struct Rgb {
    pub data: Vec<u8>,
    pub w: usize,
    pub h: usize,
}
pub fn size(w: usize, h: usize) -> Result<usize, Error> {
    w.checked_mul(h)
        .filter(|n| {
            w > 0
                && h > 0
                && w <= crate::resources::limit(65_536)
                && h <= crate::resources::limit(65_536)
                && *n <= crate::resources::limit(16_000_000)
                && n.checked_mul(3)
                    .is_some_and(|bytes| bytes <= isize::MAX as usize)
        })
        .ok_or_else(|| fail(ErrorCode::ResourceLimit))
}
pub struct Work(pub usize);
impl Work {
    fn add(&mut self, n: usize) -> Result<(), Error> {
        self.0 = self
            .0
            .checked_add(n)
            .filter(|n| *n <= crate::resources::limit(256_000_000))
            .ok_or_else(|| fail(ErrorCode::ResourceLimit))?;
        Ok(())
    }
}
fn filter(x: f64, cubic: bool) -> f64 {
    let x = x.abs();
    if !cubic {
        if x < 1. {
            1. - x
        } else {
            0.
        }
    } else if x < 1. {
        (1.5 * x - 2.5) * x * x + 1.
    } else if x < 2. {
        (((x - 5.) * x + 8.) * x - 4.) * -0.5
    } else {
        0.
    }
}
fn coefficients(
    input: usize,
    end: f32,
    output: usize,
    cubic: bool,
    work: &mut Work,
) -> Result<Vec<(usize, Vec<i32>)>, Error> {
    let scale = end as f64 / output as f64;
    let fs = scale.max(1.);
    let support = fs * if cubic { 2. } else { 1. };
    let mut out = Vec::with_capacity(output);
    for i in 0..output {
        let center = (i as f64 + 0.5) * scale;
        let start = ((center - support + 0.5) as isize).max(0) as usize;
        let stop = ((center + support + 0.5) as usize).min(input);
        let count = stop.saturating_sub(start);
        work.add(count)?;
        let mut weights: Vec<_> = (0..count)
            .map(|j| filter((j as f64 + start as f64 - center + 0.5) * (1. / fs), cubic))
            .collect();
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            if sum != 0. {
                *w /= sum;
            }
        }
        let fixed = weights
            .into_iter()
            .map(|w| ((w * (1u32 << 22) as f64) + if w < 0. { -0.5 } else { 0.5 }) as i32)
            .collect();
        out.push((start, fixed));
    }
    Ok(out)
}
fn axis(
    im: Rgb,
    output: usize,
    end: f32,
    vertical: bool,
    cubic: bool,
    work: &mut Work,
) -> Result<Rgb, Error> {
    let input = if vertical { im.h } else { im.w };
    if output == input && end == input as f32 {
        return Ok(im);
    }
    let (w, h) = if vertical {
        (im.w, output)
    } else {
        (output, im.h)
    };
    let pixels = size(w, h)?;
    let coeffs = coefficients(input, end, output, cubic, work)?;
    let cross = if vertical { im.w } else { im.h };
    let taps: usize = coeffs.iter().map(|(_, c)| c.len()).sum();
    work.add(
        taps.checked_mul(cross)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| fail(ErrorCode::ResourceLimit))?,
    )?;
    let mut data = vec![0; pixels * 3];
    // Traverse horizontal output in row order, keeping source/output accesses
    // local. Each channel retains the exact tap order and integer rounding.
    for row in 0..h {
        for col in 0..w {
            let (start, ks) = &coeffs[if vertical { row } else { col }];
            let mut sums = [1i64 << 21; 3];
            for (k, &weight) in ks.iter().enumerate() {
                let pos = if vertical {
                    ((start + k) * im.w + col) * 3
                } else {
                    (row * im.w + start + k) * 3
                };
                let weight = weight as i64;
                sums[0] += im.data[pos] as i64 * weight;
                sums[1] += im.data[pos + 1] as i64 * weight;
                sums[2] += im.data[pos + 2] as i64 * weight;
            }
            let dest = (row * w + col) * 3;
            for c in 0..3 {
                data[dest + c] = (sums[c] >> 22).clamp(0, 255) as u8;
            }
        }
    }
    Ok(Rgb { data, w, h })
}
pub fn resize(
    im: Rgb,
    w: usize,
    h: usize,
    end: (f32, f32),
    cubic: bool,
    work: &mut Work,
) -> Result<Rgb, Error> {
    size(w, h)?;
    // Image.resize 在极窄高图上先纵向缩小，保留其两次量化次序。
    if im.h > im.w.saturating_mul(100) && h < im.h {
        let im = axis(im, h, end.1, true, cubic, work)?;
        axis(im, w, end.0, false, cubic, work)
    } else {
        let im = axis(im, w, end.0, false, cubic, work)?;
        axis(im, h, end.1, true, cubic, work)
    }
}
pub fn reduce(im: Rgb, fx: usize, fy: usize, work: &mut Work) -> Result<Rgb, Error> {
    let (w, h) = (im.w.div_ceil(fx), im.h.div_ceil(fy));
    let mut data = vec![0; size(w, h)? * 3];
    work.add(im.data.len())?;
    for y in 0..h {
        for x in 0..w {
            let (nx, ny) = (fx.min(im.w - x * fx), fy.min(im.h - y * fy));
            let n = nx * ny;
            // 与 division_UINT32 一致，先做float32除法再截断；不是直接整数均值。
            let multiplier = ((1u32 << 24) as f32 / n as f32) as u64;
            for c in 0..3 {
                let mut sum = n as u64 / 2;
                for yy in 0..ny {
                    for xx in 0..nx {
                        sum += im.data[((y * fy + yy) * im.w + x * fx + xx) * 3 + c] as u64;
                    }
                }
                data[(y * w + x) * 3 + c] = ((sum * multiplier) >> 24) as u8;
            }
        }
    }
    Ok(Rgb { data, w, h })
}
