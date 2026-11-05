//! OpenCV 4 INTER_CUBIC/BORDER_REPLICATE: 1/32 coordinates and Q15 taps.
//! Based on imgwarp.cpp 4.13.0; attribution in licenses/OPENCV-WARP.txt.
use rapiddoc_core::Error;
use std::sync::LazyLock;

fn cubic(x: f32) -> [f32; 4] {
    let a = -0.75;
    let z = x + 1.;
    let t = 1. - x;
    let c0 = ((a * z - 5. * a) * z + 8. * a) * z - 4. * a;
    let c1 = ((a + 2.) * x - (a + 3.)) * x * x + 1.;
    let c2 = ((a + 2.) * t - (a + 3.)) * t * t + 1.;
    [c0, c1, c2, 1. - c0 - c1 - c2]
}

static TAPS: LazyLock<[[i16; 16]; 1024]> = LazyLock::new(|| {
    let mut table = [[0i16; 16]; 1024];
    for fy in 0..32 {
        let y = cubic(fy as f32 / 32.);
        for fx in 0..32 {
            let x = cubic(fx as f32 / 32.);
            let taps = &mut table[fy * 32 + fx];
            for row in 0..4 {
                for col in 0..4 {
                    taps[row * 4 + col] = (y[row] * x[col] * 32768.)
                        .round_ties_even()
                        .clamp(-32768., 32767.) as i16;
                }
            }
            let diff = taps.iter().map(|&v| v as i32).sum::<i32>() - 32768;
            if diff != 0 {
                let (mut smallest, mut largest) = (10, 10);
                for row in 2..4 {
                    for col in 2..4 {
                        let i = row * 4 + col;
                        if taps[i] < taps[smallest] {
                            smallest = i;
                        } else if taps[i] > taps[largest] {
                            largest = i;
                        }
                    }
                }
                let i = if diff < 0 { largest } else { smallest };
                taps[i] = (taps[i] as i32 - diff) as i16;
            }
        }
    }
    table
});

/// warpAffine INTER_CUBIC, constant-white border, inverse matrix.
pub(crate) fn affine(
    bytes: &[u8],
    width: usize,
    height: usize,
    m: [f64; 6],
    w: usize,
    h: usize,
) -> Vec<u8> {
    let mut out = vec![255u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            // OpenCV affine blockline uses Q10 arithmetic before the 1/32 table.
            let px = (((m[0] * x as f64 * 1024.).round_ties_even() as i64
                + (m[1] * y as f64 * 1024. + m[2] * 1024.).round_ties_even() as i64
                + 16)
                >> 5) as i32;
            let py = (((m[3] * x as f64 * 1024.).round_ties_even() as i64
                + (m[4] * y as f64 * 1024. + m[5] * 1024.).round_ties_even() as i64
                + 16)
                >> 5) as i32;
            let (ix, iy) = (px >> 5, py >> 5);
            let taps = &TAPS[((py & 31) * 32 + (px & 31)) as usize];
            for c in 0..3 {
                let mut sum = 0i32;
                for ky in 0..4 {
                    for kx in 0..4 {
                        let xx = ix + kx - 1;
                        let yy = iy + ky - 1;
                        let v = if xx < 0 || yy < 0 || xx >= width as i32 || yy >= height as i32 {
                            255
                        } else {
                            bytes[(yy as usize * width + xx as usize) * 3 + c]
                        };
                        sum += v as i32 * taps[(ky * 4 + kx) as usize] as i32;
                    }
                }
                out[(y * w + x) * 3 + c] = ((sum + 16384) >> 15).clamp(0, 255) as u8;
            }
        }
    }
    out
}

pub(super) fn resample(
    bytes: &[u8],
    width: usize,
    height: usize,
    m: [f64; 9],
    w: usize,
    h: usize,
) -> Result<Vec<u8>, Error> {
    let mut out = vec![0u8; w * h * 3];
    // Match warpPerspective's horizontal blocks and coordinate operation order.
    let block_width = (1024 / h.min(16)).min(w);
    for y in 0..h {
        for base in (0..w).step_by(block_width) {
            let x0 = m[0] * base as f64 + m[1] * y as f64 + m[2];
            let y0 = m[3] * base as f64 + m[4] * y as f64 + m[5];
            let w0 = m[6] * base as f64 + m[7] * y as f64 + m[8];
            for offset in 0..block_width.min(w - base) {
                let denominator = w0 + m[6] * offset as f64;
                let reciprocal = if denominator != 0. {
                    32. / denominator
                } else {
                    0.
                };
                let px = ((x0 + m[0] * offset as f64) * reciprocal)
                    .clamp(i32::MIN as f64, i32::MAX as f64)
                    .round_ties_even() as i32;
                let py = ((y0 + m[3] * offset as f64) * reciprocal)
                    .clamp(i32::MIN as f64, i32::MAX as f64)
                    .round_ties_even() as i32;
                let ix = (px >> 5).clamp(i16::MIN as i32, i16::MAX as i32);
                let iy = (py >> 5).clamp(i16::MIN as i32, i16::MAX as i32);
                let taps = &TAPS[((py & 31) * 32 + (px & 31)) as usize];
                let rows: [usize; 4] = std::array::from_fn(|i| {
                    (iy + i as i32 - 1).clamp(0, height as i32 - 1) as usize * width * 3
                });
                let cols: [usize; 4] = std::array::from_fn(|i| {
                    (ix + i as i32 - 1).clamp(0, width as i32 - 1) as usize * 3
                });
                for channel in 0..3 {
                    let mut sum = 0i32;
                    for row in 0..4 {
                        for col in 0..4 {
                            sum += bytes[rows[row] + cols[col] + channel] as i32
                                * taps[row * 4 + col] as i32;
                        }
                    }
                    out[(y * w + base + offset) * 3 + channel] =
                        ((sum + 16384) >> 15).clamp(0, 255) as u8;
                }
            }
        }
    }
    Ok(out)
}
