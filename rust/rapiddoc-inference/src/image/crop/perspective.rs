//! Exceptional four-point transforms: OpenCV 4.13 getPerspectiveTransform and
//! JacobiSVD (modules/imgproc/src/imgwarp.cpp, modules/core/src/lapack.cpp).
//! Attribution and licenses: licenses/OPENCV-WARP.txt.

pub(super) fn inverse(p: [[f64; 2]; 4], w: usize, h: usize) -> [f64; 9] {
    let target = [
        [0., 0.],
        [w as f64, 0.],
        [w as f64, h as f64],
        [0., h as f64],
    ];
    let mut a = [[0.; 9]; 8];
    for i in 0..4 {
        let [x, y] = p[i];
        let [u, v] = target[i];
        let product = |x: f64, y: f64| (-x as f32 * y as f32) as f64;
        a[i] = [x, y, 1., 0., 0., 0., product(x, u), product(y, u), u];
        a[i + 4] = [0., 0., 0., x, y, 1., product(x, v), product(y, v), v];
    }
    let m = solve(a).unwrap_or_else(|| {
        for row in &mut a {
            row[8] = -row[8];
        }
        let mut ata = [[0.; 9]; 9];
        for i in 0..9 {
            for j in i..9 {
                let mut sum = 0.;
                for row in &a {
                    sum += row[i] * row[j];
                }
                ata[i][j] = sum;
                ata[j][i] = sum;
            }
        }
        smallest_left_vector(ata)
    });
    let cof = [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ];
    let det = m[0] * cof[0] + m[1] * cof[3] + m[2] * cof[6];
    if det == 0. {
        [0.; 9]
    } else {
        let d = 1. / det;
        cof.map(|v| v * d)
    }
}

fn solve(original: [[f64; 9]; 8]) -> Option<[f64; 9]> {
    let mut a = original;
    for i in 0..8 {
        let mut pivot = i;
        for j in i + 1..8 {
            if a[j][i].abs() > a[pivot][i].abs() {
                pivot = j;
            }
        }
        if a[pivot][i].abs() < f64::EPSILON * 100. {
            return None;
        }
        a.swap(i, pivot);
        let d = -1. / a[i][i];
        for j in i + 1..8 {
            let alpha = a[j][i] * d;
            for k in i + 1..9 {
                a[j][k] += alpha * a[i][k];
            }
        }
    }
    let mut m = [0.; 9];
    m[8] = 1.;
    for i in (0..8).rev() {
        let mut sum = a[i][8];
        for j in i + 1..8 {
            sum -= a[i][j] * m[j];
        }
        m[i] = sum / a[i][i];
    }
    let mut residual = 0.;
    for row in original {
        let mut value = 0.;
        for i in 0..8 {
            value += row[i] * m[i];
        }
        residual += (value - row[8]).powi(2);
    }
    (residual.sqrt() < 1e-8).then_some(m)
}

fn smallest_left_vector(mut at: [[f64; 9]; 9]) -> [f64; 9] {
    let eps = f64::EPSILON * 10.;
    let norm = |r: [f64; 9]| {
        let mut n = 0.;
        for v in r {
            n += v * v;
        }
        n
    };
    let mut weights = at.map(norm);
    for _ in 0..30 {
        let mut changed = false;
        for i in 0..8 {
            for j in i + 1..9 {
                let (a, b) = (weights[i], weights[j]);
                let mut p = 0.;
                for k in 0..9 {
                    p += at[i][k] * at[j][k];
                }
                if p.abs() <= eps * (a * b).sqrt() {
                    continue;
                }
                p *= 2.;
                let beta = a - b;
                // OpenCV uses its scaled helper, rather than the platform hypot.
                let (hi, lo) = (p.abs().max(beta.abs()), p.abs().min(beta.abs()));
                let gamma = if hi == 0. {
                    0.
                } else {
                    let r = lo / hi;
                    hi * (1. + r * r).sqrt()
                };
                let (c, s) = if beta < 0. {
                    let s = ((gamma - beta) * 0.5 / gamma).sqrt();
                    (p / (gamma * s * 2.), s)
                } else {
                    let c = ((gamma + beta) / (gamma * 2.)).sqrt();
                    (c, p / (gamma * c * 2.))
                };
                let (mut a, mut b) = (0., 0.);
                for k in 0..9 {
                    let x = c * at[i][k] + s * at[j][k];
                    let y = -s * at[i][k] + c * at[j][k];
                    at[i][k] = x;
                    at[j][k] = y;
                    a += x * x;
                    b += y * y;
                }
                weights[i] = a;
                weights[j] = b;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    weights = at.map(|r| norm(r).sqrt());
    for i in 0..8 {
        let mut j = i;
        for k in i + 1..9 {
            if weights[j] < weights[k] {
                j = k;
            }
        }
        if i != j {
            weights.swap(i, j);
            at.swap(i, j);
        }
    }
    let mut state = 0x12345678u64;
    for i in 0..9 {
        let mut sd = weights[i];
        for _ in 0..100 {
            if sd > f64::MIN_POSITIVE {
                break;
            }
            for k in 0..9 {
                state = (state as u32 as u64)
                    .wrapping_mul(4164903690)
                    .wrapping_add(state >> 32);
                at[i][k] = if state & 256 != 0 { 1. / 9. } else { -1. / 9. };
            }
            for _ in 0..2 {
                for j in 0..i {
                    let mut dot = 0.;
                    for k in 0..9 {
                        dot += at[i][k] * at[j][k];
                    }
                    let mut sum = 0.;
                    for k in 0..9 {
                        at[i][k] -= dot * at[j][k];
                        sum += at[i][k].abs();
                    }
                    let scale = if sum > eps * 100. { 1. / sum } else { 0. };
                    for k in 0..9 {
                        at[i][k] *= scale;
                    }
                }
            }
            sd = norm(at[i]).sqrt();
        }
        let scale = if sd > f64::MIN_POSITIVE { 1. / sd } else { 0. };
        for k in 0..9 {
            at[i][k] *= scale;
        }
    }
    at[8]
}
