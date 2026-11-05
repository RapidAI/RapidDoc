//! OpenCV Canny(50,150,L1,3) and probabilistic Hough for the OCR table deskew.
//! Derived from OpenCV 4.12 canny.cpp/hough.cpp; see OPENCV-RASTER-NOTICE.txt.
pub fn canny(gray: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut dx = vec![0i32; w * h];
    let mut dy = dx.clone();
    let mut mag = vec![0i32; (w + 2) * (h + 2)];
    let stride = w + 2;
    let pixel = |x: isize, y: isize| {
        gray[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize] as i32
    };
    for y in 0..h {
        for x in 0..w {
            let (xx, yy) = (x as isize, y as isize);
            let mut gx = 0;
            let mut gy = 0;
            for k in -1..=1 {
                let weight = if k == 0 { 2 } else { 1 };
                gx += weight * (pixel(xx + 1, yy + k) - pixel(xx - 1, yy + k));
                gy += weight * (pixel(xx + k, yy + 1) - pixel(xx + k, yy - 1));
            }
            dx[y * w + x] = gx;
            dy[y * w + x] = gy;
            mag[(y + 1) * stride + x + 1] = gx.abs() + gy.abs();
        }
    }
    let mut state = vec![1u8; (w + 2) * (h + 2)];
    let mut stack = vec![];
    for y in 0..h {
        for x in 0..w {
            let p = (y + 1) * stride + x + 1;
            let m = mag[p];
            if m <= 50 {
                continue;
            }
            let gx = dx[y * w + x];
            let gy = dy[y * w + x];
            let tx = gx.abs() * 13573;
            let ty = gy.abs() << 15;
            let keep = if ty < tx {
                m > mag[p - 1] && m >= mag[p + 1]
            } else if ty > tx + (gx.abs() << 16) {
                m > mag[p - stride] && m >= mag[p + stride]
            } else {
                let s = if gx ^ gy < 0 { -1 } else { 1 };
                m > mag[(p as isize - stride as isize - s) as usize]
                    && m > mag[(p as isize + stride as isize + s) as usize]
            };
            if keep {
                state[p] = if m > 150 {
                    stack.push(p);
                    2
                } else {
                    0
                };
            }
        }
    }
    while let Some(p) = stack.pop() {
        for oy in -1..=1 {
            for ox in -1..=1 {
                let q = (p as isize + oy * stride as isize + ox) as usize;
                if state[q] == 0 {
                    state[q] = 2;
                    stack.push(q);
                }
            }
        }
    }
    (0..w * h)
        .map(|i| {
            if state[(i / w + 1) * stride + i % w + 1] == 2 {
                255
            } else {
                0
            }
        })
        .collect()
}
pub fn hough(edges: &[u8], w: usize, h: usize) -> Vec<[i32; 4]> {
    let theta = (std::f64::consts::PI / 1800.) as f32;
    let na = ((std::f64::consts::PI / theta as f64).floor() as usize) + 1;
    let na = if (std::f64::consts::PI - (na - 1) as f64 * theta as f64).abs() < theta as f64 / 2. {
        na - 1
    } else {
        na
    };
    let nr = (w + h) * 2 + 1;
    let offset = (nr - 1) / 2;
    let trig = (0..na)
        .map(|n| {
            let a = n as f64 * theta as f64;
            [a.cos() as f32, a.sin() as f32]
        })
        .collect::<Vec<_>>();
    let mut accum = vec![0i32; na * nr];
    let mut mask = edges.iter().map(|&x| x != 0).collect::<Vec<_>>();
    let mut points = (0..w * h)
        .filter(|&i| mask[i])
        .map(|i| [(i % w) as i64, (i / w) as i64])
        .collect::<Vec<_>>();
    let mut state = u64::MAX;
    let mut lines = vec![];
    let minimum = (30f64.max(w.min(h) as f64 * 0.25)).round_ties_even() as i64;
    for count in (1..=points.len()).rev() {
        state = (state as u32 as u64)
            .wrapping_mul(4164903690)
            .wrapping_add(state >> 32);
        let idx = (state as u32 as usize) % count;
        let p = points[idx];
        points[idx] = points[count - 1];
        if !mask[p[1] as usize * w + p[0] as usize] {
            continue;
        }
        let (mut best, mut angle) = (39, 0);
        for (n, t) in trig.iter().enumerate() {
            let rho = (p[0] as f32 * t[0] + p[1] as f32 * t[1]).round_ties_even() as isize
                + offset as isize;
            let v = &mut accum[n * nr + rho as usize];
            *v += 1;
            if *v > best {
                best = *v;
                angle = n;
            }
        }
        if best < 40 {
            continue;
        }
        let (a, b) = (-trig[angle][1], trig[angle][0]);
        let xf = a.abs() > b.abs();
        let (x0, y0, dx, dy) = if xf {
            (
                p[0],
                (p[1] << 16) + 32768,
                if a > 0. { 1 } else { -1 },
                (b * 65536. / a.abs()).round_ties_even() as i64,
            )
        } else {
            (
                (p[0] << 16) + 32768,
                p[1],
                (a * 65536. / b.abs()).round_ties_even() as i64,
                if b > 0. { 1 } else { -1 },
            )
        };
        let mut ends = [p; 2];
        for k in 0..2 {
            let sign = if k == 0 { 1 } else { -1 };
            let (mut x, mut y, mut gap) = (x0, y0, 0);
            loop {
                let (xx, yy) = if xf { (x, y >> 16) } else { (x >> 16, y) };
                if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                    break;
                }
                if mask[yy as usize * w + xx as usize] {
                    gap = 0;
                    ends[k] = [xx, yy];
                } else {
                    gap += 1;
                    if gap > 8 {
                        break;
                    }
                }
                x += sign * dx;
                y += sign * dy;
            }
        }
        let good = (ends[1][0] - ends[0][0]).abs() >= minimum
            || (ends[1][1] - ends[0][1]).abs() >= minimum;
        for k in 0..2 {
            let sign = if k == 0 { 1 } else { -1 };
            let (mut x, mut y) = (x0, y0);
            loop {
                let (xx, yy) = if xf { (x, y >> 16) } else { (x >> 16, y) };
                let index = yy as usize * w + xx as usize;
                if mask[index] {
                    if good {
                        for (n, t) in trig.iter().enumerate() {
                            let rho = (xx as f32 * t[0] + yy as f32 * t[1]).round_ties_even()
                                as isize
                                + offset as isize;
                            accum[n * nr + rho as usize] -= 1;
                        }
                    }
                    mask[index] = false;
                }
                if [xx, yy] == ends[k] {
                    break;
                }
                x += sign * dx;
                y += sign * dy;
            }
        }
        if good {
            lines.push([
                ends[0][0] as i32,
                ends[0][1] as i32,
                ends[1][0] as i32,
                ends[1][1] as i32,
            ]);
        }
    }
    lines
}
pub fn angle(gray: &[u8], w: usize, h: usize) -> f64 {
    let lines = hough(&canny(gray, w, h), w, h);
    let mut estimates = lines
        .iter()
        .filter_map(|p| {
            let (x, y) = ((p[2] - p[0]) as f64, (p[3] - p[1]) as f64);
            let angle = (y.atan2(x).to_degrees() + 45.).rem_euclid(90.) - 45.;
            (angle.abs() <= 5.).then_some((angle, x.hypot(y)))
        })
        .collect::<Vec<_>>();
    if estimates.len() < 3 {
        return 0.;
    }
    estimates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let midpoint = estimates.iter().map(|p| p.1).sum::<f64>() / 2.;
    let mut total = 0.;
    for (a, l) in estimates {
        total += l;
        if total >= midpoint {
            return if a.abs() < 0.1 { 0. } else { a };
        }
    }
    0.
}
