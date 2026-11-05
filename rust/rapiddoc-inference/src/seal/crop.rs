//! Port of RapidDoc seal_crop.py and the homography branch of seal_det_warp.py.
//! Original PaddleOCR contribution is Apache-2.0; see licenses/PADDLE-SEAL.txt.
use super::{error, P};
use crate::{crop::Crop, layout_geos::Context};
use imageproc::point::Point;
use rapiddoc_core::Error;
use std::path::Path;
pub struct Cropper {
    geos: Context,
}
fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1]]
}
fn norm(p: P) -> f64 {
    (p[0] * p[0] + p[1] * p[1]).sqrt()
}
fn distance(a: P, b: P) -> f64 {
    norm(sub(a, b))
}
fn mean(p: &[P]) -> P {
    let n = p.len() as f64;
    [
        p.iter().map(|p| p[0]).sum::<f64>() / n,
        p.iter().map(|p| p[1]).sum::<f64>() / n,
    ]
}
impl Cropper {
    pub fn load(path: &Path) -> Result<Self, Error> {
        Ok(Self {
            geos: Context::load(path)?,
        })
    }
    pub fn crop(
        &self,
        bytes: &[u8],
        width: usize,
        height: usize,
        points: &[P],
    ) -> Result<Crop, Error> {
        crate::preprocess::check_size(width, height, 64_000_000)?;
        if width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(bytes.len())
            || points.len() < 3
            || points
                .iter()
                .flatten()
                .any(|p| !p.is_finite() || p.abs() > rapiddoc_core::resources::limit(1_000_000.))
        {
            return Err(error("Invalid seal crop input"));
        }
        let points = points
            .iter()
            .map(|p| p.map(|x| x as i32 as f64))
            .collect::<Vec<_>>();
        let integer = points
            .iter()
            .map(|p| Point::new(p[0] as i32, p[1] as i32))
            .collect::<Vec<_>>();
        let (rect, _) = crate::detection_geometry::rectangle(&integer)
            .ok_or_else(|| error("Degenerate seal crop rectangle"))?;
        let rectangle = crate::crop::quadrilateral(
            bytes,
            width,
            height,
            rect.map(|p| p.map(|x| x as f32)),
            64_000_000,
        )?;
        if self
            .geos
            .seal_overlap(&points, &rect)?
            .is_none_or(|iou| iou >= 0.7)
        {
            return Ok(rectangle);
        }
        let sampled = sample_groups(&points);
        let mut unique = sampled.clone();
        unique.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
        unique.dedup();
        if sampled.len() < 4 || unique.len() < 4 {
            return Ok(rectangle);
        }
        let (head, tail) = head_tail(&sampled);
        let n = sampled.len();
        let head_end = (head + 1) % n;
        let tail_end = if (tail + 1) % n == 0 {
            n
        } else {
            (tail + 1) % n
        };
        let padded = sampled
            .iter()
            .chain(sampled.iter())
            .copied()
            .collect::<Vec<_>>();
        let mut top = resample(&padded[head_end..tail_end], 15)?;
        let mut bottom = resample(&padded[tail_end..head_end + n], 15)?;
        if mean(&top)[1] - mean(&bottom)[1] > 0. {
            std::mem::swap(&mut top, &mut bottom)
        }
        let curved = top
            .into_iter()
            .chain(bottom)
            .map(|p| p.map(|x| x as f32 as f64))
            .collect::<Vec<_>>();
        // AutoRectifier catches homography failures and uses the axis-aligned crop
        // of its resampled polygon. Retain that fallback, rather than dropping text.
        match rectify(bytes, width, height, &curved) {
            Ok(crop) => Ok(crop),
            Err(_) => bounding_crop(bytes, width, height, &curved),
        }
    }
}
fn sample_groups(line: &[P]) -> Vec<P> {
    let lengths = line
        .windows(2)
        .map(|p| distance(p[0], p[1]))
        .collect::<Vec<_>>();
    let average = lengths.iter().sum::<f64>() / (lengths.len() as f64 + 1e-8);
    let mut groups = vec![vec![0usize]];
    for (i, &length) in lengths.iter().enumerate() {
        if length < 0.9 * average {
            groups
                .iter_mut()
                .find(|g| g.contains(&i))
                .unwrap()
                .push(i + 1);
        } else {
            groups.push(vec![i + 1]);
        }
    }
    if distance(line[0], *line.last().unwrap()) < 0.9 * average {
        let last = groups.pop().unwrap();
        if let Some(first) = groups.first_mut() {
            first.extend(last)
        } else {
            return vec![];
        }
    }
    groups
        .into_iter()
        .map(|g| {
            let n = g.len() as f64;
            [
                g.iter().map(|&i| line[i][0]).sum::<f64>() / n,
                g.iter().map(|&i| line[i][1]).sum::<f64>() / n,
            ]
            .map(|v| v as i32 as f64)
        })
        .collect()
}
fn angle(a: P, b: P) -> f64 {
    let an = norm(a) + 1e-8;
    let bn = norm(b) + 1e-8;
    ((a[0] / an) * (b[0] / bn) + (a[1] / an) * (b[1] / bn))
        .clamp(-1., 1.)
        .acos()
}
fn head_tail(p: &[P]) -> (usize, usize) {
    let n = p.len();
    if n == 4 {
        let slope = |a: usize, b: usize| {
            let v = sub(p[a], p[b]);
            (v[1] / (v[0] + 1e-8)).abs()
        };
        let (horizontal, vertical) = if slope(1, 0) + slope(3, 2) < slope(2, 1) + slope(0, 3) {
            ([[0, 1], [2, 3]], [[3, 0], [1, 2]])
        } else {
            ([[3, 0], [1, 2]], [[0, 1], [2, 3]])
        };
        let sum = |edges: [[usize; 2]; 2]| {
            edges
                .iter()
                .map(|e| distance(p[e[0]], p[e[1]]))
                .sum::<f64>()
        };
        let chosen = if sum(vertical) > sum(horizontal) * 2. {
            horizontal
        } else {
            vertical
        };
        return (chosen[0][0], chosen[1][0]);
    }
    let edges = (0..n)
        .map(|i| sub(p[(i + 1) % n], p[i]))
        .collect::<Vec<_>>();
    let center = mean(p);
    let dist = (0..n)
        .map(|i| distance(p[i], center).max(distance(p[(i + 1) % n], center)))
        .collect::<Vec<_>>();
    let maximum = dist.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let score = (0..n)
        .map(|i| {
            let before = edges[(i + n - 1) % n];
            let after = edges[(i + 1) % n];
            let position = if n % 2 == 0 && (i == n / 2 - 1 || i == n - 1) {
                0.1
            } else {
                0.
            };
            0.5 * ((angle(edges[i], before) + angle(edges[i], after)) / std::f64::consts::PI)
                + 0.15 * (angle(before, after) / std::f64::consts::PI)
                + 0.35 * dist[i] / maximum
                + position
        })
        .collect::<Vec<_>>();
    let gauss = (0..n - 3)
        .map(|i| {
            let x = i as f64 / (n - 4) as f64;
            (1. / ((2. * std::f64::consts::PI).sqrt() * 0.5))
                * (-((x - 0.5) / 0.5).powi(2) / 2.).exp()
        })
        .collect::<Vec<_>>();
    let gmax = gauss.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut best = f64::NEG_INFINITY;
    let mut head = 0;
    let mut tail = 0;
    for i in 0..n {
        for (j, &g) in gauss.iter().enumerate() {
            let s = score[i] + score[(i + j + 2) % n] * (g / gmax) * 0.3;
            if s > best {
                best = s;
                head = i;
                tail = (i + j + 2) % n;
            }
        }
    }
    if (head + 1) % n > (tail + 1) % n {
        std::mem::swap(&mut head, &mut tail)
    }
    (head, tail)
}
fn resample(line: &[P], n: usize) -> Result<Vec<P>, Error> {
    if line.len() < 2 {
        return Err(error("Invalid seal sideline"));
    }
    let lengths = line
        .windows(2)
        .map(|p| distance(p[0], p[1]))
        .collect::<Vec<_>>();
    let mut cumulative = vec![0.];
    for &d in &lengths {
        cumulative.push(cumulative.last().unwrap() + d)
    }
    let delta = cumulative.last().unwrap() / (n as f64 + 1e-8);
    let mut edge = 0;
    let mut result = vec![line[0]];
    for i in 1..n {
        let position = i as f64 * delta;
        while edge + 1 < cumulative.len() && position >= cumulative[edge + 1] {
            edge += 1
        }
        if edge >= lengths.len() {
            break;
        }
        let ratio = (position - cumulative[edge]) / lengths[edge];
        let d = sub(line[edge + 1], line[edge]);
        result.push([line[edge][0] + d[0] * ratio, line[edge][1] + d[1] * ratio]);
    }
    result.push(*line.last().unwrap());
    Ok(result)
}
// seal_det_warp imports numpy.sqrt, retaining FP32 in the vertical branch.
fn f32_distance(a: P, b: P) -> f64 {
    let dx = a[0] as f32 - b[0] as f32;
    let dy = a[1] as f32 - b[1] as f32;
    (dx * dx + dy * dy).sqrt() as f64
}
fn pairwise_sum<T: Copy + Default + std::ops::Add<Output = T>>(v: &[T]) -> T {
    if v.len() < 8 {
        return v.iter().copied().fold(T::default(), |a, b| a + b);
    }
    if v.len() <= 128 {
        let mut r: [T; 8] = v[..8].try_into().ok().unwrap();
        let stop = v.len() / 8 * 8;
        for chunk in v[8..stop].chunks_exact(8) {
            for i in 0..8 {
                r[i] = r[i] + chunk[i]
            }
        }
        let sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        return v[stop..].iter().copied().fold(sum, |a, b| a + b);
    }
    let split = (v.len() / 2) / 8 * 8;
    pairwise_sum(&v[..split]) + pairwise_sum(&v[split..])
}
fn world(points: &[P], float32: bool) -> Vec<P> {
    let n = points.len();
    let half = n / 2;
    let edges = points
        .windows(2)
        .map(|p| {
            if float32 {
                f32_distance(p[0], p[1])
            } else {
                distance(p[0], p[1])
            }
        })
        .collect::<Vec<_>>();
    let heights = (0..half)
        .map(|i| {
            if float32 {
                f32_distance(points[i], points[n - 1 - i])
            } else {
                distance(points[i], points[n - 1 - i])
            }
        })
        .collect::<Vec<_>>();
    let height = if float32 {
        let values = heights.iter().map(|&v| v as f32).collect::<Vec<_>>();
        (pairwise_sum(&values) / half as f32).round_ties_even() as f64
    } else {
        (pairwise_sum(&heights) / half as f64).round_ties_even()
    };
    let mut xs = vec![0.];
    for i in 0..half - 1 {
        let last = *xs.last().unwrap();
        let a = edges[i];
        let b = edges[edges.len() - 1 - i];
        xs.push(if float32 {
            (last as f32 + (a as f32 + b as f32) / 2.) as f64
        } else {
            last + (a + b) / 2.
        });
    }
    let center = xs.last().unwrap() / 2.;
    #[cfg(test)]
    if let Some(path) = std::env::var_os("RAPIDDOC_SEAL_TRACE") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(file,"{}",serde_json::json!({"world_points":points,"float32":float32,"edges":edges,"heights":heights,"xs":xs})).unwrap();
    }
    xs.iter()
        .map(|&x| [x - center, -height / 2.])
        .chain(xs.iter().rev().map(|&x| [x - center, height / 2.]))
        .collect()
}
fn rectify(bytes: &[u8], width: usize, height: usize, points: &[P]) -> Result<Crop, Error> {
    let bounds = |axis: usize| {
        let min = points.iter().map(|p| p[axis]).fold(f64::INFINITY, f64::min) as i32;
        let max = points
            .iter()
            .map(|p| p[axis])
            .fold(f64::NEG_INFINITY, f64::max) as i32;
        (max - min) as f64
    };
    let span_x = bounds(0);
    if span_x == 0. {
        return Err(error("Vertical seal division by zero"));
    }
    let horizontal = bounds(1) / span_x <= 1.5;
    let mut image = points.to_vec();
    let object = if horizontal {
        // Python converts the rounded FP32 coordinates to a list first, so
        // this branch computes its world geometry in FP64.
        world(&image, false)
    } else {
        image.rotate_left(1);
        let adjusted = image
            .iter()
            .map(|p| {
                [
                    p[1] as f32 as f64,
                    (height as f64 - p[0] - 1.) as f32 as f64,
                ]
            })
            .collect::<Vec<_>>();
        world(&adjusted, true)
            .into_iter()
            .map(|p| [-p[1] as f32 as f64, p[0] as f32 as f64])
            .collect()
    };
    let n = image.len();
    let mut patches = vec![];
    for i in 0..n / 2 - 1 {
        let quad = [image[i], image[i + 1], image[n - 2 - i], image[n - 1 - i]];
        let coords = [
            object[i],
            object[i + 1],
            object[n - 2 - i],
            object[n - 1 - i],
        ]
        .map(|p| p.map(|v| v as f32));
        let axis = usize::from(!horizontal);
        let w = (coords[1][axis] - coords[0][axis]).abs();
        let h = (coords[3][1 - axis] - coords[0][1 - axis]).abs();
        patches.push(warp_linear(bytes, width, height, quad, w as f64, h as f64)?);
    }
    let w = patches.iter().map(|p| p.width).sum::<usize>();
    let h = patches
        .iter()
        .map(|p| p.height)
        .max()
        .ok_or_else(|| error("Empty seal rectification"))?;
    crate::preprocess::check_size(w, h, 64_000_000)?;
    let mut result = Crop {
        width: w,
        height: h,
        bgr: vec![0; w * h * 3],
    };
    let mut offset = 0;
    for patch in patches {
        for y in 0..patch.height {
            let start = (y * w + offset) * 3;
            result.bgr[start..start + patch.width * 3]
                .copy_from_slice(&patch.bgr[y * patch.width * 3..(y + 1) * patch.width * 3]);
        }
        offset += patch.width;
    }
    if !horizontal {
        let mut rotated = vec![0; result.bgr.len()];
        for y in 0..h {
            for x in 0..w {
                let to = (x * h + (h - 1 - y)) * 3;
                rotated[to..to + 3]
                    .copy_from_slice(&result.bgr[(y * w + x) * 3..(y * w + x) * 3 + 3]);
            }
        }
        result = Crop {
            width: h,
            height: w,
            bgr: rotated,
        };
    }
    Ok(result)
}
fn bounding_crop(bytes: &[u8], width: usize, height: usize, points: &[P]) -> Result<Crop, Error> {
    let bounds = |axis: usize, limit: usize| {
        let min = points
            .iter()
            .map(|p| p[axis])
            .fold(f64::INFINITY, f64::min)
            .clamp(0., (limit - 1) as f64) as usize;
        let max = points
            .iter()
            .map(|p| p[axis])
            .fold(f64::NEG_INFINITY, f64::max)
            .clamp(0., (limit - 1) as f64) as usize;
        (min, max)
    };
    let (x0, x1) = bounds(0, width);
    let (y0, y1) = bounds(1, height);
    let w = x1 - x0;
    let h = y1 - y0;
    crate::preprocess::check_size(w, h, 64_000_000)?;
    let mut bgr = Vec::with_capacity(w * h * 3);
    for y in y0..y1 {
        bgr.extend_from_slice(&bytes[(y * width + x0) * 3..(y * width + x1) * 3]);
    }
    Ok(Crop {
        width: w,
        height: h,
        bgr,
    })
}
// OpenCV's DECOMP_LU path, followed by its double precision 3x3 inverse.
fn perspective(points: [P; 4], w: f64, h: f64) -> Result<[f64; 9], Error> {
    let dest = [[0., 0.], [w, 0.], [w, h], [0., h]];
    let mut a = [[0.; 8]; 8];
    let mut b = [0.; 8];
    for i in 0..4 {
        let [x, y] = points[i];
        let [u, v] = dest[i];
        // getPerspectiveTransform multiplies Point2f members before assigning
        // them to its double-precision system, so these four products are FP32.
        a[i] = [
            x,
            y,
            1.,
            0.,
            0.,
            0.,
            (-x as f32 * u as f32) as f64,
            (-y as f32 * u as f32) as f64,
        ];
        a[i + 4] = [
            0.,
            0.,
            0.,
            x,
            y,
            1.,
            (-x as f32 * v as f32) as f64,
            (-y as f32 * v as f32) as f64,
        ];
        b[i] = u;
        b[i + 4] = v;
    }
    for i in 0..8 {
        let mut pivot = i;
        for j in i + 1..8 {
            if a[j][i].abs() > a[pivot][i].abs() {
                pivot = j
            }
        }
        if a[pivot][i].abs() < f64::EPSILON * 100. {
            return Err(error("Singular seal homography"));
        }
        if pivot != i {
            a.swap(i, pivot);
            b.swap(i, pivot)
        }
        let d = -1. / a[i][i];
        for j in i + 1..8 {
            let alpha = a[j][i] * d;
            for k in i + 1..8 {
                a[j][k] += alpha * a[i][k]
            }
            b[j] += alpha * b[i];
        }
    }
    for i in (0..8).rev() {
        let mut s = b[i];
        for j in i + 1..8 {
            s -= a[i][j] * b[j]
        }
        b[i] = s / a[i][i];
    }
    let m = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1.];
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det == 0. {
        return Err(error("Singular seal homography inverse"));
    }
    let reciprocal = 1. / det;
    Ok([
        (m[4] * m[8] - m[5] * m[7]) * reciprocal,
        (m[2] * m[7] - m[1] * m[8]) * reciprocal,
        (m[1] * m[5] - m[2] * m[4]) * reciprocal,
        (m[5] * m[6] - m[3] * m[8]) * reciprocal,
        (m[0] * m[8] - m[2] * m[6]) * reciprocal,
        (m[2] * m[3] - m[0] * m[5]) * reciprocal,
        (m[3] * m[7] - m[4] * m[6]) * reciprocal,
        (m[1] * m[6] - m[0] * m[7]) * reciprocal,
        (m[0] * m[4] - m[1] * m[3]) * reciprocal,
    ])
}
fn warp_linear(
    bytes: &[u8],
    width: usize,
    height: usize,
    points: [P; 4],
    world_w: f64,
    world_h: f64,
) -> Result<Crop, Error> {
    let w = world_w as usize;
    let h = world_h as usize;
    crate::preprocess::check_size(w, h, 64_000_000)?;
    let m = perspective(points, world_w, world_h)?.map(|v| v as f32);
    #[cfg(test)]
    if let Some(path) = std::env::var_os("RAPIDDOC_SEAL_TRACE") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(
            file,
            "{}",
            serde_json::json!({"points":points,"world_w":world_w,"world_h":world_h,"inverse":m})
        )
        .unwrap();
    }
    let (lanes, fused) = vector_shape();
    let vector_end = w / lanes * lanes;
    let mut bgr = vec![0; w * h * 3];
    for y in 0..h {
        let mx = y as f32 * m[1] + m[2];
        let my = y as f32 * m[4] + m[5];
        let mw = y as f32 * m[7] + m[8];
        for x in 0..w {
            let vector = x < vector_end;
            let coordinate = |a: f32, b: f32, c: f32, base: f32| {
                if vector {
                    if fused {
                        a.mul_add(x as f32, base)
                    } else {
                        a * x as f32 + base
                    }
                } else {
                    (x as f32 * a + y as f32 * b) + c
                }
            };
            let den = coordinate(m[6], m[7], m[8], mw);
            let sx = coordinate(m[0], m[1], m[2], mx) / den;
            let sy = coordinate(m[3], m[4], m[5], my) / den;
            if !sx.is_finite() || !sy.is_finite() {
                continue;
            }
            let ix = sx.floor() as i32;
            let iy = sy.floor() as i32;
            let fx = sx - ix as f32;
            let fy = sy - iy as f32;
            let sample = |xx: i32, yy: i32, c: usize| {
                if xx >= 0 && yy >= 0 && (xx as usize) < width && (yy as usize) < height {
                    bytes[(yy as usize * width + xx as usize) * 3 + c] as f32
                } else {
                    0.
                }
            };
            for c in 0..3 {
                let p00 = sample(ix, iy, c);
                let p01 = sample(ix.saturating_add(1), iy, c);
                let p10 = sample(ix, iy.saturating_add(1), c);
                let p11 = sample(ix.saturating_add(1), iy.saturating_add(1), c);
                let interpolate = |a: f32, b: f32, t: f32| {
                    if vector && fused {
                        t.mul_add(b - a, a)
                    } else {
                        a + t * (b - a)
                    }
                };
                let top = interpolate(p00, p01, fx);
                let bottom = interpolate(p10, p11, fx);
                bgr[(y * w + x) * 3 + c] = interpolate(top, bottom, fy)
                    .round_ties_even()
                    .clamp(0., 255.) as u8;
            }
        }
    }
    Ok(Crop {
        width: w,
        height: h,
        bgr,
    })
}

// OpenCV 5's dispatched linear kernel processes two float vectors per block,
// followed by its non-contracted scalar tail. Keep both arithmetic paths.
fn vector_shape() -> (usize, bool) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let fused = std::is_x86_feature_detected!("fma");
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
        {
            return (32, fused);
        }
        if std::is_x86_feature_detected!("avx2") {
            return (16, fused);
        }
        return (8, false);
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        (8, cfg!(target_arch = "aarch64"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the frozen original Python crops and packaged GEOS runtime"]
    fn original_polygon_crops() {
        let root = std::path::PathBuf::from(
            std::env::var_os("RAPIDDOC_SEAL_REFERENCE_DIR")
                .expect("Set RAPIDDOC_SEAL_REFERENCE_DIR"),
        );
        let geos = std::path::PathBuf::from(
            std::env::var_os("RAPIDDOC_SEAL_GEOS_LIBRARY").expect("Set RAPIDDOC_SEAL_GEOS_LIBRARY"),
        );
        let data = std::fs::read(root.join("input.bgr")).unwrap();
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("cases.json")).unwrap()).unwrap();
        let cropper = Cropper::load(&geos).unwrap();
        let width = cases["width"].as_u64().unwrap() as usize;
        let height = cases["height"].as_u64().unwrap() as usize;
        let mut results = vec![];
        for case in cases["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let trace = root.join(format!("{name}.native-plan.jsonl"));
            std::fs::write(&trace, []).unwrap();
            std::env::set_var("RAPIDDOC_SEAL_TRACE", &trace);
            let points: Vec<P> = serde_json::from_value(case["points"].clone()).unwrap();
            let crop = cropper.crop(&data, width, height, &points).unwrap();
            let expected = std::fs::read(root.join(format!("{name}.expected.bgr"))).unwrap();
            std::fs::write(root.join(format!("{name}.actual.bgr")), &crop.bgr).unwrap();
            let different = crop
                .bgr
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a != b)
                .count();
            let max_error = crop
                .bgr
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap_or(0);
            let passed = crop.width as u64 == case["expected_width"].as_u64().unwrap()
                && crop.height as u64 == case["expected_height"].as_u64().unwrap()
                && crop.bgr == expected;
            let row = serde_json::json!({"name":name,"passed":passed,"width":crop.width,"height":crop.height,"different_values":different,"max_error":max_error});
            eprintln!("{row}");
            results.push(row);
        }
        std::env::remove_var("RAPIDDOC_SEAL_TRACE");
        std::fs::write(
            root.join("native-test-summary.json"),
            serde_json::to_vec_pretty(&results).unwrap(),
        )
        .unwrap();
        assert!(results.iter().all(|r|r["passed"]==true),"Original seal crop pixels differ; inspect native-test-summary.json and per-patch plans");
    }
}
