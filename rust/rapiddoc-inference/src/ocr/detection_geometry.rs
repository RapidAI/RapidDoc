//! OpenCV FP32 rotating calipers/boxPoints 合同；许可见 licenses/OPENCV-GEOMETRY.txt。
use imageproc::{geometry::convex_hull, point::Point};

pub(crate) fn rectangle(points: &[Point<i32>]) -> Option<([[f64; 2]; 4], f64)> {
    rectangle_angle(points).map(|(points, side, _)| (points, side))
}

pub(crate) fn rectangle_angle(points: &[Point<i32>]) -> Option<([[f64; 2]; 4], f64, f32)> {
    let mut hull = convex_hull(points);
    let n = hull.len();
    if n < 3 {
        return None;
    }
    // 与 OpenCV 逆时针 hull 的初始端点及单调索引循环移位保持一致。
    let first = (0..n).max_by_key(|&i| (hull[i].x, hull[i].y))?;
    hull.rotate_left(first);
    let indices: Vec<_> = hull
        .iter()
        .map(|p| points.iter().position(|q| q == p).unwrap_or(0))
        .collect();
    let min = (0..n).min_by_key(|&i| indices[i])?;
    let max = (0..n).max_by_key(|&i| indices[i])?;
    let ascending = (max + 1) % n == min;
    let start = if ascending { min } else { max };
    if (0..n - 1).all(|i| (indices[(start + i) % n] < indices[(start + i + 1) % n]) == ascending) {
        hull.rotate_left(start);
    }
    let p: Vec<[f32; 2]> = hull.iter().map(|p| [p.x as f32, p.y as f32]).collect();
    calipers(p)
}

fn calipers(p: Vec<[f32; 2]>) -> Option<([[f64; 2]; 4], f64, f32)> {
    let n = p.len();
    let mut v = Vec::with_capacity(n);
    let mut inv = Vec::with_capacity(n);
    let (mut left, mut right, mut top, mut bottom) = (0, 0, 0, 0);
    for i in 0..n {
        if p[i][0] < p[left][0] {
            left = i;
        }
        if p[i][0] > p[right][0] {
            right = i;
        }
        if p[i][1] > p[top][1] {
            top = i;
        }
        if p[i][1] < p[bottom][1] {
            bottom = i;
        }
        let dx = p[(i + 1) % n][0] - p[i][0];
        let dy = p[(i + 1) % n][1] - p[i][1];
        v.push([dx, dy]);
        inv.push((1. / ((dx as f64).powi(2) + (dy as f64).powi(2)).sqrt()) as f32);
    }
    let mut seq = [bottom, right, top, left];
    let mut best = f32::MAX;
    let mut saved = (0, 0, 1f32, 0f32, 0f32, 0f32);
    for _ in 0..n {
        let r = [
            v[seq[0]],
            [v[seq[1]][1], -v[seq[1]][0]],
            [-v[seq[2]][0], -v[seq[2]][1]],
            [-v[seq[3]][1], v[seq[3]][0]],
        ];
        let mut main = 0;
        for i in 1..4 {
            if r[i][1] * r[main][0] - r[i][0] * r[main][1] < 0. {
                main = i;
            }
        }
        let index = seq[main];
        let x = v[index][0] * inv[index];
        let y = v[index][1] * inv[index];
        let (a, b) = match main {
            0 => (x, y),
            1 => (y, -x),
            2 => (-x, -y),
            _ => (-y, x),
        };
        seq[main] = (seq[main] + 1) % n;
        let dx = p[seq[1]][0] - p[seq[3]][0];
        let dy = p[seq[1]][1] - p[seq[3]][1];
        let width = dx * a + dy * b;
        let dx = p[seq[2]][0] - p[seq[0]][0];
        let dy = p[seq[2]][1] - p[seq[0]][1];
        let height = -dx * b + dy * a;
        let area = width * height;
        if area <= best {
            best = area;
            saved = (seq[3], seq[0], a, b, width, height);
        }
    }
    let (left, bottom, a, b, w, h) = saved;
    let c1 = a * p[left][0] + p[left][1] * b;
    let c2 = -b * p[bottom][0] + p[bottom][1] * a;
    let idet = 1. / (a * a - (-b) * b);
    let corner = [(c1 * a - c2 * b) * idet, (a * c2 - (-b) * c1) * idet];
    let u = [a * w, b * w];
    let v = [-b * h, a * h];
    let cx = corner[0] + (u[0] + v[0]) * 0.5;
    let cy = corner[1] + (u[1] + v[1]) * 0.5;
    let mut width = ((v[0] as f64).powi(2) + (v[1] as f64).powi(2)).sqrt() as f32;
    let mut height = ((u[0] as f64).powi(2) + (u[1] as f64).powi(2)).sqrt() as f32;
    let angle = if u[0] == 0. && u[1] > 0. {
        std::mem::swap(&mut width, &mut height);
        -std::f64::consts::FRAC_PI_2
    } else {
        -(u[0] as f64).atan2(u[1] as f64)
    };
    let degrees = (angle * 180. / std::f64::consts::PI) as f32;
    let angle = degrees as f64 * std::f64::consts::PI / 180.;
    let a = angle.sin() as f32 * 0.5;
    let b = angle.cos() as f32 * 0.5;
    let (ah, aw, bh, bw) = (a * height, a * width, b * height, b * width);
    let first = [[cx - ah - bw, cy + bh - aw], [cx + ah - bw, cy - bh - aw]];
    // OpenCV 4 RotatedRect::points reflects the opposite corners; OpenCV 5
    // evaluates all four. Preserve each version's FP32 operation order.
    // Re-evaluating center +/- terms is algebraically equivalent but can shift
    // a rounded FP32 vertex across an integer boundary before Pyclipper casts.
    let mut rect = match crate::crop::sampling() {
        crate::crop::Sampling::OpenCv4 => [
            first[0],
            first[1],
            [2. * cx - first[0][0], 2. * cy - first[0][1]],
            [2. * cx - first[1][0], 2. * cy - first[1][1]],
        ],
        crate::crop::Sampling::OpenCv5 => [
            first[0],
            first[1],
            [cx + ah + bw, cy - bh + aw],
            [cx - ah + bw, cy + bh + aw],
        ],
    }
    .map(|p| p.map(f64::from));
    rect.sort_by(|a, b| a[0].total_cmp(&b[0]));
    if rect[0][1] > rect[1][1] {
        rect.swap(0, 1);
    }
    if rect[2][1] > rect[3][1] {
        rect.swap(2, 3);
    }
    Some((
        [rect[0], rect[2], rect[3], rect[1]],
        width.min(height) as f64,
        degrees,
    ))
}

/// Float input is required by the original custom 45-degree vertex adjustment.
pub(crate) fn rectangle_float(points: &[[f64; 2]]) -> Option<[[f64; 2]; 4]> {
    let original = points
        .iter()
        .map(|p| p.map(|x| x as f32))
        .collect::<Vec<_>>();
    let mut sorted = original.clone();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    sorted.dedup();
    fn cross(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f64 {
        (b[0] - a[0]) as f64 * (c[1] - a[1]) as f64 - (b[1] - a[1]) as f64 * (c[0] - a[0]) as f64
    }
    let mut hull = Vec::new();
    for &p in &sorted {
        while hull.len() > 1 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0. {
            hull.pop();
        }
        hull.push(p)
    }
    let lower = hull.len();
    for &p in sorted.iter().rev().skip(1) {
        while hull.len() > lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0. {
            hull.pop();
        }
        hull.push(p)
    }
    hull.pop();
    if hull.len() < 3 {
        return None;
    }
    let first = (0..hull.len()).max_by(|&a, &b| {
        hull[a][0]
            .total_cmp(&hull[b][0])
            .then(hull[a][1].total_cmp(&hull[b][1]))
    })?;
    hull.rotate_left(first);
    let indices = hull
        .iter()
        .map(|p| original.iter().position(|q| q == p).unwrap_or(0))
        .collect::<Vec<_>>();
    let n = hull.len();
    let min = (0..n).min_by_key(|&i| indices[i])?;
    let max = (0..n).max_by_key(|&i| indices[i])?;
    let ascending = (max + 1) % n == min;
    let start = if ascending { min } else { max };
    if (0..n - 1).all(|i| (indices[(start + i) % n] < indices[(start + i + 1) % n]) == ascending) {
        hull.rotate_left(start)
    }
    calipers(hull).map(|(quad, _, _)| quad)
}
