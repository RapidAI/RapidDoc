//! 整数四边形 LINE_8 / 16 位扫描线；参考 OpenCV drawing.cpp，见许可文件。
use image::{GrayImage, Luma};
use imageproc::point::Point;

fn code(p: Point<i32>, w: i32, h: i32) -> u8 {
    u8::from(p.x < 0) + 2 * u8::from(p.x >= w) + 4 * u8::from(p.y < 0) + 8 * u8::from(p.y >= h)
}
fn clip(mut a: Point<i32>, mut b: Point<i32>, w: i32, h: i32) -> (Point<i32>, Point<i32>, bool) {
    let (mut ca, mut cb) = (code(a, w, h), code(b, w, h));
    if ca & cb == 0 && ca | cb != 0 {
        if ca & 12 != 0 {
            let y = if ca < 8 { 0 } else { h - 1 };
            a.x += ((y - a.y) as f64 * (b.x - a.x) as f64 / (b.y - a.y) as f64) as i32;
            a.y = y;
            ca = code(a, w, h);
        }
        if cb & 12 != 0 {
            let y = if cb < 8 { 0 } else { h - 1 };
            b.x += ((y - b.y) as f64 * (b.x - a.x) as f64 / (b.y - a.y) as f64) as i32;
            b.y = y;
            cb = code(b, w, h);
        }
        if ca & cb == 0 && ca | cb != 0 {
            if ca != 0 {
                let x = if ca == 1 { 0 } else { w - 1 };
                a.y += ((x - a.x) as f64 * (b.y - a.y) as f64 / (b.x - a.x) as f64) as i32;
                a.x = x;
                ca = 0;
            }
            if cb != 0 {
                let x = if cb == 1 { 0 } else { w - 1 };
                b.y += ((x - b.x) as f64 * (b.y - a.y) as f64 / (b.x - a.x) as f64) as i32;
                b.x = x;
                cb = 0;
            }
        }
    }
    (a, b, ca | cb == 0)
}

pub(crate) fn fill(mask: &mut GrayImage, points: &[Point<i32>]) {
    let (w, h) = (mask.width() as i32, mask.height() as i32);
    let mut edges = Vec::new();
    for i in 0..points.len() {
        let (a, b) = (points[i], points[(i + 1) % points.len()]);
        let (ca, cb, inside) = clip(a, b, w, h);
        if inside {
            let (mut p, mut q) = (ca, cb);
            if q.x < p.x {
                std::mem::swap(&mut p, &mut q);
            }
            let dx = q.x - p.x;
            let dy = (q.y - p.y).abs();
            let sy = if q.y >= p.y { 1 } else { -1 };
            let vert = dy > dx;
            let (major, minor) = if vert { (dy, dx) } else { (dx, dy) };
            let mut err = major - 2 * minor;
            for _ in 0..=major {
                mask.put_pixel(p.x as u32, p.y as u32, Luma([1]));
                let diagonal = err < 0;
                err += -2 * minor + if diagonal { 2 * major } else { 0 };
                if vert {
                    p.y += sy;
                    if diagonal {
                        p.x += 1;
                    }
                } else {
                    p.x += 1;
                    if diagonal {
                        p.y += sy;
                    }
                }
            }
        }
        if a.y != b.y {
            let (ay, by) = if ca.y != cb.y {
                (ca.y, cb.y)
            } else {
                (a.y, b.y)
            };
            let dx = ((cb.x - ca.x) as i64 * 65536) / (by - ay) as i64;
            let (y0, y1, x) = if a.y < b.y {
                (a.y, b.y, ca.x as i64 * 65536 + (a.y - ay) as i64 * dx)
            } else {
                (b.y, a.y, cb.x as i64 * 65536 + (b.y - by) as i64 * dx)
            };
            edges.push((y0, y1, x, dx));
        }
    }
    for y in 0..h {
        let mut xs: Vec<_> = edges
            .iter()
            .filter(|e| e.0 <= y && y < e.1)
            .map(|e| e.2 + (y - e.0) as i64 * e.3)
            .collect();
        xs.sort_unstable();
        for pair in xs.as_chunks::<2>().0 {
            let left = ((pair[0] + 65535) >> 16).max(0);
            let right = (pair[1] >> 16).min((w - 1) as i64);
            for x in left..=right {
                mask.put_pixel(x as u32, y as u32, Luma([1]));
            }
        }
    }
}
