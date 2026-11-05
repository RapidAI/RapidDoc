//! UNET 固定白色/灰度、2 像素 LINE_AA 绘线。参考 OpenCV drawing.cpp，许可见 OPENCV-DRAWING.txt。
use crate::{error, preprocess::check_size, table_lines::Line};
use rapiddoc_core::{Error, ErrorCode};
const ONE: i64 = 65536;
type Point = [i64; 2];
const SLOPE: [i64; 32] = [
    181, 181, 181, 182, 182, 183, 184, 185, 187, 188, 190, 192, 194, 196, 198, 201, 203, 206, 209,
    211, 214, 218, 221, 224, 227, 231, 235, 238, 242, 246, 250, 254,
];
const FILTER: [i64; 64] = [
    168, 177, 185, 194, 202, 210, 218, 224, 231, 236, 241, 246, 249, 252, 254, 254, 254, 254, 252,
    249, 246, 241, 236, 231, 224, 218, 210, 202, 194, 185, 177, 168, 158, 149, 140, 131, 122, 114,
    105, 97, 89, 82, 75, 68, 62, 56, 50, 45, 40, 36, 32, 28, 25, 22, 19, 16, 14, 12, 11, 9, 8, 7,
    5, 5,
];

pub fn draw(lines: &[Line], width: usize, height: usize) -> Result<Vec<u8>, Error> {
    let count = check_size(width, height, 4_000_000)?;
    if width > rapiddoc_core::resources::limit(4096)
        || height > rapiddoc_core::resources::limit(4096)
        || lines.len() > rapiddoc_core::resources::limit(2000)
    {
        return Err(error(
            ErrorCode::ResourceLimit,
            "UNET 绘图超过尺寸/线段限制",
        ));
    }
    if lines
        .iter()
        .flatten()
        .any(|v| !v.is_finite() || v.abs() > rapiddoc_core::resources::limit(1_000_000.))
    {
        return Err(error(ErrorCode::InvalidInput, "UNET 绘图线段无效"));
    }
    let mut canvas = Canvas {
        pixels: vec![0; count],
        width: width as i64,
        height: height as i64,
    };
    let mut work = 50_000_000usize;
    for line in lines {
        // Python int() 截断到零，再以 thickness=2 的 margin 先裁剪。
        let mut a = [line[0] as i64 + 2, line[1] as i64 + 2];
        let mut b = [line[2] as i64 + 2, line[3] as i64 + 2];
        if !clip(&mut a, &mut b, width as i64 + 4, height as i64 + 4) {
            continue;
        }
        a = a.map(|v| v - 2);
        b = b.map(|v| v - 2);
        let cost = ((a[0] - b[0]).abs() + (a[1] - b[1]).abs() + 8) as usize * 32;
        work = work
            .checked_sub(cost)
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "UNET 绘线工作量超过预算"))?;
        canvas.thick(a.map(|v| v * ONE), b.map(|v| v * ONE));
    }
    Ok(canvas.pixels)
}

fn clip(a: &mut Point, b: &mut Point, width: i64, height: i64) -> bool {
    let code = |p: Point| {
        i32::from(p[0] < 0)
            + 2 * i32::from(p[0] >= width)
            + 4 * i32::from(p[1] < 0)
            + 8 * i32::from(p[1] >= height)
    };
    let (mut ca, mut cb) = (code(*a), code(*b));
    if ca & cb == 0 && ca | cb != 0 {
        if ca & 12 != 0 {
            let y = if ca < 8 { 0 } else { height - 1 };
            a[0] += ((y - a[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            a[1] = y;
            ca = code(*a);
        }
        if cb & 12 != 0 {
            let y = if cb < 8 { 0 } else { height - 1 };
            b[0] += ((y - b[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            b[1] = y;
            cb = code(*b);
        }
        if ca & cb == 0 && ca | cb != 0 {
            if ca != 0 {
                let x = if ca == 1 { 0 } else { width - 1 };
                a[1] += ((x - a[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                a[0] = x;
                ca = 0;
            }
            if cb != 0 {
                let x = if cb == 1 { 0 } else { width - 1 };
                b[1] += ((x - b[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                b[0] = x;
                cb = 0;
            }
        }
    }
    ca | cb == 0
}
struct Canvas {
    pixels: Vec<u8>,
    width: i64,
    height: i64,
}
impl Canvas {
    fn blend(&mut self, x: i64, y: i64, alpha: i64) {
        if x < 0 || y < 0 || x >= self.width || y >= self.height {
            return;
        }
        let pixel = &mut self.pixels[(y * self.width + x) as usize];
        let mut v = *pixel as i64;
        v += ((255 - v) * alpha + 127) >> 8;
        v += ((255 - v) * alpha + 127) >> 8;
        *pixel = v as u8;
    }
    fn aa(&mut self, mut a: Point, mut b: Point) {
        if !clip(&mut a, &mut b, self.width * ONE, self.height * ONE) {
            return;
        }
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let horizontal = dx.abs() > dy.abs();
        let axis = usize::from(!horizontal);
        let other = 1 - axis;
        if b[axis] < a[axis] {
            std::mem::swap(&mut a, &mut b);
        }
        let major = b[axis] - a[axis];
        let minor = b[other] - a[other];
        let step = (minor << 16) / (major | 1);
        b[axis] += ONE;
        let mut remaining = (b[axis] >> 16) - (a[axis] >> 16);
        a[other] += ((step * -(a[axis] & (ONE - 1))) >> 16) + ONE / 2;
        let mut slope = (step >> 11) & 63;
        if step < 0 {
            slope ^= 63;
        }
        slope = if slope & 32 != 0 {
            256
        } else {
            SLOPE[slope as usize]
        };
        let i = (a[axis] >> 9) & 0x78;
        let j = (b[axis] >> 9) & 0x78;
        let t0 = slope << 7;
        let t1 = ((0x78 - i) | 4) * slope;
        let t2 = (j | 4) * slope;
        let mut ep = [0; 9];
        ep[8] = slope;
        ep[1] = (((((j - i) & 0x78) | 4) * slope) >> 8) & 0x1ff;
        ep[3] = ep[1];
        ep[2] = (t1 >> 8) & 0x1ff;
        ep[4] = (((((j - i) + 0x80) | 4) * slope) >> 8) & 0x1ff;
        ep[5] = ((t1 + t0) >> 8) & 0x1ff;
        ep[6] = (t2 >> 8) & 0x1ff;
        ep[7] = ((t2 + t0) >> 8) & 0x1ff;
        let mut position = a[axis] >> 16;
        let mut scanned = 0;
        while remaining >= 0 {
            let sub = (a[other] >> 16) - 1;
            let dist = ((a[other] >> 11) & 31) as usize;
            let start = ((i64::from(scanned >= 2) + 1) & (scanned | 2)) as usize;
            let end = ((i64::from(remaining >= 2) + 1) & (remaining | 2)) as usize;
            let correction = ep[start * 3 + end];
            for (offset, index) in [dist + 32, dist, 63 - dist].into_iter().enumerate() {
                let alpha = ((correction * FILTER[index]) >> 8) & 255;
                if horizontal {
                    self.blend(position, sub + offset as i64, alpha);
                } else {
                    self.blend(sub + offset as i64, position, alpha);
                }
            }
            position += 1;
            a[other] += step;
            scanned += 1;
            remaining -= 1;
        }
    }
    fn thick(&mut self, a: Point, b: Point) {
        let dx = (a[0] - b[0]) as f64 / ONE as f64;
        let dy = (b[1] - a[1]) as f64 / ONE as f64;
        let squared = dx * dx + dy * dy;
        if squared > f64::EPSILON {
            let r = ONE as f64 / squared.sqrt();
            let dp = [
                (dy * r).round_ties_even() as i64,
                (dx * r).round_ties_even() as i64,
            ];
            self.fill(&[
                [a[0] + dp[0], a[1] + dp[1]],
                [a[0] - dp[0], a[1] - dp[1]],
                [b[0] - dp[0], b[1] - dp[1]],
                [b[0] + dp[0], b[1] + dp[1]],
            ]);
        }
        // thickness=2 的 EllipseEx 端帽采用 delta=90，包含首尾重复顶点。
        for p in [a, b] {
            self.fill(&[
                [p[0] + ONE, p[1]],
                [p[0], p[1] + ONE],
                [p[0] - ONE, p[1]],
                [p[0], p[1] - ONE],
                [p[0] + ONE, p[1]],
            ]);
        }
    }
    fn fill(&mut self, points: &[Point]) {
        let n = points.len();
        let mut minimum = 0;
        let mut max_y = points[0][1];
        let mut min_x = points[0][0];
        let mut max_x = min_x;
        let mut previous = points[n - 1];
        for (i, &point) in points.iter().enumerate() {
            if point[1] < points[minimum][1] {
                minimum = i;
            }
            max_y = max_y.max(point[1]);
            min_x = min_x.min(point[0]);
            max_x = max_x.max(point[0]);
            self.aa(previous, point);
            previous = point;
        }
        let ymin = (points[minimum][1] + ONE / 2) >> 16;
        let ymax = ((max_y + ONE / 2) >> 16).min(self.height - 1);
        if ((max_x + ONE / 2) >> 16) < 0
            || ymax < 0
            || ((min_x + ONE / 2) >> 16) >= self.width
            || ymin >= self.height
        {
            return;
        }
        #[derive(Clone, Copy)]
        struct Edge {
            index: usize,
            x: i64,
            dx: i64,
            end: i64,
        }
        let mut edges = [Edge {
            index: minimum,
            x: -ONE,
            dx: 0,
            end: ymin,
        }; 2];
        let mut remaining = n as i64;
        for y in ymin..=ymax {
            if y < ymax || y == ymin {
                for (side, edge) in edges.iter_mut().enumerate() {
                    if y >= edge.end {
                        let direction = if side == 0 { 1 } else { n - 1 };
                        let mut old = edge.index;
                        let mut index = (old + direction) % n;
                        loop {
                            let before = remaining;
                            remaining -= 1;
                            if before <= 0 {
                                break;
                            }
                            let end = (points[index][1] + ONE / 2) >> 16;
                            if end > y {
                                edge.end = end;
                                edge.dx = ((points[index][0] - points[old][0]) * 2 + end - y)
                                    / (2 * (end - y));
                                edge.x = points[old][0];
                                edge.index = index;
                                break;
                            }
                            old = index;
                            index = (index + direction) % n;
                        }
                    }
                }
            }
            if remaining < 0 {
                break;
            }
            if y >= 0 {
                let left = ((edges[0].x.min(edges[1].x) + ONE - 1) >> 16).max(0);
                let right = (edges[0].x.max(edges[1].x) >> 16).min(self.width - 1);
                if left <= right {
                    self.pixels
                        [(y * self.width + left) as usize..=(y * self.width + right) as usize]
                        .fill(255);
                }
            }
            for edge in &mut edges {
                edge.x += edge.dx;
            }
        }
    }
}
