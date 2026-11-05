//! 整数线段、多边形掩码和抗锯齿推理线；定点规则与参考后端一致。
const ONE: i64 = 65536;
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

/// 先处理上下边界再处理左右边界，端点转换向零截断。
fn clip(width: i64, height: i64, mut a: [i64; 2], mut b: [i64; 2]) -> (bool, [i64; 2], [i64; 2]) {
    let right = width - 1;
    let bottom = height - 1;
    let code = |p: [i64; 2]| {
        (p[0] < 0) as i32
            + (p[0] > right) as i32 * 2
            + (p[1] < 0) as i32 * 4
            + (p[1] > bottom) as i32 * 8
    };
    let mut c1 = code(a);
    let mut c2 = code(b);
    if c1 & c2 == 0 && c1 | c2 != 0 {
        if c1 & 12 != 0 {
            let y = if c1 < 8 { 0 } else { bottom };
            a[0] += ((y - a[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            a[1] = y;
            c1 = (a[0] < 0) as i32 + (a[0] > right) as i32 * 2;
        }
        if c2 & 12 != 0 {
            let y = if c2 < 8 { 0 } else { bottom };
            b[0] += ((y - b[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            b[1] = y;
            c2 = (b[0] < 0) as i32 + (b[0] > right) as i32 * 2;
        }
        if c1 & c2 == 0 && c1 | c2 != 0 {
            if c1 != 0 {
                let x = if c1 == 1 { 0 } else { right };
                a[1] += ((x - a[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                a[0] = x;
                c1 = 0;
            }
            if c2 != 0 {
                let x = if c2 == 1 { 0 } else { right };
                b[1] += ((x - b[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                b[0] = x;
                c2 = 0;
            }
        }
    }
    (c1 | c2 == 0, a, b)
}

/// 单通道八连通边界保持左到右的中点裁决。
fn line8(data: &mut [u8], width: usize, height: usize, a: [i64; 2], b: [i64; 2], value: u8) {
    let (inside, mut a, mut b) = clip(width as i64, height as i64, a, b);
    if !inside {
        return;
    }
    if a[0] > b[0] {
        std::mem::swap(&mut a, &mut b);
    }
    let dx = b[0] - a[0];
    let dy = (b[1] - a[1]).abs();
    let sign = if b[1] >= a[1] { 1 } else { -1 };
    let vertical = dy > dx;
    let (major, minor) = if vertical { (dy, dx) } else { (dx, dy) };
    let mut error = major - 2 * minor;
    for _ in 0..=major {
        data[a[1] as usize * width + a[0] as usize] = value;
        let advance = error < 0;
        error += -2 * minor + if advance { 2 * major } else { 0 };
        a[0] += if vertical { advance as i64 } else { 1 };
        a[1] += if vertical {
            sign
        } else {
            sign * advance as i64
        };
    }
}

/// 定点抗锯齿滤波对每个通道做两次整数覆盖混合。
#[allow(clippy::too_many_arguments)]
fn line_aa(
    data: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    a: [i64; 2],
    b: [i64; 2],
    value: u8,
) {
    let (inside, mut a, mut b) = clip((width as i64) << 16, (height as i64) << 16, a, b);
    if !inside {
        return;
    }
    let horizontal = (b[0] - a[0]).abs() > (b[1] - a[1]).abs();
    let major = if horizontal { 0 } else { 1 };
    let minor = 1 - major;
    if a[major] > b[major] {
        std::mem::swap(&mut a, &mut b);
    }
    let step = ((b[minor] - a[minor]) << 16) / ((b[major] - a[major]) | 1);
    let last = b[major] + ONE;
    let count = (last >> 16) - (a[major] >> 16);
    let mut coordinate = a[minor] + ((step * -(a[major] & 65535)) >> 16) + 32768;
    let mut slope = (step >> 11) & 63;
    if step < 0 {
        slope ^= 63;
    }
    slope = if slope & 32 != 0 {
        256
    } else {
        SLOPE[slope as usize]
    };
    let i = (a[major] >> 9) & 120;
    let j = (last >> 9) & 120;
    let t0 = slope << 7;
    let t1 = ((120 - i) | 4) * slope;
    let t2 = (j | 4) * slope;
    let ep = [
        0,
        (((((j - i) & 120) | 4) * slope) >> 8) & 511,
        (t1 >> 8) & 511,
        (((((j - i) & 120) | 4) * slope) >> 8) & 511,
        (((((j - i) + 128) | 4) * slope) >> 8) & 511,
        ((t1 + t0) >> 8) & 511,
        (t2 >> 8) & 511,
        ((t2 + t0) >> 8) & 511,
        slope,
    ];
    for scount in 0..=count {
        let axis = (a[major] >> 16) + scount;
        let minor_axis = (coordinate >> 16) - 1;
        let remaining = count - scount;
        let correction = ep[((((scount >= 2) as i64 + 1) & (scount | 2)) * 3
            + (((remaining >= 2) as i64 + 1) & (remaining | 2)))
            as usize];
        let dist = ((coordinate >> 11) & 31) as usize;
        for (offset, weight) in [FILTER[dist + 32], FILTER[dist], FILTER[63 - dist]]
            .iter()
            .enumerate()
        {
            let (x, y) = if horizontal {
                (axis, minor_axis + offset as i64)
            } else {
                (minor_axis + offset as i64, axis)
            };
            if x >= 0 && y >= 0 && x < width as i64 && y < height as i64 {
                let alpha = ((correction * weight) >> 8) & 255;
                for channel in 0..channels {
                    let target = &mut data[(y as usize * width + x as usize) * channels + channel];
                    let mut v = *target as i64;
                    v += ((value as i64 - v) * alpha + 127) >> 8;
                    v += ((value as i64 - v) * alpha + 127) >> 8;
                    *target = v as u8;
                }
            }
        }
        coordinate += step;
    }
}

/// 凸多边形分别推进两条边，实体填充位于抗锯齿轮廓之内。
fn convex_aa(
    data: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    points: &[[i64; 2]],
    value: u8,
) {
    let n = points.len();
    if n == 0 {
        return;
    }
    for i in 0..n {
        line_aa(
            data,
            width,
            height,
            channels,
            points[(i + n - 1) % n],
            points[i],
            value,
        );
    }
    let mut low = 0;
    for i in 1..n {
        if points[i][1] < points[low][1] {
            low = i;
        }
    }
    let ymin = (points[low][1] + 32768) >> 16;
    let xmax = (points.iter().map(|v| v[0]).max().unwrap() + 32768) >> 16;
    let xmin = (points.iter().map(|v| v[0]).min().unwrap() + 32768) >> 16;
    let raw_ymax = (points.iter().map(|v| v[1]).max().unwrap() + 32768) >> 16;
    if xmax < 0 || raw_ymax < 0 || xmin >= width as i64 || ymin >= height as i64 {
        return;
    }
    let ymax = ((points.iter().map(|v| v[1]).max().unwrap() + 32768) >> 16).min(height as i64 - 1);
    let mut edge = [
        [low as i64, 1, -ONE, 0, ymin],
        [low as i64, n as i64 - 1, -ONE, 0, ymin],
    ];
    let mut remaining = n as i64;
    for y in ymin..=ymax {
        if y < ymax || y == ymin {
            for item in &mut edge {
                if y >= item[4] {
                    let mut index0 = item[0] as usize;
                    let direction = item[1] as usize;
                    let mut index = (index0 + direction) % n;
                    loop {
                        let old = remaining;
                        remaining -= 1;
                        if old <= 0 {
                            break;
                        }
                        let ty = (points[index][1] + 32768) >> 16;
                        if ty > y {
                            let xs = points[index0][0];
                            let xe = points[index][0];
                            item[4] = ty;
                            item[3] = ((xe - xs) * 2 + ty - y) / (2 * (ty - y));
                            item[2] = xs;
                            item[0] = index as i64;
                            break;
                        }
                        index0 = index;
                        index = (index + direction) % n;
                    }
                }
            }
        }
        if remaining < 0 {
            break;
        }
        if y >= 0 && y < height as i64 {
            let left = edge[0][2].min(edge[1][2]);
            let right = edge[0][2].max(edge[1][2]);
            let x1 = ((left + 65535) >> 16).max(0);
            let x2 = (right >> 16).min(width as i64 - 1);
            if x1 <= x2 {
                data[(y as usize * width + x1 as usize) * channels
                    ..(y as usize * width + x2 as usize + 1) * channels]
                    .fill(value);
            }
        }
        for item in &mut edge {
            item[2] += item[3];
        }
    }
}

/// 批量绘制推理线只复制一次图像，端帽使用确定性圆形多边形。
pub fn lines(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    lines: &[[i32; 4]],
    value: u8,
    thickness: usize,
) -> Result<Vec<u8>, String> {
    if width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(channels))
        != Some(data.len())
        || thickness == 0
        || thickness > 1024
    {
        return Err("invalid line image or thickness".into());
    }
    let mut output = data.to_vec();
    for line in lines {
        let mut start = [line[0] as i64, line[1] as i64];
        let mut end = [line[2] as i64, line[3] as i64];
        if thickness > 1
            && [start, end]
                .iter()
                .any(|p| p[0] < 0 || p[1] < 0 || p[0] >= width as i64 || p[1] >= height as i64)
        {
            let margin = thickness as i64;
            let (_, a, b) = clip(
                width as i64 + margin * 2,
                height as i64 + margin * 2,
                [start[0] + margin, start[1] + margin],
                [end[0] + margin, end[1] + margin],
            );
            start = [a[0] - margin, a[1] - margin];
            end = [b[0] - margin, b[1] - margin];
        }
        let a = [start[0] << 16, start[1] << 16];
        let b = [end[0] << 16, end[1] << 16];
        if thickness == 1 {
            line_aa(&mut output, width, height, channels, a, b, value);
            continue;
        }
        let dx = (start[0] - end[0]) as f64;
        let dy = (end[1] - start[1]) as f64;
        let length = dx.hypot(dy);
        let radius = (thickness as i64) << 15;
        if length > 0. {
            let factor = (radius + ((thickness & 1) as i64) * 32768) as f64 / length;
            let px = (dy * factor).round_ties_even() as i64;
            let py = (dx * factor).round_ties_even() as i64;
            let points = [
                [a[0] + px, a[1] + py],
                [a[0] - px, a[1] - py],
                [b[0] - px, b[1] - py],
                [b[0] + px, b[1] + py],
            ];
            convex_aa(&mut output, width, height, channels, &points, value);
        }
        let rounded = (radius + 32768) >> 16;
        let delta = if rounded < 3 {
            90
        } else if rounded < 10 {
            30
        } else if rounded < 15 {
            18
        } else {
            5
        };
        for center in [a, b] {
            let mut points = Vec::new();
            for angle in (0..=360).step_by(delta) {
                let radians = (angle as f64).to_radians();
                let x = (center[0] as f64 + radius as f64 * radians.cos() as f32 as f64)
                    .round_ties_even() as i64;
                let y = (center[1] as f64 + radius as f64 * radians.sin() as f32 as f64)
                    .round_ties_even() as i64;
                if points.last() != Some(&[x, y]) {
                    points.push([x, y]);
                }
            }
            convex_aa(&mut output, width, height, channels, &points, value);
        }
    }
    Ok(output)
}

/// 任意整数多边形按偶奇扫描线填充，另绘制八连通边界。
pub fn polygons(
    width: usize,
    height: usize,
    polygons: &[Vec<[i32; 2]>],
    value: u8,
) -> Result<Vec<u8>, String> {
    let mut output = vec![
        0;
        width
            .checked_mul(height)
            .ok_or("polygon image size overflow")?
    ];
    let mut edges = Vec::new();
    for polygon in polygons {
        for i in 0..polygon.len() {
            let p0 = polygon[(i + polygon.len() - 1) % polygon.len()];
            let p1 = polygon[i];
            let a = [p0[0] as i64, p0[1] as i64];
            let b = [p1[0] as i64, p1[1] as i64];
            line8(&mut output, width, height, a, b, value);
            if a[1] != b[1] {
                let (_, c0, c1) = clip(width as i64, height as i64, a, b);
                let (cy0, cy1) = if c0[1] != c1[1] {
                    (c0[1], c1[1])
                } else {
                    (a[1], b[1])
                };
                let dx = ((c1[0] - c0[0]) << 16) / (cy1 - cy0);
                let mut x = (c0[0] << 16) + (a[1] - cy0) * dx;
                if a[1] > b[1] {
                    x += (b[1] - a[1]) * dx;
                }
                edges.push((a[1].min(b[1]), a[1].max(b[1]), x, dx));
            }
        }
    }
    let mut intersections = Vec::new();
    for y in 0..height as i64 {
        intersections.clear();
        for &(y0, y1, x, dx) in &edges {
            if y >= y0 && y < y1 {
                intersections.push(x + (y - y0) * dx);
            }
        }
        intersections.sort_unstable();
        for pair in intersections.as_chunks::<2>().0 {
            let left = ((pair[0] + 65535) >> 16).max(0);
            let right = (pair[1] >> 16).min(width as i64 - 1);
            if left <= right {
                output[y as usize * width + left as usize..y as usize * width + right as usize + 1]
                    .fill(value);
            }
        }
    }
    Ok(output)
}
