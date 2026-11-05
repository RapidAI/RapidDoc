//! 对独立图像字节执行数值操作，不访问 Python、PDFium 或 OpenCV。
use std::collections::HashMap;

/// 按声明位深从自有连续字节中读取像素。
fn read(data: &[u8], index: usize, depth: u8) -> f32 {
    match depth {
        1 => data[index] as f32,
        2 => u16::from_ne_bytes([data[index * 2], data[index * 2 + 1]]) as f32,
        _ => f32::from_ne_bytes(data[index * 4..index * 4 + 4].try_into().unwrap()),
    }
}

/// 目标位深使用饱和偶数舍入，浮点保持原始 IEEE 字节。
fn write(output: &mut [u8], index: usize, value: f32, depth: u8) {
    match depth {
        1 => output[index] = value.round_ties_even().clamp(0., 255.) as u8,
        2 => output[index * 2..index * 2 + 2]
            .copy_from_slice(&(value.round_ties_even().clamp(0., 65535.) as u16).to_ne_bytes()),
        _ => output[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes()),
    }
}

/// 分离卷积仅缓存当前需要的源行，避免创建全幅中间张量。
#[allow(clippy::too_many_arguments)]
pub fn resize(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    depth: u8,
    target_w: usize,
    target_h: usize,
    mode: u8,
    xindices: &[Vec<usize>],
    xweights: &[Vec<f32>],
    yindices: &[Vec<usize>],
    yweights: &[Vec<f32>],
) -> Result<Vec<u8>, String> {
    if data.len()
        != width
            .checked_mul(height)
            .and_then(|v| v.checked_mul(channels))
            .and_then(|v| v.checked_mul(depth as usize))
            .ok_or("image size overflow")?
        || xindices.len()
            != if mode == 4 {
                target_w * channels
            } else {
                target_w
            }
        || yindices.len() != target_h
        || xweights.len() != xindices.len()
        || yweights.len() != target_h
        || xindices.iter().flatten().any(|&x| x >= width)
        || yindices.iter().flatten().any(|&y| y >= height)
        || xindices
            .iter()
            .zip(xweights)
            .any(|(i, w)| i.len() != w.len())
        || yindices
            .iter()
            .zip(yweights)
            .any(|(i, w)| i.len() != w.len())
    {
        return Err("invalid resize buffer or coefficient table".into());
    }
    let output_size = target_w
        .checked_mul(target_h)
        .and_then(|v| v.checked_mul(channels))
        .and_then(|v| v.checked_mul(depth as usize))
        .ok_or("output size overflow")?;
    let mut output = vec![0; output_size];
    if depth == 1 && [4, 5].contains(&mode) {
        let bits = if mode == 4 { 8 } else { 7 };
        let rounding = 1i32 << (bits - 1);
        for y in 0..target_h {
            for x in 0..target_w {
                for c in 0..channels {
                    let coefficient = if mode == 4 { x * channels + c } else { x };
                    let vertical = |column: usize| -> i32 {
                        let a = data[(yindices[y][0] * width + xindices[coefficient][column])
                            * channels
                            + c] as i32;
                        let b = data[(yindices[y][1] * width + xindices[coefficient][column])
                            * channels
                            + c] as i32;
                        (a * yweights[y][0] as i32 + b * yweights[y][1] as i32 + rounding) >> bits
                    };
                    output[(y * target_w + x) * channels + c] =
                        ((vertical(0) * xweights[coefficient][0] as i32
                            + vertical(1) * xweights[coefficient][1] as i32
                            + rounding)
                            >> bits) as u8;
                }
            }
        }
        return Ok(output);
    }
    let mut rows: HashMap<usize, Vec<f32>> = HashMap::new();
    for y in 0..target_h {
        rows.retain(|key, _| yindices[y].contains(key));
        for &source_y in &yindices[y] {
            rows.entry(source_y).or_insert_with(|| {
                let mut row = vec![0f32; target_w * channels];
                for x in 0..target_w {
                    for channel in 0..channels {
                        let mut sum = 0f32;
                        for (&index, &weight) in xindices[x].iter().zip(&xweights[x]) {
                            let pixel =
                                read(data, (source_y * width + index) * channels + channel, depth);
                            sum += pixel * weight;
                        }
                        row[x * channels + channel] = sum;
                    }
                }
                row
            });
        }
        // 当前行只解析一次缓存索引，避免每个像素重复进行哈希查找。
        let vertical_rows: Vec<&[f32]> = yindices[y]
            .iter()
            .map(|index| rows[index].as_slice())
            .collect();
        for x in 0..target_w * channels {
            if depth == 1 && mode == 1 {
                let mut sum = 0i32;
                for (tap, &weight) in yweights[y].iter().enumerate() {
                    sum += (((vertical_rows[tap][x] as i64 >> 4) * weight as i64) >> 16) as i32;
                }
                output[(y * target_w * channels) + x] = ((sum + 2) >> 2).clamp(0, 255) as u8;
            } else if depth == 1 && mode == 6 && x < target_w * channels / 8 * 8 {
                let mut sum = vertical_rows[3][x] * (yweights[y][3] / (1u32 << 22) as f32);
                for index in (0..3).rev() {
                    sum = vertical_rows[index][x]
                        .mul_add(yweights[y][index] / (1u32 << 22) as f32, sum);
                }
                write(&mut output, y * target_w * channels + x, sum, depth);
            } else if depth == 1 && [2, 6].contains(&mode) {
                let mut sum = 0i64;
                for (tap, &weight) in yweights[y].iter().enumerate() {
                    sum += vertical_rows[tap][x] as i64 * weight as i64;
                }
                output[y * target_w * channels + x] = ((sum + (1 << 21)) >> 22).clamp(0, 255) as u8;
            } else {
                let mut sum = 0f32;
                for (tap, &weight) in yweights[y].iter().enumerate() {
                    sum += vertical_rows[tap][x] * weight;
                }
                if mode == 7 {
                    sum = (sum + 0.5).floor();
                }
                write(&mut output, y * target_w * channels + x, sum, depth);
            }
        }
    }
    Ok(output)
}

/// 压缩标签并查集路径，不保留图像对象。
fn find(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}

/// 两遍扫描标记八连通域，按首次遇到的分区生成确定标签。
pub fn components(
    data: &[u8],
    width: usize,
    height: usize,
) -> Result<(Vec<u8>, Vec<[i32; 5]>), String> {
    if width.checked_mul(height) != Some(data.len())
        || width > i32::MAX as usize
        || height > i32::MAX as usize
    {
        return Err("invalid component mask".into());
    }
    let mut labels = vec![0usize; data.len()];
    let mut parents = vec![0usize];
    for y in 0..height {
        for x in 0..width {
            if data[y * width + x] == 0 {
                continue;
            }
            // 四邻接标签使用栈数组，密集二值图不产生逐像素堆分配。
            let mut roots = [0usize; 4];
            let neighbors = [
                if x > 0 { labels[y * width + x - 1] } else { 0 },
                if y > 0 && x > 0 {
                    labels[(y - 1) * width + x - 1]
                } else {
                    0
                },
                if y > 0 {
                    labels[(y - 1) * width + x]
                } else {
                    0
                },
                if y > 0 && x + 1 < width {
                    labels[(y - 1) * width + x + 1]
                } else {
                    0
                },
            ];
            let mut minimum = usize::MAX;
            for (i, neighbor) in neighbors.into_iter().enumerate() {
                if neighbor != 0 {
                    roots[i] = find(&mut parents, neighbor);
                    minimum = minimum.min(roots[i]);
                }
            }
            let label = if minimum != usize::MAX {
                for root in roots {
                    if root != 0 {
                        parents[root] = minimum;
                    }
                }
                minimum
            } else {
                let index = parents.len();
                parents.push(index);
                index
            };
            labels[y * width + x] = label;
        }
    }
    let roots: Vec<usize> = (0..parents.len()).map(|v| find(&mut parents, v)).collect();
    let mut mapping = vec![0usize; parents.len()];
    let mut count = 1;
    // 二乘二块按行优先发现分区，保持历史八连通组件顺序。
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    if y + dy < height && x + dx < width {
                        let root = roots[labels[(y + dy) * width + x + dx]];
                        if root != 0 && mapping[root] == 0 {
                            mapping[root] = count;
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    let mut stats = vec![[i32::MAX, i32::MAX, -1, -1, 0]; count];
    let mut output = Vec::with_capacity(data.len() * 4);
    for (index, label) in labels.into_iter().enumerate() {
        let id = mapping[roots[label]];
        output.extend_from_slice(&(id as i32).to_ne_bytes());
        let x = (index % width) as i32;
        let y = (index / width) as i32;
        let stat = &mut stats[id];
        stat[0] = stat[0].min(x);
        stat[1] = stat[1].min(y);
        stat[2] = stat[2].max(x);
        stat[3] = stat[3].max(y);
        stat[4] += 1;
    }
    for stat in &mut stats {
        if stat[4] == 0 {
            *stat = [-1, -1, 0, 0, 0];
        } else {
            stat[2] -= stat[0] - 1;
            stat[3] -= stat[1] - 1;
        }
    }
    Ok((output, stats))
}

/// 八邻域边界追踪与直线简化；遍历语义对应公开参考实现。
pub fn contours(
    data: &[u8],
    width: usize,
    height: usize,
    external: bool,
) -> Result<Vec<Vec<[i32; 2]>>, String> {
    if width.checked_mul(height) != Some(data.len()) {
        return Err("invalid contour mask".into());
    }
    let stride = width + 2;
    let mut pixels = vec![0i16; stride * (height + 2)];
    for y in 0..height {
        for x in 0..width {
            pixels[(y + 1) * stride + x + 1] = i16::from(data[y * width + x] != 0);
        }
    }
    let deltas: [isize; 8] = [
        1,
        1 - stride as isize,
        -(stride as isize),
        -1 - stride as isize,
        -1,
        stride as isize - 1,
        stride as isize,
        stride as isize + 1,
    ];
    let mut result = Vec::new();
    for y in 1..=height {
        let mut previous = 0i16;
        let mut last_boundary = y * stride;
        for x in 1..=width {
            let index = y * stride + x;
            let current = pixels[index];
            if current == previous {
                continue;
            }
            let outer = previous == 0 && current == 1;
            let hole = current == 0 && previous >= 1;
            if !outer && !hole {
                previous = current;
                if previous & -2 != 0 {
                    last_boundary = index;
                }
                continue;
            }
            if hole && previous & -2 != 0 {
                last_boundary = index - 1;
            }
            if external && (hole || pixels[last_boundary] > 0) {
                previous = current;
                continue;
            }
            let start = index - usize::from(hole);
            let mut direction = if hole { 0 } else { 4 };
            let end = direction;
            let mut first;
            loop {
                direction = (direction + 7) & 7;
                first = (start as isize + deltas[direction]) as usize;
                if pixels[first] != 0 || direction == end {
                    break;
                }
            }
            let mut points = Vec::new();
            if direction == end {
                pixels[start] = -126;
                points.push([(start % stride) as i32 - 1, (start / stride) as i32 - 1]);
            } else {
                let mut position = start;
                let mut previous_direction = direction ^ 4;
                let mut terminated = false;
                for _ in 0..pixels.len() * 8 {
                    let end = direction;
                    let mut next_direction = direction + 1;
                    let mut following;
                    loop {
                        following = (position as isize + deltas[next_direction & 7]) as usize;
                        if pixels[following] != 0 {
                            break;
                        }
                        next_direction += 1;
                    }
                    direction = next_direction & 7;
                    if (direction as u32).wrapping_sub(1) < end as u32 {
                        pixels[position] = -126;
                    } else if pixels[position] == 1 {
                        pixels[position] = 2;
                    }
                    if direction != previous_direction {
                        points.push([
                            (position % stride) as i32 - 1,
                            (position / stride) as i32 - 1,
                        ]);
                        previous_direction = direction;
                    }
                    if following == start && position == first {
                        terminated = true;
                        break;
                    }
                    position = following;
                    direction = (direction + 4) & 7;
                }
                if !terminated {
                    return Err("contour traversal did not terminate".into());
                }
            }
            result.push(points);
            last_boundary = start;
            previous = pixels[index];
        }
    }
    result.reverse();
    Ok(result)
}

/// 以 IEEE 偶数舍入量化到半精度，然后返回可直接参与运算的 float32。
fn half_round(value: f64) -> f32 {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 2047) as i32 - 1023;
    if exponent < -14 {
        return ((value * 16777216.).round_ties_even() / 16777216.) as f32;
    }
    if exponent >= 16 {
        return if value.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    let mask = (1u64 << 42) - 1;
    let remainder = bits & mask;
    let mut rounded = bits & !mask;
    if remainder > 1u64 << 41 || (remainder == 1u64 << 41 && (rounded >> 42) & 1 != 0) {
        rounded += 1u64 << 42;
    }
    f64::from_bits(rounded) as f32
}

/// 加乘中间使用 double 精度，结果只量化一次至约定的半精度。
fn half_fma(a: f32, b: f32, c: f32) -> f32 {
    half_round(a as f64 * b as f64 + c as f64)
}

/// 连续三次采样使用同一权重式，八位图明确保留半精度数值契约。
fn cubic_weights(x: f32, half: bool) -> [f32; 4] {
    if half {
        let x = half_round(x as f64);
        let a2 = half_round((x * x) as f64);
        let b = half_round((1. - x) as f64);
        let b2 = half_round((b * b) as f64);
        let w0 = half_round((-0.75 * half_round((x * b2) as f64)) as f64);
        let w3 = half_round((-0.75 * half_round((a2 * b) as f64)) as f64);
        let w1 = half_fma(a2, half_fma(1.25, x, -2.25), 1.);
        let w2 = half_round((half_round((half_round((1. - w0) as f64) - w1) as f64) - w3) as f64);
        [w0, w1, w2, w3]
    } else {
        let a2 = x * x;
        let b = 1. - x;
        let w0 = -0.75 * (x * (b * b));
        let w3 = -0.75 * (a2 * b);
        let w1 = a2.mul_add(1.25f32.mul_add(x, -2.25), 1.);
        [w0, w1, 1. - w0 - w1 - w3, w3]
    }
}

/// 采样边界时使用显式常数或复制模式，不调用其他视觉运行时。
#[allow(clippy::too_many_arguments)]
fn sample(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    depth: u8,
    x: i64,
    y: i64,
    c: usize,
    replicate: bool,
    value: f32,
) -> f32 {
    if !replicate && (x < 0 || y < 0 || x >= width as i64 || y >= height as i64) {
        return value;
    }
    let x = x.clamp(0, width as i64 - 1) as usize;
    let y = y.clamp(0, height as i64 - 1) as usize;
    read(data, (y * width + x) * channels + c, depth)
}

/// 分行逆映射避免整幅坐标临时数组，释放 GIL 后可独立并发执行。
#[allow(clippy::too_many_arguments)]
pub fn warp(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    depth: u8,
    target_w: usize,
    target_h: usize,
    mode: u8,
    replicate: bool,
    value: f32,
    inverse: &[f64; 9],
    affine: bool,
) -> Result<Vec<u8>, String> {
    if width == 0
        || height == 0
        || data.len()
            != width
                .checked_mul(height)
                .and_then(|v| v.checked_mul(channels))
                .and_then(|v| v.checked_mul(depth as usize))
                .ok_or("image size overflow")?
    {
        return Err("invalid warp image".into());
    }
    let length = target_w
        .checked_mul(target_h)
        .and_then(|v| v.checked_mul(channels))
        .and_then(|v| v.checked_mul(depth as usize))
        .ok_or("output size overflow")?;
    let mut output = vec![0u8; length];
    let half = depth == 1 && channels != 2;
    let m: Vec<f32> = inverse.iter().map(|&v| v as f32).collect();
    for y in 0..target_h {
        let row: Vec<f32> = (0..3)
            .map(|i| {
                if depth == 4 {
                    (y as f32).mul_add(m[i * 3 + 1], m[i * 3 + 2])
                } else {
                    (y as f64 * inverse[i * 3 + 1] + inverse[i * 3 + 2]) as f32
                }
            })
            .collect();
        let cubic_row: Vec<f32> = (0..3)
            .map(|i| (y as f32).mul_add(m[i * 3 + 1], m[i * 3 + 2]))
            .collect();
        for x in 0..target_w {
            let (sx, sy) = if mode == 2 {
                if affine {
                    (
                        (x as f32).mul_add(m[0], cubic_row[0]),
                        (x as f32).mul_add(m[3], cubic_row[1]),
                    )
                } else {
                    let d = cubic_row[2] as f64 + x as f64 * m[6] as f64;
                    if d == 0. {
                        (0., 0.)
                    } else {
                        (
                            ((cubic_row[0] as f64 + x as f64 * m[0] as f64) / d) as f32,
                            ((cubic_row[1] as f64 + x as f64 * m[3] as f64) / d) as f32,
                        )
                    }
                }
            } else if x < target_w / 8 * 8 {
                let d = (x as f32).mul_add(m[6], row[2]);
                if d == 0. {
                    (0., 0.)
                } else {
                    (
                        (x as f32).mul_add(m[0], row[0]) / d,
                        (x as f32).mul_add(m[3], row[1]) / d,
                    )
                }
            } else {
                let d = (x as f64 * inverse[6] + y as f64 * inverse[7] + inverse[8]) as f32;
                if d == 0. {
                    (0., 0.)
                } else {
                    (
                        ((x as f64 * inverse[0] + y as f64 * inverse[1] + inverse[2]) / d as f64)
                            as f32,
                        ((x as f64 * inverse[3] + y as f64 * inverse[4] + inverse[5]) / d as f64)
                            as f32,
                    )
                }
            };
            let bx = sx.floor() as i64;
            let by = sy.floor() as i64;
            let alpha = sx - bx as f32;
            let beta = sy - by as f32;
            let wx = cubic_weights(alpha, half);
            let wy = cubic_weights(beta, half);
            for c in 0..channels {
                let pixel = |ix: i64, iy: i64| {
                    sample(
                        data, width, height, channels, depth, ix, iy, c, replicate, value,
                    )
                };
                let result = if mode == 0 {
                    pixel(sx.round_ties_even() as i64, sy.round_ties_even() as i64)
                } else if mode == 1 {
                    let p00 = pixel(bx, by);
                    let p01 = pixel(bx + 1, by);
                    let p10 = pixel(bx, by + 1);
                    let p11 = pixel(bx + 1, by + 1);
                    let top = alpha.mul_add(p01 - p00, p00);
                    let bottom = alpha.mul_add(p11 - p10, p10);
                    beta.mul_add(bottom - top, top)
                } else {
                    let mut sum = 0f32;
                    for (j, &yweight) in wy.iter().enumerate() {
                        let mut horizontal = pixel(bx - 1, by + j as i64 - 1) * wx[0];
                        if half {
                            horizontal = half_round(horizontal as f64);
                        }
                        for (i, &xweight) in wx.iter().enumerate().skip(1) {
                            let p = pixel(bx + i as i64 - 1, by + j as i64 - 1);
                            horizontal = if half {
                                half_fma(p, xweight, horizontal)
                            } else {
                                p.mul_add(xweight, horizontal)
                            };
                        }
                        sum = if half {
                            half_fma(horizontal, yweight, sum)
                        } else {
                            horizontal.mul_add(yweight, sum)
                        };
                    }
                    sum
                };
                write(
                    &mut output,
                    (y * target_w + x) * channels + c,
                    result,
                    depth,
                );
            }
        }
    }
    Ok(output)
}

/// 旋转卡尺接收有序凸包，等面积时使用最后一次接触边的裁决。
pub fn minimum_rectangle(hull: &[[f32; 2]]) -> [f32; 5] {
    let n = hull.len();
    if n == 0 {
        return [0., 0., 0., 0., -90.];
    }
    if n == 1 {
        return [hull[0][0], hull[0][1], 0., 0., -90.];
    }
    if n == 2 {
        let dx = (hull[0][0] - hull[1][0]) as f64;
        let dy = (hull[0][1] - hull[1][1]) as f64;
        let mut w = 0.;
        let mut h = dx.hypot(dy) as f32;
        let mut angle = -90.;
        if dx == 0. {
            std::mem::swap(&mut w, &mut h);
        } else if dy < 0. {
            angle = dy.atan2(dx).to_degrees() as f32;
            std::mem::swap(&mut w, &mut h);
        } else if dy > 0. {
            angle = -dx.atan2(dy).to_degrees() as f32;
        }
        return [
            (hull[0][0] + hull[1][0]) * 0.5,
            (hull[0][1] + hull[1][1]) * 0.5,
            w,
            h,
            angle,
        ];
    }
    let vectors: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            [
                hull[(i + 1) % n][0] - hull[i][0],
                hull[(i + 1) % n][1] - hull[i][1],
            ]
        })
        .collect();
    let lengths: Vec<f32> = vectors
        .iter()
        .map(|v| (1. / (v[0] as f64).hypot(v[1] as f64)) as f32)
        .collect();
    let extremum = |axis: usize, maximum: bool| {
        let mut idx = 0;
        for i in 1..n {
            if if maximum {
                hull[i][axis] > hull[idx][axis]
            } else {
                hull[i][axis] < hull[idx][axis]
            } {
                idx = i;
            }
        }
        idx
    };
    let mut seq = [
        extremum(1, false),
        extremum(0, true),
        extremum(1, true),
        extremum(0, false),
    ];
    let mut area = f32::INFINITY;
    let (mut left, mut a, mut b, mut width, mut height, mut bottom) =
        (0, 0f32, 0f32, 0f32, 0f32, 0);
    for _ in 0..n {
        let v0 = vectors[seq[0]];
        let v1 = vectors[seq[1]];
        let v2 = vectors[seq[2]];
        let v3 = vectors[seq[3]];
        let rot = [v0, [v1[1], -v1[0]], [-v2[0], -v2[1]], [-v3[1], v3[0]]];
        let mut main = 0;
        for i in 1..4 {
            if rot[i][1].mul_add(rot[main][0], -rot[i][0] * rot[main][1]) < 0. {
                main = i;
            }
        }
        let lead_x = vectors[seq[main]][0] * lengths[seq[main]];
        let lead_y = vectors[seq[main]][1] * lengths[seq[main]];
        let (ba, bb) = [
            (lead_x, lead_y),
            (lead_y, -lead_x),
            (-lead_x, -lead_y),
            (-lead_y, lead_x),
        ][main];
        seq[main] = (seq[main] + 1) % n;
        let dx = hull[seq[1]][0] - hull[seq[3]][0];
        let dy = hull[seq[1]][1] - hull[seq[3]][1];
        let bw = dx.mul_add(ba, dy * bb);
        let dx = hull[seq[2]][0] - hull[seq[0]][0];
        let dy = hull[seq[2]][1] - hull[seq[0]][1];
        let bh = (-dx).mul_add(bb, dy * ba);
        if bw * bh <= area {
            area = bw * bh;
            left = seq[3];
            a = ba;
            b = bb;
            width = bw;
            height = bh;
            bottom = seq[0];
        }
    }
    let c1 = a.mul_add(hull[left][0], hull[left][1] * b);
    let c2 = (-b).mul_add(hull[bottom][0], hull[bottom][1] * a);
    let inverse = 1. / a.mul_add(a, b * b);
    let px = c1.mul_add(a, -c2 * b) * inverse;
    let py = a.mul_add(c2, b * c1) * inverse;
    let edge1 = [a * width, b * width];
    let edge2 = [-b * height, a * height];
    let mut w = (edge2[0] as f64).hypot(edge2[1] as f64) as f32;
    let mut h = (edge1[0] as f64).hypot(edge1[1] as f64) as f32;
    let angle = if edge1[0] == 0. && edge1[1] > 0. {
        std::mem::swap(&mut w, &mut h);
        -90.
    } else {
        -(edge1[0] as f64).atan2(edge1[1] as f64).to_degrees() as f32
    };
    [
        px + (edge1[0] + edge2[0]) * 0.5,
        py + (edge1[1] + edge2[1]) * 0.5,
        w,
        h,
        angle,
    ]
}

/// 矩形极值使用单调队列逐轴处理，复杂度不随核面积增长。
pub fn morphology(
    data: &[u8],
    width: usize,
    height: usize,
    kw: usize,
    kh: usize,
    dilate: bool,
) -> Result<Vec<u8>, String> {
    if width.checked_mul(height) != Some(data.len()) || kw == 0 || kh == 0 {
        return Err("invalid morphology buffer".into());
    }
    let border = if dilate { 0 } else { 255 };
    let pass = |input: &[u8], vertical: bool, kernel: usize| {
        let length = if vertical { height } else { width };
        let lines = if vertical { width } else { height };
        let mut output = vec![0u8; data.len()];
        for line in 0..lines {
            let mut queue = std::collections::VecDeque::<(usize, u8)>::new();
            for p in 0..length + kernel - 1 {
                let coordinate = p as isize - (kernel / 2) as isize;
                let value = if coordinate < 0 || coordinate >= length as isize {
                    border
                } else {
                    input[if vertical {
                        coordinate as usize * width + line
                    } else {
                        line * width + coordinate as usize
                    }]
                };
                while queue
                    .back()
                    .is_some_and(|v| if dilate { v.1 <= value } else { v.1 >= value })
                {
                    queue.pop_back();
                }
                queue.push_back((p, value));
                while queue.front().is_some_and(|v| v.0 + kernel <= p) {
                    queue.pop_front();
                }
                if p + 1 >= kernel {
                    let position = p + 1 - kernel;
                    output[if vertical {
                        position * width + line
                    } else {
                        line * width + position
                    }] = queue.front().unwrap().1;
                }
            }
        }
        output
    };
    Ok(pass(&pass(data, false, kw), true, kh))
}

/// 通用凸包保留首个原始索引，并按循环单调输入顺序确定起点。
pub fn convex_hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut values: Vec<([f32; 2], usize)> = points
        .iter()
        .copied()
        .enumerate()
        .map(|(i, p)| (p, i))
        .collect();
    values.sort_unstable_by(|a, b| {
        a.0[0]
            .total_cmp(&b.0[0])
            .then_with(|| a.0[1].total_cmp(&b.0[1]))
            .then(a.1.cmp(&b.1))
    });
    values.dedup_by(|a, b| a.0 == b.0);
    if values.len() < 2 {
        return values.into_iter().map(|v| v.0).collect();
    }
    let cross = |a: ([f32; 2], usize), b: ([f32; 2], usize), c: ([f32; 2], usize)| {
        (b.0[0] - a.0[0]) as f64 * (c.0[1] - a.0[1]) as f64
            - (b.0[1] - a.0[1]) as f64 * (c.0[0] - a.0[0]) as f64
    };
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for &point in &values {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0.
        {
            lower.pop();
        }
        lower.push(point);
    }
    for &point in values.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0.
        {
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    let n = lower.len();
    let mut start = 0;
    for i in 1..n {
        if lower[i].0[0] > lower[start].0[0]
            || (lower[i].0[0] == lower[start].0[0] && lower[i].0[1] > lower[start].0[1])
        {
            start = i;
        }
    }
    lower.rotate_left(start);
    if n >= 3 {
        for increasing in [true, false] {
            let start = if increasing {
                lower.iter().enumerate().min_by_key(|(_, v)| v.1).unwrap().0
            } else {
                lower.iter().enumerate().max_by_key(|(_, v)| v.1).unwrap().0
            };
            if (0..n - 1)
                .all(|i| (lower[(start + i) % n].1 < lower[(start + i + 1) % n].1) == increasing)
            {
                lower.rotate_left(start);
                break;
            }
        }
    }
    lower.into_iter().map(|v| v.0).collect()
}

/// 小型矩阵使用显式主元和融合加乘消元，避免 BLAS 选择不同舍入顺序。
pub fn linear_solve(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = matrix.len();
    if n == 0 || n > 16 || rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    for i in 0..n {
        let mut pivot = i;
        for j in i + 1..n {
            if a[j][i].abs() > a[pivot][i].abs() {
                pivot = j;
            }
        }
        if a[pivot][i].abs() < f64::EPSILON * 100. {
            return None;
        }
        if pivot != i {
            // 已消元列不再读取，交换整行不改变有效列的计算顺序。
            a.swap(i, pivot);
            b.swap(i, pivot);
        }
        let reciprocal = -1. / a[i][i];
        let (prior, following) = a.split_at_mut(i + 1);
        let pivot_row = &prior[i];
        for (offset, row) in following.iter_mut().enumerate() {
            let alpha = row[i] * reciprocal;
            for (value, &coefficient) in row[i + 1..].iter_mut().zip(&pivot_row[i + 1..]) {
                *value = alpha.mul_add(coefficient, *value);
            }
            let j = i + 1 + offset;
            b[j] = alpha.mul_add(b[i], b[j]);
        }
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for (k, &value) in b.iter().enumerate().skip(i + 1) {
            sum = (-a[i][k]).mul_add(value, sum);
        }
        b[i] = sum / a[i][i];
    }
    Some(b)
}
