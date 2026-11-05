//! OpenCV 4.13 closed approxPolyDP and the original custom mask vertices.
//! Intel/OpenCV notices are distributed with the runtime sources and wheel.
use image::GrayImage;
use imageproc::contours::{find_contours, BorderType};
use rapiddoc_core::{Error, ErrorCode};
type P = [f64; 2];
fn error(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}
fn distance(a: P, b: P) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}
fn contour_area(p: &[P]) -> f64 {
    (0..p.len())
        .map(|i| {
            let q = p[(i + 1) % p.len()];
            p[i][0] * q[1] - q[0] * p[i][1]
        })
        .sum::<f64>()
        .abs()
        * 0.5
}
fn simple(p: &[P]) -> Vec<P> {
    if p.len() <= 2 {
        return p.to_vec();
    }
    (0..p.len())
        .filter_map(|i| {
            let a = p[(i + p.len() - 1) % p.len()];
            let b = p[i];
            let c = p[(i + 1) % p.len()];
            let before = [
                ((b[0] - a[0]) as i64).signum(),
                ((b[1] - a[1]) as i64).signum(),
            ];
            let after = [
                ((c[0] - b[0]) as i64).signum(),
                ((c[1] - b[1]) as i64).signum(),
            ];
            (before != after).then_some(b)
        })
        .collect()
}
pub fn mask_polygon(
    mask: &[u8],
    width: usize,
    height: usize,
    max_allowed_dist: f64,
) -> Result<Option<(Vec<P>, bool)>, Error> {
    if width.checked_mul(height) != Some(mask.len()) {
        return Err(error("Invalid layout mask shape"));
    }
    let mut image = GrayImage::new((width + 2) as u32, (height + 2) as u32);
    let stride = width + 2;
    for y in 0..height {
        let start = (y + 1) * stride + 1;
        image.as_mut()[start..start + width].copy_from_slice(&mask[y * width..(y + 1) * width]);
    }
    let contours = find_contours::<i32>(&image);
    let mut selected = None;
    let mut best = -1.;
    for c in contours
        .into_iter()
        .rev()
        .filter(|c| c.border_type == BorderType::Outer && c.parent.is_none())
    {
        let points = c
            .points
            .iter()
            .map(|p| [(p.x - 1) as f64, (p.y - 1) as f64])
            .collect::<Vec<_>>();
        let points = simple(&points);
        let area = contour_area(&points);
        if area > best {
            best = area;
            selected = Some(points)
        }
    }
    let Some(points) = selected else {
        return Ok(None);
    };
    let mut perimeter = 0.;
    for i in 0..points.len() {
        let p = points[i];
        let q = points[(i + 1) % points.len()];
        let dx = (p[0] - q[0]) as f32;
        let dy = (p[1] - q[1]) as f32;
        perimeter += ((dx * dx + dy * dy) as f64).sqrt();
    }
    let polygon = approximate(&points, perimeter * 0.004)?;
    #[cfg(test)]
    if std::env::var_os("RAPIDDOC_LAYOUT_TRACE").is_some() {
        eprintln!(
            "contour {:?}; approximate {:?}; custom {:?}",
            points,
            polygon,
            custom_vertices(&polygon, max_allowed_dist)
        );
    }
    Ok(Some(custom_vertices(&polygon, max_allowed_dist)))
}
pub(crate) fn approximate(points: &[P], epsilon: f64) -> Result<Vec<P>, Error> {
    let n = points.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let eps = epsilon * epsilon;
    let mut pos = 0;
    let mut far = 0;
    let mut start = [0., 0.];
    let mut close = false;
    for _ in 0..3 {
        let mut max = 0.;
        pos = (pos + far) % n;
        start = points[pos];
        pos = (pos + 1) % n;
        for j in 1..n {
            let p = points[pos];
            pos = (pos + 1) % n;
            let dx = p[0] - start[0];
            let dy = p[1] - start[1];
            let d = dx * dx + dy * dy;
            if d > max {
                max = d;
                far = j
            }
        }
        close = max <= eps;
    }
    let mut stack = vec![];
    let mut output = vec![];
    if close {
        output.push(start)
    } else {
        let a = pos % n;
        let b = (far + a) % n;
        stack.push((b, a));
        stack.push((a, b));
    }
    while let Some((a, b)) = stack.pop() {
        let end = points[b];
        let start = points[a];
        pos = (a + 1) % n;
        let mut max = 0.;
        let mut split = a;
        if pos != b {
            let dx = end[0] - start[0];
            let dy = end[1] - start[1];
            let length = dx * dx + dy * dy;
            if length == 0. {
                return Err(error("Degenerate closed layout contour"));
            }
            while pos != b {
                let p = points[pos];
                let current = pos;
                pos = (pos + 1) % n;
                let projection = (p[0] - start[0]) * dx + (p[1] - start[1]) * dy;
                let dist = if projection < 0. {
                    ((p[0] - start[0]).powi(2) + (p[1] - start[1]).powi(2)) * length
                } else if projection > length {
                    ((p[0] - end[0]).powi(2) + (p[1] - end[1]).powi(2)) * length
                } else {
                    ((p[1] - start[1]) * dx - (p[0] - start[0]) * dy).powi(2)
                };
                if dist > max {
                    max = dist;
                    split = current
                }
            }
            close = max <= eps * length;
        } else {
            close = true
        }
        if close {
            output.push(start)
        } else {
            stack.push((split, b));
            stack.push((a, split));
        }
    }
    let count = output.len();
    if count <= 2 {
        return Ok(output);
    }
    let mut new_count = count;
    pos = 0;
    let mut start = output[count - 1];
    let mut current = output[pos];
    pos = (pos + 1) % count;
    let mut write = 0;
    let mut i = 0;
    while i < count && new_count > 2 {
        let end = output[pos];
        pos = (pos + 1) % count;
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dist = ((current[0] - start[0]) * dy - (current[1] - start[1]) * dx).abs();
        let inner = (current[0] - start[0]) * (end[0] - current[0])
            + (current[1] - start[1]) * (end[1] - current[1]);
        if dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx != 0. && dy != 0. && inner >= 0. {
            new_count -= 1;
            output[write] = end;
            start = end;
            write = (write + 1) % count;
            current = output[pos];
            pos = (pos + 1) % count;
            i += 2;
            continue;
        }
        output[write] = current;
        start = current;
        write = (write + 1) % count;
        current = end;
        i += 1;
    }
    output.truncate(new_count);
    Ok(output)
}
fn custom_vertices(poly: &[P], max_dist: f64) -> (Vec<P>, bool) {
    let n = poly.len();
    if n == 0 {
        return (vec![], false);
    }
    let mut convex = vec![];
    let mut angles = vec![];
    let mut vectors = vec![];
    for i in 0..n {
        let prev = poly[(i + n - 1) % n];
        let cur = poly[i];
        let next = poly[(i + 1) % n];
        let v1 = [prev[0] - cur[0], prev[1] - cur[1]];
        let v2 = [next[0] - cur[0], next[1] - cur[1]];
        convex.push(
            (cur[0] - prev[0]) * (next[1] - cur[1]) - (cur[1] - prev[1]) * (next[0] - cur[0]) < 0.,
        );
        let a = distance(v1, [0., 0.]);
        let b = distance(v2, [0., 0.]);
        let dot = ((v1[0] / a) * (v2[0] / b) + (v1[1] / a) * (v2[1] / b)).clamp(-1., 1.);
        angles.push(dot.acos() * 180. / std::f64::consts::PI);
        vectors.push((v1, v2, a, b));
    }
    let concave = (0..n).filter(|&i| !convex[i]).collect::<Vec<_>>();
    let mut groups = vec![];
    let mut current = vec![];
    for &index in &concave {
        if current
            .last()
            .is_some_and(|&prev| index - prev != 1 && !(prev == n - 1 && index == 0))
        {
            if current.len() >= 2 {
                groups.extend(&current)
            }
            current.clear();
        }
        current.push(index)
    }
    if current.len() >= 2 {
        groups.extend(current)
    }
    let mut preserve = std::collections::BTreeSet::new();
    if concave.len() >= 2 && concave[0] == 0 && concave[concave.len() - 1] == n - 1 {
        if groups.contains(&0) && groups.contains(&(n - 1)) {
            preserve.extend(groups)
        }
    } else {
        preserve.extend(groups)
    }
    let kept = (0..n)
        .filter(|&i| convex[i] || (preserve.contains(&i) && angles[i] >= 120.))
        .collect::<Vec<_>>();
    let mut final_points = std::collections::BTreeSet::new();
    for j in 0..kept.len() {
        let index = kept[j];
        let next = kept[(j + 1) % kept.len()];
        final_points.insert(index);
        let dist = distance(poly[index], poly[next]);
        if dist > max_dist {
            let mut intermediate = vec![];
            let mut k = (index + 1) % n;
            while k != next {
                intermediate.push(k);
                k = (k + 1) % n;
            }
            if !intermediate.is_empty() {
                let needed = (dist / max_dist).ceil() as usize - 1;
                if intermediate.len() <= needed {
                    final_points.extend(intermediate)
                } else {
                    let step = intermediate.len() as f64 / needed as f64;
                    for k in 0..needed {
                        final_points.insert(intermediate[(k as f64 * step) as usize]);
                    }
                }
            }
        }
    }
    let floating = final_points
        .iter()
        .any(|&i| convex[i] && (angles[i] - 45.).abs() < 1.);
    let points = final_points
        .into_iter()
        .map(|i| {
            if convex[i] && (angles[i] - 45.).abs() < 1. {
                let (v1, v2, a, b) = vectors[i];
                let mut dir = [v1[0] / a + v2[0] / b, v1[1] / a + v2[1] / b];
                let length = distance(dir, [0., 0.]);
                dir[0] /= length;
                dir[1] /= length;
                let d = (a + b) / 2.;
                [poly[i][0] + dir[0] * d, poly[i][1] + dir[1] * d]
            } else {
                poly[i]
            }
        })
        .collect();
    (points, floating)
}
