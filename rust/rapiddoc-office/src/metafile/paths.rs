// Port of metafile-render 0.3.0 backends/paths.py, MIT (Xiaomeng Zhao).
use super::{
    clipper, composite,
    dib_decode::RgbaImage,
    geometry::{self, FlattenBudget},
    image_session::ImageSession,
    malformed,
    mask::Mask,
    model::*,
};
use crate::Result;
use std::sync::Arc;
pub type Subpaths = Vec<(Vec<Point>, bool)>;
pub fn subpath_mask(
    subpaths: &[(Vec<Point>, bool)],
    size: [usize; 2],
    fill_rule: &str,
    budget: &mut FlattenBudget,
) -> Result<Mask> {
    let mut paths = Vec::new();
    for (points, _) in subpaths {
        let p = clipper::quantize(points, true)?;
        if p.len() >= 3 {
            paths.push(p);
        }
    }
    let mut mask = Mask::new(size, 0)?;
    if !paths.is_empty() {
        mask.contours(&clipper::union(&paths, fill_rule == "evenodd", budget)?);
    }
    Ok(mask)
}
pub fn path_mask(
    path: &Path,
    matrix: Matrix,
    size: [usize; 2],
    fill_rule: &str,
    budget: &mut FlattenBudget,
) -> Result<Mask> {
    let subpaths = geometry::flatten(&geometry::transform(path, matrix), 0.25, 12, budget)?;
    subpath_mask(&subpaths, size, fill_rule, budget)
}
pub fn clip_mask(
    clip: &Arc<[Clip]>,
    matrix: Matrix,
    size: [usize; 2],
    budget: &mut FlattenBudget,
    session: &mut ImageSession,
) -> Result<Option<Arc<Mask>>> {
    if clip.is_empty() {
        return Ok(None);
    }
    if let Some(mask) = session.get_mask(clip, matrix, size) {
        return Ok(Some(mask));
    }
    let mut current = Mask::new(size, 255)?;
    for op in clip.iter() {
        let incoming = path_mask(&op.path, matrix, size, &op.fill_rule, budget)?;
        if op.mode == "copy" {
            current = incoming;
        } else {
            for (a, b) in current.pixels.iter_mut().zip(incoming.pixels) {
                *a = match op.mode.as_str() {
                    "and" => (u32::from(*a) * u32::from(b) / 255) as u8,
                    "or" => (*a).max(b),
                    "xor" => *a ^ b,
                    _ => (*a).saturating_sub(b),
                };
            }
        }
    }
    let mask = Arc::new(current);
    session.put_mask(clip, matrix, size, Arc::clone(&mask));
    Ok(Some(mask))
}
pub fn apply_clip(
    layer: &mut RgbaImage,
    clip: &Arc<[Clip]>,
    matrix: Matrix,
    budget: &mut FlattenBudget,
    session: &mut ImageSession,
) -> Result<()> {
    if let Some(mask) = clip_mask(clip, matrix, layer.size(), budget, session)? {
        for (p, &m) in layer.pixels.chunks_exact_mut(4).zip(&mask.pixels) {
            p[3] = (u32::from(p[3]) * u32::from(m) / 255) as u8;
        }
    }
    Ok(())
}
pub fn dash_polyline(points: &[Point], dashes: &[f64], closed: bool) -> Vec<Vec<Point>> {
    if points.len() < 2 {
        return vec![];
    }
    if dashes.is_empty() {
        return vec![points.to_vec()];
    }
    let pattern: Vec<_> = dashes.iter().map(|v| v.max(0.5)).collect();
    let (mut index, mut remaining, mut drawing) = (0, pattern[0], true);
    let mut visible = Vec::new();
    let mut current: Option<Vec<Point>> = None;
    for pair in points.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let delta = [b[0] - a[0], b[1] - a[1]];
        let length = delta[0].hypot(delta[1]);
        if length <= 1e-9 {
            continue;
        }
        let mut consumed = 0.;
        while consumed < length {
            let advance = remaining.min(length - consumed);
            if drawing && advance > 0. {
                let first = [
                    a[0] + delta[0] * consumed / length,
                    a[1] + delta[1] * consumed / length,
                ];
                let second = [
                    a[0] + delta[0] * (consumed + advance) / length,
                    a[1] + delta[1] * (consumed + advance) / length,
                ];
                let path = current.get_or_insert_with(|| vec![first]);
                if path.last() != Some(&first) {
                    path.push(first);
                }
                path.push(second);
            }
            consumed += advance;
            remaining -= advance;
            if remaining <= 1e-9 {
                if drawing {
                    if let Some(path) = current.take() {
                        visible.push(path);
                    }
                }
                index = (index + 1) % pattern.len();
                remaining = pattern[index];
                drawing = !drawing;
            }
        }
    }
    if let Some(path) = current {
        visible.push(path);
    }
    if closed
        && visible.len() > 1
        && visible[0][0] == points[0]
        && visible.last().unwrap().last() == points.last()
    {
        let mut tail = visible.pop().unwrap();
        tail.pop();
        tail.append(&mut visible[0]);
        visible[0] = tail;
    }
    visible
}
pub fn offset_stroke(
    points: &[Point],
    closed: bool,
    width: f64,
    pen: &Pen,
    miter: f64,
    size: [usize; 2],
    budget: &mut FlattenBudget,
) -> Result<Mask> {
    let mut mask = Mask::new(size, 0)?;
    let p = clipper::quantize(points, closed)?;
    if p.len() < if closed { 3 } else { 2 } {
        return Ok(mask);
    }
    let join = match pen.join.as_str() {
        "round" => 1,
        "bevel" => 0,
        "miter" => 2,
        _ => return Err(malformed()),
    };
    let end = if closed {
        1
    } else {
        match pen.cap.as_str() {
            "round" => 4,
            "square" => 3,
            "flat" => 2,
            _ => return Err(malformed()),
        }
    };
    mask.contours(&clipper::offset(&p, join, end, miter, width, budget)?);
    Ok(mask)
}
pub fn stroke_mask(
    subpaths: &[(Vec<Point>, bool)],
    size: [usize; 2],
    width: f64,
    pen: &Pen,
    dashes: &[f64],
    miter: f64,
    budget: &mut FlattenBudget,
) -> Result<Mask> {
    let mut mask = Mask::new(size, 0)?;
    for (points, closed) in subpaths {
        for segment in dash_polyline(points, dashes, *closed) {
            let incoming = offset_stroke(
                &segment,
                *closed && dashes.is_empty(),
                width,
                pen,
                miter,
                size,
                budget,
            )?;
            for (a, b) in mask.pixels.iter_mut().zip(incoming.pixels) {
                *a = (*a).max(b);
            }
        }
    }
    Ok(mask)
}
pub fn hatch(size: [usize; 2], brush: &Brush) -> Result<RgbaImage> {
    let mut layer = RgbaImage::blank(size)?;
    for y in 0..size[1] {
        for x in 0..size[0] {
            let painted = (matches!(brush.hatch, 0 | 4 | 5) && y % 8 == 0)
                || (matches!(brush.hatch, 1 | 4 | 5) && x % 8 == 0)
                || (matches!(brush.hatch, 2 | 4)
                    && (x as i64 - y as i64 + size[1] as i64).rem_euclid(8) == 0)
                || (matches!(brush.hatch, 3 | 5) && (x + y) % 8 == 0);
            if painted {
                layer.pixels[(y * size[0] + x) * 4..][..4].copy_from_slice(&[
                    brush.color.red,
                    brush.color.green,
                    brush.color.blue,
                    brush.color.alpha,
                ]);
            }
        }
    }
    Ok(layer)
}
fn solid(size: [usize; 2], color: Color, mask: &Mask) -> Result<RgbaImage> {
    let mut layer = RgbaImage::blank(size)?;
    for (p, &m) in layer.pixels.chunks_exact_mut(4).zip(&mask.pixels) {
        p.copy_from_slice(&[
            color.red,
            color.green,
            color.blue,
            (u32::from(color.alpha) * u32::from(m) / 255) as u8,
        ]);
    }
    Ok(layer)
}
pub fn render_path(
    command: &PathCommand,
    matrix: Matrix,
    size: [usize; 2],
    raster_scale: u32,
    budget: &mut FlattenBudget,
    session: &mut ImageSession,
) -> Result<RgbaImage> {
    let mut layer = RgbaImage::blank(size)?;
    let subpaths = geometry::flatten(
        &geometry::transform(&command.path, matrix),
        0.25,
        12,
        budget,
    )?;
    if command.fill && command.brush.kind != "null" {
        let mask = subpath_mask(&subpaths, size, &command.fill_rule, budget)?;
        let fill = if command.brush.kind == "hatch" {
            let mut fill = hatch(size, &command.brush)?;
            for (p, &m) in fill.pixels.chunks_exact_mut(4).zip(&mask.pixels) {
                p[3] = (u32::from(p[3]) * u32::from(m) / 255) as u8;
            }
            fill
        } else {
            solid(size, command.brush.color, &mask)?
        };
        composite::source_over(&mut layer.pixels, fill.pixels())?;
    }
    if command.stroke && !command.pen.null {
        let scale = (matrix.a.abs() + matrix.d.abs()) / 2.;
        let pen_width = command.pen.width
            * if command.pen.cosmetic {
                raster_scale as f64
            } else {
                scale.max(1e-9)
            };
        let width = pen_width.min(size[0].max(size[1]) as f64 * 2.).max(1.);
        let dash_scale = if command.pen.cosmetic {
            raster_scale as f64
        } else {
            scale
        };
        let dashes: Vec<_> = command.pen.dashes.iter().map(|d| d * dash_scale).collect();
        let mask = stroke_mask(
            &subpaths,
            size,
            width,
            &command.pen,
            &dashes,
            command.miter_limit,
            budget,
        )?;
        let stroke = solid(size, command.pen.color, &mask)?;
        composite::source_over(&mut layer.pixels, stroke.pixels())?;
    }
    apply_clip(&mut layer, &command.clip, matrix, budget, session)?;
    Ok(layer)
}
