//! Upstream 2c6c0ad: writing direction affects reading copies, never physical ownership.
use super::*;
pub(super) fn zero_angle() -> Option<i32> {
    Some(0)
}
fn point([x, y]: [f64; 2], a: i32) -> [f64; 2] {
    match a {
        90 => [y, -x],
        180 => [-x, -y],
        270 => [-y, x],
        _ => [x, y],
    }
}
pub(super) fn rotated(b: Box4, a: i32) -> Box4 {
    let p = point([b[0], b[1]], a);
    let q = point([b[2], b[3]], a);
    [
        p[0].min(q[0]),
        p[1].min(q[1]),
        p[0].max(q[0]),
        p[1].max(q[1]),
    ]
}
pub(super) fn metadata(
    c: &Value,
    t: Box4,
    a: i32,
) -> Result<(Option<i32>, Option<Box4>, Option<[f64; 2]>), Error> {
    let direction = if c["writing_angle"].is_null() {
        Some(0)
    } else {
        number(&c["writing_angle"])?.and_then(|v| {
            let relative = (v.to_degrees().rem_euclid(360.) - a as f64).rem_euclid(360.);
            let cardinal = ((relative / 90.).round_ties_even() * 90.).rem_euclid(360.);
            (((relative - cardinal + 180.).rem_euclid(360.) - 180.).abs() <= 0.1)
                .then_some(cardinal as i32)
        })
    };
    let advance = if direction.is_some_and(|v| v != 0) {
        normalize_bbox(&c["loose_bbox"])?.and_then(|b| local(b, t, a))
    } else {
        None
    };
    let origin = c["origin"]
        .as_array()
        .filter(|v| v.len() == 2)
        .and_then(|p| {
            let x = number(&p[0]).ok().flatten()?;
            let y = number(&p[1]).ok().flatten()?;
            let [x, y] = point([x - t[0], y - t[1]], a);
            let (w, h) = (t[2] - t[0], t[3] - t[1]);
            let [dx, dy] = match a {
                90 => [0., w],
                180 => [w, h],
                270 => [h, 0.],
                _ => [0., 0.],
            };
            Some([x + dx, y + dy])
        });
    Ok((direction, advance, origin))
}
pub(super) fn baseline(g: &Pending) -> Option<i32> {
    let Some(a) = g.writing_angle.filter(|a| *a != 0) else {
        return g.writing_angle;
    };
    let Some(origin) = g.origin else {
        return Some(a);
    };
    let Some(reference) = g.advance_bbox.or_else(|| {
        g.text
            .chars()
            .any(rapiddoc_native::text_boundary::python_alnum)
            .then_some(g.bbox)
    }) else {
        return Some(a);
    };
    let b = rotated(reference, a);
    let p = point(origin, a);
    let (above, below) = (p[1] - b[1], b[3] - p[1]);
    Some(
        if above >= -0.1 * (b[3] - b[1])
            && below > 0.
            && below >= 2. * above
            && below - above >= 0.4 * (b[3] - b[1])
        {
            (a + 180) % 360
        } else {
            a
        },
    )
}
fn supports(glyphs: &[Glyph], a: i32) -> bool {
    let anchors: Vec<_> = glyphs
        .iter()
        .filter(|g| g.writing_angle == Some(a))
        .filter_map(|g| g.origin.map(|p| (rotated(g.bbox, a), point(p, a))))
        .collect();
    if anchors.len() < 2 {
        return false;
    }
    let scale = median(anchors.iter().map(|(b, _)| b[3] - b[1]).collect());
    for g in glyphs.iter().filter(|g| g.writing_angle != Some(a)) {
        let Some(origin) = g.origin else { return false };
        if g.writing_angle
            .is_some_and(|v| (v - a).rem_euclid(360) != 180)
        {
            return false;
        }
        let b = rotated(g.bbox, a);
        let p = point(origin, a);
        let neighbors: Vec<_> = anchors
            .iter()
            .filter(|(peer, base)| {
                (p[1] - base[1]).abs() <= 0.15 * scale
                    && (peer[0] - b[2]).max(b[0] - peer[2]).max(0.) <= 2. * scale
            })
            .collect();
        if neighbors.len() < 2 {
            return false;
        }
        let lo = neighbors
            .iter()
            .map(|(_, p)| p[0])
            .fold(f64::INFINITY, f64::min);
        let hi = neighbors
            .iter()
            .map(|(_, p)| p[0])
            .fold(f64::NEG_INFINITY, f64::max);
        if hi - lo < 0.5 * scale {
            return false;
        }
        let center = (b[1] + b[3]) / 2. - p[1];
        let side = median(
            neighbors
                .iter()
                .map(|(b, p)| (b[1] + b[3]) / 2. - p[1])
                .collect(),
        );
        if center * side <= 0. || (center - side).abs() > 0.6 * scale {
            return false;
        }
    }
    true
}
pub(crate) fn writing_angle(glyphs: &[Glyph]) -> Option<i32> {
    let directions: BTreeSet<_> = glyphs.iter().map(|g| g.writing_angle).collect();
    if directions.is_empty() {
        return Some(0);
    }
    if directions.len() == 1 {
        return *directions.first().unwrap();
    }
    let accepted: Vec<_> = directions
        .iter()
        .filter_map(|&a| a)
        .filter(|&a| supports(glyphs, a))
        .collect();
    (accepted.len() == 1).then(|| accepted[0])
}
pub(crate) fn reading_rows(
    mut glyphs: Vec<Glyph>,
    height: f64,
) -> Result<(Vec<Glyph>, Vec<Vec<usize>>, f64), Error> {
    let a = writing_angle(&glyphs).ok_or_else(invalid)?;
    if a == 0 {
        let kernel = if crate::legacy_page::budget::current().host {
            rapiddoc_native::table_cell_visual_groups_page
        } else {
            rapiddoc_native::table_cell_visual_groups
        };
        let rows = kernel(
            glyphs.iter().map(|g| g.bbox).collect(),
            glyphs.iter().map(|g| g.glyph_id).collect(),
            glyphs.iter().map(|g| g.visual_row).collect(),
            height,
        )?;
        return Ok((glyphs, rows, height));
    }
    for g in &mut glyphs {
        g.bbox = rotated(g.bbox, a);
        g.advance_bbox = g.advance_bbox.map(|b| rotated(b, a));
    }
    let height = median(glyphs.iter().map(|g| g.bbox[3] - g.bbox[1]).collect());
    let kernel = if crate::legacy_page::budget::current().host {
        rapiddoc_native::table_visual_rows_page
    } else {
        rapiddoc_native::table_visual_rows
    };
    let rows = kernel(
        glyphs.iter().map(|g| g.bbox).collect(),
        glyphs.iter().map(|g| g.glyph_id).collect(),
        height,
    )?;
    for (index, row) in rows.iter().enumerate() {
        for &i in row {
            glyphs[i].visual_row = index as i64;
        }
    }
    Ok((glyphs, rows, height))
}
