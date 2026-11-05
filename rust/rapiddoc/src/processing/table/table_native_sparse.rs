//! 旧sparse_common/sparse_multiline的共享物理原语和列证据。
use super::{
    table_native_text::{Glyph, Row, Text},
    table_native_vector::{local, normalized, number, validate_text},
};
use crate::legacy_page::{invalid, limit, Work};
use rapiddoc_core::Error;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
mod tests {
    #[test]
    fn old_median_cluster_members_and_tolerance_edges() {
        let f: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-table-rule-band-extra.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["clusters"].as_array().unwrap() {
            let values: Vec<f64> = serde_json::from_value(c["values"].clone()).unwrap();
            let result = super::clusters(
                values,
                c["tolerance"].as_f64().unwrap(),
                &mut crate::legacy_page::Work(0),
            )
            .unwrap();
            assert_eq!(serde_json::json!(result), c["expected"]);
        }
    }
}
pub(super) fn dimensions(input: &Value) -> Result<(f64, f64, i64), Error> {
    let b = input["table_bbox"]
        .as_array()
        .filter(|b| b.len() == 4)
        .ok_or_else(invalid)?;
    let b = [
        number(&b[0])?,
        number(&b[1])?,
        number(&b[2])?,
        number(&b[3])?,
    ];
    let a = input.get("angle").map_or(Ok(0.), number)?.trunc() as i64;
    let a = a.rem_euclid(360);
    let a = if [0, 90, 180, 270].contains(&a) { a } else { 0 };
    let (w, h) = if [90, 270].contains(&a) {
        (b[3] - b[1], b[2] - b[0])
    } else {
        (b[2] - b[0], b[3] - b[1])
    };
    Ok((w, h, a))
}
pub(super) fn text(v: &Value) -> Result<Text, Error> {
    let text = validate_text(v)?;
    if text.median_glyph_height > 1e9 || text.median_glyph_width.abs() > 1e9 {
        return Err(limit());
    }
    if text
        .rows
        .iter()
        .enumerate()
        .any(|(i, r)| r.row_index != i as i64)
        || text
            .glyphs
            .iter()
            .any(|g| g.visual_row < 0 || g.visual_row as usize >= text.rows.len())
    {
        return Err(invalid());
    }
    let ids: BTreeSet<_> = text.glyphs.iter().map(|g| g.glyph_id).collect();
    if text
        .rows
        .iter()
        .any(|r| r.glyph_ids.iter().any(|i| !ids.contains(i)))
    {
        return Err(invalid());
    }
    Ok(text)
}
fn median(values: &[f64]) -> f64 {
    let m = values.len() / 2;
    if values.len() % 2 == 1 {
        values[m]
    } else {
        (values[m - 1] + values[m]) / 2.
    }
}
pub(super) fn clusters(
    mut values: Vec<f64>,
    tolerance: f64,
    work: &mut Work,
) -> Result<Vec<(f64, Vec<f64>)>, Error> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for value in values {
        work.charge(1)?;
        if groups
            .last()
            .is_some_and(|g| (value - median(g)).abs() <= tolerance)
        {
            groups.last_mut().unwrap().push(value)
        } else {
            groups.push(vec![value])
        }
    }
    Ok(groups.into_iter().map(|g| (median(&g), g)).collect())
}
#[derive(Serialize)]
pub(super) struct Rectangle {
    pub(super) bbox: [f64; 4],
    pub(super) fill_visible: bool,
    pub(super) stroke_visible: bool,
}
pub(super) fn rectangles(input: &Value, width: f64, height: f64) -> Result<Vec<Rectangle>, Error> {
    let Some(table) = normalized(&input["table_bbox"])? else {
        return Ok(Vec::new());
    };
    let (_, _, angle) = dimensions(input)?;
    let list = input.get("rectangles").map_or(Ok(&[][..]), |a| {
        a.as_array().map(|a| a.as_slice()).ok_or_else(invalid)
    })?;
    if list.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let mut out = Vec::new();
    for r in list {
        let count = r["segment_count"].as_i64().ok_or_else(invalid)?;
        let fill = r["fill_visible"].as_bool().ok_or_else(invalid)?;
        let stroke = r["stroke_visible"].as_bool().ok_or_else(invalid)?;
        if count != 5 || !(fill || stroke) {
            continue;
        }
        let Some(b) = normalized(&r["bbox"])?.and_then(|r| local(r, table, angle, 0.)) else {
            continue;
        };
        let b = [
            b[0].max(0.),
            b[1].max(0.),
            b[2].min(width),
            b[3].min(height),
        ];
        if b[2] <= b[0] || b[3] <= b[1] {
            continue;
        }
        out.push(Rectangle {
            bbox: b,
            fill_visible: fill,
            stroke_visible: stroke,
        })
    }
    Ok(out)
}
pub(super) fn occupancy(
    row: &Row,
    glyphs: &BTreeMap<usize, &Glyph>,
    x: &[f64],
    work: &mut Work,
) -> Result<BTreeSet<usize>, Error> {
    let mut out = BTreeSet::new();
    work.charge(row.glyph_ids.len() * x.len())?;
    for i in &row.glyph_ids {
        let g = glyphs.get(i).ok_or_else(invalid)?;
        let c = (g.bbox[0] + g.bbox[2]) / 2.;
        if let Some(col) = x.windows(2).position(|p| p[0] <= c && c <= p[1]) {
            out.insert(col);
        }
    }
    Ok(out)
}
pub(super) fn target(text: &Text) -> Option<usize> {
    let mut counts = BTreeMap::new();
    for r in &text.rows {
        if (2..=20).contains(&r.tokens.len()) {
            *counts.entry(r.tokens.len()).or_insert(0) += 1
        }
    }
    counts.iter().rev().find(|(_, n)| **n >= 2).map(|(n, _)| *n)
}
fn refine(text: &Text, x: &[f64], work: &mut Work) -> Result<Vec<f64>, Error> {
    let mut out = x.to_vec();
    for i in 1..x.len() - 1 {
        let boundary = out[i];
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for row in &text.rows {
            for t in &row.tokens {
                work.charge(x.len())?;
                let crossed = x[1..x.len() - 1]
                    .iter()
                    .filter(|b| t.bbox[0] < **b && **b < t.bbox[2])
                    .count();
                if crossed > 1 {
                    continue;
                }
                let c = (t.bbox[0] + t.bbox[2]) / 2.;
                if c < boundary {
                    left.push(t.bbox[2])
                } else if c > boundary {
                    right.push(t.bbox[0])
                }
            }
        }
        if !left.is_empty() && !right.is_empty() {
            let low = left.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let high = right.into_iter().fold(f64::INFINITY, f64::min);
            if out[i - 1] < low && low < high && high < out[i + 1] {
                out[i] = (low + high) / 2.
            }
        }
    }
    Ok(out)
}
pub(super) fn text_tracks(
    text: &Text,
    width: f64,
    cols: usize,
    work: &mut Work,
) -> Result<Option<Vec<f64>>, Error> {
    let rows: Vec<_> = text
        .rows
        .iter()
        .filter(|r| r.tokens.len() == cols)
        .collect();
    if rows.len() < 2 {
        return Ok(None);
    }
    let mut x = vec![0.];
    for col in 0..cols - 1 {
        work.charge(rows.len())?;
        let low = rows
            .iter()
            .map(|r| r.tokens[col].bbox[2])
            .fold(f64::NEG_INFINITY, f64::max);
        let high = rows
            .iter()
            .map(|r| r.tokens[col + 1].bbox[0])
            .fold(f64::INFINITY, f64::min);
        let b = if low < high {
            (low + high) / 2.
        } else {
            let mut mid: Vec<_> = rows
                .iter()
                .map(|r| (r.tokens[col].bbox[2] + r.tokens[col + 1].bbox[0]) / 2.)
                .collect();
            mid.sort_by(|a, b| a.partial_cmp(b).unwrap());
            median(&mid)
        };
        if !(0. < b && b < width) {
            return Ok(None);
        }
        x.push(b)
    }
    x.push(width);
    let x = refine(text, &x, work)?;
    let minimum = (0.50 * text.median_glyph_width).max(1.);
    Ok(x.windows(2).all(|p| p[1] - p[0] >= minimum).then_some(x))
}
fn rectangle_tracks(
    rectangles: &[Rectangle],
    width: f64,
    tolerance: f64,
    outer: f64,
    work: &mut Work,
) -> Result<Vec<f64>, Error> {
    let mut x = vec![0., width];
    for (c, m) in clusters(
        rectangles
            .iter()
            .flat_map(|r| [r.bbox[0], r.bbox[2]])
            .collect(),
        tolerance,
        work,
    )? {
        if m.len() < 2 {
            continue;
        }
        x.push(if c <= outer {
            0.
        } else if width - c <= outer {
            width
        } else {
            c
        })
    }
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    x.dedup();
    Ok(if x.windows(2).any(|p| p[1] <= p[0]) {
        Vec::new()
    } else {
        x
    })
}
#[derive(Serialize)]
pub(super) struct Hypothesis {
    pub(super) evidence: String,
    pub(super) x_tracks: Vec<f64>,
    pub(super) physical_boundaries: BTreeSet<usize>,
    pub(super) filled_band_count: usize,
}
pub(super) fn hypothesis(
    text: &Text,
    width: f64,
    rules: &[Rule],
    rectangles: &[Rectangle],
    d: &mut Map<String, Value>,
    work: &mut Work,
) -> Result<Option<Hypothesis>, Error> {
    let Some(cols) = target(text) else {
        d.insert("first_rejection_gate".into(), json!("column_count"));
        return Ok(None);
    };
    let mut tracks = text_tracks(text, width, cols, work)?;
    let tolerance = (0.25 * text.median_glyph_height).max(1.);
    let rect = rectangle_tracks(
        rectangles,
        width,
        tolerance,
        tolerance.max(1.50 * text.median_glyph_height),
        work,
    )?;
    let mut boundaries = BTreeSet::new();
    if rect.len() == cols + 1 {
        tracks = Some(rect);
        boundaries = (1..cols).collect()
    }
    let Some(tracks) = tracks else {
        d.insert("first_rejection_gate".into(), json!("column_tracks"));
        return Ok(None);
    };
    let centers = rectangles
        .iter()
        .filter(|r| {
            r.fill_visible
                && r.bbox[3] - r.bbox[1] >= 0.75 * text.median_glyph_height
                && r.bbox[2] - r.bbox[0] >= 0.08 * width
        })
        .map(|r| (r.bbox[1] + r.bbox[3]) / 2.)
        .collect();
    let bands = clusters(
        centers,
        tolerance.max(0.50 * text.median_glyph_height),
        work,
    )?
    .len();
    let maximum = text
        .rows
        .iter()
        .map(|r| r.bbox[3])
        .max_by(f64::total_cmp)
        .unwrap_or(0.);
    let internal = rules.iter().any(|r| {
        r.orientation == "horizontal"
            && r.end - r.start >= 0.90 * width
            && 0.01 * maximum < r.coordinate
            && r.coordinate < 0.99 * maximum
    });
    let evidence = if bands >= 2 && !boundaries.is_empty() {
        "filled_record"
    } else if internal {
        "rule_band"
    } else {
        "keyed_record"
    };
    d.insert("target_cols".into(), json!(cols));
    d.insert("x_tracks".into(), json!(tracks));
    d.insert("physical_boundaries".into(), json!(boundaries));
    d.insert("filled_band_count".into(), json!(bands));
    d.insert("evidence".into(), json!(evidence));
    Ok(Some(Hypothesis {
        evidence: evidence.into(),
        x_tracks: tracks,
        physical_boundaries: boundaries,
        filled_band_count: bands,
    }))
}
pub(super) fn overlapping(text: &Text, bottom: f64) -> bool {
    text.rows.iter().any(|r| {
        (r.bbox[1] + r.bbox[3]) / 2. > bottom
            && r.bbox[3] - r.bbox[1] > 3. * text.median_glyph_height
    })
}
pub(super) fn gutters(
    text: &Text,
    x: &[f64],
    physical: &BTreeSet<usize>,
    bottom: f64,
    d: &mut Map<String, Value>,
    work: &mut Work,
) -> Result<bool, Error> {
    let mut by_row = BTreeMap::<i64, Vec<&Glyph>>::new();
    for g in &text.glyphs {
        by_row.entry(g.visual_row).or_default().push(g)
    }
    let body: Vec<_> = text
        .glyphs
        .iter()
        .filter(|g| {
            let r = &text.rows[g.visual_row as usize];
            (r.bbox[1] + r.bbox[3]) / 2. > bottom
        })
        .collect();
    let tolerance = (0.03 * text.median_glyph_width).max(0.15);
    let mut supports = Vec::new();
    for (index, boundary) in x.iter().enumerate().take(x.len() - 1).skip(1) {
        work.charge(body.len())?;
        if body
            .iter()
            .any(|g| g.bbox[0] + tolerance < *boundary && *boundary < g.bbox[2] - tolerance)
        {
            d.insert("failed_gutter_boundary".into(), json!(index));
            d.insert("failed_gutter_reason".into(), json!("glyph_crossing"));
            return Ok(false);
        }
        if physical.contains(&index) {
            supports.push(text.rows.len());
            continue;
        }
        let (mut comparable, mut stable) = (0usize, 0usize);
        for r in &text.rows {
            if (r.bbox[1] + r.bbox[3]) / 2. <= bottom {
                continue;
            }
            let gs = by_row.get(&r.row_index).map_or(&[][..], Vec::as_slice);
            work.charge(gs.len())?;
            let left = gs
                .iter()
                .filter(|g| g.bbox[2] <= *boundary)
                .map(|g| g.bbox[2])
                .max_by(f64::total_cmp);
            let right = gs
                .iter()
                .filter(|g| g.bbox[0] >= *boundary)
                .map(|g| g.bbox[0])
                .min_by(f64::total_cmp);
            if let (Some(left), Some(right)) = (left, right) {
                comparable += 1;
                if right - left >= (0.10 * text.median_glyph_width).max(0.25) {
                    stable += 1
                }
            }
        }
        if comparable < 3 || (stable as f64) / (comparable as f64) < 0.80 {
            d.insert("failed_gutter_boundary".into(), json!(index));
            d.insert("failed_gutter_reason".into(), json!("support"));
            d.insert("failed_gutter_comparable".into(), json!(comparable));
            d.insert("failed_gutter_stable".into(), json!(stable));
            return Ok(false);
        }
        supports.push(stable)
    }
    d.insert("gutter_supports".into(), json!(supports));
    Ok(true)
}
#[derive(Deserialize, Serialize)]
pub(super) struct Rule {
    pub(super) orientation: String,
    pub(super) coordinate: f64,
    pub(super) start: f64,
    pub(super) end: f64,
    pub(super) width: f64,
}
pub(super) fn local_rules(input: &Value, w: f64, h: f64) -> Result<Vec<Rule>, Error> {
    let a = input.get("angle").map_or(Ok(0.), number)?.trunc() as i64;
    let a = a.rem_euclid(360);
    let a = if [0, 90, 180, 270].contains(&a) { a } else { 0 };
    let Some(b) = normalized(&input["table_bbox"])? else {
        return Ok(Vec::new());
    };
    let lines = input.get("drawing_lines").map_or(Ok(&[][..]), |a| {
        a.as_array().map(|a| a.as_slice()).ok_or_else(invalid)
    })?;
    if lines.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let mut out = Vec::new();
    for r in lines {
        let Some(b) = normalized(&r["bbox"])?.and_then(|r| local(r, b, a, 0.)) else {
            continue;
        };
        let width = number(&r["width"])?;
        let (rw, rh) = (b[2] - b[0], b[3] - b[1]);
        if rw >= (3. * rh.max(0.1)).max(1.) {
            out.push(Rule {
                orientation: "horizontal".into(),
                coordinate: (b[1] + b[3]) / 2.,
                start: b[0].max(0.),
                end: b[2].min(w),
                width: width.max(rh),
            })
        } else if rh >= (3. * rw.max(0.1)).max(1.) {
            out.push(Rule {
                orientation: "vertical".into(),
                coordinate: (b[0] + b[2]) / 2.,
                start: b[1].max(0.),
                end: b[3].min(h),
                width: width.max(rw),
            })
        }
    }
    Ok(out)
}
