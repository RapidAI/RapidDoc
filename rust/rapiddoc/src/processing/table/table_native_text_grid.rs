//! 旧text_grid.py少线/无线/key-value候选，普通JSON页面原语与NativeTableText输入。
use super::{
    table_native_candidate,
    table_native_text::{Row, Text},
    table_native_vector::{local, normalized, number, validate_text},
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
fn median(mut a: Vec<f64>) -> f64 {
    a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = a.len() / 2;
    if a.len() % 2 == 1 {
        a[m]
    } else {
        (a[m - 1] + a[m]) / 2.
    }
}
fn column_count(text: &Text) -> Option<usize> {
    let mut counts = BTreeMap::new();
    for r in &text.rows {
        if r.tokens.len() >= 2 {
            *counts.entry(r.tokens.len()).or_insert(0) += 1
        }
    }
    counts
        .iter()
        .rev()
        .find(|(_, n)| **n >= 2)
        .map(|(n, _)| *n)
        .or_else(|| {
            if text.rows.len() <= 3 {
                counts.keys().next_back().copied()
            } else {
                None
            }
        })
}
fn columns(row: &Row, x: &[f64], work: &mut Work) -> Result<Option<Vec<usize>>, Error> {
    let mut out = Vec::new();
    for token in &row.tokens {
        work.charge(x.len())?;
        let c = (token.bbox[0] + token.bbox[2]) / 2.;
        let Some(col) = x.windows(2).position(|p| p[0] <= c && c <= p[1]) else {
            return Ok(None);
        };
        if out.last().is_some_and(|last| col < *last) {
            return Ok(None);
        }
        out.push(col)
    }
    Ok(Some(out))
}
fn infer_columns(
    text: &Text,
    width: f64,
    cols: usize,
    work: &mut Work,
) -> Result<Option<(Vec<f64>, f64, f64)>, Error> {
    let dense: Vec<_> = text
        .rows
        .iter()
        .filter(|r| r.tokens.len() == cols)
        .collect();
    if dense.is_empty() || dense.len() < 2 && text.rows.len() > 3 {
        return Ok(None);
    }
    let mut tracks = vec![0.];
    let mut supports = Vec::new();
    let gap = (0.15 * text.median_glyph_height).max(1.);
    for col in 0..cols - 1 {
        let mut midpoints = Vec::new();
        let mut valid = 0;
        work.charge(dense.len())?;
        for row in &dense {
            let (a, b) = (&row.tokens[col], &row.tokens[col + 1]);
            valid += usize::from(b.bbox[0] - a.bbox[2] >= gap);
            midpoints.push((a.bbox[2] + b.bbox[0]) / 2.)
        }
        let support = valid as f64 / dense.len() as f64;
        if support < 0.60 {
            return Ok(None);
        }
        tracks.push(median(midpoints));
        supports.push(support)
    }
    tracks.push(width);
    if !tracks.windows(2).all(|p| p[1] > p[0]) {
        return Ok(None);
    }
    let first = dense.iter().map(|r| r.row_index).min().unwrap();
    let margin = (0.08 * text.median_glyph_height).max(0.5);
    for row in &text.rows {
        if row.row_index < first {
            continue;
        }
        work.charge(row.tokens.len() * cols)?;
        if row.tokens.iter().any(|t| {
            tracks[1..cols]
                .iter()
                .any(|x| t.bbox[0] + margin < *x && *x < t.bbox[2] - margin)
        }) {
            return Ok(None);
        }
    }
    let mut supported = vec![BTreeSet::new(); cols];
    let (mut aligned, mut total) = (0usize, 0usize);
    for row in &text.rows {
        let mut prev = None;
        for token in &row.tokens {
            work.charge(tracks.len())?;
            total += 1;
            let c = (token.bbox[0] + token.bbox[2]) / 2.;
            let Some(col) = tracks.windows(2).position(|p| p[0] <= c && c <= p[1]) else {
                continue;
            };
            if prev.is_some_and(|p| col < p) {
                continue;
            }
            supported[col].insert(row.row_index);
            aligned += 1;
            prev = Some(col)
        }
    }
    let rows = text.rows.len().max(1);
    let fractions: Vec<_> = supported
        .iter()
        .map(|r| r.len() as f64 / rows as f64)
        .collect();
    let anchor = table_native_candidate::sum(&fractions) / cols as f64;
    if anchor < 0.60 {
        return Ok(None);
    }
    let alignment = if total == 0 {
        0.
    } else {
        aligned as f64 / total as f64
    };
    let gaps = if supports.is_empty() {
        1.
    } else {
        rapiddoc_native::table_decisiveness_mean(&supports)?
    };
    Ok(Some((tracks, anchor, alignment.min(gaps))))
}
struct Geometry {
    box_: Option<[f64; 4]>,
    width: f64,
    height: f64,
    rules: Vec<[f64; 4]>,
    stripes: Vec<[f64; 4]>,
}
fn geometry(input: &Value) -> Result<Geometry, Error> {
    let bbox = normalized(&input["table_bbox"])?;
    let angle = input.get("angle").map_or(Ok(0.), number)?.trunc() as i64;
    let n = angle.rem_euclid(360);
    let angle = if [0, 90, 180, 270].contains(&n) { n } else { 0 };
    let lines = input.get("drawing_lines").map_or(Ok(&[][..]), |a| {
        a.as_array().map(|a| a.as_slice()).ok_or_else(invalid)
    })?;
    let rects = input.get("rectangles").map_or(Ok(&[][..]), |a| {
        a.as_array().map(|a| a.as_slice()).ok_or_else(invalid)
    })?;
    if lines.len() + rects.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let Some(b) = bbox else {
        return Ok(Geometry {
            box_: None,
            width: 0.,
            height: 0.,
            rules: Vec::new(),
            stripes: Vec::new(),
        });
    };
    let (width, height) = if angle == 90 || angle == 270 {
        (b[3] - b[1], b[2] - b[0])
    } else {
        (b[2] - b[0], b[3] - b[1])
    };
    let mut rules = Vec::new();
    let mut stripes = Vec::new();
    for r in lines {
        if let Some(b) = normalized(&r["bbox"])?.and_then(|a| local(a, b, angle, 0.)) {
            rules.push(b)
        }
    }
    for r in rects {
        if r["segment_count"].as_i64().ok_or_else(invalid)? != 5
            || !r["fill_visible"].as_bool().ok_or_else(invalid)?
        {
            continue;
        }
        if let Some(b) = normalized(&r["bbox"])?.and_then(|a| local(a, b, angle, 0.)) {
            stripes.push(b)
        }
    }
    Ok(Geometry {
        box_: Some(b),
        width,
        height,
        rules,
        stripes,
    })
}
impl Geometry {
    fn physical(&self, text: &Text) -> f64 {
        if self.box_.is_none() {
            return 0.;
        }
        let long = self
            .rules
            .iter()
            .filter(|b| b[2] - b[0] >= 0.50 * self.width && b[2] - b[0] >= 4. * (b[3] - b[1]))
            .count();
        let stripe = self
            .stripes
            .iter()
            .filter(|b| {
                b[2] - b[0] >= 0.60 * self.width
                    && 0.30 * text.median_glyph_height <= b[3] - b[1]
                    && b[3] - b[1] <= (3. * text.median_glyph_height).min(0.50 * self.height)
            })
            .count();
        ((long + stripe) as f64 / 3.).min(1.)
    }
    fn separates(&self, a: &Row, b: &Row, work: &mut Work) -> Result<bool, Error> {
        if self.box_.is_none() {
            return Ok(true);
        }
        work.charge(self.rules.len())?;
        Ok(self.rules.iter().any(|r| {
            let (w, h) = (r[2] - r[0], r[3] - r[1]);
            let cy = (r[1] + r[3]) / 2.;
            w >= 0.60 * self.width && w >= 4. * h.max(0.1) && a.bbox[3] <= cy && cy <= b.bbox[1]
        }))
    }
}
struct Grouping {
    rows: Vec<Row>,
    dense: Vec<[i64; 2]>,
    subsets: Vec<[i64; 2]>,
}
fn grouping(text: &Text, x: &[f64], g: &Geometry, work: &mut Work) -> Result<Grouping, Error> {
    let mut groups: Vec<Vec<&Row>> = Vec::new();
    let mut occupancy: Vec<BTreeSet<usize>> = Vec::new();
    let mut dense = Vec::new();
    let mut subsets = Vec::new();
    let gap = (0.40 * text.median_glyph_height).max(0.75);
    let minimum = 2usize.max((0.60 * (x.len() - 1) as f64).ceil() as usize);
    for row in &text.rows {
        let cols: BTreeSet<_> = columns(row, x, work)?
            .unwrap_or_default()
            .into_iter()
            .collect();
        if let Some(last) = groups.last() {
            let previous = *last.last().unwrap();
            let near = row.bbox[1] - previous.bbox[3] <= gap;
            let separator = g.separates(previous, row, work)?;
            let parent = occupancy.last().unwrap();
            if cols == *parent && cols.len() >= minimum && near && !separator {
                dense.push([previous.row_index, row.row_index])
            }
            if !cols.is_empty()
                && cols.len() < parent.len()
                && cols.is_subset(parent)
                && near
                && !separator
            {
                groups.last_mut().unwrap().push(row);
                subsets.push([previous.row_index, row.row_index]);
                continue;
            }
        }
        groups.push(vec![row]);
        occupancy.push(cols);
    }
    let rows = groups
        .iter()
        .enumerate()
        .map(|(i, rows)| {
            let mut tokens: Vec<_> = rows.iter().flat_map(|r| r.tokens.iter().cloned()).collect();
            tokens.sort_by(|a, b| {
                a.bbox[0]
                    .partial_cmp(&b.bbox[0])
                    .unwrap()
                    .then(a.bbox[1].partial_cmp(&b.bbox[1]).unwrap())
            });
            Row {
                row_index: i as i64,
                bbox: [
                    rows.iter().map(|r| r.bbox[0]).fold(f64::INFINITY, f64::min),
                    rows.iter().map(|r| r.bbox[1]).fold(f64::INFINITY, f64::min),
                    rows.iter()
                        .map(|r| r.bbox[2])
                        .fold(f64::NEG_INFINITY, f64::max),
                    rows.iter()
                        .map(|r| r.bbox[3])
                        .fold(f64::NEG_INFINITY, f64::max),
                ],
                tokens,
                glyph_ids: rows
                    .iter()
                    .flat_map(|r| r.glyph_ids.iter().copied())
                    .collect(),
            }
        })
        .collect();
    Ok(Grouping {
        rows,
        dense,
        subsets,
    })
}
fn row_tracks(rows: &[Row], height: f64) -> Option<Vec<f64>> {
    if rows.len() < 2 {
        return None;
    }
    let centers: Vec<_> = rows.iter().map(|r| (r.bbox[1] + r.bbox[3]) / 2.).collect();
    let mut tracks = vec![0.];
    tracks.extend(centers.windows(2).map(|p| (p[0] + p[1]) / 2.));
    tracks.push(height);
    tracks.windows(2).all(|p| p[1] > p[0]).then_some(tracks)
}
fn header_spans(
    row: &Row,
    next: &Row,
    x: &[f64],
    work: &mut Work,
) -> Result<Option<Vec<(usize, usize)>>, Error> {
    if row.tokens.len() < 2 {
        if row.tokens.len() != 1 {
            return Ok(None);
        }
        let b = row.tokens[0].bbox;
        let width = b[2] - b[0];
        if width <= 0. {
            return Ok(None);
        }
        work.charge(x.len())?;
        let cols: Vec<_> = x
            .windows(2)
            .enumerate()
            .filter(|(_, p)| (b[2].min(p[1]) - b[0].max(p[0])).max(0.) / width >= 0.20)
            .map(|(i, _)| i)
            .collect();
        return Ok(
            if cols.len() >= 2 && cols.windows(2).all(|p| p[1] == p[0] + 1) {
                Some(vec![(cols[0], *cols.last().unwrap())])
            } else {
                None
            },
        );
    }
    let Some(child) = columns(next, x, work)? else {
        return Ok(None);
    };
    let child: BTreeSet<_> = child.into_iter().collect();
    if child.len() < row.tokens.len() {
        return Ok(None);
    }
    let centers: Vec<_> = row
        .tokens
        .iter()
        .map(|t| (t.bbox[0] + t.bbox[2]) / 2.)
        .collect();
    let gaps: Vec<_> = centers
        .windows(2)
        .filter(|p| p[1] > p[0])
        .map(|p| p[1] - p[0])
        .collect();
    let maximum = if gaps.is_empty() {
        f64::INFINITY
    } else {
        1.5 * median(gaps)
    };
    let mut groups = vec![Vec::new(); row.tokens.len()];
    for c in child {
        work.charge(centers.len())?;
        let center = (x[c] + x[c + 1]) / 2.;
        let owner = centers
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (center - **a)
                    .abs()
                    .partial_cmp(&(center - **b).abs())
                    .unwrap()
            })
            .unwrap()
            .0;
        if (center - centers[owner]).abs() <= maximum {
            groups[owner].push(c)
        }
    }
    let mut spans = Vec::new();
    for (i, cols) in groups.iter().enumerate() {
        if cols.is_empty()
            || !cols.windows(2).all(|p| p[1] == p[0] + 1)
            || !(x[cols[0]] <= centers[i] && centers[i] <= x[cols.last().unwrap() + 1])
        {
            return Ok(None);
        }
        spans.push((cols[0], *cols.last().unwrap()))
    }
    Ok(Some(spans))
}
fn representable(rows: &[Row], x: &[f64], first: usize, work: &mut Work) -> Result<bool, Error> {
    for i in 0..first {
        let spans = if i + 1 < rows.len() {
            header_spans(&rows[i], &rows[i + 1], x, work)?
        } else {
            None
        };
        let covered: BTreeSet<_> = if let Some(spans) = spans {
            spans.iter().flat_map(|(a, b)| *a..=*b).collect()
        } else {
            let Some(c) = columns(&rows[i], x, work)? else {
                return Ok(false);
            };
            c.into_iter().collect()
        };
        if covered != (0..x.len() - 1).collect() {
            return Ok(false);
        }
    }
    Ok(true)
}
fn specs(
    rows: &[Row],
    x: &[f64],
    y: &[f64],
    first: usize,
    work: &mut Work,
) -> Result<Option<(Vec<Value>, f64)>, Error> {
    let cols = x.len() - 1;
    let mut result = Vec::new();
    let mut stable = 0;
    let cell = |r: usize, a: usize, b: usize| json!({"row":r,"col":a,"rowspan":1,"colspan":b-a+1,"bbox":[x[a],y[r],x[b+1],y[r+1]]});
    for (i, row) in rows.iter().enumerate() {
        let spans = if i < first && i + 1 < rows.len() {
            header_spans(row, &rows[i + 1], x, work)?
        } else {
            None
        };
        let mut occupied = BTreeSet::new();
        if let Some(spans) = &spans {
            for (a, b) in spans {
                result.push(cell(i, *a, *b));
                occupied.extend(*a..=*b)
            }
            stable += 1
        } else {
            let Some(c) = columns(row, x, work)? else {
                return Ok(None);
            };
            occupied.extend(c);
            stable += usize::from(occupied.len() >= 2)
        }
        work.charge(cols)?;
        for c in 0..cols {
            if occupied.contains(&c) && spans.is_some() {
                continue;
            }
            result.push(cell(i, c, c))
        }
        if result.len() > rapiddoc_core::resources::limit(2000) {
            return Err(limit());
        }
    }
    Ok(Some((
        result,
        if rows.is_empty() {
            0.
        } else {
            stable as f64 / rows.len() as f64
        },
    )))
}
fn rejected(d: &mut Map<String, Value>, gate: &str) -> Value {
    d.insert("first_rejection_gate".into(), json!(gate));
    Value::Null
}
struct Build<'a> {
    text: &'a Text,
    raw_text: &'a Value,
    g: &'a Geometry,
    source: &'a str,
    physical: f64,
}
impl Build<'_> {
    fn run(&self, d: &mut Map<String, Value>, work: &mut Work) -> Result<Value, Error> {
        let text = self.text;
        d.insert("source".into(), json!(self.source));
        d.insert("raw_visual_rows".into(), json!(text.rows.len()));
        let Some(cols) = column_count(text) else {
            return Ok(rejected(d, "column_count"));
        };
        if self.source == "text_grid" && cols < 3 || self.source == "key_value" && cols != 2 {
            return Ok(rejected(d, "column_count"));
        }
        if self.source != "sparse_grid" && text.rows.len() < 3 {
            return Ok(rejected(d, "row_count"));
        }
        if self.g.box_.is_none() {
            return Ok(rejected(d, "table_geometry"));
        }
        if cols > rapiddoc_core::resources::limit(200) {
            return Err(limit());
        }
        let Some((x, anchor, alignment)) = infer_columns(text, self.g.width, cols, work)? else {
            return Ok(rejected(d, "column_tracks"));
        };
        let groups = grouping(text, &x, self.g, work)?;
        let rows = &groups.rows;
        d.insert("logical_rows".into(), json!(rows.len()));
        d.insert("dense_row_ambiguities".into(), json!(groups.dense));
        d.insert("subset_continuation_merges".into(), json!(groups.subsets));
        if !groups.dense.is_empty() {
            return Ok(rejected(d, "dense_row_ambiguity"));
        }
        let Some(y) = row_tracks(rows, self.g.height) else {
            return Ok(rejected(d, "row_tracks"));
        };
        let mut dense = Vec::new();
        for row in rows {
            let c: BTreeSet<_> = columns(row, &x, work)?
                .unwrap_or_default()
                .into_iter()
                .collect();
            if c.len() == cols {
                dense.push(row.row_index as usize)
            }
        }
        let first = dense.into_iter().min().unwrap_or(0);
        let representable = representable(rows, &x, first, work)?;
        d.insert("first_dense_row".into(), json!(first));
        d.insert("header_representable".into(), json!(representable));
        if !representable {
            return Ok(rejected(d, "header_requires_rowspan"));
        }
        let Some((specs, stability)) = specs(rows, &x, &y, first, work)? else {
            return Ok(rejected(d, "grid_specs"));
        };
        if stability < 0.60 {
            return Ok(rejected(d, "row_stability"));
        }
        let structure = if self.source == "sparse_grid" {
            anchor.max(self.physical)
        } else {
            alignment
        };
        work.charge(
            text.glyphs.len() * specs.len() * 3 + text.glyphs.len() * text.glyphs.len() * 2,
        )?;
        let result = table_native_candidate::build(
            &json!({"source":self.source,"rows":rows.len(),"cols":cols,"specs":specs,"text":self.raw_text,"structure_support":structure,"row_stability":stability,"column_stability":alignment,"require_atomic_tokens":true,"diagnostics":d}),
        )?;
        *d = result["diagnostics"]
            .as_object()
            .cloned()
            .ok_or_else(invalid)?;
        let candidate = result["candidate"].clone();
        if candidate.is_null() {
            if d.get("first_rejection_gate").is_none_or(Value::is_null) {
                let gate = d
                    .get("candidate_rejection_gate")
                    .cloned()
                    .unwrap_or(json!("candidate_hard_gate"));
                d.insert("first_rejection_gate".into(), gate);
            }
            return Ok(candidate);
        }
        d.insert("first_rejection_gate".into(), Value::Null);
        d.insert("score".into(), candidate["score"].clone());
        Ok(candidate)
    }
}
pub fn generate(v: &Value) -> Result<Value, Error> {
    generate_with_work(v, &mut Work(0))
}
pub(in crate::processing) fn generate_with_work(
    v: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    validate_tree(v)?;
    let text = validate_text(&v["text"])?;
    let g = geometry(&v["table_input"])?;
    let physical = g.physical(&text);
    let mut candidates = Vec::new();
    let mut diagnostics = Vec::new();
    for source in ["sparse_grid", "text_grid", "key_value"] {
        if source == "sparse_grid" && physical <= 0. {
            continue;
        }
        let mut d = Map::new();
        let c = Build {
            text: &text,
            raw_text: &v["text"],
            g: &g,
            source,
            physical: if source == "sparse_grid" {
                physical
            } else {
                0.
            },
        }
        .run(&mut d, work)?;
        diagnostics.push(Value::Object(d));
        if !c.is_null() {
            candidates.push(c)
        }
    }
    let html = candidates
        .iter()
        .map(table_native_candidate::render)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({"candidates":candidates,"diagnostics":diagnostics,"html":html}))
}
