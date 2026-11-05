//! 旧sparse_multiline主分支与整行分组带回退；仅接受给定页面原语。
use super::{
    table_native_candidate, table_native_sparse as sparse,
    table_native_text::{Row, Text},
    table_native_vector::normalized,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
mod grouped;
pub(super) mod spatial;
type Diagnostic = Map<String, Value>;
#[derive(Serialize)]
struct Spec {
    row: usize,
    col: usize,
    rowspan: usize,
    colspan: usize,
    bbox: [f64; 4],
}
struct Logical {
    indices: Vec<usize>,
    top: f64,
    bottom: f64,
}
struct Body {
    rows: Vec<Logical>,
    key: usize,
    occupancy: Vec<BTreeSet<usize>>,
}
fn center(r: &Row) -> f64 {
    (r.bbox[1] + r.bbox[3]) / 2.
}
fn median(mut a: Vec<usize>) -> f64 {
    a.sort();
    let m = a.len() / 2;
    if a.len() % 2 == 1 {
        a[m] as f64
    } else {
        (a[m - 1] + a[m]) as f64 / 2.
    }
}
fn reject<T>(d: &mut Diagnostic, gate: &str) -> Option<T> {
    d.insert("first_rejection_gate".into(), json!(gate));
    None
}
fn internal(r: &sparse::Rule, w: f64, h: f64) -> bool {
    r.orientation == "horizontal"
        && r.end - r.start >= 0.90 * w
        && 0.01 * h < r.coordinate
        && r.coordinate < 0.99 * h
}
fn header_bottom(
    text: &Text,
    x: &[f64],
    rules: &[sparse::Rule],
    w: f64,
    h: f64,
    evidence: &str,
) -> Option<f64> {
    if text.rows.len() < 3 {
        return None;
    }
    let hint = text.rows[1..]
        .iter()
        .find(|r| r.tokens.len() == x.len() - 1)
        .unwrap_or(&text.rows[1]);
    let candidates: Vec<_> = rules
        .iter()
        .filter(|r| internal(r, w, h) && r.coordinate < center(hint))
        .map(|r| r.coordinate)
        .collect();
    let boundary = if let Some(b) = candidates.into_iter().max_by(f64::total_cmp) {
        b
    } else if ["filled_record", "keyed_record"].contains(&evidence) {
        return Some((center(&text.rows[0]) + center(&text.rows[1])) / 2.);
    } else {
        (center(&text.rows[hint.row_index as usize - 1]) + center(hint)) / 2.
    };
    (text.rows[0].bbox[3] - 0.5 <= boundary && boundary <= text.rows.last().unwrap().bbox[1] + 0.5)
        .then_some(boundary)
}
fn key_column(occupancy: &[BTreeSet<usize>], cols: usize) -> (usize, usize) {
    let stats: Vec<_> = (0..cols)
        .map(|c| {
            let (mut runs, mut count, mut previous) = (0, 0, false);
            for o in occupancy {
                let flag = o.contains(&c);
                if flag && !previous {
                    runs += 1;
                }
                count += usize::from(flag);
                previous = flag;
            }
            (c, runs, count)
        })
        .collect();
    let first = stats[0];
    if first.2 >= (0.80 * occupancy.len() as f64).ceil() as usize {
        return (first.0, first.1);
    }
    let repeated: Vec<_> = stats.iter().filter(|s| s.1 >= 3).copied().collect();
    let selected = if !repeated.is_empty() {
        let maximum = stats.iter().map(|s| s.2).max().unwrap();
        if let Some(s) = repeated
            .iter()
            .find(|s| s.2 as f64 >= 0.60 * maximum as f64)
        {
            *s
        } else {
            *repeated
                .iter()
                .max_by_key(|s| (s.1, s.2, std::cmp::Reverse(s.0)))
                .unwrap()
        }
    } else if first.2 >= 2 {
        first
    } else {
        *stats
            .iter()
            .max_by_key(|s| (s.2, std::cmp::Reverse(s.0)))
            .unwrap()
    };
    (selected.0, selected.1)
}
fn split(
    rows: &[&Row],
    key: usize,
    occupied: &BTreeMap<usize, BTreeSet<usize>>,
    short: bool,
    work: &mut Work,
) -> Result<Vec<Vec<usize>>, Error> {
    let mut anchors = Vec::new();
    let mut previous = None;
    for (i, r) in rows.iter().enumerate() {
        if occupied[&(r.row_index as usize)].contains(&key) {
            if !short || previous != Some(i.saturating_sub(1)) || i == 0 {
                anchors.push(i);
            }
            previous = Some(i);
        }
    }
    if anchors.is_empty() {
        return Ok(Vec::new());
    }
    let mut centered = BTreeMap::new();
    if !short {
        for (owner, a) in anchors.iter().enumerate() {
            let a = *a;
            if a == 0 || a + 1 >= rows.len() {
                continue;
            }
            let (leading, k, trailing) = (rows[a - 1], rows[a], rows[a + 1]);
            let l = &occupied[&(leading.row_index as usize)];
            if l.is_empty()
                || l != &occupied[&(trailing.row_index as usize)]
                || !l.is_disjoint(&occupied[&(k.row_index as usize)])
            {
                continue;
            }
            if (center(leading) + center(trailing) - 2. * center(k)).abs()
                <= 0.5 * (k.bbox[3] - k.bbox[1])
            {
                centered.insert(a - 1, owner);
            }
        }
    }
    work.charge(rows.len() * anchors.len())?;
    let mut groups = vec![Vec::new(); anchors.len()];
    for (i, row) in rows.iter().enumerate() {
        let owner = anchors.partition_point(|a| *a <= i).saturating_sub(1);
        groups[*centered.get(&i).unwrap_or(&owner)].push(row.row_index as usize);
    }
    Ok(groups.into_iter().filter(|g| !g.is_empty()).collect())
}
fn logical(
    text: &Text,
    x: &[f64],
    rules: &[sparse::Rule],
    geometry: (f64, f64, f64),
    evidence: &str,
    d: &mut Diagnostic,
    work: &mut Work,
) -> Result<Option<Body>, Error> {
    let (bottom, w, h) = geometry;
    let rows: Vec<_> = text.rows.iter().filter(|r| center(r) > bottom).collect();
    if rows.len() < 2 {
        return Ok(None);
    }
    let glyphs = text.glyphs.iter().map(|g| (g.glyph_id, g)).collect();
    let occ = rows
        .iter()
        .map(|r| sparse::occupancy(r, &glyphs, x, work))
        .collect::<Result<Vec<_>, _>>()?;
    let occupied: BTreeMap<_, _> = rows
        .iter()
        .zip(&occ)
        .map(|(r, o)| (r.row_index as usize, o.clone()))
        .collect();
    if evidence == "rule_band"
        && !rules.iter().any(|r| {
            internal(r, w, h)
                && bottom + 0.5 < r.coordinate
                && r.coordinate < center(rows.last().unwrap())
        })
    {
        let (mut seen, mut gap) = (false, false);
        for o in &occ {
            if o.contains(&0) {
                if seen && gap {
                    return Ok(reject(d, "ambiguous_body_rowspan"));
                }
                seen = true;
            } else if seen && *o != BTreeSet::from([x.len() - 2]) {
                gap = true;
            }
        }
    }
    let (key, runs) = key_column(&occ, x.len() - 1);
    let flags: Vec<_> = occ.iter().map(|o| o.contains(&key)).collect();
    let (mut run_lengths, mut gap_lengths) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < flags.len() {
        let mut j = i + 1;
        while j < flags.len() && flags[j] == flags[i] {
            j += 1;
        }
        if flags[i] {
            run_lengths.push(j - i)
        } else {
            gap_lengths.push(j - i)
        }
        i = j;
    }
    let short = evidence == "keyed_record"
        && runs >= 3
        && !run_lengths.is_empty()
        && median(run_lengths) <= 2.
        && !gap_lengths.is_empty()
        && median(gap_lengths) >= 2.;
    if evidence == "keyed_record" && !short {
        return Ok(reject(d, "record_key_support"));
    }
    let mut bands = vec![bottom, h];
    bands.extend(
        rules
            .iter()
            .filter(|r| internal(r, w, h) && r.coordinate > bottom + 0.5)
            .map(|r| r.coordinate),
    );
    bands.sort_by(f64::total_cmp);
    bands.dedup();
    work.charge(bands.len() * rows.len())?;
    let mut groups = Vec::new();
    for p in bands.windows(2).filter(|p| p[1] - p[0] > 0.5) {
        let band: Vec<_> = rows
            .iter()
            .filter(|r| p[0] <= center(r) && center(r) <= p[1])
            .copied()
            .collect();
        groups.extend(split(&band, key, &occupied, short, work)?);
    }
    if groups.len() < 2 {
        return Ok(None);
    }
    let extents: Vec<_> = groups
        .iter()
        .map(|g| {
            (
                g.iter()
                    .map(|i| text.rows[*i].bbox[1])
                    .fold(f64::INFINITY, f64::min),
                g.iter()
                    .map(|i| text.rows[*i].bbox[3])
                    .fold(f64::NEG_INFINITY, f64::max),
            )
        })
        .collect();
    let mut boundaries = vec![bottom];
    for p in extents.windows(2) {
        boundaries.push(if p[0].1 <= p[1].0 {
            (p[0].1 + p[1].0) / 2.
        } else {
            ((p[0].0 + p[0].1) / 2. + (p[1].0 + p[1].1) / 2.) / 2.
        });
    }
    boundaries.push(h);
    if boundaries.windows(2).any(|p| p[1] <= p[0]) {
        return Ok(None);
    }
    let rows: Vec<_> = groups
        .into_iter()
        .enumerate()
        .map(|(i, indices)| Logical {
            indices,
            top: boundaries[i],
            bottom: boundaries[i + 1],
        })
        .collect();
    let occupancy = rows
        .iter()
        .map(|r| {
            r.indices
                .iter()
                .flat_map(|i| occupied[i].iter().copied())
                .collect()
        })
        .collect();
    d.extend(json!({"key_col":key,"key_runs":runs,"group_short_key_runs":short,"logical_body_rows":rows.len(),"body_groups":rows.iter().map(|r|&r.indices).collect::<Vec<_>>()} ).as_object().unwrap().clone());
    Ok(Some(Body {
        rows,
        key,
        occupancy,
    }))
}
fn separator(text: &Text, rules: &[sparse::Rule], bottom: f64, w: f64) -> Option<f64> {
    let rows: Vec<_> = text.rows.iter().filter(|r| center(r) < bottom).collect();
    if rows.len() < 2 {
        return None;
    }
    let mut best: Option<&sparse::Rule> = None;
    for r in rules.iter().filter(|r| {
        r.orientation == "horizontal"
            && r.end - r.start >= 0.20 * w
            && rows[0].bbox[3] - 0.5 < r.coordinate
            && r.coordinate < bottom - 0.5
    }) {
        let key = (r.end - r.start, -(r.coordinate - bottom / 2.).abs());
        if best.is_none_or(|b| key > (b.end - b.start, -(b.coordinate - bottom / 2.).abs())) {
            best = Some(r);
        }
    }
    best.map(|r| r.coordinate)
}
fn header_specs(
    text: &Text,
    x: &[f64],
    bottom: f64,
    y: f64,
    rules: &[sparse::Rule],
    work: &mut Work,
) -> Result<Option<Vec<Spec>>, Error> {
    let cols = x.len() - 1;
    let tol = 1f64.max(0.25 * text.median_glyph_height);
    let mut absent = BTreeSet::new();
    let mut out = Vec::new();
    for c in 0..cols {
        work.charge(rules.len())?;
        let cov = rapiddoc_native::interval_coverage(
            rules
                .iter()
                .filter(|r| r.orientation == "horizontal" && (r.coordinate - y).abs() <= tol)
                .map(|r| (r.start, r.end))
                .collect(),
            x[c],
            x[c + 1],
        )?;
        if 0.20 < cov && cov < 0.80 {
            return Ok(None);
        }
        if cov <= 0.20 {
            absent.insert(c);
            out.push(Spec {
                row: 0,
                col: c,
                rowspan: 2,
                colspan: 1,
                bbox: [x[c], 0., x[c + 1], bottom],
            });
        }
    }
    let glyphs = text.glyphs.iter().map(|g| (g.glyph_id, g)).collect();
    let mut occupied = BTreeSet::new();
    for r in text
        .rows
        .iter()
        .filter(|r| y < center(r) && center(r) < bottom)
    {
        occupied.extend(sparse::occupancy(r, &glyphs, x, work)?);
    }
    let top: Vec<_> = text
        .rows
        .iter()
        .filter(|r| 0. <= center(r) && center(r) <= y)
        .flat_map(|r| r.tokens.iter())
        .map(|t| (t.bbox[0] + t.bbox[2]) / 2.)
        .collect();
    work.charge(cols * top.len())?;
    let present: BTreeSet<_> = (0..cols).filter(|c| !absent.contains(c)).collect();
    let mut groups = vec![Vec::new(); top.len()];
    for c in occupied.intersection(&present) {
        let center = (x[*c] + x[*c + 1]) / 2.;
        let mut best = None;
        for (i, t) in top.iter().enumerate() {
            let dist = (center - t).abs();
            if best.is_none_or(|(_, b)| dist < b) {
                best = Some((i, dist));
            }
        }
        let Some((i, _)) = best else {
            return Ok(None);
        };
        groups[i].push(*c);
    }
    let mut covered = BTreeSet::new();
    for (c, g) in top.iter().zip(groups) {
        if g.is_empty() {
            continue;
        }
        if g.windows(2).any(|p| p[1] != p[0] + 1) {
            return Ok(None);
        }
        let (a, b) = (g[0], *g.last().unwrap());
        if !(x[a] <= *c && *c <= x[b + 1]) {
            return Ok(None);
        }
        out.push(Spec {
            row: 0,
            col: a,
            rowspan: 1,
            colspan: b - a + 1,
            bbox: [x[a], 0., x[b + 1], y],
        });
        covered.extend(g);
    }
    for c in present.difference(&covered) {
        out.push(Spec {
            row: 0,
            col: *c,
            rowspan: 1,
            colspan: 1,
            bbox: [x[*c], 0., x[*c + 1], y],
        });
    }
    for c in present {
        out.push(Spec {
            row: 1,
            col: c,
            rowspan: 1,
            colspan: 1,
            bbox: [x[c], y, x[c + 1], bottom],
        });
    }
    Ok(Some(out))
}
fn body_specs(
    text: &Text,
    x: &[f64],
    body: &Body,
    offset: usize,
    evidence: &str,
    work: &mut Work,
) -> Result<Vec<Spec>, Error> {
    let mut spans = BTreeMap::new();
    if evidence == "filled_record" && body.key > 0 {
        work.charge(body.rows.iter().map(|r| r.indices.len()).sum::<usize>() * text.glyphs.len())?;
        let occupied: Vec<_> = body
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                text.glyphs.iter().any(|g| {
                    r.indices.contains(&(g.visual_row as usize))
                        && x[0] <= (g.bbox[0] + g.bbox[2]) / 2.
                        && (g.bbox[0] + g.bbox[2]) / 2. <= x[1]
                })
            })
            .map(|(i, _)| i)
            .collect();
        for (i, start) in occupied.iter().enumerate() {
            spans.insert(
                *start,
                occupied.get(i + 1).copied().unwrap_or(body.rows.len()) - start,
            );
        }
    }
    let mut covered = BTreeSet::new();
    let mut out = Vec::new();
    for (i, r) in body.rows.iter().enumerate() {
        if let Some(n) = spans.get(&i) {
            out.push(Spec {
                row: offset + i,
                col: 0,
                rowspan: *n,
                colspan: 1,
                bbox: [x[0], r.top, x[1], body.rows[i + n - 1].bottom],
            });
            covered.extend(i..i + n);
        } else if !covered.contains(&i) {
            out.push(Spec {
                row: offset + i,
                col: 0,
                rowspan: 1,
                colspan: 1,
                bbox: [x[0], r.top, x[1], r.bottom],
            });
        }
        for c in 1..x.len() - 1 {
            out.push(Spec {
                row: offset + i,
                col: c,
                rowspan: 1,
                colspan: 1,
                bbox: [x[c], r.top, x[c + 1], r.bottom],
            });
        }
    }
    Ok(out)
}
fn primary(
    v: &Value,
    text: &Text,
    d: &mut Diagnostic,
    work: &mut Work,
) -> Result<Option<Value>, Error> {
    let input = &v["table_input"];
    let Some(b) = normalized(&input["table_bbox"])? else {
        return Ok(None);
    };
    let (_, _, a) = sparse::dimensions(input)?;
    let (w, h) = if [90, 270].contains(&a) {
        (b[3] - b[1], b[2] - b[0])
    } else {
        (b[2] - b[0], b[3] - b[1])
    };
    let rules = sparse::local_rules(input, w, h)?;
    let rects = sparse::rectangles(input, w, h)?;
    let Some(hyp) = sparse::hypothesis(text, w, &rules, &rects, d, work)? else {
        return Ok(None);
    };
    let x = &hyp.x_tracks;
    let Some(bottom) = header_bottom(text, x, &rules, w, h, &hyp.evidence) else {
        return Ok(reject(d, "header_boundary"));
    };
    if sparse::overlapping(text, bottom) {
        return Ok(reject(d, "overlapping_formula_rows"));
    }
    let Some(body) = logical(text, x, &rules, (bottom, w, h), &hyp.evidence, d, work)? else {
        if d.get("first_rejection_gate").is_none_or(Value::is_null) {
            reject::<Value>(d, "logical_rows");
        }
        return Ok(None);
    };
    if hyp.evidence != "filled_record" {
        let (mut seen, mut gap) = (false, false);
        for o in &body.occupancy {
            if o.contains(&0) {
                if seen && gap {
                    return Ok(reject(d, "ambiguous_body_rowspan"));
                }
                seen = true;
            } else if seen {
                gap = true;
            }
        }
    }
    if !sparse::gutters(text, x, &hyp.physical_boundaries, bottom, d, work)? {
        return Ok(reject(d, "gutter_support"));
    }
    let sep = separator(text, &rules, bottom, w);
    let header_rows = if sep.is_some() { 2 } else { 1 };
    if (header_rows + body.rows.len()) * (x.len() - 1) > rapiddoc_core::resources::limit(2000) {
        return Err(limit());
    }
    let mut specs = if let Some(y) = sep {
        let Some(s) = header_specs(text, x, bottom, y, &rules, work)? else {
            return Ok(reject(d, "header_topology"));
        };
        s
    } else {
        (0..x.len() - 1)
            .map(|c| Spec {
                row: 0,
                col: c,
                rowspan: 1,
                colspan: 1,
                bbox: [x[c], 0., x[c + 1], bottom],
            })
            .collect()
    };
    specs.extend(body_specs(
        text,
        x,
        &body,
        header_rows,
        &hyp.evidence,
        work,
    )?);
    work.charge(text.glyphs.len() * specs.len() * 3 + text.glyphs.len() * text.glyphs.len() * 2)?;
    let output = table_native_candidate::build(
        &json!({"source":"sparse_multiline","rows":header_rows+body.rows.len(),"cols":x.len()-1,"specs":specs,"text":v["text"],"structure_support":1.,"row_stability":1.,"column_stability":1.,"use_grid_index":true,"issues":[format!("evidence={}",hyp.evidence),format!("header_rows={header_rows}"),format!("logical_body_rows={}",body.rows.len()),format!("key_col={}",body.key),format!("filled_band_count={}",hyp.filled_band_count)]}),
    )?;
    let c = &output["candidate"];
    let record = &output["diagnostics"];
    if c.is_null() {
        d.insert(
            "first_rejection_gate".into(),
            record
                .get("candidate_rejection_gate")
                .cloned()
                .unwrap_or(json!("candidate_hard_gate")),
        );
        return Ok(None);
    }
    d.insert("candidate_diagnostics".into(), record.clone());
    d.insert("candidate_components".into(),json!({"text_capture":c["text_capture"],"order_consistency":c["order_consistency"],"score":c["score"],"ambiguous_glyph_ratio":record.get("ambiguous_glyph_ratio").unwrap_or(&json!(1.))}));
    if c["text_capture"].as_f64().unwrap() < 1.
        || c["order_consistency"].as_f64().unwrap() < 1.
        || c["score"].as_f64().unwrap() < 0.98
        || record["ambiguous_glyph_ratio"].as_f64().unwrap_or(1.) > 0.
    {
        return Ok(reject(d, "verified_integrity"));
    }
    if hyp.evidence == "keyed_record" && body.rows.len() < 3 {
        return Ok(reject(d, "record_count"));
    }
    d.extend(json!({"first_rejection_gate":null,"grid":{"rows":c["rows"],"cols":c["cols"]},"header_bottom":bottom,"header_rows":header_rows,"score":c["score"],"token_split_count":record.get("token_split_count").unwrap_or(&json!(0))}).as_object().unwrap().clone());
    Ok(Some(c.clone()))
}
pub fn generate(v: &Value) -> Result<Value, Error> {
    generate_with_work(v, &mut Work(0))
}
pub(in crate::processing) fn generate_with_work(
    v: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    validate_tree(v)?;
    let text = sparse::text(&v["text"])?;
    let mut d = json!({"source":"sparse_multiline"})
        .as_object()
        .unwrap()
        .clone();
    let mut c = primary(v, &text, &mut d, work)?;
    let mut diagnostics = vec![Value::Object(d)];
    if c.is_none() {
        let mut d = json!({"source":"sparse_multiline"})
            .as_object()
            .unwrap()
            .clone();
        c = grouped::build(v, &text, &mut d, work)?;
        diagnostics.push(Value::Object(d));
    }
    let candidates: Vec<_> = c.into_iter().collect();
    let html = candidates
        .iter()
        .map(table_native_candidate::render)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({"candidates":candidates,"diagnostics":diagnostics,"html":html}))
}

pub fn project_text(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let chars = v["chars"].as_array().ok_or_else(invalid)?;
    if chars.len() > rapiddoc_core::resources::limit(20000) {
        return Err(limit());
    }
    let a = super::table_native_text::angle(&v["angle"])?;
    let chars = chars.iter().collect::<Vec<_>>();
    Ok(json!(spatial::project(
        &chars,
        &v["table_bbox"],
        a as i64,
        &mut Work(0)
    )?))
}
