//! 旧sparse_hybrid：物理线/矩形与文字对齐融合，给定页面原语，不读取PDF。
use super::{
    table_native_candidate, table_native_sparse as sparse,
    table_native_text::Text,
    table_native_vector::{local, normalized},
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

struct Layout {
    cols: usize,
    start: usize,
    run: Vec<usize>,
}
struct Hypothesis {
    evidence: &'static str,
    x: Vec<f64>,
    physical: BTreeSet<usize>,
    start: usize,
    reliability: f64,
}
struct Support {
    minimum: f64,
    overall: f64,
    occupancy: Vec<BTreeSet<usize>>,
}
#[derive(Serialize)]
struct Spec {
    row: usize,
    col: usize,
    rowspan: usize,
    colspan: usize,
    bbox: [f64; 4],
}
fn unique(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values.dedup();
    values
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = values.len() / 2;
    if values.len() % 2 == 1 {
        values[m]
    } else {
        (values[m - 1] + values[m]) / 2.
    }
}
fn coverage(intervals: Vec<(f64, f64)>, start: f64, end: f64) -> Result<f64, Error> {
    if end <= start {
        Ok(0.)
    } else {
        rapiddoc_native::interval_coverage(intervals, start, end)
    }
}
type VerticalEvidence = (Vec<f64>, Vec<(f64, f64)>);
fn vertical(
    rules: &[sparse::Rule],
    width: f64,
    height: f64,
    tolerance: f64,
    work: &mut Work,
) -> Result<VerticalEvidence, Error> {
    let rules: Vec<_> = rules
        .iter()
        .filter(|r| r.orientation == "vertical")
        .collect();
    let clusters = sparse::clusters(
        rules.iter().map(|r| r.coordinate).collect(),
        tolerance,
        work,
    )?;
    let mut positions = vec![0., width];
    let mut coverages: Vec<(f64, f64)> = Vec::new();
    for (coordinate, _) in clusters {
        work.charge(rules.len())?;
        let covered = coverage(
            rules
                .iter()
                .filter(|r| (r.coordinate - coordinate).abs() <= tolerance)
                .map(|r| (r.start, r.end))
                .collect(),
            0.,
            height,
        )?;
        if covered < 0.75 {
            continue;
        }
        let snapped = if coordinate <= tolerance {
            0.
        } else if width - coordinate <= tolerance {
            width
        } else {
            coordinate
        };
        positions.push(snapped);
        if let Some((_, c)) = coverages.iter_mut().find(|(x, _)| *x == snapped) {
            *c = c.max(covered)
        } else {
            coverages.push((snapped, covered))
        }
    }
    Ok((unique(positions), coverages))
}
fn rectangle_edges(
    input: &Value,
    text: &Text,
    width: f64,
    height: f64,
    tolerance: f64,
    work: &mut Work,
) -> Result<Vec<f64>, Error> {
    let Some(table) = normalized(&input["table_bbox"])? else {
        return Ok(Vec::new());
    };
    let (_, _, angle) = sparse::dimensions(input)?;
    let rects = input.get("rectangles").map_or(Ok(&[][..]), |r| {
        r.as_array().map(|a| a.as_slice()).ok_or_else(invalid)
    })?;
    if rects.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let mut edges = Vec::new();
    for r in rects {
        let count = r["segment_count"].as_i64().ok_or_else(invalid)?;
        let fill = r["fill_visible"].as_bool().ok_or_else(invalid)?;
        let stroke = r["stroke_visible"].as_bool().ok_or_else(invalid)?;
        if count != 5 || !(fill || stroke) {
            continue;
        }
        let Some(b) = normalized(&r["bbox"])?.and_then(|b| local(b, table, angle, 0.)) else {
            continue;
        };
        let (w, h) = (b[2] - b[0], b[3] - b[1]);
        let thin =
            h <= 1.5f64.max(0.4 * text.median_glyph_height) && w >= 2. * text.median_glyph_width;
        let header = b[1] <= (0.30 * height).min(3. * text.median_glyph_height)
            && h <= 3. * text.median_glyph_height
            && w >= 2. * text.median_glyph_width;
        if thin || header {
            edges.extend([b[0].max(0.), b[2].min(width)]);
        }
    }
    let mut out = vec![0., width];
    for (coordinate, members) in sparse::clusters(edges, tolerance, work)? {
        if members.len() >= 2 || coordinate <= tolerance || width - coordinate <= tolerance {
            out.push(coordinate);
        }
    }
    Ok(unique(out))
}
fn dense(text: &Text) -> Option<Layout> {
    let mut counts = BTreeMap::new();
    for row in &text.rows {
        if row.tokens.len() >= 2 {
            *counts.entry(row.tokens.len()).or_insert(0) += 1;
        }
    }
    let mut best = None;
    for (cols, count) in counts {
        let mut run = Vec::new();
        let mut longest = Vec::new();
        for row in text.rows.iter().filter(|r| r.tokens.len() == cols) {
            let i = row.row_index as usize;
            if run.last().is_some_and(|last| *last + 1 != i) {
                run.clear();
            }
            run.push(i);
            if run.len() >= longest.len() {
                longest = run.clone();
            }
        }
        if longest.len() < 2 {
            continue;
        }
        let key = (longest.len(), count, cols, longest.clone());
        if best.as_ref().is_none_or(|b| key > *b) {
            best = Some(key);
        }
    }
    best.map(|(_, _, cols, run)| Layout {
        cols,
        start: run[0],
        run,
    })
}
fn support(
    text: &Text,
    x: &[f64],
    start: usize,
    work: &mut Work,
) -> Result<Option<Support>, Error> {
    let cols = x.len().saturating_sub(1);
    let rows = &text.rows[start..];
    if rows.len() < 2 || cols < 2 {
        return Ok(None);
    }
    let glyphs = text.glyphs.iter().map(|g| (g.glyph_id, g)).collect();
    let occupied = rows
        .iter()
        .map(|r| sparse::occupancy(r, &glyphs, x, work))
        .collect::<Result<Vec<_>, _>>()?;
    let minimum_cols = 2usize.max((0.60 * cols as f64).ceil() as usize);
    if occupied.iter().any(|o| o.len() < minimum_cols) {
        return Ok(None);
    }
    // key_col的最频首列必然满足以下逐列minimum门；保留原first-seen平局顺序。
    let mut counts: Vec<(usize, usize)> = Vec::new();
    for o in &occupied {
        let key = *o.first().unwrap();
        if let Some((_, n)) = counts.iter_mut().find(|(c, _)| *c == key) {
            *n += 1;
        } else {
            counts.push((key, 1));
        }
    }
    let mut key = counts[0];
    for entry in counts {
        if entry.1 > key.1 {
            key = entry;
        }
    }
    if occupied.iter().filter(|o| o.contains(&key.0)).count() as f64 / (occupied.len() as f64)
        < 0.60
    {
        return Ok(None);
    }
    work.charge(cols * occupied.len())?;
    let supports: Vec<_> = (0..cols)
        .map(|c| occupied.iter().filter(|o| o.contains(&c)).count() as f64 / occupied.len() as f64)
        .collect();
    let minimum = supports.iter().copied().fold(f64::INFINITY, f64::min);
    if minimum < 0.60 {
        return Ok(None);
    }
    // Python statistics.mean按float精确比率求和后一次舍入，支持域在[.6,1]。
    let overall = rapiddoc_native::table_decisiveness_mean(&supports)?;
    if overall < 0.80 {
        return Ok(None);
    }
    Ok(Some(Support {
        minimum,
        overall,
        occupancy: occupied,
    }))
}
fn hypotheses(
    input: &Value,
    text: &Text,
    rules: &[sparse::Rule],
    w: f64,
    h: f64,
    work: &mut Work,
) -> Result<Vec<Hypothesis>, Error> {
    work.charge(text.rows.len() * text.rows.len())?;
    let Some(layout) = dense(text) else {
        return Ok(Vec::new());
    };
    if layout.cols > rapiddoc_core::resources::limit(200) {
        return Err(limit());
    }
    let tol = 1f64.max(0.25 * text.median_glyph_height);
    let (vertical, coverages) = vertical(rules, w, h, tol, work)?;
    let rects = rectangle_edges(input, text, w, h, tol, work)?;
    let physical = unique(vertical.iter().chain(&rects).copied().collect());
    let rects = unique(
        rects
            .into_iter()
            .map(|p| {
                if p <= tol {
                    0.
                } else if w - p <= tol {
                    w
                } else {
                    p
                }
            })
            .collect(),
    );
    let mut tx = vec![0.];
    let mut valid = true;
    for col in 0..layout.cols - 1 {
        work.charge(layout.run.len())?;
        let b = median(
            layout
                .run
                .iter()
                .map(|i| {
                    let r = &text.rows[*i];
                    (r.tokens[col].bbox[2] + r.tokens[col + 1].bbox[0]) / 2.
                })
                .collect(),
        );
        valid &= 0. < b && b < w;
        tx.push(b);
    }
    tx.push(w);
    valid &= tx.windows(2).all(|p| p[0] < p[1]);
    let mut out = Vec::new();
    let cols = vertical.len() - 1;
    if cols >= 2
        && (cols == layout.cols || cols == layout.cols + 1)
        && support(text, &vertical, layout.start, work)?.is_some()
    {
        let reliability = if cols == layout.cols {
            1.
        } else {
            coverages
                .iter()
                .filter(|(p, _)| tol < *p && *p < w - tol)
                .map(|(_, v)| *v)
                .fold(1., f64::min)
        };
        out.push(Hypothesis {
            evidence: "vertical_text",
            physical: (1..vertical.len() - 1).collect(),
            x: vertical,
            start: layout.start,
            reliability: reliability.min(1.),
        });
    }
    if rects.len().saturating_sub(1) == layout.cols
        && layout.cols >= 2
        && support(text, &rects, layout.start, work)?.is_some()
    {
        out.push(Hypothesis {
            evidence: "rect_text",
            physical: (1..rects.len() - 1).collect(),
            x: rects,
            start: layout.start,
            reliability: 1.,
        });
    }
    if valid && out.is_empty() {
        let physical = (1..tx.len() - 1)
            .filter(|i| physical.iter().any(|p| (tx[*i] - p).abs() <= tol))
            .collect();
        out.push(Hypothesis {
            evidence: "text_network",
            physical,
            x: tx,
            start: layout.start,
            reliability: 1.,
        });
    }
    let mut dedup: Vec<(Vec<i64>, Hypothesis)> = Vec::new();
    for hyp in out.into_iter().take(8) {
        let signature: Vec<_> = hyp
            .x
            .iter()
            .map(|x| (x / tol.max(0.1)).round_ties_even() as i64)
            .collect();
        if let Some((_, existing)) = dedup.iter_mut().find(|(key, _)| *key == signature) {
            if hyp.reliability > existing.reliability {
                *existing = hyp;
            }
        } else {
            dedup.push((signature, hyp));
        }
    }
    Ok(dedup.into_iter().map(|(_, h)| h).collect())
}
fn y_tracks(
    text: &Text,
    rules: &[sparse::Rule],
    height: f64,
    work: &mut Work,
) -> Result<Option<Vec<f64>>, Error> {
    if text.rows.len() < 2 {
        return Ok(None);
    }
    let mut tracks = vec![0.];
    for pair in text.rows.windows(2) {
        work.charge(rules.len())?;
        let mut best: Option<&sparse::Rule> = None;
        for r in rules.iter().filter(|r| {
            r.orientation == "horizontal"
                && pair[0].bbox[3] - 0.5 <= r.coordinate
                && r.coordinate <= pair[1].bbox[1] + 0.5
        }) {
            if best.is_none_or(|b| r.end - r.start > b.end - b.start) {
                best = Some(r);
            }
        }
        tracks.push(best.map_or(
            ((pair[0].bbox[1] + pair[0].bbox[3]) / 2. + (pair[1].bbox[1] + pair[1].bbox[3]) / 2.)
                / 2.,
            |r| r.coordinate,
        ));
    }
    tracks.push(height);
    Ok(tracks.windows(2).all(|p| p[0] < p[1]).then_some(tracks))
}
fn spec(row: usize, col: usize, rowspan: usize, colspan: usize, x: &[f64], y: &[f64]) -> Spec {
    Spec {
        row,
        col,
        rowspan,
        colspan,
        bbox: [x[col], y[row], x[col + colspan], y[row + rowspan]],
    }
}
fn specs(
    text: &Text,
    hyp: &Hypothesis,
    y: &[f64],
    rules: &[sparse::Rule],
    work: &mut Work,
) -> Result<Option<Vec<Spec>>, Error> {
    let x = &hyp.x;
    let cols = x.len() - 1;
    if hyp.start > 2 {
        return Ok(None);
    }
    if text.rows.len() * cols > rapiddoc_core::resources::limit(2000) {
        return Err(limit());
    }
    let mut out = Vec::new();
    if hyp.start == 2 {
        let tol = 1f64.max(0.25 * text.median_glyph_height);
        let mut absent = BTreeSet::new();
        for col in 0..cols {
            work.charge(rules.len())?;
            let c = coverage(
                rules
                    .iter()
                    .filter(|r| r.orientation == "horizontal" && (r.coordinate - y[1]).abs() <= tol)
                    .map(|r| (r.start, r.end))
                    .collect(),
                x[col],
                x[col + 1],
            )?;
            if 0.20 < c && c < 0.80 {
                return Ok(None);
            }
            if c <= 0.20 {
                absent.insert(col);
                out.push(spec(0, col, 2, 1, x, y));
            }
        }
        let present: BTreeSet<_> = (0..cols).filter(|c| !absent.contains(c)).collect();
        let glyphs = text.glyphs.iter().map(|g| (g.glyph_id, g)).collect();
        let occupied = sparse::occupancy(&text.rows[1], &glyphs, x, work)?;
        // 原zip会先丢弃无法落列的中心，再按索引配对；不能改为filter_map token。
        let token_cols: Vec<_> = text.rows[0]
            .tokens
            .iter()
            .filter_map(|t| {
                let c = (t.bbox[0] + t.bbox[2]) / 2.;
                x.windows(2).position(|p| p[0] <= c && c <= p[1])
            })
            .collect();
        let top: Vec<_> = text.rows[0]
            .tokens
            .iter()
            .zip(token_cols)
            .filter(|(_, c)| present.contains(c))
            .map(|(t, _)| t)
            .collect();
        let mut assigned = vec![Vec::new(); top.len()];
        work.charge(occupied.len() * top.len())?;
        for c in occupied.intersection(&present) {
            let center = (x[*c] + x[*c + 1]) / 2.;
            let mut best = None;
            for (i, t) in top.iter().enumerate() {
                let d = (center - (t.bbox[0] + t.bbox[2]) / 2.).abs();
                if best.is_none_or(|(_, b)| d < b) {
                    best = Some((i, d));
                }
            }
            let Some((owner, _)) = best else {
                return Ok(None);
            };
            assigned[owner].push(*c);
        }
        let mut covered = BTreeSet::new();
        for (t, group) in top.iter().zip(&assigned) {
            if group.is_empty() || group.windows(2).any(|p| p[1] != p[0] + 1) {
                return Ok(None);
            }
            let (first, last) = (group[0], *group.last().unwrap());
            let center = (t.bbox[0] + t.bbox[2]) / 2.;
            if !(x[first] <= center && center <= x[last + 1]) {
                return Ok(None);
            }
            out.push(spec(0, first, 1, last - first + 1, x, y));
            covered.extend(group.iter().copied());
        }
        for c in present.difference(&covered) {
            out.push(spec(0, *c, 1, 1, x, y));
        }
        for c in present {
            out.push(spec(1, c, 1, 1, x, y));
        }
    } else {
        for row in 0..hyp.start {
            for col in 0..cols {
                out.push(spec(row, col, 1, 1, x, y));
            }
        }
    }
    for row in hyp.start..text.rows.len() {
        for col in 0..cols {
            out.push(spec(row, col, 1, 1, x, y));
        }
    }
    Ok(Some(out))
}
fn splits(
    text: &Text,
    hyp: &Hypothesis,
    specs: &[Spec],
    work: &mut Work,
) -> Result<Option<usize>, Error> {
    let cols = hyp.x.len() - 1;
    let mut owners = vec![vec![usize::MAX; cols]; text.rows.len()];
    for (i, s) in specs.iter().enumerate() {
        for row in owners.iter_mut().skip(s.row).take(s.rowspan) {
            for col in row.iter_mut().skip(s.col).take(s.colspan) {
                *col = i;
            }
        }
    }
    let glyphs: BTreeMap<_, _> = text.glyphs.iter().map(|g| (g.glyph_id, g)).collect();
    let margin = 0.25f64.max(0.05 * text.median_glyph_width);
    let mut justified = 0;
    for row in &text.rows {
        for t in &row.tokens {
            work.charge(cols * (t.glyph_ids.len() + 1))?;
            for b in 1..cols {
                if owners[row.row_index as usize][b - 1] == owners[row.row_index as usize][b] {
                    continue;
                }
                let boundary = hyp.x[b];
                if !(t.bbox[0] + margin < boundary && boundary < t.bbox[2] - margin) {
                    continue;
                }
                if !hyp.physical.contains(&b) {
                    return Ok(None);
                }
                for id in &t.glyph_ids {
                    let g = glyphs.get(id).ok_or_else(invalid)?;
                    if g.bbox[0] + margin < boundary && boundary < g.bbox[2] - margin {
                        return Ok(None);
                    }
                }
                justified += 1;
            }
        }
    }
    Ok(Some(justified))
}
fn reject(d: &mut Map<String, Value>, gate: Value) -> Option<Value> {
    d.insert("first_rejection_gate".into(), gate);
    None
}
fn candidate(
    v: &Value,
    text: &Text,
    rules: &[sparse::Rule],
    hyp: &Hypothesis,
    height: f64,
    d: &mut Map<String, Value>,
    work: &mut Work,
) -> Result<Option<Value>, Error> {
    let Some(y) = y_tracks(text, rules, height, work)? else {
        return Ok(reject(d, json!("row_tracks")));
    };
    let Some(support) = support(text, &hyp.x, hyp.start, work)? else {
        return Ok(reject(d, json!("anchor_support")));
    };
    let Some(specs) = specs(text, hyp, &y, rules, work)? else {
        return Ok(reject(d, json!("header_topology")));
    };
    let Some(justified) = splits(text, hyp, &specs, work)? else {
        return Ok(reject(d, json!("token_split")));
    };
    work.charge(text.glyphs.len() * specs.len() * 3 + text.glyphs.len() * text.glyphs.len() * 2)?;
    let output = table_native_candidate::build(
        &json!({"source":"sparse_hybrid","rows":y.len()-1,"cols":hyp.x.len()-1,"specs":specs,"text":v["text"],"structure_support":hyp.reliability,"row_stability":1.,"column_stability":support.overall.min(1.),"use_grid_index":true,
        "issues":[format!("evidence={}",hyp.evidence),format!("body_start={}",hyp.start),format!("minimum_column_support={:.4}",support.minimum),format!("overall_anchor_support={:.4}",support.overall),format!("physical_boundaries={}",hyp.physical.len()),format!("physically_justified_token_splits={justified}")]}),
    )?;
    let c = &output["candidate"];
    if c.is_null() {
        return Ok(reject(
            d,
            output["diagnostics"]
                .get("candidate_rejection_gate")
                .cloned()
                .unwrap_or(json!("candidate_hard_gate")),
        ));
    }
    if c["text_capture"].as_f64().unwrap() < 1.
        || c["order_consistency"].as_f64().unwrap() < 1.
        || c["score"].as_f64().unwrap() < 0.98
        || output["diagnostics"]["ambiguous_glyph_ratio"]
            .as_f64()
            .unwrap_or(1.)
            > 0.
    {
        return Ok(reject(d, json!("verified_integrity")));
    }
    d.extend(json!({"first_rejection_gate":null,"grid":{"rows":c["rows"],"cols":c["cols"]},"body_start":hyp.start,"x_tracks":hyp.x,"physical_boundaries":hyp.physical,"minimum_column_support":support.minimum,"overall_anchor_support":support.overall,"body_occupancies":support.occupancy,"physically_justified_token_splits":justified,"score":c["score"]}).as_object().unwrap().clone());
    Ok(Some(c.clone()))
}
fn result(candidates: Vec<Value>, diagnostics: Vec<Value>) -> Result<Value, Error> {
    let html = candidates
        .iter()
        .map(table_native_candidate::render)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({"candidates":candidates,"diagnostics":diagnostics,"html":html}))
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
    let input = &v["table_input"];
    let Some(b) = normalized(&input["table_bbox"])? else {
        return result(Vec::new(), Vec::new());
    };
    let (_, _, angle) = sparse::dimensions(input)?;
    let (w, h) = if [90, 270].contains(&angle) {
        (b[3] - b[1], b[2] - b[0])
    } else {
        (b[2] - b[0], b[3] - b[1])
    };
    // Resource bound applies even when early physical evidence would reject.
    let rects = input.get("rectangles").map_or(Ok(&[][..]), |r| {
        r.as_array().map(|r| r.as_slice()).ok_or_else(invalid)
    })?;
    if rects.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let rules = sparse::local_rules(input, w, h)?;
    let count = rules
        .iter()
        .filter(|r| r.orientation == "horizontal" && r.end - r.start >= 0.50 * w)
        .count();
    let (tracks, _) = vertical(
        &rules,
        w,
        h,
        1f64.max(0.25 * text.median_glyph_height),
        work,
    )?;
    if count < 2 && tracks.len() < 3 {
        return result(
            Vec::new(),
            vec![
                json!({"source":"sparse_hybrid","first_rejection_gate":"physical_sparse_evidence","long_horizontal_rules":count,"vertical_tracks":tracks.len()}),
            ],
        );
    }
    let hypotheses = hypotheses(input, &text, &rules, w, h, work)?;
    let mut candidates = Vec::new();
    let mut diagnostics = Vec::new();
    for hyp in hypotheses {
        let mut d =
            json!({"source":"sparse_hybrid","evidence":hyp.evidence,"long_horizontal_rules":count})
                .as_object()
                .unwrap()
                .clone();
        if let Some(c) = candidate(v, &text, &rules, &hyp, h, &mut d, work)? {
            candidates.push(c);
        }
        diagnostics.push(Value::Object(d));
    }
    let mut topologies = BTreeSet::new();
    for c in &candidates {
        let mut spans: Vec<_> = c["cells"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                (
                    s["row"].as_u64().unwrap(),
                    s["col"].as_u64().unwrap(),
                    s["rowspan"].as_u64().unwrap(),
                    s["colspan"].as_u64().unwrap(),
                )
            })
            .collect();
        spans.sort();
        topologies.insert((
            c["rows"].as_u64().unwrap(),
            c["cols"].as_u64().unwrap(),
            spans,
        ));
    }
    if topologies.len() > 1 {
        diagnostics.push(json!({"source":"sparse_hybrid","first_rejection_gate":"topology_ambiguity","topology_count":topologies.len()}));
        candidates.clear();
    }
    candidates.truncate(1);
    result(candidates, diagnostics)
}
