//! 旧banded_numeric：重复数值行、公共字形间隙和物理表头；给定原语，不读取PDF。
use super::{
    table_native_candidate,
    table_native_multiline::spatial,
    table_native_sparse as sparse,
    table_native_text::{self, Text},
    table_native_vector::{local, number, python_set},
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
type Diagnostic = Map<String, Value>;
#[derive(Serialize)]
struct Spec {
    row: usize,
    col: usize,
    rowspan: usize,
    colspan: usize,
    bbox: [f64; 4],
}
pub fn numeric_value(s: &str) -> bool {
    let s = s.trim_matches(python_space);
    if ["O", "✓", "✔", "✗", "✘", "×"].contains(&s) {
        return true;
    }
    let mut cs = s.chars().peekable();
    if cs
        .peek()
        .is_some_and(|c| ['<', '>', '≤', '≥', '+', '-'].contains(c))
    {
        cs.next();
    }
    if !cs.next().is_some_and(|c| python_decimal(c).is_some()) {
        return false;
    }
    let mut rest: Vec<_> = cs.collect();
    if rest
        .last()
        .is_some_and(|c| ['%', 'k', 'K', 'm', 'M', 'b', 'B'].contains(c))
    {
        rest.pop();
    }
    rest.iter().all(|c| {
        python_decimal(*c).is_some()
            || python_space(*c)
            || ['.', ',', '–', '—', '/', '+', '-'].contains(c)
    })
}
fn median(mut a: Vec<f64>) -> f64 {
    a.sort_by(f64::total_cmp);
    let m = a.len() / 2;
    if a.len() % 2 == 1 {
        a[m]
    } else {
        (a[m - 1] + a[m]) / 2.
    }
}
fn bbox(v: &Value) -> Result<[f64; 4], Error> {
    let b = v.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
    Ok([
        number(&b[0])?,
        number(&b[1])?,
        number(&b[2])?,
        number(&b[3])?,
    ])
}
fn gutter(
    text: &Text,
    indices: &BTreeSet<i64>,
    lower: f64,
    upper: f64,
    work: &mut Work,
) -> Result<Option<f64>, Error> {
    work.charge(text.glyphs.len())?;
    let mut intervals: Vec<_> = text
        .glyphs
        .iter()
        .filter(|g| indices.contains(&g.visual_row) && g.bbox[2] > lower && g.bbox[0] < upper)
        .map(|g| (g.bbox[0].max(lower), g.bbox[2].min(upper)))
        .collect();
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut gaps = Vec::new();
    let mut cursor = lower;
    for (start, end) in intervals {
        if start > cursor {
            gaps.push((cursor, start));
        }
        cursor = cursor.max(end);
    }
    if cursor < upper {
        gaps.push((cursor, upper));
    }
    let mut best = None;
    for (a, b) in gaps {
        if best.is_none_or(|(l, r)| b - a > r - l) {
            best = Some((a, b));
        }
    }
    Ok(best.and_then(|(a, b)| {
        (b - a >= 0.6f64.max(0.08 * text.median_glyph_height)).then_some((a + b) / 2.)
    }))
}
fn filled_tracks(
    input: &Value,
    text: &Text,
    cols: usize,
    width: f64,
    work: &mut Work,
) -> Result<Option<Vec<f64>>, Error> {
    let table = bbox(&input["table_bbox"])?;
    let (_, _, angle) = sparse::dimensions(input)?;
    let rects = input.get("rectangles").map_or(Ok(&[][..]), |a| {
        a.as_array().map(|a| a.as_slice()).ok_or_else(invalid)
    })?;
    if rects.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    work.charge(rects.len())?;
    let mut boxes = Vec::new();
    for r in rects {
        let fill = r["fill_visible"].as_bool().ok_or_else(invalid)?;
        let count = r["segment_count"].as_i64().ok_or_else(invalid)?;
        if !fill || count != 5 {
            continue;
        }
        let Some(b) = local(bbox(&r["bbox"])?, table, angle, 0.) else {
            continue;
        };
        if b[1] <= text.rows[0].bbox[1]
            && b[3] >= text.rows[0].bbox[3]
            && b[3] - b[1] > text.median_glyph_height
        {
            boxes.push(b);
        }
    }
    boxes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if boxes.len() != cols
        || boxes.windows(2).any(|p| {
            (p[0][2] - p[1][0]).abs() > 1.
                || (p[0][1] - p[1][1]).abs() > 1.
                || (p[0][3] - p[1][3]).abs() > 1.
        })
    {
        return Ok(None);
    }
    let mut tracks = vec![0.];
    tracks.extend(boxes.iter().take(cols - 1).map(|b| b[2]));
    tracks.push(width);
    if text.rows[0].tokens.len() > cols {
        return Err(invalid());
    }
    if text.rows[0].tokens.iter().enumerate().any(|(c, t)| {
        !(tracks[c] <= t.bbox[0] && t.bbox[0] <= t.bbox[2] && t.bbox[2] <= tracks[c + 1])
    }) {
        return Ok(None);
    }
    Ok(Some(tracks))
}
fn reject<T>(d: &mut Diagnostic, gate: &str) -> Option<T> {
    d.insert("first_rejection_gate".into(), json!(gate));
    None
}
fn strict_index(c: &Value, fallback: usize) -> Result<i64, Error> {
    let Some(v) = c.get("char_idx") else {
        return Ok(fallback as i64);
    };
    let id = if let Some(s) = v.as_str() {
        table_native_text::numeric_string(s)?
            .and_then(|s| s.parse().ok())
            .ok_or_else(invalid)?
    } else {
        table_native_text::number(v)?.ok_or_else(invalid)?.trunc() as i64
    };
    if id.unsigned_abs() > rapiddoc_core::resources::limit(1_000_000_000_000) {
        return Err(limit());
    }
    Ok(id)
}
fn build(
    v: &Value,
    text: &Text,
    d: &mut Diagnostic,
    work: &mut Work,
) -> Result<Option<Value>, Error> {
    d.extend(
        json!({"evidence":"banded_numeric","first_rejection_gate":"repeated_data_rows"})
            .as_object()
            .unwrap()
            .clone(),
    );
    let mut dense: Vec<_> = text
        .rows
        .iter()
        .filter(|r| r.tokens.len() >= 2 && r.tokens[1..].iter().all(|t| numeric_value(&t.text)))
        .collect();
    if dense.len() < 3 {
        return Ok(None);
    }
    let mut counts: Vec<(usize, usize)> = Vec::new();
    for r in &dense {
        if let Some((_, n)) = counts.iter_mut().find(|(n, _)| *n == r.tokens.len()) {
            *n += 1;
        } else {
            counts.push((r.tokens.len(), 1));
        }
    }
    let mut chosen = counts[0];
    for c in counts {
        if c.1 > chosen.1 {
            chosen = c;
        }
    }
    let (cols, count) = chosen;
    dense.retain(|r| r.tokens.len() == cols);
    if count < 3 || cols > 20 {
        return Ok(None);
    }
    let first = dense[0].row_index as usize;
    if !(1..=3).contains(&first) || text.rows.len() < first + 3 {
        return Ok(None);
    }
    let input = &v["table_input"];
    let (w, h, _) = sparse::dimensions(input)?;
    let rules = sparse::local_rules(input, w, h)?;
    let horizontal: Vec<_> = rules
        .iter()
        .filter(|r| r.orientation == "horizontal")
        .collect();
    let tol = 0.6f64.max(0.12 * text.median_glyph_height);
    let filled = filled_tracks(input, text, cols, w, work)?;
    let outer: Vec<_> = horizontal
        .iter()
        .filter(|r| r.end - r.start >= 0.9 * w)
        .collect();
    if filled.is_none()
        && !(outer.iter().any(|r| r.coordinate < 3. * tol)
            && outer.iter().any(|r| r.coordinate > h - 3. * tol))
    {
        return Ok(reject(d, "physical_boundary"));
    }
    let centers: Vec<_> = (0..cols)
        .map(|c| {
            median(
                dense
                    .iter()
                    .map(|r| (r.tokens[c].bbox[0] + r.tokens[c].bbox[2]) / 2.)
                    .collect(),
            )
        })
        .collect();
    if centers
        .windows(2)
        .any(|p| p[1] - p[0] < 2. * text.median_glyph_width)
    {
        return Ok(None);
    }
    let mut leaf = first - 1;
    if text.rows[leaf].tokens.len() < cols - 1 && first == 2 {
        leaf = 0;
    }
    let grouped =
        first == 3 && text.rows[0].tokens.len() == 1 && text.rows[2].tokens.len() == cols - 1;
    let mut gutter_rows: BTreeSet<_> = (first..text.rows.len()).map(|i| i as i64).collect();
    if grouped {
        gutter_rows.insert(leaf as i64);
    } else {
        gutter_rows.extend((0..first).map(|i| i as i64));
    }
    let tracks = if let Some(x) = filled {
        x
    } else {
        let mut x = vec![0.];
        for c in 0..cols - 1 {
            let lower = dense
                .iter()
                .map(|r| r.tokens[c].bbox[2])
                .fold(f64::NEG_INFINITY, f64::max);
            let upper = dense
                .iter()
                .map(|r| r.tokens[c + 1].bbox[0])
                .fold(f64::INFINITY, f64::min);
            let Some(b) = gutter(text, &gutter_rows, lower, upper, work)? else {
                return Ok(reject(d, "shared_gutter"));
            };
            x.push(b);
        }
        x.push(w);
        x
    };
    let glyphs: BTreeMap<_, _> = text.glyphs.iter().map(|g| (g.glyph_id, g)).collect();
    for row in &text.rows[first..] {
        let mut values = vec![Vec::new(); cols];
        work.charge(row.glyph_ids.len() * cols)?;
        for id in &row.glyph_ids {
            let g = glyphs.get(id).ok_or_else(invalid)?;
            let owners: Vec<_> = (0..cols)
                .filter(|c| tracks[*c] <= g.bbox[0] && g.bbox[2] <= tracks[*c + 1])
                .collect();
            if owners.len() != 1 {
                return Ok(reject(d, "body_glyph_split"));
            }
            values[owners[0]].push(*g);
        }
        if values.iter().any(Vec::is_empty) {
            return Ok(reject(d, "body_occupancy"));
        }
        for value in values.iter_mut().skip(1) {
            value.sort_by(|a, b| a.bbox[0].partial_cmp(&b.bbox[0]).unwrap());
            let joined = value.iter().map(|g| g.text.as_str()).collect::<String>();
            if !numeric_value(&joined) {
                return Ok(reject(d, "body_occupancy"));
            }
        }
    }
    let mut y = vec![0.];
    for p in text.rows.windows(2) {
        y.push(
            table_native_candidate::sum(&[p[0].bbox[1], p[0].bbox[3], p[1].bbox[1], p[1].bbox[3]])
                / 4.,
        );
    }
    y.push(h);
    let mut spans = BTreeMap::new();
    if grouped {
        let mut groups = Vec::new();
        for t in &text.rows[1].tokens {
            if t.bbox[2] < tracks[1] {
                continue;
            }
            work.charge(horizontal.len() + cols)?;
            let matches: Vec<_> = horizontal
                .iter()
                .filter(|r| {
                    t.bbox[3] < r.coordinate
                        && r.coordinate < text.rows[2].bbox[1]
                        && r.start <= t.bbox[0]
                        && t.bbox[2] <= r.end
                })
                .collect();
            if matches.len() != 1 {
                return Ok(reject(d, "group_rules"));
            }
            let rule = matches[0];
            let cs: Vec<_> = (1..cols)
                .filter(|c| rule.start - tol <= centers[*c] && centers[*c] <= rule.end + tol)
                .collect();
            if cs.is_empty() || cs.windows(2).any(|p| p[1] != p[0] + 1) {
                return Ok(None);
            }
            groups.push(cs);
        }
        let mut covered: Vec<_> = groups.iter().flatten().copied().collect();
        covered.sort();
        if covered != (1..cols).collect::<Vec<_>>() {
            return Ok(None);
        }
        if text
            .glyphs
            .iter()
            .any(|g| [0, 2].contains(&g.visual_row) && g.bbox[0] < tracks[1])
        {
            return Ok(None);
        }
        spans.insert((0, 0), (3, 1));
        spans.insert((0, 1), (1, cols - 1));
        for g in groups {
            spans.insert((1, g[0]), (1, g.len()));
        }
    }
    if text.rows.len() * cols > rapiddoc_core::resources::limit(2000) {
        return Err(limit());
    }
    let mut occupied = BTreeSet::new();
    let mut specs = Vec::new();
    for row in 0..text.rows.len() {
        for col in 0..cols {
            if occupied.contains(&(row, col)) {
                continue;
            }
            let (rs, cs) = spans.get(&(row, col)).copied().unwrap_or((1, 1));
            for r in row..row + rs {
                for c in col..col + cs {
                    occupied.insert((r, c));
                }
            }
            specs.push(Spec {
                row,
                col,
                rowspan: rs,
                colspan: cs,
                bbox: [tracks[col], y[row], tracks[col + cs], y[row + rs]],
            });
        }
    }
    work.charge(specs.len() * text.glyphs.len())?;
    for s in &specs {
        for g in &text.glyphs {
            let cx = (g.bbox[0] + g.bbox[2]) / 2.;
            if s.row as i64 <= g.visual_row
                && g.visual_row < (s.row + s.rowspan) as i64
                && s.bbox[0] < cx
                && cx < s.bbox[2]
                && (g.bbox[0] < s.bbox[0] || g.bbox[2] > s.bbox[2])
            {
                return Ok(reject(d, "header_glyph_split"));
            }
        }
    }
    work.charge(text.glyphs.len() * specs.len() * 3 + text.glyphs.len() * text.glyphs.len() * 2)?;
    let output = table_native_candidate::build(
        &json!({"source":"sparse_hybrid","rows":text.rows.len(),"cols":cols,"specs":specs,"text":v["text"],"structure_support":1.,"row_stability":1.,"column_stability":1.,"issues":["evidence=banded_numeric"],"use_grid_index":true}),
    )?;
    let mut candidate = output["candidate"].clone();
    let record = &output["diagnostics"];
    d.extend(
        json!({"x_tracks":tracks,"candidate_diagnostics":record})
            .as_object()
            .unwrap()
            .clone(),
    );
    if candidate.is_null()
        || candidate["score"].as_f64().unwrap() < 0.98
        || candidate["text_capture"].as_f64().unwrap() < 1.
        || record["ambiguous_glyph_ratio"].as_f64().unwrap_or(1.) > 0.
    {
        return Ok(reject(d, "verified_integrity"));
    }
    let chars = input.get("chars").map_or(Ok(&[][..]), |c| {
        c.as_array().map(|c| c.as_slice()).ok_or_else(invalid)
    })?;
    if chars.len() > rapiddoc_core::resources::limit(20000) {
        return Err(limit());
    }
    let mut raw: Vec<(i64, &Value)> = Vec::new();
    let mut positions = BTreeMap::new();
    for (i, c) in chars.iter().enumerate() {
        let id = strict_index(c, i)?;
        if let Some(p) = positions.get(&id) {
            raw[*p] = (id, c);
        } else {
            positions.insert(id, raw.len());
            raw.push((id, c));
        }
    }
    let mut visible: Vec<_> = raw
        .iter()
        .filter(|(_, c)| {
            !spatial::raw_string(c.get("char"), false)
                .trim_matches(python_space)
                .is_empty()
        })
        .map(|(i, _)| *i)
        .collect();
    visible.sort();
    for cell in candidate["cells"].as_array_mut().unwrap() {
        let ids: Vec<_> = cell["source_char_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap())
            .collect();
        let sources: BTreeSet<_> = ids.iter().copied().collect();
        let mut selected: Vec<_> = python_set::ordered_signed(&ids)
            .iter()
            .filter_map(|i| positions.get(i).map(|p| raw[*p].1))
            .collect();
        work.charge(raw.len())?;
        for (i, c) in &raw {
            let s = spatial::raw_string(c.get("char"), false);
            if s.is_empty() || !s.chars().all(python_space) {
                continue;
            }
            let p = visible.partition_point(|v| v < i);
            if p > 0
                && p < visible.len()
                && sources.contains(&visible[p - 1])
                && sources.contains(&visible[p])
            {
                selected.push(*c);
            }
        }
        let mut indexed = selected
            .into_iter()
            .map(|c| Ok((strict_index(c, 0)?, c)))
            .collect::<Result<Vec<_>, Error>>()?;
        indexed.sort_by_key(|(i, _)| *i);
        let selected = indexed.iter().map(|(_, c)| *c).collect::<Vec<_>>();
        let content = spatial::project(
            &selected,
            &input["table_bbox"],
            sparse::dimensions(input)?.2,
            work,
        )?;
        let content = content
            .split(python_space)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let squeeze = |s: &str| s.chars().filter(|c| !python_space(*c)).collect::<String>();
        if squeeze(&content) != squeeze(cell["content"].as_str().unwrap())
            && cell["row"].as_u64().unwrap() >= first as u64
        {
            return Ok(reject(d, "body_text_integrity"));
        }
        cell["content"] = json!(content);
    }
    d.insert("first_rejection_gate".into(), Value::Null);
    Ok(Some(candidate))
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
    let mut d = Diagnostic::new();
    let candidate = build(v, &text, &mut d, work)?;
    let html = candidate
        .as_ref()
        .map(table_native_candidate::render)
        .transpose()?;
    Ok(json!({"candidate":candidate,"diagnostics":d,"html":html}))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Value {
        serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-table-banded-numeric.json"),
            )
            .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn old_banded_numeric_grammar() {
        for c in fixture()["numbers"].as_array().unwrap() {
            assert_eq!(
                numeric_value(c["text"].as_str().unwrap()),
                c["expected"].as_bool().unwrap(),
                "{}",
                c["text"]
            );
        }
    }
    #[test]
    fn old_gutters_and_signed_source_set_order() {
        let f = fixture();
        for c in f["gutters"].as_array().unwrap() {
            let text = sparse::text(&c["text"]).unwrap();
            let indices = serde_json::from_value(c["indices"].clone()).unwrap();
            assert_eq!(
                json!(gutter(
                    &text,
                    &indices,
                    c["lower"].as_f64().unwrap(),
                    c["upper"].as_f64().unwrap(),
                    &mut Work(0)
                )
                .unwrap()),
                c["expected"]
            );
        }
        for c in f["sets"].as_array().unwrap() {
            let input: Vec<i64> = serde_json::from_value(c["input"].clone()).unwrap();
            assert_eq!(json!(python_set::ordered_signed(&input)), c["expected"]);
        }
    }
}
