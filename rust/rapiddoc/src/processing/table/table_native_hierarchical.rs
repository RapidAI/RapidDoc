//! Upstream 72656be/2c6c0ad: independently proven child cells constrain every fallback.
use super::{
    table_native_candidate as candidate, table_native_sparse as sparse,
    table_native_sparse_hybrid as hybrid,
    table_native_text::{Row, Text},
};
use crate::legacy_page::{invalid, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
type Box4 = [f64; 4];
pub(super) struct Evidence {
    bbox: Box4,
    child: Option<Value>,
}
fn shift(b: Box4, x: f64, y: f64) -> Box4 {
    [b[0] - x, b[1] - y, b[2] - x, b[3] - y]
}
fn subset(text: &Text, b: Box4) -> Option<Text> {
    let ids: BTreeSet<_> = text
        .glyphs
        .iter()
        .filter(|g| {
            let (x, y) = ((g.bbox[0] + g.bbox[2]) / 2., (g.bbox[1] + g.bbox[3]) / 2.);
            b[0] <= x && x < b[2] && b[1] <= y && y < b[3]
        })
        .map(|g| g.glyph_id)
        .collect();
    let mut rows = Vec::new();
    let mut mapping = BTreeMap::new();
    for row in &text.rows {
        let mut tokens: Vec<_> = row
            .tokens
            .iter()
            .filter(|t| t.glyph_ids.iter().any(|i| ids.contains(i)))
            .cloned()
            .collect();
        if tokens.is_empty() {
            continue;
        }
        if tokens
            .iter()
            .any(|t| t.glyph_ids.iter().any(|i| !ids.contains(i)))
        {
            return None;
        }
        mapping.insert(row.row_index, rows.len() as i64);
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for token in &mut tokens {
            token.bbox = shift(token.bbox, b[0], b[1]);
            for i in 0..4 {
                bounds[i] = if i < 2 {
                    bounds[i].min(token.bbox[i])
                } else {
                    bounds[i].max(token.bbox[i])
                };
            }
        }
        rows.push(Row {
            row_index: rows.len() as i64,
            bbox: bounds,
            tokens,
            glyph_ids: row
                .glyph_ids
                .iter()
                .copied()
                .filter(|i| ids.contains(i))
                .collect(),
        });
    }
    if rows.is_empty() {
        return None;
    }
    let glyphs = text
        .glyphs
        .iter()
        .filter(|g| ids.contains(&g.glyph_id))
        .map(|g| {
            let mut g = g.clone();
            g.bbox = shift(g.bbox, b[0], b[1]);
            g.advance_bbox = g.advance_bbox.map(|a| shift(a, b[0], b[1]));
            g.origin = g.origin.map(|[x, y]| [x - b[0], y - b[1]]);
            g.visual_row = mapping[&g.visual_row];
            g
        })
        .collect();
    Some(Text {
        glyphs,
        rows,
        median_glyph_width: text.median_glyph_width,
        median_glyph_height: text.median_glyph_height,
    })
}
fn topology(c: &Value) -> Value {
    json!([
        c["rows"],
        c["cols"],
        c["cells"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| json!([s["row"], s["col"], s["rowspan"], s["colspan"]]))
            .collect::<Vec<_>>()
    ])
}
fn subtable(input: &Value, text: &Text, b: Box4, work: &mut Work) -> Result<Option<Value>, Error> {
    let Some(text) = subset(text, b).filter(|t| t.rows.len() >= 3) else {
        return Ok(None);
    };
    let mut input = input.clone();
    input["table_bbox"] = json!(b);
    let output = hybrid::generate_with_work(&json!({"table_input":input,"text":text}), work)?;
    let mut accepted = Vec::new();
    for c in output["candidates"].as_array().ok_or_else(invalid)? {
        if c["score"].as_f64().unwrap() >= 0.98
            && c["text_capture"] == 1.0
            && c["order_consistency"] == 1.0
            && c["rows"].as_u64().unwrap() as usize == text.rows.len()
            && c["cols"].as_u64().unwrap() >= 2
            && c["cells"].as_array().unwrap().iter().all(|s| {
                s["rowspan"] == 1
                    && s["colspan"] == 1
                    && !s["content"]
                        .as_str()
                        .unwrap()
                        .trim_matches(rapiddoc_native::text_boundary::python_space)
                        .is_empty()
            })
        {
            accepted.push(c)
        }
    }
    let Some(first) = accepted.first() else {
        return Ok(None);
    };
    let signature = topology(first);
    if accepted.iter().any(|c| topology(c) != signature) {
        return Ok(None);
    }
    Ok(accepted
        .into_iter()
        .max_by(|a, b| {
            a["score"]
                .as_f64()
                .unwrap()
                .total_cmp(&b["score"].as_f64().unwrap())
        })
        .cloned())
}
fn local_input(input: &Value) -> Result<Value, Error> {
    let (w, h, _) = sparse::dimensions(input)?;
    let mut converted = Vec::new();
    for r in sparse::local_rules(input, w, h)? {
        let half = 0.05f64.max(r.width / 2.);
        let b = if r.orientation == "horizontal" {
            [r.start, r.coordinate - half, r.end, r.coordinate + half]
        } else {
            [r.coordinate - half, r.start, r.coordinate + half, r.end]
        };
        converted.push(json!({"bbox":b,"width":r.width,"orientation":r.orientation}));
    }
    Ok(
        json!({"table_bbox":[0.,0.,w,h],"page_size":[w,h],"angle":0,"chars":[],"drawing_lines":converted,"rectangles":[]}),
    )
}
fn repeated(text: &Text) -> bool {
    // Python Counter.most_common resolves tied counts in first-seen order.
    let mut counts = Vec::<(usize, usize)>::new();
    for row in &text.rows {
        let n = row.tokens.len();
        if n >= 2 {
            if let Some(pair) = counts.iter_mut().find(|p| p.0 == n) {
                pair.1 += 1
            } else {
                counts.push((n, 1));
            }
        }
    }
    let mut best = None;
    for pair in counts {
        if best.is_none_or(|(_, n)| pair.1 > n) {
            best = Some(pair)
        }
    }
    let Some((cols, support)) = best else {
        return false;
    };
    if support < 3 || (support as f64) < 0.6 * text.rows.len() as f64 {
        return false;
    }
    let rows: Vec<_> = text
        .rows
        .iter()
        .filter(|r| r.tokens.len() == cols)
        .collect();
    let tolerance = 0.5f64.max(0.2 * text.median_glyph_height);
    for col in 0..cols {
        let left = rows
            .iter()
            .map(|r| r.tokens[col].bbox[0])
            .fold(f64::INFINITY, f64::min);
        let right = rows
            .iter()
            .map(|r| r.tokens[col].bbox[0])
            .fold(f64::NEG_INFINITY, f64::max);
        if right - left > tolerance {
            return false;
        }
        if col + 1 < cols {
            let next = rows
                .iter()
                .map(|r| r.tokens[col + 1].bbox[0])
                .fold(f64::INFINITY, f64::min);
            let end = rows
                .iter()
                .map(|r| r.tokens[col].bbox[2])
                .fold(f64::NEG_INFINITY, f64::max);
            if next - end < 2. * text.median_glyph_width {
                return false;
            }
        }
    }
    true
}
pub(super) fn find(input: &Value, text: &Text, work: &mut Work) -> Result<Vec<Evidence>, Error> {
    if text.glyphs.iter().any(|g| g.writing_angle != Some(0)) {
        return Ok(Vec::new());
    }
    let (w, h, _) = sparse::dimensions(input)?;
    let rules = sparse::local_rules(input, w, h)?;
    if !rules.iter().any(|r| r.orientation == "vertical") {
        return Ok(Vec::new());
    }
    let tolerance = 0.5f64.max(0.12 * text.median_glyph_height);
    let horizontal: Vec<_> = rules
        .iter()
        .filter(|r| r.orientation == "horizontal")
        .collect();
    let full: Vec<_> = horizontal
        .iter()
        .copied()
        .filter(|r| r.end - r.start >= 0.9 * w)
        .collect();
    let local = local_input(input)?;
    let mut output = Vec::<Evidence>::new();
    for line in rules.iter().filter(|r| r.orientation == "vertical") {
        if !(0.1 * w < line.coordinate && line.coordinate < 0.8 * w) {
            continue;
        }
        let partial: Vec<_> = horizontal
            .iter()
            .filter(|r| {
                (r.start - line.coordinate).abs() <= tolerance
                    && w - r.end <= 3. * tolerance
                    && line.start - tolerance <= r.coordinate
                    && r.coordinate <= line.end + tolerance
            })
            .map(|r| r.coordinate)
            .collect();
        let separators = sparse::clusters(partial, tolerance, work)?;
        if separators.len() < 3 {
            continue;
        }
        let upper = full
            .iter()
            .filter(|r| line.start - 2. * tolerance <= r.coordinate && r.coordinate <= line.start)
            .map(|r| r.coordinate)
            .max_by(f64::total_cmp);
        let lower = full
            .iter()
            .filter(|r| line.end <= r.coordinate && r.coordinate <= line.end + 2. * tolerance)
            .map(|r| r.coordinate)
            .min_by(f64::total_cmp);
        let (Some(top), Some(bottom)) = (upper, lower) else {
            continue;
        };
        let bbox = [line.coordinate, top, w, bottom];
        if output.iter().any(|e| {
            (bbox[0] - e.bbox[0]).abs() <= tolerance && (bbox[1] - e.bbox[1]).abs() <= tolerance
        }) {
            continue;
        }
        let Some(subset) = subset(text, bbox).filter(|t| t.rows.len() == separators.len() + 1)
        else {
            continue;
        };
        let child = subtable(&local, text, bbox, work)?;
        if child.is_none() && !repeated(&subset) {
            continue;
        }
        let Some(parent) = self::subset(text, [0., top, line.coordinate, bottom]) else {
            continue;
        };
        if parent.rows.len() >= subset.rows.len() {
            continue;
        }
        output.push(Evidence { bbox, child });
    }
    Ok(output)
}
pub(super) fn preserves(candidate: &Value, evidence: &[Evidence]) -> bool {
    if evidence.is_empty() {
        return true;
    }
    let mut owners = BTreeMap::new();
    for (i, cell) in candidate["cells"].as_array().unwrap().iter().enumerate() {
        for index in cell["source_char_indices"].as_array().unwrap() {
            owners.insert(index.as_i64().unwrap(), i);
        }
    }
    for region in evidence {
        let Some(child) = &region.child else {
            return false;
        };
        let mut child_owners = BTreeSet::new();
        for cell in child["cells"].as_array().unwrap() {
            let occupied: BTreeSet<_> = cell["source_char_indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| owners.get(&i.as_i64().unwrap()).copied())
                .collect();
            if occupied.contains(&None) || occupied.len() != 1 {
                return false;
            }
            if !child_owners.insert(*occupied.first().unwrap()) {
                return false;
            }
        }
    }
    true
}
fn spec(row: &Value, col: Value, rowspan: Value, colspan: Value, b: Box4) -> Value {
    json!({"row":row,"col":col,"rowspan":rowspan,"colspan":colspan,"bbox":b})
}
pub(super) fn build(
    input: &Value,
    text: &Text,
    evidence: &[Evidence],
    work: &mut Work,
) -> Result<Option<Value>, Error> {
    if evidence.len() != 1 {
        return Ok(None);
    };
    let region = &evidence[0];
    let [split, top, w, bottom] = region.bbox;
    let Some(child) = &region.child else {
        return Ok(None);
    };
    let local = local_input(input)?;
    let h = local["page_size"][1].as_f64().unwrap();
    if text
        .glyphs
        .iter()
        .any(|g| (g.bbox[1] + g.bbox[3]) / 2. >= bottom)
    {
        return Ok(None);
    }
    let Some(upper) = subtable(&local, text, [0., 0., w, top], work)? else {
        return Ok(None);
    };
    if upper["cols"] != 2 {
        return Ok(None);
    }
    let child_cols = child["cols"].as_u64().unwrap();
    let child_rows = child["rows"].as_u64().unwrap();
    let upper_rows = upper["rows"].as_u64().unwrap();
    let mut specs = Vec::new();
    for cell in upper["cells"].as_array().unwrap() {
        let first = cell["col"] == 0;
        specs.push(spec(
            &cell["row"],
            json!(if first { 0 } else { 1 }),
            json!(1),
            json!(if first { 1 } else { child_cols }),
            [
                if first { 0. } else { split },
                cell["bbox"][1].as_f64().unwrap(),
                if first { split } else { w },
                cell["bbox"][3].as_f64().unwrap(),
            ],
        ));
    }
    specs.push(spec(
        &json!(upper_rows),
        json!(0),
        json!(child_rows),
        json!(1),
        [0., top, split, h],
    ));
    for cell in child["cells"].as_array().unwrap() {
        let b: Box4 = serde_json::from_value(cell["bbox"].clone()).map_err(|_| invalid())?;
        let mut b = shift(b, -split, -top);
        if cell["row"].as_u64().unwrap() == child_rows - 1 {
            b[3] = h;
        }
        specs.push(spec(
            &json!(cell["row"].as_u64().unwrap() + upper_rows),
            json!(cell["col"].as_u64().unwrap() + 1),
            json!(1),
            json!(1),
            b,
        ));
    }
    work.charge(specs.len().saturating_mul(text.glyphs.len()))?;
    let output = candidate::build(
        &json!({"source":"sparse_hybrid","rows":upper_rows+child_rows,"cols":child_cols+1,"specs":specs,"text":text,"structure_support":upper["score"].as_f64().unwrap().min(child["score"].as_f64().unwrap()),"row_stability":1.,"column_stability":1.,"issues":["evidence=hierarchical_rule_bands"],"use_grid_index":true,"require_atomic_tokens":true}),
    )?;
    let c = &output["candidate"];
    if c.is_null() || c["text_capture"] != 1.0 || c["order_consistency"] != 1.0 {
        return Ok(None);
    }
    for original in upper["cells"].as_array().unwrap() {
        let Some(combined) = c["cells"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["row"] == original["row"] && s["col"] == original["col"])
        else {
            return Ok(None);
        };
        let ids = |v: &Value| {
            v["source_char_indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| i.as_i64().unwrap())
                .collect::<BTreeSet<_>>()
        };
        if ids(original) != ids(combined) {
            return Ok(None);
        }
    }
    Ok(preserves(c, evidence).then(|| c.clone()))
}
