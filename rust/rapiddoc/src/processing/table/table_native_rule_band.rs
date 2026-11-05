//! 旧rule_band.py三条完整横线下的换行表头与描述列同步重启候选。
use super::{table_native_candidate, table_native_sparse as sparse};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
pub fn inspect_columns(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let text = sparse::text(&v["text"])?;
    let (w, h, _) = sparse::dimensions(&v["table_input"])?;
    let rules = sparse::local_rules(&v["table_input"], w, h)?;
    let rects = sparse::rectangles(&v["table_input"], w, h)?;
    let mut d = Map::new();
    let mut work = Work(0);
    let hyp = sparse::hypothesis(&text, w, &rules, &rects, &mut d, &mut work)?;
    let bottom = v
        .get("header_bottom")
        .map_or(Ok(0.), super::table_native_vector::number)?;
    let mut gutter = Map::new();
    let mut stable = None;
    let mut occupancy = None;
    if let Some(h) = &hyp {
        stable = Some(sparse::gutters(
            &text,
            &h.x_tracks,
            &h.physical_boundaries,
            bottom,
            &mut gutter,
            &mut work,
        )?);
        let glyphs = text.glyphs.iter().map(|g| (g.glyph_id, g)).collect();
        occupancy = Some(
            text.rows
                .iter()
                .map(|r| sparse::occupancy(r, &glyphs, &h.x_tracks, &mut work))
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
    Ok(
        json!({"rules":rules,"rectangles":rects,"hypothesis":hyp,"hypothesis_diagnostics":d,"stable_gutters":stable,"gutter_diagnostics":gutter,"occupancy":occupancy,"overlapping_formula_rows":sparse::overlapping(&text,bottom)}),
    )
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
    let (w, h, _) = sparse::dimensions(&v["table_input"])?;
    let rules = sparse::local_rules(&v["table_input"], w, h)?;
    let tolerance = (0.15 * text.median_glyph_height).max(1.);
    let mut horizontal: Vec<_> = rules
        .iter()
        .filter(|r| r.orientation == "horizontal" && r.end - r.start >= 0.9 * w)
        .map(|r| r.coordinate)
        .collect();
    horizontal.sort_by(|a, b| a.partial_cmp(b).unwrap());
    horizontal.dedup();
    if horizontal.len() != 3
        || horizontal[0] > tolerance
        || h - horizontal[2] > tolerance
        || rules.iter().any(|r| {
            r.orientation == "vertical" && tolerance < r.coordinate && r.coordinate < w - tolerance
        })
    {
        return result(Vec::new(), Vec::new());
    }
    let bottom = horizontal[1];
    let header: Vec<_> = text
        .rows
        .iter()
        .filter(|r| r.bbox[3] <= bottom + 0.5)
        .collect();
    let body: Vec<_> = text
        .rows
        .iter()
        .filter(|r| r.bbox[1] >= bottom - 0.5)
        .collect();
    if !(1..=3).contains(&header.len())
        || body.len() < 3
        || header.len() + body.len() != text.rows.len()
        || sparse::overlapping(&text, bottom)
    {
        return result(Vec::new(), Vec::new());
    }
    let rectangles = sparse::rectangles(&v["table_input"], w, h)?;
    let mut ignored = Map::new();
    let Some(hypothesis) = sparse::hypothesis(&text, w, &rules, &rectangles, &mut ignored, work)?
    else {
        return result(Vec::new(), Vec::new());
    };
    let x = &hypothesis.x_tracks;
    let cols = x.len() - 1;
    let glyphs: BTreeMap<_, _> = text.glyphs.iter().map(|g| (g.glyph_id, g)).collect();
    let occupancy = body
        .iter()
        .map(|r| sparse::occupancy(r, &glyphs, x, work))
        .collect::<Result<Vec<_>, _>>()?;
    let full: BTreeSet<_> = (0..cols).collect();
    let common = occupancy.iter().skip(1).fold(occupancy[0].clone(), |a, b| {
        a.intersection(b).copied().collect::<BTreeSet<_>>()
    });
    let prefix = common.first().copied().unwrap_or(0);
    let mut groups = BTreeMap::new();
    if prefix > 0 {
        let suffix: BTreeSet<_> = (prefix..cols).collect();
        let starts: Vec<_> = occupancy
            .iter()
            .enumerate()
            .filter(|(_, o)| **o == full)
            .map(|(i, _)| i)
            .collect();
        let ends: Vec<_> = starts.iter().skip(1).copied().chain([body.len()]).collect();
        if prefix < 2
            || suffix.len() < 2
            || common != suffix
            || starts.len() < 3
            || starts[0] != 0
            || occupancy.iter().any(|o| *o != full && *o != suffix)
            || starts.iter().zip(&ends).any(|(a, b)| b - a < 2)
        {
            return result(Vec::new(), Vec::new());
        }
        groups.extend(starts.into_iter().zip(ends));
    } else if occupancy.iter().any(|o| *o != full) {
        return result(Vec::new(), Vec::new());
    }
    if groups.is_empty() && header.len() == 1 {
        return result(Vec::new(), Vec::new());
    }
    let ho = header
        .iter()
        .map(|r| sparse::occupancy(r, &glyphs, x, work))
        .collect::<Result<Vec<_>, _>>()?;
    if ho
        .iter()
        .flat_map(|s| s.iter().copied())
        .collect::<BTreeSet<_>>()
        != full
        || header.len() > 1 && ho.last() != Some(&full)
    {
        return result(Vec::new(), Vec::new());
    }
    let margin = (0.05 * text.median_glyph_width).max(0.25);
    for row in &header {
        work.charge(row.tokens.len() * cols)?;
        if row.tokens.iter().any(|t| {
            x[1..cols]
                .iter()
                .any(|b| t.bbox[0] + margin < *b && *b < t.bbox[2] - margin)
        }) {
            return result(Vec::new(), Vec::new());
        }
    }
    if !sparse::gutters(
        &text,
        x,
        &hypothesis.physical_boundaries,
        bottom,
        &mut ignored,
        work,
    )? {
        return result(Vec::new(), Vec::new());
    }
    let mut y = vec![bottom];
    for pair in body.windows(2) {
        if pair[0].bbox[3] > pair[1].bbox[1] + 0.5 {
            return result(Vec::new(), Vec::new());
        }
        y.push((pair[0].bbox[3] + pair[1].bbox[1]) / 2.)
    }
    y.push(h);
    if (body.len() + 1) * cols > rapiddoc_core::resources::limit(2000) {
        return Err(limit());
    }
    let mut specs: Vec<_> = (0..cols)
        .map(|c| json!({"row":0,"col":c,"rowspan":1,"colspan":1,"bbox":[x[c],0.,x[c+1],bottom]}))
        .collect();
    for i in 0..body.len() {
        for c in 0..cols {
            if c < prefix && !groups.contains_key(&i) {
                continue;
            }
            let end = if c < prefix { groups[&i] } else { i + 1 };
            specs.push(json!({"row":i+1,"col":c,"rowspan":end-i,"colspan":1,"bbox":[x[c],y[i],x[c+1],y[end]]}))
        }
    }
    work.charge(text.glyphs.len() * specs.len() * 3 + text.glyphs.len() * text.glyphs.len() * 2)?;
    let output = table_native_candidate::build(
        &json!({"source":"sparse_multiline","rows":body.len()+1,"cols":cols,"specs":specs,"text":v["text"],"structure_support":1.,"row_stability":1.,"column_stability":1.,"issues":["evidence=physical_header_band",format!("wrapped_header_rows={}",header.len()),format!("descriptor_columns={prefix}")],"use_grid_index":true}),
    )?;
    let record = output["diagnostics"]
        .as_object()
        .cloned()
        .ok_or_else(invalid)?;
    let mut diagnostic = json!({"source":"sparse_multiline","evidence":"physical_header_band"});
    diagnostic.as_object_mut().unwrap().extend(record.clone());
    let c = &output["candidate"];
    if c.is_null()
        || c["text_capture"].as_f64().unwrap() < 1.
        || c["order_consistency"].as_f64().unwrap() < 1.
        || c["score"].as_f64().unwrap() < 0.98
        || record
            .get("ambiguous_glyph_ratio")
            .and_then(Value::as_f64)
            .unwrap_or(1.)
            > 0.
    {
        return result(Vec::new(), vec![diagnostic]);
    }
    result(vec![c.clone()], vec![diagnostic])
}
