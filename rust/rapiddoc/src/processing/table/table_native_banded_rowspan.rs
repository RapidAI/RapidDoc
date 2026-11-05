//! DocVortex 0.6.9: recover centered row labels only with repeated physical bands.
use super::{
    table_native_banded_numeric::numeric_value, table_native_candidate,
    table_native_sparse as sparse, table_native_text::Text,
};
use crate::legacy_page::Work;
use rapiddoc_core::Error;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len() / 2;
    if values.len() % 2 == 1 {
        values[n]
    } else {
        (values[n - 1] + values[n]) / 2.
    }
}
pub(super) fn candidate(
    input: &Value,
    original: &Value,
    refined: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    let old: Text =
        serde_json::from_value(original.clone()).map_err(|_| crate::legacy_page::invalid())?;
    let text: Text =
        serde_json::from_value(refined.clone()).map_err(|_| crate::legacy_page::invalid())?;
    if text.rows.len() <= old.rows.len() {
        return Ok(Value::Null);
    }
    let (width, height, _) = sparse::dimensions(input)?;
    let rules = sparse::local_rules(input, width, height)?;
    let tolerance = 0.6f64.max(0.12 * text.median_glyph_height);
    let mut boundaries: Vec<_> = rules
        .iter()
        .filter(|r| r.orientation == "horizontal" && r.end - r.start >= 0.9 * width)
        .map(|r| r.coordinate)
        .collect();
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup();
    if boundaries.len() < 5
        || boundaries[0] > tolerance
        || height - boundaries[boundaries.len() - 1] > tolerance
        || rules.iter().any(|r| {
            r.orientation == "vertical"
                && r.coordinate > tolerance
                && r.coordinate < width - tolerance
        })
        || rules.iter().any(|r| {
            r.orientation == "horizontal"
                && boundaries
                    .iter()
                    .all(|y| (r.coordinate - y).abs() > tolerance)
        })
    {
        return Ok(Value::Null);
    }
    let rects = sparse::rectangles(input, width, height)?;
    let Some(h) = sparse::hypothesis(&old, width, &rules, &rects, &mut Map::new(), work)? else {
        return Ok(Value::Null);
    };
    let tracks = h.x_tracks;
    if tracks.len() < 4 {
        return Ok(Value::Null);
    }
    let cols = tracks.len() - 1;
    let header: Vec<_> = text
        .rows
        .iter()
        .filter(|r| r.bbox[3] <= boundaries[1])
        .collect();
    if header.len() != 1
        || header[0].bbox[1] < boundaries[0]
        || text
            .glyphs
            .iter()
            .any(|g| !(0..cols).any(|c| tracks[c] <= g.bbox[0] && g.bbox[2] <= tracks[c + 1]))
    {
        return Ok(Value::Null);
    }
    let mut specs:Vec<_>=(0..cols).map(|c|json!({"row":0,"col":c,"rowspan":1,"colspan":1,"bbox":[tracks[c],0.,tracks[c+1],boundaries[1]]})).collect();
    let mut logical = 1;
    for pair in boundaries[1..].windows(2) {
        let (top, bottom) = (pair[0], pair[1]);
        work.charge(text.glyphs.len() * cols)?;
        let members: Vec<_> = text
            .glyphs
            .iter()
            .filter(|g| top < (g.bbox[1] + g.bbox[3]) / 2. && (g.bbox[1] + g.bbox[3]) / 2. < bottom)
            .collect();
        let labels: Vec<_> = members
            .iter()
            .copied()
            .filter(|g| g.bbox[2] <= tracks[1])
            .collect();
        let data: Vec<_> = members
            .iter()
            .copied()
            .filter(|g| g.bbox[0] >= tracks[1])
            .collect();
        let indices: BTreeSet<_> = data.iter().map(|g| g.visual_row).collect();
        if labels.is_empty()
            || indices.len() != 2
            || labels
                .iter()
                .map(|g| g.visual_row)
                .collect::<BTreeSet<_>>()
                .len()
                != 1
        {
            return Ok(Value::Null);
        }
        let rows: Vec<Vec<_>> = indices
            .iter()
            .map(|i| {
                data.iter()
                    .copied()
                    .filter(|g| g.visual_row == *i)
                    .collect()
            })
            .collect();
        let centers: Vec<_> = rows
            .iter()
            .map(|row| median(row.iter().map(|g| (g.bbox[1] + g.bbox[3]) / 2.).collect()))
            .collect();
        let label = median(
            labels
                .iter()
                .map(|g| (g.bbox[1] + g.bbox[3]) / 2.)
                .collect(),
        );
        if !(centers[0] < label && label < centers[1])
            || (label - (centers[0] + centers[1]) / 2.).abs() > tolerance
            || members
                .iter()
                .any(|g| g.bbox[1] < top || g.bbox[3] > bottom)
        {
            return Ok(Value::Null);
        }
        for row in &rows {
            for col in 1..cols {
                let mut owned: Vec<_> = row
                    .iter()
                    .filter(|g| tracks[col] <= g.bbox[0] && g.bbox[2] <= tracks[col + 1])
                    .collect();
                owned.sort_by(|a, b| a.bbox[0].total_cmp(&b.bbox[0]));
                if owned.is_empty()
                    || !numeric_value(&owned.iter().map(|g| g.text.as_str()).collect::<String>())
                {
                    return Ok(Value::Null);
                }
            }
        }
        let upper = rows[0]
            .iter()
            .map(|g| g.bbox[3])
            .fold(f64::NEG_INFINITY, f64::max);
        let lower = rows[1]
            .iter()
            .map(|g| g.bbox[1])
            .fold(f64::INFINITY, f64::min);
        if upper >= lower {
            return Ok(Value::Null);
        }
        let cut = (upper + lower) / 2.;
        specs.push(json!({"row":logical,"col":0,"rowspan":2,"colspan":1,"bbox":[tracks[0],top,tracks[1],bottom]}));
        for (offset, (y0, y1)) in [(top, cut), (cut, bottom)].iter().enumerate() {
            for col in 1..cols {
                specs.push(json!({"row":logical+offset,"col":col,"rowspan":1,"colspan":1,"bbox":[tracks[col],y0,tracks[col+1],y1]}));
            }
        }
        logical += 2;
    }
    let output = table_native_candidate::build(
        &json!({"source":"sparse_hybrid","rows":logical,"cols":cols,"specs":specs,"text":refined,"structure_support":1.,"row_stability":1.,"column_stability":1.,"issues":["evidence=banded_rowspan"],"use_grid_index":true}),
    )?;
    let c = &output["candidate"];
    if c.is_null()
        || c["text_capture"].as_f64().unwrap_or(0.) < 1.
        || c["order_consistency"].as_f64().unwrap_or(0.) < 1.
        || c["score"].as_f64().unwrap_or(0.) < 0.98
        || output["diagnostics"]["ambiguous_glyph_ratio"]
            .as_f64()
            .unwrap_or(1.)
            > 0.
    {
        return Ok(Value::Null);
    }
    let sources: Vec<_> = c["cells"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|cell| {
            cell["source_char_indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_i64().unwrap())
        })
        .collect();
    if sources.len() != text.glyphs.len()
        || sources.iter().collect::<BTreeSet<_>>().len() != sources.len()
    {
        return Ok(Value::Null);
    }
    Ok(c.clone())
}
