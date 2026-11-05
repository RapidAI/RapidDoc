//! 旧engine.py的输入早期门与已有矢量诊断守卫；不替代完整_evaluate候选执行。
use super::{
    table_native_rule,
    table_native_vector::{normalized, number, validate_text},
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn records<'a>(v: &'a Value, key: &str, max: usize) -> Result<&'a [Value], Error> {
    let a = v.get(key).map_or(Ok(&[][..]), |v| {
        v.as_array().map(|a| a.as_slice()).ok_or_else(invalid)
    })?;
    if a.len() > crate::legacy_page::budget::records(max) {
        return Err(limit());
    }
    Ok(a)
}
fn indices(v: Option<&Value>) -> Result<BTreeSet<u64>, Error> {
    let Some(v) = v else {
        return Ok(BTreeSet::new());
    };
    let a = v.as_array().ok_or_else(invalid)?;
    if a.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    a.iter()
        .map(|v| {
            v.as_u64()
                .filter(|n| *n <= rapiddoc_core::resources::limit(1000))
                .ok_or_else(invalid)
        })
        .collect()
}
fn verified_rect(c: &Value) -> bool {
    c["source"] == "vector_grid"
        && c["score"].as_f64().is_some_and(|s| s >= 0.95)
        && c["issues"].as_array().is_some_and(|a| {
            a.iter().any(|v| v == "evidence=rect_grid")
                && a.iter().any(|v| v == "ambiguous_separator_ratio=0.0000")
        })
}
#[cfg(test)]
mod capacity_tests {
    use super::*;
    #[test]
    fn python_primitive_gate_rejects_candidate_without_rejecting_document() {
        let _scope = crate::legacy_page::budget::host_page();
        let request = json!({"table_input":{"table_bbox":[0.,0.,100.,100.],
            "page_size":[100.,100.], "drawing_lines":vec![json!({"bbox":[1.,1.,10.,2.]});5001]},
            "text":{"glyphs":[],"rows":[],"median_glyph_width":1.,"median_glyph_height":1.}});
        let out = inspect(&request).unwrap();
        assert_eq!(out["primitive_count"], 5001);
        assert_eq!(out["input_rejection_gate"], "primitive_limit");
    }
}
pub fn inspect(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut work = Work(0);
    let input = &v["table_input"];
    let box_ = normalized(&input["table_bbox"])?;
    let page = input["page_size"]
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or_else(invalid)?;
    let (w, h) = (number(&page[0])?, number(&page[1])?);
    let lines = records(input, "drawing_lines", 20000)?;
    let rects = records(input, "rectangles", 20000)?;
    if lines.len() + rects.len() > rapiddoc_core::resources::limit(20000) {
        return Err(limit());
    }
    let mut primitive_count = 0;
    if let Some(t) = box_ {
        for p in lines.iter().chain(rects) {
            work.charge(1)?;
            if let Some(b) = normalized(&p["bbox"])? {
                if b[2].min(t[2]) > b[0].max(t[0]) && b[3].min(t[3]) > b[1].max(t[1]) {
                    primitive_count += 1
                }
            }
        }
    }
    let gate = if box_
        .is_none_or(|b| w <= 0. || h <= 0. || b[0] < 0. || b[1] < 0. || b[2] > w || b[3] > h)
    {
        Some("input_geometry")
    } else if primitive_count > 5000 {
        Some("primitive_limit")
    } else {
        None
    };
    let attempts = records(v, "vector_attempts", 1000)?;
    let candidates = records(v, "candidates", 1000)?;
    let text = validate_text(&v["text"])?;
    // 复用现有候选schema/资源验证；这里不把select结果当成完整引擎决策。
    table_native_rule::select(&json!({"candidates":candidates,"remove_undercounted":false}))?;
    for c in candidates {
        number(&c["text_capture"])?;
        number(&c["order_consistency"])?;
    }
    let (mut alias_blank, mut undercount, mut minimum) = (false, false, 0u64);
    let mut line_grids = Vec::new();
    let mut rect_grids = Vec::new();
    for a in attempts {
        if a["evidence"] == "line_grid" {
            line_grids.push(&a["grid"]);
            let mut hypotheses = vec![a];
            hypotheses.extend(records(a, "track_hypotheses", 1000)?);
            work.charge(hypotheses.len())?;
            for h in hypotheses {
                if h["first_rejection_gate"] == "physical_row_undercount" {
                    undercount = true
                }
                if h["first_rejection_gate"] == "empty_row" {
                    let empty = indices(h.get("empty_rows"))?;
                    let affected = indices(h.get("alias_affected_rows"))?;
                    if !empty.is_disjoint(&affected) {
                        alias_blank = true
                    }
                }
            }
            if a["first_rejection_gate"] == "physical_row_undercount" && a["grid"].is_object() {
                if let Some(tracks) = a["canonical_tracks"].as_array() {
                    let cols = a["grid"]["cols"]
                        .as_u64()
                        .filter(|n| *n <= rapiddoc_core::resources::limit(1000))
                        .ok_or_else(invalid)?;
                    if tracks.len() == cols as usize + 1 {
                        minimum = minimum.max(cols)
                    }
                }
            }
        } else if a["evidence"] == "rect_grid" {
            rect_grids.push(&a["grid"])
        }
    }
    work.charge(line_grids.len() * rect_grids.len())?;
    let conflict = line_grids.iter().filter(|g| g.is_object()).any(|l| {
        rect_grids
            .iter()
            .filter(|g| g.is_object())
            .any(|r| l["rows"] != r["rows"] || l["cols"] != r["cols"])
    });
    let rectangles: Vec<_> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| verified_rect(c))
        .collect();
    let resolved = if rectangles.len() == 1 {
        let (index, c) = rectangles[0];
        let rows = c["rows"].as_u64().ok_or_else(invalid)?;
        let cols = c["cols"].as_u64().ok_or_else(invalid)?;
        let matches = attempts
            .iter()
            .filter(|a| a["evidence"] == "line_grid")
            .any(|a| {
                a["first_rejection_gate"] == "physical_row_undercount"
                    && a["grid"].is_object()
                    && a["grid"]["cols"].as_u64() == Some(cols)
                    && a["grid"]
                        .get("rows")
                        .map_or(Some(0), Value::as_u64)
                        .is_some_and(|r| r < rows)
            });
        if matches
            && rows == text.rows.len() as u64
            && number(&c["text_capture"])? >= 1.
            && number(&c["order_consistency"])? >= 1.
        {
            Some(index)
        } else {
            None
        }
    } else {
        None
    };
    let kept: Vec<_> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c["cols"].as_u64().is_some_and(|n| n >= minimum))
        .map(|(i, _)| i)
        .collect();
    Ok(
        json!({"primitive_count":primitive_count,"input_rejection_gate":gate,"alias_affected_physical_blank_row":alias_blank,"physical_row_undercount":undercount,"minimum_columns":minimum,"column_consistent":kept,"attempted_line_rect_grid_conflict":conflict,"resolved_rect_undercount":resolved}),
    )
}
