//! 旧规则表候选完整普通组装：核心/表题/自动表注、成员分离和方向还原。
use super::path_graphics::{array, bbox, integer, num, union};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn sub(a: &Value, b: &Value) -> Value {
    if let (Some(a), Some(b)) = (a.as_i64(), b.as_i64()) {
        json!(a - b)
    } else {
        json!(a.as_f64().unwrap() - b.as_f64().unwrap())
    }
}
pub(super) fn rotate(b: &[Value; 4], p: &[Value], angle: i64) -> Value {
    match angle {
        270 => json!([b[1], sub(&p[1], &b[2]), b[3], sub(&p[1], &b[0])]),
        90 => json!([sub(&p[0], &b[3]), b[0], sub(&p[0], &b[1]), b[2]]),
        180 => json!([
            sub(&p[0], &b[2]),
            sub(&p[1], &b[3]),
            sub(&p[0], &b[0]),
            sub(&p[1], &b[1])
        ]),
        _ => json!(b),
    }
}
pub fn expand(v: &Value) -> Result<Value, Error> {
    expand_with_work(v, &mut Work(0))
}
pub(crate) fn expand_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    expand_with_core(v, None, work)
}
pub(super) fn expand_with_core(
    v: &Value,
    core_rows: Option<&[Value]>,
    work: &mut Work,
) -> Result<Value, Error> {
    expand_with_inputs(v, core_rows, None, None, None, None, work)
}
pub(super) fn expand_with_inputs(
    v: &Value,
    core_rows: Option<&[Value]>,
    prepared_core: Option<&[&super::rule_annotation_preparation::Row]>,
    rows_override: Option<&[super::rule_annotation_preparation::Row]>,
    lines_override: Option<&[&Value]>,
    notes_cache: Option<&mut super::table_footnotes::FootnoteCache>,
    work: &mut Work,
) -> Result<Value, Error> {
    validate_tree(v)?;
    let page = array(&v["page_size"], 2)?;
    if page.len() != 2 {
        return Err(invalid());
    }
    page.iter().map(num).collect::<Result<Vec<_>, _>>()?;
    let angle = integer(v, "angle", 0)?;
    let possible = v
        .get("has_possible_table_notes")
        .map_or(Ok(true), |v| v.as_bool().ok_or_else(invalid))?;
    let rows = if rows_override.is_some() {
        &[][..]
    } else {
        array(&v["rows"], 4096)?
    };
    let owned_lines;
    let lines = if let Some(lines) = lines_override {
        lines
    } else {
        owned_lines = array(&v["lines"], 4096)?.iter().collect::<Vec<_>>();
        &owned_lines
    };
    let prep_request = value_object!({"rows":rows,"rule_bboxes":v["rule_bboxes"],"core_row_indices":v["core_row_indices"],"median_height":v["median_height"],"caption_source_index":v["caption_source_index"],"footnote_row_indices":json!([])});
    let prep = super::rule_annotation_preparation::prepare_with_rows(
        &prep_request,
        core_rows,
        rows_override,
        prepared_core,
        work,
    )?;
    let notes = if possible {
        let request = value_object!({"rows":rows,"rule_bbox":prep["rule_bbox"],"core_line_indices":prep["core_line_indices"],"page_size":page,"angle":angle,"median_height":v["median_height"]});
        super::table_footnotes::collect_cached(
            &request,
            rows_override,
            Some(lines),
            notes_cache,
            work,
        )?
    } else {
        json!({"footnote_rows":[],"annotation":null})
    };
    let mut annotations = prep["annotations"].as_array().unwrap().clone();
    if !notes["annotation"].is_null() {
        annotations.push(notes["annotation"].clone());
    }
    let core = bbox(&prep["core_local_bbox"])?;
    // Reference path includes caption/core/footnote rows, then keeps core first on ties.
    let mut included = Vec::new();
    for r in prep["caption_rows"].as_array().unwrap() {
        included.push(bbox(&r["bbox"])?);
    }
    if let Some(core) = prepared_core {
        included.extend(core.iter().map(|r| r.b.clone()));
    } else if let Some(core_rows) = core_rows {
        for r in core_rows {
            included.push(bbox(&r["bbox"])?);
        }
    } else {
        for i in array(&v["core_row_indices"], 4096)? {
            let i = i
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
                .filter(|i| *i < rows.len())
                .ok_or_else(invalid)?;
            included.push(bbox(&rows[i]["bbox"])?);
        }
    }
    for r in notes["footnote_rows"].as_array().unwrap() {
        included.push(bbox(&r["bbox"])?);
    }
    work.charge(included.len() + 1)?;
    let mut outer = included.first().ok_or_else(invalid)?.clone();
    for b in &included[1..] {
        outer = union(outer, b);
    }
    let local = union(core.clone(), &outer);
    let mut ids: BTreeSet<i64> = prep["core_line_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    for a in &annotations {
        let members = a["line_indices"].as_array().unwrap();
        work.charge(members.len() * 16 + 1)?;
        for i in members {
            ids.remove(&i.as_i64().unwrap());
        }
    }
    let candidate = json!({"bbox":rotate(&local.raw,page,angle),"local_bbox":local.raw,"angle":angle,"score":0.0,"core_bbox":rotate(&core.raw,page,angle),"line_indices":ids,"annotations":annotations,"inferred_grid":[],"preserve_inline_font_styles":false,"inferred_grid_authoritative":false});
    let out = json!({"candidate":candidate});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_full_rule_candidate_expansion() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-rule-candidate-expansion.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                expand(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn shared_budget() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-rule-candidate-expansion.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let q = &f["cases"][0]["request"];
        let mut work = Work(0);
        assert!(expand_with_work(q, &mut work).is_ok());
        let cost = work.0;
        assert!(expand_with_work(q, &mut Work(10_000_000 - cost + 1)).is_err());
        assert!(expand_with_work(q, &mut Work(0)).is_ok());
    }
}
