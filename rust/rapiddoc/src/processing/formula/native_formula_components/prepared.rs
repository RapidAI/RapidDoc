//! Exact final-page prefix through fallback index extraction and semantic separation.
use super::super::{native_page_preparation as prep, page_containers as containers};
use super::*;
use std::collections::BTreeMap;
fn page_size(v: &Value) -> Result<[f64; 2], Error> {
    size(v.get("page_size").unwrap_or(&v["source"]["page_size"]))
}
fn rules(v: &Value) -> Result<&[Value], Error> {
    if v.get("drawing_lines").is_some() {
        optional(v, "drawing_lines", 5000)
    } else {
        optional(&v["source"], "drawing_lines", 5000)
    }
}
fn request(v: &Value, lines: Vec<Value>) -> Result<Value, Error> {
    Ok(
        json!({"lines":lines,"page_size":page_size(v)?,"table_bboxes":optional(v,"table_bboxes",5000)?,"drawing_lines":rules(v)?}),
    )
}
fn inventory(blocks: &[Value]) -> Result<Vec<[f64; 4]>, Error> {
    let mut out = Vec::new();
    for b in blocks {
        if b["type"] == "equation" && b["bbox"].as_array().is_some_and(|a| a.len() == 4) {
            out.push(bbox(&b["bbox"])?.n)
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(out)
}
fn canonical(v: &Value, excluded: &BTreeSet<i64>, w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut ls = Vec::new();
    for l in optional(v, "canonical_formula_source_lines", 4096)? {
        if !excluded.contains(&integer(l, "source_index", 0)?) {
            let mut l = l.clone();
            l["style_scale_repaired"] = json!(true);
            ls.push(l)
        }
    }
    let ls = prep::merge_with_work(&request(v, ls)?, w)?;
    let ls = array(&ls, 4096)?;
    let mut normal = ls
        .iter()
        .filter(|l| l["formula_candidate_only"] != true)
        .cloned()
        .collect::<Vec<_>>();
    normal.extend(
        ls.iter()
            .filter(|l| l["formula_candidate_only"] == true)
            .cloned(),
    );
    let out = input_with_work(&request(v, normal)?, w)?;
    Ok(array(&out["formula_blocks"], 4000)?.to_vec())
}
fn attach(blocks: &mut [Value], inks: &[Value], w: &mut Work) -> Result<(), Error> {
    for raw in inks {
        let ink = bbox(raw)?;
        let mut matches = Vec::new();
        for (i, block) in blocks.iter().enumerate() {
            w.charge(1)?;
            if block["type"] != "equation" || ![0, 180].contains(&integer(block, "angle", 0)?) {
                continue;
            }
            let b = bbox(&block["bbox"])?;
            let x = (ink.n[0] - b.n[2]).max(b.n[0] - ink.n[2]).max(0.);
            let y = (ink.n[1] - b.n[3]).max(b.n[1] - ink.n[3]).max(0.);
            if axis(&ink, &b, 1) >= 0.8
                && x.hypot(y) <= 0.25 * (ink.n[3] - ink.n[1])
                && ink.n[1] >= b.n[1] - 2.
                && ink.n[3] <= b.n[3] + 2.
            {
                matches.push(i)
            }
        }
        if matches.len() == 1 {
            let b = &mut blocks[matches[0]];
            b["bbox"] = json!(combined([&bbox(&b["bbox"])?, &ink])?.raw);
            if !b["_tight_output_bbox"].is_null() {
                b["_tight_output_bbox"] =
                    json!(combined([&bbox(&b["_tight_output_bbox"])?, &ink])?.raw)
            }
        }
    }
    Ok(())
}
pub(super) fn prepare(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let p = &v["page"];
    p.as_object().ok_or_else(invalid)?;
    let sz = page_size(p)?;
    let mut semantic = Vec::new();
    let mut unresolved = Vec::new();
    for l in array(&p["remaining_lines"], 4096)? {
        if l["semantic_type"].is_null() {
            unresolved.push(l.clone())
        } else {
            semantic.push(l.clone())
        }
    }
    let containers = optional(p, "fixed_blocks", 4000)?
        .iter()
        .map(|b| b["bbox"].clone())
        .collect::<Vec<_>>();
    let indices = containers::extract_index_with_work(
        &json!({"lines":unresolved,"page_size":sz,"container_bboxes":containers,"require_heading":true}),
        w,
    )?;
    let mut index_blocks = array(&indices["index_blocks"], 4000)?.to_vec();
    let mut input = Vec::new();
    for l in array(&indices["remaining_lines"], 4096)? {
        if l["semantic_type"].is_null() {
            input.push(l.clone())
        } else {
            semantic.push(l.clone())
        }
    }
    input.extend_from_slice(optional(p, "formula_candidate_lines", 4096)?);
    let mut styles = BTreeMap::new();
    for l in &input {
        styles.insert(
            integer(l, "source_index", 0)?,
            l["style_scale_repaired"].clone(),
        );
    }
    let canonical_geometry = p["canonical_formula_geometry"] == true;
    if canonical_geometry {
        for l in &mut input {
            l["style_scale_repaired"] = json!(true)
        }
    }
    let formulas = input_with_work(&request(p, input)?, w)?;
    let mut blocks = array(&formulas["formula_blocks"], 4000)?.to_vec();
    let mut remaining = array(&formulas["remaining_lines"], 4096)?.to_vec();
    if canonical_geometry && !optional(p, "canonical_formula_source_lines", 4096)?.is_empty() {
        let excluded = semantic
            .iter()
            .map(|l| integer(l, "source_index", 0))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let replay = canonical(p, &excluded, w)?;
        if inventory(&replay)? == inventory(&blocks)? {
            blocks = replay
        }
    }
    attach(&mut blocks, optional(p, "formula_ink_bboxes", 4096)?, w)?;
    if canonical_geometry {
        for l in &mut remaining {
            if let Some(style) = styles.get(&integer(l, "source_index", 0)?) {
                l["style_scale_repaired"] = style.clone()
            }
        }
    }
    let fallback = containers::extract_index_with_work(
        &json!({"lines":remaining,"page_size":sz,"container_bboxes":containers}),
        w,
    )?;
    index_blocks.extend_from_slice(array(&fallback["index_blocks"], 4000)?);
    remaining = Vec::new();
    for l in array(&fallback["remaining_lines"], 4096)? {
        if l["semantic_type"].is_null() {
            remaining.push(l.clone())
        } else {
            semantic.push(l.clone())
        }
    }
    let out = json!({"formula_blocks":blocks,"index_blocks":index_blocks,"semantic_lines":semantic,"remaining_lines":remaining,"canonical_formula_source_lines":[]});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
