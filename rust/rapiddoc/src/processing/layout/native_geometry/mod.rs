//! 原文档几何计划应用、校准字符重拆及样式/链接证据分片。
mod evidence;
mod planner;
mod runs;
use super::{
    native_lines,
    page_containers::{coerce, optional, size},
    path_graphics::{array, bbox, integer, num},
    table_text_styles::preparation,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeMap;
#[derive(Clone)]
struct Resplit {
    source: Value,
    members: Vec<Value>,
    positions: Vec<Vec<usize>>,
}
pub fn build_document_plan(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    bounded(planner::build(v, &mut Work(0))?)
}
/// Original document prefix, ending before page-marginal and Form classifiers.
/// The same shared work budget covers plan generation and pre-container repairs.
pub fn prepare_document(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_document_with_work(v, &mut Work(0))
}
pub(super) fn prepare_document_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_document_owned(v.clone(), w)
}
pub(super) fn prepare_document_owned(mut v: Value, w: &mut Work) -> Result<Value, Error> {
    let mut profile = crate::processing::NativeStageProfile::new();
    let mut result = planner::build(&v, w)?;
    profile.mark("build_geometry_plan");
    array(&v["pages"], 1000)?;
    let Value::Array(mut pages) = v["pages"].take() else {
        return Err(invalid());
    };
    drop(v);
    for (index, page) in pages.iter_mut().enumerate() {
        page["source"]["lines"] = apply_plan(
            &page["source"]["lines"],
            &result["plan"],
            index as i64,
            false,
            w,
        )?;
        page["geometry"]["loose_bboxes"] = result["remaining_loose_bboxes"][index].clone();
    }
    let mut out = value_object!({"plan":result["plan"].take()});
    profile.mark("apply_geometry_plan");
    out["pages"] = Value::Array(pages);
    bounded(out)
}
fn index<'a>(
    plan: &'a Value,
    key: &str,
    page: i64,
    w: &mut Work,
) -> Result<BTreeMap<i64, &'a Value>, Error> {
    let mut out = BTreeMap::new();
    for item in optional(plan, key, 20000)? {
        w.charge(1)?;
        if integer(item, "page_index", 0)? == page {
            out.insert(integer(item, "source_index", 0)?, item);
        }
    }
    Ok(out)
}
pub fn apply(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    bounded(apply_with_work(v, &mut Work(0))?)
}
fn bounded(value: Value) -> Result<Value, Error> {
    if crate::legacy_page::budget::output_exceeds(&value) {
        Err(limit())
    } else {
        Ok(value)
    }
}
fn apply_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let page = integer(v, "page_index", 0)?;
    let allow = v["allow_y_trim"].as_bool().ok_or_else(invalid)?;
    apply_plan(&v["lines"], &v["plan"], page, allow, w)
}
fn apply_plan(
    lines: &Value,
    plan: &Value,
    page: i64,
    allow: bool,
    w: &mut Work,
) -> Result<Value, Error> {
    let repairs = index(plan, "line_repairs", page, w)?;
    let inks = index(plan, "line_ink_bboxes", page, w)?;
    let baselines = index(plan, "line_baselines", page, w)?;
    let scales = index(plan, "line_style_scales", page, w)?;
    let local = index(plan, "local_metric_repair_sources", page, w)?;
    let inflated = optional(plan, "style_inflated_runs", 20000)?;
    let mut lines = array(lines, 4096)?.to_vec();
    for line in &mut lines {
        let id = integer(line, "source_index", 0)?;
        bbox(&line["bbox"])?;
        if let Some(entry) = inks.get(&id) {
            bbox(&entry["bbox"])?;
            line["ink_bbox"] = entry["bbox"].clone()
        }
        if let Some(entry) = baselines.get(&id) {
            num(&entry["baseline"])?;
            line["baseline"] = entry["baseline"].clone()
        }
        let scale = scales
            .get(&id)
            .map(|entry| num(&entry["scale"]))
            .transpose()?;
        if let Some(scale) = scale {
            line["em_height"] = json!(scale)
        }
        let em = line.get("em_height").map_or(Ok(0.), num)?;
        let mut style = false;
        if !line["font_signature"].is_null() && em > 0. {
            let sig = array(&line["font_signature"], 2)?;
            if sig.len() != 2 {
                return Err(invalid());
            }
            let raw = sig[0].as_str().ok_or_else(invalid)?;
            let name = preparation::family(&Some((raw.to_owned(), sig[1].clone())))
                .unwrap_or_else(|| "<unknown>".into());
            let style_size = (em * 4.).round_ties_even() / 4.;
            for run in inflated {
                w.charge(1)?;
                let a = array(run, 6)?;
                if a.len() != 6 {
                    return Err(invalid());
                }
                if a[0] == name
                    && (style_size - num(&a[1])?).abs() <= 0.25
                    && crate::legacy_page::eq(&sig[1], &a[2])
                    && integer(line, "angle", 0)? == a[4].as_i64().ok_or_else(invalid)?
                {
                    style = true;
                    break;
                }
            }
        }
        line["style_scale_repaired"] = json!(local.contains_key(&id) || style);
        let Some(repair) = repairs.get(&id) else {
            continue;
        };
        let state = repair["state"].as_str().ok_or_else(invalid)?;
        let is_local = local.contains_key(&id);
        if state == "trim_y" && !allow && !is_local {
            continue;
        }
        let mut layout = repair["layout_bbox"].clone();
        bbox(&layout)?;
        bbox(&repair["source_bbox"])?;
        let local_state = if state == "repair_xy" && !allow && !is_local {
            layout[1] = repair["source_bbox"][1].clone();
            layout[3] = repair["source_bbox"][3].clone();
            "repair_x"
        } else {
            state
        };
        line["source_bbox"] = repair["source_bbox"].clone();
        line["ink_bbox"] = repair["ink_bbox"].clone();
        if !repair["baseline"].is_null() {
            num(&repair["baseline"])?;
            line["baseline"] = repair["baseline"].clone()
        }
        line["geometry_state"] = json!(local_state);
        num(&repair["confidence"])?;
        line["geometry_confidence"] = repair["confidence"].clone();
        line["split_y_candidate"] =
            json!(repair["split_y_candidate"].as_bool().ok_or_else(invalid)?);
        line["bbox"] = layout;
        let height = num(&repair["em_height"])?;
        line["em_height"] = json!(scale.unwrap_or(height));
        if (allow || is_local) && ["trim_y", "repair_xy"].contains(&state) {
            line["effective_height"] = json!(height)
        }
    }
    Ok(Value::Array(lines))
}
pub fn resplit(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    bounded(resplit_with_work(v, &mut Work(0))?)
}
fn resplit_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let (lines, resplits, source_lines) = runs::resplit(v, w)?;
    let (styles, links) = evidence::realign(v, &resplits, w)?;
    let maps = resplits
        .into_iter()
        .map(|(id, r)| {
            let mut entry = json!({"source_index":id,"member_char_positions":r.positions});
            entry["source"] = r.source;
            entry["members"] = Value::Array(r.members);
            entry
        })
        .collect();
    let mut out = json!({});
    out["lines"] = Value::Array(lines);
    out["source_lines"] = Value::Array(source_lines);
    out["resplits"] = Value::Array(maps);
    out["style_lines"] = Value::Array(styles);
    out["link_lines"] = Value::Array(links);
    Ok(out)
}
/// Resume the original post-container geometry sequence using a document plan.
/// Plan construction remains a separate document-wide stage.
pub fn finish_page(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    finish_page_with_work(v, &mut Work(0))
}
pub(super) fn finish_page_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    finish_page_with_plan(v, &v["plan"], w)
}
pub(super) fn finish_page_with_plan(v: &Value, plan: &Value, w: &mut Work) -> Result<Value, Error> {
    finish_page_core(v["prepared"].clone(), v, plan, w)
}
pub(super) fn finish_page_owned(mut v: Value, plan: &Value, w: &mut Work) -> Result<Value, Error> {
    let prepared = v["prepared"].take();
    finish_page_core(prepared, &v, plan, w)
}
fn finish_page_core(mut out: Value, v: &Value, plan: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut profile = crate::processing::NativeStageProfile::new();
    let page = v.get("page_index").cloned().unwrap_or(json!(0));
    let source = &out["source"];
    let mut next = optional(source, "lines", 4096)?
        .iter()
        .map(|l| integer(l, "source_index", 0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(-1)
        .checked_add(1)
        .ok_or_else(limit)?;
    let mut detached = std::collections::BTreeSet::new();
    if !plan.is_null() {
        let repaired = apply_plan(
            &out["remaining_lines"],
            plan,
            page.as_i64().ok_or_else(invalid)?,
            true,
            w,
        )?;
        let mut line_boxes = optional(source, "lines", 4096)?
            .iter()
            .map(|l| Ok((integer(l, "source_index", 0)?, l["bbox"].clone())))
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        for l in array(&repaired, 4096)? {
            line_boxes.insert(integer(l, "source_index", 0)?, l["bbox"].clone());
        }
        let mut visual = Vec::new();
        for c in optional(plan, "char_repairs", 20000)? {
            if c["page_index"] == page {
                visual.push(value_object!({"char_idx":c["char_idx"],"bbox":c["layout_bbox"]}));
            }
        }
        let mut request = value_object!({"page_size":source["page_size"],"visual_bboxes":visual,"source_index_start":next,"line_bboxes":line_boxes.iter().map(|(id,b)|value_object!({"source_index":id,"bbox":b})).collect::<Vec<_>>(),"style_lines":v.get("style_lines").cloned().unwrap_or(json!([])),"link_lines":v.get("link_lines").cloned().unwrap_or(json!([]))});
        request["lines"] = repaired;
        let mut geometry = resplit_with_work(&request, w)?;
        profile.mark("finish_geometry_resplit");
        next = next.max(
            array(&geometry["lines"], 4096)?
                .iter()
                .map(|l| integer(l, "source_index", 0))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .unwrap_or(-1)
                .checked_add(1)
                .ok_or_else(limit)?,
        );
        out["remaining_lines"] = geometry["lines"].take();
        for entry in array(&geometry["resplits"], 4096)? {
            detached.insert(integer(entry, "source_index", 0)?);
        }
        let updated = array(&geometry["source_lines"], 4096)?
            .iter()
            .map(|l| Ok((integer(l, "source_index", 0)?, l)))
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        optional(&out["source"], "lines", 4096)?;
        for l in out["source"]["lines"].as_array_mut().unwrap() {
            if let Some(repaired) = updated.get(&integer(l, "source_index", 0)?) {
                *l = (**repaired).clone();
            }
        }
        out["style_lines"] = geometry["style_lines"].take();
        out["link_lines"] = geometry["link_lines"].take();
    } else {
        for key in ["style_lines", "link_lines"] {
            out[key] = v.get(key).cloned().unwrap_or(json!([]));
        }
    }
    out["source_index_start"] = json!(next);
    profile.mark("finish_geometry_updates");
    if v["include_post_text_roles"] == true
        || v["include_post_footnotes"] == true
        || v["prepare_page"] == true
    {
        super::page_containers::finish_calibrated_page_owned(
            out,
            w,
            &detached,
            Some((
                page.as_i64().ok_or_else(invalid)?,
                plan["document_style_anomaly"] == true,
                v["include_post_footnotes"] == true || v["prepare_page"] == true,
                if v["prepare_page"] == true {
                    Some(v)
                } else {
                    None
                },
            )),
        )
    } else {
        super::page_containers::finish_calibrated_page_owned(out, w, &detached, None)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn diff(a: &Value, b: &Value, path: &str) -> Option<String> {
        if a == b {
            return None;
        }
        match (a, b) {
            (Value::Object(a), Value::Object(b)) => {
                for k in a.keys().chain(b.keys()) {
                    if !a.contains_key(k) || !b.contains_key(k) {
                        return Some(format!("{path}/{k}: missing"));
                    }
                    if let Some(d) = diff(&a[k], &b[k], &format!("{path}/{k}")) {
                        return Some(d);
                    }
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Some(format!("{path}: length {} vs {}", a.len(), b.len()));
                }
                for (i, (a, b)) in a.iter().zip(b).enumerate() {
                    if let Some(d) = diff(a, b, &format!("{path}/{i}")) {
                        return Some(d);
                    }
                }
            }
            _ => return Some(format!("{path}: {a} vs {b}")),
        }
        None
    }
    #[test]
    fn old_document_geometry_prefix() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-geometry-prefix.json");
        let fixtures: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in fixtures["cases"].as_array().unwrap() {
            let out =
                prepare_document(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"]);
            }
        }
    }
    #[test]
    fn old_native_document_geometry_plan() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-geometry-plans.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = build_document_plan(&c["request"])
                .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"]);
            }
        }
        let edges: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-geometry-plan-edges.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in edges["cases"].as_array().unwrap() {
            let out = build_document_plan(&c["request"])
                .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"]);
            }
        }
        let proofs: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-geometry-plan-proofs.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in proofs["cases"].as_array().unwrap() {
            let out = build_document_plan(&c["request"])
                .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"]);
            }
        }
        let fonts: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-geometry-font-contract.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in fonts["cases"].as_array().unwrap() {
            let out = build_document_plan(&c["request"]);
            if c.get("error").is_some() {
                assert!(out.is_err(), "{}", c["label"]);
                continue;
            }
            let out = out.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"]);
            }
        }
    }
    #[test]
    fn old_native_geometry_aliases() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-geometry-aliases.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let actual = finish_page(&c["request"]).unwrap();
            if let Some(d) = diff(&actual, &c["expected"], "") {
                panic!("{}: {d}", c["label"]);
            }
        }
    }
    #[test]
    fn old_native_geometry_pages() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-geometry-pages.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let actual =
                finish_page(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = diff(&actual, &c["expected"], "") {
                panic!("{}: {d}", c["label"]);
            }
        }
    }
    #[test]
    fn old_native_geometry() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-geometry.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for key in ["apply", "resplit"] {
            for c in f[key].as_array().unwrap() {
                let out = if key == "apply" {
                    apply(&c["request"])
                } else {
                    resplit(&c["request"])
                }
                .unwrap_or_else(|e| panic!("{key}/{}: {e}", c["label"]));
                if let Some(d) = diff(&out, &c["expected"], "") {
                    panic!("{key}/{}: {d}", c["label"]);
                }
            }
        }
    }
}
