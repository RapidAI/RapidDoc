//! Original document reference-region events and hanging-entry grouping.
mod groups;
mod regions;
mod text;
use super::{
    page_containers::{median, optional, size},
    path_graphics::{array, bbox, integer, num, Box4},
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) * 0.5
}
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) * 0.5
}
fn axis(a: &Box4, b: &Box4) -> f64 {
    let den = (a.n[2] - a.n[0]).min(b.n[2] - b.n[0]);
    if den <= 0. {
        0.
    } else {
        (a.n[2].min(b.n[2]) - a.n[0].max(b.n[0])).max(0.) / den
    }
}
fn height(l: &Value, b: &Box4) -> Result<f64, Error> {
    super::native_titles::reference_height(l, b)
}
fn ls(p: &Value) -> Result<&[Value], Error> {
    array(&p["remaining_lines"], 4096)
}
fn sz(p: &Value) -> Result<[f64; 2], Error> {
    size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))
}
fn txt(l: &Value) -> Result<&str, Error> {
    l["text"].as_str().ok_or_else(invalid)
}
fn sid(l: &Value) -> Result<i64, Error> {
    integer(l, "source_index", 0)
}
fn edge(l: &Value) -> bool {
    matches!(
        l["semantic_type"].as_str(),
        Some("header" | "footer" | "page_number")
    )
}
fn prune(p: &mut Value, claimed: &BTreeSet<i64>) -> Result<(), Error> {
    let mut out = Vec::new();
    for group in optional(p, "page_footnote_groups", 4096)? {
        let ids = array(group, 4096)?
            .iter()
            .map(|v| v.as_i64().ok_or_else(invalid))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|i| !claimed.contains(i))
            .collect::<BTreeSet<_>>();
        if !ids.is_empty() {
            out.push(json!(ids))
        }
    }
    p["page_footnote_groups"] = json!(out);
    Ok(())
}
fn corridors(lines: &[Value], p: &Value, w: &mut Work) -> Result<Vec<(f64, f64)>, Error> {
    let size = sz(p)?;
    let containers = optional(p, "fixed_blocks", 4000)?
        .iter()
        .map(|b| bbox(&b["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut free = Vec::new();
    for l in lines {
        w.charge(containers.len() + 1)?;
        let b = bbox(&l["bbox"])?;
        if !containers
            .iter()
            .any(|c| super::path_graphics::overlap_first(&b, c) >= 0.8)
        {
            free.push(super::native_titles::reference_lane_line(l))
        }
    }
    let lanes = super::page_containers::layout_lanes(&free, size, w)?;
    let mut out = Vec::new();
    for l in &lanes {
        let b = bbox(&json!([num(&l["left"])?, 0., num(&l["right"])?, 1.]))?;
        let c = super::page_containers::layout_corridor(&b, &lanes, size[0]).ok_or_else(invalid)?;
        if !out.contains(&c) {
            out.push(c)
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    Ok(out)
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0))
}
pub(super) fn prepare_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_owned(v.clone(), w)
}
pub(super) fn prepare_owned(mut v: Value, w: &mut Work) -> Result<Value, Error> {
    array(&v["pages"], 1000)?;
    let Value::Array(mut pages) = v["pages"].take() else {
        return Err(invalid());
    };
    drop(v);
    regions::mark(&mut pages, w)?;
    for (i, p) in pages.iter_mut().enumerate() {
        if p["numbered_references"] == true {
            groups::group(p, w)?
        }
        super::native_titles::compact_reference_titles(p, i as i64, w)?;
    }
    let mut out = json!({});
    out["pages"] = Value::Array(pages);
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
/// The same original grouping is called again by final page assembly, including author-year regions.
pub fn group(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut p = v["page"].clone();
    groups::group(&mut p, &mut Work(0))?;
    Ok(p)
}
pub(super) fn group_with_work(p: &mut Value, w: &mut Work) -> Result<(), Error> {
    groups::group(p, w)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn check(name: &str, f: fn(&Value) -> Result<Value, Error>) {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(format!("../../build/migration/{name}.json")),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let out = f(&c["request"]);
            if !c["error"].is_null() {
                assert!(out.is_err(), "{}", c["label"]);
                continue;
            }
            let out = out.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
        }
    }
    #[test]
    fn original_raw_source_reference_chain() {
        check(
            "native-reference-chain",
            super::super::native_document::prepare_references,
        )
    }
    #[test]
    fn original_reference_document() {
        check("native-reference-document-canonical", prepare)
    }
    #[test]
    fn original_reference_groups() {
        check("native-reference-groups", group)
    }
    #[test]
    fn original_reference_text() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-reference-text.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let s = c["text"].as_str().unwrap();
            assert_eq!(json!(text::heading(s)), c["heading"], "heading: {s:?}");
            assert_eq!(
                json!(text::number(s).map(|(n, b)| json!([n, b]))),
                c["number"],
                "number: {s:?}"
            );
        }
    }
}
