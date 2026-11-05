//! 原容器前首张通讯脚注与附录标题分类。
mod headings;
use super::{
    page_containers::{combined, height, layout_corridor, layout_lanes, median, optional, size},
    path_graphics::{array, bbox, integer, num, Box4},
    table_block::output::field,
    table_text_styles::font_styles,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use regex::Regex;
use serde_json::{json, Value};
use std::{collections::BTreeSet, sync::OnceLock};
const SPACE: &str = r"[\u{9}-\u{d}\u{1c}-\u{20}\u{85}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}]";
fn text(l: &Value) -> Result<&str, Error> {
    l["text"].as_str().ok_or_else(invalid)
}
fn mapped(s: &str) -> String {
    s.chars()
        .map(|c| python_decimal(c).map_or(c, |d| (b'0' + d) as char))
        .collect()
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let d = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if d > 0. {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / d
    } else {
        0.
    }
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) * 0.5
}
fn words(s: &str) -> usize {
    s.split(python_space).filter(|s| !s.is_empty()).count()
}
pub(super) fn bold(l: &Value) -> Result<bool, Error> {
    let sig = &l["font_signature"];
    let (name, flags) = if sig.is_null() {
        (json!(""), json!(0))
    } else {
        let a = array(sig, 2)?;
        if a.len() != 2 {
            return Err(invalid());
        }
        (a[0].clone(), a[1].clone())
    };
    Ok(
        font_styles(
            &json!({"font":{"name":name,"flags":flags,"weight":l["dominant_font_weight"]}}),
        )
        .0,
    )
}
fn promote(l: &mut Value, id: i64) {
    l["semantic_type"] = json!("paragraph_title");
    l["structural_title"] = json!(true);
    l["explicit_section_title"] = json!(true);
    l["title_band_id"] = json!(id);
    l["title_suppressed"] = json!(false);
}
pub(super) fn correspondence(source: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut out = source.clone();
    let sz = size(&source["page_size"])?;
    let mut ls = array(&source["lines"], 4096)?.to_vec();
    let bs = ls
        .iter()
        .map(|l| bbox(&l["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut lower = Vec::new();
    let mut body = Vec::new();
    for (i, l) in ls.iter().enumerate() {
        w.charge(text(l)?.len() + 1)?;
        if integer(l, "angle", 0)? != 0 {
            continue;
        }
        if bs[i].n[1] >= 0.65 * sz[1] && l["semantic_type"].is_null() {
            lower.push(i)
        }
        if bs[i].n[1] > 0.2 * sz[1]
            && bs[i].n[1] < 0.75 * sz[1]
            && bs[i].n[2] - bs[i].n[0] > 0.3 * sz[0]
        {
            body.push(height(l, &bs[i])?)
        }
    }
    let contacts = lower
        .iter()
        .copied()
        .filter(|&i| field(ls[i]["text"].as_str().unwrap()) == Some("contact"))
        .collect::<Vec<_>>();
    if contacts.is_empty() || body.is_empty() {
        return Ok(out);
    }
    let em = median(body);
    let lanes = layout_lanes(&ls, sz, w)?;
    let mut ordered = lower.clone();
    ordered.sort_by(|&i, &j| {
        bs[i].n[1]
            .total_cmp(&bs[j].n[1])
            .then(bs[i].n[0].total_cmp(&bs[j].n[0]))
    });
    for anchor in contacts {
        let Some(corridor) = layout_corridor(&bs[anchor], &lanes, sz[0]) else {
            continue;
        };
        let mut fields = Vec::new();
        for &i in &lower {
            w.charge(lanes.len() + 1)?;
            let c = layout_corridor(&bs[i], &lanes, sz[0]);
            if field(text(&ls[i])?).is_some()
                && (c == Some(corridor) || c == Some((0., sz[0])))
                && (bs[i].n[1] - bs[anchor].n[1]).abs() <= 5. * em
            {
                fields.push(i)
            }
        }
        if fields.is_empty() {
            continue;
        }
        let top = fields
            .iter()
            .map(|&i| bs[i].n[1])
            .fold(f64::INFINITY, f64::min);
        let mut members = Vec::<usize>::new();
        for &i in &ordered {
            w.charge(1)?;
            let cx = (bs[i].n[0] + bs[i].n[2]) * 0.5;
            if bs[i].n[1] < top || cx < corridor.0 || cx > corridor.1 {
                continue;
            }
            if height(&ls[i], &bs[i])? > 0.85 * em {
                break;
            }
            if let Some(&previous) = members.last() {
                if bs[i].n[1] - bs[previous].n[3] > 1.5 * em || axis(&bs[anchor], &bs[i], 0) < 0.5 {
                    break;
                }
                static RE: OnceLock<Regex> = OnceLock::new();
                if RE
                    .get_or_init(|| Regex::new(&format!(r"^{SPACE}*[0-9]+[.)]{SPACE}+")).unwrap())
                    .is_match(&mapped(text(&ls[i])?))
                {
                    break;
                }
            }
            members.push(i)
        }
        let roles = members
            .iter()
            .filter_map(|&i| field(ls[i]["text"].as_str().unwrap()))
            .collect::<BTreeSet<_>>();
        if members.contains(&anchor) && roles.len() >= 2 {
            for i in members {
                ls[i]["semantic_type"] = json!("page_footnote")
            }
        }
    }
    out["lines"] = json!(ls);
    Ok(out)
}
pub(super) fn appendix_headings(
    source: &Value,
    containers: &[Value],
    w: &mut Work,
) -> Result<Value, Error> {
    headings::classify(source, containers, w)
}
pub(super) fn final_headings(
    source: &Value,
    containers: &[Value],
    w: &mut Work,
) -> Result<Value, Error> {
    headings::classify_mode(source, containers, w, false)
}
pub fn classify(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let w = &mut Work(0);
    let out = if v["kind"] == "correspondence" {
        correspondence(&v["source"], w)?
    } else if v["kind"] == "appendix" {
        appendix_headings(&v["source"], optional(v, "container_bboxes", 5000)?, w)?
    } else {
        return Err(invalid());
    };
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_precontainer_classifiers() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-precontainers.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let actual = classify(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            assert!(
                crate::legacy_page::eq(&actual, &c["expected"]),
                "{}\nactual: {actual}\nexpected: {}",
                c["label"],
                c["expected"]
            );
        }
    }
}
