//! Original post-container caption/title/quotation/inline-prose classifications.
mod boundaries;
mod cjk;
mod display;
mod prose;
use super::{
    native_precontainers::bold,
    native_vector_formulas::english_words,
    page_containers::{combined, height, median, optional, size, strong_caption},
    path_graphics::{array, bbox, integer, num, Box4},
};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use regex::Regex;
use serde_json::{json, Value};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};
const SPACE: &str = r"[\u{9}-\u{d}\u{1c}-\u{20}\u{85}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}]";
fn text(l: &Value) -> Result<&str, Error> {
    l["text"].as_str().ok_or_else(invalid)
}
fn trim(s: &str) -> &str {
    s.trim_matches(python_space)
}
fn words(s: &str) -> usize {
    s.split(python_space).filter(|s| !s.is_empty()).count()
}
fn em(l: &Value) -> Result<f64, Error> {
    l.get("effective_height").map_or(Ok(0.), num)
}
fn weight(l: &Value, d: f64) -> Result<f64, Error> {
    if l["dominant_font_weight"].is_null() {
        Ok(d)
    } else {
        let n = num(&l["dominant_font_weight"])?;
        Ok(if n == 0. { d } else { n })
    }
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) * 0.5
}
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) * 0.5
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let d = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if d > 0. {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / d
    } else {
        0.
    }
}
fn cjk(c: char) -> bool {
    ('\u{3400}'..='\u{9fff}').contains(&c)
}
fn mapped(s: &str) -> String {
    s.chars()
        .map(|c| python_decimal(c).map_or(c, |d| (b'0' + d) as char))
        .collect()
}
fn ordered(mut ids: Vec<usize>, bs: &[Box4]) -> Vec<usize> {
    ids.sort_by(|&i, &j| {
        bs[i].n[1]
            .total_cmp(&bs[j].n[1])
            .then(bs[i].n[0].total_cmp(&bs[j].n[0]))
    });
    ids
}
fn bounds(ids: &[usize], bs: &[Box4]) -> Result<Box4, Error> {
    combined(ids.iter().map(|&i| &bs[i]))
}
fn next_group(ls: &[Value]) -> Result<i64, Error> {
    ls.iter()
        .filter(|l| !l["paragraph_group"].is_null())
        .map(|l| integer(l, "paragraph_group", 0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(-1)
        .checked_add(1)
        .ok_or_else(limit)
}
fn band(ids: &[usize], ls: &mut [Value]) -> Result<(), Error> {
    let id = ids
        .iter()
        .map(|&i| integer(&ls[i], "source_index", 0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .ok_or_else(invalid)?;
    let group = next_group(ls)?;
    for &i in ids {
        ls[i]["semantic_type"] = json!("paragraph_title");
        ls[i]["structural_title"] = json!(true);
        ls[i]["explicit_section_title"] = json!(true);
        ls[i]["title_band_id"] = json!(id);
        ls[i]["paragraph_group"] = json!(group);
        ls[i]["title_suppressed"] = json!(false);
    }
    Ok(())
}
fn mark(ids: impl IntoIterator<Item = usize>, ls: &mut [Value], group: i64) {
    for i in ids {
        ls[i]["semantic_type"] = json!("text");
        ls[i]["title_suppressed"] = json!(true);
        ls[i]["paragraph_group"] = json!(group);
    }
}
pub fn classify(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let out = classify_with_work(v, &mut Work(0))?;
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub(super) fn classify_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    classify_lines_owned(array(&v["lines"], 4096)?.to_vec(), v, w)
}
pub(super) fn classify_lines_owned(
    ls: Vec<Value>,
    v: &Value,
    w: &mut Work,
) -> Result<Value, Error> {
    classify_lines_with_reference(ls, v, None, w)
}
pub(super) fn classify_lines_with_reference(
    ls: Vec<Value>,
    v: &Value,
    reference: Option<&[Value]>,
    w: &mut Work,
) -> Result<Value, Error> {
    classify_reference_mode(ls, v, reference, w, true)
}
fn classify_reference_mode(
    mut ls: Vec<Value>,
    v: &Value,
    reference: Option<&[Value]>,
    w: &mut Work,
    compact: bool,
) -> Result<Value, Error> {
    let sz = size(&v["page_size"])?;
    let bs = ls
        .iter()
        .map(|l| bbox(&l["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    for l in &mut ls {
        w.charge(text(l)?.len() + 1)?;
        if integer(l, "angle", 0)? == 0 && strong_caption(text(l)?) {
            l["caption_start"] = json!(true)
        }
    }
    boundaries::bilingual(&mut ls, &bs, w)?;
    boundaries::appendix(&mut ls, &bs, sz, w)?;
    prose::quotes(&mut ls, &bs, w)?;
    prose::inline(&mut ls, &bs, sz, optional(v, "inline_regions", 4000)?, w)?;
    let reference = match reference {
        Some(rows) => {
            if rows.len() > rapiddoc_core::resources::limit(4096) {
                return Err(limit());
            }
            rows
        }
        None => optional(v, "reference_lines", 4096)?,
    };
    let mut reference: Vec<Cow<'_, Value>> = reference.iter().map(Cow::Borrowed).collect();
    for alias in optional(v, "reference_aliases", 4096)? {
        let r = alias["reference_index"].as_u64().ok_or_else(invalid)? as usize;
        let i = alias["line_index"].as_u64().ok_or_else(invalid)? as usize;
        let l = ls.get(i).ok_or_else(invalid)?;
        let target = reference.get_mut(r).ok_or_else(invalid)?;
        // Display freezes membership/text/font at this checkpoint, then follows
        // aliased rows for later geometry/semantic changes. Characters and other
        // line fields are never inspected in the frozen reference snapshot.
        *target = Cow::Owned(if compact {
            let mut snapshot = serde_json::Map::new();
            for key in ["text", "angle", "effective_height", "font_signature"] {
                if let Some(value) = l.get(key) {
                    snapshot.insert(key.into(), value.clone());
                }
            }
            snapshot.insert("_role_alias".into(), json!(i));
            Value::Object(snapshot)
        } else {
            let mut snapshot = l.clone();
            snapshot["_role_alias"] = json!(i);
            snapshot
        });
    }
    display::classify(
        &mut ls,
        &bs,
        sz,
        optional(v, "visual_bboxes", 5000)?,
        optional(v, "table_bboxes", 5000)?,
        integer(v, "page_index", 0)?,
        &reference,
        w,
    )?;
    let mut containers = optional(v, "visual_bboxes", 5000)?.to_vec();
    containers.extend_from_slice(optional(v, "table_bboxes", 5000)?);
    if containers.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    cjk::restore(
        &mut ls,
        &bs,
        sz,
        &containers,
        integer(v, "page_index", 0)?,
        w,
    )?;
    Ok(Value::Array(ls))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_role_reference_preserves_aliases_and_work() {
        let f: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!("../../../../../../tests/fixtures/native_owned_line_reference.json.gz")
                [..],
        ))
        .unwrap();
        let aliases: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../../tests/fixtures/native_role_snapshot_reference.json.gz"
            )[..],
        ))
        .unwrap();
        for host in [false, true] {
            let _scope = host.then(crate::legacy_page::budget::host_page);
            for c in f["roles"]
                .as_array()
                .unwrap()
                .iter()
                .chain(aliases["cases"].as_array().unwrap())
            {
                let q = &c["request"];
                let (mut a, mut b) = (Work(0), Work(0));
                let old = classify_reference_mode(
                    q["lines"].as_array().unwrap().clone(),
                    q,
                    None,
                    &mut a,
                    false,
                );
                let new = classify_reference_mode(
                    q["lines"].as_array().unwrap().clone(),
                    q,
                    None,
                    &mut b,
                    true,
                );
                assert_eq!(a.0, b.0, "{}", c["label"]);
                match (old, new) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(a, b, "{}", c["label"]);
                        assert_eq!(b, c["expected"], "{}", c["label"]);
                    }
                    (Err(a), Err(b)) => {
                        assert_eq!(a.code, b.code);
                        assert_eq!(a.message, b.message);
                    }
                    other => panic!("{}: {other:?}", c["label"]),
                }
            }
        }
    }
    #[test]
    fn old_post_container_roles() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-text-roles.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = classify(&c["request"]);
            if !c["error"].is_null() {
                assert!(out.is_err(), "{}", c["label"]);
                continue;
            }
            let out = out.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            assert!(
                eq(&out, &c["expected"]),
                "{}\nactual: {out}\nexpected: {}",
                c["label"],
                c["expected"]
            );
        }
    }
    #[test]
    fn old_post_text_document() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-post-text-document.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = super::super::native_document::prepare_pages(&c["request"]);
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
    fn role_source_identity() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-text-role-aliases-canonical.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for key in ["cases", "pages"] {
            for c in f[key].as_array().unwrap() {
                let out = if key == "cases" {
                    classify(&c["request"])
                } else {
                    super::super::native_geometry::finish_page(&c["request"])
                }
                .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                if let Some(d) =
                    super::super::native_document::tests::diff(&out, &c["expected"], "")
                {
                    panic!("{}: {d}", c["label"])
                }
            }
        }
    }
}
