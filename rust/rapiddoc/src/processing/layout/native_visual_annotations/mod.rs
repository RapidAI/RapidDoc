//! Existing visual_annotations rules, identity-indexed blocks and original relation arbitration.
mod bands;
mod candidates;
mod continuations;
mod regions;
mod relations;
mod xycut;
pub(super) fn recover_with_work(
    bs: &mut Vec<Value>,
    rules: &[Value],
    w: &mut Work,
) -> Result<(), Error> {
    bands::recover(bs, rules, w)
}
use super::{
    native_marginals::local,
    page_containers::{coerce, combined, median, optional, size, strong_caption},
    path_graphics::{array, num, Box4},
};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::{python_alnum, python_space};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use unicode_normalization::UnicodeNormalization;
#[derive(Clone)]
struct Parent {
    members: Vec<usize>,
    angle: i64,
    b: Box4,
    numeric: bool,
}
#[derive(Clone)]
struct Relation {
    parent: Parent,
    direction: &'static str,
    gap: f64,
    projection: f64,
    coverage: f64,
    offset: f64,
}
type Assignments = Vec<(usize, Relation)>;
type Candidates = Vec<(usize, &'static str)>;
fn truth(v: &Value) -> bool {
    super::native_text_assembly::python_truth(v)
}
fn text(b: &Value) -> &str {
    b["content"].as_str().unwrap_or("")
}
fn angle(b: &Value) -> Result<i64, Error> {
    super::native_final_output::block_angle(b)
}
fn local_box(b: &Value, p: [f64; 2], a: i64) -> Result<Option<Box4>, Error> {
    Ok(coerce(&b["bbox"])?.map(|b| local(&b, p, a)))
}
fn own_box(b: &Value, p: [f64; 2]) -> Result<Option<Box4>, Error> {
    local_box(b, p, angle(b)?)
}
fn height(b: &Value, p: [f64; 2]) -> Result<f64, Error> {
    let hs = optional(b, "_line_heights", 4096)?
        .iter()
        .filter(|n| n.is_number() || n.is_boolean())
        .map(|n| {
            if let Some(b) = n.as_bool() {
                Ok(f64::from(b))
            } else {
                num(n)
            }
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|&n| n > 0.)
        .collect::<Vec<_>>();
    if !hs.is_empty() {
        return Ok(median(hs));
    }
    let Some(bb) = own_box(b, p)? else {
        return Ok(1.);
    };
    let rows = b["_local_line_bboxes"]
        .as_array()
        .filter(|r| !r.is_empty())
        .map_or(1, Vec::len);
    Ok(((bb.n[3] - bb.n[1]) / rows as f64).max(1.))
}
fn first_row(b: &Value, p: [f64; 2], a: i64) -> Result<Option<Box4>, Error> {
    let mut rows = vec![];
    if let Some(rs) = b["_local_line_bboxes"].as_array() {
        for r in rs {
            if let Some(b) = coerce(r)? {
                rows.push(b)
            }
        }
    }
    rows.sort_by(|a, b| a.n[1].total_cmp(&b.n[1]).then(a.n[0].total_cmp(&b.n[0])));
    if rows.is_empty() {
        local_box(b, p, a)
    } else {
        Ok(Some(rows.remove(0)))
    }
}
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) * 0.5
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) * 0.5
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let min = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if min > 0. {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / min
    } else {
        0.
    }
}
fn regex(t: &str) -> regex::Regex {
    static R: std::sync::OnceLock<std::sync::Mutex<BTreeMap<String, regex::Regex>>> =
        std::sync::OnceLock::new();
    let mut r = R.get_or_init(Default::default).lock().unwrap();
    r.entry(t.to_owned())
        .or_insert_with(|| {
            regex::Regex::new(
                &t.replace("\\s", SPACE)
                    .replace("\\S", &format!("[^{}]", &SPACE[1..SPACE.len() - 1])),
            )
            .unwrap()
        })
        .clone()
}
const SPACE: &str = r"[\u{9}-\u{d}\u{1c}-\u{20}\u{85}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}]";
fn fold(t: &str) -> String {
    t.chars()
        .map(|c| match c {
            'İ' | 'ı' => 'i',
            'ſ' => 's',
            'K' => 'k',
            _ => c,
        })
        .collect()
}
fn normalized(t: &str) -> String {
    t.nfkc()
        .collect::<String>()
        .trim_matches(python_space)
        .to_owned()
}
fn words(t: &str) -> usize {
    regex(r"[A-Za-z]+").find_iter(t).count()
}
fn graph(t: &str) -> bool {
    let t = fold(t);
    regex(r"(?i)graph|chart|plot").find_iter(&t).any(|m| {
        let word = |c: char| c == '_' || python_alnum(c);
        !t[..m.start()].chars().next_back().is_some_and(word)
            && !t[m.end()..].chars().next().is_some_and(word)
    })
}
fn font_conflict(a: &Value, b: &Value) -> bool {
    let (Some(a), Some(b)) = (
        a["_font_signatures"].as_array(),
        b["_font_signatures"].as_array(),
    ) else {
        return false;
    };
    !a.is_empty() && !b.is_empty() && !a.iter().any(|a| b.iter().any(|b| eq(a, b)))
}
fn interval(b: &Value) -> Option<(f64, f64)> {
    let a = b["_lane_interval"].as_array()?;
    if a.len() != 2 {
        return None;
    }
    let n = |v: &Value| super::table_native_text::number(v).ok().flatten();
    let (l, r) = (n(&a[0])?, n(&a[1])?);
    (r > l).then_some((l, r))
}
fn distinct(a: &Value, b: &Value) -> bool {
    a["_lane_is_span"] == false
        && b["_lane_is_span"] == false
        && interval(a)
            .zip(interval(b))
            .is_some_and(|(a, b)| a.1 <= b.0 || b.1 <= a.0)
}
fn same_parent(a: &Parent, b: &Parent) -> bool {
    a.members == b.members && a.angle == b.angle && a.b.n == b.b.n && a.numeric == b.numeric
}
fn kind(cs: &Candidates, i: usize) -> Option<&'static str> {
    cs.iter().find(|(j, _)| *j == i).map(|(_, k)| *k)
}
fn assign(v: &mut Assignments, i: usize, r: Relation) {
    if let Some((_, old)) = v.iter_mut().find(|(j, _)| *j == i) {
        *old = r
    } else {
        v.push((i, r))
    }
}
fn merge_blocks(bs: &[Value], ids: &[usize]) -> Result<Value, Error> {
    super::native_text_assembly::merge_internal_blocks(bs, ids)
}
pub fn run(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let p = size(&v["page_size"])?;
    let bs = array(&v["blocks"], 4096)?.to_vec();
    let w = &mut Work(0);
    let mut cs = candidates::collect(&bs, w)?;
    let hs = cs
        .iter()
        .map(|(i, _)| height(&bs[*i], p))
        .collect::<Result<Vec<_>, _>>()?;
    let h = if hs.is_empty() { 8. } else { median(hs) };
    let mut parents = relations::parents(&bs, p, h, w)?;
    relations::table_panels(&bs, p, &mut parents, &mut cs, w)?;
    let out = json!({"candidates":cs.iter().map(|(i,k)|json!([i,k])).collect::<Vec<_>>(),"parents":parents.iter().map(parent_json).collect::<Vec<_>>(),"relations":cs.iter().map(|(i,k)|Ok(json!({"index":i,"kind":k,"relation":relations::choose(*i,k,&bs,p,&parents,&cs.iter().map(|(i,_)|*i).collect(),w)?.as_ref().map(relation_json)}))).collect::<Result<Vec<_>,Error>>()?});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
fn parent_json(p: &Parent) -> Value {
    json!({"member_indices":p.members,"angle":p.angle,"local_bbox":p.b.raw,"native_numeric_grid":p.numeric})
}
fn relation_json(r: &Relation) -> Value {
    json!({"parent":parent_json(&r.parent),"direction":r.direction,"normalized_gap":r.gap,"projection_overlap":r.projection,"annotation_coverage":r.coverage,"center_offset":r.offset})
}
pub fn bind(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    bind_with_work(v, &mut Work(0))
}
pub(super) fn bind_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let p = size(&v["page_size"])?;
    let mut bs = array(&v["blocks"], 4096)?.to_vec();
    let mut cs = candidates::collect(&bs, w)?;
    if cs.is_empty() {
        return Ok(json!({"blocks":bs,"regions":[]}));
    }
    let h = median(
        cs.iter()
            .map(|(i, _)| height(&bs[*i], p))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let mut parents = relations::parents(&bs, p, h, w)?;
    if parents.is_empty() {
        return Ok(json!({"blocks":bs,"regions":[]}));
    }
    relations::table_panels(&bs, p, &mut parents, &mut cs, w)?;
    let annotations = cs.iter().map(|(i, _)| *i).collect::<BTreeSet<_>>();
    let mut assignments = vec![];
    for &(i, k) in &cs {
        if let Some(r) = relations::choose(i, k, &bs, p, &parents, &annotations, w)? {
            assignments.push((i, r))
        }
    }
    // JSON carries parent identity as an original block index, never value equality.
    for &(i, k) in &cs {
        let Some(parent) = bs[i].get("_annotation_band_parent_index") else {
            continue;
        };
        let j = parent.as_u64().ok_or_else(invalid)? as usize;
        let Some(parent) = parents.iter().find(|p| p.members == vec![j]) else {
            continue;
        };
        let previous = assignments.iter().find(|(j, _)| *j == i).map(|(_, r)| r);
        if k == "caption"
            && parent.numeric
            && previous.is_some_and(|r| !r.parent.numeric)
            && graph(text(&bs[i]))
        {
            continue;
        }
        let pb = coerce(&bs[j]["bbox"])?.ok_or_else(invalid)?;
        let ab = coerce(&bs[i]["bbox"])?.ok_or_else(invalid)?;
        assign(
            &mut assignments,
            i,
            Relation {
                parent: parent.clone(),
                direction: if ab.n[3] <= pb.n[1] { "above" } else { "below" },
                gap: 0.,
                projection: 1.,
                coverage: 1.,
                offset: 0.,
            },
        );
    }
    for (i, r) in continuations::bilingual(&bs, p, &cs, &assignments)? {
        assign(&mut assignments, i, r)
    }
    let cross = continuations::cross(&bs, p, &parents, &cs, &assignments, w)?;
    for (i, r) in cross {
        if let Some((_, k)) = cs.iter_mut().find(|(j, _)| *j == i) {
            *k = "caption"
        } else {
            cs.push((i, "caption"))
        }
        assign(&mut assignments, i, r)
    }
    let used = continuations::footnotes(&mut bs, p, &cs, &assignments, w)?;
    for (i, r) in &assignments {
        bs[*i]["type"] = json!(kind(&cs, *i).ok_or_else(invalid)?);
        bs[*i]["_visual_annotation_direction"] = json!(r.direction);
        if kind(&cs, *i) == Some("caption")
            && r.parent.members.len() == 1
            && !r.parent.numeric
            && parents.iter().any(|p| p.numeric)
            && graph(text(&bs[*i]))
        {
            bs[*i]["_native_annotation_parent_bbox"] = bs[r.parent.members[0]]["bbox"].clone()
        }
    }
    let regions = regions::build(&assignments, &bs, p, w)?;
    let mut remap = BTreeMap::new();
    let mut out = vec![];
    for (i, b) in bs.into_iter().enumerate() {
        if !used.contains(&i) {
            remap.insert(i, out.len());
            out.push(b)
        }
    }
    let regions = regions
        .into_iter()
        .map(|ids| {
            ids.into_iter()
                .map(|i| remap.get(&i).copied().ok_or_else(invalid))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let out = json!({"blocks":out,"regions":regions});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub fn recover_bands(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut bs = array(&v["blocks"], 4096)?.to_vec();
    bands::recover(&mut bs, optional(v, "drawing_lines", 5000)?, &mut Work(0))?;
    let out = json!(bs);
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub(super) fn sort_proxies(bs: &[Value], w: &mut Work) -> Result<Vec<usize>, Error> {
    xycut::sort(bs, w)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_python_annotation_bands() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-annotation-bands.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let got = recover_bands(&c["request"]);
            if c.get("error").is_some() {
                assert!(got.is_err(), "{}", c["label"])
            } else {
                let got = got.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                if let Some(d) =
                    super::super::native_document::tests::diff(&got, &c["expected"], "")
                {
                    panic!("{}: {d}", c["label"])
                }
            }
        }
    }
    #[test]
    fn original_python_visual_annotations() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-visual-annotations.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let got = if c["kind"] == "bind" {
                bind(&c["request"])
            } else {
                run(&c["request"])
            };
            if c.get("error").is_some() {
                assert!(got.is_err(), "{}", c["label"])
            } else {
                let got = got.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                if let Some(d) =
                    super::super::native_document::tests::diff(&got, &c["expected"], "")
                {
                    panic!("{}: {d}", c["label"]);
                }
            }
        }
    }
}
