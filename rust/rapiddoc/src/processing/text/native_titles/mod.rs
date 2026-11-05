//! Original full page titles and document title-profile/structural/recurrent passes.
mod compact;
mod cross;
mod document;
mod evidence;
mod final_rules;
mod lane;
mod lane_evidence;
mod page_structural;
mod profile;
mod prototype;
mod recurrent;
mod refine;
mod selected;
mod structural;
use super::{
    native_marginals::local,
    page_containers::{combined, median, optional, size},
    path_graphics::{array, bbox, integer, num, Box4},
};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone)]
struct Lane {
    left: f64,
    right: f64,
    span: bool,
    ids: Vec<usize>,
}
struct State {
    lines: Vec<Value>,
    boxes: Vec<Box4>,
}
fn height(l: &Value, b: &Box4) -> Result<f64, Error> {
    if l["style_scale_repaired"] == true {
        let em = l.get("em_height").map_or(Ok(0.), num)?;
        if em > 0. {
            return Ok(em.max(0.1));
        }
    }
    let h = if l["effective_height"].is_null() {
        0.
    } else {
        num(&l["effective_height"])?
    };
    Ok((if h != 0. { h } else { b.n[3] - b.n[1] }).max(0.1))
}
fn lane_line(l: &Value) -> Value {
    let mut out = l.clone();
    if out["effective_height"].is_null() {
        out["effective_height"] = json!(0.);
    }
    if out["em_height"].is_null() && out["style_scale_repaired"] != true {
        out["em_height"] = json!(0.);
    }
    out
}
pub(super) fn reference_height(l: &Value, b: &Box4) -> Result<f64, Error> {
    height(l, b)
}
pub(super) fn reference_lane_line(l: &Value) -> Value {
    lane_line(l)
}
pub(super) fn compact_reference_titles(
    p: &mut Value,
    page: i64,
    w: &mut Work,
) -> Result<(), Error> {
    compact::classify(p, page, w)
}
struct Profile {
    h: f64,
    font: Value,
    weight: Option<f64>,
    gap: f64,
    support: Vec<(Value, f64)>,
    count: usize,
}
impl Profile {
    fn raw(&self) -> Value {
        json!({"body_height":self.h,"body_font":self.font,"body_weight":self.weight,"regular_gap":self.gap,"style_support":self.support.iter().map(|(font,ratio)|json!({"font":font,"ratio":ratio})).collect::<Vec<_>>(),"body_row_count":self.count})
    }
}
impl State {
    fn h(&self, i: usize) -> Result<f64, Error> {
        height(&self.lines[i], &self.boxes[i])
    }
    fn unresolved(&self, i: usize) -> bool {
        self.lines[i]["semantic_type"].is_null()
    }
    fn mark(&mut self, i: usize, t: &str) {
        self.lines[i]["semantic_type"] = json!(t)
    }
    fn order(&self, ids: &mut [usize]) {
        ids.sort_by(|&i, &j| {
            self.boxes[i].n[1]
                .total_cmp(&self.boxes[j].n[1])
                .then(self.boxes[i].n[0].total_cmp(&self.boxes[j].n[0]))
                .then(
                    integer(&self.lines[i], "source_index", 0)
                        .unwrap()
                        .cmp(&integer(&self.lines[j], "source_index", 0).unwrap()),
                )
        })
    }
    fn bounds(&self, ids: &[usize]) -> Result<Box4, Error> {
        combined(ids.iter().map(|&i| &self.boxes[i]))
    }
    fn profile(&self, lane: &mut Lane, w: &mut Work) -> Result<Profile, Error> {
        let available = lane
            .ids
            .iter()
            .copied()
            .filter(|&i| self.unresolved(i))
            .collect::<Vec<_>>();
        if available.is_empty() {
            return Ok(Profile {
                h: 1.,
                font: Value::Null,
                weight: None,
                gap: 0.35,
                support: vec![],
                count: 0,
            });
        }
        let width = (lane.right - lane.left).max(0.1);
        let long = available
            .iter()
            .copied()
            .filter(|&i| self.boxes[i].n[2] - self.boxes[i].n[0] >= 0.45 * width)
            .collect::<Vec<_>>();
        let body = if long.is_empty() {
            available.clone()
        } else {
            long
        };
        let h = median(
            body.iter()
                .map(|&i| self.h(i))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let mut font_support = Vec::<(Value, f64)>::new();
        let mut style = Vec::<(Value, f64)>::new();
        let mut total = 0.;
        let body_ids = body.iter().copied().collect::<BTreeSet<_>>();
        for i in available {
            w.charge(1)?;
            let l = &self.lines[i];
            if l["font_signature"].is_null() || coverage(l)? < 0.75 {
                continue;
            }
            let width = (self.boxes[i].n[2] - self.boxes[i].n[0]).max(0.1);
            add(&mut style, &l["font_signature"], width);
            total += width;
            if body_ids.contains(&i) && (0.75..=1.35).contains(&(self.h(i)? / h)) {
                add(&mut font_support, &l["font_signature"], width)
            }
        }
        let mut font = Value::Null;
        let mut support = f64::NEG_INFINITY;
        for (f, v) in font_support {
            if v > support {
                font = f;
                support = v
            }
        }
        if total > 0. {
            for (_, v) in &mut style {
                *v /= total
            }
        }
        let weights = body
            .iter()
            .filter_map(|&i| {
                let l = &self.lines[i];
                (!l["dominant_font_weight"].is_null()
                    && (font.is_null() || eq(&l["font_signature"], &font)))
                .then_some(i)
            })
            .map(|i| Ok((i, self.h(i)?)))
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter()
            .filter(|(_, height)| (0.75..=1.35).contains(&(*height / h)))
            .map(|(i, _)| num(&self.lines[i]["dominant_font_weight"]))
            .collect::<Result<Vec<_>, _>>()?;
        // The original gap estimator sorts lane members, even when its result is cached later.
        self.order(&mut lane.ids);
        let heights = lane
            .ids
            .iter()
            .map(|&i| self.h(i))
            .collect::<Result<Vec<_>, _>>()?;
        let skip = lane
            .ids
            .windows(2)
            .map(|p| {
                eq(
                    &self.lines[p[0]]["visual_row_id"],
                    &self.lines[p[1]]["visual_row_id"],
                ) && (self.lines[p[0]]["split_from_row"] == true
                    || self.lines[p[1]]["split_from_row"] == true)
            })
            .collect();
        w.charge(lane.ids.len() + 1)?;
        let (gap, _) = rapiddoc_native::geometry_plan::statistics::lane_gap(
            lane.ids.iter().map(|&i| self.boxes[i].n).collect(),
            heights,
            lane.ids
                .iter()
                .map(|&i| self.lines[i]["restored_inline_cluster"] == true)
                .collect(),
            skip,
        )
        .ok_or_else(invalid)?;
        Ok(Profile {
            h: h.max(0.1),
            font,
            weight: (!weights.is_empty()).then(|| median(weights)),
            gap,
            support: style,
            count: body.len(),
        })
    }
}
fn add(s: &mut Vec<(Value, f64)>, f: &Value, v: f64) {
    if let Some((_, old)) = s.iter_mut().find(|(old, _)| eq(old, f)) {
        *old += v
    } else {
        s.push((f.clone(), v))
    }
}
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) / 2.
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) / 2.
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let den = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if den <= 0. {
        0.
    } else {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / den
    }
}
fn coverage(l: &Value) -> Result<f64, Error> {
    l.get("font_coverage").map_or(Ok(0.), num)
}
fn weight(l: &Value) -> Result<Option<f64>, Error> {
    if l["dominant_font_weight"].is_null() {
        Ok(None)
    } else {
        Ok(Some(num(&l["dominant_font_weight"])?))
    }
}
fn weights_ok(a: &Value, b: &Value) -> Result<bool, Error> {
    Ok(match (weight(a)?, weight(b)?) {
        (Some(a), Some(b)) => (a - b).abs() < 100. || a.max(b) < 1.15 * a.min(b),
        _ => true,
    })
}
fn fonts_ok(a: &Value, b: &Value) -> Result<bool, Error> {
    Ok((a["font_signature"].is_null()
        || b["font_signature"].is_null()
        || coverage(a)? < 0.75
        || coverage(b)? < 0.75
        || eq(&a["font_signature"], &b["font_signature"]))
        && weights_ok(a, b)?)
}
fn family(f: &Value) -> Result<Option<String>, Error> {
    if f.is_null() {
        return Ok(None);
    }
    let a = array(f, 2)?;
    if a.len() != 2 {
        return Err(invalid());
    }
    let name = a[0].as_str().ok_or_else(invalid)?.to_owned();
    Ok(super::table_text_styles::preparation::family(&Some((
        name,
        a[1].clone(),
    ))))
}
fn regular(l: &Value, body: &Value) -> Result<bool, Error> {
    if body.is_null() || l["font_signature"].is_null() || coverage(l)? < 0.5 {
        return Ok(false);
    }
    let Some(f) = family(&l["font_signature"])? else {
        return Ok(false);
    };
    for font in optional(body, "regular_fonts", 4096)? {
        if family(font)?.as_ref() == Some(&f) {
            return Ok(true);
        }
    }
    Ok(false)
}
fn share_family(a: &Value, b: &Value) -> Result<bool, Error> {
    let a = family(a)?;
    let b = family(b)?;
    Ok(a.is_some() && b.is_some() && a == b)
}
fn inside(b: &Box4, containers: &[Box4]) -> bool {
    containers
        .iter()
        .any(|c| c.n[0] <= cx(b) && cx(b) <= c.n[2] && c.n[1] <= cy(b) && cy(b) <= c.n[3])
}
fn near(b: &Box4, containers: &[Box4], h: f64) -> bool {
    containers.iter().any(|c| {
        axis(b, c, 0) >= 0.35
            && b.n[2] - b.n[0] <= 0.8 * (c.n[2] - c.n[0]).max(0.1)
            && (b.n[1] - c.n[3]).max(c.n[1] - b.n[3]).max(0.) <= 2. * h
    })
}
fn gap(s: &State, a: usize, b: usize) -> Result<f64, Error> {
    let h = s.h(a)?;
    Ok(if s.lines[a]["restored_inline_cluster"] == true {
        (s.boxes[b].n[1] - s.boxes[a].n[3]).max(-0.25 * h)
    } else {
        s.boxes[b].n[1] - (s.boxes[a].n[1] + h)
    })
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0))
}
pub fn classify_page_structure(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    page_structural::run(v, &mut Work(0))
}
pub fn refine_final_titles(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    final_rules::run(v, &mut Work(0))
}
pub(super) fn refine_final_titles_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    final_rules::run(v, w)
}
pub(super) fn classify_page_structure_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    page_structural::run(v, w)
}
#[cfg(test)]
mod final_rule_tests {
    use super::*;
    #[test]
    fn existing_python_final_title_rules() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-final-title-rules-typed.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut failures = vec![];
        for c in f["cases"].as_array().unwrap() {
            let out = refine_final_titles(&c["request"]);
            if !c["error"].is_null() {
                assert!(out.is_err(), "{}", c["label"]);
                continue;
            }
            match out {
                Err(e) => failures.push(format!("{}: {e}", c["label"])),
                Ok(out) => {
                    if let Some(d) =
                        super::super::native_document::tests::diff(&out, &c["expected"], "")
                    {
                        failures.push(format!("{}: {d}", c["label"]))
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} differences: {}",
            failures.len(),
            failures
                .iter()
                .take(25)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
pub(super) fn final_roman_headings(
    ls: &mut [Value],
    bs: &[Box4],
    cs: &[Box4],
    sz: [f64; 2],
    w: &mut Work,
) -> Result<(), Error> {
    final_rules::roman(ls, bs, cs, sz, w)
}
#[cfg(test)]
mod structure_tests {
    use super::*;
    #[test]
    fn existing_python_page_structure() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-page-structure.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut failures = vec![];
        for c in f["cases"].as_array().unwrap() {
            let out = classify_page_structure(&c["request"]);
            if !c["error"].is_null() {
                assert!(out.is_err(), "{}", c["label"]);
                continue;
            }
            match out {
                Err(e) => failures.push(format!("{}: {e}", c["label"])),
                Ok(out) => {
                    if let Some(d) =
                        super::super::native_document::tests::diff(&out, &c["expected"], "")
                    {
                        failures.push(format!("{}: {d}", c["label"]))
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} differences: {}",
            failures.len(),
            failures
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
pub fn infer_document_profile(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    profile::infer(v, &mut Work(0))
}
pub(super) fn infer_document_profile_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    profile::infer(v, w)
}
/// Complete original document title sequence; probe classification never mutates page inputs.
pub fn classify_document(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    classify_document_with_work(v, &mut Work(0))
}
pub(super) fn classify_document_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    classify_document_owned(v.clone(), w)
}
pub(super) fn classify_document_owned(mut v: Value, w: &mut Work) -> Result<Value, Error> {
    array(&v["pages"], 1000)?;
    let title = infer_document_profile_with_work(&v, w)?;
    let Value::Array(mut pages) = v["pages"].take() else {
        return Err(invalid());
    };
    structural::classify(
        &mut pages,
        &v["canonical_body_profile"],
        &v["body_profile"],
        &title,
        w,
    )?;
    recurrent::classify(&mut pages, w)?;
    let mut out = json!({"title_profile":title});
    out["pages"] = Value::Array(pages);
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub(super) fn prepare_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let p = size(&v["page_size"])?;
    let page = integer(v, "page_index", 0)?;
    let raw = array(&v["lines"], 4096)?.to_vec();
    let boxes = raw
        .iter()
        .map(|l| bbox(&l["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut s = State { lines: raw, boxes };
    if page > 0 {
        evidence::front(&mut s, p, w)?
    }
    let containers = optional(v, "container_bboxes", 4000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let body = &v["document_body_profile"];
    let mut contexts = Vec::new();
    let angles = s
        .lines
        .iter()
        .filter(|l| l["semantic_type"].is_null() && l["title_suppressed"] != true)
        .map(|l| integer(l, "angle", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for a in angles {
        let ids = s
            .lines
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                integer(l, "angle", 0).ok() == Some(a)
                    && (l["semantic_type"].is_null() || l["explicit_section_title"] == true)
                    && l["title_suppressed"] != true
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if ids.is_empty() {
            continue;
        }
        for i in 0..s.lines.len() {
            if integer(&s.lines[i], "angle", 0)? == a {
                s.boxes[i] = local(&bbox(&s.lines[i]["bbox"])?, p, a)
            }
        }
        let local_size = if a == 90 || a == 270 { [p[1], p[0]] } else { p };
        let em = median(ids.iter().map(|&i| s.h(i)).collect::<Result<Vec<_>, _>>()?);
        let request = json!({"line_geometry":ids.iter().map(|&i|json!([lane_line(&s.lines[i]),s.boxes[i].raw])).collect::<Vec<_>>(),"local_page_width":local_size[0],"median_height":em});
        let value = if v["final_page_objects"] == true {
            super::table_text_styles::lanes::infer_formula_layout(&request, w)?
        } else {
            super::table_text_styles::lanes::infer_layout(&request, w)?
        };
        let mut lanes = Vec::new();
        for l in array(&value, 4096)? {
            let mut members = array(&l["line_indices"], 4096)?
                .iter()
                .map(|i| {
                    i.as_u64()
                        .and_then(|i| ids.get(i as usize).copied())
                        .ok_or_else(invalid)
                })
                .collect::<Result<Vec<_>, _>>()?;
            s.order(&mut members);
            lanes.push(Lane {
                left: num(&l["left"])?,
                right: num(&l["right"])?,
                span: l["is_span"] == true,
                ids: members,
            })
        }
        let physical = evidence::physical(&s, &ids, w)?;
        let local_containers = containers
            .iter()
            .map(|b| local(b, p, a))
            .collect::<Vec<_>>();
        let suppressed = evidence::suppression(&s, &ids, &lanes, &local_containers, em, w)?;
        let mut bottom = None;
        if page == 0 {
            bottom = document::select(&mut s, &mut lanes, local_size, body, w)?;
            bottom = document::expand(&mut s, &ids, local_size, bottom, w)?;
            bottom = document::additional(&mut s, &ids, local_size, bottom, body, w)?
        }
        let fallback = document::fallback(&s, &mut lanes, body, w)?;
        cross::centered(
            &mut s,
            &ids,
            &mut lanes,
            local_size,
            &local_containers,
            page,
            bottom,
            w,
        )?;
        cross::emphasized(
            &mut s,
            &ids,
            &mut lanes,
            local_size,
            &local_containers,
            bottom,
            body,
            w,
        )?;
        let mut lane_values = Vec::new();
        let mut gap_map = BTreeMap::new();
        for item in &physical {
            gap_map.insert(
                integer(item, "source_index", 0)?,
                (
                    if item["above"].is_null() {
                        None
                    } else {
                        Some(num(&item["above"])?)
                    },
                    if item["below"].is_null() {
                        None
                    } else {
                        Some(num(&item["below"])?)
                    },
                ),
            );
        }
        let suppressed_set = suppressed.iter().copied().collect::<BTreeSet<_>>();
        for lane in &mut lanes {
            let profile = s.profile(lane, w)?;
            if v["classify_page"] == true {
                self::lane::classify(
                    &mut s,
                    lane,
                    &profile,
                    local_size,
                    &local_containers,
                    bottom,
                    fallback,
                    &gap_map,
                    &suppressed_set,
                    body,
                    &v["document_title_profile"],
                    page,
                    w,
                )?;
            }
            lane_values.push(json!({"left":lane.left,"right":lane.right,"is_span":lane.span,"line_indices":lane.ids,"profile":profile.raw()}))
        }
        if v["classify_page"] == true {
            let caption = optional(v, "caption_container_bboxes", 4000)?
                .iter()
                .map(|b| bbox(b).map(|b| local(&b, p, a)))
                .collect::<Result<Vec<_>, _>>()?;
            refine::finish(
                &mut s,
                &ids,
                &mut lanes,
                &local_containers,
                &caption,
                body,
                page,
                a,
                w,
            )?;
        }
        contexts.push(json!({"angle":a,"line_indices":ids,"median_height":em,"local_page_size":local_size,"lanes":lane_values,"physical_gaps":physical,"grid_title_suppressions":suppressed,"document_title_bottom":bottom,"preserve_front_matter_boundaries":fallback}));
    }
    let out = if v["classify_page"] == true {
        json!({"lines":s.lines})
    } else {
        json!({"lines":s.lines,"contexts":contexts})
    };
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_title_edge_contracts() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-title-contracts.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let out = prepare(&c["request"]);
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
    fn original_raw_document_text_chain() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-document-text-chain.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let out = super::super::native_document::prepare_text(&c["request"]);
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
    fn original_document_title_sequence() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-document-titles.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let out = classify_document(&c["request"]);
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
    fn original_document_title_profiles() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-title-profiles.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let out = infer_document_profile(&c["request"]);
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
    fn original_complete_page_titles() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-page-titles.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let out = prepare(&c["request"]);
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
    fn original_page_title_layout_prefix() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-title-layout.json");
        let data: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in data["cases"].as_array().unwrap() {
            let actual = prepare(&c["request"]);
            if !c["error"].is_null() {
                assert!(actual.is_err(), "{}", c["label"]);
                continue;
            }
            let actual = actual.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&actual, &c["expected"], "")
            {
                panic!("{}: {d}", c["label"])
            }
        }
    }
}
