//! Complete original formula recognition and ordered mixed/detached/band arbitration.
//! Dense-row/baseline preparation is shared with the original merge implementations.
mod anchors;
mod bands;
mod core;
mod dense;
mod evidence;
mod materialize;
mod prepared;
mod recovery;
mod tail;
use super::{
    native_marginals::local,
    page_containers::{clip, coerce, combined, median, optional, size},
    path_graphics::{array, bbox, integer, num, Box4},
};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
use std::collections::BTreeSet;
#[derive(Clone)]
struct Row {
    index: usize,
    b: Box4,
}
#[derive(Clone)]
struct Lane {
    left: f64,
    right: f64,
    rows: Vec<Row>,
}
#[derive(Clone)]
struct Anchor {
    row: Row,
    below: bool,
    above: bool,
    repeated: bool,
}
struct Page {
    lines: Vec<Value>,
    size: [f64; 2],
    tables: Vec<Box4>,
}
impl Page {
    fn validate(&mut self, w: &mut Work) -> Result<(), Error> {
        for line in &mut self.lines {
            let raw = line.as_object().ok_or_else(invalid)?.clone();
            let mut normalized = super::native_lines::defaults();
            normalized.as_object_mut().unwrap().extend(raw);
            *line = normalized;
            let text = line["text"].as_str().ok_or_else(invalid)?;
            w.charge(text.len() + 1)?;
            bbox(&line["bbox"])?;
            integer(line, "source_index", 0)?;
            integer(line, "angle", 0)?;
            integer(line, "run_index", 0)?;
            if !line["visual_row_id"].is_null() {
                integer(line, "visual_row_id", 0)?;
            }
            if !line["font_signature"].is_null() {
                let f = array(&line["font_signature"], 2)?;
                if f.len() != 2 || !f[0].is_string() || !(f[1].is_i64() || f[1].is_boolean()) {
                    return Err(invalid());
                }
            }
            for k in ["chars", "native_math_words", "inline_math_regions"] {
                array(&line[k], 20000)?;
            }
        }
        Ok(())
    }
    fn l(&self, r: &Row) -> &Value {
        &self.lines[r.index]
    }
    fn id(&self, r: &Row) -> Result<i64, Error> {
        integer(self.l(r), "source_index", 0)
    }
    fn h(&self, r: &Row) -> Result<f64, Error> {
        super::native_titles::reference_height(self.l(r), &r.b)
    }
    fn t(&self, r: &Row) -> Result<&str, Error> {
        self.l(r)["text"].as_str().ok_or_else(invalid)
    }
    fn coverage(&self, r: &Row) -> Result<f64, Error> {
        self.l(r).get("font_coverage").map_or(Ok(0.), num)
    }
    fn bounds(&self, rows: &[Row]) -> Result<Box4, Error> {
        combined(rows.iter().map(|r| &r.b))
    }
    fn lanes(&self, rows: &[Row], width: f64, em: f64, w: &mut Work) -> Result<Vec<Lane>, Error> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        let raw = super::table_text_styles::lanes::infer_formula_layout(
            &json!({"line_geometry":rows.iter().map(|r|json!([super::native_titles::reference_lane_line(self.l(r)),r.b.raw])).collect::<Vec<_>>(),"local_page_width":width,"median_height":em}),
            w,
        )?;
        let mut out = Vec::new();
        for lane in array(&raw, 4096)? {
            let mut members = array(&lane["line_indices"], 4096)?
                .iter()
                .map(|i| {
                    i.as_u64()
                        .and_then(|i| rows.get(i as usize))
                        .cloned()
                        .ok_or_else(invalid)
                })
                .collect::<Result<Vec<_>, _>>()?;
            members.sort_by(|a, b| {
                a.b.n[1]
                    .total_cmp(&b.b.n[1])
                    .then(a.b.n[0].total_cmp(&b.b.n[0]))
                    .then(self.id(a).unwrap().cmp(&self.id(b).unwrap()))
            });
            if lane["is_span"] == true {
                continue;
            }
            out.push(Lane {
                left: num(&lane["left"])?,
                right: num(&lane["right"])?,
                rows: members,
            })
        }
        Ok(out)
    }
}
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) * 0.5
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) * 0.5
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let den = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if den <= 0. {
        0.
    } else {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / den
    }
}
fn intersects(a: &Box4, b: &Box4) -> bool {
    a.n[2].min(b.n[2]) > a.n[0].max(b.n[0]) && a.n[3].min(b.n[3]) > a.n[1].max(b.n[1])
}
fn crosses(p: &Page, a: &Row, b: &Row) -> Result<bool, Error> {
    let a = bbox(&p.l(a)["bbox"])?;
    let b = bbox(&p.l(b)["bbox"])?;
    let x0 = cx(&a).min(cx(&b)) - 0.1;
    let x1 = cx(&a).max(cx(&b)) + 0.1;
    let y0 = cy(&a).min(cy(&b)) - 0.1;
    let y1 = cy(&a).max(cy(&b)) + 0.1;
    Ok(p.tables
        .iter()
        .any(|t| t.n[0] < x1 && t.n[2] > x0 && t.n[1] < y1 && t.n[3] > y0))
}
fn connected(p: &Page, a: &Row, b: &Row) -> Result<bool, Error> {
    if p.l(a)["angle"] != p.l(b)["angle"] || crosses(p, a, b)? {
        return Ok(false);
    }
    let h = p.h(a)?.max(p.h(b)?);
    let vg = (a.b.n[1] - b.b.n[3]).max(b.b.n[1] - a.b.n[3]).max(0.);
    if axis(&a.b, &b.b, 1) < 0.2 && vg > 0.6 * h {
        return Ok(false);
    }
    Ok(axis(&a.b, &b.b, 0) > 0.
        || (a.b.n[0] - b.b.n[2]).max(b.b.n[0] - a.b.n[2]).max(0.) <= 1.5 * h)
}
fn seed(p: &Page, a: &Row, b: &Row, detached: bool) -> Result<bool, Error> {
    let difference = (cy(&a.b) - cy(&b.b)).abs();
    let h = p.h(a)?.max(p.h(b)?);
    Ok(if detached {
        a.b.n[3].min(b.b.n[3]) > a.b.n[1].max(b.b.n[1]) || difference <= h
    } else {
        axis(&a.b, &b.b, 1) >= 0.3 || difference <= 0.6 * h
    })
}
fn margin(b: &Box4, h: f64) -> bool {
    b.n[3] <= 0.05 * h || b.n[1] >= 0.95 * h
}
pub(super) fn split_number(t: &str) -> Option<(String, String)> {
    let t = t.trim_matches(python_space);
    let end = t.chars().last()?;
    if !")）﹚".contains(end) {
        return None;
    }
    let start = t
        .char_indices()
        .rev()
        .skip(1)
        .find(|(_, c)| "()（）﹙﹚\r\n".contains(*c))?;
    if !"(（﹙".contains(start.1) {
        return None;
    }
    let inner = &t[start.0 + start.1.len_utf8()..t.len() - end.len_utf8()];
    if inner.is_empty() || t[..start.0].contains('\n') {
        return None;
    }
    Some((
        t[..start.0].trim_end_matches(python_space).to_owned(),
        t[start.0..].to_owned(),
    ))
}
fn standalone(t: &str) -> Option<String> {
    let (a, b) = split_number(t)?;
    a.is_empty().then_some(b)
}
fn math(t: &str) -> bool {
    super::native_vector_formulas::formula_math(t)
}
fn sentence(t: &str) -> bool {
    super::native_vector_formulas::formula_sentence(t)
}
fn sm(c: char) -> bool {
    math(&c.to_string()) && !"¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞↉".contains(c)
}
fn ink(p: &Page, index: usize, angle: i64) -> Result<Row, Error> {
    let l = &p.lines[index];
    Ok(Row {
        index,
        b: local(
            &bbox(if l["ink_bbox"].is_null() {
                &l["bbox"]
            } else {
                &l["ink_bbox"]
            })?,
            p.size,
            angle,
        ),
    })
}
fn layout(p: &Page, indices: &[usize], w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut lines = Vec::new();
    for &i in indices {
        if integer(&p.lines[i], "angle", 0)? != 0 {
            continue;
        }
        let b = bbox(&p.lines[i]["bbox"])?;
        w.charge(p.tables.len() + 1)?;
        if p.tables
            .iter()
            .any(|t| super::path_graphics::overlap_first(&b, t) >= 0.8)
        {
            continue;
        }
        lines.push(super::native_titles::reference_lane_line(&p.lines[i]));
    }
    if lines.is_empty() {
        return Ok(vec![]);
    }
    let geometry = lines
        .iter()
        .map(|l| {
            Ok(json!([
                l,
                bbox(if l["ink_bbox"].is_null() {
                    &l["bbox"]
                } else {
                    &l["ink_bbox"]
                })?
                .raw
            ]))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let h = median(
        lines
            .iter()
            .map(|l| {
                super::native_titles::reference_height(
                    l,
                    &bbox(if l["ink_bbox"].is_null() {
                        &l["bbox"]
                    } else {
                        &l["ink_bbox"]
                    })?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    let raw = super::table_text_styles::lanes::infer_formula_layout(
        &json!({"line_geometry":geometry,"local_page_width":p.size[0],"median_height":h,"recalculate_intervals":false}),
        w,
    )?;
    let mut stable = array(&raw, 4096)?
        .iter()
        .filter(|l| l["is_span"] == false && l["line_indices"].as_array().unwrap().len() >= 3)
        .cloned()
        .collect::<Vec<_>>();
    stable.sort_by(|a, b| {
        a["left"]
            .as_f64()
            .unwrap()
            .total_cmp(&b["left"].as_f64().unwrap())
    });
    let mut distinct: Vec<Value> = vec![];
    for lane in stable {
        if distinct
            .last()
            .is_some_and(|last| lane["left"].as_f64().unwrap() < last["right"].as_f64().unwrap())
        {
            if lane["line_indices"].as_array().unwrap().len()
                > distinct.last().unwrap()["line_indices"]
                    .as_array()
                    .unwrap()
                    .len()
            {
                *distinct.last_mut().unwrap() = lane
            }
        } else {
            distinct.push(lane)
        }
    }
    Ok(distinct)
}
fn regex_words(t: &str, min: usize, max: Option<usize>) -> Vec<String> {
    let re = regex::Regex::new(&format!(
        "[A-Za-z]{{{},{}}}",
        min,
        max.map_or(String::new(), |v| v.to_string())
    ))
    .unwrap();
    let word = |c| c == '_' || rapiddoc_native::text_boundary::python_alnum(c);
    re.find_iter(t)
        .filter(|m| {
            t[..m.start()].chars().last().is_none_or(|c| !word(c))
                && t[m.end()..].chars().next().is_none_or(|c| !word(c))
        })
        .map(|m| m.as_str().to_owned())
        .collect()
}
fn number(t: &str) -> bool {
    super::native_vector_formulas::formula_number(t)
}
pub fn build(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    build_with_work(v, &mut Work(0))
}
/// Complete original formula-block recovery, preserving source-line mutations.
pub fn build_full(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    full_with_work(v, &mut Work(0))
}
pub(super) fn sanitize_text(t: &str, newlines: bool) -> String {
    materialize::sanitize(t, newlines)
}
/// Original dense-row, same-baseline and complete formula recovery sequence.
pub fn prepare_input(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    input_with_work(v, &mut Work(0))
}
pub fn prepare_page(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepared::prepare(v, &mut Work(0))
}
pub(super) fn prepare_page_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepared::prepare(v, w)
}
pub(super) fn input_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut p = Page {
        lines: array(&v["lines"], 4096)?.to_vec(),
        size: size(&v["page_size"])?,
        tables: optional(v, "table_bboxes", 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?,
    };
    p.validate(w)?;
    let restored = dense::restore(&mut p, w)?;
    if v["dense_only"] == true {
        return Ok(json!({"lines":restored,"source_lines":p.lines}));
    }
    p.lines = restored;
    super::native_page_preparation::formula_merge_baseline(&mut p.lines, p.size, &p.tables, w)?;
    let q = json!({"lines":p.lines,"page_size":p.size,"table_bboxes":v.get("table_bboxes").cloned().unwrap_or(json!([])),"drawing_lines":v.get("drawing_lines").cloned().unwrap_or(json!([]))});
    let mut out = full_with_work(&q, w)?;
    let formula_input = out.as_object_mut().unwrap().remove("lines").unwrap();
    out["formula_input"] = formula_input;
    Ok(out)
}
pub(super) fn full_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut p = Page {
        lines: array(&v["lines"], 4096)?.to_vec(),
        size: size(&v["page_size"])?,
        tables: optional(v, "table_bboxes", 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?,
    };
    let drawings = optional(v, "drawing_lines", 5000)?;
    p.validate(w)?;
    let (blocks, claimed, paragraphs) = core::build(&mut p, drawings, w)?;
    let (blocks, remaining) = tail::finish(&mut p, drawings, blocks, claimed, paragraphs, w)?;
    let out = json!({"formula_blocks":blocks,"remaining_lines":remaining,"lines":p.lines});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub(super) fn build_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut p = Page {
        lines: array(&v["lines"], 4096)?.to_vec(),
        size: size(&v["page_size"])?,
        tables: optional(v, "table_bboxes", 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?,
    };
    p.validate(w)?;
    let (blocks, claimed, paragraphs) =
        core::build(&mut p, optional(v, "drawing_lines", 5000)?, w)?;
    let out = json!({"formula_blocks":blocks,"claimed_line_indices":claimed,"paragraph_lines":paragraphs,"lines":p.lines});
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
    fn original_final_page_formula_prefix() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-final-formula.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut failures = Vec::new();
        for c in data["cases"].as_array().unwrap() {
            let out = prepare_page(&c["request"]);
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
    #[test]
    fn original_final_formula_proofs_and_chains() {
        for name in ["native-final-formula-proofs", "native-final-formula-chain"] {
            let data: Value = serde_json::from_slice(
                &std::fs::read(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join(format!("../../build/migration/{name}.json")),
                )
                .unwrap(),
            )
            .unwrap();
            for c in data["cases"].as_array().unwrap() {
                let out = if name.ends_with("chain") {
                    super::super::native_document::prepare_formulas(&c["request"])
                } else {
                    prepare_page(&c["request"])
                };
                if !c["error"].is_null() {
                    assert!(out.is_err(), "{}", c["label"]);
                    continue;
                }
                let out = out.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                if let Some(d) =
                    super::super::native_document::tests::diff(&out, &c["expected"], "")
                {
                    panic!("{}: {d}", c["label"])
                }
            }
        }
    }
    #[test]
    fn original_formula_input_sequence() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-formula-input.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut failures = Vec::new();
        for c in data["cases"].as_array().unwrap() {
            let out = prepare_input(&c["request"]);
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
    #[test]
    fn original_complete_formula_recovery() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-formula-full.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let q = &c["request"];
            let out = (|| -> Result<Value, Error> {
                if q.get("kind").is_none() {
                    return build_full(q);
                }
                let w = &mut Work(0);
                let mut p = Page {
                    lines: array(&q["lines"], 4096)?.to_vec(),
                    size: size(&q["page_size"])?,
                    tables: optional(q, "table_bboxes", 5000)?
                        .iter()
                        .map(bbox)
                        .collect::<Result<Vec<_>, _>>()?,
                };
                let indices = (0..p.lines.len()).collect::<Vec<_>>();
                let blocks = match q["kind"].as_str().unwrap() {
                    "mixed" => recovery::mixed(&mut p, &indices, w)?.0,
                    "detached" => {
                        let flags = recovery::flags(&p, w)?;
                        recovery::detached(
                            &mut p,
                            &indices,
                            optional(q, "drawing_lines", 5000)?,
                            &flags,
                            w,
                        )?
                        .0
                    }
                    "bands" => bands::build(&mut p, optional(q, "drawing_lines", 5000)?, w)?,
                    _ => return Err(invalid()),
                };
                Ok(json!({"formula_blocks":blocks,"lines":p.lines}))
            })();
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
    fn original_formula_spatial_prefix() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-formula-components.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let out = build(&c["request"]);
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
}
