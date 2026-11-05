//! Exact existing post-aggregation tight boxes and final normalized block/line output.
use super::{
    native_marginals::local,
    page_containers::{clip, coerce, combined, optional, size},
    path_graphics::{array, bbox, Box4},
};
use crate::legacy_page::{invalid, limit, validate_tree};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
fn angle(b: &Value) -> Result<i64, Error> {
    let v = &b["angle"];
    let n = if v.is_null() || v == false || v == "" {
        0
    } else if let Some(n) = v.as_i64() {
        n
    } else if let Some(n) = v.as_bool() {
        i64::from(n)
    } else if let Some(n) = v.as_f64() {
        n as i64
    } else if let Some(t) = v.as_str() {
        t.trim_matches(python_space)
            .parse()
            .map_err(|_| invalid())?
    } else {
        return Err(invalid());
    };
    Ok(n.rem_euclid(360))
}
pub(super) fn block_angle(b: &Value) -> Result<i64, Error> {
    angle(b)
}
fn from_local(b: &Box4, p: [f64; 2], a: i64) -> Box4 {
    let lp = if a == 90 || a == 270 { [p[1], p[0]] } else { p };
    local(b, lp, (360 - a) % 360)
}
fn unit(b: &Box4, p: [f64; 2]) -> Value {
    let mut ticks = [0i64; 4];
    for i in 0..4 {
        ticks[i] = (b.n[i] / p[i % 2] * 1000.)
            .round_ties_even()
            .clamp(0., 1000.) as i64;
    }
    for i in 0..2 {
        if ticks[i + 2] <= ticks[i] {
            if ticks[i] < 1000 {
                ticks[i + 2] = ticks[i] + 1
            } else {
                ticks[i] = (ticks[i + 2] - 1).max(0)
            }
        }
    }
    json!(ticks.map(|n| n as f64 / 1000.))
}
pub(super) fn tight(bs: &mut [Value], p: [f64; 2]) -> Result<(), Error> {
    for b in bs {
        let obj = b.as_object_mut().ok_or_else(invalid)?;
        let mut candidate = coerce(&obj.remove("_tight_output_bbox").unwrap_or(Value::Null))?;
        let rows = obj
            .remove("_local_output_line_bboxes")
            .unwrap_or(Value::Null);
        let repaired = obj.remove("_output_bbox_repaired").unwrap_or(json!(false));
        if repaired == true {
            if let Some(rows) = rows.as_array().filter(|r| !r.is_empty()) {
                let boxes = rows.iter().map(coerce).collect::<Result<Vec<_>, _>>()?;
                if boxes.iter().all(Option::is_some) {
                    let boxes = boxes.into_iter().flatten().collect::<Vec<_>>();
                    b["_local_line_bboxes"] =
                        json!(boxes.iter().map(|b| b.raw.clone()).collect::<Vec<_>>());
                    let a = angle(b)?;
                    let raw = boxes
                        .iter()
                        .map(|b| from_local(b, p, a))
                        .collect::<Vec<_>>();
                    candidate = Some(combined(raw.iter())?);
                }
            }
        }
        if let Some(candidate) = candidate {
            if let Some(c) = clip(&candidate, p)? {
                b["bbox"] = json!(c.raw)
            }
        }
    }
    Ok(())
}
fn lines(b: &Value, p: [f64; 2]) -> Result<Value, Error> {
    let Some(rows) = b["_local_line_bboxes"].as_array() else {
        return Ok(json!([]));
    };
    let a = angle(b)?;
    let mut out = vec![];
    for row in rows {
        let Some(values) = row.as_array() else {
            return Ok(json!([]));
        };
        let mut n = vec![];
        for v in values {
            let Some(x) = super::table_native_text::number(v)? else {
                return Ok(json!([]));
            };
            n.push(x)
        }
        let Some(coerced) = coerce(&json!(n))? else {
            return Ok(json!([]));
        };
        if n.len() != 4 || n[2] <= n[0] || n[3] <= n[1] {
            return Ok(json!([]));
        }
        let Some(page) = clip(&from_local(&coerced, p, a), p)? else {
            return Ok(json!([]));
        };
        out.push(json!({"bbox":unit(&page,p)}));
    }
    Ok(json!(out))
}
pub(super) fn normalize(b: &Value, p: [f64; 2]) -> Result<Value, Error> {
    let Some(raw) = coerce(&b["bbox"])? else {
        return Ok(Value::Null);
    };
    let Some(bounds) = clip(&raw, p)? else {
        return Ok(Value::Null);
    };
    if p[0] <= 0. || p[1] <= 0. {
        return Ok(Value::Null);
    }
    let Some(content) = b["content"].as_str() else {
        return Ok(Value::Null);
    };
    let content = super::native_formula_components::sanitize_text(content, true);
    let ty = b["type"].as_str().unwrap_or("");
    let ty = if [
        "text",
        "table",
        "image",
        "equation",
        "code",
        "list",
        "doc_title",
        "paragraph_title",
        "ref_text",
        "header",
        "footer",
        "page_number",
        "caption",
        "footnote",
        "page_footnote",
        "aside_text",
        "index",
    ]
    .contains(&ty)
    {
        ty
    } else {
        "text"
    };
    if !["image", "equation", "header", "footer", "list"].contains(&ty)
        && content.trim_matches(python_space).is_empty()
    {
        return Ok(Value::Null);
    }
    let mut out = json!({"type":ty,"bbox":unit(&bounds,p),"angle":if ty=="image"{0}else{angle(b)?},"content":if ty=="equation"{""}else{&content}});
    let mut regions = vec![];
    for region in optional(b, "_inline_math_regions", 20000)? {
        if let Some(raw) = coerce(region)? {
            if let Some(clipped) = clip(&raw, p)? {
                regions.push(unit(&clipped, p));
            }
        }
    }
    if !regions.is_empty() {
        out["_inline_math_regions"] = json!(regions)
    }
    let images = &b["_table_images"];
    if super::native_text_assembly::python_truth(images) {
        let mut result = vec![];
        for img in array(images, 5000)? {
            let mut img = img.clone();
            img.as_object().ok_or_else(invalid)?;
            img["bbox"] = unit(&bbox(&img["bbox"])?, p);
            result.push(img)
        }
        out["_table_images"] = json!(result)
    }
    if [
        "text",
        "ref_text",
        "doc_title",
        "paragraph_title",
        "caption",
        "footnote",
    ]
    .contains(&ty)
    {
        out["lines"] = lines(b, p)?
    }
    let members = optional(b, "_text_lines", 4096)?;
    if let Some(first) = members.first() {
        if !first["reference_start"].is_null() {
            out["_reference_start"] = first["reference_start"].clone();
        }
    }
    if super::native_text_assembly::python_truth(&b["_protected_hard_break_before"]) {
        out["_paragraph_boundary"] = json!(true)
    }
    if let Some(parent) = coerce(&b["_native_annotation_parent_bbox"])? {
        out["_native_annotation_parent_bbox"] = unit(&parent, p)
    }
    Ok(out)
}
pub fn run(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let p = size(&v["page_size"])?;
    let mut bs = array(&v["blocks"], 4096)?.to_vec();
    let out = match v["kind"].as_str().ok_or_else(invalid)? {
        "tight" => {
            tight(&mut bs, p)?;
            json!(bs)
        }
        "lines" => json!(bs
            .iter()
            .map(|b| lines(b, p))
            .collect::<Result<Vec<_>, _>>()?),
        "normalize" => json!(bs
            .iter()
            .map(|b| normalize(b, p))
            .collect::<Result<Vec<_>, _>>()?),
        "reference_order" => json!(reference_order(
            bs,
            p,
            &v["page"],
            &mut crate::legacy_page::Work(0)
        )?),
        "narrow_order" => json!(narrow_order(bs, p, &mut crate::legacy_page::Work(0))?),
        _ => return Err(invalid()),
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
fn line_height(l: &Value) -> Result<f64, Error> {
    super::native_titles::reference_height(l, &bbox(&l["bbox"])?)
}
fn truth(v: &Value) -> bool {
    super::native_text_assembly::python_truth(v)
}
fn layout_lines(bs: &[Value], ids: &[usize]) -> Result<Vec<Value>, Error> {
    let mut ls = vec![];
    for &i in ids {
        ls.extend_from_slice(optional(&bs[i], "_text_lines", 4096)?)
    }
    Ok(ls)
}
pub(super) fn reference_order(
    mut bs: Vec<Value>,
    p: [f64; 2],
    page: &Value,
    w: &mut crate::legacy_page::Work,
) -> Result<Vec<Value>, Error> {
    let regions = optional(page, "reference_regions", 5000)?;
    if !truth(&page["numbered_references"]) || regions.is_empty() {
        return Ok(bs);
    }
    let top = regions
        .iter()
        .map(|b| bbox(b).map(|b| b.n[1]))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let mut ids = vec![];
    for (i, b) in bs.iter().enumerate() {
        if b["type"] != "text" || !b["_reference_group"].is_null() {
            continue;
        }
        let bb = bbox(&b["bbox"])?;
        let ls = optional(b, "_text_lines", 4096)?;
        if bb.n[3] < top && !ls.is_empty() {
            let hs = ls.iter().map(line_height).collect::<Result<Vec<_>, _>>()?;
            if bb.n[2] - bb.n[0] >= 3. * super::page_containers::median(hs) {
                ids.push(i)
            }
        }
    }
    if ids.len() < 4 {
        return Ok(bs);
    }
    let layout = super::native_layout::build(&layout_lines(&bs, &ids)?, p, 0, &[], w)?;
    if layout.lanes.len() < 2 {
        return Ok(bs);
    }
    let mut keys = vec![];
    for &i in &ids {
        let b = bbox(&bs[i]["bbox"])?;
        let Some(lane) = layout.corridor(&b)? else {
            return Ok(bs);
        };
        if lane == (0., p[0]) {
            return Ok(bs);
        }
        keys.push((i, lane.0, b.n[1]))
    }
    let lo = *ids.iter().min().unwrap();
    let hi = *ids.iter().max().unwrap();
    if bs[lo..=hi].iter().any(|b| {
        ["image", "table", "equation", "paragraph_title"]
            .iter()
            .any(|t| b["type"] == *t)
    }) {
        return Ok(bs);
    }
    keys.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)));
    let replacements = keys.iter().map(|k| bs[k.0].clone()).collect::<Vec<_>>();
    for (i, b) in ids.into_iter().zip(replacements) {
        bs[i] = b
    }
    Ok(bs)
}
pub(super) fn narrow_order(
    mut bs: Vec<Value>,
    p: [f64; 2],
    w: &mut crate::legacy_page::Work,
) -> Result<Vec<Value>, Error> {
    if !bs.iter().any(|b| truth(&b["_caption_graphic"])) {
        return Ok(bs);
    }
    let marginal = |b: &Value| {
        ["header", "footer", "page_number", "aside_text"]
            .iter()
            .any(|t| b["type"] == *t)
    };
    let content = bs
        .iter()
        .enumerate()
        .filter(|(_, b)| !marginal(b))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let body = content
        .iter()
        .copied()
        .filter(|&i| {
            bs[i]["type"] == "text"
                && optional(&bs[i], "_text_lines", 4096).is_ok_and(|ls| ls.len() >= 2)
        })
        .collect::<Vec<_>>();
    let layout = super::native_layout::build(&layout_lines(&bs, &body)?, p, 0, &[], w)?;
    if layout.lanes.len() < 2 || body.len() < 3 {
        return Ok(bs);
    }
    let mut keys = vec![];
    for &i in &content {
        let b = bbox(&bs[i]["bbox"])?;
        let Some(lane) = layout.corridor(&b)? else {
            return Ok(bs);
        };
        if lane == (0., p[0]) {
            return Ok(bs);
        }
        keys.push((i, lane.0, b.n[1], b.n[0]))
    }
    keys.sort_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then(a.2.total_cmp(&b.2))
            .then(a.3.total_cmp(&b.3))
    });
    let replacements = keys.iter().map(|k| bs[k.0].clone()).collect::<Vec<_>>();
    for (i, b) in content.into_iter().zip(replacements) {
        bs[i] = b
    }
    Ok(bs)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_python_final_output() {
        for name in ["native-final-output", "native-final-order-edges"] {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                let got = run(&c["request"]);
                if c.get("error").is_some() {
                    assert!(got.is_err(), "{}", c["label"])
                } else {
                    assert_eq!(
                        got.unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                        c["expected"],
                        "{}",
                        c["label"]
                    );
                }
            }
        }
    }
}
