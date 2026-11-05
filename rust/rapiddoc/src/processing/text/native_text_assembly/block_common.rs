//! Exact internal-block metadata merge and lane/corridor helpers.
use super::*;
fn set_number(v: &Value) -> Result<f64, Error> {
    v.as_bool()
        .map(|b| Ok(f64::from(b)))
        .unwrap_or_else(|| num(v))
}
/// Python numeric sets keep the first inserted representative (True and 1 compare equal).
pub(super) fn numeric_set(values: impl IntoIterator<Item = Value>) -> Result<Vec<Value>, Error> {
    let mut out = vec![];
    for v in values {
        set_number(&v)?;
        if !out.iter().any(|x| eq(x, &v)) {
            out.push(v)
        }
    }
    out.sort_by(|a, b| set_number(a).unwrap().total_cmp(&set_number(b).unwrap()));
    Ok(out)
}
pub(super) fn sort_set(v: &mut Vec<Value>) {
    v.sort_by(|a, b| {
        let a = a.as_array().unwrap();
        let b = b.as_array().unwrap();
        a[0].as_str().unwrap().cmp(b[0].as_str().unwrap()).then(
            set_number(&a[1])
                .unwrap()
                .total_cmp(&set_number(&b[1]).unwrap()),
        )
    });
}
pub(super) fn rows(v: &Value) -> Result<Vec<Box4>, Error> {
    optional(v, "_local_line_bboxes", 4096)?
        .iter()
        .map(bbox)
        .collect()
}
pub(super) fn key(v: &Value) -> Result<(f64, f64), Error> {
    let r = rows(v)?;
    let r = r.first().ok_or_else(invalid)?;
    Ok((r.n[1], r.n[0]))
}
pub(super) fn angle(v: &Value) -> Result<i64, Error> {
    Ok(integer(v, "angle", 0)?.rem_euclid(360))
}
pub(super) fn heights(v: &Value, positive: bool) -> Result<Vec<f64>, Error> {
    optional(v, "_line_heights", 4096)?
        .iter()
        .filter(|v| v.is_number() || v.is_boolean())
        .map(num)
        .filter(|v| !positive || v.as_ref().is_ok_and(|n| *n > 0.))
        .collect()
}
pub(super) fn merge(bs: &[Value], ids: &[usize], spaces: bool) -> Result<Value, Error> {
    let mut ids = ids.to_vec();
    ids.sort_by(|&i, &j| {
        if spaces {
            num(&bs[i]["bbox"][0])
                .unwrap()
                .total_cmp(&num(&bs[j]["bbox"][0]).unwrap())
        } else {
            key(&bs[i])
                .unwrap()
                .partial_cmp(&key(&bs[j]).unwrap())
                .unwrap()
        }
    });
    let mut b = bs[*ids.first().ok_or_else(invalid)?].clone();
    let boxes = ids
        .iter()
        .map(|&i| bbox(&bs[i]["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    b["bbox"] = json!(combined(boxes.iter())?.raw);
    let ts = ids
        .iter()
        .map(|&i| bs[i]["content"].as_str().unwrap_or("").to_owned())
        .collect::<Vec<_>>();
    b["content"] = json!(if spaces {
        ts.iter()
            .map(|s| s.trim_matches(python_space))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        join_text(&ts)?
    });
    let mut visual = vec![];
    let mut fonts = vec![];
    for &i in &ids {
        for r in optional(&bs[i], "_visual_row_ids", 4096)? {
            visual.push(r.clone());
        }
        for f in optional(&bs[i], "_font_signatures", 4096)? {
            if !fonts.iter().any(|v| eq(v, f)) {
                fonts.push(f.clone())
            }
        }
    }
    sort_set(&mut fonts);
    b["_visual_row_ids"] = json!(numeric_set(visual)?);
    b["_font_signatures"] = json!(fonts);
    b["_single_run_row_id"] = Value::Null;
    for field in [
        "_text_lines",
        "_local_line_bboxes",
        "_local_output_line_bboxes",
        "_line_heights",
        "_inline_math_regions",
    ] {
        let mut items = vec![];
        for &i in &ids {
            items.extend_from_slice(optional(&bs[i], field, 20000)?)
        }
        b[field] = json!(items)
    }
    for field in ["_output_bbox_repaired", "_paragraph_formula_context"] {
        b[field] = json!(ids.iter().any(|&i| bs[i][field] == true))
    }
    Ok(b)
}
pub(super) fn interval(v: &Value, span: bool) -> Option<(f64, f64)> {
    if !span && v.get("_lane_is_span") != Some(&json!(false)) {
        return None;
    }
    let a = v["_lane_interval"].as_array()?;
    if a.len() != 2 {
        return None;
    }
    let l = num(&a[0]).ok()?;
    let r = num(&a[1]).ok()?;
    (r > l).then_some((l, r))
}
pub(super) fn role(a: &Value, b: &Value, h: f64) -> bool {
    match (a["_lane_is_span"].as_bool(), b["_lane_is_span"].as_bool()) {
        (Some(x), Some(y)) => {
            if x != y {
                return false;
            }
            if x {
                return true;
            }
            match (interval(a, false), interval(b, false)) {
                (Some(a), Some(b)) => {
                    (a.0 - b.0).abs() <= 0.75 * h && (a.1 - b.1).abs() <= 0.75 * h
                }
                _ => false,
            }
        }
        (None, None) => true,
        _ => false,
    }
}
pub(super) fn reference(v: &Value, page: f64) -> f64 {
    interval(v, false).map_or(page, |(l, r)| r - l)
}
pub(super) fn compatible(a: &Value, b: &Value, page: f64, h: f64) -> f64 {
    match (interval(a, false), interval(b, false)) {
        (Some(a), Some(b)) if (a.0 - b.0).abs() <= 0.75 * h && (a.1 - b.1).abs() <= 0.75 * h => {
            (a.1 - a.0).min(b.1 - b.0)
        }
        _ => page,
    }
}
pub(super) fn skips(bs: &[Value], i: usize, j: usize, h: f64) -> Result<bool, Error> {
    let ar = rows(&bs[i])?;
    let br = rows(&bs[j])?;
    let a = ar.last().ok_or_else(invalid)?;
    let b = br.first().ok_or_else(invalid)?;
    let top = cy(a).min(cy(b));
    let bottom = cy(a).max(cy(b));
    let corridor = bbox(&json!([
        a.n[0].min(b.n[0]),
        top,
        a.n[2].max(b.n[2]),
        bottom
    ]))?;
    for (k, v) in bs.iter().enumerate() {
        if k == i || k == j {
            continue;
        }
        if let Some(other) = rows(v)?.first() {
            let c = cy(other);
            if top + 0.1 * h < c && c < bottom - 0.1 * h && axis(&corridor, other, 0) >= 0.2 {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
pub(super) fn conflict(a: &Value, b: &Value) -> bool {
    let Some(a) = a["_font_signatures"].as_array() else {
        return false;
    };
    let Some(b) = b["_font_signatures"].as_array() else {
        return false;
    };
    !a.is_empty() && !b.is_empty() && !a.iter().any(|a| b.iter().any(|b| eq(a, b)))
}
pub(super) fn replace(
    bs: Vec<Value>,
    mut replacements: BTreeMap<usize, Value>,
    consumed: BTreeSet<usize>,
) -> Vec<Value> {
    bs.into_iter()
        .enumerate()
        .filter_map(|(i, b)| {
            if let Some(r) = replacements.remove(&i) {
                Some(r)
            } else {
                (!consumed.contains(&i)).then_some(b)
            }
        })
        .collect()
}
