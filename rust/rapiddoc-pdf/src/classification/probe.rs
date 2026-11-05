//! Original image/text coverage and paint evidence, with no Pillow round-trip.
use rapiddoc_core::{Error, Limits};
use serde_json::{json, Value};
use std::path::Path;
type Box4 = [f64; 4];
fn bbox(v: &Value) -> Box4 {
    let a = v.as_array().unwrap();
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
        a[3].as_f64().unwrap(),
    ]
}
fn area(b: Box4) -> f64 {
    (b[2] - b[0]).max(0.) * (b[3] - b[1]).max(0.)
}
fn intersect(a: Box4, b: Box4) -> Box4 {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}
fn union_area(boxes: &[Box4]) -> f64 {
    let boxes = boxes
        .iter()
        .copied()
        .filter(|b| area(*b) != 0.)
        .collect::<Vec<_>>();
    let mut edges = boxes.iter().flat_map(|b| [b[0], b[2]]).collect::<Vec<_>>();
    edges.sort_by(f64::total_cmp);
    edges.dedup();
    let mut total = 0.;
    for pair in edges.windows(2) {
        let (left, right) = (pair[0], pair[1]);
        let mut intervals = boxes
            .iter()
            .filter(|b| b[0] < right && b[2] > left)
            .map(|b| (b[1], b[3]))
            .collect::<Vec<_>>();
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut end = f64::NEG_INFINITY;
        for (bottom, top) in intervals {
            total += (right - left) * (top - bottom.max(end)).max(0.);
            end = end.max(top);
        }
    }
    total
}
fn objects(facts: &Value) -> impl Iterator<Item = (i64, Box4)> + '_ {
    facts["objects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| (o[0].as_i64().unwrap(), bbox(&o[1])))
}
pub fn sparse_bad(facts: &Value, sample: &Value) -> bool {
    if super::rules::quality(&[sample.clone()])["abnormal_ratio"]
        .as_f64()
        .unwrap()
        >= 0.03
    {
        return true;
    }
    let total = sample["char_count"].as_f64().unwrap();
    if total > 0. && sample["unicode_map_error_count"].as_f64().unwrap() / total >= 0.04 {
        return true;
    }
    let frame = bbox(&facts["bbox"]);
    let empty = sample["cleaned_text"].as_str().unwrap().is_empty();
    let mut content = false;
    let mut images = vec![];
    for (kind, b) in objects(facts).filter(|(k, _)| empty || *k == 3) {
        let b = intersect(b, frame);
        if area(b) != 0. {
            content = true;
            if kind == 3 {
                images.push(b)
            }
        }
    }
    (area(frame) != 0. && union_area(&images) / area(frame) >= 0.8) || (empty && content)
}
pub fn high_coverage(facts: &Value) -> bool {
    let frame = bbox(&facts["bbox"]);
    let a = (frame[2] - frame[0]).abs() * (frame[3] - frame[1]).abs();
    let sum = objects(facts)
        .filter(|(k, _)| *k == 3)
        .map(|(_, b)| area(b))
        .sum::<f64>();
    a > 0. && (sum / a).min(1.) >= 0.8
}
fn visual(b: Box4, frame: Box4, rotation: i64) -> Box4 {
    let [l, bottom, r, t] = frame;
    let mut points = vec![];
    for x in [b[0], b[2]] {
        for y in [b[1], b[3]] {
            points.push(match rotation {
                90 => [y - bottom, x - l],
                180 => [r - x, y - bottom],
                270 => [t - y, r - x],
                _ => [x - l, t - y],
            });
        }
    }
    [
        points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
        points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
        points
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max),
        points
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max),
    ]
}
pub fn image_text(
    library: &Path,
    data: &[u8],
    actual: usize,
    facts: &Value,
    limits: &Limits,
) -> Result<Value, Error> {
    let mut result =
        json!({"image_backed_ocr":false,"candidate_chars":0,"unpainted_chars":0,"probed":false});
    let frame = bbox(&facts["bbox"]);
    let images = objects(facts)
        .filter(|(k, _)| *k == 3)
        .map(|(_, b)| intersect(b, frame))
        .filter(|b| area(*b) != 0.)
        .collect::<Vec<_>>();
    if area(frame) == 0. || union_area(&images) / area(frame) < 0.8 {
        return Ok(result);
    }
    let mut boxes = vec![];
    let mut valid = 0usize;
    for c in facts["chars"].as_array().unwrap() {
        let code = c[0].as_u64().unwrap_or(0) as u32;
        if code == 0 || code > 0x10ffff || char::from_u32(code).is_some_and(super::rules::space) {
            continue;
        }
        valid += 1;
        let b = intersect(bbox(&c[4]), frame);
        if area(b) != 0.
            && union_area(
                &images
                    .iter()
                    .map(|&image| intersect(b, image))
                    .collect::<Vec<_>>(),
            ) >= area(b) * 0.8
        {
            boxes.push(visual(b, frame, facts["rotation"].as_i64().unwrap()));
        }
    }
    result["candidate_chars"] = json!(boxes.len());
    if (boxes.len() as f64) < 50f64.max(valid as f64 * 0.5) {
        return Ok(result);
    }
    let width = facts["size"][0].as_f64().unwrap();
    let height = facts["size"][1].as_f64().unwrap();
    let png = super::difference(
        library,
        data,
        actual,
        1.5f64.min(1024. / width.max(height)),
        None,
        limits,
    )?;
    let image = image::load_from_memory(&png)
        .map_err(|e| super::error(&e.to_string()))?
        .into_luma8();
    result["probed"] = json!(true);
    let (sx, sy) = (image.width() as f64 / width, image.height() as f64 / height);
    let mut unpainted = 0;
    for b in boxes {
        let x0 = (b[0] * sx).floor().max(0.) as u32;
        let y0 = (b[1] * sy).floor().max(0.) as u32;
        let x1 = (b[2] * sx).ceil().min(image.width() as f64).max(0.) as u32;
        let y1 = (b[3] * sy).ceil().min(image.height() as f64).max(0.) as u32;
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let painted = (y0..y1).any(|y| (x0..x1).any(|x| image.get_pixel(x, y)[0] != 0));
        if !painted {
            unpainted += 1
        }
    }
    result["unpainted_chars"] = json!(unpainted);
    result["image_backed_ocr"] = json!((unpainted as f64) >= 50f64.max(valid as f64 * 0.5));
    Ok(result)
}
