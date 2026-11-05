//! 旧 txt_spans_bbox_extract 与 _extract_text_from_pdf；保留分组、FP64/FP32 和回退语义。
use super::coordinates;
use crate::legacy_page::native_text::{pairwise, rotation, START, STOP};
use crate::legacy_page::{invalid, limit, truth, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
pub mod detection;

pub(super) fn number(v: &Value) -> Result<f64, Error> {
    v.as_f64()
        .filter(|n| n.is_finite() && n.abs() <= 1e12)
        .ok_or_else(invalid)
}
pub(super) fn op(a: &Value, b: &Value, subtract: bool) -> Result<Value, Error> {
    if let (Some(a), Some(b)) = (a.as_i64(), b.as_i64()) {
        return Ok(json!(if subtract {
            a.checked_sub(b)
        } else {
            a.checked_add(b)
        }
        .ok_or_else(limit)?));
    }
    Ok(json!(if subtract {
        number(a)? - number(b)?
    } else {
        number(a)? + number(b)?
    }))
}
fn compare(a: &Value, b: &Value) -> std::cmp::Ordering {
    a.as_f64()
        .unwrap()
        .partial_cmp(&b.as_f64().unwrap())
        .unwrap()
}
fn poly(b: &[Value; 4]) -> Value {
    json!([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
}
pub(super) fn useful(v: &Value) -> Result<[Value; 8], Error> {
    let a = v.as_array().filter(|a| a.len() == 8).ok_or_else(invalid)?;
    for n in a {
        number(n)?;
    }
    if number(&a[6])? <= 0. || number(&a[7])? <= 0. {
        return Err(invalid());
    }
    Ok(a.clone().try_into().unwrap())
}
#[derive(Clone)]
struct Span {
    bbox: [Value; 4],
    cy: f64,
    height: f64,
    size: f64,
}
pub(super) fn inside(a: [f64; 4], b: [f64; 4], text: &str) -> bool {
    let x = (a[0] + a[2]) / 2.;
    let y = (a[1] + a[3]) / 2.;
    let h = b[3] - b[1];
    if b[0] < x && x < b[2] && b[1] < y && y < b[3] {
        return true;
    }
    if STOP.contains(&text) {
        b[2] - h < a[0] && a[0] < b[2] && x > b[0] && b[1] < y && y < b[3]
    } else if START.contains(&text) {
        b[0] < a[2] && a[2] < b[0] + h && x < b[2] && b[1] < y && y < b[3]
    } else {
        false
    }
}
fn merged(mut spans: Vec<Span>, work: &mut Work) -> Result<Vec<[Value; 4]>, Error> {
    spans.sort_by(|a, b| {
        compare(&a.bbox[1], &b.bbox[1]).then_with(|| compare(&a.bbox[0], &b.bbox[0]))
    });
    let mut lines: Vec<Vec<Span>> = Vec::new();
    for s in spans {
        let mut dest = None;
        for (i, line) in lines.iter().enumerate() {
            work.charge(line.len())?;
            let heights = line.iter().map(|s| s.height).collect::<Vec<_>>();
            let centers = line.iter().map(|s| s.cy).collect::<Vec<_>>();
            let h = pairwise(&heights) / line.len() as f64;
            let cy = pairwise(&centers) / line.len() as f64;
            if (s.cy - cy).abs() < h * 0.8 {
                dest = Some(i);
                break;
            }
        }
        if let Some(i) = dest {
            lines[i].push(s);
        } else {
            lines.push(vec![s]);
        }
    }
    let mut out = Vec::new();
    for mut line in lines {
        line.sort_by(|a, b| compare(&a.bbox[0], &b.bbox[0]));
        let mut current = line[0].clone();
        for s in &line[1..] {
            work.charge(1)?;
            if number(&s.bbox[0])? - number(&current.bbox[2])? <= (current.size + s.size) / 2. * 0.6
            {
                for i in 0..4 {
                    let order = compare(&s.bbox[i], &current.bbox[i]);
                    if (i < 2 && order.is_lt()) || (i >= 2 && order.is_gt()) {
                        current.bbox[i] = s.bbox[i].clone();
                    }
                }
                // 字号沿用 current 的首 span，而非合并后的均值。
            } else {
                out.push(current.bbox);
                current = s.clone();
            }
        }
        out.push(current.bbox);
    }
    Ok(out)
}
fn split(boxes: Vec<Value>, masks: &[Value], work: &mut Work) -> Result<Vec<Value>, Error> {
    let masks = masks
        .iter()
        .map(|v| coordinates(&v["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = Vec::new();
    for p in boxes {
        let b = [
            number(&p[0][0])?,
            number(&p[0][1])?,
            number(&p[2][0])?,
            number(&p[2][1])?,
        ];
        let h = b[3] - b[1];
        let mut intervals = Vec::new();
        for mask in &masks {
            work.charge(1)?;
            let min_h = h.min(mask[3] - mask[1]);
            if min_h > 0. && (b[3].min(mask[3]) - b[1].max(mask[1])).max(0.) / min_h > 0.8 {
                intervals.push([mask[0], mask[2]]);
            }
        }
        intervals.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        let mut merged: Vec<[f64; 2]> = Vec::new();
        for m in intervals {
            if let Some(last) = merged.last_mut().filter(|last| last[1] >= m[0]) {
                last[1] = last[1].max(m[1]);
            } else {
                merged.push(m);
            }
        }
        let mut start = b[0];
        let mut ranges = Vec::new();
        for m in merged {
            if m[0] > b[2] || m[1] < start {
                continue;
            }
            if start < m[0] {
                ranges.push([start, m[0] - 1.]);
            }
            start = (m[1] + 1.).max(start);
        }
        if start <= b[2] {
            ranges.push([start, b[2]]);
        }
        for range in ranges {
            let f = |n: f64| json!(n as f32 as f64);
            out.push(poly(&[f(range[0]), f(b[1]), f(range[1]), f(b[3])]));
            if out.len() > rapiddoc_core::resources::limit(20_000) {
                return Err(limit());
            }
        }
    }
    Ok(out)
}
/// 返回原生裁图坐标检测框；masks 为已相对 useful_list 调整的公式/复选框 bbox。
pub fn boxes(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let mut work = Work(0);
    let mut region = request["region"].clone();
    let masks = match request.get("masks") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.clone(),
        _ => return Err(invalid()),
    };
    if masks.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let result = derive(
        &request["native_page"],
        &mut region,
        request.get("scale").unwrap_or(&json!(1.0)),
        &useful(&request["useful_list"])?,
        &masks,
        &mut work,
    )?;
    Ok(json!({"region":region,"boxes":result}))
}
fn derive(
    page: &Value,
    region: &mut Value,
    scale: &Value,
    u: &[Value; 8],
    masks: &[Value],
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    let factor = number(scale)?;
    if factor <= 0. || factor > 1000. {
        return Err(invalid());
    }
    let region_bbox = super::rect(region)?.map(|n| n / factor);
    let blocks = page["blocks"].as_array().ok_or_else(invalid)?;
    let mut spans = Vec::new();
    let mut seen = 0usize;
    for block in blocks {
        for line in block["lines"].as_array().ok_or_else(invalid)? {
            if !rotation(&line["rotation"]) {
                continue;
            }
            for s in line["spans"].as_array().ok_or_else(invalid)? {
                seen += 1;
                work.charge(1)?;
                if seen > rapiddoc_core::resources::limit(100_000) {
                    return Err(limit());
                }
                let b = coordinates(&s["bbox"])?;
                let text = s["text"].as_str().ok_or_else(invalid)?;
                if !inside(b, region_bbox, text) {
                    continue;
                }
                let size = if let Some(font) = s.get("font") {
                    if !font.is_object() {
                        return Err(invalid());
                    }
                    font.get("size").map_or(Ok(10.), number)?
                } else {
                    10.
                };
                spans.push(Span {
                    bbox: s["bbox"].as_array().unwrap().clone().try_into().unwrap(),
                    cy: (b[1] + b[3]) / 2.,
                    height: b[3] - b[1],
                    size,
                });
            }
        }
    }
    let mut boxes = Vec::new();
    for b in merged(spans, work)? {
        let mut out: [Value; 4] = std::array::from_fn(|_| Value::Null);
        for i in 0..4 {
            let n = if let (Some(a), Some(b)) = (b[i].as_i64(), scale.as_i64()) {
                json!(a.checked_mul(b).ok_or_else(limit)?)
            } else {
                json!(number(&b[i])? * factor)
            };
            let n = op(&op(&n, &u[i % 2], false)?, &u[2 + i % 2], true)?;
            out[i] = if n.is_i64() || n.is_u64() {
                n
            } else {
                json!(format!("{:.2}", number(&n)?)
                    .parse::<f64>()
                    .map_err(|_| invalid())?)
            };
        }
        boxes.push(poly(&out));
    }
    let boxes = if masks.is_empty() {
        boxes
    } else {
        split(boxes, masks, work)?
    };
    if boxes.is_empty() {
        region["need_ocr_det"] = json!(true);
    }
    Ok(boxes)
}
pub(super) fn adjusted(masks: &[Value], u: &[Value; 8]) -> Result<Vec<Value>, Error> {
    let mut out = Vec::new();
    for m in masks {
        coordinates(&m["bbox"])?;
        let mut b = Vec::new();
        for i in 0..4 {
            b.push(op(
                &op(&m["bbox"][i], &u[2 + i % 2], true)?,
                &u[i % 2],
                false,
            )?);
        }
        if number(&b[2])? < 0.
            || number(&b[3])? < 0.
            || number(&b[0])? > number(&u[6])?
            || number(&b[1])? > number(&u[7])?
        {
            continue;
        }
        out.push(json!({"bbox":b}));
    }
    Ok(out)
}
fn native_dets(
    boxes: &[Value],
    u: &[Value; 8],
    region: &Value,
    fp32: bool,
) -> Result<Vec<Value>, Error> {
    let mut out = Vec::new();
    for b in boxes {
        if number(&b[2][0])? - number(&b[0][0])? < 3. {
            continue;
        }
        let mut poly = Vec::new();
        for p in b.as_array().unwrap() {
            for k in 0..2 {
                let n = number(&p[k])?;
                poly.push(if fp32 {
                    json!(((n as f32 - number(&u[k])? as f32) + number(&u[2 + k])? as f32) as f64)
                } else {
                    json!(n - number(&u[k])? + number(&u[2 + k])?)
                });
            }
        }
        out.push(json!({"category_id":15,"original_label":region["original_label"],"original_order":region["original_order"],"poly":poly,"score":1.0,"text":""}));
    }
    Ok(out)
}
/// 原生阶段追加空文字 span；需要 OCR 检测的任务另列，未调用任何模型。
pub fn plan(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let plan = &request["plan"];
    let tasks = plan["ocr"].as_array().ok_or_else(invalid)?;
    if tasks.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mode = request
        .get("use_det_mode")
        .and_then(Value::as_str)
        .unwrap_or("auto");
    if !matches!(mode, "auto" | "txt" | "ocr") {
        return Err(invalid());
    }
    let enabled = request
        .get("ocr_enable")
        .map_or(Some(false), Value::as_bool)
        .ok_or_else(invalid)?;
    let enabled = enabled
        || (mode != "ocr"
            && matches!(
                request["native_page"]["rotate_label"].as_str(),
                Some("90" | "180" | "270")
            ));
    let mut model = plan["model"].clone();
    let dets = model["layout_dets"].as_array_mut().ok_or_else(invalid)?;
    if dets.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut masks = Vec::new();
    for categories in [&[8, 13, 14][..], &[200][..]] {
        for d in dets.iter() {
            if categories.iter().any(|n| d["category_id"] == *n) {
                coordinates(&d["bbox"])?;
                masks.push(d.clone());
            }
        }
    }
    if masks.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut native = Vec::new();
    let mut pending = Vec::new();
    let mut added = Vec::new();
    let mut work = Work(0);
    for t in tasks {
        let index = t["layout_index"].as_u64().ok_or_else(invalid)? as usize;
        let r = dets.get_mut(index).ok_or_else(invalid)?;
        let u = useful(&t["useful_list"])?;
        let m = adjusted(&masks, &u)?;
        if !enabled && mode != "ocr" {
            if let Some(text) = r
                .get("native_text")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                added.push(json!({"category_id":15,"poly":r["poly"],"score":1.0,"text":text,"docvortex_native_text":true,"original_label":r["original_label"],"original_order":r["original_order"]}));
            } else {
                let boxes = derive(
                    &request["native_page"],
                    r,
                    request.get("scale").unwrap_or(&json!(1.0)),
                    &u,
                    &m,
                    &mut work,
                )?;
                added.extend(native_dets(&boxes, &u, r, !m.is_empty())?);
                native.push(json!({"layout_index":index,"boxes":boxes}));
            }
        }
        let need = r.get("need_ocr_det").is_some_and(truth);
        if enabled || need || mode == "ocr" {
            r.as_object_mut()
                .ok_or_else(invalid)?
                .remove("need_ocr_det");
            let mut task = t.clone();
            task["masks"] = json!(m);
            task["ocr_enable"] = json!(enabled || need);
            pending.push(task);
        }
        if added.len() > rapiddoc_core::resources::limit(20_000) {
            return Err(limit());
        }
    }
    dets.extend(added);
    Ok(json!({"model":model,"native":native,"detect":pending}))
}
