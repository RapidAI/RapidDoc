//! 旧表格公式/图片坐标提示与RapidTableModel.predict的结构模型输入准备。
use super::{
    coordinates,
    native_ocr::{number, op, useful},
    table_text,
};
use crate::legacy_page::{invalid, limit, truth, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};

fn list(v: &Value) -> Result<&[Value], Error> {
    if v.is_null() {
        return Ok(&[]);
    }
    v.as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
        .map(Vec::as_slice)
        .ok_or_else(invalid)
}
fn flag(v: &Value, k: &str) -> Result<bool, Error> {
    v[k].as_bool().ok_or_else(invalid)
}
fn poly_bbox(v: &Value) -> Result<Value, Error> {
    let p = v.as_array().filter(|p| p.len() == 8).ok_or_else(invalid)?;
    let b = json!([p[0], p[1], p[4], p[5]]);
    coordinates(&b)?;
    Ok(b)
}
fn points(b: &Value) -> Value {
    json!([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
}
fn inside(a: [f64; 4], b: [f64; 4]) -> bool {
    a[0] >= b[0] && a[1] >= b[1] && a[2] <= b[2] && a[3] <= b[3]
}
fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let (x0, y0, x1, y1) = (
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    );
    if x1 < x0 || y1 < y0 {
        return 0.;
    }
    let aa = (a[2] - a[0]) * (a[3] - a[1]);
    let bb = (b[2] - b[0]) * (b[3] - b[1]);
    if aa == 0. || bb == 0. {
        return 0.;
    }
    let area = (x1 - x0) * (y1 - y0);
    area / (aa + bb - area)
}
fn local(b: &Value, u: &[Value; 8]) -> Result<Value, Error> {
    coordinates(b)?;
    Ok(Value::Array(
        (0..4)
            .map(|i| op(&op(&b[i], &u[i % 2], false)?, &u[2 + i % 2], true))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
/// 返回JSON副本；图片像素由宿主按uuid持有，不将PIL对象序列化为JSON。
pub fn plan(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let mut page = request["page"].clone();
    let mut table = request["table"].clone();
    if !page.is_object() || !table.is_object() {
        return Err(invalid());
    }
    let u = useful(&request["useful_list"])?;
    let scale = request["scale"]
        .as_f64()
        .filter(|n| n.is_finite() && *n > 0. && *n <= 1000.)
        .ok_or_else(invalid)?;
    let mut formulas = Vec::new();
    if flag(request, "formula_enabled")? {
        for m in list(&request["formulas"])?
            .iter()
            .chain(list(&request["checkboxes"])?)
        {
            let b = &m["bbox"];
            coordinates(b)?;
            let mut out = Vec::new();
            for i in 0..4 {
                out.push(op(&op(&b[i], &u[2 + i % 2], true)?, &u[i % 2], false)?);
            }
            if number(&out[2])? < 0.
                || number(&out[3])? < 0.
                || number(&out[0])? > number(&u[6])?
                || number(&out[1])? > number(&u[7])?
            {
                continue;
            }
            let mut adjusted = json!({"bbox":out});
            for k in ["latex", "checkbox"] {
                if truth(&m[k]) {
                    adjusted[k] = m[k].clone();
                }
            }
            formulas.push(adjusted);
        }
    }
    let mut fills = Vec::new();
    if flag(request, "image_enabled")? {
        let table_box = coordinates(&poly_bbox(&table["poly"])?)?.map(|n| n / scale);
        if table_box.iter().any(|n| !n.is_finite() || n.abs() > 1e12) {
            return Err(limit());
        }
        let original = list(&page["ori_image_list"])?;
        let extract = flag(request, "extract_original_image")?
            && !matches!(page["rotate_label"].as_str(), Some("90" | "180" | "270"));
        let mut updated = original.to_vec();
        if extract {
            for image in &mut updated {
                let b = coordinates(&image["bbox"])?;
                if inside(b, table_box) && iou(b, table_box) < 0.9 {
                    let mut scaled = Vec::new();
                    for i in 0..4 {
                        scaled.push(
                            if let (Some(a), Some(s)) =
                                (image["bbox"][i].as_i64(), request["scale"].as_i64())
                            {
                                json!(a.checked_mul(s).ok_or_else(limit)?)
                            } else {
                                json!(number(&image["bbox"][i])? * scale)
                            },
                        );
                    }
                    image["ori_bbox"] = json!(scaled);
                    image["ocr_bbox"] = points(&local(&image["ori_bbox"], &u)?);
                    fills.push(image.clone());
                }
            }
        }
        if extract && !updated.is_empty() {
            page["ori_image_list"] = json!(updated);
        }
        if fills.is_empty() && truth(&table["layout_image_list"]) {
            let mut layout = list(&table["layout_image_list"])?.to_vec();
            for image in &mut layout {
                let b = poly_bbox(&image["poly"])?;
                image["ori_bbox"] = b.clone();
                image["bbox"] = b;
                image["ocr_bbox"] = points(&local(&image["ori_bbox"], &u)?);
                fills.push(image.clone());
            }
            table["layout_image_list"] = json!(layout);
        }
        if !truth(&page["table_fill_image_list"]) {
            page["table_fill_image_list"] = json!(fills);
        } else {
            let previous = page["table_fill_image_list"]
                .as_array_mut()
                .ok_or_else(invalid)?;
            if previous.len() + fills.len() > rapiddoc_core::resources::limit(2000) {
                return Err(limit());
            }
            previous.extend(fills.clone());
        }
    }
    Ok(json!({"page":page,"table":table,"formulas":formulas,"fill_images":fills}))
}

fn point_bbox(q: &Value) -> Result<[f64; 4], Error> {
    let q = q.as_array().filter(|q| q.len() == 4).ok_or_else(invalid)?;
    for p in q {
        let p = p.as_array().filter(|p| p.len() == 2).ok_or_else(invalid)?;
        for v in p {
            coordinates(&json!([v, v, v, v]))?;
        }
    }
    // 旧points_to_bbox取p1.x/p1.y/p2.x/p3.y，而不是四点的min/max。
    coordinates(&json!([q[0][0], q[0][1], q[1][0], q[2][1]]))
}
pub fn payload(png: &[u8], request: &Value) -> Result<table_text::Prepared, Error> {
    validate_tree(request)?;
    let p = table_text::rgb(png)?;
    let mut data = p.data;
    let mut work = Work(0);
    let mut area = 0usize;
    let input = &request["ocr_result"];
    if !truth(input) {
        return Ok(table_text::Prepared {
            metadata: json!({"ocr_result":null,"needs_ocr":true}),
            png: table_text::encode(&data, p.width, p.height)?,
        });
    }
    let columns = input
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(invalid)?;
    let mut rows = Vec::new();
    let n = columns[0]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(20_000))
        .ok_or_else(invalid)?
        .len();
    for a in columns {
        if a.as_array().is_none_or(|a| a.len() != n) {
            return Err(invalid());
        }
    }
    for ((q, text), score) in columns[0]
        .as_array()
        .unwrap()
        .iter()
        .zip(columns[1].as_array().unwrap())
        .zip(columns[2].as_array().unwrap())
    {
        point_bbox(q)?;
        text.as_str().ok_or_else(invalid)?;
        number(score)?;
        rows.push([q.clone(), text.clone(), score.clone()]);
    }
    let skip = flag(request, "skip_text_in_image")?;
    for image in list(&request["fill_images"])? {
        let b = point_bbox(&image["ocr_bbox"])?;
        let uuid = image["uuid"].as_str().ok_or_else(invalid)?;
        let ints = b.map(|n| n as i64);
        let (x0, y0, x1, y1) = (
            ints[0].min(ints[2]),
            ints[1].min(ints[3]),
            ints[0].max(ints[2]),
            ints[1].max(ints[3]),
        );
        let x0 = x0.clamp(0, p.width as i64) as usize;
        let y0 = y0.clamp(0, p.height as i64) as usize;
        let x1 = (x1 + 1).clamp(0, p.width as i64) as usize;
        let y1 = (y1 + 1).clamp(0, p.height as i64) as usize;
        area = area
            .checked_add((x1 - x0) * (y1 - y0))
            .filter(|n| *n <= rapiddoc_core::resources::limit(64_000_000))
            .ok_or_else(limit)?;
        for y in y0..y1 {
            data[(y * p.width + x0) * 3..(y * p.width + x1) * 3].fill(255);
        }
        if skip {
            let mut kept = Vec::new();
            for row in rows {
                work.charge(1)?;
                if !inside(point_bbox(&row[0])?, b) {
                    kept.push(row);
                }
            }
            rows = kept;
        }
        rows.push([image["ocr_bbox"].clone(), json!(uuid), json!(1)]);
        if rows.len() > rapiddoc_core::resources::limit(20_000) {
            return Err(limit());
        }
    }
    let left = request["inline_left"]
        .as_str()
        .filter(|s| s.len() <= 64)
        .ok_or_else(invalid)?;
    let right = request["inline_right"]
        .as_str()
        .filter(|s| s.len() <= 64)
        .ok_or_else(invalid)?;
    for m in list(&request["formulas"])? {
        let text = if truth(&m["latex"]) {
            format!("{left}{}{right}", m["latex"].as_str().ok_or_else(invalid)?)
        } else if truth(&m["checkbox"]) {
            m["checkbox"].as_str().ok_or_else(invalid)?.to_owned()
        } else {
            continue;
        };
        let b = coordinates(&m["bbox"])?.map(|n| n as f32 as f64);
        rows.push([
            points(&json!(b)),
            json!(table_text::normalize(&text)),
            json!(1),
        ]);
        if rows.len() > rapiddoc_core::resources::limit(20_000) {
            return Err(limit());
        }
    }
    let columns = (0..3)
        .map(|k| rows.iter().map(|r| r[k].clone()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    Ok(table_text::Prepared {
        metadata: json!({"ocr_result":columns,"needs_ocr":false}),
        png: table_text::encode(&data, p.width, p.height)?,
    })
}
