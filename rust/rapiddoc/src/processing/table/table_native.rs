//! 旧get_ocr_result_list_table/_extract_table_text_from_pdf；复用原生字符分配内核。
use super::{
    coordinates,
    native_ocr::{number, op, useful},
    table_text,
};
use crate::legacy_page::{assets::NativeCrops, invalid, limit, native_text, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};

fn factor(request: &Value) -> Result<f64, Error> {
    request["scale"]
        .as_f64()
        .filter(|n| n.is_finite() && *n > 0. && *n <= 1000.)
        .ok_or_else(invalid)
}
/// 保留原始det四点；仅用于原生填充的bbox按旧倾斜判断纠正，再转全局坐标并向零取整。
pub fn spans(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let scale = factor(request)?;
    let u = useful(&request["useful_list"])?;
    let boxes = request["boxes"]
        .as_array()
        .filter(|b| b.len() <= rapiddoc_core::resources::limit(1000))
        .ok_or_else(invalid)?;
    let mut spans = Vec::new();
    for q in boxes {
        let a = q.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
        let mut p = [[0.; 2]; 4];
        for (i, v) in a.iter().enumerate() {
            let v = v.as_array().filter(|v| v.len() == 2).ok_or_else(invalid)?;
            for j in 0..2 {
                p[i][j] = v[j]
                    .as_f64()
                    .filter(|n| n.is_finite() && n.abs() <= 1e6)
                    .ok_or_else(invalid)?;
            }
        }
        let width = p[2][0] - p[0][0];
        if width < 3. {
            continue;
        }
        let h = ((p[3][1] - p[0][1]) + (p[2][1] - p[1][1])) / 2.;
        if !(0.8 * h <= p[2][1] - p[0][1] && p[2][1] - p[0][1] <= 1.2 * h) {
            let x = p.iter().map(|p| p[0]).sum::<f64>() / 4.;
            let y = p.iter().map(|p| p[1]).sum::<f64>() / 4.;
            p[0] = [x - width / 2., y - h / 2.];
            p[2] = [x + width / 2., y + h / 2.];
        }
        let mut bbox = [0i64; 4];
        for i in 0..4 {
            let v = (p[if i < 2 { 0 } else { 2 }][i % 2] - number(&u[i % 2])?
                + number(&u[2 + i % 2])?)
                / scale;
            if !v.is_finite() || v.abs() > 1e9 {
                return Err(limit());
            }
            bbox[i] = v as i64;
        }
        spans.push(json!({"ori_bbox":q,"bbox":bbox,"score":1,"content":"","type":"text"}));
    }
    Ok(json!(spans))
}
fn word_points(b: &Value, u: &[Value; 8], scale: &Value) -> Result<Value, Error> {
    coordinates(b)?;
    let mut out = Vec::new();
    for i in 0..4 {
        let product = if let (Some(a), Some(b)) = (b[i].as_i64(), scale.as_i64()) {
            json!(a.checked_mul(b).ok_or_else(limit)?)
        } else {
            json!(number(&b[i])? * number(scale)?)
        };
        let v = op(&op(&product, &u[i % 2], false)?, &u[2 + i % 2], true)?;
        number(&v)?;
        out.push(v);
    }
    Ok(json!([
        [out[0], out[1]],
        [out[2], out[1]],
        [out[2], out[3]],
        [out[0], out[3]]
    ]))
}
pub fn extract(png: &[u8], request: &Value) -> Result<Value, Error> {
    let mut spans = spans(request)?.as_array().unwrap().clone();
    let scale = factor(request)?;
    let u = useful(&request["useful_list"])?;
    let words = request["word_boxes"].as_bool().ok_or_else(invalid)?;
    let p = table_text::rgb(png)?;
    let poly = request["table"]["poly"]
        .as_array()
        .filter(|a| a.len() == 8)
        .ok_or_else(invalid)?;
    let bbox = coordinates(&json!([poly[0], poly[1], poly[4], poly[5]]))?.map(|x| x / scale);
    if bbox.iter().any(|n| !n.is_finite() || n.abs() > 1e9) {
        return Err(limit());
    }
    let block = json!([
        bbox[0] as i64,
        bbox[1] as i64,
        bbox[2] as i64,
        bbox[3] as i64,
        null,
        null,
        null,
        "text",
        null,
        null,
        null,
        null,
        1
    ]);
    let mut crops = NativeCrops::new(&p, &json!({"scale":scale,"page_index":0}))?;
    let mut work = Work(0);
    native_text::fill_mode(
        &request["native_page"],
        &mut spans,
        &[block],
        &[],
        &mut work,
        native_text::FillMode {
            word_boxes: words,
            boxed_chars: true,
        },
        |b| crops.crop(b),
    )?;
    let (mut boxes, mut texts, mut scores) = (Vec::new(), Vec::new(), Vec::new());
    for span in spans {
        if words {
            if let Some(ws) = span["_native_words"].as_array() {
                for w in ws {
                    if boxes.len() >= rapiddoc_core::resources::limit(20_000) {
                        return Err(limit());
                    }
                    boxes.push(word_points(&w[2], &u, &request["scale"])?);
                    texts.push(json!(table_text::normalize(w[0].as_str().unwrap())));
                    scores.push(json!(1));
                }
            }
        } else if let Some(text) = span["content"].as_str().filter(|s| !s.is_empty()) {
            boxes.push(span["ori_bbox"].clone());
            texts.push(json!(table_text::normalize(text)));
            scores.push(span["score"].clone());
        }
    }
    if boxes.is_empty() {
        Ok(json!([]))
    } else {
        Ok(json!([boxes, texts, scores]))
    }
}
