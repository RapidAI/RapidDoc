//! _detect_table_text/_run_table_ocr line-level policy and _process_single_table model HTML attachment.
use super::coordinates;
use crate::legacy_page::{assets::Pixels, invalid, limit, unsupported, validate_tree};
use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
use rapiddoc_core::Error;
use serde_json::{json, Value};

pub struct Prepared {
    pub metadata: Value,
    pub png: Vec<u8>,
}
pub(super) fn encode(data: &[u8], w: usize, h: usize) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(data, w as u32, h as u32, ExtendedColorType::Rgb8)
        .map_err(|_| invalid())?;
    if out.len() > rapiddoc_core::resources::limit(64_000_000) {
        return Err(limit());
    }
    Ok(out)
}
pub(super) fn rgb(png: &[u8]) -> Result<Pixels, Error> {
    if png.len() < 33 || !png.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(invalid());
    }
    let (w, h) = (
        u32::from_be_bytes(png[16..20].try_into().unwrap()) as usize,
        u32::from_be_bytes(png[20..24].try_into().unwrap()) as usize,
    );
    w.checked_mul(h)
        .filter(|n| *n > 0 && *n <= rapiddoc_core::resources::limit(4_000_000))
        .ok_or_else(limit)?;
    let p = Pixels::decode(png)?;
    if p.channels != 3 {
        return Err(unsupported("表格OCR只接受8-bit RGB PNG"));
    }
    if p.width * p.height > rapiddoc_core::resources::limit(4_000_000) {
        return Err(limit());
    }
    Ok(p)
}
/// 输入为已经变换到表裁图坐标的公式/checkbox bbox，遮罩floor/ceil且右下端不包含。
pub fn detection(png: &[u8], request: &Value) -> Result<Prepared, Error> {
    let p = detection_pixels(png, request)?;
    Ok(Prepared {
        metadata: json!({"width":p.width,"height":p.height}),
        png: encode(&p.data, p.width, p.height)?,
    })
}
/// Production detector consumes the same masked RGB bytes without a PNG round trip.
pub(crate) fn detection_pixels(png: &[u8], request: &Value) -> Result<Pixels, Error> {
    validate_tree(request)?;
    let mut p = rgb(png)?;
    let masks = request["masks"].as_array().ok_or_else(invalid)?;
    if masks.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut area = 0usize;
    for m in masks {
        if m.get("bbox").is_none_or(Value::is_null) {
            continue;
        }
        let b = coordinates(&m["bbox"])?;
        let b = [
            b[0].floor().clamp(0., p.width as f64) as usize,
            b[1].floor().clamp(0., p.height as f64) as usize,
            b[2].ceil().clamp(0., p.width as f64) as usize,
            b[3].ceil().clamp(0., p.height as f64) as usize,
        ];
        area = area
            .checked_add(b[2].saturating_sub(b[0]) * b[3].saturating_sub(b[1]))
            .filter(|v| *v <= rapiddoc_core::resources::limit(64_000_000))
            .ok_or_else(limit)?;
        if b[2] > b[0] && b[3] > b[1] {
            for y in b[1]..b[3] {
                p.data[(y * p.width + b[0]) * 3..(y * p.width + b[2]) * 3].fill(255);
            }
        }
    }
    Ok(p)
}
/// 仅在第一轮检测为空且短边触发后才调用，保留已经遮罩的图，不再次加入公式坐标。
pub fn padding(png: &[u8], padding: usize) -> Result<Prepared, Error> {
    if padding > rapiddoc_core::resources::limit(1024) {
        return Err(limit());
    }
    let p = padding_pixels(rgb(png)?, padding)?;
    Ok(Prepared {
        metadata: json!({"width":p.width,"height":p.height}),
        png: encode(&p.data, p.width, p.height)?,
    })
}
pub(crate) fn padding_pixels(mut p: Pixels, padding: usize) -> Result<Pixels, Error> {
    if padding > rapiddoc_core::resources::limit(1024) {
        return Err(limit());
    }
    let (w, h) = (p.width + padding * 2, p.height + padding * 2);
    let count = w
        .checked_mul(h)
        .filter(|n| *n <= rapiddoc_core::resources::limit(4_000_000))
        .ok_or_else(limit)?;
    let mut data = vec![255; count * 3];
    for y in 0..p.height {
        let dst = ((y + padding) * w + padding) * 3;
        data[dst..dst + p.width * 3]
            .copy_from_slice(&p.data[y * p.width * 3..(y + 1) * p.width * 3]);
    }
    p.width = w;
    p.height = h;
    p.data = data;
    Ok(p)
}
pub fn restore(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let w = request["width"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= rapiddoc_core::resources::limit(4_000_000))
        .ok_or_else(invalid)?;
    let h = request["height"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= rapiddoc_core::resources::limit(4_000_000))
        .ok_or_else(invalid)?;
    if w * h > rapiddoc_core::resources::limit(4_000_000) {
        return Err(limit());
    }
    let pad = request["padding"]
        .as_u64()
        .filter(|n| *n <= rapiddoc_core::resources::limit(1024))
        .ok_or_else(invalid)? as f32;
    let rows = request["boxes"]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
        .ok_or_else(invalid)?;
    let mut out = Vec::new();
    for row in rows {
        let points = row
            .as_array()
            .filter(|a| a.len() == 4)
            .ok_or_else(invalid)?;
        let mut coords = Vec::new();
        for point in points {
            let xy = point
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(invalid)?;
            let mut p = Vec::new();
            for (i, v) in xy.iter().enumerate() {
                let v = v
                    .as_f64()
                    .filter(|v| v.is_finite() && v.abs() <= 1_000_000.)
                    .ok_or_else(invalid)?;
                let max = if i == 0 { w - 1 } else { h - 1 } as f32;
                p.push(((v as f32 - pad).clamp(0., max)) as f64);
            }
            coords.push(p);
        }
        out.push(coords);
    }
    Ok(json!(out))
}
pub fn normalize(text: &str) -> String {
    let text = text.trim_matches(rapiddoc_native::text_boundary::python_space);
    let chars: Vec<_> = text.chars().collect();
    let text = if text == "香" {
        "否"
    } else if text == "哦樂" {
        "哦"
    } else if chars.len() == 2 && chars[0].is_ascii_digit() && chars[1] == '號' {
        &text[..1]
    } else {
        text
    };
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}
/// RapidOcrModel.__call__默认drop_score=0.5；先过滤，才执行表格文字归一化。
pub fn apply_full_ocr(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let boxes = request["boxes"]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
        .ok_or_else(invalid)?;
    let results = request["results"].as_array().ok_or_else(invalid)?;
    if boxes.len() != results.len() {
        return Err(invalid());
    }
    let mut kept_boxes = Vec::new();
    let mut kept_results = Vec::new();
    for (q, r) in boxes.iter().zip(results) {
        let r = r.as_array().filter(|r| r.len() == 2).ok_or_else(invalid)?;
        let score = r[1]
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or_else(invalid)?;
        if score >= 0.5 {
            kept_boxes.push(q.clone());
            kept_results.push(json!(r));
        }
    }
    apply_ocr(&json!({"boxes":kept_boxes,"results":kept_results}))
}
pub fn apply_ocr(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let boxes = request["boxes"]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
        .ok_or_else(invalid)?;
    let results = request["results"].as_array().ok_or_else(invalid)?;
    if boxes.len() != results.len() {
        return Err(invalid());
    }
    if boxes.is_empty() {
        return Ok(json!([]));
    }
    let mut out_boxes = Vec::new();
    let mut texts = Vec::new();
    let mut scores = Vec::new();
    for (b, r) in boxes.iter().zip(results) {
        let b = b.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
        let mut points = Vec::new();
        for p in b {
            let p = p.as_array().filter(|a| a.len() == 2).ok_or_else(invalid)?;
            let mut xy = Vec::new();
            for v in p {
                xy.push(
                    v.as_f64()
                        .filter(|v| v.is_finite() && v.abs() <= 1_000_000.)
                        .ok_or_else(invalid)? as f32 as f64,
                );
            }
            points.push(xy);
        }
        let r = r.as_array().filter(|a| a.len() == 2).ok_or_else(invalid)?;
        let text = if r[0].is_null() {
            ""
        } else {
            r[0].as_str().ok_or_else(invalid)?
        };
        r[1].as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(invalid)?;
        out_boxes.push(points);
        texts.push(normalize(text));
        scores.push(r[1].clone());
    }
    Ok(json!([out_boxes, texts, scores]))
}
/// 模型返回后去layout图片、提取首<table>至末</table>，并保存页级公式/图片坐标。
pub fn apply_prediction(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let mut table = request["table"].clone();
    let obj = table.as_object_mut().ok_or_else(invalid)?;
    obj.remove("layout_image_list");
    let html = if request["html_code"].is_null() {
        return Ok(table);
    } else {
        request["html_code"].as_str().ok_or_else(invalid)?
    };
    if request["text_fallback"] == true {
        table["html"] = json!(html);
        table["text_fallback"] = json!(true);
    } else {
        let (Some(start), Some(end)) = (html.find("<table>"), html.rfind("</table>")) else {
            return Ok(table);
        };
        let end = end + 8;
        table["html"] = json!(if end >= start { &html[start..end] } else { "" });
    }
    let scale = request["scale"]
        .as_f64()
        .filter(|v| v.is_finite() && *v > 0. && *v <= 1000.)
        .ok_or_else(invalid)?;
    for (key, list, source) in [
        ("formula_boxes", "formulas", "bbox"),
        ("img_boxes", "fill_images", "ori_bbox"),
    ] {
        let items = request[list]
            .as_array()
            .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
            .ok_or_else(invalid)?;
        let mut boxes = Vec::new();
        for item in items {
            if item.get("bbox").is_none() {
                continue;
            }
            let b = coordinates(&item[source])?;
            let mut out = Vec::new();
            for n in b {
                let n = n / scale;
                if !n.is_finite() || n.abs() > 1e12 {
                    return Err(limit());
                }
                out.push(n as i64);
            }
            boxes.push(out);
        }
        if !boxes.is_empty() {
            table[key] = json!(boxes);
        }
    }
    Ok(table)
}

#[cfg(test)]
mod pixel_tests {
    use super::*;

    #[test]
    fn masked_rgb_and_thin_retry_preserve_pixels_and_errors() {
        let (w, h) = (13, 7);
        let data: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        let png = encode(&data, w, h).unwrap();
        let request = json!({"masks":[{"bbox":[1.8,-2.0,4.2,3.5]},{"bbox":null}]});
        let p = detection_pixels(&png, &request).unwrap();
        assert_eq!((p.width, p.height, p.channels), (w, h, 3));
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                let expected = if (1..5).contains(&x) && y < 4 {
                    &[255, 255, 255][..]
                } else {
                    &data[i..i + 3]
                };
                assert_eq!(&p.data[i..i + 3], expected);
            }
        }
        let encoded = detection(&png, &request).unwrap();
        assert_eq!(rgb(&encoded.png).unwrap().data, p.data);
        let padded = padding_pixels(p, 8).unwrap();
        let old = padding(&encoded.png, 8).unwrap();
        assert_eq!(rgb(&old.png).unwrap().data, padded.data);
        for y in 0..padded.height {
            for x in 0..padded.width {
                if x < 8 || x >= 8 + w || y < 8 || y >= 8 + h {
                    assert_eq!(
                        &padded.data[(y * padded.width + x) * 3..(y * padded.width + x + 1) * 3],
                        &[255, 255, 255]
                    );
                }
            }
        }
        assert!(detection_pixels(b"bad", &request).is_err());
        assert!(detection_pixels(&png, &json!({"masks":[{"bbox":[1]}]})).is_err());
        assert_eq!(
            padding(b"bad", 1025).err().unwrap().code,
            rapiddoc_core::ErrorCode::ResourceLimit
        );
    }
}
