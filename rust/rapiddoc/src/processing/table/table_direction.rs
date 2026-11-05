//! 旧表格原生文字角度投票、RapidOrientationModel调用门槛和正交像素旋转。
use super::table_text::Prepared;
use super::{coordinates, native_ocr, table_text};
use crate::legacy_page::{invalid, limit, truth, unsupported, validate_tree};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};

// JSON适配器接收有限number或ASCII float字符串；无效旧float转换仍走旧fallback。
fn rotation(v: &Value) -> Result<Option<f64>, Error> {
    if !truth(v) {
        return Ok(Some(0.));
    }
    let n = if let Some(n) = v.as_f64() {
        Some(n)
    } else if let Some(b) = v.as_bool() {
        Some(if b { 1. } else { 0. })
    } else if let Some(s) = v.as_str() {
        let s = s.trim_matches(|c: char| c.is_whitespace());
        if !s.is_ascii() {
            return Err(unsupported(
                "表格角度字符串仅支持ASCII float；可先转换为JSON number",
            ));
        }
        if s.contains('_') {
            return Err(unsupported(
                "表格角度字符串不支持数字下划线；可先转换为JSON number",
            ));
        }
        s.parse::<f64>().ok()
    } else {
        None
    };
    Ok(n.filter(|n| n.is_finite()))
}

pub fn plan(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let width = request["width"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(invalid)?;
    let height = request["height"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(invalid)?;
    width
        .checked_mul(height)
        .filter(|n| *n <= rapiddoc_core::resources::limit(4_000_000))
        .ok_or_else(limit)?;
    let scale = request["scale"]
        .as_f64()
        .filter(|n| n.is_finite() && *n > 0. && *n <= 1000.)
        .ok_or_else(invalid)?;
    let page = &request["page"];
    if !page.is_object() {
        return Err(invalid());
    }
    let poly = request["table"]["poly"]
        .as_array()
        .filter(|p| p.len() == 8)
        .ok_or_else(invalid)?;
    let b = coordinates(&json!([poly[0], poly[1], poly[4], poly[5]]))?.map(|x| x / scale);
    if b.iter().any(|x| !x.is_finite() || x.abs() > 1e12) {
        return Err(limit());
    }
    let boxes = request["boxes"]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
        .ok_or_else(invalid)?;
    for q in boxes {
        let q = q.as_array().filter(|q| q.len() == 4).ok_or_else(invalid)?;
        for xy in q {
            let xy = xy.as_array().filter(|q| q.len() == 2).ok_or_else(invalid)?;
            for j in 0..2 {
                xy[j]
                    .as_f64()
                    .filter(|n| n.is_finite() && n.abs() <= 1e6)
                    .ok_or_else(invalid)?;
            }
        }
    }
    let pdf_not_rotate = !matches!(page["rotate_label"].as_str(), Some("90" | "180" | "270"));
    let mut angles = Vec::new();
    if pdf_not_rotate {
        let page_rotation = rotation(&page["rotation"])?.unwrap_or(0.).rem_euclid(360.);
        let blocks = page["blocks"].as_array().ok_or_else(invalid)?;
        for block in blocks {
            for line in block["lines"].as_array().ok_or_else(invalid)? {
                let Some(r) = rotation(&line["rotation"])? else {
                    continue;
                };
                let degrees = r * (180. / std::f64::consts::PI) + page_rotation;
                if !degrees.is_finite() || degrees.abs() > 1e12 {
                    return Err(limit());
                }
                // Python math.degrees -> round(ties-even) -> integer modulo, including negative angles.
                let angle = (degrees.round_ties_even() as i64).rem_euclid(360);
                for span in line["spans"].as_array().ok_or_else(invalid)? {
                    let bbox = coordinates(&span["bbox"])?;
                    let text = span["text"].as_str().ok_or_else(invalid)?;
                    if !text.trim_matches(python_space).is_empty()
                        && native_ocr::inside(bbox, b, text)
                    {
                        angles.push(angle);
                    }
                }
            }
        }
    }
    // Counter.most_common的同票规则：首次出现的角度优先，而不是数值较小的角度。
    let mut counts = [0usize; 360];
    for a in &angles {
        counts[*a as usize] += 1;
    }
    let mut winner = 0;
    let mut best = 0;
    for a in &angles {
        if counts[*a as usize] > best {
            best = counts[*a as usize];
            winner = *a;
        }
    }
    let needs_classifier = angles.is_empty() && height as f64 / width as f64 > 1.2;
    Ok(
        json!({"rotate_label":winner.to_string(),"angles":angles,"needs_classifier":needs_classifier,"pdf_not_rotate":pdf_not_rotate}),
    )
}

/// Conservative model-only correction, scoped to table crops, not whole pages.
pub fn select_model_rotation(request: &Value) -> Result<Value, Error> {
    let label = request["rotation_degrees"]
        .as_u64()
        .filter(|v| matches!(v, 0 | 90 | 180 | 270))
        .ok_or_else(invalid)?;
    let score = request["score"]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(invalid)?;
    let margin = request["margin"]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(invalid)?;
    Ok(json!(
        if matches!(label, 90 | 270) && score >= 0.9 && margin >= 0.2 {
            label.to_string()
        } else {
            "0".into()
        }
    ))
}

/// classifier_bgr保留_process_single_table向旧分类器传BGR ndarray的实际通道次序。
pub fn dimensions(png: &[u8]) -> Result<(usize, usize), Error> {
    // Keep full RGB decoding/validation; reading only the header would accept bad PNGs.
    let p = table_text::rgb(png)?;
    Ok((p.width, p.height))
}

/// Internal no-op rotation avoids re-encoding validated RGB pixels. Other angles
/// retain image() semantics, including the original identity behavior for 180.
pub fn oriented_png<'a>(png: &'a [u8], angle: &str) -> Result<std::borrow::Cow<'a, [u8]>, Error> {
    if !matches!(angle, "90" | "270") {
        dimensions(png)?;
        Ok(std::borrow::Cow::Borrowed(png))
    } else {
        Ok(std::borrow::Cow::Owned(image(png, angle, false)?.png))
    }
}

pub fn image(png: &[u8], angle: &str, classifier_bgr: bool) -> Result<Prepared, Error> {
    let p = table_text::rgb(png)?;
    let (w, h) = if matches!(angle, "90" | "270") {
        (p.height, p.width)
    } else {
        (p.width, p.height)
    };
    let mut out = vec![0; p.data.len()];
    for y in 0..p.height {
        for x in 0..p.width {
            let (xx, yy) = match angle {
                "90" => (y, p.width - 1 - x),
                "270" => (p.height - 1 - y, x),
                _ => (x, y),
            };
            let src = (y * p.width + x) * 3;
            let dst = (yy * w + xx) * 3;
            out[dst..dst + 3].copy_from_slice(&p.data[src..src + 3]);
            if classifier_bgr {
                out.swap(dst, dst + 2);
            }
        }
    }
    Ok(Prepared {
        metadata: json!({"width":w,"height":h}),
        png: table_text::encode(&out, w, h)?,
    })
}
