//! 旧 txt_spans_extract/fill_char_in_spans；JSON 坐标替代 pdftext Bbox 对象。
use super::*;
use rapiddoc_native::text_boundary::python_space;
use std::collections::BTreeMap;

pub(crate) const STOP: &[&str] = &[
    ".", "!", "?", "。", "！", "？", ")", "）", "\"", "”", ":", "：", ";", "；", "]", "】", "}",
    ">", "》", "、", ",", "，", "-", "—", "–",
];
pub(crate) const START: &[&str] = &["(", "（", "\"", "“", "【", "{", "《", "<", "「", "『", "["];
fn median(mut a: Vec<f64>) -> f64 {
    a.sort_by(f64::total_cmp);
    let n = a.len();
    if n % 2 == 1 {
        a[n / 2]
    } else {
        (a[n / 2 - 1] + a[n / 2]) / 2.
    }
}
fn difference(a: &Value, b: &Value) -> Value {
    if let (Some(a), Some(b)) = (a.as_i64(), b.as_i64()) {
        json!(a - b)
    } else {
        json!(num(a) - num(b))
    }
}
fn grid_index(value: f64, height: f64) -> Result<i64, Error> {
    let value = value / height;
    if !value.is_finite() || value < i64::MIN as f64 || value >= i64::MAX as f64 {
        return Err(limit());
    }
    Ok(value as i64)
}
pub(crate) fn rotation(v: &Value) -> bool {
    let r = if !truth(v) {
        0.
    } else if let Some(n) = v.as_f64() {
        n
    } else if let Some(s) = v.as_str() {
        s.trim_matches(python_space)
            .parse::<f64>()
            .unwrap_or(f64::NAN)
    } else if v == true {
        1.
    } else {
        f64::NAN
    };
    let degrees = (r * (180. / std::f64::consts::PI)).rem_euclid(360.);
    r.is_finite()
        && [0., 90., 180., 270.]
            .iter()
            .any(|angle| ((degrees - angle + 180.).rem_euclid(360.) - 180.).abs() < 0.1)
}
fn inside_char(c: &Value, s: &Value) -> bool {
    let a = bb(&c["bbox"]);
    let b = bb(&s["bbox"]);
    let x = (a[0] + a[2]) / 2.;
    let y = (a[1] + a[3]) / 2.;
    let h = b[3] - b[1];
    if !(b[1] < y && y < b[3] && (y - (b[1] + b[3]) / 2.).abs() < h * 0.33) {
        return false;
    }
    if b[0] < x && x < b[2] {
        return true;
    }
    let text = c["char"].as_str().unwrap();
    if STOP.contains(&text) {
        b[2] - h < a[0] && a[0] < b[2] && x > b[0]
    } else if START.contains(&text) {
        b[0] < a[2] && a[2] < b[0] + h && x < b[2]
    } else {
        false
    }
}
#[derive(Default)]
pub(crate) struct FillMode {
    pub word_boxes: bool,
    pub boxed_chars: bool,
}
fn content(
    span: &mut Value,
    chars: &[Value],
    work: &mut Work,
    mode: &FillMode,
) -> Result<(), Error> {
    let words = mode.word_boxes;
    if chars.is_empty() {
        return Ok(());
    }
    let mut chars = chars.to_vec();
    chars.sort_by_key(|c| c["char_idx"].as_i64().unwrap());
    if words {
        // 表格适配器随后按useful_list/scale转换原生字符框；raw word文本不做连字或Unicode替换。
        span["_native_words"] = json!(chars
            .iter()
            .map(|c| json!([c["char"], 1, c["bbox"]]))
            .collect::<Vec<_>>());
    }
    let width = median(
        chars
            .iter()
            .map(|c| {
                let b = bb(&c["bbox"]);
                b[2] - b[0]
            })
            .collect(),
    );
    let whitespace = chars.iter().any(|c| {
        let s = c["char"].as_str().unwrap();
        !s.is_empty() && s.chars().all(python_space)
    });
    let mut text = String::new();
    for (position, c) in chars.iter().enumerate() {
        // 保留旧 list.index(dict) 在完全重复字典上取首项的行为。
        let mut first = position;
        // pdftext.Bbox未实现值相等；每个JSON字符出现位置投影为独立Bbox对象。
        if !mode.boxed_chars {
            for (i, other) in chars.iter().enumerate() {
                work.charge(1)?;
                if eq(c, other) {
                    first = i;
                    break;
                }
            }
        }
        let s = c["char"].as_str().unwrap();
        text.push_str(s);
        if let Some(next) = chars.get(first + 1) {
            if !words
                && !whitespace
                && bb(&next["bbox"])[0] - bb(&c["bbox"])[2] > width * 0.25
                && s != " "
                && next["char"] != " "
            {
                text.push(' ');
            }
        }
    }
    text = text.replace("\r\n", "").replace('\u{0002}', "-");
    for _ in 0..2 {
        for (a, b) in [
            ("ﬁ", "fi"),
            ("ﬂ", "fl"),
            ("ﬀ", "ff"),
            ("ﬃ", "ffi"),
            ("ﬄ", "ffl"),
            ("ﬅ", "ft"),
            ("ﬆ", "st"),
        ] {
            text = text.replace(a, b);
        }
    }
    span["content"] = json!(text.trim_matches(python_space));
    Ok(())
}

pub(super) fn fill<F>(
    page: &Value,
    spans: &mut Vec<Value>,
    blocks: &[Value],
    discarded: &[Value],
    work: &mut Work,
    crop: F,
) -> Result<(), Error>
where
    F: FnMut(&Value) -> Result<Option<String>, Error>,
{
    fill_mode(
        page,
        spans,
        blocks,
        discarded,
        work,
        FillMode::default(),
        crop,
    )
}

pub(crate) fn fill_mode<F>(
    page: &Value,
    spans: &mut Vec<Value>,
    blocks: &[Value],
    discarded: &[Value],
    work: &mut Work,
    mode: FillMode,
    mut crop: F,
) -> Result<(), Error>
where
    F: FnMut(&Value) -> Result<Option<String>, Error>,
{
    validate_tree(page)?;
    if spans.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    if spans
        .iter()
        .any(|s| !s.is_object() || !s["type"].is_string())
    {
        return Err(invalid());
    }
    let mut chars = Vec::new();
    let mut lines = Vec::new();
    for block in page["blocks"].as_array().ok_or_else(invalid)? {
        for line in block["lines"].as_array().ok_or_else(invalid)? {
            if !rotation(&line["rotation"]) {
                continue;
            }
            coord(&line["bbox"], false)?;
            lines.push(line);
            if lines.len() > rapiddoc_core::resources::limit(10_000) {
                return Err(limit());
            }
            for native_span in line["spans"].as_array().ok_or_else(invalid)? {
                if !native_span["text"].is_string() {
                    return Err(invalid());
                }
                for c in native_span["chars"].as_array().ok_or_else(invalid)? {
                    coord(&c["bbox"], false)?;
                    if !c["char"].is_string()
                        || !c["char_idx"].as_i64().is_some_and(|n| {
                            n.unsigned_abs() <= rapiddoc_core::resources::limit(1_000_000_000)
                        })
                    {
                        return Err(invalid());
                    }
                    chars.push(c.clone());
                    if chars.len() > rapiddoc_core::resources::limit(100_000) {
                        return Err(limit());
                    }
                }
            }
        }
    }
    let mut heights = Vec::new();
    for s in spans
        .iter_mut()
        .filter(|s| s["type"] == "text" && !truth(&s["docvortex_native_text"]))
    {
        coord(&s["bbox"], true)?;
        if !s["content"].is_string() {
            return Err(invalid());
        }
        let b = bb(&s["bbox"]);
        s["height"] = difference(&s["bbox"][3], &s["bbox"][1]);
        s["width"] = difference(&s["bbox"][2], &s["bbox"][0]);
        heights.push(b[3] - b[1]);
    }
    if heights.is_empty() {
        return Ok(());
    }
    let median_h = median(heights);
    let mut useful = Vec::new();
    let mut unused = Vec::new();
    let mut vertical = Vec::new();
    for (i, s) in spans
        .iter()
        .enumerate()
        .filter(|(_, s)| s["type"] == "text" && !truth(&s["docvortex_native_text"]))
    {
        for (j, b) in blocks.iter().chain(discarded).enumerate() {
            work.charge(1)?;
            let coords = b.as_array().filter(|b| b.len() >= 8).ok_or_else(invalid)?;
            if matches!(
                b[7].as_str(),
                Some("image_body" | "table_body" | "interline_equation")
            ) {
                continue;
            }
            let bbox = Value::Array(coords[..4].to_vec());
            coord(&bbox, false)?;
            if overlap(&s["bbox"], &bbox) > 0.5 {
                let h = num(&s["height"]);
                let w = num(&s["width"]);
                if h > median_h * 3. && h > w * 3. {
                    vertical.push(i);
                } else if j < blocks.len() || blocks.iter().any(|a| eq(a, b)) {
                    useful.push(i);
                } else {
                    unused.push(i);
                }
                break;
            }
        }
    }
    let mut removed = vec![false; spans.len()];
    for line in lines {
        for &i in &vertical {
            work.charge(1)?;
            if overlap(&line["bbox"], &spans[i]["bbox"]) > 0.5 {
                let mut text = spans[i]["content"].as_str().unwrap().to_owned();
                for s in line["spans"].as_array().unwrap() {
                    text.push_str(s["text"].as_str().unwrap());
                }
                spans[i]["content"] = json!(text);
                break;
            }
        }
    }
    for &i in &vertical {
        if spans[i]["content"] == "" {
            removed[i] = true;
        }
    }
    let mut selected = useful;
    selected.extend(unused);
    selected.sort_by(|&a, &b| {
        num(&spans[a]["bbox"][1])
            .partial_cmp(&num(&spans[b]["bbox"][1]))
            .unwrap()
    });
    let mut grid: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for &i in &selected {
        let b = bb(&spans[i]["bbox"]);
        let start = grid_index(b[1], median_h)?;
        let end = grid_index(b[3], median_h)?;
        let count = end
            .checked_sub(start)
            .and_then(|n| n.checked_add(1))
            .filter(|n| *n <= rapiddoc_core::resources::limit(100_001) && *n > 0)
            .ok_or_else(limit)?;
        work.charge(count as usize)?;
        for row in start..=end {
            grid.entry(row).or_default().push(i);
        }
    }
    let mut assigned: Vec<Vec<Value>> = (0..spans.len()).map(|_| Vec::new()).collect();
    for c in chars {
        let b = bb(&c["bbox"]);
        let key = grid_index((b[1] + b[3]) / 2., median_h)?;
        if let Some(candidates) = grid.get(&key) {
            for &i in candidates {
                work.charge(1)?;
                if inside_char(&c, &spans[i]) {
                    assigned[i].push(c);
                    break;
                }
            }
        }
    }
    let mut need = Vec::new();
    for i in selected {
        content(&mut spans[i], &assigned[i], work, &mode)?;
        let s = &mut spans[i];
        if s["content"].as_str().unwrap().chars().count() as f64 * num(&s["height"])
            < num(&s["width"]) * 0.5
        {
            need.push(i);
        }
        let s = s.as_object_mut().unwrap();
        s.remove("height");
        s.remove("width");
        s.remove("chars");
    }
    for i in need {
        if let Some(id) = crop(&spans[i]["bbox"])? {
            spans[i]["content"] = json!("");
            spans[i]["score"] = json!(1.);
            spans[i]["np_img"] = json!(id);
        } else {
            removed[i] = true;
        }
    }
    let mut i = 0;
    spans.retain(|_| {
        let keep = !removed[i];
        i += 1;
        keep
    });
    Ok(())
}

// NumPy 2.4 的连续 FP64 pairwise reduction 次序，许可见 licenses/NUMPY-PAIRWISE.txt。
pub(crate) fn pairwise(values: &[f64]) -> f64 {
    if values.len() < 8 {
        return values.iter().fold(-0., |a, b| a + b);
    }
    if values.len() > 128 {
        let middle = (values.len() / 2) / 8 * 8;
        return pairwise(&values[..middle]) + pairwise(&values[middle..]);
    }
    let mut sums: [f64; 8] = values[..8].try_into().unwrap();
    let full = values.len() / 8 * 8;
    for chunk in values[8..full].as_chunks::<8>().0 {
        for i in 0..8 {
            sums[i] += chunk[i];
        }
    }
    let sum =
        ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
    values[full..].iter().fold(sum, |a, b| a + b)
}
pub(super) fn contrast(rgb: &[u8]) -> f64 {
    let gray: Vec<_> = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            ((p[0] as u32 * 9798 + p[1] as u32 * 19235 + p[2] as u32 * 3735 + 16384) >> 15) as f64
        })
        .collect();
    let mean = gray.iter().sum::<f64>() / gray.len() as f64;
    let squared: Vec<_> = gray
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .collect();
    let std = (pairwise(&squared) / gray.len() as f64).sqrt();
    ((std / (mean + 1e-6)) * 100.).round_ties_even() / 100.
}

#[cfg(test)]
mod tests {
    #[test]
    fn old_opencv_numpy_contrast_rounding_matches() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let data: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/native-contrast.json")).unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let rgb: Vec<_> = c["rgb"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            assert_eq!(super::contrast(&rgb), c["expected"].as_f64().unwrap());
        }
    }
}
