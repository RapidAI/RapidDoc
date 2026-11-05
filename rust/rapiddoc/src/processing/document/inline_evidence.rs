//! 旧inline.detection字体/绘图样式与链接注解的普通JSON检测阶段。
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{inline_evidence as kernel, python_printable, text_boundary::python_space};
use serde_json::{json, Value};
use std::collections::BTreeSet;
type Box4 = [f64; 4];
fn number(v: &Value) -> Result<f64, Error> {
    let n = v
        .as_f64()
        .or_else(|| v.as_bool().map(|b| u8::from(b) as f64))
        .filter(|n| n.is_finite())
        .ok_or_else(invalid)?;
    if n.abs() > 1e9 {
        return Err(limit());
    }
    Ok(n)
}
fn bbox(v: &Value) -> Result<Option<Box4>, Error> {
    if v.is_null() {
        return Ok(None);
    }
    let Some(a) = v.as_array().filter(|a| a.len() == 4) else {
        return Ok(None);
    };
    let b = [
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ];
    Ok((b[2] > b[0] && b[3] > b[1]).then_some(b))
}
fn array(v: &Value, n: usize) -> Result<&Vec<Value>, Error> {
    let a = v.as_array().ok_or_else(invalid)?;
    let n = if n > 8 {
        crate::legacy_page::budget::records(n)
    } else {
        n
    };
    if a.len() > n {
        return Err(limit());
    }
    Ok(a)
}
fn optional<'a>(v: &'a Value, k: &str) -> Result<&'a [Value], Error> {
    v.get(k)
        .map_or(Ok(&[][..]), |x| array(x, 5000).map(Vec::as_slice))
}
fn optional_input(v: Option<&Value>) -> Result<&[Value], Error> {
    v.map_or(Ok(&[][..]), |v| array(v, 5000).map(Vec::as_slice))
}
pub(super) fn compact(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{200b}' | '\u{2060}' | '\u{feff}' | '\u{ad}' => {}
            '\u{2}' => out.push('-'),
            'ﬀ' => out.push_str("ff"),
            'ﬁ' => out.push_str("fi"),
            'ﬂ' => out.push_str("fl"),
            'ﬃ' => out.push_str("ffi"),
            'ﬄ' => out.push_str("ffl"),
            'ﬅ' | 'ﬆ' => out.push_str("st"),
            _ if python_space(c) || matches!(c as u32,0..=8|11..=12|14..=31|127..=159) => {}
            _ => out.push(c),
        }
    }
    out
}
fn integer(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_bool().map(i64::from))
}
struct Line<'a> {
    b: Option<Box4>,
    angle: i64,
    source: i64,
    chars: Vec<&'a Value>,
}
fn line(v: &Value) -> Result<Line<'_>, Error> {
    let mut chars = optional_chars(v)?;
    if chars.iter().all(|c| integer(&c["char_idx"]).is_some()) {
        chars.sort_by_key(|c| integer(&c["char_idx"]).unwrap());
    }
    Ok(Line {
        b: bbox(&v["bbox"])?,
        angle: integer(&v["angle"]).unwrap_or(0).rem_euclid(360),
        source: integer(&v["source_index"]).ok_or_else(invalid)?,
        chars,
    })
}
fn optional_chars(v: &Value) -> Result<Vec<&Value>, Error> {
    let a = v
        .get("chars")
        .map_or(Ok(&[][..]), |v| array(v, 20000).map(Vec::as_slice))?;
    let mut out = Vec::new();
    for c in a {
        if !c.is_object() {
            continue;
        }
        c["char"]
            .as_str()
            .filter(|s| s.len() <= rapiddoc_core::resources::limit(256))
            .ok_or_else(invalid)?;
        out.push(c)
    }
    Ok(out)
}
fn styles(lines: &[Line<'_>], drawings: &[Value], work: &mut Work) -> Result<Value, Error> {
    let mut chars = Vec::new();
    let mut inputs = Vec::new();
    let mut rules = Vec::new();
    for l in lines {
        if l.angle != 0 {
            continue;
        }
        let Some(b) = l.b else { continue };
        let mut members = Vec::new();
        for (i, c) in l.chars.iter().enumerate() {
            let s = c["char"].as_str().unwrap();
            let text = compact(s);
            let marker = !text.is_empty()
                && text
                    .chars()
                    .all(|c| "•◦‣⁃▪▫●○■□∙·▶▷►▸▹\u{f0b7}".contains(c));
            let rawbox = bbox(&c["bbox"])?.unwrap_or([0.; 4]);
            members.push(chars.len());
            chars.push(kernel::Character {
                bbox: rawbox,
                tight: bbox(&c["tight_bbox"])?,
                index: i,
                fragment: text,
                visible: s.chars().all(python_printable)
                    && !(!s.is_empty() && s.chars().all(python_space)),
                space: !s.is_empty() && s.chars().all(python_space),
                marker,
                bold: super::table_text_styles::font_styles(c).0,
            });
        }
        inputs.push(kernel::Line {
            bbox: b,
            source: l.source,
            members,
        });
    }
    for d in drawings {
        if d["orientation"] != "horizontal" {
            continue;
        }
        let Some(b) = bbox(&d["bbox"])? else { continue };
        rules.push(kernel::Drawing {
            bbox: b,
            width: if d["width"].is_null() {
                0.
            } else {
                number(&d["width"])?
            }
            .max(0.),
        });
    }
    work.charge(
        chars.len().saturating_mul(8).saturating_add(
            rules.len().saturating_mul(
                chars
                    .len()
                    .saturating_add(inputs.len().saturating_mul(inputs.len()))
                    .saturating_add(1),
            ),
        ),
    )?;
    let result = kernel::styles(&chars, inputs, &rules).ok_or_else(limit)?;
    Ok(json!(result
        .into_iter()
        .map(|(bbox, text, ranges, source_index)| {
            let ranges: Vec<_> = ranges
                .into_iter()
                .map(|(start, end, bits)| {
                    let mut styles = Vec::new();
                    for (mask, s) in [(1, "bold"), (2, "underline"), (4, "strikethrough")] {
                        if bits & mask != 0 {
                            styles.push(s)
                        }
                    }
                    json!({"start":start,"end":end,"styles":styles})
                })
                .collect();
            json!({"bbox":bbox,"text":text,"style_ranges":ranges,"source_index":source_index})
        })
        .collect::<Vec<_>>()))
}
struct Annotation<'a> {
    target: &'a str,
    boxes: Vec<Box4>,
}
fn intersection(a: Box4, b: Box4) -> f64 {
    (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.)
}
fn hit(region: Box4, b: Box4) -> bool {
    let x = (b[0] + b[2]) / 2.;
    let y = (b[1] + b[3]) / 2.;
    (region[0] <= x && x <= region[2] && region[1] <= y && y <= region[3])
        || intersection(region, b) / ((b[2] - b[0]) * (b[3] - b[1])).max(0.01) >= 0.5
}
fn links(lines: &[Line<'_>], annotations: &[Value], work: &mut Work) -> Result<Value, Error> {
    if annotations.is_empty() {
        return Ok(json!([]));
    }
    let mut ann = Vec::new();
    let mut regions = 0usize;
    for a in annotations {
        let target = a["target"].as_str().ok_or_else(invalid)?;
        let mut boxes = Vec::new();
        for b in array(&a["bboxes"], 5000)? {
            let b = bbox(b)?.ok_or_else(invalid)?;
            boxes.push(b)
        }
        regions = regions.saturating_add(boxes.len());
        if regions > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        ann.push(Annotation { target, boxes });
    }
    let n: usize = lines.iter().map(|l| l.chars.len()).sum();
    work.charge((lines.len().saturating_add(n)).saturating_mul(regions))?;
    let mut out = Vec::new();
    for l in lines {
        if !matches!(l.angle, 0 | 90 | 180 | 270) {
            continue;
        }
        let Some(b) = l.b else { continue };
        let nearby: Vec<_> = ann
            .iter()
            .filter(|a| a.boxes.iter().any(|r| intersection(b, *r) > 0.))
            .collect();
        if nearby.is_empty() {
            continue;
        }
        let mut text = String::new();
        let mut offset = 0usize;
        let mut active: Option<&str> = None;
        let mut start = 0usize;
        let mut ranges = Vec::new();
        for c in &l.chars {
            let fragment = compact(c["char"].as_str().unwrap());
            if fragment.is_empty() {
                continue;
            }
            let mut targets = BTreeSet::new();
            if let Some(b) = bbox(&c["bbox"])? {
                for a in &nearby {
                    if a.boxes.iter().any(|r| hit(*r, b)) {
                        targets.insert(a.target);
                    }
                }
            }
            let target = if targets.len() == 1 {
                targets.first().copied()
            } else {
                None
            };
            if target != active {
                if let Some(target) = active {
                    ranges.push(json!({"start":start,"end":offset,"target":target}))
                }
                start = offset;
                active = target;
            }
            offset += fragment.chars().count();
            text.push_str(&fragment);
        }
        if let Some(target) = active {
            ranges.push(json!({"start":start,"end":offset,"target":target}))
        }
        if !text.is_empty() && !ranges.is_empty() {
            out.push(json!({"bbox":b,"text":text,"link_ranges":ranges,"source_index":l.source}))
        }
    }
    out.sort_by(|a, b| {
        a["source_index"]
            .as_i64()
            .cmp(&b["source_index"].as_i64())
            .then_with(|| {
                a["bbox"][1]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["bbox"][1].as_f64().unwrap())
            })
            .then_with(|| {
                a["bbox"][0]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["bbox"][0].as_f64().unwrap())
            })
    });
    Ok(json!(out))
}
pub fn detect(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    detect_with_work(v, &mut Work(0))
}
pub(super) fn detect_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    detect_parts(
        &v["lines"],
        v.get("drawing_lines"),
        v.get("annotations"),
        work,
    )
}
pub(super) fn detect_parts(
    raw: &Value,
    drawings: Option<&Value>,
    annotations: Option<&Value>,
    work: &mut Work,
) -> Result<Value, Error> {
    let raw = array(raw, 4096)?;
    let lines = raw.iter().map(line).collect::<Result<Vec<_>, _>>()?;
    let bytes: usize = lines
        .iter()
        .flat_map(|l| &l.chars)
        .map(|c| c["char"].as_str().unwrap().len())
        .sum();
    work.charge(bytes.saturating_mul(4))?;
    let result = json!({"style_lines":styles(&lines,optional_input(drawings)?,work)?,"link_lines":links(&lines,optional_input(annotations)?,work)?});
    if crate::legacy_page::budget::output_exceeds_limit(
        &result,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn same(a: &Value, b: &Value, path: &str) {
        match (a, b) {
            (Value::Array(a), Value::Array(b)) => {
                assert_eq!(a.len(), b.len(), "{path}/length");
                for (i, (a, b)) in a.iter().zip(b).enumerate() {
                    same(a, b, &format!("{path}/{i}"))
                }
            }
            (Value::Object(a), Value::Object(b)) => {
                assert_eq!(
                    a.keys().collect::<Vec<_>>(),
                    b.keys().collect::<Vec<_>>(),
                    "{path}/keys"
                );
                for (k, a) in a {
                    same(a, &b[k], &format!("{path}/{k}"))
                }
            }
            _ => assert_eq!(a, b, "{path}"),
        }
    }
    #[test]
    fn old_style_and_link_evidence() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-inline-evidence.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-inline-evidence-edges.json");
        let g: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"]
            .as_array()
            .unwrap()
            .iter()
            .chain(g["cases"].as_array().unwrap())
        {
            let r = detect(&c["request"]).unwrap_or_else(|e| panic!("{}: {e:?}", c["label"]));
            same(&r, &c["expected"], c["label"].as_str().unwrap())
        }
    }
    #[test]
    fn inline_evidence_keeps_incoming_budget() {
        let q = json!({"lines":[{"text":"ab","angle":0,"source_index":0,"bbox":[0.,0.,20.,10.],"chars":[{"char":"ab","bbox":[0.,0.,20.,10.]}]}]});
        let mut work = Work(9_999_999);
        assert_eq!(
            detect_with_work(&q, &mut work).unwrap_err().code,
            rapiddoc_core::ErrorCode::ResourceLimit
        );
        assert!(detect(&q).is_ok());
    }
}
