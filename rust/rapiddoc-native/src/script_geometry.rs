//! 旧_script_geometry的普通JSON适配；不执行inline token精炼或整表语义HTML。
use crate::{
    error, python_decimal,
    text_boundary::{python_alnum, python_space},
};
use rapiddoc_core::{Error, ErrorCode};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
mod kernel;
mod math_identifiers;
pub mod refine;
fn invalid() -> Error {
    error(ErrorCode::InvalidInput, "上下标几何输入无效")
}
fn limit() -> Error {
    error(ErrorCode::ResourceLimit, "上下标几何输入超过预算")
}
fn number(v: &Value) -> Option<f64> {
    let n = if let Some(b) = v.as_bool() {
        f64::from(u8::from(b))
    } else if let Some(n) = v.as_f64() {
        n
    } else {
        v.as_str()?.trim().parse().ok()?
    };
    n.is_finite().then_some(n)
}
fn bbox(v: &Value) -> Option<[f64; 4]> {
    let a = v.as_array()?;
    if a.len() != 4 {
        return None;
    }
    let b = [
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ];
    (b[2] > b[0] && b[3] > b[1]).then_some(b)
}
fn source(c: &Value) -> Option<i64> {
    c.get("char_idx")?.as_i64()
}
fn scalar(v: Option<&Value>) -> Result<String, Error> {
    Ok(match v {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) => "None".into(),
        Some(Value::Bool(b)) => if *b { "True" } else { "False" }.into(),
        Some(n) if n.is_number() => n.to_string(),
        _ => return Err(invalid()),
    })
}
pub fn text_flags(s: &str) -> u32 {
    let nonempty = !s.is_empty();
    let mut chars = s.chars();
    let first = chars.next();
    let single = first.is_some() && chars.next().is_none();
    let local = single
        && first.is_some_and(|c| {
            c.is_ascii_alphanumeric()
                || matches!(c as u32,0xff10..=0xff19|0xff21..=0xff3a|0xff41..=0xff5a)
        });
    u32::from(nonempty && s.chars().all(python_space))
        | (u32::from(matches!(s, "\r" | "\n")) << 1)
        | (u32::from(nonempty && s.chars().all(python_alnum)) << 2)
        | (u32::from(nonempty && s.chars().all(|c| python_decimal(c).is_some())) << 3)
        | (u32::from(local) << 4)
        | (u32::from(matches!(s, "/" | "⁄")) << 5)
        | (u32::from(matches!(s, "," | "，" | "-" | "–" | "—")) << 6)
}
fn records(v: &Value) -> Result<Vec<kernel::Record>, Error> {
    let chars = v["chars"].as_array().ok_or_else(invalid)?;
    if chars.len() > rapiddoc_core::resources::limit(1500) {
        return Err(limit());
    }
    for name in ["tight_bboxes", "origins"] {
        if let Some(map) = v.get(name) {
            if !map.is_object() {
                return Err(invalid());
            }
            if map.as_object().unwrap().len() > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
        }
    }
    let protected: HashSet<usize> = match v.get("protected_body_indices") {
        None => HashSet::new(),
        Some(a) => {
            let a = a
                .as_array()
                .filter(|a| a.len() <= rapiddoc_core::resources::limit(1500))
                .ok_or_else(invalid)?;
            a.iter()
                .map(|i| {
                    i.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(invalid)
                })
                .collect::<Result<_, _>>()?
        }
    };
    let mut fonts = HashMap::new();
    let mut output = Vec::with_capacity(chars.len());
    let mut bytes = 0usize;
    for (i, c) in chars.iter().enumerate() {
        if !c.is_object() {
            return Err(invalid());
        }
        let text = scalar(c.get("char"))?;
        bytes += text.len();
        if text.len() > rapiddoc_core::resources::limit(256)
            || bytes > rapiddoc_core::resources::limit(1_000_000)
        {
            return Err(limit());
        }
        let loose = bbox(&c["bbox"]).unwrap_or([0.; 4]);
        let index = source(c);
        let key = index.map(|n| n.to_string());
        let tight = key.as_ref().and_then(|k| bbox(&v["tight_bboxes"][k]));
        let origin = key
            .as_ref()
            .and_then(|k| v["origins"][k].as_array())
            .and_then(|a| Some([number(a.first()?)?, number(a.get(1)?)?]));
        if loose
            .iter()
            .chain(tight.iter().flatten())
            .chain(origin.iter().flatten())
            .any(|n| n.abs() > 1e9)
        {
            return Err(limit());
        }
        let mut flags = text_flags(&text) | (u32::from(protected.contains(&i)) << 7);
        if flags & 3 == 0 && tight.is_some() && origin.is_some() {
            flags |= 256;
        }
        let font = if let Some(name) = c["font"]["name"].as_str().filter(|s| !s.is_empty()) {
            bytes += name.len();
            if bytes > rapiddoc_core::resources::limit(1_000_000) {
                return Err(limit());
            }
            let key = (
                name,
                c["font"]["flags"].as_i64(),
                c["font"]["weight"].as_i64(),
            );
            let next = fonts.len() as i64;
            *fonts.entry(key).or_insert(next)
        } else {
            -1
        };
        output.push((loose, tight, origin, flags, font));
    }
    Ok(output)
}
pub fn classify(v: &Value) -> Result<Value, Error> {
    let result = kernel::classify(records(v)?).ok_or_else(invalid)?;
    Ok(json!(result
        .into_iter()
        .map(|r| ["body", "sup", "sub"][r as usize])
        .collect::<Vec<_>>()))
}
/// 给后续token精炼复用的数值引用补充；roles必须与chars一一对应。
pub fn numeric_indices(v: &Value) -> Result<Value, Error> {
    let records = records(v)?;
    let roles = match v.get("roles") {
        None => vec![0; records.len()],
        Some(a) => {
            let a = a
                .as_array()
                .filter(|a| a.len() == records.len())
                .ok_or_else(invalid)?;
            a.iter()
                .map(|r| match r.as_str() {
                    Some("body") => Ok(0),
                    Some("sup") => Ok(1),
                    Some("sub") => Ok(2),
                    _ => Err(invalid()),
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let mut indices = kernel::numeric_indices(records, &roles);
    indices.sort_unstable();
    Ok(json!(indices))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_script_geometry_and_unicode_flags() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/script-geometry.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                classify(&c["request"]).unwrap(),
                c["expected"]["roles"],
                "{}",
                c["label"]
            );
            assert_eq!(
                numeric_indices(&c["request"]).unwrap(),
                c["expected"]["numeric_indices"],
                "{}",
                c["label"]
            );
        }
        let extra: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/script-numeric-extra.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in extra["cases"].as_array().unwrap() {
            assert_eq!(
                numeric_indices(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        let mut facts = f["flags"].as_array().unwrap().iter().peekable();
        for code in 0..=0x10ffff {
            let Some(c) = char::from_u32(code) else {
                continue;
            };
            let mut buffer = [0; 4];
            let s = c.encode_utf8(&mut buffer);
            let expected = if facts.peek().is_some_and(|f| f["text"].as_str() == Some(s)) {
                facts.next().unwrap()["expected"].as_u64().unwrap()
            } else {
                0
            };
            assert_eq!(u64::from(text_flags(s)), expected, "U+{code:04X}");
        }
        assert!(facts.next().is_none());
    }
}
