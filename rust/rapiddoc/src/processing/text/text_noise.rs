//! 旧浅灰短数字噪声候选、按需颜色请求和原始来源号清理。
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{
    python_decimal,
    text_boundary::python_space,
    text_noise::{numeric_noise_candidates, NoiseLine},
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn array(v: &Value, max: usize) -> Result<&[Value], Error> {
    let a = v.as_array().ok_or_else(invalid)?;
    let max = if max > 8 {
        crate::legacy_page::budget::records(max)
    } else {
        max
    };
    if a.len() > max {
        return Err(limit());
    }
    Ok(a)
}
fn num(v: &Value) -> Result<f64, Error> {
    let n = v.as_f64().filter(|n| n.is_finite()).ok_or_else(invalid)?;
    if n.abs() > 1e9 {
        return Err(limit());
    }
    Ok(n)
}
fn bbox(v: &Value) -> Result<[f64; 4], Error> {
    let a = array(v, 4)?;
    if a.len() != 4 {
        return Err(invalid());
    }
    Ok([num(&a[0])?, num(&a[1])?, num(&a[2])?, num(&a[3])?])
}
fn index(v: &Value) -> Result<i64, Error> {
    v.as_i64().ok_or_else(invalid)
}
fn visible(v: &Value, work: &mut Work) -> Result<bool, Error> {
    let s = v
        .get("char")
        .map_or(Ok(""), |v| v.as_str().ok_or_else(invalid))?;
    work.charge(s.len() + 1)?;
    Ok(!s.trim_matches(python_space).is_empty())
}
fn char_index(v: &Value) -> Result<Option<i64>, Error> {
    match v.get("char_idx") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => Ok(Some(index(v)?)),
    }
}
fn chars(v: &Value) -> Result<&[Value], Error> {
    match v.get("chars") {
        None => Ok(&[]),
        Some(v) => array(v, 20000),
    }
}
fn signature(v: &Value) -> Result<(String, bool), Error> {
    if v.is_null() {
        return Ok(("null".into(), false));
    }
    let a = array(v, 32)?;
    let mut key = Vec::new();
    for p in a {
        key.push(match p {
            Value::String(s) if s.len() <= rapiddoc_core::resources::limit(256) => p.clone(),
            Value::Null => Value::Null,
            Value::Bool(b) => json!(i32::from(*b)),
            Value::Number(_) if p.as_i64().is_some() => p.clone(),
            _ => return Err(invalid()),
        });
    }
    Ok((
        serde_json::to_string(&key).map_err(|_| invalid())?,
        !a.is_empty(),
    ))
}
fn features(s: &str) -> (bool, bool, bool, bool) {
    let t = s.trim_matches(python_space);
    let c: Vec<_> = t.chars().collect();
    let digit = (1..=3).contains(&c.len()) && c.iter().all(|c| python_decimal(*c).is_some());
    let (mut english, mut run, mut chinese) = (0usize, 0usize, 0usize);
    for c in s.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_alphabetic() {
            run += 1
        } else {
            if run >= 2 {
                english += 1
            }
            run = 0;
        }
        if ('\u{3400}'..='\u{9fff}').contains(&c) {
            chinese += 1;
        }
    }
    let mut i = 0;
    while i < c.len() && i < 2 && python_decimal(c[i]).is_some() {
        i += 1
    }
    let entry = if i > 0 && i < c.len() && matches!(c[i], '.' | ')' | '、') {
        i += 1;
        while i < c.len() && python_space(c[i]) {
            i += 1
        }
        i < c.len() && (c[i].is_ascii_alphabetic() || ('\u{3400}'..='\u{9fff}').contains(&c[i]))
    } else {
        false
    };
    (
        digit,
        english + chinese >= 5,
        english >= 5 || chinese >= 12,
        entry,
    )
}
type Candidates = Vec<(usize, Vec<usize>)>;
fn detect<'a>(v: &'a Value, work: &mut Work) -> Result<(&'a Value, Candidates), Error> {
    validate_tree(v)?;
    let source = &v["source"];
    source.as_object().ok_or_else(invalid)?;
    let lines = array(&source["lines"], 4096)?;
    array(&source["chars"], 20000)?;
    let mut fonts = BTreeMap::new();
    let mut records: Vec<NoiseLine> = Vec::new();
    let mut candidates = 0;
    for line in lines {
        line.as_object().ok_or_else(invalid)?;
        index(&line["source_index"])?;
        let s = line["text"].as_str().ok_or_else(invalid)?;
        work.charge(s.len())?;
        if s.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        let (digit, mixed, long, entry) = features(s);
        let (signature, has_font) = signature(&line["font_signature"])?;
        let next = fonts.len();
        let font = *fonts.entry(signature).or_insert(next);
        let h = line.get("effective_height").map_or(Ok(0.), num)?;
        let upright = line.get("angle").map_or(Ok(0.), num)? == 0.;
        let eligible = upright && line["semantic_type"].is_null() && has_font && digit && h > 0.;
        candidates += usize::from(eligible);
        records.push((
            bbox(&line["bbox"])?,
            h,
            upright,
            has_font,
            font,
            digit,
            mixed,
            long,
            entry,
            eligible,
        ));
        work.charge(chars(line)?.len())?;
        for c in chars(line)? {
            c.as_object().ok_or_else(invalid)?;
            visible(c, work)?;
            char_index(c)?;
        }
    }
    let visual = match v.get("visual_bboxes") {
        None => Vec::new(),
        Some(v) => array(v, 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?,
    };
    // 固定内核包含逐候选的visual平方和正文/目录扫描、排序；先付保守上界。
    let scans = visual
        .len()
        .checked_mul(visual.len())
        .and_then(|n| n.checked_add(lines.len() * 20 + 1))
        .and_then(|n| n.checked_mul(candidates))
        .ok_or_else(limit)?;
    work.charge(scans)?;
    Ok((source, numeric_noise_candidates(&records, &visual)))
}
fn requests(
    source: &Value,
    candidates: &Candidates,
    work: &mut Work,
) -> Result<BTreeSet<i64>, Error> {
    let lines = source["lines"].as_array().unwrap();
    let mut requested = BTreeSet::new();
    for (i, refs) in candidates {
        for i in std::iter::once(i).chain(refs) {
            for c in chars(&lines[*i])? {
                work.charge(1)?;
                if visible(c, work)? && c.get("char_idx").is_some() {
                    requested.insert(index(&c["char_idx"])?);
                }
            }
        }
    }
    if requested.len() > rapiddoc_core::resources::limit(20000) {
        return Err(limit());
    }
    Ok(requested)
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    prepare_with_work(v, &mut Work(0))
}
pub(crate) fn prepare_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let (source, candidates) = detect(v, work)?;
    let requested = requests(source, &candidates, work)?;
    Ok(
        json!({"candidates":candidates.into_iter().map(|(i,r)|json!({"line_index":i,"reference_indices":r})).collect::<Vec<_>>(),"requested_char_indices":requested}),
    )
}
type Color = [u32; 4];
fn paints(v: &Value) -> Result<BTreeMap<i64, Option<Color>>, Error> {
    let map = v.as_object().ok_or_else(invalid)?;
    if map.len() > rapiddoc_core::resources::limit(20000) {
        return Err(limit());
    }
    let mut out = BTreeMap::new();
    for (k, v) in map {
        let n = k.parse::<i64>().map_err(|_| invalid())?;
        if n.to_string() != *k {
            return Err(invalid());
        }
        let color = if v.is_null() {
            None
        } else {
            let a = array(v, 4)?;
            if a.len() != 4 {
                return Err(invalid());
            }
            let mut c = [0; 4];
            for (i, v) in a.iter().enumerate() {
                c[i] = v.as_u64().filter(|n| *n <= 255).ok_or_else(invalid)? as u32;
            }
            Some(c)
        };
        out.insert(n, color);
    }
    Ok(out)
}
fn colors(
    line: &Value,
    paint: &BTreeMap<i64, Option<Color>>,
    work: &mut Work,
) -> Result<Vec<Option<Color>>, Error> {
    let mut out = Vec::new();
    for c in chars(line)? {
        work.charge(1)?;
        if visible(c, work)? {
            out.push(char_index(c)?.and_then(|i| paint.get(&i).copied().flatten()));
        }
    }
    Ok(out)
}
pub fn apply(v: &Value) -> Result<Value, Error> {
    apply_with_work(v, &mut Work(0))
}
pub(crate) fn apply_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let (source, candidates) = detect(v, work)?;
    apply_detected(source.clone(), candidates, v, work)
}
pub(crate) fn apply_owned(mut v: Value, work: &mut Work) -> Result<Value, Error> {
    let (_, candidates) = detect(&v, work)?;
    let source = v["source"].take();
    apply_detected(source, candidates, &v, work)
}
fn apply_detected(
    mut source: Value,
    candidates: Candidates,
    v: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    let mut excluded = BTreeSet::new();
    let mut requested = BTreeSet::new();
    if let Some(paint) = v.get("paint").filter(|p| !p.is_null()) {
        let paint = paints(paint)?;
        if !candidates.is_empty() {
            requested = requests(&source, &candidates, work)?;
        }
        let lines = source["lines"].as_array().unwrap();
        for (i, refs) in candidates {
            let cs = colors(&lines[i], &paint, work)?;
            if cs.is_empty()
                || cs.iter().any(|c| {
                    c.is_none_or(|c| {
                        c[3] == 0
                            || *c[..3].iter().min().unwrap() < 170
                            || *c[..3].iter().max().unwrap() > 225
                            || c[..3].iter().max().unwrap() - c[..3].iter().min().unwrap() > 35
                    })
                })
            {
                continue;
            }
            let mut references = Vec::new();
            for j in refs {
                references.extend(
                    colors(&lines[j], &paint, work)?
                        .into_iter()
                        .flatten()
                        .filter(|c| c[3] > 0),
                );
            }
            let dark = references
                .iter()
                .filter(|c| {
                    0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64 <= 120.
                })
                .count();
            let white = references
                .iter()
                .filter(|c| *c[..3].iter().min().unwrap() >= 245)
                .count();
            if !references.is_empty()
                && (dark as f64 >= 0.8 * references.len() as f64
                    || white as f64 >= 0.8 * references.len() as f64)
            {
                excluded.insert(index(&lines[i]["source_index"])?);
            }
        }
    }
    if !excluded.is_empty() {
        let mut indices = BTreeSet::new();
        for line in source["lines"].as_array().unwrap() {
            if excluded.contains(&index(&line["source_index"])?) {
                for c in chars(line)? {
                    work.charge(1)?;
                    indices.insert(char_index(c)?);
                }
            }
        }
        let lines = source["lines"].as_array_mut().unwrap();
        lines.retain(|l| !excluded.contains(&l["source_index"].as_i64().unwrap()));
        let mut kept = Vec::new();
        for c in source["chars"].as_array().unwrap() {
            work.charge(1)?;
            if !indices.contains(&char_index(c)?) {
                kept.push(c.clone());
            }
        }
        source["chars"] = Value::Array(kept);
    }
    let result = crate::processing::with_source(
        source,
        json!({"excluded_source_indices":excluded,"requested_char_indices":requested}),
    );
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
    #[test]
    fn old_text_noise() {
        for name in ["native-text-noise", "native-text-noise-edges"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                assert_eq!(
                    prepare(&c["request"]).unwrap(),
                    c["prepared"],
                    "prepare {}",
                    c["label"]
                );
                assert_eq!(
                    apply(&c["request"]).unwrap(),
                    c["expected"],
                    "apply {}",
                    c["label"]
                );
            }
        }
    }
    #[test]
    fn shared_budget() {
        let q = json!({"source":{"lines":[{"text":"1","bbox":[0,0,1,1],"effective_height":1,"source_index":0,"font_signature":["n",0]}],"chars":[]}});
        assert!(prepare_with_work(&q, &mut Work(9_999_999)).is_err());
        assert!(apply_with_work(&q, &mut Work(9_999_999)).is_err());
        assert!(apply(&q).is_ok());
    }
}
