//! 旧_collect_document_sources的局部段落及薄图片/装饰文字线阶段。
use super::native_lines;
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, python_printable, text_boundary::python_space};
use serde_json::{json, Value};
use std::collections::BTreeMap;
type Box4 = [f64; 4];
fn number(v: &Value) -> Result<f64, Error> {
    let n = v.as_f64().filter(|n| n.is_finite()).ok_or_else(invalid)?;
    if n.abs() > 1e9 {
        return Err(limit());
    }
    Ok(n)
}
fn rect(v: &Value) -> Result<Box4, Error> {
    let a = v.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
    Ok([
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ])
}
fn array(v: &Value, max: usize) -> Result<&Vec<Value>, Error> {
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
fn optional<'a>(v: &'a Value, key: &str) -> Result<&'a [Value], Error> {
    v.get(key)
        .map_or(Ok(&[][..]), |a| array(a, 5000).map(Vec::as_slice))
}
fn local(b: Box4, p: [f64; 2], a: i64) -> Box4 {
    match a {
        90 => [b[1], p[0] - b[2], b[3], p[0] - b[0]],
        180 => [p[0] - b[2], p[1] - b[3], p[0] - b[0], p[1] - b[1]],
        270 => [p[1] - b[3], b[0], p[1] - b[1], b[2]],
        _ => b,
    }
}
fn median(mut a: Vec<f64>) -> f64 {
    a.sort_by(f64::total_cmp);
    let n = a.len();
    if n % 2 == 1 {
        a[n / 2]
    } else {
        (a[n / 2 - 1] + a[n / 2]) / 2.
    }
}
fn lower(s: &str) -> bool {
    s.len() == 1 && s.as_bytes()[0].is_ascii_lowercase()
}
fn upper(s: &str) -> bool {
    s.len() == 1 && s.as_bytes()[0].is_ascii_uppercase()
}
fn marked(s: &str) -> bool {
    let s = s.trim_start_matches(python_space);
    let cs: Vec<_> = s.chars().collect();
    let mut i = 0;
    if cs.first().is_some_and(|c| c.is_ascii_lowercase()) {
        i = 1
    } else {
        while i < cs.len() && python_decimal(cs[i]).is_some() {
            i += 1
        }
    }
    i > 0 && i + 1 < cs.len() && matches!(cs[i], '.' | ')') && python_space(cs[i + 1])
}
struct Line {
    v: Value,
    b: Box4,
    h: f64,
    em: f64,
    angle: i64,
    group: Option<i64>,
}
impl Line {
    fn text(&self) -> &str {
        self.v["text"].as_str().unwrap()
    }
    fn free(&self) -> bool {
        self.v["semantic_type"].is_null()
    }
    fn assign(&mut self, group: i64) {
        self.v["semantic_type"] = json!("text");
        self.v["title_suppressed"] = json!(true);
        self.v["paragraph_group"] = json!(group);
        self.group = Some(group)
    }
}
fn read(raw: &Value) -> Result<Line, Error> {
    read_owned(raw.clone())
}
fn read_owned(raw: Value) -> Result<Line, Error> {
    let Value::Object(obj) = raw else {
        return Err(invalid());
    };
    let mut v = native_lines::defaults();
    v["text"] = obj.get("text").cloned().unwrap_or(Value::Null);
    v["text"].as_str().ok_or_else(invalid)?;
    native_lines::markers(&mut v);
    for (k, x) in obj {
        v[k] = x
    }
    let b = rect(&v["bbox"])?;
    let angle = v["angle"].as_i64().ok_or_else(invalid)?;
    if !matches!(angle, 0 | 90 | 180 | 270) {
        return Err(invalid());
    }
    v["source_index"].as_i64().ok_or_else(invalid)?;
    if v["source_bbox"].is_null() {
        v["source_bbox"] = v["bbox"].clone()
    }
    let em = number(&v["em_height"])?;
    let effective = number(&v["effective_height"])?;
    let repaired = v["style_scale_repaired"].as_bool().ok_or_else(invalid)?;
    let h = (if repaired && em > 0. {
        em
    } else if effective != 0. {
        effective
    } else {
        b[3] - b[1]
    })
    .max(0.1);
    let group = if v["paragraph_group"].is_null() {
        None
    } else {
        Some(
            v["paragraph_group"]
                .as_i64()
                .filter(|g| g.unsigned_abs() <= rapiddoc_core::resources::limit(1_000_000_000))
                .ok_or_else(invalid)?,
        )
    };
    if !v["semantic_type"].is_null() && !v["semantic_type"].is_string() {
        return Err(invalid());
    }
    let chars = array(&v["chars"], 20000)?;
    for c in chars {
        c["char"].as_str().ok_or_else(invalid)?;
        rect(&c["bbox"])?;
    }
    Ok(Line {
        v,
        b,
        h,
        em,
        angle,
        group,
    })
}
fn next_group(lines: &[Line]) -> i64 {
    lines.iter().filter_map(|l| l.group).max().unwrap_or(-1) + 1
}
fn drop_caps(lines: &mut [Line], work: &mut Work) -> Result<(), Error> {
    let mut group = next_group(lines);
    let text_work: usize = lines.iter().map(|l| l.text().len()).sum();
    for first in 0..lines.len() {
        let f = &lines[first];
        let chars = f.v["chars"].as_array().unwrap();
        work.charge(chars.len())?;
        if f.angle != 0 || chars.len() < 4 || !upper(chars[0]["char"].as_str().unwrap()) {
            continue;
        }
        let mut index = 1;
        while index < chars.len() {
            let s = chars[index]["char"].as_str().unwrap();
            if s.is_empty() || !s.chars().all(python_space) {
                break;
            }
            let b = rect(&chars[index]["bbox"])?;
            if b[2] > b[0] || b[3] > b[1] {
                break;
            }
            index += 1;
        }
        let suffix: Vec<_> = chars[index..]
            .iter()
            .take_while(|c| lower(c["char"].as_str().unwrap()))
            .collect();
        if !(2..=11).contains(&suffix.len()) {
            continue;
        }
        let cap = rect(&chars[0]["bbox"])?;
        let mut body = rect(&suffix[0]["bbox"])?;
        let mut heights = Vec::new();
        for c in suffix {
            let b = rect(&c["bbox"])?;
            heights.push(b[3] - b[1]);
            body = [
                body[0].min(b[0]),
                body[1].min(b[1]),
                body[2].max(b[2]),
                body[3].max(b[3]),
            ];
        }
        let em = median(heights);
        let ch = cap[3] - cap[1];
        let gap = body[0] - cap[2];
        if em <= 0.
            || ch < 3. * em
            || gap < -0.1 * em
            || gap > 1.5 * em
            || body[3] < cap[1] + 0.2 * ch
            || body[3] > cap[1] + 0.75 * ch
        {
            continue;
        }
        work.charge(lines.len().saturating_mul(20).saturating_add(text_work))?;
        let edge = lines
            .iter()
            .enumerate()
            .filter(|(i, l)| {
                *i != first
                    && (l.b[0] - body[0]).abs() <= 0.5 * em
                    && body[1] < l.b[1]
                    && l.b[1] < cap[3]
            })
            .map(|(_, l)| l.b[2])
            .reduce(f64::max)
            .unwrap_or(body[2]);
        let mut peers: Vec<_> = lines
            .iter()
            .enumerate()
            .filter(|(i, l)| {
                *i != first
                    && l.angle == 0
                    && l.b[0] >= cap[0] - 0.4 * em
                    && l.b[2] <= edge + 0.5 * em
                    && l.h >= 0.85 * em
                    && l.h <= 1.15 * em
                    && l.b[1] >= body[1] - 0.2 * em
            })
            .map(|(i, _)| i)
            .collect();
        peers.sort_by(|&i, &j| {
            lines[i].b[1]
                .total_cmp(&lines[j].b[1])
                .then(lines[i].b[0].total_cmp(&lines[j].b[0]))
        });
        let mut run = vec![first];
        let mut bottom = body[3];
        for i in peers {
            let l = &lines[i];
            if !l.free() || marked(l.text()) || l.b[1] - bottom > 0.85 * em {
                break;
            }
            if l.b[1] < cap[3] - em && l.b[0] < body[0] - 0.5 * em {
                continue;
            }
            run.push(i);
            bottom = bottom.max(l.b[3]);
        }
        let wrapped = run[1..]
            .iter()
            .filter(|&&i| {
                (lines[i].b[0] - body[0]).abs() <= 0.5 * em && lines[i].b[1] > body[1] + 0.5 * em
            })
            .count();
        let returned = run[1..]
            .iter()
            .filter(|&&i| {
                lines[i].b[1] >= cap[3] - em && (lines[i].b[0] - cap[0]).abs() <= 0.5 * em
            })
            .count();
        if wrapped < 2 || returned < 2 {
            continue;
        }
        if index > 1 {
            let chars = lines[first].v["chars"].as_array().unwrap();
            let prefix: String = chars[..index]
                .iter()
                .map(|c| c["char"].as_str().unwrap())
                .collect();
            if let Some(rest) = lines[first].text().strip_prefix(&prefix) {
                let text = format!("{}{}", chars[0]["char"].as_str().unwrap(), rest);
                lines[first].v["text"] = json!(text)
            }
        }
        for i in run {
            lines[i].assign(group)
        }
        group += 1;
    }
    Ok(())
}
fn hanging(lines: &mut [Line], work: &mut Work) -> Result<(), Error> {
    let mut group = next_group(lines);
    let text_work: usize = lines.iter().map(|l| l.text().len()).sum();
    for marker in 0..lines.len() {
        let m = &lines[marker];
        let text = m.text().trim_matches(python_space);
        if m.angle != 0
            || !m.free()
            || text.len() != 2
            || !text.as_bytes()[0].is_ascii_lowercase()
            || !matches!(text.as_bytes()[1], b'.' | b')')
        {
            continue;
        }
        work.charge(lines.len().saturating_mul(20).saturating_add(text_work))?;
        let em = m.h;
        let mb = m.b;
        let hosts: Vec<_> = lines
            .iter()
            .enumerate()
            .filter(|(i, l)| {
                *i != marker
                    && l.angle == 0
                    && l.free()
                    && l.group.is_none()
                    && (l.b[3] - mb[3]).abs() <= 0.25 * em
                    && l.b[0] - mb[2] >= 0.5 * em
                    && l.b[0] - mb[2] <= 3. * em
                    && l.text()
                        .split(python_space)
                        .filter(|s| !s.is_empty())
                        .count()
                        >= 5
            })
            .map(|(i, _)| i)
            .collect();
        if hosts.len() != 1 {
            continue;
        }
        let first = hosts[0];
        let fb = lines[first].b;
        let font = lines[first].v["font_signature"].clone();
        let mut candidates: Vec<_> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                l.angle == 0 && l.b[1] > fb[1] + 0.5 * em && (l.b[0] - fb[0]).abs() <= 0.2 * em
            })
            .map(|(i, _)| i)
            .collect();
        candidates.sort_by(|&i, &j| lines[i].b[1].total_cmp(&lines[j].b[1]));
        let mut run = vec![first];
        for &i in &candidates {
            let l = &lines[i];
            let gap = l.b[1] - lines[*run.last().unwrap()].b[3];
            if !l.free()
                || l.group.is_some()
                || l.h < 0.85 * em
                || l.h > 1.15 * em
                || !eq(&l.v["font_signature"], &font)
                || gap < -0.15 * em
                || gap > 0.55 * em
                || marked(l.text())
            {
                break;
            }
            run.push(i);
        }
        if run.len() < 3 {
            continue;
        }
        let bottom = lines[*run.last().unwrap()].b[3];
        let mut member = vec![false; lines.len()];
        for &i in &run {
            member[i] = true
        }
        if candidates.iter().any(|&i| {
            let l = &lines[i];
            !member[i]
                && l.h >= 0.85 * em
                && l.h <= 1.15 * em
                && !eq(&l.v["font_signature"], &font)
                && l.b[1] - bottom >= -0.15 * em
                && l.b[1] - bottom <= 0.55 * em
        }) {
            continue;
        }
        let right = run.iter().map(|&i| lines[i].b[2]).reduce(f64::max).unwrap();
        if lines.iter().enumerate().any(|(i, l)| {
            i != marker
                && !member[i]
                && l.b[1] >= fb[1]
                && l.b[1] <= bottom
                && l.b[0] > fb[0] + 0.2 * em
                && l.b[0] < right
        }) {
            continue;
        }
        lines[marker].assign(group);
        for i in run {
            lines[i].assign(group)
        }
        group += 1;
    }
    Ok(())
}
fn coerce_rule(v: &Value) -> Result<Option<Value>, Error> {
    let b = rect(&v["bbox"])?;
    let b = [
        b[0].min(b[2]),
        b[1].min(b[3]),
        b[0].max(b[2]),
        b[1].max(b[3]),
    ];
    if b[0] >= b[2] || b[1] >= b[3] {
        return Ok(None);
    }
    let width = number(&v["width"])?.max(0.);
    let orientation = v["orientation"]
        .as_str()
        .filter(|s| matches!(*s, "horizontal" | "vertical"))
        .ok_or_else(invalid)?;
    Ok(Some(
        json!({"bbox":b,"width":width,"orientation":orientation}),
    ))
}
fn private(c: char) -> bool {
    matches!(c as u32,0xe000..=0xf8ff|0xf0000..=0xffffd|0x100000..=0x10fffd)
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0))
}
pub(super) fn prepare_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let raw = array(&v["lines"], 4096)?;
    let lines = raw.iter().map(read).collect::<Result<Vec<_>, _>>()?;
    prepare_lines(lines, v, work)
}
pub(super) fn prepare_owned(mut v: Value, work: &mut Work) -> Result<Value, Error> {
    array(&v["lines"], 4096)?;
    let Value::Array(raw) = v["lines"].take() else {
        return Err(invalid());
    };
    let lines = raw
        .into_iter()
        .map(read_owned)
        .collect::<Result<Vec<_>, _>>()?;
    prepare_lines(lines, &v, work)
}
fn prepare_lines(mut lines: Vec<Line>, v: &Value, work: &mut Work) -> Result<Value, Error> {
    let a = array(&v["page_size"], 2)?;
    if a.len() != 2 {
        return Err(invalid());
    }
    let p = [number(&a[0])?, number(&a[1])?];
    if p.iter().any(|n| *n <= 0.) {
        return Err(invalid());
    }
    work.charge(
        lines
            .iter()
            .map(|l| l.text().len() + l.v["chars"].as_array().unwrap().len())
            .sum(),
    )?;
    drop_caps(&mut lines, work)?;
    hanging(&mut lines, work)?;
    let local_lines = if v["include_local_lines"] == true {
        Some(lines.iter().map(|l| l.v.clone()).collect::<Vec<_>>())
    } else {
        None
    };
    let mut rules = Vec::new();
    for r in optional(v, "drawing_lines")? {
        if let Some(r) = coerce_rule(r)? {
            rules.push(r)
        }
    }
    let mut heights: Vec<_> = lines.iter().map(|l| l.em).filter(|e| *e > 0.).collect();
    heights.sort_by(f64::total_cmp);
    if !heights.is_empty() {
        let em = heights[heights.len() / 2];
        for info in optional(v, "image_infos")? {
            work.charge(1)?;
            let b = rect(&info["bbox"])?;
            let h = b[3] - b[1];
            if b[2] - b[0] >= 2. * em && h > 0. && h <= 0.1 * em {
                if let Some(r) =
                    coerce_rule(&json!({"bbox":info["bbox"],"width":h,"orientation":"horizontal"}))?
                {
                    rules.push(r)
                }
            }
        }
    }
    let mut kept = Vec::new();
    for line in lines {
        let b = local(line.b, p, line.angle);
        let (pw, ph) = if matches!(line.angle, 90 | 270) {
            (p[1], p[0])
        } else {
            (p[0], p[1])
        };
        let mut counts = BTreeMap::new();
        let mut total = 0usize;
        let mut private_count = 0usize;
        for c in line
            .text()
            .chars()
            .filter(|c| !python_space(*c) && (python_printable(*c) || private(*c)))
        {
            total += 1;
            private_count += usize::from(private(c));
            *counts.entry(c).or_insert(0usize) += 1;
        }
        let repeated = counts.values().copied().max().unwrap_or(0);
        let w = (b[2] - b[0]).max(0.);
        let h = (b[3] - b[1]).max(0.1);
        if total >= 8
            && private_count as f64 / total as f64 >= 0.8
            && repeated as f64 / total as f64 >= 0.8
            && w >= 0.4 * pw
            && w / h >= 20.
            && (b[1] + b[3]) / 2. <= 0.2 * ph
        {
            rules.push(json!({"bbox":line.v["bbox"],"width":h.min(1.),"orientation":if matches!(line.angle,90|270){"vertical"}else{"horizontal"}}));
        } else {
            kept.push(line.v)
        }
    }
    if rules.len() > rapiddoc_core::resources::limit(10000) {
        return Err(limit());
    }
    let mut result = json!({});
    result["lines"] = Value::Array(kept);
    result["drawing_lines"] = Value::Array(rules);
    if let Some(lines) = local_lines {
        result["local_lines"] = Value::Array(lines);
    }
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
    fn dense_source_records_are_not_a_production_document_limit() {
        let records = json!(vec![Value::Null; 5001]);
        assert!(array(&records, 5000).is_err());
        {
            let _scope = crate::legacy_page::budget::host_page();
            assert_eq!(array(&records, 5000).unwrap().len(), 5001);
            assert!(rect(&json!([0., 0., 1., 1., 2.])).is_err());
            assert!(array(&Value::Null, 5000).is_err());
        }
        assert!(array(&records, 5000).is_err());
    }
    #[test]
    fn owned_source_lines_match_installed_outputs_and_work() {
        let fixture: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!("../../../../../tests/fixtures/native_source_owned_reference.json.gz")
                [..],
        ))
        .unwrap();
        for host in [false, true] {
            let _scope = host.then(crate::legacy_page::budget::host_page);
            for case in fixture["cases"].as_array().unwrap() {
                let q = &case["request"];
                let (mut a, mut b) = (Work(0), Work(0));
                let old = prepare_with_work(q, &mut a);
                let actual = prepare_owned(q.clone(), &mut b);
                assert_eq!(a.0, b.0, "{}", case["label"]);
                match (old, actual) {
                    (Ok(left), Ok(right)) => {
                        assert_eq!(left, right, "{}", case["label"]);
                        assert_eq!(right, case["expected"], "{}", case["label"]);
                    }
                    (Err(left), Err(right)) => {
                        assert_eq!(left.code, right.code);
                        assert_eq!(left.message, right.message);
                        assert_eq!(case["error"], true);
                    }
                    other => panic!("{}: {other:?}", case["label"]),
                }
            }
        }
    }
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
    fn old_source_line_preparation() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-source-lines.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let result = prepare(&c["request"]).unwrap_or_else(|e| panic!("{}: {e:?}", c["label"]));
            same(&result, &c["expected"], c["label"].as_str().unwrap());
        }
    }
    #[test]
    fn source_stage_keeps_incoming_work_budget() {
        let q = json!({"page_size":[400.,600.],"lines":[{"text":"a)","bbox":[10.,10.,18.,20.],"angle":0,"source_index":0}]});
        let mut work = Work(9_999_999);
        assert_eq!(
            prepare_with_work(&q, &mut work).unwrap_err().code,
            rapiddoc_core::ErrorCode::ResourceLimit
        );
        assert!(prepare(&q).is_ok());
    }
}
