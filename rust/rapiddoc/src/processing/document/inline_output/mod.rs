//! 旧链接/样式到InlineSpan的物化；块和证据均为明确阶段输入。
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::{python_alnum, python_space};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};
mod conflicts;
mod matching;
mod payloads;
mod scripts;
mod spacing;
mod spans;
pub(crate) fn unescape_html(text: &str) -> String {
    spans::unescape(text)
}
pub fn prepare_spacing(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_spacing_with_work(v, &mut Work(0))
}
pub(super) fn prepare_spacing_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    spacing::prepare(v, w)
}
pub(super) fn prepare_spacing_lines(
    lines: &Value,
    options: &Value,
    w: &mut Work,
) -> Result<Value, Error> {
    spacing::prepare_lines(lines, options, w)
}
pub fn apply_spacing(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    apply_spacing_with_work(v, &mut Work(0))
}
pub(super) fn apply_spacing_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    spacing::apply(v, w)
}
#[derive(Clone, Debug)]
struct Range {
    start: usize,
    end: usize,
    styles: u8,
    target: Option<String>,
    source: i64,
}
#[derive(Clone)]
struct Line {
    bbox: [f64; 4],
    text: Vec<char>,
    source: i64,
    ranges: Vec<Range>,
}
#[derive(Clone)]
struct Token {
    value: char,
    start: usize,
    end: usize,
    gap: bool,
}
fn config() -> &'static Value {
    static C: OnceLock<Value> = OnceLock::new();
    C.get_or_init(|| serde_json::from_str(include_str!("config.json")).unwrap())
}
fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}
fn text(s: &[char]) -> String {
    s.iter().collect()
}
fn blank(s: &[char]) -> bool {
    !s.is_empty() && s.iter().all(|c| python_space(*c))
}
fn number(v: &Value) -> Result<f64, Error> {
    let n = v.as_f64().filter(|n| n.is_finite()).ok_or_else(invalid)?;
    if n.abs() > 1e9 {
        return Err(limit());
    }
    Ok(n)
}
fn bbox(v: &Value) -> Result<Option<[f64; 4]>, Error> {
    let Some(a) = v.as_array().filter(|a| a.len() == 4) else {
        return Ok(None);
    };
    let b = [
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ];
    Ok((b[0] < b[2] && b[1] < b[3]).then_some(b))
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
fn listed(k: &str, s: &Value) -> bool {
    config()[k].as_array().unwrap().contains(s)
}
fn style_mask(v: &Value) -> Result<u8, Error> {
    let a = array(v, 32)?;
    let mut mask = 0;
    for s in a {
        if let Some(i) = config()["styles"]
            .as_array()
            .unwrap()
            .iter()
            .position(|v| v == s)
        {
            mask |= 1 << i
        }
    }
    Ok(mask)
}
fn style_names(mask: u8) -> Vec<Value> {
    config()["styles"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (mask & (1 << i) != 0).then_some(v.clone()))
        .collect()
}
fn style_key(mask: u8) -> Vec<&'static str> {
    config()["styles"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (mask & (1 << i) != 0).then(|| v.as_str().unwrap()))
        .collect()
}
fn lines(v: &Value, link: bool) -> Result<Vec<Line>, Error> {
    let a = if v.is_null() {
        return Ok(Vec::new());
    } else {
        array(v, 4096)?
    };
    let mut out = Vec::new();
    for l in a {
        let b = bbox(&l["bbox"])?.ok_or_else(invalid)?;
        let text = chars(l["text"].as_str().ok_or_else(invalid)?);
        let source = l["source_index"].as_i64().ok_or_else(invalid)?;
        let mut ranges = Vec::new();
        for r in array(&l[if link { "link_ranges" } else { "style_ranges" }], 5000)? {
            let start = r["start"].as_u64().ok_or_else(invalid)? as usize;
            let end = r["end"].as_u64().ok_or_else(invalid)? as usize;
            if start > text.len() || end > text.len() {
                return Err(invalid());
            }
            ranges.push(Range {
                start,
                end,
                styles: if link { 0 } else { style_mask(&r["styles"])? },
                target: if link {
                    Some(r["target"].as_str().ok_or_else(invalid)?.to_owned())
                } else {
                    None
                },
                source,
            });
        }
        out.push(Line {
            bbox: b,
            text,
            source,
            ranges,
        });
    }
    Ok(out)
}
fn assignments(
    blocks: &[Value],
    lines: &[Line],
    p: [f64; 2],
    work: &mut Work,
) -> Result<BTreeMap<usize, Vec<Line>>, Error> {
    work.charge(blocks.len().saturating_mul(lines.len()))?;
    let mut targets = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        if !listed("natural", &b["type"]) || !b["content"].is_string() {
            continue;
        }
        if let Some(mut bb) = bbox(&b["bbox"])? {
            if bb.iter().all(|n| *n >= 0. && *n <= 1.) {
                for j in 0..4 {
                    bb[j] *= p[j % 2]
                }
            }
            targets.push((i, bb))
        }
    }
    let mut out: BTreeMap<usize, Vec<Line>> = BTreeMap::new();
    for l in lines {
        let a = l.bbox;
        let x = (a[0] + a[2]) / 2.;
        let y = (a[1] + a[3]) / 2.;
        let mut best: Option<(usize, (bool, f64, f64))> = None;
        for &(i, b) in &targets {
            let inside = b[0] <= x && x <= b[2] && b[1] <= y && y <= b[3];
            let ratio = (a[2].min(b[2]) - a[0].max(b[0])).max(0.)
                * (a[3].min(b[3]) - a[1].max(b[1])).max(0.)
                / ((a[2] - a[0]) * (a[3] - a[1])).max(0.01);
            if !inside && ratio < 0.5 {
                continue;
            }
            let score = (inside, ratio, -((b[2] - b[0]) * (b[3] - b[1])));
            if best.as_ref().is_none_or(|(_, old)| score > *old) {
                best = Some((i, score))
            }
        }
        if let Some((i, _)) = best {
            out.entry(i).or_default().push(l.clone())
        }
    }
    for a in out.values_mut() {
        a.sort_by(|a, b| {
            a.source
                .cmp(&b.source)
                .then(a.bbox[1].total_cmp(&b.bbox[1]))
                .then(a.bbox[0].total_cmp(&b.bbox[0]))
        })
    }
    Ok(out)
}
fn project(content: &[char], work: &mut Work) -> Result<Vec<Token>, Error> {
    let mut out = Vec::new();
    let (mut i, mut gap) = (0, false);
    while i < content.len() {
        work.charge(1)?;
        if content[i..].starts_with(&['\\', '(']) {
            if let Some(end) = formula_end(content, i + 2, work)? {
                i = end + 2;
                gap = true;
                continue;
            }
        }
        let fragment = super::inline_evidence::compact(&content[i].to_string());
        for (j, value) in fragment.chars().enumerate() {
            out.push(Token {
                value,
                start: i,
                end: i + 1,
                gap: gap && j == 0,
            })
        }
        if !fragment.is_empty() {
            gap = false
        }
        i += 1;
    }
    Ok(out)
}
fn formula_end(content: &[char], start: usize, work: &mut Work) -> Result<Option<usize>, Error> {
    for i in start..content.len().saturating_sub(1) {
        work.charge(1)?;
        if content[i..].starts_with(&['\\', ')']) {
            return Ok(Some(i));
        }
    }
    Ok(None)
}
fn merge_styles(ranges: &[Range]) -> Vec<Range> {
    let mut events: BTreeMap<usize, [i32; 6]> = BTreeMap::new();
    for r in ranges {
        if r.start >= r.end || r.styles == 0 {
            continue;
        }
        for (p, d) in [(r.start, 1), (r.end, -1)] {
            let e = events.entry(p).or_default();
            for (i, n) in e.iter_mut().enumerate() {
                if r.styles & (1 << i) != 0 {
                    *n += d
                }
            }
        }
    }
    let mut active = [0; 6];
    let mut previous = None;
    let mut out: Vec<Range> = Vec::new();
    for (p, delta) in events {
        let mask = active
            .iter()
            .enumerate()
            .fold(0u8, |m, (i, n)| m | if *n > 0 { 1 << i } else { 0 });
        if let Some(start) = previous {
            if start < p && mask != 0 {
                if out
                    .last()
                    .is_some_and(|r| r.end == start && r.styles == mask)
                {
                    out.last_mut().unwrap().end = p
                } else {
                    out.push(Range {
                        start,
                        end: p,
                        styles: mask,
                        target: None,
                        source: 0,
                    })
                }
            }
        }
        for i in 0..6 {
            active[i] += delta[i]
        }
        previous = Some(p)
    }
    out
}
fn raw_ranges(content: &[char], tokens: &[Token], ranges: &[Range], link: bool) -> Vec<Range> {
    let mut out = Vec::new();
    for r in ranges {
        let mut current: Option<Range> = None;
        for t in tokens.iter().take(r.end).skip(r.start) {
            if current.as_ref().is_some_and(|c| {
                (link && t.gap) || (t.start > c.end && !blank(&content[c.end..t.start]))
            }) {
                out.push(current.take().unwrap())
            }
            if let Some(c) = &mut current {
                c.end = c.end.max(t.end)
            } else {
                let mut rr = r.clone();
                rr.start = t.start;
                rr.end = t.end;
                current = Some(rr)
            }
        }
        if let Some(r) = current {
            out.push(r)
        }
    }
    out
}
fn merge_raw_styles(content: &[char], mut ranges: Vec<Range>) -> Vec<Range> {
    ranges.sort_by(|a, b| {
        (a.start, a.end, style_key(a.styles)).cmp(&(b.start, b.end, style_key(b.styles)))
    });
    let mut out: Vec<Range> = Vec::new();
    for r in ranges {
        if r.start >= r.end || r.styles == 0 {
            continue;
        }
        if out.last().is_some_and(|p| {
            p.styles == r.styles && (r.start <= p.end || blank(&content[p.end..r.start]))
        }) {
            out.last_mut().unwrap().end = out.last().unwrap().end.max(r.end)
        } else {
            out.push(r)
        }
    }
    out
}
fn boundary_gap(s: &[char]) -> bool {
    if s.windows(2).any(|w| w == ['\\', '('] || w == ['\\', ')']) {
        return false;
    }
    !chars(&spans::unescape(&text(s)))
        .iter()
        .any(|c| python_alnum(*c))
}
fn merge_raw_links(content: &[char], mut ranges: Vec<Range>) -> Vec<Range> {
    ranges.sort_by(|a, b| {
        (a.start, a.end, a.source, &a.target).cmp(&(b.start, b.end, b.source, &b.target))
    });
    let mut out: Vec<Range> = Vec::new();
    for r in ranges {
        if r.start >= r.end || r.target.as_ref().is_none_or(String::is_empty) {
            continue;
        }
        if out.last().is_some_and(|p| {
            p.target == r.target
                && p.source.checked_add(1) == Some(r.source)
                && r.start >= p.end
                && boundary_gap(&content[p.end..r.start])
        }) {
            let p = out.last_mut().unwrap();
            p.end = r.end;
            p.source = r.source
        } else {
            out.push(r)
        }
    }
    out
}
pub fn apply(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    apply_with_work(v, &mut Work(0))
}
pub fn prepare_scripts(v: &Value) -> Result<Value, Error> {
    payloads::prepare(v)
}
pub(super) fn prepare_scripts_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    payloads::prepare_with_work(v, w)
}
pub(super) fn prepare_scripts_parts(
    lines: &[Value],
    page: &Value,
    tight: &Value,
    origins: &Value,
    chars: &Value,
    drawings: &Value,
    w: &mut Work,
) -> Result<Value, Error> {
    payloads::prepare_parts(lines, page, tight, origins, chars, drawings, w)
}
pub(super) fn apply_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    core(v, work, false).map(|(blocks, _)| blocks)
}
pub fn trace(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    trace_with_work(v, &mut Work(0))
}
pub(super) fn trace_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let (blocks, materialized_ranges) = core(v, w, true)?;
    let out = json!({"blocks":blocks,"materialized_ranges":materialized_ranges});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
fn core(v: &Value, work: &mut Work, diagnostics: bool) -> Result<(Value, Vec<Value>), Error> {
    let raw = array(&v["blocks"], 4000)?;
    let mut blocks = raw.clone();
    let p = array(&v["page_size"], 2)?;
    if p.len() != 2 {
        return Err(invalid());
    }
    let p = [number(&p[0])?, number(&p[1])?];
    if p.iter().any(|n| *n <= 0.) {
        return Err(invalid());
    }
    let styles = lines(&v["style_lines"], false)?;
    let links = lines(&v["link_lines"], true)?;
    let mut style_raw: BTreeMap<usize, Vec<Range>> = BTreeMap::new();
    let mut link_raw = BTreeMap::new();
    for (link, ls) in [(true, &links), (false, &styles)] {
        for (i, mut ls) in assignments(&blocks, ls, p, work)? {
            if !link {
                for l in &mut ls {
                    for r in &mut l.ranges {
                        for s in 0..6 {
                            if !config()["targets"][config()["styles"][s].as_str().unwrap()]
                                .as_array()
                                .unwrap()
                                .contains(&blocks[i]["type"])
                            {
                                r.styles &= !(1 << s)
                            }
                        }
                    }
                    l.ranges.retain(|r| r.styles != 0)
                }
            }
            let content = chars(blocks[i]["content"].as_str().unwrap());
            if content.is_empty() || (!link && !ls.iter().any(|l| !l.ranges.is_empty())) {
                continue;
            }
            let tokens = project(&content, work)?;
            let ranges = matching::match_ranges(&tokens, &ls, link, work)?;
            let raw = raw_ranges(&content, &tokens, &ranges, link);
            if !raw.is_empty() {
                if link {
                    link_raw.insert(i, merge_raw_links(&content, raw));
                } else {
                    style_raw.insert(i, merge_raw_styles(&content, raw));
                }
            }
        }
    }
    let diagnostic_ranges = if v["script_lines"].is_null() {
        Vec::new()
    } else {
        scripts::apply(
            &mut blocks,
            p,
            &v["script_lines"],
            &mut style_raw,
            diagnostics,
            work,
        )?
    };
    for (i, b) in blocks.iter_mut().enumerate() {
        spans::materialize(
            b,
            style_raw.remove(&i).unwrap_or_default(),
            link_raw.remove(&i).unwrap_or_default(),
            work,
        )?
    }
    let out = json!(blocks);
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok((out, diagnostic_ranges))
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
    fn old_complete_inline_output() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-inline-output.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let r = apply(&c["request"]).unwrap_or_else(|e| panic!("{}: {e:?}", c["label"]));
            same(&r, &c["expected"], c["label"].as_str().unwrap())
        }
    }
    #[test]
    fn output_keeps_incoming_work_budget() {
        let q = json!({"blocks":[{"type":"text","content":"ab","bbox":[0.,0.,1.,1.]}],"page_size":[400.,600.],"style_lines":[],"link_lines":[]});
        let mut work = Work(9_999_999);
        assert_eq!(
            apply_with_work(&q, &mut work).unwrap_err().code,
            rapiddoc_core::ErrorCode::ResourceLimit
        );
        assert!(apply(&q).is_ok());
    }
    #[test]
    fn old_complete_script_output_and_diagnostics() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-script-output.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let result = trace(&c["request"]).unwrap_or_else(|e| panic!("{}: {e:?}", c["label"]));
            same(&result, &c["expected"], c["label"].as_str().unwrap());
        }
    }
    #[test]
    fn old_script_payloads() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-script-payloads.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let result =
                prepare_scripts(&c["request"]).unwrap_or_else(|e| panic!("{}: {e:?}", c["label"]));
            same(&result, &c["expected"], c["label"].as_str().unwrap());
        }
    }
}
