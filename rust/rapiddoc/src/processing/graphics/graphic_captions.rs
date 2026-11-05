//! 旧图体上下沿图题保护；只以旧英文fig/figure数字种子启动。
use super::path_graphics::{array, bbox, integer, num, union, Box4};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_alnum, python_space},
};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use unicode_normalization::UnicodeNormalization;
pub(super) fn axis(a: &Box4, b: &Box4, i: usize) -> f64 {
    let short = (a.n[i + 2] - a.n[i]).min(b.n[i + 2] - b.n[i]);
    if short > 0. {
        (a.n[i + 2].min(b.n[i + 2]) - a.n[i].max(b.n[i])).max(0.) / short
    } else {
        0.
    }
}
fn center(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) / 2.
}
fn fold_char(c: char) -> char {
    match c {
        'İ' | 'ı' => 'i',
        'ſ' => 's',
        'K' => 'k',
        _ => c.to_ascii_lowercase(),
    }
}
fn prefix(c: &[char], i: usize, s: &str) -> bool {
    s.chars()
        .enumerate()
        .all(|(j, ch)| c.get(i + j).is_some_and(|c| fold_char(*c) == ch))
}
fn word(c: char) -> bool {
    c == '_' || python_alnum(c)
}
fn alpha(c: char) -> bool {
    fold_char(c).is_ascii_alphabetic()
}
pub(super) fn tail_reference(c: &[char]) -> bool {
    let mut i = 0;
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1;
    }
    let c = &c[i..];
    let Some(&first) = c.first() else {
        return false;
    };
    if matches!(first, '(' | '（') {
        let mut j = 1;
        while c.get(j).is_some_and(|c| !matches!(c, ')' | '）')) {
            j += 1;
        }
        if j > 1 && c.get(j).is_some() {
            j += 1;
            while c.get(j).is_some_and(|c| python_space(*c)) {
                j += 1;
            }
            if c.get(j).is_some_and(|c| "、,，和及与".contains(*c)) {
                return true;
            }
        }
    }
    if ".:：-–—()[]（）".contains(first) {
        return false;
    }
    if ",;&，；、".contains(first) {
        return true;
    }
    for s in [
        "and",
        "or",
        "show",
        "shows",
        "illustrate",
        "illustrates",
        "present",
        "presents",
        "depict",
        "depicts",
        "demonstrate",
        "demonstrates",
        "list",
        "lists",
        "summarise",
        "summarises",
        "summarize",
        "summarizes",
        "report",
        "reports",
        "compare",
        "compares",
        "provide",
        "provides",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "can",
    ] {
        if prefix(c, 0, s) && c.get(s.len()).is_none_or(|c| !word(*c)) {
            return true;
        }
    }
    for s in [
        "和", "及", "与", "为", "是", "展示", "显示", "给出", "说明", "列出", "分别", "可见",
        "表明", "描述", "呈现", "汇总", "所示", "中",
    ] {
        if c.iter().take(s.chars().count()).copied().eq(s.chars()) {
            return true;
        }
    }
    first.is_ascii_lowercase()
}
/// `_is_strong_caption_text` on the domain already admitted by `_FIGURE_CAPTION_LINE_RE`.
fn strong_figure(s: &str) -> bool {
    let normalized: String = s.nfkc().collect();
    let c: Vec<_> = normalized.trim_matches(python_space).chars().collect();
    if !prefix(&c, 0, "fig") {
        return false;
    }
    let mut i = 3;
    if prefix(&c, i, "ure") {
        i += 3;
    }
    if !c
        .get(i)
        .is_some_and(|c| *c == '.' || python_space(*c) || c.is_ascii_digit())
    {
        return false;
    }
    if c.get(i) == Some(&'.') {
        i += 1;
    }
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1;
    }
    let start = i;
    while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        i += 1;
    }
    if i == start {
        return false;
    }
    loop {
        let saved = i;
        let mut j = i;
        while c.get(j).is_some_and(|c| python_space(*c)) {
            j += 1;
        }
        if !c.get(j).is_some_and(|c| "-./–—".contains(*c)) {
            break;
        }
        j += 1;
        while c.get(j).is_some_and(|c| python_space(*c)) {
            j += 1;
        }
        if c.get(j).is_some_and(|c| alpha(*c)) {
            j += 1;
        }
        let start = j;
        while c.get(j).is_some_and(|c| python_decimal(*c).is_some()) {
            j += 1;
        }
        if j == start {
            i = saved;
            break;
        }
        i = j;
    }
    if c.get(i).is_some_and(|c| alpha(*c)) {
        i += 1;
    }
    !tail_reference(&c[i..])
}
pub(super) fn source_note(s: &str, chinese: bool) -> bool {
    let c: Vec<_> = s.trim_start_matches(python_space).chars().collect();
    for text in ["source", "sources", "note", "notes"] {
        if prefix(&c, 0, text) {
            let mut i = text.len();
            while c.get(i).is_some_and(|c| python_space(*c)) {
                i += 1;
            }
            if c.get(i)
                .is_some_and(|c| *c == ':' || (chinese && *c == '：'))
            {
                return true;
            }
        }
    }
    chinese
        && ["资料来源", "数据来源", "来源", "注"].iter().any(|text| {
            let n = text.chars().count();
            if !c.iter().take(n).copied().eq(text.chars()) {
                return false;
            }
            let mut i = n;
            while c.get(i).is_some_and(|c| python_space(*c)) {
                i += 1;
            }
            c.get(i).is_some_and(|c| matches!(c, ':' | '：'))
        })
}
pub(super) fn words(s: &str) -> usize {
    let c: Vec<_> = s.chars().collect();
    let (mut i, mut count) = (0, 0);
    while i < c.len() {
        if c[i].is_ascii_alphabetic() {
            let start = i;
            while i < c.len() && c[i].is_ascii_alphabetic() {
                i += 1;
            }
            if i - start >= 2
                && (start == 0 || !word(c[start - 1]))
                && (i == c.len() || !word(c[i]))
            {
                count += 1;
            }
        } else {
            i += 1;
        }
    }
    count
}
struct Line {
    b: Box4,
    id: i64,
    angle: f64,
    signature: Option<Value>,
    family: Option<String>,
    weight: f64,
    caption: bool,
    strong: bool,
    words: usize,
    numeric: bool,
    note: bool,
    terminal: bool,
}
pub(super) fn protect(
    lines: &mut [Value],
    candidates: &[Box4],
    em: f64,
    work: &mut Work,
) -> Result<BTreeSet<i64>, Error> {
    if lines.len() > rapiddoc_core::resources::limit(4096)
        || candidates.len() > rapiddoc_core::resources::limit(5000)
    {
        return Err(limit());
    }
    let mut parsed = Vec::new();
    for l in lines.iter() {
        let text = l["text"].as_str().ok_or_else(invalid)?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        work.charge(text.len() * 8 + 1)?;
        let signature = l.get("font_signature").filter(|v| !v.is_null()).cloned();
        let family = if let Some(s) = &signature {
            let a = array(s, 2)?;
            if a.len() != 2 {
                return Err(invalid());
            }
            let name = a[0].as_str().ok_or_else(invalid)?;
            if name.len() > rapiddoc_core::resources::limit(4096) {
                return Err(limit());
            }
            work.charge(name.len() + 1)?;
            Some((name.into(), a[1].clone()))
        } else {
            None
        };
        let weight = l
            .get("dominant_font_weight")
            .filter(|v| !v.is_null())
            .map_or(Ok(400.), num)?;
        let cap = super::concentric_graphics::caption(text);
        parsed.push(Line {
            b: bbox(&l["bbox"])?,
            id: integer(l, "source_index", 0)?,
            angle: l.get("angle").map_or(Ok(0.), num)?,
            family: super::table_text_styles::preparation::family(&family),
            signature,
            weight: if weight == 0. { 400. } else { weight },
            caption: cap,
            strong: cap && strong_figure(text),
            words: words(text),
            numeric: super::bar_groups::numeric(text),
            note: source_note(text, false),
            terminal: text
                .trim_end_matches(python_space)
                .ends_with(['.', '!', '?']),
        });
    }
    let mut ordered: Vec<_> = parsed
        .iter()
        .enumerate()
        .filter(|(_, l)| l.angle == 0.)
        .map(|(i, _)| i)
        .collect();
    work.charge(ordered.len() * 16 + 1)?;
    ordered.sort_by(|a, b| {
        parsed[*a].b.n[1]
            .partial_cmp(&parsed[*b].b.n[1])
            .unwrap()
            .then_with(|| parsed[*a].b.n[0].partial_cmp(&parsed[*b].b.n[0]).unwrap())
            .then(parsed[*a].id.cmp(&parsed[*b].id))
    });
    let mut protected = BTreeSet::new();
    for (at, &seed_id) in ordered.iter().enumerate() {
        let seed = &parsed[seed_id];
        if !seed.caption {
            continue;
        }
        let mut wrapped = vec![seed_id];
        if seed.signature.is_some() {
            for &i in &ordered[at + 1..] {
                work.charge(1)?;
                let tail = &parsed[i];
                let prior = &parsed[*wrapped.last().unwrap()];
                if tail.b.n[1] - prior.b.n[3] > 0.75 * em {
                    break;
                }
                if seed.family.is_none()
                    || tail.family != seed.family
                    || (tail.weight - seed.weight).abs() >= 100.
                    || (tail.b.n[0] - seed.b.n[0]).abs() > 0.3 * em
                {
                    continue;
                }
                if center(&tail.b) <= center(&prior.b) {
                    continue;
                }
                if tail.words < 2
                    || tail.caption
                    || tail.b.n[2] - tail.b.n[0] > 1.5 * (seed.b.n[2] - seed.b.n[0])
                {
                    break;
                }
                wrapped.push(i);
            }
        }
        let mut wb = seed.b.clone();
        for i in &wrapped[1..] {
            work.charge(1)?;
            wb = union(wb, &parsed[*i].b);
        }
        work.charge(candidates.len() * 2 + 1)?;
        if wrapped.len() >= 2
            && seed.strong
            && candidates.iter().any(|c| {
                axis(&wb, c, 0) >= 0.35
                    && 0. <= c.n[1] - wb.n[3]
                    && c.n[1] - wb.n[3] <= 4. * em
                    && seed.b.n[3] <= c.n[1]
            })
        {
            work.charge(lines.len() + wrapped.len())?;
            let mut group = -1i64;
            // Python max(default=-1) does not clamp a nonempty set of negative groups.
            let mut found = false;
            for l in lines.iter() {
                if let Some(v) = l.get("paragraph_group").filter(|v| !v.is_null()) {
                    let value = v.as_i64().ok_or_else(invalid)?;
                    if value.unsigned_abs() > rapiddoc_core::resources::limit(1_000_000_000) {
                        return Err(limit());
                    }
                    if !found || value > group {
                        group = value;
                        found = true;
                    }
                }
            }
            group = group.checked_add(1).ok_or_else(limit)?;
            for i in &wrapped {
                let l = lines[*i].as_object_mut().ok_or_else(invalid)?;
                l.insert("semantic_type".into(), json!("caption"));
                l.insert("title_suppressed".into(), json!(true));
                l.insert("paragraph_group".into(), json!(group));
                protected.insert(parsed[*i].id);
            }
        }
        let matching: Vec<_> = candidates
            .iter()
            .filter(|c| {
                axis(&seed.b, c, 0) >= 0.35
                    && ((c.n[3] - 2.5 * em <= center(&seed.b)
                        && center(&seed.b) <= c.n[3] + 2.5 * em)
                        || (-0.25 * em <= c.n[1] - seed.b.n[3] && c.n[1] - seed.b.n[3] <= 4. * em))
            })
            .collect();
        if matching.is_empty() {
            continue;
        }
        protected.insert(seed.id);
        let mut previous = seed;
        let above = matching.iter().any(|c| seed.b.n[3] <= c.n[1] + 0.25 * em);
        let top = matching.iter().map(|c| c.n[1]).reduce(f64::min).unwrap();
        for &i in &ordered[at + 1..] {
            work.charge(1)?;
            let tail = &parsed[i];
            if tail.b.n[1] - previous.b.n[3] > 0.75 * em {
                break;
            }
            if center(&tail.b) <= center(&previous.b) {
                continue;
            }
            let inset = above
                && tail.b.n[0] >= seed.b.n[0] - 0.25 * em
                && tail.b.n[2] <= seed.b.n[2] + 0.25 * em
                && !tail.numeric
                && !tail.caption
                && !tail.note
                && tail.b.n[3] <= top;
            if above && !inset {
                break;
            }
            if (!inset && (tail.b.n[0] - seed.b.n[0]).abs() > em)
                || (seed.signature.is_some()
                    && tail.signature.is_some()
                    && !eq(
                        seed.signature.as_ref().unwrap(),
                        tail.signature.as_ref().unwrap(),
                    ))
            {
                continue;
            }
            protected.insert(tail.id);
            previous = tail;
            if tail.terminal {
                break;
            }
        }
    }
    Ok(protected)
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut work = Work(0);
    let em = num(&v["median_height"])?;
    if em <= 0. {
        return Err(invalid());
    }
    let mut lines = array(&v["lines"], 4096)?.to_vec();
    let candidates = array(&v["candidate_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let ids = protect(&mut lines, &candidates, em, &mut work)?;
    let out = json!({"lines":lines,"protected_source_indices":ids});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_budget() {
        let mut lines = vec![json!({"text":"Fig 1: Title","bbox":[0,0,100,10],"source_index":0})];
        assert!(protect(&mut lines, &[], 10., &mut Work(10_000_000)).is_err());
        assert!(protect(&mut lines, &[], 10., &mut Work(0)).is_ok());
    }
    #[test]
    fn old_graphic_captions() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-graphic-captions.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                prepare(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
}
