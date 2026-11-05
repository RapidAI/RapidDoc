use super::*;
use crate::legacy_model::{table_native_multiline::spatial, table_text_styles::preparation::fold};
use std::sync::OnceLock;
fn text(v: &Value) -> Result<&str, Error> {
    v.as_str().ok_or_else(invalid)
}
fn font(name: &str) -> bool {
    let name = fold(&name.replace(['-', '_'], ""));
    [
        "mono",
        "courier",
        "consolas",
        "menlo",
        "typewriter",
        "fixed",
        "code",
    ]
    .iter()
    .any(|h| name.contains(h))
}
fn fallback_font(l: &Line) -> Result<bool, Error> {
    let sig = &l.v["font_signature"];
    if sig.is_null() {
        Ok(false)
    } else {
        let a = array(sig, 2)?;
        if a.len() != 2 {
            return Err(invalid());
        }
        Ok(font(text(&a[0])?))
    }
}
fn visible(ch: &Value) -> bool {
    ch.is_object()
        && ch
            .get("char")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.trim_matches(python_space).is_empty())
}
pub(super) fn estimated(l: &Line, w: &mut Work) -> Result<f64, Error> {
    let chars = optional(l.v, "chars", 20000)?;
    w.charge(chars.len() + 1)?;
    let n = chars.iter().filter(|c| visible(c)).count();
    if n > 0 {
        return Ok(n as f64);
    }
    let width = (l.b.n[2] - l.b.n[0]).max(0.1);
    let glyph =
        l.v.get("median_glyph_width")
            .filter(|v| !v.is_null())
            .map_or(Ok(width), num)?;
    let glyph = if glyph == 0. { width } else { glyph };
    Ok((width / glyph.max(0.1)).max(1.))
}
fn wide(c: char) -> bool {
    static R: OnceLock<Vec<[u32; 2]>> = OnceLock::new();
    let r = R.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("wide.json")).unwrap();
        serde_json::from_value(v["ranges"].clone()).unwrap()
    });
    let n = c as u32;
    let i = r.partition_point(|r| r[0] <= n);
    i > 0 && n <= r[i - 1][1]
}
pub(super) fn monospace(lines: &[&Line], w: &mut Work) -> Result<(f64, [Vec<f64>; 2]), Error> {
    let (mut supported, mut total) = (0., 0.);
    let mut widths = [Vec::new(), Vec::new()];
    for l in lines {
        w.charge(
            l.v["font_signature"]
                .get(0)
                .and_then(Value::as_str)
                .map_or(0, str::len)
                + 1,
        )?;
        let fallback = fallback_font(l)?;
        for ch in optional(l.v, "chars", 20000)? {
            w.charge(1)?;
            if !ch.is_object() {
                continue;
            }
            let value = ch
                .get("char")
                .filter(|v| !v.is_null())
                .map_or(Ok(""), text)?;
            if value.trim_matches(python_space).is_empty() {
                continue;
            }
            total += 1.;
            let name = ch
                .get("font")
                .and_then(|v| v.get("name"))
                .filter(|v| !v.is_null())
                .map_or(Ok(""), text)?;
            w.charge(name.len() + 1)?;
            if fallback || font(name) {
                supported += 1.;
            }
            let raw = &ch["bbox"];
            let Some(a) = raw.as_array().filter(|a| a.len() == 4) else {
                continue;
            };
            let mut numbers = Vec::new();
            for v in a {
                if let Ok(Some(n)) = super::super::table_native_text::number(v) {
                    numbers.push(n);
                }
            }
            if numbers.len() != 4 {
                continue;
            }
            let width = numbers[2] - numbers[0];
            if width > 0.1 {
                widths[usize::from(wide(value.chars().next().unwrap()))].push(width);
            }
        }
    }
    if total > 0. {
        return Ok((supported / total, widths));
    }
    let support = lines
        .iter()
        .map(|l| estimated(l, w))
        .collect::<Result<Vec<_>, _>>()?;
    let total = super::super::table_native_candidate::sum(&support);
    if total <= 0. {
        return Ok((0., widths));
    }
    let mut count = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if fallback_font(l)? {
            count.push(support[i]);
        }
        if let Some(v) = l.v.get("median_glyph_width").filter(|v| !v.is_null()) {
            let v = num(v)?;
            if v > 0. {
                widths[0].push(v);
            }
        }
    }
    Ok((
        super::super::table_native_candidate::sum(&count) / total,
        widths,
    ))
}
pub(super) fn stable(widths: &[Vec<f64>; 2]) -> bool {
    if widths.iter().map(Vec::len).sum::<usize>() < 3 {
        return false;
    }
    let mut ms = [None, None];
    for (i, vs) in widths.iter().enumerate() {
        if vs.is_empty() {
            continue;
        }
        let m = median(vs.clone());
        let mad = median(vs.iter().map(|v| (v - m).abs()).collect());
        if mad / m.max(0.1) > 0.2 {
            return false;
        }
        ms[i] = Some(m);
    }
    if let (Some(n), Some(w)) = (ms[0], ms[1]) {
        (1.25..=2.25).contains(&(w / n))
    } else {
        true
    }
}
fn lower(c: char) -> char {
    match c {
        'İ' | 'ı' => 'i',
        'ſ' => 's',
        'K' => 'k',
        _ => c.to_ascii_lowercase(),
    }
}
fn word(c: char) -> bool {
    c == '_' || python_alnum(c)
}
fn prefix(c: &[char], s: &str) -> bool {
    let mut n = 0;
    for ch in s.chars() {
        if c.get(n).map(|c| lower(*c)) != Some(ch) {
            return false;
        }
        n += 1;
    }
    true
}
fn decimals(c: &[char], i: &mut usize) -> bool {
    let start = *i;
    while c.get(*i).is_some_and(|c| python_decimal(*c).is_some()) {
        *i += 1;
    }
    *i > start
}
fn spaces(c: &[char], i: &mut usize) {
    while c.get(*i).is_some_and(|c| python_space(*c)) {
        *i += 1;
    }
}
pub(super) fn heading(s: &str) -> bool {
    let c: Vec<_> = s.trim_start_matches(python_space).chars().collect();
    let mut i = if prefix(&c, "algorithm") {
        9
    } else if c.starts_with(&['算', '法']) {
        2
    } else {
        return false;
    };
    spaces(&c, &mut i);
    decimals(&c, &mut i)
        && c.get(i)
            .is_some_and(|c| ".:".contains(*c) || python_space(*c))
}
pub(super) fn control(s: &str) -> bool {
    let c: Vec<_> = s.trim_start_matches(python_space).chars().collect();
    let mut i = 0;
    let mut n = 0;
    if decimals(&c, &mut n) {
        spaces(&c, &mut n);
        if c.get(n).is_some_and(|c| ".:".contains(*c)) {
            n += 1;
            spaces(&c, &mut n);
            i = n;
        }
    }
    [
        "if", "else", "end", "for", "while", "input", "output", "return", "repeat", "until",
        "require", "ensure",
    ]
    .iter()
    .any(|key| prefix(&c[i..], key) && c.get(i + key.len()).is_none_or(|c| !word(*c)))
}
pub(super) fn section(s: &str) -> bool {
    let c: Vec<_> = s.trim_start_matches(python_space).chars().collect();
    let mut i = 0;
    if !decimals(&c, &mut i) {
        return false;
    }
    while c.get(i) == Some(&'.') && c.get(i + 1).is_some_and(|c| python_decimal(*c).is_some()) {
        i += 1;
        decimals(&c, &mut i);
    }
    if c.get(i) == Some(&'.') {
        i += 1;
    }
    let start = i;
    spaces(&c, &mut i);
    i > start && c.get(i).is_some_and(|c| c.is_ascii_uppercase())
}
pub(super) fn syntax(s: &str) -> bool {
    if s.contains(":=") || s.contains("->") || s.contains(['{', '}']) {
        return true;
    }
    let c: Vec<_> = s.chars().collect();
    let mut open: Option<usize> = None;
    for i in 0..c.len() {
        if i == 0 || !word(c[i - 1]) {
            for key in [
                "if", "else", "while", "for", "return", "def", "function", "begin", "end",
            ] {
                if prefix(&c[i..], key) && c.get(i + key.len()).is_none_or(|c| !word(*c)) {
                    return true;
                }
            }
        }
        if c[i] == '。' {
            open = None;
        }
        if c[i] == '(' {
            let mut j = i;
            while j > 0 && python_space(c[j - 1]) {
                j -= 1;
            }
            if j > 0 && word(c[j - 1]) && open.is_none() {
                open = Some(i);
            }
        }
        if c[i] == ')' && open.is_some_and(|at| at + 2 <= i) {
            let mut next = i + 1;
            spaces(&c, &mut next);
            if c.get(next).is_some_and(|c| ";:".contains(*c)) {
                return true;
            }
        }
    }
    false
}
pub(in crate::processing) fn sanitize(s: &str) -> String {
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
    let mut c = Vec::new();
    for ch in s.chars() {
        if ['\u{200b}', '\u{2060}', '\u{feff}'].contains(&ch) {
            continue;
        }
        c.push(match ch {
            '\u{85}' | '\u{2028}' | '\u{2029}' => '\n',
            '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}' => ' ',
            _ => ch,
        });
    }
    let mut out = String::new();
    for (i, &ch) in c.iter().enumerate() {
        if matches!(ch, '\u{2}' | '\u{ad}') && i > 0 && c[i - 1].is_ascii_alphabetic() {
            let mut j = i + 1;
            while c.get(j).is_some_and(|c| matches!(c, '\t' | ' ')) {
                j += 1;
            }
            if c.get(j).is_none_or(|c| *c == '\n') {
                out.push('-');
                continue;
            }
        }
        if ch == '\u{ad}' {
            continue;
        }
        let ch = if ch == '\t' { ' ' } else { ch };
        if matches!(ch,'\u{0}'..='\u{8}'|'\u{b}'|'\u{c}'|'\u{e}'..='\u{1f}'|'\u{7f}'..='\u{9f}') {
            continue;
        }
        out.push(ch);
    }
    out.trim_matches(python_space).to_owned()
}
fn identity(v: &Value) -> Option<i64> {
    if let Some(n) = v.as_i64() {
        Some(n)
    } else if let Some(b) = v.as_bool() {
        Some(i64::from(b))
    } else if let Some(n) = v.as_f64() {
        if n.is_finite() && n.abs() <= 1e15 {
            Some(n.trunc() as i64)
        } else {
            None
        }
    } else {
        v.as_str()?.trim_matches(python_space).parse().ok()
    }
}
pub(super) fn materialize(lines: &[&Line], c: &Candidate, w: &mut Work) -> Result<String, Error> {
    let mut chars = Vec::new();
    let mut seen = BTreeSet::new();
    for l in lines {
        for (i, ch) in optional(l.v, "chars", 20000)?.iter().enumerate() {
            w.charge(1)?;
            if !ch.is_object() {
                continue;
            }
            let key = if let Some(n) = identity(&ch["char_idx"]) {
                (0, i128::from(n))
            } else {
                (1, i as i128 + i128::from(l.id) * 1_000_000)
            };
            if seen.insert(key) {
                chars.push(ch);
            }
        }
    }
    let content = spatial::project_options(&chars, &json!(c.b.raw), c.angle, false, true, w)?;
    if !content.is_empty() {
        return Ok(content);
    }
    let mut ordered = lines.to_vec();
    ordered.sort_by(|a, b| {
        let y = |l: &Line| format!("{:.1}", cy(&l.b)).parse::<f64>().unwrap();
        y(a).partial_cmp(&y(b))
            .unwrap()
            .then(cx(&a.b).partial_cmp(&cx(&b.b)).unwrap())
            .then(a.id.cmp(&b.id))
    });
    let raw = ordered
        .iter()
        .filter_map(|l| l.v["text"].as_str().filter(|s| !s.is_empty()))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(sanitize(&raw))
}
