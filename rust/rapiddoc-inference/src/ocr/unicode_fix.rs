//! ftfy 6.3.1 default fix_text, valid Unicode scalar input. Apache-2.0: licenses/FTFY.txt.
//! Tables/regexes are exported from the actual legacy dependency, not inferred from examples.
use crate::error;
use fancy_regex::Regex;
use rapiddoc_core::{Error, ErrorCode};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    sync::LazyLock,
};
use unicode_normalization::UnicodeNormalization;

const TEXT: usize = 65_536;
const SCRATCH: usize = TEXT * 4;
const STEPS: usize = 128;
fn limit(message: impl std::fmt::Display) -> Error {
    error(ErrorCode::ResourceLimit, message)
}
#[derive(Default)]
pub(crate) struct Work {
    units: usize,
}
impl Work {
    fn charge(&mut self, n: usize) -> Result<(), Error> {
        self.units = self
            .units
            .checked_add(n)
            .filter(|n| *n <= rapiddoc_core::resources::limit(20_000_000))
            .ok_or_else(|| limit("ftfy修复工作量超过预算"))?;
        Ok(())
    }
}
#[derive(Deserialize)]
struct Codec {
    name: String,
    decode: Vec<String>,
    encode: HashMap<u32, u8>,
}
#[derive(Deserialize)]
struct Raw {
    codecs: Vec<Codec>,
    ligatures: HashMap<u32, String>,
    width: HashMap<u32, String>,
    controls: HashSet<u32>,
    entities: HashMap<String, String>,
    invalid_refs: HashMap<u32, String>,
    invalid_points: HashSet<u32>,
    patterns: HashMap<String, String>,
}
struct Rules {
    data: Raw,
    badness: Regex,
    inconsistent: Regex,
    altered: Regex,
    lossy: Regex,
    a_grave: Regex,
    html: Regex,
    ansi: Regex,
}
static RULES: LazyLock<Rules> = LazyLock::new(|| {
    let data: Raw =
        serde_json::from_str(include_str!("unicode_fix_rules.json")).expect("frozen ftfy tables");
    let re = |key: &str| Regex::new(&data.patterns[key]).expect("frozen ftfy regex");
    Rules {
        badness: re("badness"),
        inconsistent: re("inconsistent"),
        altered: re("altered"),
        lossy: re("lossy"),
        a_grave: re("a_grave"),
        html: re("html"),
        ansi: re("ansi"),
        data,
    }
});
fn matches(re: &Regex, s: &str, work: &mut Work) -> Result<bool, Error> {
    work.charge(s.len())?;
    re.is_match(s).map_err(limit)
}
fn replace<F>(re: &Regex, s: &str, work: &mut Work, mut f: F) -> Result<String, Error>
where
    F: FnMut(&str, &mut Work) -> Result<String, Error>,
{
    work.charge(s.len())?;
    let mut out = String::with_capacity(s.len());
    let mut end = 0;
    for found in re.find_iter(s) {
        let found = found.map_err(limit)?;
        out.push_str(&s[end..found.start()]);
        out.push_str(&f(found.as_str(), work)?);
        end = found.end();
        if out.len() > rapiddoc_core::resources::limit(SCRATCH) {
            return Err(limit("ftfy中间文本超过预算"));
        }
    }
    out.push_str(&s[end..]);
    if out.len() > rapiddoc_core::resources::limit(SCRATCH) {
        return Err(limit("ftfy中间文本超过预算"));
    }
    Ok(out)
}
fn translate(s: &str, map: &HashMap<u32, String>, work: &mut Work) -> Result<String, Error> {
    work.charge(s.len())?;
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if let Some(v) = map.get(&(c as u32)) {
            out.push_str(v);
        } else {
            out.push(c);
        }
    }
    if out.len() > rapiddoc_core::resources::limit(SCRATCH) {
        return Err(limit("ftfy字符映射超过预算"));
    }
    Ok(out)
}
fn c1(s: &str, work: &mut Work) -> Result<String, Error> {
    work.charge(s.len())?;
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if ('\u{80}'..='\u{9f}').contains(&c) {
            out.push_str(&RULES.data.codecs[1].decode[c as usize]);
        } else {
            out.push(c);
        }
    }
    Ok(out)
}
fn numeric_entity(m: &str) -> String {
    if let Some(value) = RULES.data.entities.get(m) {
        return value.clone();
    }
    let Some(body) = m.strip_prefix("&#").and_then(|s| s.strip_suffix(';')) else {
        return m.into();
    };
    let (digits, radix) = body
        .strip_prefix(['x', 'X'])
        .map_or((body, 10), |s| (s, 16));
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii() && c.is_digit(radix)) {
        return m.into();
    }
    let number = u32::from_str_radix(digits, radix).unwrap_or(u32::MAX);
    let value = if let Some(value) = RULES.data.invalid_refs.get(&number) {
        value.clone()
    } else if number > 0x10ffff || (0xd800..=0xdfff).contains(&number) {
        "\u{fffd}".into()
    } else if RULES.data.invalid_points.contains(&number) {
        String::new()
    } else {
        char::from_u32(number).unwrap().to_string()
    };
    // 原_unescape_fixup连合法数字分号实体也保留，不能改成普通HTML解码器。
    if value.contains(';') {
        m.into()
    } else {
        value
    }
}
fn utf8(bytes: &[u8]) -> Option<String> {
    if !bytes.contains(&0xed) && !bytes.contains(&0xc0) {
        return std::str::from_utf8(bytes).ok().map(str::to_owned);
    }
    let mut result = String::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(&[0xc0, 0x80]) {
            result.push('\0');
            i += 2;
            continue;
        }
        if i + 6 <= bytes.len()
            && bytes[i] == 0xed
            && (0xa0..=0xaf).contains(&bytes[i + 1])
            && (0x80..=0xbf).contains(&bytes[i + 2])
            && bytes[i + 3] == 0xed
            && (0xb0..=0xbf).contains(&bytes[i + 4])
            && (0x80..=0xbf).contains(&bytes[i + 5])
        {
            let point = (((bytes[i + 1] & 15) as u32) << 16)
                + (((bytes[i + 2] & 63) as u32) << 10)
                + (((bytes[i + 4] & 15) as u32) << 6)
                + (bytes[i + 5] & 63) as u32
                + 0x10000;
            result.push(char::from_u32(point)?);
            i += 6;
            continue;
        }
        let n = match bytes[i] {
            0..=0x7f => 1,
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => return None,
        };
        result.push_str(std::str::from_utf8(bytes.get(i..i + n)?).ok()?);
        i += n;
    }
    Some(result)
}
fn fix_encoding(s: &str, depth: usize, work: &mut Work) -> Result<String, Error> {
    if depth > rapiddoc_core::resources::limit(64) {
        return Err(limit("ftfy编码修复递归超过预算"));
    }
    let mut current = s.to_owned();
    for _ in 0..rapiddoc_core::resources::limit(STEPS) {
        let next = encoding_step(&current, depth, work)?;
        if next == current {
            return Ok(next);
        }
        current = next;
    }
    Err(limit("ftfy编码修复迭代超过预算"))
}
fn encoding_step(s: &str, depth: usize, work: &mut Work) -> Result<String, Error> {
    work.charge(s.len())?;
    if s.is_ascii() || !matches(&RULES.badness, s, work)? {
        return Ok(s.to_owned());
    }
    for codec in &RULES.data.codecs {
        work.charge(s.len())?;
        let Some(bytes) = s
            .chars()
            .map(|c| codec.encode.get(&(c as u32)).copied())
            .collect::<Option<Vec<u8>>>()
        else {
            continue;
        };
        let mut latin: String = bytes.iter().map(|b| char::from(*b)).collect();
        if codec.name != "macroman" && matches(&RULES.altered, &latin, work)? {
            latin = replace(&RULES.a_grave, &latin, work, |_, _| {
                Ok("\u{c3}\u{a0} ".into())
            })?;
            latin = replace(&RULES.altered, &latin, work, |m, _| {
                Ok(m.replace(' ', "\u{a0}"))
            })?;
        }
        if codec.name.starts_with("sloppy") {
            latin = replace(&RULES.lossy, &latin, work, |_, _| {
                Ok("\u{ef}\u{bf}\u{bd}".into())
            })?;
        }
        let bytes: Vec<_> = latin.chars().map(|c| c as u8).collect();
        if let Some(fixed) = utf8(&bytes) {
            return Ok(fixed);
        }
    }
    if matches(&RULES.inconsistent, s, work)? {
        let fixed = replace(&RULES.inconsistent, s, work, |m, work| {
            if m.len() < s.len() && matches(&RULES.badness, m, work)? {
                fix_encoding(m, depth + 1, work)
            } else {
                Ok(m.into())
            }
        })?;
        if fixed != s {
            return Ok(fixed);
        }
    }
    // Latin1→严格Windows1252的成功结果与末尾sloppy C1修复相同；无需解释步骤。
    c1(s, work)
}
fn segment(s: &str, html: bool, work: &mut Work) -> Result<String, Error> {
    let mut text = s.to_owned();
    for _ in 0..rapiddoc_core::resources::limit(STEPS) {
        let previous = text.clone();
        if html {
            text = replace(&RULES.html, &text, work, |m, _| Ok(numeric_entity(m)))?;
        }
        text = fix_encoding(&text, 0, work)?;
        text = c1(&text, work)?;
        text = translate(&text, &RULES.data.ligatures, work)?;
        text = translate(&text, &RULES.data.width, work)?;
        work.charge(text.len())?;
        text = text
            .chars()
            .map(|c| match c {
                '\u{2bc}' | '\u{2018}'..='\u{201b}' => '\'',
                '\u{201c}'..='\u{201f}' => '"',
                _ => c,
            })
            .collect();
        text = text
            .replace("\r\n", "\n")
            .replace(['\r', '\u{2028}', '\u{2029}', '\u{85}'], "\n");
        // Rust字符串和本模块严格UTF8/CESU8解码不会含孤立UTF16 surrogate。
        text = replace(&RULES.ansi, &text, work, |_, _| Ok(String::new()))?;
        work.charge(text.len())?;
        text.retain(|c| !RULES.data.controls.contains(&(c as u32)));
        work.charge(text.len())?;
        text = text.nfc().collect();
        if text.len() > rapiddoc_core::resources::limit(SCRATCH) {
            return Err(limit("ftfy归一化超过中间文本预算"));
        }
        if text == previous {
            return Ok(text);
        }
    }
    Err(limit("ftfy文本修复迭代超过预算"))
}
pub(crate) fn with_work(s: &str, work: &mut Work) -> Result<String, Error> {
    if s.len() > rapiddoc_core::resources::limit(TEXT) {
        return Err(limit("ftfy输入超过64KB"));
    }
    let mut result = String::with_capacity(s.len());
    let mut html = true;
    for line in s.split_inclusive('\n') {
        if line.contains('<') {
            html = false;
        }
        result.push_str(&segment(line, html, work)?);
        if result.len() > rapiddoc_core::resources::limit(TEXT) {
            return Err(limit("ftfy输出超过64KB"));
        }
    }
    Ok(result)
}
/// Old ftfy.fix_text defaults, Unicode16 NFC. Errors never return partially repaired text.
pub fn fix_text(s: &str) -> Result<String, Error> {
    with_work(s, &mut Work::default())
}
