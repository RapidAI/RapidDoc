//! 旧normalize_table_cell_text及HTML直接文本节点归一化；复用现有HTML树，不调用Python。
use crate::legacy_page::{invalid, limit, Work};
use crate::legacy_table_merge::html::Dom;
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};
#[derive(Deserialize)]
struct Rules {
    named: BTreeMap<String, String>,
    attr_named: BTreeMap<String, String>,
    c1: BTreeMap<u32, String>,
    invalid_attrs: BTreeSet<u32>,
    attr_replacements: BTreeMap<u32, String>,
    void: BTreeSet<String>,
    list_attrs: BTreeMap<String, BTreeSet<String>>,
}
fn rules() -> &'static Rules {
    static R: OnceLock<Rules> = OnceLock::new();
    R.get_or_init(|| serde_json::from_str(include_str!("table_html_rules.json")).unwrap())
}
pub(crate) fn is_void(tag: &str) -> bool {
    rules().void.contains(tag)
}
pub(crate) fn list_attribute(tag: &str, name: &str) -> bool {
    rules()
        .list_attrs
        .get("*")
        .is_some_and(|s| s.contains(name))
        || rules()
            .list_attrs
            .get(tag)
            .is_some_and(|s| s.contains(name))
}
fn cjk(c: char) -> bool {
    ('\u{3400}'..='\u{9fff}').contains(&c)
}
fn latin(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '$'
}
fn punct(c: char) -> bool {
    "，。、“”‘’；：？！、：（）《》【】".contains(c)
}
pub fn text(s: &str) -> Result<String, Error> {
    if s.len() > rapiddoc_core::resources::limit(2_000_000) {
        return Err(limit());
    }
    if !s.chars().any(cjk) {
        return Ok(s.into());
    }
    let chars = s.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if !python_space(chars[i]) {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let begin = i;
        while i < chars.len() && python_space(chars[i]) {
            i += 1;
        }
        let drop = begin > 0 && i < chars.len() && {
            let (a, b) = (chars[begin - 1], chars[i]);
            cjk(a) && cjk(b)
                || (cjk(a) || latin(a)) && punct(b)
                || punct(a) && (cjk(b) || latin(b))
                || latin(a) && cjk(b)
                || cjk(a) && latin(b)
        };
        if !drop {
            out.extend(chars[begin..i].iter());
        }
    }
    Ok(out)
}
fn numeric(digits: &str, radix: u32, attr: bool) -> String {
    let n = u32::from_str_radix(digits, radix).unwrap_or(u32::MAX);
    if attr {
        if let Some(s) = rules().attr_replacements.get(&n) {
            return s.clone();
        }
        if rules().invalid_attrs.contains(&n) {
            return String::new();
        }
    } else if let Some(s) = rules().c1.get(&n) {
        return s.clone();
    }
    if n == 0 || n > 0x10ffff || (0xd800..=0xdfff).contains(&n) {
        "\u{fffd}".into()
    } else {
        char::from_u32(n).unwrap().into()
    }
}
pub(crate) fn decode(s: &str, attr: bool) -> Result<String, Error> {
    if s.len() > rapiddoc_core::resources::limit(2_000_000) {
        return Err(limit());
    }
    let mut out = String::new();
    let mut i = 0;
    while i < s.len() {
        if s.as_bytes()[i] != b'&' {
            let c = s[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
            continue;
        }
        let begin = i;
        i += 1;
        if s.as_bytes().get(i) == Some(&b'#') {
            let mut j = i + 1;
            let radix = if matches!(s.as_bytes().get(j), Some(b'x' | b'X')) {
                j += 1;
                16
            } else {
                10
            };
            let digits = j;
            while j < s.len()
                && s.as_bytes()[j].is_ascii()
                && (s.as_bytes()[j] as char).is_digit(radix)
            {
                j += 1;
            }
            if j > digits {
                if !attr && radix == 10 && s.as_bytes().get(j).is_some_and(u8::is_ascii_hexdigit) {
                    return Err(invalid());
                }
                out.push_str(&numeric(&s[digits..j], radix, attr));
                if s.as_bytes().get(j) == Some(&b';') {
                    j += 1;
                }
                i = j;
                continue;
            }
        } else if attr {
            let mut j = i;
            let mut count = 0;
            while j < s.len() && count < 64 {
                let c = s[j..].chars().next().unwrap();
                if !c.is_ascii_alphanumeric() {
                    break;
                }
                j += c.len_utf8();
                count += 1;
            }
            if j > i {
                if s.as_bytes().get(j) == Some(&b';') {
                    j += 1;
                }
                let candidate = &s[i..j];
                let selected = if s.as_bytes().get(j) == Some(&b'=') {
                    None
                } else {
                    rules().attr_named.get(candidate)
                };
                if let Some(value) = selected {
                    out.push_str(value);
                } else {
                    out.push('&');
                    out.push_str(candidate);
                }
                i = j;
                continue;
            }
        } else {
            let mut j = i;
            if s.as_bytes().get(j).is_some_and(u8::is_ascii_alphabetic) {
                while s
                    .as_bytes()
                    .get(j)
                    .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'.'))
                {
                    j += 1;
                }
                let name = &s[i..j];
                if let Some(value) = rules().named.get(name) {
                    out.push_str(value);
                } else {
                    out.push('&');
                    out.push_str(name);
                }
                if s.as_bytes().get(j) == Some(&b';') {
                    j += 1;
                }
                i = j;
                continue;
            }
        }
        out.push('&');
        i = begin + 1;
    }
    if out.len() > rapiddoc_core::resources::limit(2_000_000) {
        return Err(limit());
    }
    Ok(out)
}
pub fn html(input: &str) -> Result<String, Error> {
    if input.len() > rapiddoc_core::resources::limit(2_000_000) {
        return Err(limit());
    }
    if let Ok(decoded) = decode(input, false) {
        if text(&decoded)? == decoded {
            return Ok(input.into());
        }
    }
    let mut work = Work(0);
    let mut dom = Dom::parse_normalization(input, &mut work)?;
    let mut changed = false;
    for i in 1..dom.nodes.len() {
        let parent = dom.nodes[i].parent.ok_or_else(invalid)?;
        if !matches!(dom.nodes[parent].tag.as_str(), "td" | "th")
            || !(dom.nodes[i].tag.is_empty() || dom.nodes[i].tag.starts_with('!'))
        {
            continue;
        }
        let normalized = text(&dom.nodes[i].text)?;
        if normalized != dom.nodes[i].text {
            changed = true;
            dom.nodes[i].text = normalized;
            dom.nodes[i].tag.clear();
        }
    }
    if !changed {
        return Ok(input.into());
    }
    let result = dom.normalized_html();
    if result.len() > rapiddoc_core::resources::limit(4_000_000) {
        return Err(limit());
    }
    Ok(result)
}
