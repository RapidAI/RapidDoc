//! Binary bookmarks and TOC run cleanup follow the Python DOC adapter.
use super::fib::Pair;
use crate::docx::spans::{self, Span};
use std::collections::BTreeMap;
fn region(table: &[u8], pair: Pair) -> Option<&[u8]> {
    table.get(pair.fc as usize..(pair.fc as usize).checked_add(pair.lcb as usize)?)
}
fn u16(data: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        data.get(p..p.checked_add(2)?)?.try_into().ok()?,
    ))
}
pub(super) fn plc(data: &[u8], width: usize) -> (Vec<u32>, Vec<&[u8]>) {
    if data.len() < 4 || (data.len() - 4) % (width + 4) != 0 {
        return (vec![], vec![]);
    }
    let n = (data.len() - 4) / (width + 4);
    let cps = data[..(n + 1) * 4]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let items = (0..n)
        .map(|i| &data[(n + 1) * 4 + i * width..(n + 1) * 4 + (i + 1) * width])
        .collect();
    (cps, items)
}
pub fn bookmarks(
    table: &[u8],
    names: Pair,
    starts: Pair,
    ends: Pair,
) -> BTreeMap<u32, Vec<String>> {
    let (Some(data), Some(starts), Some(ends)) = (
        region(table, names),
        region(table, starts),
        region(table, ends),
    ) else {
        return BTreeMap::new();
    };
    let extended = u16(data, 0) == Some(0xffff);
    let (Some(count), Some(extra)) = (
        u16(data, if extended { 2 } else { 0 }),
        u16(data, if extended { 4 } else { 2 }),
    ) else {
        return BTreeMap::new();
    };
    let (cps, items) = plc(starts, 4);
    let (end_cps, _) = plc(ends, 0);
    let mut out = BTreeMap::<u32, Vec<String>>::new();
    let mut cursor = if extended { 6usize } else { 4 };
    for i in 0..count as usize {
        let length = if extended {
            let n = u16(data, cursor);
            cursor += 2;
            n
        } else {
            let n = data.get(cursor).copied().map(u16::from);
            cursor += 1;
            n
        };
        let Some(length) = length else { continue };
        if length == 0xffff {
            continue;
        }
        let n = length as usize * if extended { 2 } else { 1 };
        let Some(payload) = data.get(cursor..cursor + n) else {
            break;
        };
        cursor += n + extra as usize;
        let name = if extended {
            String::from_utf16_lossy(
                &payload
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            )
        } else {
            encoding_rs::WINDOWS_1252.decode(payload).0.into_owned()
        };
        if name.is_empty() || name.starts_with("_GoBack") {
            continue;
        }
        let Some(end) = items
            .get(i)
            .and_then(|v| u16(v, 0))
            .map(usize::from)
            .filter(|n| *n < end_cps.len().saturating_sub(1))
        else {
            continue;
        };
        if let Some(start) = cps.get(i).filter(|cp| end_cps[end] >= **cp) {
            out.entry(*start).or_default().push(name);
        }
    }
    out
}
pub fn anchor(bookmarks: &BTreeMap<u32, Vec<String>>, start: u32, end: u32) -> Option<String> {
    if start >= end {
        return None;
    }
    let names: Vec<_> = bookmarks.range(start..end).flat_map(|(_, v)| v).collect();
    names
        .iter()
        .find(|s| s.starts_with("_Toc"))
        .or_else(|| names.first())
        .map(|s| (*s).clone())
}
pub fn toc_anchor(content: &[Span]) -> Option<String> {
    for s in content {
        if let Span::Link { url, content } = s {
            if let Some(anchor) = url.strip_prefix('#').filter(|s| !s.is_empty()) {
                return Some(anchor.into());
            }
            if let Some(anchor) = toc_anchor(content) {
                return Some(anchor);
            }
        }
    }
    None
}
pub fn toc_runs(content: Vec<Span>) -> Vec<Span> {
    // Match runs, including multiple runs sharing a hyperlink, rather than a whole link's text.
    fn flatten(content: Vec<Span>, url: Option<&str>, out: &mut Vec<Span>) {
        for s in content {
            match s {
                Span::Link { url, content } => flatten(content, Some(&url), out),
                s => out.push(if let Some(url) = url {
                    Span::Link {
                        url: url.into(),
                        content: vec![s],
                    }
                } else {
                    s
                }),
            }
        }
    }
    fn text(s: &mut Span) -> Option<&mut String> {
        match s {
            Span::Text { content, .. } => Some(content),
            Span::Link { content, .. } => content.last_mut().and_then(text),
            _ => None,
        }
    }
    static PAGE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static SUFFIX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let page = PAGE.get_or_init(|| regex::Regex::new(r"^[\s\t]*\d+[\s\t]*$").unwrap());
    let suffix = SUFFIX.get_or_init(|| regex::Regex::new(r"\t+\s*\d+\s*$").unwrap());
    let mut out = vec![];
    flatten(content, None, &mut out);
    while out
        .last_mut()
        .and_then(text)
        .is_some_and(|s| page.is_match(s))
    {
        out.pop();
    }
    if let Some(last) = out.last_mut().and_then(text) {
        *last = suffix.replace(last, "").into_owned();
    }
    let mut grouped = vec![];
    for span in spans::strip(out) {
        if let Span::Link { url, content } = &span {
            if let Some(Span::Link {
                url: previous,
                content: children,
            }) = grouped.last_mut()
            {
                if previous == url {
                    for child in content {
                        spans::append(children, child.clone());
                    }
                    continue;
                }
            }
        }
        grouped.push(span);
    }
    grouped
}
