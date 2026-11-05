//! UTF-16 range styles and master inheritance; no lossy JSON round trips.
use super::{
    records::{self, u32, Budget, Record},
    styles::{self, CharacterRun, MasterLevel, ParagraphRun, StyleRuns},
};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
pub fn is_control(c: char) -> bool {
    static RANGES: std::sync::OnceLock<Vec<[u32; 2]>> = std::sync::OnceLock::new();
    let ranges = RANGES
        .get_or_init(|| serde_json::from_str(include_str!("control_categories.json")).unwrap());
    let code = c as u32;
    let i = ranges.partition_point(|r| r[1] < code);
    ranges.get(i).is_some_and(|r| r[0] <= code)
}
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub text: String,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strike: bool,
    pub baseline: Option<i16>,
    pub hyperlink: Option<String>,
}
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Paragraph {
    pub runs: Vec<Run>,
    pub depth: u16,
    pub list_kind: Option<String>,
    pub start: Option<i64>,
    pub pp9rt: u8,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Content {
    pub paragraphs: Vec<Paragraph>,
    pub text_type: u32,
}
pub fn decode(r: Record<'_>) -> Option<String> {
    match r.kind {
        0xfa0 => Some(records::utf16(r.payload)),
        0xfa8 => Some(
            r.payload
                .iter()
                .map(|b| cp1252(*b))
                .collect::<String>()
                .trim_end_matches('\0')
                .into(),
        ),
        _ => None,
    }
}
fn cp1252(b: u8) -> char {
    const SPECIAL: [char; 32] = [
        '€', '�', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '�', 'Ž', '�', '�', '‘',
        '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '�', 'ž', 'Ÿ',
    ];
    if (0x80..0xa0).contains(&b) {
        SPECIAL[(b - 0x80) as usize]
    } else {
        b as char
    }
}
fn at<T: Clone + Default>(runs: &[T], offset: usize, count: impl Fn(&T) -> u32) -> T {
    let mut cursor = 0usize;
    for r in runs {
        let end = cursor + count(r) as usize;
        if cursor <= offset && offset < end {
            return r.clone();
        }
        cursor = end;
    }
    T::default()
}
fn master(levels: &[MasterLevel], depth: u16) -> MasterLevel {
    if levels.is_empty() {
        MasterLevel::default()
    } else {
        levels[(depth as usize).min(levels.len() - 1)].clone()
    }
}
fn whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
fn flush(
    paragraphs: &mut Vec<Paragraph>,
    runs: &mut Vec<Run>,
    styles: &StyleRuns,
    levels: &[MasterLevel],
    start: usize,
) {
    let p: ParagraphRun = at(&styles.paragraphs, start, |r| r.count);
    let c: CharacterRun = at(&styles.characters, start, |r| r.count);
    let m = master(levels, p.depth);
    if runs
        .iter()
        .any(|r| !r.text.trim_matches(whitespace).is_empty())
    {
        paragraphs.push(Paragraph {
            runs: std::mem::take(runs),
            depth: p.depth,
            list_kind: if p.bullet.unwrap_or(m.bullet.unwrap_or(false)) {
                Some("unordered".into())
            } else {
                None
            },
            start: None,
            pp9rt: c.pp9rt,
        });
    } else {
        runs.clear();
    }
}
pub fn build(
    text: &str,
    styles: &StyleRuns,
    levels: &[MasterLevel],
    links: &[(u32, u32, String)],
    is_control: impl Fn(char) -> bool,
) -> Vec<Paragraph> {
    let mut paragraphs = vec![];
    let mut runs: Vec<Run> = vec![];
    let (mut offset, mut start) = (0, 0);
    for c in text.chars() {
        let p: ParagraphRun = at(&styles.paragraphs, offset, |r| r.count);
        let m = master(levels, p.depth);
        let explicit: CharacterRun = at(&styles.characters, offset, |r| r.count);
        let link = links
            .iter()
            .find(|(a, b, _)| *a as usize <= offset && offset < *b as usize)
            .map(|(_, _, s)| s.clone());
        let style = Run {
            text: String::new(),
            bold: explicit.bold.or(m.bold).unwrap_or(false),
            italic: explicit.italic.or(m.italic).unwrap_or(false),
            underline: explicit
                .underline
                .unwrap_or(m.underline.unwrap_or(false) || link.is_some()),
            strike: explicit.strike.unwrap_or(false),
            baseline: explicit.baseline.or(m.baseline),
            hyperlink: link,
        };
        if c == '\r' {
            flush(&mut paragraphs, &mut runs, styles, levels, start);
            start = offset + 1;
        } else if c == '\u{b}' || c == '\n' || c == '\t' || !is_control(c) {
            let character = if c == '\u{b}' || c == '\n' { '\n' } else { c };
            let same = runs.last().is_some_and(|r| {
                let mut previous = r.clone();
                previous.text.clear();
                previous == style
            });
            if same {
                runs.last_mut().unwrap().text.push(character);
            } else {
                let mut next = style;
                next.text.push(character);
                runs.push(next);
            }
        }
        offset += c.len_utf16();
    }
    if !runs.is_empty() {
        flush(&mut paragraphs, &mut runs, styles, levels, start);
    }
    paragraphs
}
pub fn targets(document: Record<'_>, budget: &mut Budget) -> Result<BTreeMap<u32, String>> {
    let containers = records::descendants(document, budget).collect::<Result<Vec<_>>>()?;
    let mut out = BTreeMap::new();
    for r in containers
        .into_iter()
        .filter(|r| r.kind == 0xfd7 && r.version == 15)
    {
        let mut id = None;
        let mut strings = vec![];
        for c in records::iter(r.payload, 0, None, false, budget) {
            let c = c?;
            if c.kind == 0xfd3 {
                id = u32(c.payload, 0);
            } else if c.kind == 0xfba {
                strings.push(records::utf16(c.payload));
            }
        }
        if let (Some(id), Some(target)) = (id, strings.last()) {
            if let Some(safe) = crate::links::sanitize(Some(target)).filter(|s| {
                s.split_once(':').is_some_and(|(scheme, _)| {
                    matches!(
                        scheme.to_ascii_lowercase().as_str(),
                        "http" | "https" | "mailto"
                    )
                })
            }) {
                out.insert(id, safe.into());
            }
        }
    }
    Ok(out)
}
pub fn interactive(
    related: &[Record<'_>],
    links: &BTreeMap<u32, String>,
    budget: &mut Budget,
) -> Result<Vec<(u32, u32, String)>> {
    let mut flat = vec![];
    for r in related {
        flat.push(*r);
        if r.version == 15 {
            flat.extend(records::descendants(*r, budget).collect::<Result<Vec<_>>>()?);
        }
    }
    let mut pending = None;
    let mut out = vec![];
    for r in flat {
        if r.kind == 0xff3 {
            pending = u32(r.payload, 4);
            continue;
        }
        if r.kind == 0xfdf && pending.is_some() {
            if let (Some(a), Some(b), Some(target)) = (
                u32(r.payload, 0),
                u32(r.payload, 4),
                pending.and_then(|id| links.get(&id)),
            ) {
                if b > a {
                    out.push((a, b, target.clone()));
                }
            }
            pending = None;
        }
    }
    Ok(out)
}
pub fn parse(
    records: &[Record<'_>],
    masters: &BTreeMap<u16, Vec<MasterLevel>>,
    links: &BTreeMap<u32, String>,
    budget: &mut Budget,
    is_control: impl Fn(char) -> bool + Copy,
) -> Result<Vec<Content>> {
    let mut result = vec![];
    let mut text_type = 4;
    for (i, r) in records.iter().enumerate() {
        if r.kind == 0xf9f {
            text_type = u32(r.payload, 0).filter(|n| *n != 0).unwrap_or(4);
            continue;
        }
        let Some(text) = decode(*r) else {
            continue;
        };
        let tail = &records[i + 1..];
        let end = tail
            .iter()
            .position(|r| matches!(r.kind, 0xfa0 | 0xfa8 | 1011))
            .unwrap_or(tail.len());
        let related = &tail[..end];
        let styles = related
            .iter()
            .find(|r| r.kind == 0xfa1)
            .map(|r| styles::parse(r.payload, text.encode_utf16().count()))
            .unwrap_or_default();
        let spans = interactive(related, links, budget)?;
        let levels = u16::try_from(text_type)
            .ok()
            .and_then(|kind| masters.get(&kind))
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        result.push(Content {
            paragraphs: build(&text, &styles, levels, &spans, is_control),
            text_type,
        });
    }
    Ok(result)
}
