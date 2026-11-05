//! Auxiliary story parsing shares Word field/style semantics with the main story.
use super::{
    formatting::{sprms, Formatting},
    pieces::{Piece, TextStream},
    styles::{Character, Stylesheet},
};
use crate::docx::spans::{self, Span};
use std::collections::BTreeSet;
pub fn character(
    text: &TextStream,
    index: usize,
    pieces: &[Piece],
    formatting: &Formatting,
    styles: &Stylesheet,
) -> Character {
    let fc = text.fcs[index];
    let paragraph = Formatting::at(&formatting.paragraphs, fc)
        .map(|r| r.style_id)
        .unwrap_or(0);
    let chpx = Formatting::at(&formatting.characters, fc)
        .map(|r| r.grpprl.as_slice())
        .unwrap_or(&[]);
    let id = sprms(chpx)
        .into_iter()
        .filter(|(op, _)| *op == 0x4a30)
        .map(|(_, v)| u16::from_le_bytes([v[0], v[1]]))
        .last()
        .unwrap_or(paragraph);
    let base = &styles.get(id).character;
    base.apply(chpx, base)
        .apply(&pieces[text.piece_indexes[index] as usize].prm, base)
}
/// Python _auxiliary_contents deduplicates per section/kind and filters digit-only pages.
pub fn contents(paragraphs: Vec<Vec<Span>>) -> Vec<Vec<Span>> {
    let mut seen = BTreeSet::new();
    paragraphs
        .into_iter()
        .map(|s| rich(s, true))
        .filter(|s| {
            let plain = spans::visible(s);
            !plain.is_empty() && !plain.chars().all(digit) && seen.insert(plain)
        })
        .collect()
}
/// DocConverter._rich_text flushes ordinary runs before formulas; formulas do
/// not inherit a field hyperlink, and only the first text group trims its edges.
pub fn rich(content: Vec<Span>, trim: bool) -> Vec<Span> {
    fn flatten(content: Vec<Span>, url: Option<&str>, out: &mut Vec<Span>) {
        for s in content {
            match s {
                Span::Link { url, content } => flatten(content, Some(&url), out),
                Span::Text { .. } => out.push(if let Some(url) = url {
                    Span::Link {
                        url: url.into(),
                        content: vec![s],
                    }
                } else {
                    s
                }),
                Span::Equation { content } => out.push(Span::Equation {
                    content: content.replace('<', "\\lt ").replace('>', "\\gt "),
                }),
            }
        }
    }
    fn flush(group: &mut Vec<Span>, out: &mut Vec<Span>, trim: bool) {
        let group = std::mem::take(group);
        let group = if trim && out.is_empty() {
            spans::strip(group)
        } else {
            group
        };
        for span in group {
            if let Span::Link { url, content } = &span {
                if let Some(Span::Link {
                    url: previous,
                    content: children,
                }) = out.last_mut()
                {
                    if previous == url {
                        for child in content {
                            spans::append(children, child.clone());
                        }
                        continue;
                    }
                }
            }
            spans::append(out, span);
        }
    }
    let mut runs = vec![];
    flatten(content, None, &mut runs);
    let mut group = vec![];
    let mut out = vec![];
    for run in runs {
        if matches!(run, Span::Equation { .. }) {
            flush(&mut group, &mut out, trim);
            spans::append(&mut out, run);
        } else {
            group.push(run);
        }
    }
    flush(&mut group, &mut out, trim);
    if trim {
        spans::strip(out)
    } else {
        out
    }
}
fn digit(c: char) -> bool {
    static RANGES: std::sync::OnceLock<Vec<[u32; 2]>> = std::sync::OnceLock::new();
    let ranges =
        RANGES.get_or_init(|| serde_json::from_str(include_str!("digit_ranges.json")).unwrap());
    let n = c as u32;
    let i = ranges.partition_point(|r| r[1] < n);
    ranges.get(i).is_some_and(|r| r[0] <= n)
}

#[derive(Debug, serde::Serialize)]
pub struct NoteRange {
    pub reference: u32,
    pub start: u32,
    pub end: u32,
    pub label: String,
}
pub fn note_ranges(
    table: &[u8],
    refs: super::fib::Pair,
    text: super::fib::Pair,
    base: u32,
) -> (std::collections::BTreeMap<u32, String>, Vec<NoteRange>) {
    let region = |p: super::fib::Pair| {
        table.get(p.fc as usize..(p.fc as usize).checked_add(p.lcb as usize)?)
    };
    let (Some(refs), Some(text)) = (region(refs), region(text)) else {
        return Default::default();
    };
    let (cps, items) = super::navigation::plc(refs, 2);
    let (texts, _) = super::navigation::plc(text, 0);
    let mut references = std::collections::BTreeMap::new();
    let mut ranges = vec![];
    for i in 0..items.len().min(texts.len().saturating_sub(1)) {
        let (Some(start), Some(end)) = (base.checked_add(texts[i]), base.checked_add(texts[i + 1]))
        else {
            continue;
        };
        let label = (i + 1).to_string();
        references.insert(cps[i], label.clone());
        ranges.push(NoteRange {
            reference: cps[i],
            start,
            end,
            label,
        });
    }
    (references, ranges)
}
