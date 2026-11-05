//! MS-DOC FTXBXS/PlcfSpa and comment indices; no unbound-story guessing.
use super::{fib::Pair, navigation::plc};
use std::collections::BTreeMap;
pub struct Range {
    pub anchor: u32,
    pub start: u32,
    pub end: u32,
}
fn region(table: &[u8], pair: Pair) -> Option<&[u8]> {
    table.get(pair.fc as usize..(pair.fc as usize).checked_add(pair.lcb as usize)?)
}
fn u32(d: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(d[p..p + 4].try_into().unwrap())
}
pub fn textboxes(
    table: &[u8],
    text: Pair,
    shapes: Pair,
    base: u32,
    count: u32,
    diagnostics: &mut Vec<String>,
) -> Vec<Range> {
    let (Some(text), Some(shapes)) = (region(table, text), region(table, shapes)) else {
        if text.lcb != 0 {
            diagnostics.push("DOC_TEXTBOX_INDEX: invalid textbox/shape range".into());
        }
        return vec![];
    };
    let (cps, entries) = plc(text, 22);
    if count != 0 && entries.is_empty() {
        diagnostics.push("DOC_TEXTBOX_INDEX: malformed textbox PLC".into());
    }
    let (anchors, descriptors) = plc(shapes, 26);
    let mut by_shape = BTreeMap::new();
    for (cp, d) in anchors.iter().zip(descriptors) {
        by_shape.entry(u32(d, 0)).or_insert(*cp);
    }
    let mut out = vec![];
    for (i, entry) in entries
        .iter()
        .enumerate()
        .take(entries.len().saturating_sub(1))
    {
        if u16::from_le_bytes(entry[8..10].try_into().unwrap()) != 0 {
            continue;
        }
        let Some(&anchor) = by_shape.get(&u32(entry, 14)) else {
            diagnostics.push(format!("DOC_TEXTBOX_INDEX: skipped unbound textbox {i}"));
            continue;
        };
        if cps[i] >= cps[i + 1] || cps[i + 1] > count || cps[i + 1] - cps[i] <= 1 {
            diagnostics.push(format!("DOC_TEXTBOX_INDEX: skipped invalid textbox {i}"));
            continue;
        }
        if let (Some(start), Some(end)) = (base.checked_add(cps[i]), base.checked_add(cps[i + 1])) {
            out.push(Range { anchor, start, end });
        }
    }
    out
}
pub fn comments(
    table: &[u8],
    refs: Pair,
    text: Pair,
    base: u32,
    count: u32,
    main: u32,
) -> (BTreeMap<u32, String>, Vec<super::auxiliary::NoteRange>) {
    let (Some(refs), Some(text)) = (region(table, refs), region(table, text)) else {
        return Default::default();
    };
    let (cps, entries) = plc(refs, 30);
    let (texts, _) = plc(text, 0);
    let mut references = BTreeMap::new();
    let mut out = vec![];
    for i in 0..entries.len().min(texts.len().saturating_sub(1)) {
        if cps[i] >= main
            || texts[i] >= texts[i + 1]
            || texts[i + 1] > count
            || references.contains_key(&cps[i])
        {
            continue;
        }
        if let (Some(start), Some(end)) =
            (base.checked_add(texts[i]), base.checked_add(texts[i + 1]))
        {
            let label = format!("comment {}", i + 1);
            references.insert(cps[i], label.clone());
            out.push(super::auxiliary::NoteRange {
                reference: cps[i],
                start,
                end,
                label,
            });
        }
    }
    (references, out)
}
