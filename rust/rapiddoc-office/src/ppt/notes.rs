//! NotesAtom slide-id binding and the original direct-text fallback.
use super::{
    document::{self, Layout},
    records::{self, u16, u32, Budget, Record},
    shapes,
    styles::MasterLevel,
    text::{self, Paragraph},
};
use crate::Result;
use std::collections::BTreeMap;
pub fn hidden(slide: Record<'_>, budget: &mut Budget) -> Result<bool> {
    for r in records::descendants(slide, budget) {
        let r = r?;
        if r.kind == 1017 && r.payload.len() >= 12 {
            return Ok(u16(r.payload, 10).unwrap_or(0) & 4 != 0);
        }
    }
    Ok(false)
}
fn keep(p: &Paragraph) -> bool {
    let text = p.runs.iter().map(|r| r.text.as_str()).collect::<String>();
    let text = text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c));
    !text.is_empty() && text != "*"
}
pub fn paragraphs(
    note: Record<'_>,
    masters: &BTreeMap<u16, Vec<MasterLevel>>,
    links: &BTreeMap<u32, String>,
    budget: &mut Budget,
) -> Result<Vec<Paragraph>> {
    let mut out = vec![];
    let collection = shapes::collect(note, budget)?;
    for shape in &collection.shapes {
        if let Some(content) = shapes::shape_text(shape, &[], masters, links, budget)? {
            out.extend(content.paragraphs.into_iter().filter(keep));
        }
    }
    if !out.is_empty() {
        return Ok(out);
    }
    let records = records::descendants(note, budget).collect::<Result<Vec<_>>>()?;
    for content in text::parse(&records, masters, links, budget, text::is_control)? {
        out.extend(content.paragraphs.into_iter().filter(keep));
    }
    Ok(out)
}
pub fn bind(
    layout: &Layout<'_>,
    data: &[u8],
    masters: &BTreeMap<u16, Vec<MasterLevel>>,
    links: &BTreeMap<u32, String>,
    budget: &mut Budget,
) -> Result<(BTreeMap<u32, Vec<Paragraph>>, Vec<Vec<Paragraph>>)> {
    let mut bound = BTreeMap::new();
    let mut unbound = vec![];
    for (reference, _) in document::entries(layout.document, 2, budget)? {
        let Some(offset) = layout.persist.get(&reference) else {
            continue;
        };
        let Some(note) =
            records::at(data, *offset as usize, None, false, budget)?.filter(|r| r.kind == 1008)
        else {
            continue;
        };
        let mut id = None;
        for r in records::descendants(note, budget) {
            let r = r?;
            if r.kind == 1009 {
                id = u32(r.payload, 0);
                break;
            }
        }
        let paragraphs = paragraphs(note, masters, links, budget)?;
        if paragraphs.is_empty() {
            continue;
        }
        if let Some(id) = id.filter(|n| *n != 0) {
            bound.insert(id, paragraphs);
        } else {
            unbound.push(paragraphs);
        }
    }
    Ok((bound, unbound))
}
