//! Original UserEdit/persist newest-first lookup and top-level recovery.
use super::{
    records::{self, u32, Budget, Record},
    styles::{self, MasterLevel},
};
use crate::{Error, ErrorKind, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
pub struct Layout<'a> {
    pub document: Record<'a>,
    pub persist: BTreeMap<u32, u32>,
    pub recovered: bool,
}
/// Original parser preflight: Current User compatibility, complete recursive
/// record validation, latest saved document and CryptSession detection.
pub fn prepare<'a>(data: &'a [u8], current: &[u8]) -> Result<(Layout<'a>, Budget)> {
    if !current.is_empty() {
        if records::u16(current, 2).is_some_and(|v| v != 0xff6) {
            return Err(Error::malformed(
                "PowerPoint 95 or earlier Current User stream is unsupported",
            ));
        }
        if u32(current, 12) == Some(0xf3d1c4df) {
            return Err(Error::new(
                ErrorKind::Encrypted,
                "password-protected PPT is unsupported",
            ));
        }
    }
    let mut budget = Budget::default();
    let mut cursor = 0;
    while let Some(root) = records::at(data, cursor, None, cursor == 0, &mut budget)? {
        cursor += 8 + root.payload.len();
        for r in records::descendants(root, &mut budget) {
            r?;
        }
        if cursor >= data.len() {
            break;
        }
    }
    let layout = locate(data, current, &mut budget)?;
    for r in records::descendants(layout.document, &mut budget) {
        if r?.kind == 0x2f14 {
            return Err(Error::new(
                ErrorKind::Encrypted,
                "encrypted PPT record stream is unsupported",
            ));
        }
    }
    Ok((layout, budget))
}
pub fn merge(map: &mut BTreeMap<u32, u32>, payload: &[u8]) -> Result<()> {
    let mut p = 0;
    while p < payload.len() {
        let head = u32(payload, p)
            .ok_or_else(|| Error::malformed("PowerPoint persist directory is truncated"))?;
        p += 4;
        let first = head & 0xfffff;
        let count = head >> 20;
        if count == 0 || p + count as usize * 4 > payload.len() {
            return Err(Error::malformed(
                "PowerPoint persist directory entry is invalid",
            ));
        }
        for i in 0..count {
            let offset = u32(payload, p).unwrap();
            p += 4;
            map.entry(first + i).or_insert(offset);
        }
    }
    Ok(())
}
pub fn locate<'a>(data: &'a [u8], current_user: &[u8], budget: &mut Budget) -> Result<Layout<'a>> {
    let mut edit_offset = u32(current_user, 16).unwrap_or(0) as usize;
    let mut persist = BTreeMap::new();
    let mut doc_id = None;
    let mut seen = BTreeSet::new();
    let lookup = (|| -> Result<Option<Record<'a>>> {
        let mut ended = false;
        for _ in 0..100 {
            if edit_offset == 0 {
                ended = true;
                break;
            }
            if !seen.insert(edit_offset) {
                return Err(Error::malformed("PowerPoint UserEdit chain is cyclic"));
            }
            let edit = records::at(data, edit_offset, None, true, budget)?
                .filter(|r| r.kind == 0xff5)
                .ok_or_else(|| Error::malformed("PowerPoint UserEditAtom is invalid"))?;
            if doc_id.is_none() {
                doc_id = u32(edit.payload, 16);
            }
            let offset = u32(edit.payload, 12)
                .ok_or_else(|| Error::malformed("PowerPoint persist directory offset is missing"))?
                as usize;
            let directory = records::at(data, offset, None, true, budget)?
                .filter(|r| r.kind == 0x1772)
                .ok_or_else(|| Error::malformed("PowerPoint PersistDirectoryAtom is invalid"))?;
            merge(&mut persist, directory.payload)?;
            let previous = u32(edit.payload, 8).unwrap_or(0) as usize;
            if previous == edit_offset {
                return Err(Error::malformed(
                    "PowerPoint UserEdit chain points to itself",
                ));
            }
            edit_offset = previous;
        }
        if !ended {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "UserEdit chain exceeds original max_user_edit_chain=100",
            ));
        }
        if let Some(offset) = doc_id.and_then(|id| persist.get(&id)) {
            return Ok(
                records::at(data, *offset as usize, None, true, budget)?.filter(|r| r.kind == 1000)
            );
        }
        Ok(None)
    })();
    match lookup {
        Ok(Some(document)) => {
            return Ok(Layout {
                document,
                persist,
                recovered: false,
            })
        }
        Err(e) if e.kind == ErrorKind::ResourceLimit => return Err(e),
        _ => {}
    }
    let mut document = None;
    for r in records::iter(data, 0, None, false, budget) {
        let r = r?;
        if r.kind == 1000 && r.version == 15 {
            document = Some(r);
        }
    }
    let document =
        document.ok_or_else(|| Error::malformed("PowerPoint DocumentContainer is missing"))?;
    let mut recovered = BTreeMap::new();
    for r in records::iter(data, 0, None, false, budget) {
        let r = r?;
        if r.kind == 0x1772 {
            let _ = merge(&mut recovered, r.payload);
        }
    }
    Ok(Layout {
        document,
        persist: recovered,
        recovered: true,
    })
}
pub fn entries(
    document: Record<'_>,
    instance: u16,
    budget: &mut Budget,
) -> Result<Vec<(u32, u32)>> {
    let mut container = None;
    for r in records::descendants(document, budget) {
        let r = r?;
        if r.kind == 0xff0 && r.instance == instance {
            container = Some(r);
            break;
        }
    }
    let mut out = vec![];
    if let Some(container) = container {
        for r in records::iter(container.payload, 0, None, false, budget) {
            let r = r?;
            if r.kind == 1011 {
                if let Some(reference) = u32(r.payload, 0).filter(|n| *n != 0) {
                    let id = u32(r.payload, 12).unwrap_or(0);
                    if instance != 1 || id != 0 {
                        out.push((reference, id));
                    }
                }
            }
        }
    }
    Ok(out)
}
pub fn size(document: Record<'_>, budget: &mut Budget) -> Result<[u32; 2]> {
    for r in records::descendants(document, budget) {
        let r = r?;
        if r.kind == 1001 {
            if let (Some(w), Some(h)) = (u32(r.payload, 0), u32(r.payload, 4)) {
                if w > 0 && h > 0 && w < 100000 && h < 100000 {
                    return Ok([w, h]);
                }
            }
        }
    }
    Ok([5760, 4320])
}
pub fn master_id(slide: Record<'_>, budget: &mut Budget) -> Result<Option<u32>> {
    for r in records::descendants(slide, budget) {
        let r = r?;
        if r.kind == 1007 {
            return Ok(u32(r.payload, 12).filter(|n| *n != 0));
        }
    }
    Ok(None)
}
pub fn master_styles(
    r: Record<'_>,
    budget: &mut Budget,
) -> Result<BTreeMap<u16, Vec<MasterLevel>>> {
    let mut out = BTreeMap::new();
    for r in records::descendants(r, budget) {
        let r = r?;
        if r.kind == 0xfa3 {
            out.entry(r.instance)
                .or_insert_with(|| styles::master(r.payload, r.instance));
        }
    }
    Ok(out)
}
pub type Master<'a> = (Record<'a>, BTreeMap<u16, Vec<MasterLevel>>);
pub fn masters<'a>(
    layout: &Layout<'a>,
    data: &'a [u8],
    budget: &mut Budget,
) -> Result<Vec<(u32, Master<'a>)>> {
    let mut masters = vec![];
    for (reference, id) in entries(layout.document, 1, budget)? {
        let Some(offset) = layout.persist.get(&reference) else {
            continue;
        };
        let Some(record) = records::at(data, *offset as usize, None, false, budget)?
            .filter(|r| matches!(r.kind, 1016 | 1006))
        else {
            continue;
        };
        let styles = master_styles(record, budget)?;
        if !masters.iter().any(|(key, _)| *key == id) {
            masters.push((id, (record, styles)));
        }
    }
    if masters.is_empty() {
        let mut cursor = 0;
        while let Some(root) = records::at(data, cursor, None, false, budget)? {
            cursor += 8 + root.payload.len();
            if root.kind != 1016 {
                continue;
            }
            let styles = master_styles(root, budget)?;
            if masters.is_empty() {
                masters.push((0, (root, styles)));
            }
        }
    }
    Ok(masters)
}
pub fn external_text<'a>(
    document: Record<'a>,
    budget: &mut Budget,
) -> Result<BTreeMap<u32, Vec<Record<'a>>>> {
    let mut found = None;
    for r in records::descendants(document, budget) {
        let r = r?;
        if r.kind == 0xff0 && r.instance == 0 {
            found = Some(r);
            break;
        }
    }
    let mut groups = BTreeMap::new();
    let mut current = None;
    if let Some(container) = found {
        for r in records::iter(container.payload, 0, None, false, budget) {
            let r = r?;
            if r.kind == 1011 {
                current = u32(r.payload, 0).filter(|n| *n != 0);
                if let Some(reference) = current {
                    groups.entry(reference).or_insert_with(Vec::new);
                }
                continue;
            }
            if let Some(reference) = current {
                groups.get_mut(&reference).unwrap().push(r);
            }
        }
    }
    Ok(groups)
}
#[derive(Debug, Serialize)]
pub struct Info {
    pub document_offset: usize,
    pub persist: BTreeMap<u32, u32>,
    pub recovered: bool,
    pub size: [u32; 2],
    pub slides: Vec<(u32, u32)>,
    pub masters: Vec<(u32, u32)>,
    pub notes: Vec<u32>,
    pub record_visits: usize,
}
pub fn inspect(data: &[u8], current_user: &[u8]) -> Result<Info> {
    let mut budget = Budget::default();
    let layout = locate(data, current_user, &mut budget)?;
    let size = size(layout.document, &mut budget)?;
    let slides = entries(layout.document, 0, &mut budget)?;
    let masters = entries(layout.document, 1, &mut budget)?;
    let notes = entries(layout.document, 2, &mut budget)?
        .into_iter()
        .map(|(r, _)| r)
        .collect();
    Ok(Info {
        document_offset: layout.document.offset,
        persist: layout.persist,
        recovered: layout.recovered,
        size,
        slides,
        masters,
        notes,
        record_visits: budget.count,
    })
}
