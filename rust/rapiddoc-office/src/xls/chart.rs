//! Original BRAI reference subset: no evaluation, cache guessing or cross-sheet
//! unions. BIFF token class bits are masked exactly like the Python reader.
use super::records::{u16, Record};
use serde::Serialize;
use std::collections::BTreeSet;
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct Selection {
    pub sheet_index: usize,
    pub rows: Vec<u32>,
    pub cols: Vec<u32>,
}
fn reference(tokens: &[u8], current: usize, external: &[Option<usize>]) -> Option<Selection> {
    let token = *tokens.first()? & 0x1f;
    let (sheet, offset, area) = match token {
        4 if tokens.len() >= 5 => (current, 1, false),
        5 if tokens.len() >= 9 => (current, 1, true),
        0x1a if tokens.len() >= 7 => (
            external.get(u16(tokens, 1)? as usize).copied().flatten()?,
            3,
            false,
        ),
        0x1b if tokens.len() >= 11 => (
            external.get(u16(tokens, 1)? as usize).copied().flatten()?,
            3,
            true,
        ),
        _ => return None,
    };
    if !area {
        return Some(Selection {
            sheet_index: sheet,
            rows: vec![u16(tokens, offset)? as u32],
            cols: vec![(u16(tokens, offset + 2)? & 0xff) as u32],
        });
    }
    let (a, b, c, d) = (
        u16(tokens, offset)? as u32,
        u16(tokens, offset + 2)? as u32,
        (u16(tokens, offset + 4)? & 0xff) as u32,
        (u16(tokens, offset + 6)? & 0xff) as u32,
    );
    // Ordinary PtgArea leaves reversed columns empty, while Area3d sorts them.
    let (c, d) = if token == 0x1b {
        (c.min(d), c.max(d))
    } else {
        (c, d)
    };
    Some(Selection {
        sheet_index: sheet,
        rows: (a.min(b)..=a.max(b)).collect(),
        cols: (c..=d).collect(),
    })
}
pub fn selection<'a>(
    records: impl IntoIterator<Item = Record<'a>>,
    current: usize,
    external: &[Option<usize>],
) -> Option<Selection> {
    let mut rows = BTreeSet::new();
    let mut cols = BTreeSet::new();
    let mut sheets = BTreeSet::new();
    let mut found = false;
    for r in records {
        if r.kind != 0x1051 || r.payload.len() < 8 {
            continue;
        }
        let n = u16(r.payload, 6)? as usize;
        if n == 0 {
            continue;
        }
        found = true;
        let tokens = r.payload.get(8..8 + n)?;
        let reference = reference(tokens, current, external)?;
        sheets.insert(reference.sheet_index);
        rows.extend(reference.rows);
        cols.extend(reference.cols);
    }
    if !found || rows.is_empty() || cols.is_empty() || sheets.len() != 1 {
        return None;
    }
    Some(Selection {
        sheet_index: *sheets.first().unwrap(),
        rows: rows.into_iter().collect(),
        cols: cols.into_iter().collect(),
    })
}
