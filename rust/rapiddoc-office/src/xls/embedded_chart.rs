//! Workbook/Book object streams use BIFF or MS-OGRAPH, never OOXML chart XML.
use super::{
    number_format,
    records::{self, at, u16, Budget, BOF, EOF},
    strings, Cell, Descriptor, Workbook, Worksheet,
};
use crate::{Error, ErrorKind, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

fn extent(sheet: &Worksheet) -> (Vec<u32>, Vec<u32>) {
    (
        sheet
            .cells
            .keys()
            .map(|p| p.0)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        sheet
            .cells
            .keys()
            .map(|p| p.1)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    )
}
fn render(sheet: &Worksheet, rows: &[u32], cols: &[u32]) -> Result<Option<String>> {
    if rows.is_empty() || cols.is_empty() {
        return Ok(None);
    }
    if rows.len().saturating_mul(cols.len()) > 4_000_000 {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "workbook extent exceeds original max_grid_slots",
        ));
    }
    Ok(Some(crate::spreadsheet::table::html(
        &sheet.project_cells()?.synthetic(rows, cols)?,
    )))
}
fn excel(data: Arc<[u8]>) -> Result<Option<String>> {
    let workbook = Workbook::parse_stream(data)?;
    let charts: Vec<_> = workbook.chart_sheets.iter().filter(|c| c.visible).collect();
    let chart = charts
        .iter()
        .find(|c| Some(c.order) == workbook.active_sheet_index)
        .copied()
        .or_else(|| {
            if charts.len() == 1 {
                Some(charts[0])
            } else {
                None
            }
        });
    if let Some(c) = chart {
        if let Some(sheet) = c
            .source_sheet_name
            .as_ref()
            .and_then(|n| workbook.sheets.iter().find(|s| &s.name == n))
        {
            if !c.source_rows.is_empty() && !c.source_cols.is_empty() {
                return render(sheet, &c.source_rows, &c.source_cols);
            }
        }
    }
    let sheets: Vec<_> = workbook
        .sheets
        .iter()
        .filter(|s| s.visible && !s.cells.is_empty())
        .collect();
    if sheets.len() != 1 {
        return Ok(None);
    }
    let (rows, cols) = extent(sheets[0]);
    render(sheets[0], &rows, &cols)
}
fn unicode(data: &[u8], offset: usize, short: bool) -> Option<String> {
    let (count, p) = if short {
        (*data.get(offset)? as usize, offset + 1)
    } else {
        (u16(data, offset)? as usize, offset + 2)
    };
    let flags = *data.get(p)?;
    if flags & 0xfe != 0 {
        return None;
    }
    let width = if flags & 1 != 0 { 2 } else { 1 };
    let bytes = data.get(p + 1..p + 1 + count * width)?;
    let text = if width == 2 {
        String::from_utf16(
            &bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect::<Vec<_>>(),
        )
        .ok()?
    } else {
        encoding_rs::WINDOWS_1252
            .decode_without_bom_handling_and_without_replacement(bytes)?
            .into_owned()
    };
    Some(strings::clean(&text))
}
fn included(candidates: Vec<u32>, payload: Option<&[u8]>) -> Vec<u32> {
    let Some(p) = payload.filter(|p| !p.is_empty() && p.len() % 4 == 0) else {
        return candidates;
    };
    let selected: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|n| {
            p.chunks_exact(4)
                .any(|b| *n >= u16(b, 0).unwrap() as u32 && *n < u16(b, 2).unwrap() as u32)
        })
        .collect();
    if selected.is_empty() {
        candidates
    } else {
        selected
    }
}
fn graph(data: &[u8]) -> Result<Option<String>> {
    let mut records = vec![];
    let mut budget = Budget::default();
    let mut cursor = 0;
    while let Some(r) = records::charged(data, cursor, &mut budget)? {
        cursor = r.next;
        records.push(r);
    }
    let bofs: Vec<_> = records.iter().filter(|r| r.kind == BOF).collect();
    if bofs.len() != 2
        || u16(bofs[0].payload, 0) != Some(0x680)
        || u16(bofs[0].payload, 2) != Some(5)
        || u16(bofs[1].payload, 0) != Some(0x680)
        || u16(bofs[1].payload, 2) != Some(0x8000)
    {
        return Ok(None);
    }
    let second = bofs[1].offset;
    let mut date1904 = false;
    let mut formats = BTreeMap::new();
    let (mut in_chart, mut in_datasheet) = (false, false);
    let (mut rows, mut cols) = (None, None);
    let mut sheet = Worksheet::empty(
        &Descriptor {
            name: "Data".into(),
            offset: 0,
            visible: true,
            sheet_type: 0,
        },
        0,
        false,
    );
    for r in records {
        if r.offset == second {
            in_chart = true;
            continue;
        }
        if !in_chart {
            match r.kind {
                0x22 => date1904 = u16(r.payload, 0) == Some(1),
                0x41e => {
                    if let (Some(id), Some(value)) =
                        (u16(r.payload, 0), unicode(r.payload, 2, false))
                    {
                        if !value.is_empty() {
                            formats.insert(id, value);
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        match r.kind {
            EOF => break,
            0x1052 => {
                in_datasheet = true;
                continue;
            }
            0x1053 => {
                rows = Some(r.payload);
                continue;
            }
            0x1054 => {
                cols = Some(r.payload);
                continue;
            }
            _ => {}
        }
        if !in_datasheet || r.payload.len() < 7 {
            continue;
        }
        let row = u16(r.payload, 0).unwrap() as u32;
        let col = u16(r.payload, 2).unwrap() as u32;
        if row >= 4000 || col >= 256 {
            continue;
        }
        let id = u16(r.payload, 5).unwrap();
        let text = match r.kind {
            0x204 => unicode(r.payload, 7, true),
            3 => records::f64(r.payload, 7)
                .map(|v| {
                    number_format::number(
                        v,
                        formats
                            .get(&id)
                            .map(String::as_str)
                            .or_else(|| number_format::builtin(id)),
                        date1904,
                    )
                })
                .transpose()?,
            1 => Some(String::new()),
            _ => None,
        };
        if let Some(text) = text {
            sheet.cells.insert(
                (row, col),
                Cell {
                    row,
                    col,
                    value: strings::RichText { text, runs: vec![] },
                    hyperlink: None,
                },
            );
        }
    }
    let (rr, cc) = extent(&sheet);
    render(&sheet, &included(rr, rows), &included(cc, cols))
}
pub fn extract(data: Arc<[u8]>) -> Result<Option<String>> {
    let Some(first) = at(&data, 0).filter(|r| r.kind == BOF) else {
        return Ok(None);
    };
    let result = match u16(first.payload, 0) {
        Some(0x500 | 0x600) => excel(data),
        Some(0x680) => graph(&data),
        _ => Ok(None),
    };
    match result {
        Err(e) if e.kind == ErrorKind::Malformed => Ok(None),
        other => other,
    }
}
