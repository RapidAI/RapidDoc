//! BIFF drawing/OBJ/TXO association and native/image-comment equation priority.
use super::{
    chart,
    parser::{Cell, Workbook, Worksheet},
    records::{at, u16, u32, Budget},
    strings::{self, RichText},
};
use crate::{
    equation::{image_mtef, mtef},
    officeart::{self, Payload, Shape},
    ole::OleReader,
    Result,
};
use std::{collections::BTreeMap, sync::Arc};
#[derive(Debug, serde::Serialize)]
pub struct Image {
    pub row: u32,
    pub col: u32,
    pub image_base64: Arc<str>,
}
#[derive(Debug, serde::Serialize)]
pub struct Equation {
    pub row: u32,
    pub col: u32,
    pub latex: String,
}
#[derive(Debug, serde::Serialize)]
pub struct Chart {
    pub row: u32,
    pub col: u32,
    pub source_rows: Vec<u32>,
    pub source_cols: Vec<u32>,
    pub image_base64: Option<Arc<str>>,
}
#[derive(Debug)]
struct Object {
    kind: u16,
    id: u16,
    checked: Option<bool>,
    embedding: Option<String>,
    text: Option<RichText>,
    shape: Option<Shape>,
}
fn embedding(p: &[u8], flags: Option<u16>) -> Option<String> {
    let flags = flags?;
    if flags & (0x0002 | 0x0010 | 0x0020 | 0x0080) != 0 || p.len() < 10 {
        return None;
    }
    let size = u16(p, 0)? as usize;
    let end = 2 + size;
    if size == 0 || size % 2 != 0 || end + 4 > p.len() {
        return None;
    }
    let formula = &p[2..end];
    if formula.len() < 7 || u16(formula, 0)? & 0x7fff != 5 {
        return None;
    }
    if ![6, 2]
        .into_iter()
        .any(|i| i + 5 <= formula.len() && formula[i] == 2)
    {
        return None;
    }
    Some(format!("MBD{:08X}", u32(p, end)?))
}
fn object(payload: &[u8]) -> Option<Object> {
    let mut cursor = 0;
    let (mut kind, mut id, mut checked, mut flags, mut formula) = (None, 0, None, None, None);
    while cursor + 4 <= payload.len() {
        let sub = u16(payload, cursor)?;
        let size = u16(payload, cursor + 2)? as usize;
        let start = cursor + 4;
        let end = start + size;
        let Some(body) = payload.get(start..end) else {
            break;
        };
        match sub {
            0x0015 if body.len() >= 4 => {
                kind = u16(body, 0);
                id = u16(body, 2)?;
            }
            0x0012 if body.len() >= 2 => {
                checked = match u16(body, 0)? {
                    0 => Some(false),
                    1 => Some(true),
                    _ => None,
                }
            }
            0x0008 if body.len() >= 2 => flags = u16(body, 0),
            0x0009 => formula = Some(body),
            _ => {}
        }
        if sub == 0 {
            break;
        }
        cursor = end;
    }
    Some(Object {
        kind: kind?,
        id,
        checked,
        embedding: formula.and_then(|p| embedding(p, flags)),
        text: None,
        shape: None,
    })
}
pub(super) fn drawing_bytes(stream: &[u8], offsets: &[usize]) -> Vec<u8> {
    let mut out = vec![];
    for offset in offsets {
        let Some(r) = at(stream, *offset) else {
            continue;
        };
        out.extend_from_slice(r.payload);
        let mut cursor = r.next;
        while let Some(c) = at(stream, cursor).filter(|r| r.kind == 0x003c) {
            out.extend_from_slice(c.payload);
            cursor = c.next;
        }
    }
    out
}
fn append(sheet: &mut Worksheet, row: u32, col: u32, value: RichText) {
    if value.text.is_empty() {
        return;
    }
    if let Some(existing) = sheet.cells.get_mut(&(row, col)) {
        let separator = if existing.value.text.is_empty() {
            ""
        } else {
            "\n"
        };
        let shift = existing.value.text.chars().count() + separator.len();
        existing.value.text.push_str(separator);
        existing.value.text.push_str(&value.text);
        existing
            .value
            .runs
            .extend(value.runs.into_iter().map(|mut r| {
                r.start += shift;
                r.end += shift;
                r
            }));
    } else {
        sheet.cells.insert(
            (row, col),
            Cell {
                row,
                col,
                value,
                hyperlink: None,
            },
        );
    }
}
pub(super) fn bind(
    stream: &[u8],
    sheet: &mut Worksheet,
    fonts: &[strings::Font],
    external: &[Option<usize>],
    images: &BTreeMap<u32, Payload>,
    equations: &BTreeMap<String, String>,
    decoder: &mut image_mtef::Decoder,
    budget: &mut Budget,
    diagnostics: &mut Vec<String>,
) -> Result<()> {
    let drawing = sheet
        .resource_records
        .iter()
        .filter(|i| at(stream, **i).is_some_and(|r| r.kind == 0x00ec))
        .copied()
        .collect::<Vec<_>>();
    let drawing = drawing_bytes(stream, &drawing);
    let shapes = officeart::excel_shapes(&drawing, budget)?;
    let mut objects: Vec<Object> = vec![];
    for offset in &sheet.resource_records {
        let Some(r) = at(stream, *offset) else {
            continue;
        };
        match r.kind {
            0x005d => {
                if let Some(o) = object(r.payload) {
                    objects.push(o)
                }
            }
            0x01b6 => {
                if let Some(o) = objects.last_mut() {
                    let mut parts = vec![];
                    let mut cursor = r.next;
                    while let Some(c) = at(stream, cursor).filter(|r| r.kind == 0x003c) {
                        parts.push(c.payload);
                        cursor = c.next;
                    }
                    o.text = strings::txo(r.payload, &parts, fonts);
                }
            }
            _ => {}
        }
    }
    if shapes.len() != objects.len() {
        diagnostics.push(format!(
            "XLS_DRAWING_OBJECT_MISMATCH: {} shapes={} objects={}",
            sheet.name,
            shapes.len(),
            objects.len()
        ));
    }
    for (o, s) in objects.iter_mut().zip(shapes) {
        o.shape = Some(s);
    }
    for o in &objects {
        let Some(shape) = o.shape.as_ref().filter(|s| !s.hidden) else {
            continue;
        };
        let Some([row, col, _, _]) = shape.anchor else {
            continue;
        };
        if o.kind == 8 {
            if let Some(latex) = o
                .embedding
                .as_ref()
                .and_then(|s| equations.get(&crate::ole::casefold(s)))
                .filter(|s| !s.is_empty())
            {
                sheet.equations.push(Equation {
                    row,
                    col,
                    latex: latex.clone(),
                });
                continue;
            }
            if let Some(payload) = shape.pib.and_then(|p| images.get(&p)) {
                if let Some(latex) = decoder
                    .probe(
                        &payload.data,
                        Some(&format!("picture.{}", payload.extension)),
                        Some(payload.content_type),
                    )?
                    .filter(|s| !s.is_empty())
                {
                    sheet.equations.push(Equation { row, col, latex });
                    continue;
                }
            }
        }
        if o.kind == 6 {
            if let Some(text) = &o.text {
                append(sheet, row, col, text.clone());
            }
        } else if o.kind == 11 {
            if let Some(checked) = o.checked {
                let caption = o
                    .text
                    .as_ref()
                    .map(|s| s.text.trim_matches(crate::docx::space))
                    .unwrap_or("");
                let text = format!("{} {caption}", if checked { "[x]" } else { "[ ]" })
                    .trim_end_matches(crate::docx::space)
                    .to_owned();
                append(sheet, row, col, RichText { text, runs: vec![] });
            }
        }
        if matches!(o.kind, 8 | 5) || shape.pib.is_some() {
            if let Some(payload) = images.get(&shape.pib.unwrap_or(0)) {
                if let Some(source) = payload.serialize_with_decoder(decoder)? {
                    if o.kind != 5 {
                        sheet.images.push(Image {
                            row,
                            col,
                            image_base64: source,
                        });
                    }
                }
            }
        }
    }
    for (index, o) in objects.iter().filter(|o| o.kind == 5).enumerate() {
        let Some(shape) = o.shape.as_ref().filter(|s| !s.hidden) else {
            continue;
        };
        let Some([row, col, _, _]) = shape.anchor else {
            continue;
        };
        let axes = sheet
            .chart_records
            .get(index)
            .and_then(|offsets| {
                chart::selection(
                    offsets.iter().filter_map(|i| at(stream, *i)),
                    sheet.order,
                    external,
                )
            })
            .filter(|s| s.sheet_index == sheet.order);
        let preview = shape
            .pib
            .and_then(|p| images.get(&p))
            .map(Payload::serialize)
            .transpose()?
            .flatten();
        let (rows, cols) = if let Some(s) = axes {
            (s.rows, s.cols)
        } else if preview.is_some() {
            (vec![], vec![])
        } else {
            diagnostics.push(format!(
                "XLS_CHART_SOURCE_UNSUPPORTED: {} object_id={}",
                sheet.name, o.id
            ));
            continue;
        };
        sheet.charts.push(Chart {
            row,
            col,
            source_rows: rows,
            source_cols: cols,
            image_base64: preview,
        });
    }
    Ok(())
}
pub(super) fn native_equations(
    ole: &mut OleReader,
    diagnostics: &mut Vec<String>,
) -> Result<BTreeMap<String, String>> {
    let names = ole
        .stream_names(Some("MBD"))
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let pattern = regex::Regex::new(r"(?i)^(MBD[0-9A-F]{8})/Equat[iİı]on Nat[iİı]ve$").unwrap();
    let mut out = BTreeMap::new();
    for name in names {
        let Some(m) = pattern.captures(&name) else {
            continue;
        };
        let bytes = ole.read_stream(&name, true)?;
        let Some(latex) = mtef::equation_native(&bytes)? else {
            diagnostics.push(format!("XLS_MTEF_FALLBACK: {}", &m[1]));
            continue;
        };
        out.entry(crate::ole::casefold(&m[1])).or_insert(latex);
    }
    Ok(out)
}
impl Workbook {
    pub fn open_complete(data: Arc<[u8]>) -> Result<Self> {
        let mut ole = OleReader::open(data)?;
        Self::from_ole(&mut ole)
    }
    pub fn from_ole(ole: &mut OleReader) -> Result<Self> {
        if ole.has_stream("EncryptionInfo") || ole.has_stream("EncryptedPackage") {
            return Err(crate::Error::new(
                crate::ErrorKind::Encrypted,
                "password-protected XLS is not supported",
            ));
        }
        let stream = ole.read_stream(
            if ole.has_stream("Workbook") {
                "Workbook"
            } else {
                "Book"
            },
            true,
        )?;
        let mut diagnostics = vec![];
        let equations = native_equations(ole, &mut diagnostics)?;
        let mut workbook = Self::parse_bound(stream, &equations)?;
        diagnostics.append(&mut workbook.diagnostics);
        workbook.diagnostics = diagnostics;
        Ok(workbook)
    }
}
