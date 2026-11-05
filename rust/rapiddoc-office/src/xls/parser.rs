//! Faithful BIFF directory recovery, cell caches and rich-text materialization.
use super::{
    number_format,
    records::{at, charged, continues, f64, u16, u32, Budget, Segments, BOF, EOF},
    strings::{self, Decoded, Font, RichText},
};
use crate::{ole::OleReader, Error, ErrorKind, Result};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
#[derive(Debug, Clone, Serialize)]
pub struct Descriptor {
    pub name: String,
    pub offset: u32,
    pub visible: bool,
    pub sheet_type: u8,
}
#[derive(Debug, Clone, Serialize)]
pub struct Cell {
    pub row: u32,
    pub col: u32,
    pub value: RichText,
    pub hyperlink: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct Worksheet {
    pub name: String,
    pub visible: bool,
    pub order: usize,
    pub cells: BTreeMap<(u32, u32), Cell>,
    pub merges: Vec<(u32, u32, u32, u32)>,
    pub recovered: bool,
    /// Offsets retained for the next native OfficeArt/chart binding stage.
    pub resource_records: Vec<usize>,
    pub chart_records: Vec<Vec<usize>>,
    pub images: Vec<super::resources::Image>,
    pub equations: Vec<super::resources::Equation>,
    pub charts: Vec<super::resources::Chart>,
}
impl Worksheet {
    pub(super) fn empty(d: &Descriptor, order: usize, recovered: bool) -> Self {
        Self {
            name: d.name.clone(),
            visible: d.visible,
            order,
            cells: BTreeMap::new(),
            merges: vec![],
            recovered,
            resource_records: vec![],
            chart_records: vec![],
            images: vec![],
            equations: vec![],
            charts: vec![],
        }
    }
    fn put(&mut self, row: u32, col: u32, value: RichText) {
        if !value.text.is_empty() {
            self.cells.insert(
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
}
#[derive(Default)]
struct Globals {
    biff8: bool,
    date1904: bool,
    codepage: u16,
    descriptors: Vec<Descriptor>,
    strings: Vec<Decoded>,
    fonts: Vec<Font>,
    xfs: Vec<(usize, Option<String>)>,
    active_sheet_index: Option<usize>,
    resource_records: Vec<usize>,
    extern_sheets: Vec<Option<usize>>,
}
impl Globals {
    fn read(&self, r: &mut Segments<'_>, short: bool, rich: bool) -> Option<Decoded> {
        strings::read(r, self.biff8, self.codepage, short, rich)
    }
    fn format(&self, index: usize) -> Option<&str> {
        self.xfs.get(index).and_then(|(_, s)| s.as_deref())
    }
    fn rich(&self, d: &Decoded, index: usize) -> RichText {
        let formatted = number_format::text(&d.text, self.format(index));
        if formatted != d.text {
            RichText {
                text: formatted,
                runs: vec![],
            }
        } else {
            strings::rich(d, &self.fonts)
        }
    }
    fn numeric(&self, v: f64, index: usize) -> Result<RichText> {
        Ok(RichText {
            text: number_format::number(v, self.format(index), self.date1904)?,
            runs: vec![],
        })
    }
}
pub struct Workbook {
    pub stream: Arc<[u8]>,
    pub sheets: Vec<Worksheet>,
    pub descriptors: Vec<Descriptor>,
    pub active_sheet_index: Option<usize>,
    pub date1904: bool,
    pub biff8: bool,
    pub codepage: u16,
    pub fonts: Vec<Font>,
    pub extern_sheets: Vec<Option<usize>>,
    pub global_resource_records: Vec<usize>,
    pub diagnostics: Vec<String>,
    pub record_visits: usize,
    pub chart_sheets: Vec<ChartSheet>,
    pub resources_bound: bool,
}
#[derive(Debug, Serialize)]
pub struct ChartSheet {
    pub name: String,
    pub visible: bool,
    pub order: usize,
    pub source_sheet_name: Option<String>,
    pub source_rows: Vec<u32>,
    pub source_cols: Vec<u32>,
}
fn font(p: &[u8]) -> Font {
    if p.len() < 11 {
        return Font::default();
    }
    let flags = u16(p, 2).unwrap_or(0);
    let weight = u16(p, 6).filter(|v| *v != 0).unwrap_or(400);
    let esc = u16(p, 8).unwrap_or(0);
    Font {
        bold: weight >= 700,
        italic: flags & 2 != 0,
        strike: flags & 8 != 0,
        underline: p[10] != 0,
        superscript: esc == 1,
        subscript: esc == 2,
    }
}
fn bound(p: &[u8], g: &Globals) -> Option<Descriptor> {
    if p.len() < 8 {
        return None;
    }
    let mut r = Segments::new(vec![&p[6..]]);
    let d = g.read(&mut r, true, false)?;
    Some(Descriptor {
        name: if d.text.is_empty() {
            "Sheet".into()
        } else {
            d.text
        },
        offset: u32(p, 0)?,
        visible: p[4] & 3 == 0,
        sheet_type: p[5],
    })
}
fn read_sst(parts: Vec<&[u8]>, diagnostics: &mut Vec<String>) -> Vec<Decoded> {
    let mut r = Segments::new(parts);
    let total = r.u32();
    let count = r.u32();
    let Some(count) = count.filter(|_| total.is_some()) else {
        return vec![];
    };
    let mut out = vec![];
    for _ in 0..count {
        let Some(s) = strings::biff8(&mut r, false, true) else {
            diagnostics.push(format!(
                "XLS_SST_TRUNCATED: {} completed strings",
                out.len()
            ));
            break;
        };
        out.push(s);
    }
    out
}
fn globals(data: &[u8], budget: &mut Budget, diagnostics: &mut Vec<String>) -> Result<Globals> {
    let first = charged(data, 0, budget)?
        .filter(|r| r.kind == BOF)
        .ok_or_else(|| Error::malformed("workbook stream does not start with a BOF record"))?;
    if u16(first.payload, 2).is_some_and(|kind| kind != 5) {
        return Err(Error::malformed(
            "first BIFF substream is not workbook globals",
        ));
    }
    let version = u16(first.payload, 0);
    if !matches!(version, Some(0x0500 | 0x0600)) {
        return Err(Error::malformed(format!(
            "unsupported BIFF version: {version:?}"
        )));
    }
    let mut g = Globals {
        biff8: version == Some(0x0600),
        codepage: 1252,
        ..Default::default()
    };
    let mut formats = BTreeMap::<u16, String>::new();
    let mut raw_xfs = vec![];
    let mut internal_supbooks = vec![];
    let mut external = vec![];
    let (mut cursor, mut depth) = (first.next, 1usize);
    while cursor < data.len() {
        let Some(r) = charged(data, cursor, budget)? else {
            diagnostics.push(format!("XLS_GLOBALS_TRUNCATED: {cursor}"));
            break;
        };
        cursor = r.next;
        if r.kind == BOF {
            depth += 1;
            continue;
        }
        if r.kind == EOF {
            depth -= 1;
            if depth == 0 {
                break;
            }
            continue;
        }
        if depth != 1 {
            continue;
        }
        match r.kind {
            0x002f => {
                return Err(Error::new(
                    ErrorKind::Encrypted,
                    "password-protected XLS is not supported",
                ))
            }
            0x0042 => g.codepage = u16(r.payload, 0).filter(|v| *v != 0).unwrap_or(1252),
            0x0022 => g.date1904 = u16(r.payload, 0) == Some(1),
            0x0031 => g.fonts.push(font(r.payload)),
            0x0085 => {
                if let Some(d) = bound(r.payload, &g).filter(|d| d.sheet_type != 6) {
                    g.descriptors.push(d)
                }
            }
            0x041e => {
                if let Some(id) = u16(r.payload, 0) {
                    let mut q = Segments::new(vec![&r.payload[2..]]);
                    if let Some(s) = g.read(&mut q, !g.biff8, false) {
                        formats.insert(id, s.text);
                    }
                }
            }
            0x00e0 => raw_xfs.push((
                u16(r.payload, 0).unwrap_or(0) as usize,
                u16(r.payload, 2).unwrap_or(0),
            )),
            0x00fc if g.biff8 => {
                let (parts, next) = continues(data, r, budget)?;
                cursor = next;
                g.strings = read_sst(parts, diagnostics);
            }
            0x00eb => {
                g.resource_records.push(r.offset);
                let (_, next) = continues(data, r, budget)?;
                cursor = next;
            }
            0x01ae => {
                internal_supbooks.push(r.payload.len() >= 4 && u16(r.payload, 2) == Some(0x0401))
            }
            0x0017 => external.push(r.payload),
            0x003d if r.payload.len() >= 12 => {
                g.active_sheet_index = u16(r.payload, 10).map(usize::from)
            }
            _ => {}
        }
    }
    g.xfs = raw_xfs
        .into_iter()
        .map(|(font, id)| {
            (
                font,
                formats
                    .get(&id)
                    .filter(|s| !s.is_empty())
                    .cloned()
                    .or_else(|| number_format::builtin(id).map(str::to_owned)),
            )
        })
        .collect();
    for p in external {
        let count = (u16(p, 0).unwrap_or(0) as usize).min(p.len().saturating_sub(2) / 6);
        for i in 0..count {
            let offset = 2 + i * 6;
            let book = u16(p, offset).unwrap() as usize;
            let first = u16(p, offset + 2).unwrap();
            let last = u16(p, offset + 4).unwrap();
            g.extern_sheets.push(
                if internal_supbooks.get(book) == Some(&true) && first == last && first < 0xfffe {
                    Some(first as usize)
                } else {
                    None
                },
            );
        }
    }
    Ok(g)
}
pub(super) fn cell_ref(p: &[u8]) -> Option<(u32, u32, usize)> {
    let col = u16(p, 2)?;
    if col >= 256 {
        return None;
    }
    Some((u16(p, 0)? as u32, col as u32, u16(p, 4)? as usize))
}
pub(super) fn error_literal(code: u8) -> Option<&'static str> {
    Some(match code {
        0 => "#NULL!",
        7 => "#DIV/0!",
        15 => "#VALUE!",
        23 => "#REF!",
        29 => "#NAME?",
        36 => "#NUM!",
        42 => "#N/A",
        43 => "#GETTING_DATA",
        _ => return None,
    })
}
fn rk(value: u32) -> f64 {
    let n = if value & 2 != 0 {
        (value as i32 >> 2) as f64
    } else {
        f64::from_bits((value as u64 & 0xffff_fffc) << 32)
    };
    if value & 1 != 0 {
        n / 100.
    } else {
        n
    }
}
fn label(parts: Vec<&[u8]>, g: &Globals, rich_record: bool) -> Option<Decoded> {
    if parts.first()?.len() < 6 {
        return None;
    }
    let mut adjusted = parts;
    adjusted[0] = &adjusted[0][6..];
    let mut reader = Segments::new(adjusted);
    let mut decoded = g.read(&mut reader, false, false)?;
    if rich_record {
        if let Some(n) = reader.u16() {
            for _ in 0..n {
                let Some(b) = reader.across(4) else { break };
                decoded.font_starts.push((u16(&b, 0)?, u16(&b, 2)?));
            }
        }
    }
    Some(decoded)
}
fn is_worksheet(data: &[u8], offset: usize) -> bool {
    at(data, offset).is_some_and(|r| r.kind == BOF && u16(r.payload, 2) == Some(0x0010))
}
fn read_sheet(
    data: &[u8],
    g: &Globals,
    d: &Descriptor,
    offset: usize,
    order: usize,
    recovered: bool,
    budget: &mut Budget,
    diagnostics: &mut Vec<String>,
) -> Result<Option<Worksheet>> {
    let Some(first) = charged(data, offset, budget)?
        .filter(|r| r.kind == BOF && u16(r.payload, 2) == Some(0x0010))
    else {
        return Ok(None);
    };
    let mut sheet = Worksheet::empty(d, order, recovered);
    let (mut cursor, mut depth) = (first.next, 1usize);
    let mut chart: Option<Vec<usize>> = None;
    let mut pending = None;
    let mut links = BTreeMap::new();
    while cursor < data.len() {
        let Some(r) = charged(data, cursor, budget)? else {
            diagnostics.push(format!("XLS_SHEET_TRUNCATED: {} at {cursor}", sheet.name));
            break;
        };
        cursor = r.next;
        if let Some(active) = &mut chart {
            active.push(r.offset);
            if r.kind == BOF {
                depth += 1
            } else if r.kind == EOF {
                depth -= 1;
                if depth == 1 {
                    sheet.chart_records.push(chart.take().unwrap());
                }
            }
            continue;
        }
        if r.kind == BOF {
            depth += 1;
            if depth == 2 && u16(r.payload, 2) == Some(0x0020) {
                chart = Some(vec![r.offset]);
            }
            continue;
        }
        if r.kind == EOF {
            depth -= 1;
            if depth == 0 {
                break;
            }
            continue;
        }
        if depth != 1 {
            continue;
        }
        match r.kind {
            0x00e5 => {
                let n = (u16(r.payload, 0).unwrap_or(0) as usize)
                    .min(r.payload.len().saturating_sub(2) / 8);
                for i in 0..n {
                    let p = 2 + i * 8;
                    let (a, b, c, d) = (
                        u16(r.payload, p).unwrap() as u32,
                        u16(r.payload, p + 2).unwrap() as u32,
                        u16(r.payload, p + 4).unwrap() as u32,
                        u16(r.payload, p + 6).unwrap() as u32,
                    );
                    let (row, last, col, end) = (a.min(b), a.max(b), c.min(d), c.max(d));
                    if col >= 256 || row == last && col == end {
                        continue;
                    }
                    sheet.merges.push((row, col, last.min(65535), end.min(255)));
                }
            }
            0x00fd => {
                if let (Some((row, col, xf)), Some(index)) =
                    (cell_ref(r.payload), u32(r.payload, 6))
                {
                    if let Some(s) = g.strings.get(index as usize) {
                        sheet.put(row, col, g.rich(s, xf));
                    }
                }
            }
            0x0204 | 0x00d6 => {
                let (parts, next) = continues(data, r, budget)?;
                cursor = next;
                if let (Some((row, col, xf)), Some(s)) =
                    (cell_ref(r.payload), label(parts, g, r.kind == 0x00d6))
                {
                    sheet.put(row, col, g.rich(&s, xf));
                }
            }
            0x0203 => {
                if let (Some((row, col, xf)), Some(value)) =
                    (cell_ref(r.payload), f64(r.payload, 6))
                {
                    sheet.put(row, col, g.numeric(value, xf)?);
                }
            }
            0x027e => {
                if let (Some((row, col, xf)), Some(value)) =
                    (cell_ref(r.payload), u32(r.payload, 6))
                {
                    sheet.put(row, col, g.numeric(rk(value), xf)?);
                }
            }
            0x00bd if r.payload.len() >= 6 => {
                let row = u16(r.payload, 0).unwrap() as u32;
                let first = u16(r.payload, 2).unwrap() as u32;
                let count = (r.payload.len() - 6) / 6;
                for i in 0..count {
                    let xf = u16(r.payload, 4 + i * 6).unwrap() as usize;
                    let Some(raw) = u32(r.payload, 6 + i * 6) else {
                        break;
                    };
                    let col = first + i as u32;
                    if col >= 256 {
                        break;
                    }
                    sheet.put(row, col, g.numeric(rk(raw), xf)?);
                }
            }
            0x0205 if r.payload.len() >= 8 => {
                if let Some((row, col, _)) = cell_ref(r.payload) {
                    let s = if r.payload[7] == 1 {
                        error_literal(r.payload[6])
                    } else {
                        Some(if r.payload[6] != 0 { "TRUE" } else { "FALSE" })
                    };
                    if let Some(s) = s {
                        sheet.put(
                            row,
                            col,
                            RichText {
                                text: s.into(),
                                runs: vec![],
                            },
                        );
                    }
                }
            }
            0x0006 if r.payload.len() >= 14 => {
                if let Some((row, col, xf)) = cell_ref(r.payload) {
                    let cached = &r.payload[6..14];
                    if cached[6..] == [0xff, 0xff] {
                        match cached[0] {
                            0 => pending = Some((row, col, xf)),
                            1 => sheet.put(
                                row,
                                col,
                                RichText {
                                    text: if cached[2] != 0 { "TRUE" } else { "FALSE" }.into(),
                                    runs: vec![],
                                },
                            ),
                            2 => {
                                if let Some(s) = error_literal(cached[2]) {
                                    sheet.put(
                                        row,
                                        col,
                                        RichText {
                                            text: s.into(),
                                            runs: vec![],
                                        },
                                    );
                                }
                            }
                            _ => {}
                        }
                    } else {
                        sheet.put(row, col, g.numeric(f64(r.payload, 6).unwrap(), xf)?);
                    }
                }
            }
            0x0207 => {
                let (parts, next) = continues(data, r, budget)?;
                cursor = next;
                if let Some((row, col, xf)) = pending.take() {
                    let mut q = Segments::new(parts);
                    if let Some(mut s) = g.read(&mut q, false, false) {
                        let check = s.text.trim_start_matches(crate::docx::space).to_uppercase();
                        if check.starts_with("=DISPIMG(") || check.starts_with("=_XLFN.DISPIMG(") {
                            s = Decoded {
                                text: "#NAME?".into(),
                                font_starts: vec![],
                            };
                        }
                        sheet.put(row, col, g.rich(&s, xf));
                    }
                }
            }
            0x01b8 => super::hyperlink::apply(r.payload, &mut links),
            0x00ec | 0x01b6 => {
                sheet.resource_records.push(r.offset);
                let (_, next) = continues(data, r, budget)?;
                cursor = next;
            }
            0x005d => sheet.resource_records.push(r.offset),
            _ => {}
        }
    }
    for (p, target) in links {
        if let Some(cell) = sheet.cells.get_mut(&p) {
            cell.hyperlink = Some(target);
        }
    }
    Ok(Some(sheet))
}
impl Workbook {
    pub fn open(data: Arc<[u8]>) -> Result<Self> {
        let mut ole = OleReader::open(data)?;
        if ole.has_stream("EncryptionInfo") || ole.has_stream("EncryptedPackage") {
            return Err(Error::new(
                ErrorKind::Encrypted,
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
        Self::parse_stream(stream)
    }
    pub fn parse_stream(stream: Arc<[u8]>) -> Result<Self> {
        Self::parse_internal(stream, None)
    }
    pub(super) fn parse_bound(
        stream: Arc<[u8]>,
        equations: &BTreeMap<String, String>,
    ) -> Result<Self> {
        Self::parse_internal(stream, Some(equations))
    }
    fn parse_internal(
        stream: Arc<[u8]>,
        equations: Option<&BTreeMap<String, String>>,
    ) -> Result<Self> {
        if stream.is_empty() {
            return Err(Error::malformed("empty Workbook stream"));
        }
        let mut budget = Budget::default();
        let mut diagnostics = vec![];
        let g = globals(&stream, &mut budget, &mut diagnostics)?;
        let mut decoder = crate::equation::image_mtef::Decoder::default();
        let images = if equations.is_some() {
            let bytes = super::resources::drawing_bytes(&stream, &g.resource_records);
            crate::officeart::bstore(&bytes, None, &mut budget, &mut diagnostics)?
        } else {
            BTreeMap::new()
        };
        // iter_records' zero-tail test does not clone and need not scan the tail
        // on every record: one last-nonzero index gives exactly the same stop.
        let last = stream.iter().rposition(|b| *b != 0).map_or(0, |i| i + 1);
        let mut candidates = vec![];
        let mut cursor = 0;
        while cursor < last {
            let Some(r) = at(&stream, cursor) else { break };
            cursor = r.next;
            if r.kind == BOF && u16(r.payload, 2) == Some(0x0010) {
                candidates.push(r.offset)
            }
        }
        let mut entries = g
            .descriptors
            .iter()
            .enumerate()
            .filter(|(_, d)| d.sheet_type == 0)
            .map(|(i, d)| (i, d.clone()))
            .collect::<Vec<_>>();
        if entries.is_empty() && !candidates.is_empty() {
            entries = candidates
                .iter()
                .enumerate()
                .map(|(i, offset)| {
                    (
                        i,
                        Descriptor {
                            name: format!("Recovered Sheet {}", i + 1),
                            offset: *offset as u32,
                            visible: true,
                            sheet_type: 0,
                        },
                    )
                })
                .collect();
        }
        if entries.is_empty() && candidates.is_empty() {
            return Err(Error::malformed("workbook contains no worksheet substream"));
        }
        let mut sheets = vec![];
        let mut used = BTreeSet::new();
        for (order, d) in entries {
            let (original, recovered) = if is_worksheet(&stream, d.offset as usize)
                && !used.contains(&(d.offset as usize))
            {
                (Some(d.offset as usize), false)
            } else {
                let found = candidates.iter().find(|p| !used.contains(*p)).copied();
                (found, found.is_some())
            };
            let Some(offset) = original else {
                diagnostics.push(format!("XLS_SHEET_UNREADABLE: {}", d.name));
                sheets.push(Worksheet::empty(&d, order, true));
                continue;
            };
            if recovered {
                diagnostics.push(format!(
                    "XLS_BOUNDSHEET_RECOVERED: {} {} -> {offset}",
                    d.name, d.offset
                ));
            }
            used.insert(offset);
            let mut sheet = read_sheet(
                &stream,
                &g,
                &d,
                offset,
                order,
                recovered,
                &mut budget,
                &mut diagnostics,
            )?
            .unwrap_or_else(|| Worksheet::empty(&d, order, true));
            if let Some(equations) = equations {
                super::resources::bind(
                    &stream,
                    &mut sheet,
                    &g.fonts,
                    &g.extern_sheets,
                    &images,
                    equations,
                    &mut decoder,
                    &mut budget,
                    &mut diagnostics,
                )?;
            }
            sheets.push(sheet);
        }
        let mut chart_sheets = vec![];
        for (order, d) in g
            .descriptors
            .iter()
            .enumerate()
            .filter(|(_, d)| d.sheet_type == 2)
        {
            let mut selection = None;
            if charged(&stream, d.offset as usize, &mut budget)?
                .is_some_and(|r| r.kind == BOF && u16(r.payload, 2) == Some(0x0020))
            {
                let last = stream.iter().rposition(|b| *b != 0).map_or(0, |i| i + 1);
                let mut cursor = d.offset as usize;
                let mut records = vec![];
                while cursor < last {
                    let Some(r) = charged(&stream, cursor, &mut budget)? else {
                        break;
                    };
                    cursor = r.next;
                    records.push(r);
                    if r.kind == EOF {
                        break;
                    }
                }
                selection = super::chart::selection(records, order, &g.extern_sheets);
            }
            let name = selection
                .as_ref()
                .and_then(|s| g.descriptors.get(s.sheet_index))
                .filter(|d| d.sheet_type == 0)
                .map(|d| d.name.clone());
            let (rows, cols) = if name.is_some() {
                selection.map(|s| (s.rows, s.cols)).unwrap_or_default()
            } else {
                (vec![], vec![])
            };
            chart_sheets.push(ChartSheet {
                name: d.name.clone(),
                visible: d.visible,
                order,
                source_sheet_name: name,
                source_rows: rows,
                source_cols: cols,
            });
        }
        Ok(Self {
            stream,
            sheets,
            descriptors: g.descriptors,
            active_sheet_index: g.active_sheet_index,
            date1904: g.date1904,
            biff8: g.biff8,
            codepage: g.codepage,
            fonts: g.fonts,
            extern_sheets: g.extern_sheets,
            global_resource_records: g.resource_records,
            diagnostics,
            record_visits: budget.count,
            chart_sheets,
            resources_bound: equations.is_some(),
        })
    }
}
