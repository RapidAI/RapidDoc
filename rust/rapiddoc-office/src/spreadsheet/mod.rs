//! Format-neutral XLS/XLSX table projection. Package readers supply typed cells;
//! discovery and projection never call Python or build an openpyxl object model.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
pub mod regions;
pub mod table;
pub type Position = (u32, u32);
/// Consume spreadsheet page values into the shared Office output types. Strings
/// move out of the native page boundary; no JSON encode/decode or Python bridge.
pub fn into_body_pages(
    pages: Vec<Vec<serde_json::Value>>,
) -> Result<Vec<Vec<crate::docx::body::Block>>> {
    use crate::docx::{
        body::{Block, Content, Kind},
        spans::Span,
    };
    fn block(v: serde_json::Value) -> Result<Block> {
        let serde_json::Value::Object(mut v) = v else {
            return Err(Error::malformed("invalid spreadsheet output block"));
        };
        let kind = match v
            .remove("type")
            .and_then(|s| s.as_str().map(str::to_owned))
            .as_deref()
        {
            Some("text") => Kind::Text,
            Some("paragraph_title") => Kind::ParagraphTitle,
            Some("table") => Kind::Table,
            Some("chart") => Kind::Chart,
            Some("image") => Kind::Image,
            Some("equation") => Kind::Equation,
            _ => return Err(Error::malformed("unknown spreadsheet output block")),
        };
        let content = match v.remove("content") {
            None => Content::None,
            Some(serde_json::Value::String(s)) => Content::Text(s),
            Some(serde_json::Value::Array(spans)) => Content::Spans(
                spans
                    .into_iter()
                    .map(|s| {
                        let serde_json::Value::Object(mut s) = s else {
                            return Err(Error::malformed("invalid spreadsheet text span"));
                        };
                        if s.get("type").and_then(|s| s.as_str()) != Some("text") {
                            return Err(Error::malformed("unsupported spreadsheet text span"));
                        }
                        let Some(serde_json::Value::String(content)) = s.remove("content") else {
                            return Err(Error::malformed("missing spreadsheet span text"));
                        };
                        Ok(Span::Text {
                            content,
                            styles: vec![],
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            _ => return Err(Error::malformed("invalid spreadsheet content")),
        };
        let mut b = Block::new(kind, content);
        if let Some(serde_json::Value::String(s)) = v.remove("image_base64") {
            b.image_base64 = Some(s.into());
        }
        b.level = v.remove("level").and_then(|v| v.as_i64());
        Ok(b)
    }
    pages
        .into_iter()
        .map(|p| p.into_iter().map(block).collect())
        .collect()
}
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Merge {
    pub min_row: u32,
    pub max_row: u32,
    pub min_col: u32,
    pub max_col: u32,
}
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct Cell {
    pub row: u32,
    pub col: u32,
    pub text: String,
    pub row_span: u32,
    pub col_span: u32,
    pub styles: BTreeMap<String, String>,
    pub media: Vec<String>,
    pub equations: Vec<String>,
    pub text_is_html: bool,
    pub source_row: Option<u32>,
    pub source_col: Option<u32>,
}
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Table {
    pub anchor: (u32, u32),
    pub num_rows: u32,
    pub num_cols: u32,
    pub data: Vec<Cell>,
}
#[derive(Debug, Clone, Deserialize)]
pub struct SourceCell {
    pub row: u32,
    pub col: u32,
    pub has_value: bool,
    pub semantic: bool,
    pub cell: Cell,
}
#[derive(Debug)]
pub struct Sheet {
    pub cells: BTreeMap<Position, SourceCell>,
    pub merges: Vec<Merge>,
    pub default_cell: Cell,
    merged_rows: BTreeMap<u32, Vec<(u32, u32)>>,
    hidden_rows: BTreeMap<u32, Vec<(u32, u32)>>,
    anchor_spans: BTreeMap<Position, (u32, u32)>,
}
impl Sheet {
    pub fn new(cells: Vec<SourceCell>, merges: Vec<Merge>) -> Result<Self> {
        if merges
            .iter()
            .any(|m| m.min_row > m.max_row || m.min_col > m.max_col)
        {
            return Err(Error::malformed("invalid spreadsheet merge bounds"));
        }
        let mut merged_rows = BTreeMap::<u32, Vec<(u32, u32)>>::new();
        let mut hidden_rows = BTreeMap::<u32, Vec<(u32, u32)>>::new();
        let mut anchor_spans = BTreeMap::new();
        for m in &merges {
            let rs = (m.max_row - m.min_row)
                .checked_add(1)
                .ok_or_else(|| Error::malformed("spreadsheet row span overflow"))?;
            let cs = (m.max_col - m.min_col)
                .checked_add(1)
                .ok_or_else(|| Error::malformed("spreadsheet column span overflow"))?;
            anchor_spans.insert((m.min_row, m.min_col), (rs, cs));
            for row in m.min_row..=m.max_row {
                merged_rows
                    .entry(row)
                    .or_default()
                    .push((m.min_col, m.max_col));
                let start = if row == m.min_row {
                    m.min_col.checked_add(1)
                } else {
                    Some(m.min_col)
                };
                if let Some(start) = start.filter(|s| *s <= m.max_col) {
                    hidden_rows.entry(row).or_default().push((start, m.max_col));
                }
            }
        }
        for ranges in merged_rows.values_mut().chain(hidden_rows.values_mut()) {
            ranges.sort();
        }
        Ok(Self {
            cells: cells.into_iter().map(|c| ((c.row, c.col), c)).collect(),
            merges,
            default_cell: Cell::default(),
            merged_rows,
            hidden_rows,
            anchor_spans,
        })
    }
    pub fn span(&self, row: u32, col: u32) -> (u32, u32) {
        self.anchor_spans
            .get(&(row, col))
            .copied()
            .unwrap_or((1, 1))
    }
    pub fn has_content(&self, row: u32, col: u32) -> bool {
        self.cells.get(&(row, col)).is_some_and(|c| c.has_value)
            || Self::contains(&self.merged_rows, row, col)
    }
    fn contains(index: &BTreeMap<u32, Vec<(u32, u32)>>, row: u32, col: u32) -> bool {
        if let Some(ranges) = index.get(&row) {
            for (start, end) in ranges {
                if *start > col {
                    break;
                }
                if col <= *end {
                    return true;
                }
            }
        }
        false
    }
    pub fn semantic(&self, row: u32, col: u32) -> bool {
        self.cells.get(&(row, col)).is_some_and(|c| c.semantic)
    }
    pub fn bounds(&self) -> [u32; 4] {
        let mut minr = None;
        let mut minc = None;
        let (mut maxr, mut maxc) = (0, 0);
        let mut include = |r0: u32, r1: u32, c0: u32, c1: u32| {
            minr = Some(minr.map_or(r0, |v: u32| v.min(r0)));
            minc = Some(minc.map_or(c0, |v: u32| v.min(c0)));
            maxr = maxr.max(r1);
            maxc = maxc.max(c1);
        };
        for (&(r, c), v) in &self.cells {
            if v.has_value {
                include(r, r, c, c);
            }
        }
        for m in &self.merges {
            include(m.min_row, m.max_row, m.min_col, m.max_col);
        }
        if let (Some(r), Some(c)) = (minr, minc) {
            [r, maxr, c, maxc]
        } else {
            [0, 0, 0, 0]
        }
    }
    pub fn positions(&self) -> Vec<Position> {
        self.cells
            .iter()
            .filter(|(_, c)| c.has_value)
            .map(|(p, _)| *p)
            .collect()
    }
    fn build_cell(&self, r: u32, c: u32, display_r: u32, display_c: u32, rs: u32, cs: u32) -> Cell {
        let mut cell = self
            .cells
            .get(&(r, c))
            .map(|c| c.cell.clone())
            .unwrap_or_else(|| self.default_cell.clone());
        cell.row = display_r;
        cell.col = display_c;
        cell.row_span = rs;
        cell.col_span = cs;
        cell.source_row = Some(r);
        cell.source_col = Some(c);
        cell
    }
    pub fn materialize(&self, region: &regions::Region) -> Table {
        let mut data = vec![];
        for r in region.row_start..=region.row_end {
            for c in region.col_start..=region.col_end {
                if Self::contains(&self.hidden_rows, r, c) {
                    continue;
                }
                let (rs, cs) = self.span(r, c);
                data.push(self.build_cell(
                    r,
                    c,
                    r - region.row_start,
                    c - region.col_start,
                    rs,
                    cs,
                ));
            }
        }
        Table {
            anchor: (region.col_start, region.row_start),
            num_rows: region.rows(),
            num_cols: region.cols(),
            data,
        }
    }
    pub fn synthetic(&self, rows: &[u32], cols: &[u32]) -> Result<Table> {
        if rows.is_empty() || cols.is_empty() {
            return Err(Error::malformed("empty spreadsheet row/column selection"));
        }
        let selected = rows
            .iter()
            .flat_map(|r| cols.iter().map(move |c| (*r, *c)))
            .collect::<BTreeSet<_>>();
        let mut hidden = BTreeSet::new();
        let mut spans = BTreeMap::new();
        for m in &self.merges {
            let anchor = (m.min_row, m.min_col);
            if !selected.contains(&anchor) {
                continue;
            }
            let rr = rows
                .iter()
                .filter(|r| **r >= m.min_row && **r <= m.max_row)
                .copied()
                .collect::<Vec<_>>();
            let cc = cols
                .iter()
                .filter(|c| **c >= m.min_col && **c <= m.max_col)
                .copied()
                .collect::<Vec<_>>();
            if rr.is_empty() || cc.is_empty() {
                continue;
            }
            spans.insert(anchor, (rr.len() as u32, cc.len() as u32));
            for r in rr {
                for c in &cc {
                    if (r, *c) != anchor {
                        hidden.insert((r, *c));
                    }
                }
            }
        }
        let mut data = vec![];
        for (dr, r) in rows.iter().enumerate() {
            for (dc, c) in cols.iter().enumerate() {
                if hidden.contains(&(*r, *c)) {
                    continue;
                }
                let (rs, cs) = spans.get(&(*r, *c)).copied().unwrap_or((1, 1));
                data.push(self.build_cell(*r, *c, dr as u32, dc as u32, rs, cs));
            }
        }
        Ok(Table {
            anchor: (cols[0], rows[0]),
            num_rows: rows.len() as u32,
            num_cols: cols.len() as u32,
            data,
        })
    }
    pub fn project(&self, gap: Option<u32>) -> (u32, f64, Vec<Table>) {
        let (gap, penalty, regions) = if let Some(gap) = gap {
            (gap, 0., regions::discover(self, gap))
        } else {
            regions::select(self)
        };
        let tables = regions
            .iter()
            .map(|r| self.materialize(r))
            .collect::<Vec<_>>();
        let semantic = tables
            .iter()
            .map(|t| {
                t.data
                    .iter()
                    .filter(|c| {
                        !c.text.trim_matches(crate::docx::space).is_empty()
                            || c.media
                                .iter()
                                .any(|s| !s.trim_matches(crate::docx::space).is_empty())
                            || !c.equations.is_empty()
                    })
                    .map(|c| (c.source_row.unwrap(), c.source_col.unwrap()))
                    .collect::<BTreeSet<_>>()
            })
            .collect::<Vec<_>>();
        let tables = tables
            .into_iter()
            .enumerate()
            .filter(|(i, _)| {
                !semantic.iter().enumerate().any(|(j, s)| {
                    *i != j && semantic[*i].len() < s.len() && semantic[*i].is_subset(s)
                })
            })
            .map(|(_, t)| t)
            .collect();
        (gap, penalty, tables)
    }
}
