//! BIFF semantic cells to the shared table projector (no openpyxl adapter).
use super::{
    strings::{Font, RichText},
    Workbook, Worksheet,
};
use crate::{
    spreadsheet::{self, Cell, Merge, Sheet, SourceCell},
    Error, ErrorKind, Result,
};
use std::collections::BTreeSet;
fn escape_lines(s: &str) -> String {
    crate::document_output::html::escape(s)
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\n', "<br>")
}
fn styled(s: &str, font: Font) -> String {
    let mut s = escape_lines(s);
    if s.is_empty() {
        return s;
    }
    if font.underline || font.strike {
        s = s.replace(' ', "&nbsp;")
    }
    for (on, tag) in [
        (font.superscript, "sup"),
        (font.subscript && !font.superscript, "sub"),
        (font.strike, "s"),
        (font.underline, "u"),
        (font.italic, "em"),
        (font.bold, "strong"),
    ] {
        if on {
            s = format!("<{tag}>{s}</{tag}>")
        }
    }
    s
}
fn text(value: &RichText) -> (String, bool) {
    if value.runs.is_empty() {
        return (value.text.chars().take(32767).collect(), false);
    }
    let chars = value.text.chars().collect::<Vec<_>>();
    let length = chars.len();
    let mut bounds = BTreeSet::from([0, length]);
    for r in &value.runs {
        bounds.insert(r.start.min(length));
        bounds.insert(r.end.min(length));
    }
    let bounds = bounds.into_iter().collect::<Vec<_>>();
    let mut out = String::new();
    for b in bounds.windows(2) {
        if b[0] >= b[1] {
            continue;
        }
        let part = chars[b[0]..b[1]].iter().collect::<String>();
        if let Some(run) = value.runs.iter().find(|r| r.start <= b[0] && b[0] < r.end) {
            out.push_str(&styled(&part, run.style));
        } else {
            out.push_str(&escape_lines(&part));
        }
    }
    (out, true)
}
impl Worksheet {
    pub fn project_cells(&self) -> Result<Sheet> {
        let mut cells = vec![];
        for source in self.cells.values() {
            let mut value = source.value.clone();
            if value.runs.is_empty() {
                value.text = value.text.chars().take(32767).collect();
            }
            // Merges are applied after values by the original adapter.
            if self.merges.iter().any(|&(a, b, c, d)| {
                source.row >= a
                    && source.row <= c
                    && source.col >= b
                    && source.col <= d
                    && (source.row, source.col) != (a, b)
            }) {
                continue;
            }
            let semantic = !value.text.trim_matches(crate::docx::space).is_empty()
                && !value.text.contains("DISPIMG");
            let (mut text, mut html) = text(&value);
            if value.text.contains("DISPIMG") {
                text.clear();
                html = false;
            } else if let Some(link) = source
                .hyperlink
                .as_deref()
                .and_then(|s| crate::links::sanitize(Some(s)))
            {
                if !text.is_empty() {
                    if !html {
                        text = escape_lines(&text);
                    }
                    text = format!(
                        "<a href=\"{}\">{text}</a>",
                        crate::document_output::html::escape(link)
                    );
                    html = true;
                }
            }
            cells.push(SourceCell {
                row: source.row,
                col: source.col,
                has_value: true,
                semantic,
                cell: Cell {
                    text,
                    text_is_html: html,
                    row_span: 1,
                    col_span: 1,
                    source_row: Some(source.row),
                    source_col: Some(source.col),
                    ..Default::default()
                },
            });
        }
        let merges = self
            .merges
            .iter()
            .map(|&(a, b, c, d)| Merge {
                min_row: a,
                min_col: b,
                max_row: c,
                max_col: d,
            })
            .collect::<Vec<_>>();
        for m in &merges {
            if !cells
                .iter()
                .any(|c| (c.row, c.col) == (m.min_row, m.min_col))
            {
                cells.push(SourceCell {
                    row: m.min_row,
                    col: m.min_col,
                    has_value: false,
                    semantic: false,
                    cell: Cell {
                        row_span: 1,
                        col_span: 1,
                        source_row: Some(m.min_row),
                        source_col: Some(m.min_col),
                        ..Default::default()
                    },
                });
            }
        }
        Sheet::new(cells, merges)
    }
    fn project_resources(&self, resources: &Worksheet) -> Result<Sheet> {
        let mut sheet = self.project_cells()?;
        fn cell(sheet: &mut Sheet, row: u32, col: u32) -> &mut SourceCell {
            sheet.cells.entry((row, col)).or_insert_with(|| SourceCell {
                row,
                col,
                has_value: false,
                semantic: false,
                cell: Cell {
                    row_span: 1,
                    col_span: 1,
                    source_row: Some(row),
                    source_col: Some(col),
                    ..Default::default()
                },
            })
        }
        for image in &resources.images {
            let c = cell(&mut sheet, image.row, image.col);
            c.cell
                .media
                .push(format!("<img src=\"{}\" />", image.image_base64));
            c.semantic = true;
        }
        for equation in &resources.equations {
            let c = cell(&mut sheet, equation.row, equation.col);
            c.cell.equations.push(equation.latex.clone());
            c.semantic = true;
        }
        Ok(sheet)
    }
}
impl Workbook {
    /// Diagnostic stage entry. Never silently publishes cells without drawings,
    /// chart sheets, embedded objects or text boxes that have not been bound.
    pub fn cell_pages(&self) -> Result<Vec<Vec<serde_json::Value>>> {
        if !self.resources_bound
            && (!self.global_resource_records.is_empty()
                || self
                    .sheets
                    .iter()
                    .any(|s| !s.resource_records.is_empty() || !s.chart_records.is_empty()))
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "XLS resource binding pending; cell-only output is incomplete",
            ));
        }
        let mut merge_slots = 0u64;
        for sheet in &self.sheets {
            for &(a, b, c, d) in &sheet.merges {
                merge_slots += u64::from(c - a + 1) * u64::from(d - b + 1);
                if merge_slots > 4_000_000 {
                    return Err(Error::new(
                        ErrorKind::ResourceLimit,
                        "workbook extent exceeds original max_grid_slots",
                    ));
                }
            }
        }
        let mut names = vec![];
        let mut pages = vec![];
        let mut grid = 0u64;
        for source in &self.sheets {
            let name = crate::xlsx::sheet_title(&source.name, &names)?;
            names.push(name.clone());
            if !source.visible {
                continue;
            }
            let sheet = source.project_resources(source)?;
            let (_, _, tables) = sheet.project(None);
            grid += tables
                .iter()
                .map(|t| u64::from(t.num_rows) * u64::from(t.num_cols))
                .sum::<u64>();
            if grid > 4_000_000 {
                return Err(Error::new(
                    ErrorKind::ResourceLimit,
                    "workbook extent exceeds original max_grid_slots",
                ));
            }
            let mut used = BTreeSet::new();
            let mut artifacts = vec![];
            for (index, t) in tables.iter().enumerate() {
                used.extend(t.data.iter().filter_map(|c| c.source_row.zip(c.source_col)));
                artifacts.push((
                    (t.anchor.1, t.anchor.0),
                    index,
                    spreadsheet::table::block(t, true),
                ));
            }
            for (index, chart) in source.charts.iter().enumerate() {
                let mut block = None;
                if !chart.source_rows.is_empty() && !chart.source_cols.is_empty() {
                    grid += chart.source_rows.len() as u64 * chart.source_cols.len() as u64;
                    if grid > 4_000_000 {
                        return Err(Error::new(
                            ErrorKind::ResourceLimit,
                            "workbook extent exceeds original max_grid_slots",
                        ));
                    }
                    let table = sheet.synthetic(&chart.source_rows, &chart.source_cols)?;
                    block = Some(
                        serde_json::json!({"type":"chart","content":spreadsheet::table::html(&table)}),
                    );
                } else if chart.image_base64.is_some() {
                    block = Some(serde_json::json!({"type":"chart","content":""}));
                }
                if let Some(mut block) = block {
                    if let Some(image) = &chart.image_base64 {
                        block["image_base64"] = serde_json::json!(image);
                    }
                    artifacts.push(((chart.row, chart.col), 10_000 + index, block));
                }
            }
            for (index, equation) in source.equations.iter().enumerate() {
                if !used.contains(&(equation.row, equation.col)) {
                    artifacts.push((
                        (equation.row, equation.col),
                        20_000 + index,
                        serde_json::json!({"type":"equation","content":equation.latex}),
                    ));
                }
            }
            artifacts.sort_by_key(|(p, i, _)| (*p, *i));
            let mut page = artifacts.into_iter().map(|(_, _, v)| v).collect::<Vec<_>>();
            for image in &source.images {
                if !used.contains(&(image.row, image.col)) {
                    page.push(
                        serde_json::json!({"type":"image","image_base64":image.image_base64}),
                    );
                }
            }
            pages.push((name, page));
        }
        // Preserve the original builder's lookup by source name after the
        // adapter has de-duplicated titles, including repeated-name aliases.
        let mut ordered = self
            .sheets
            .iter()
            .filter(|s| s.visible)
            .map(|s| {
                let index = pages.iter().position(|(name, _)| *name == s.name);
                (s.order, s.name.clone(), index)
            })
            .collect::<Vec<_>>();
        for chart in self.chart_sheets.iter().filter(|c| c.visible) {
            let mut page = vec![];
            if let Some(source_name) = &chart.source_sheet_name {
                if !chart.source_rows.is_empty() && !chart.source_cols.is_empty() {
                    grid += chart.source_rows.len() as u64 * chart.source_cols.len() as u64;
                    if grid > 4_000_000 {
                        return Err(Error::new(
                            ErrorKind::ResourceLimit,
                            "workbook extent exceeds original max_grid_slots",
                        ));
                    }
                    if let Some(index) = names.iter().position(|n| n == source_name) {
                        // The original builder retains the last converted
                        // worksheet's media/math maps while rendering chart
                        // sheets. Preserve this context, including hidden data.
                        let context = self
                            .sheets
                            .iter()
                            .rev()
                            .find(|s| s.visible)
                            .unwrap_or(&self.sheets[index]);
                        let source = self.sheets[index].project_resources(context)?;
                        let table = source.synthetic(&chart.source_rows, &chart.source_cols)?;
                        page.push(serde_json::json!({"type":"chart","content":spreadsheet::table::html(&table)}));
                    }
                }
            }
            let index = pages.len();
            pages.push((chart.name.clone(), page));
            ordered.push((chart.order, chart.name.clone(), Some(index)));
        }
        // Stable source order puts worksheets before charts at an equal order.
        ordered.sort_by_key(|(order, _, _)| *order);
        if ordered
            .iter()
            .filter(|(_, _, i)| i.is_some_and(|i| !pages[i].1.is_empty()))
            .count()
            > 1
        {
            for (_, name, index) in &ordered {
                if let Some(i) = index {
                    let page = &mut pages[*i].1;
                    if !page.is_empty() {
                        page.insert(0,serde_json::json!({"type":"paragraph_title","level":2,"content":[{"type":"text","content":name}]}));
                    }
                }
            }
        }
        let mut counts = vec![0; pages.len()];
        for (_, _, i) in &ordered {
            if let Some(i) = i {
                counts[*i] += 1;
            }
        }
        Ok(ordered
            .into_iter()
            .map(|(_, _, i)| match i {
                None => vec![],
                Some(i) => {
                    counts[i] -= 1;
                    if counts[i] == 0 {
                        std::mem::take(&mut pages[i].1)
                    } else {
                        pages[i].1.clone()
                    }
                }
            })
            .collect())
    }
    pub fn into_pages(self) -> Result<Vec<Vec<serde_json::Value>>> {
        self.cell_pages()
    }
}
