//! DOC table assembly using the original row boundaries and merge rules.
use super::paragraph::{Paragraph, Table};
use crate::docx::{
    body::{Block, Content, Kind},
    spans::Span,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
pub struct Item {
    pub visuals: Vec<Block>,
    pub list_id: Option<u32>,
    pub cell_list: Option<(bool, u8, String)>,
    pub cp: u32,
    pub pap: Paragraph,
    pub cell: bool,
    pub row: bool,
    pub visible: bool,
    pub html: String,
    pub block: Block,
}
struct CellPart {
    html: String,
    cell_list: Option<(bool, u8, String)>,
}
struct RawRow {
    cells: Vec<Vec<CellPart>>,
    format: Option<Table>,
}
struct Cell {
    parts: Vec<CellPart>,
    rows: usize,
    cols: usize,
}
fn formats(row: &RawRow) -> (Vec<i16>, Vec<super::paragraph::Cell>) {
    let n = row.cells.len();
    if let Some(t) = row.format.as_ref().filter(|t| t.boundaries.len() > n) {
        let boundaries = t.boundaries[..=n].to_vec();
        let mut cells = t.cells.iter().take(n).cloned().collect::<Vec<_>>();
        while cells.len() < n {
            cells.push(super::paragraph::Cell {
                right: boundaries[cells.len() + 1],
                ..Default::default()
            })
        }
        return (boundaries, cells);
    }
    (
        (0..=n).map(|n| n as i16).collect(),
        (0..n)
            .map(|n| super::paragraph::Cell {
                right: n as i16 + 1,
                ..Default::default()
            })
            .collect(),
    )
}
fn materialize(rows: Vec<RawRow>) -> String {
    let fs: Vec<_> = rows.iter().map(formats).collect();
    let edges: BTreeSet<_> = fs.iter().flat_map(|(b, _)| b.iter().copied()).collect();
    let edges: BTreeMap<_, _> = edges.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
    let mut result: Vec<(bool, Vec<Cell>)> = vec![];
    let mut origins = BTreeMap::<(usize, usize), (usize, usize)>::new();
    for (r, (boundaries, fs)) in rows.into_iter().zip(fs) {
        let ri = result.len();
        let header = r.format.as_ref().is_some_and(|t| t.header);
        result.push((header, vec![]));
        let mut keys: Vec<(usize, usize)> = vec![];
        for (i, parts) in r.cells.into_iter().enumerate() {
            let f = &fs[i];
            let left = edges[&boundaries[i]];
            let right = edges[&boundaries[i + 1]];
            let cols = right.saturating_sub(left).max(1);
            if f.horizontal_continue && !result[ri].1.is_empty() {
                let c = result[ri].1.last_mut().unwrap();
                c.cols += cols;
                c.parts.extend(parts);
                keys.last_mut().unwrap().1 = right;
                continue;
            }
            let key = (left, right);
            if let Some(&(r, c)) = origins.get(&key).filter(|_| f.vertical_continue) {
                result[r].1[c].rows += 1;
            } else {
                let ci = result[ri].1.len();
                result[ri].1.push(Cell {
                    parts,
                    cols,
                    rows: 1,
                });
                keys.push(key);
                if f.vertical_first {
                    origins.insert(key, (ri, ci));
                }
            }
        }
    }
    let mut html = String::from("<table>");
    for (header, cells) in result {
        html.push_str("<tr>");
        let tag = if header { "th" } else { "td" };
        for c in cells {
            html.push_str(&format!("<{tag}"));
            if c.rows > 1 {
                html.push_str(&format!(" rowspan=\"{}\"", c.rows))
            }
            if c.cols > 1 {
                html.push_str(&format!(" colspan=\"{}\"", c.cols))
            }
            html.push('>');
            let mut lists = vec![];
            for part in &c.parts {
                append_cell(&mut html, &mut lists, part);
            }
            close_lists(&mut html, &mut lists);
            html.push_str(&format!("</{tag}>"));
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}
fn depth(p: &Item) -> i64 {
    p.pap.table_depth.unwrap_or(1).max(0)
}
fn in_table(p: &Item) -> bool {
    p.pap.in_table.unwrap_or(false) || p.cell || p.row
}
fn close_lists(out: &mut String, stack: &mut Vec<&'static str>) {
    while let Some(tag) = stack.pop() {
        out.push_str(&format!("</{tag}>"));
    }
}
fn append_cell(out: &mut String, stack: &mut Vec<&'static str>, p: &CellPart) {
    if let Some((ordered, level, html)) = &p.cell_list {
        let tag = if *ordered { "ol" } else { "ul" };
        let level = usize::from((*level).min(8));
        while stack.len() > level + 1 {
            let tag = stack.pop().unwrap();
            out.push_str(&format!("</{tag}>"));
        }
        while stack.len() < level + 1 {
            out.push_str(&format!("<{tag}>"));
            stack.push(tag);
        }
        if stack.last() != Some(&tag) {
            let old = stack.pop().unwrap();
            out.push_str(&format!("</{old}><{tag}>"));
            stack.push(tag);
        }
        out.push_str(html);
    } else {
        close_lists(out, stack);
        out.push_str(&p.html);
    }
}
fn parse(queue: &mut VecDeque<Item>, level: i64) -> (u32, String) {
    let cp = queue.front().unwrap().cp;
    let mut rows = vec![];
    let mut cells = vec![];
    let mut current = vec![];
    while let Some(p) = queue.front() {
        if !in_table(p) || depth(p) < level {
            break;
        }
        if depth(p) > level {
            let (_, nested) = parse(queue, depth(p));
            current.push(CellPart {
                html: nested,
                cell_list: None,
            });
            continue;
        }
        let p = queue.pop_front().unwrap();
        if p.row {
            if p.visible {
                current.push(CellPart {
                    html: p.html,
                    cell_list: p.cell_list,
                })
            }
            if !current.is_empty() {
                cells.push(std::mem::take(&mut current));
            }
            if !cells.is_empty() {
                rows.push(RawRow {
                    cells: std::mem::take(&mut cells),
                    format: p.pap.table,
                });
            }
        } else if p.cell {
            if p.visible {
                current.push(CellPart {
                    html: p.html,
                    cell_list: p.cell_list,
                })
            }
            cells.push(std::mem::take(&mut current));
        } else {
            current.push(CellPart {
                html: p.html,
                cell_list: p.cell_list,
            })
        }
    }
    if !current.is_empty() {
        cells.push(current)
    }
    if !cells.is_empty() {
        rows.push(RawRow {
            cells,
            format: None,
        })
    }
    (cp, materialize(rows))
}
pub fn assemble(items: Vec<Item>) -> Vec<(u32, Block, Option<u32>)> {
    let mut q: VecDeque<_> = items.into();
    let mut out = vec![];
    while let Some(p) = q.front() {
        if in_table(p) && depth(p) > 0 {
            let (cp, html) = parse(&mut q, 1);
            if html != "<table></table>" {
                out.push((cp, Block::new(Kind::Table, Content::Text(html)), None))
            }
        } else {
            let p = q.pop_front().unwrap();
            if p.visible {
                if !matches!(&p.block.content, Content::None)
                    && !matches!(&p.block.content,Content::Spans(s) if s.is_empty())
                {
                    out.push((p.cp, p.block, p.list_id));
                }
                for b in p.visuals {
                    out.push((p.cp, b, p.list_id));
                }
            }
        }
    }
    out
}
pub fn escape(s: &str, quote: bool) -> String {
    let out = s
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    if quote {
        out.replace('"', "&quot;").replace('\'', "&#x27;")
    } else {
        out
    }
}
fn styled(content: &str, styles: &[&str]) -> String {
    let mut rendered = escape(content, false)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    if styles.contains(&"superscript") {
        rendered = format!("<sup>{rendered}</sup>");
    } else if styles.contains(&"subscript") {
        rendered = format!("<sub>{rendered}</sub>");
    }
    for (name, tag) in [
        ("underline", "u"),
        ("bold", "strong"),
        ("italic", "em"),
        ("strikethrough", "s"),
    ] {
        if styles.contains(&name) {
            rendered = format!("<{tag}>{rendered}</{tag}>");
        }
    }
    rendered
}
pub fn inline(spans: &[Span]) -> String {
    fn render(spans: &[Span], url: Option<&str>) -> String {
        let mut out = String::new();
        for s in spans {
            match s {
                Span::Text { content, styles } => {
                    if content.is_empty() {
                        continue;
                    }
                    let mut text = styled(content, styles);
                    if let Some(url) = url.and_then(|u| crate::links::sanitize(Some(u))) {
                        text = format!("<a href=\"{}\">{text}</a>", escape(url, true));
                    }
                    out.push_str(&text);
                }
                Span::Link { url, content } => out.push_str(&render(content, Some(url))),
                Span::Equation { content } => {
                    let latex = content.replace('<', "\\lt ").replace('>', "\\gt ");
                    out.push_str(&format!("<eq>{}</eq>", escape(&latex, false)));
                }
            }
        }
        out
    }
    render(spans, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn floating_visual_placeholder_does_not_consume_a_public_block_index() {
        let item = Item {
            cp: 3,
            pap: Paragraph::default(),
            cell: false,
            row: false,
            visible: true,
            html: String::new(),
            block: Block::new(Kind::Text, Content::None),
            visuals: vec![Block::new(Kind::Image, Content::None)],
            list_id: None,
            cell_list: None,
        };
        let result = assemble(vec![item]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].1.kind, Kind::Image));
    }
    #[test]
    fn merged_cells_join_paragraphs_before_rendering_list_containers() {
        let make = |i, word: &str| Item {
            cp: i,
            pap: Paragraph {
                in_table: Some(true),
                table_depth: Some(1),
                ..Default::default()
            },
            cell: true,
            row: false,
            visible: true,
            html: format!("<p>{word}</p>"),
            block: Block::new(Kind::Text, Content::None),
            visuals: vec![],
            list_id: None,
            cell_list: Some((true, 0, format!("<li>{}. {word}</li>", i + 1))),
        };
        let row = Item {
            cp: 2,
            pap: Paragraph {
                in_table: Some(true),
                table_depth: Some(1),
                table: Some(Table {
                    boundaries: vec![0, 100, 200],
                    cells: vec![
                        super::super::paragraph::Cell {
                            horizontal_first: true,
                            ..Default::default()
                        },
                        super::super::paragraph::Cell {
                            horizontal_continue: true,
                            ..Default::default()
                        },
                    ],
                    header: false,
                }),
                ..Default::default()
            },
            cell: false,
            row: true,
            visible: false,
            html: String::new(),
            block: Block::new(Kind::Text, Content::None),
            visuals: vec![],
            list_id: None,
            cell_list: None,
        };
        let blocks = assemble(vec![make(0, "First"), make(1, "Second"), row]);
        let expected: serde_json::Value =
            serde_json::from_slice(
                &std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
                    "../../tests/fixtures/office_rust_reference_068/doc_merged_cell_list.json",
                ))
                .unwrap(),
            )
            .unwrap();
        let Content::Text(html) = &blocks[0].1.content else {
            panic!()
        };
        assert_eq!(html, expected.as_str().unwrap());
    }
}
