//! Native PowerPoint 97-2003 conversion.  The binary parser owns the record
//! lifetime; this layer binds slide-local stories/resources and projects their
//! ordered paragraphs, tables, titles, lists and notes into the shared model.
use super::{document, notes, records, resources, shapes, tables, text};
use crate::{
    docx::{
        body::{Block, Content, Kind},
        spans::Span,
    },
    ole::OleReader,
    Error, ErrorKind, Result,
};

#[derive(serde::Serialize)]
pub struct Body {
    pub pages: Vec<Vec<Block>>,
    pub diagnostics: Vec<String>,
}

fn elements(paragraph: &text::Paragraph) -> Vec<crate::docx::Element> {
    use crate::docx::{Element, Formatting, Script};
    let mut values: Vec<_> = paragraph
        .runs
        .iter()
        .filter(|r| !r.text.is_empty())
        .map(|r| Element {
            text: r.text.replace('\n', " "),
            hyperlink: r.hyperlink.clone(),
            format: Some(Formatting {
                bold: r.bold,
                italic: r.italic,
                underline: r.underline,
                strikethrough: r.strike,
                script: match r.baseline.unwrap_or(0).cmp(&0) {
                    std::cmp::Ordering::Greater => Script::Super,
                    std::cmp::Ordering::Less => Script::Sub,
                    _ => Script::Baseline,
                },
                ..Default::default()
            }),
        })
        .collect();
    for value in &mut values {
        if value.format.as_ref().is_some_and(|f| f.visible()) {
            break;
        }
        value.text = value.text.trim_start_matches(crate::docx::space).into();
        if !value.text.is_empty() {
            break;
        }
    }
    for value in values.iter_mut().rev() {
        if value.format.as_ref().is_some_and(|f| f.visible()) {
            break;
        }
        value.text = value.text.trim_end_matches(crate::docx::space).into();
        if !value.text.is_empty() {
            break;
        }
    }
    values
}
fn spans(paragraph: &text::Paragraph) -> Vec<Span> {
    crate::docx::spans::from_elements(&elements(paragraph))
}
fn cell_html(paragraph: &text::Paragraph) -> String {
    elements(paragraph)
        .into_iter()
        .map(|element| crate::doc::tables::inline(&crate::docx::spans::from_elements(&[element])))
        .collect()
}
fn table_html(table: &tables::Table) -> String {
    let mut out = String::from("<table border=\"1\">");
    for row in 0..table.rows {
        out.push_str("<tr>");
        for cell in table.cells.iter().filter(|c| c.row == row) {
            let text = cell
                .paragraphs
                .iter()
                .map(|p| cell_html(p))
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("<br/>");
            out.push_str("<td");
            if cell.row_span > 1 {
                out.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
            }
            if cell.col_span > 1 {
                out.push_str(&format!(" colspan=\"{}\"", cell.col_span));
            }
            out.push('>');
            out.push_str(&text);
            out.push_str("</td>");
        }
        out.push_str("</tr>");
    }
    out.push_str("</table>");
    out
}

fn text_blocks(content: &text::Content, title: bool) -> Vec<Block> {
    let mut out = vec![];
    let mut lists: Vec<Vec<usize>> = vec![];
    let mut title_used = false;
    for p in &content.paragraphs {
        let value = spans(p);
        if value.is_empty() {
            continue;
        }
        let mut b = Block::new(Kind::Text, Content::Spans(value));
        if title && !title_used && p.list_kind.is_none() {
            b.kind = Kind::ParagraphTitle;
            b.level = Some(2);
            title_used = true;
        } else if let Some(kind) = &p.list_kind {
            let depth = p.depth as usize;
            let attribute = Some(if kind == "ordered" {
                "ordered"
            } else {
                "unordered"
            });
            lists.truncate(depth + 1);
            if lists
                .get(depth)
                .is_some_and(|path| at_list(&mut out, path).attribute != attribute)
            {
                lists.truncate(depth);
            }
            while lists.len() <= depth {
                let mut list = Block::new(Kind::List, Content::Blocks(vec![]));
                list.attribute = attribute;
                list.ilevel = Some(lists.len() as i64);
                if attribute == Some("ordered") {
                    list.start = p.start.map(|v| crate::docx::body::Integer(v.into()));
                }
                let path = if let Some(parent) = lists.last() {
                    let Content::Blocks(children) = &mut at_list(&mut out, parent).content else {
                        unreachable!()
                    };
                    let mut path = parent.clone();
                    path.push(children.len());
                    children.push(list);
                    path
                } else {
                    let path = vec![out.len()];
                    out.push(list);
                    path
                };
                lists.push(path);
            }
            let Content::Blocks(children) = &mut at_list(&mut out, lists.last().unwrap()).content
            else {
                unreachable!()
            };
            children.push(b);
            continue;
        }
        lists.clear();
        out.push(b);
    }
    out
}
fn at_list<'a>(blocks: &'a mut [Block], path: &[usize]) -> &'a mut Block {
    let b = &mut blocks[path[0]];
    if path.len() == 1 {
        return b;
    }
    let Content::Blocks(children) = &mut b.content else {
        unreachable!()
    };
    at_list(children, &path[1..])
}
fn equation(latex: String) -> Block {
    Block::new(Kind::Equation, Content::Text(latex))
}

pub fn convert(ole: &mut OleReader) -> Result<Body> {
    if ole.format()? != crate::Format::Ppt {
        return Err(Error::new(ErrorKind::Unsupported, "expected PPT"));
    }
    let data = ole.read_stream("PowerPoint Document", true)?;
    let current = ole.read_stream("Current User", false)?;
    let (layout, mut budget) = document::prepare(&data, &current)?;
    let pictures = ole.read_stream("Pictures", false)?;
    let catalog = resources::Catalog::load(&data, &pictures, &layout, &mut budget)?;
    let mut image_equations = crate::equation::image_mtef::Decoder::default();
    let masters = document::masters(&layout, &data, &mut budget)?;
    let master_map = masters
        .first()
        .map(|(_, (_, s))| s.clone())
        .unwrap_or_default();
    let links = text::targets(layout.document, &mut budget)?;
    let external_records = document::external_text(layout.document, &mut budget)?;
    let size = document::size(layout.document, &mut budget)?;
    let (notes_by_slide, unbound_notes) =
        notes::bind(&layout, &data, &master_map, &links, &mut budget)?;
    let mut pages = Vec::new();
    let mut title_promoted = false;
    let mut plan = vec![];
    let mut seen = std::collections::BTreeSet::new();
    for (reference, slide_id) in document::entries(layout.document, 0, &mut budget)? {
        let slide = if let Some(offset) = layout
            .persist
            .get(&reference)
            .filter(|o| !seen.contains(*o))
        {
            let slide = records::at(&data, *offset as usize, None, false, &mut budget)?
                .filter(|r| r.kind == 1006);
            if slide.is_some() {
                seen.insert(*offset);
            }
            slide
        } else {
            None
        };
        plan.push((reference, slide_id, slide));
    }
    if seen.is_empty() {
        plan.clear();
        for root in records::iter(&data, 0, None, false, &mut budget) {
            let root = root?;
            if root.kind == 1006 {
                plan.push((0, 0, Some(root)));
            }
        }
    }
    let mut unbound = unbound_notes.iter();
    for (reference, slide_id, slide) in plan {
        let slide_notes = if reference == 0 {
            None
        } else {
            notes_by_slide.get(&slide_id).or_else(|| unbound.next())
        };
        let Some(slide) = slide else {
            pages.push(
                slide_notes
                    .into_iter()
                    .flatten()
                    .map(|p| Block::new(Kind::PageFootnote, Content::Spans(spans(p))))
                    .collect(),
            );
            continue;
        };
        let master_id = document::master_id(slide, &mut budget)?;
        let master_map = masters
            .iter()
            .find(|(id, _)| Some(*id) == master_id)
            .or_else(|| masters.first())
            .map(|(_, (_, styles))| styles)
            .unwrap_or(&master_map);
        let external = text::parse(
            external_records
                .get(&reference)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            master_map,
            &links,
            &mut budget,
            text::is_control,
        )?;
        let collection = shapes::collect(slide, &mut budget)?;
        let mut entries = Vec::new();
        let mut boxes = Vec::new();
        let mut consumed = std::collections::BTreeSet::new();
        let mut groups: Vec<_> = collection.groups.iter().collect();
        groups.sort_by_key(|g| (!g.authoritative, g.order));
        for group in groups {
            if group
                .members
                .iter()
                .any(|i| consumed.contains(&collection.shapes[*i].key))
            {
                continue;
            }
            if let Some(table) = tables::from_group(
                group,
                &collection,
                &external,
                master_map,
                &links,
                &mut budget,
            )? {
                consumed.extend(table.shape_offsets.iter().cloned());
                boxes.push(Some(table.bbox));
                entries.push(Some(vec![Block::new(
                    Kind::Table,
                    Content::Text(table_html(&table)),
                )]));
            }
        }
        let mut text_seen = false;
        for shape in collection.shapes {
            if consumed.contains(&shape.key) {
                continue;
            }
            let children = records::children(shape.record, &mut budget)?;
            if children
                .iter()
                .any(|r| r.kind == 0xf00a && records::u32(r.payload, 4).unwrap_or(0) & 0x400 != 0)
            {
                continue;
            }
            if let Some(id) = shapes::object_id(&shape, &mut budget)? {
                if let Some(chart) = catalog.charts.get(&id) {
                    let mut block = Block::new(Kind::Chart, Content::Text(chart.clone()));
                    if catalog
                        .image_equation(&shape, &mut image_equations, &mut budget)?
                        .is_none()
                    {
                        block.image_base64 =
                            catalog.image(&shape, &mut budget, &mut image_equations)?;
                    }
                    boxes.push(Some(shape.bbox));
                    entries.push(Some(vec![block]));
                    continue;
                }
                if let Some(latex) = catalog.equations.get(&id) {
                    boxes.push(Some(shape.bbox));
                    entries.push(Some(vec![equation(latex.clone())]));
                    continue;
                }
            }
            if let Some(content) =
                shapes::shape_text(&shape, &external, master_map, &links, &mut budget)?
            {
                let mut placeholder = false;
                for child in &children {
                    if child.kind == 0xf011 {
                        placeholder |= records::descendants(*child, &mut budget)
                            .collect::<Result<Vec<_>>>()?
                            .iter()
                            .any(|r| r.kind == 3011);
                    }
                }
                let heuristic = !text_seen
                    && placeholder
                    && shape.bbox[1] <= size[1] as f64 * 0.35
                    && content.paragraphs.first().is_some_and(|p| {
                        p.list_kind.is_none()
                            && p.runs
                                .iter()
                                .map(|r| r.text.as_str())
                                .collect::<String>()
                                .trim()
                                .chars()
                                .count()
                                <= 200
                    });
                let blocks = text_blocks(&content, matches!(content.text_type, 0 | 6) || heuristic);
                if !blocks.is_empty() {
                    text_seen = true;
                    boxes.push(Some(shape.bbox));
                    entries.push(Some(blocks));
                }
            }
            if let Some(latex) =
                catalog.image_equation(&shape, &mut image_equations, &mut budget)?
            {
                boxes.push(Some(shape.bbox));
                entries.push(Some(vec![equation(latex)]));
                continue;
            }
            let [left, top, right, bottom] = shape.bbox;
            let (w, h) = (right - left, bottom - top);
            let small = w > 0.
                && h > 0.
                && size[0] > 0
                && size[1] > 0
                && (w < 0.1 * size[0] as f64
                    || h < 0.1 * size[1] as f64
                    || w * h < 0.01 * size[0] as f64 * size[1] as f64);
            if small {
                continue;
            }
            if let Some(image) = catalog.image(&shape, &mut budget, &mut image_equations)? {
                let mut block = Block::new(Kind::Image, Content::None);
                block.image_base64 = Some(image);
                boxes.push(Some(shape.bbox));
                entries.push(Some(vec![block]));
            }
        }
        if !text_seen {
            let raw = records::descendants(slide, &mut budget).collect::<Result<Vec<_>>>()?;
            for (offset, content) in
                text::parse(&raw, master_map, &links, &mut budget, text::is_control)?
                    .iter()
                    .enumerate()
            {
                let blocks = text_blocks(content, matches!(content.text_type, 0 | 6));
                if !blocks.is_empty() {
                    let y = offset as f64 * 432.;
                    boxes.push(Some([
                        288.,
                        288. + y,
                        (size[0] as f64 - 288.).max(576.),
                        576. + y,
                    ]));
                    entries.push(Some(blocks));
                }
            }
        }
        let mut page = Vec::new();
        for i in crate::pptx::reading_order::sort(&boxes) {
            page.extend(entries[i].take().unwrap());
        }
        for block in &mut page {
            if matches!(block.kind, Kind::ParagraphTitle) && !title_promoted {
                block.kind = Kind::DocTitle;
                block.level = Some(1);
                title_promoted = true;
            }
        }
        if let Some(items) = slide_notes {
            for p in items {
                page.push(Block::new(Kind::PageFootnote, Content::Spans(spans(p))));
            }
        }
        pages.push(page);
    }
    if pages.is_empty() {
        let roots =
            records::iter(&data, 0, None, false, &mut budget).collect::<Result<Vec<_>>>()?;
        let mut boxes = vec![];
        let mut blocks = vec![];
        for (i, content) in text::parse(&roots, &master_map, &links, &mut budget, text::is_control)?
            .iter()
            .enumerate()
        {
            let mut entry = text_blocks(content, matches!(content.text_type, 0 | 6));
            if !entry.is_empty() {
                let y = i as f64 * 432.;
                boxes.push(Some([288., 288. + y, 5472., 576. + y]));
                blocks.push(Some(std::mem::take(&mut entry)));
            }
        }
        let mut page = vec![];
        for i in crate::pptx::reading_order::sort(&boxes) {
            page.extend(blocks[i].take().unwrap());
        }
        if let Some(b) = page
            .iter_mut()
            .find(|b| matches!(b.kind, Kind::ParagraphTitle))
        {
            b.kind = Kind::DocTitle;
            b.level = Some(1);
        }
        if !page.is_empty() {
            pages.push(page);
        }
    }
    if pages.is_empty() {
        return Err(Error::malformed(
            "PPT contains no recoverable slides or text",
        ));
    }
    let mut diagnostics = catalog.diagnostics;
    if layout.recovered {
        diagnostics.push("PPT persist directory recovered from document records".into());
    }
    Ok(Body { pages, diagnostics })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_and_image_equations_use_display_spans_and_delimiters() {
        let parsed =
            crate::document_output::middle(vec![vec![equation("x+y".into())]], "0.9.11").unwrap();
        let middle = serde_json::to_value(&parsed.middle).unwrap();
        assert_eq!(
            middle["pdf_info"][0]["para_blocks"][0]["type"],
            "interline_equation"
        );
        assert_eq!(
            middle["pdf_info"][0]["para_blocks"][0]["lines"][0]["spans"][0]["type"],
            "interline_equation"
        );
        assert_eq!(
            crate::document_output::render::export(
                &parsed.middle,
                "mm_markdown",
                "images",
                &Default::default()
            ),
            "$$\nx+y\n$$"
        );
    }
    #[test]
    fn paragraph_lists_links_visible_spaces_and_table_markup_match_python() {
        let cases: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/office_rust_reference_068/ppt_projection.json"
        ))
        .unwrap();
        for case in cases.as_array().unwrap() {
            let content = text::Content {
                text_type: 1,
                paragraphs: serde_json::from_value(case["paragraphs"].clone()).unwrap(),
            };
            let actual = text_blocks(&content, case["title"].as_bool().unwrap());
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                case["blocks"],
                "{} blocks",
                case["name"]
            );
            let html = content
                .paragraphs
                .iter()
                .map(cell_html)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("<br/>");
            assert_eq!(
                html,
                case["html"].as_str().unwrap(),
                "{} HTML",
                case["name"]
            );
        }
    }
}
