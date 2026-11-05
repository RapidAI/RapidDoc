use super::{fib, formatting::Formatting, pieces, styles::Stylesheet};
use crate::{
    docx::body::{Block, Content, Kind},
    docx::spans::{self, Span},
    ole::OleReader,
    Error, ErrorKind, Format, Result,
};

#[derive(serde::Serialize)]
pub struct Body {
    pub pages: Vec<Vec<Block>>,
    pub diagnostics: Vec<String>,
}

pub fn convert(ole: &mut OleReader) -> Result<Body> {
    if ole.format()? != Format::Doc {
        return Err(Error::new(ErrorKind::Unsupported, "expected DOC"));
    }
    let word = ole.read_stream("WordDocument", true)?;
    let f = fib::parse(&word)?;
    if f.base.encrypted() || f.base.obfuscated() {
        return Err(Error::new(
            ErrorKind::Encrypted,
            "password-protected DOC is unsupported",
        ));
    }
    let table_name = if f.base.uses_1table() {
        "1Table"
    } else {
        "0Table"
    };
    let alternate = if table_name == "1Table" {
        "0Table"
    } else {
        "1Table"
    };
    let mut table = ole.read_stream(table_name, false)?;
    if table.is_empty() {
        table = ole.read_stream(alternate, f.base.complex())?;
    }
    let data = ole.read_stream("Data", false)?;
    let object_equations = crate::equation::mtef::object_pool(ole)?;
    let mut object_charts = std::collections::BTreeMap::new();
    let object_names: Vec<String> = ole
        .stream_names(Some("ObjectPool/"))
        .into_iter()
        .map(str::to_owned)
        .collect();
    for name in object_names {
        let parts: Vec<_> = name.split('/').collect();
        if parts.len() != 3
            || !parts[0].eq_ignore_ascii_case("ObjectPool")
            || !(parts[2].eq_ignore_ascii_case("Workbook") || parts[2].eq_ignore_ascii_case("Book"))
        {
            continue;
        }
        let Some(id) = parts[1]
            .strip_prefix('_')
            .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        if object_charts.contains_key(&id.to_string()) {
            continue;
        }
        let bytes = ole.read_stream(&name, false)?;
        if let Some(html) = crate::xls::embedded_chart::extract(bytes.into())? {
            if !html.is_empty() {
                object_charts.insert(id.to_string(), html);
            }
        }
    }
    let parsed = if f.clx().lcb != 0 {
        pieces::parse_clx(&table, f.clx())?
    } else if f.base.complex() {
        return Err(Error::malformed(
            "complex DOC is missing its CLX piece table",
        ));
    } else if f.base.fc_mac > f.base.fc_min {
        vec![pieces::Piece {
            cp_start: 0,
            cp_end: (f.base.fc_mac - f.base.fc_min).min(f.ccp_text()),
            fc: f.base.fc_min,
            compressed: true,
            prm: vec![],
        }]
    } else {
        vec![]
    };
    if parsed.is_empty() && f.total_story_cp() != 0 {
        return Err(Error::malformed("DOC contains no recoverable text pieces"));
    }
    let text = pieces::extract_mapped(&word, &parsed, f.total_story_cp(), f.text_lid(&word));
    let formatting = Formatting::load(&word, &table, f.pair(12), f.pair(13));
    let stylesheet = Stylesheet::load(&table, f.pair(1));
    let mut lists = super::lists::ListTables::load(&table, f.pair(73), f.pair(74));
    let bookmarks = super::navigation::bookmarks(&table, f.pair(21), f.pair(22), f.pair(23));
    let ranges = section_ranges(&table, f.pair(6), f.ccp_text());
    let mut pages: Vec<Vec<Block>> = (0..ranges.len()).map(|_| vec![]).collect();
    let mut items = vec![];
    let mut resources = super::resources::Resources::default();
    let (mut note_refs, foot_ranges) =
        super::auxiliary::note_ranges(&table, f.pair(2), f.pair(3), f.story_base(4));
    let (end_refs, end_ranges) =
        super::auxiliary::note_ranges(&table, f.pair(46), f.pair(47), f.story_base(8));
    note_refs.extend(end_refs);
    let (comment_refs, comment_ranges) = super::stories::comments(
        &table,
        f.pair(4),
        f.pair(5),
        f.story_base(7),
        f.rglw.get(7).copied().unwrap_or(0),
        f.ccp_text(),
    );
    note_refs.extend(comment_refs);
    let empty_refs = std::collections::BTreeMap::new();
    let reader = super::story::Reader {
        text: &text,
        pieces: &parsed,
        formatting: &formatting,
        styles: &stylesheet,
        data: &data,
        equations: &object_equations,
        charts: &object_charts,
    };
    reader.visit(0, f.ccp_text(), &note_refs, &mut resources, |p| {
        let style = stylesheet.get(p.style_id);
        emit(
            &mut items,
            p.start,
            p.content,
            p.terminator,
            p.visuals,
            style,
            &p.properties,
            lists.paragraph(p.properties.ilfo, p.properties.ilvl),
            style.toc.or_else(|| p.toc.then_some(0)),
            super::navigation::anchor(&bookmarks, p.start, p.end),
            p.caption,
        );
    })?;
    let floating = resources.floating(&table, &word, f.pair(40), f.pair(50))?;
    for (cp, visual) in floating {
        items.push(super::tables::Item {
            cp,
            pap: super::paragraph::Paragraph::default(),
            cell: false,
            row: false,
            visible: true,
            html: String::new(),
            visuals: vec![visual.block()],
            list_id: None,
            cell_list: None,
            block: Block::new(Kind::Text, Content::None),
        });
    }
    let mut elements = super::tables::assemble(items);
    for range in super::stories::textboxes(
        &table,
        f.pair(56),
        f.pair(40),
        f.story_base(9),
        f.rglw.get(9).copied().unwrap_or(0),
        &mut resources.diagnostics,
    ) {
        let mut items = vec![];
        reader.visit(range.start, range.end, &empty_refs, &mut resources, |p| {
            let style = stylesheet.get(p.style_id);
            emit(
                &mut items,
                p.start,
                p.content,
                p.terminator,
                p.visuals,
                style,
                &p.properties,
                lists.paragraph(p.properties.ilfo, p.properties.ilvl),
                style.toc.or_else(|| p.toc.then_some(0)),
                None,
                p.caption,
            );
        })?;
        let mut page = vec![];
        project(&mut page, super::tables::assemble(items));
        if !page.is_empty() {
            elements.push((
                range.anchor,
                Block::new(Kind::Textbox, Content::Blocks(page)),
                None,
            ));
        }
    }
    elements.sort_by_key(|(cp, b, _)| {
        (
            *cp,
            if matches!(b.kind, Kind::Textbox) {
                2
            } else {
                0
            },
        )
    });
    let mut section_elements: Vec<Vec<_>> = (0..pages.len()).map(|_| vec![]).collect();
    for element in elements {
        let target = ranges
            .iter()
            .position(|(a, b)| *a <= element.0 && element.0 < *b)
            .unwrap_or(pages.len() - 1);
        section_elements[target].push(element);
    }
    for (page, elements) in pages.iter_mut().zip(section_elements) {
        project(page, elements);
    }
    let header_ranges = header_story_ranges(&table, f.pair(11));
    let header_base = f.story_base(5);
    let header_boxes = super::stories::textboxes(
        &table,
        f.pair(58),
        f.pair(41),
        f.story_base(10),
        f.rglw.get(10).copied().unwrap_or(0),
        &mut resources.diagnostics,
    );
    let mut inherited_header: Option<Vec<Span>> = None;
    for (section, page) in pages.iter_mut().enumerate() {
        let group = 6 + section * 6;
        let mut headers = vec![];
        let mut footers = vec![];
        if group + 5 < header_ranges.len() {
            for slot in [0, 1, 4, 2, 3, 5] {
                let (start, end) = header_ranges[group + slot];
                let Some(start) = header_base.checked_add(start) else {
                    continue;
                };
                let Some(end) = header_base.checked_add(end) else {
                    continue;
                };
                let paragraphs = reader
                    .paragraphs(start, end, &empty_refs, &mut resources)?
                    .into_iter()
                    .map(|p| p.content);
                if matches!(slot, 0 | 1 | 4) {
                    headers.extend(paragraphs)
                } else {
                    footers.extend(paragraphs)
                }
            }
        }
        if let Some(first) = headers.first() {
            inherited_header = Some(first.clone());
        } else if let Some(inherited) = &inherited_header {
            headers.push(inherited.clone());
        }
        for range in &header_boxes {
            for slot in [0, 1, 4, 2, 3, 5] {
                if let Some(&(a, b)) = header_ranges.get(group + slot) {
                    if a <= range.anchor && range.anchor < b {
                        let paragraphs = reader
                            .paragraphs(range.start, range.end, &empty_refs, &mut resources)?
                            .into_iter()
                            .map(|p| p.content);
                        if matches!(slot, 0 | 1 | 4) {
                            headers.extend(paragraphs)
                        } else {
                            footers.extend(paragraphs)
                        }
                        break;
                    }
                }
            }
        }
        for content in super::auxiliary::contents(headers) {
            page.push(Block::new(Kind::Header, Content::Spans(content)));
        }
        for content in super::auxiliary::contents(footers) {
            page.push(Block::new(Kind::Footer, Content::Spans(content)));
        }
    }
    let mut notes: Vec<Vec<Vec<Span>>> = (0..pages.len()).map(|_| vec![]).collect();
    for (ranges_of_notes, is_endnote) in [
        (&foot_ranges, false),
        (&end_ranges, true),
        (&comment_ranges, false),
    ] {
        for note in ranges_of_notes {
            let mut paragraphs: Vec<_> = reader
                .paragraphs(note.start, note.end, &empty_refs, &mut resources)?
                .into_iter()
                .map(|p| p.content)
                .collect();
            if let Some(first) = paragraphs.first_mut() {
                let mut merged = vec![Span::Text {
                    content: format!("[{}] ", note.label),
                    styles: vec![],
                }];
                for span in std::mem::take(first) {
                    spans::append(&mut merged, span);
                }
                *first = merged;
            }
            let target = if is_endnote {
                pages.len() - 1
            } else {
                ranges
                    .iter()
                    .position(|(a, b)| *a <= note.reference && note.reference < *b)
                    .unwrap_or(pages.len() - 1)
            };
            notes[target].extend(paragraphs);
        }
    }
    for (page, notes) in pages.iter_mut().zip(notes) {
        for content in super::auxiliary::contents(notes) {
            page.push(Block::new(Kind::PageFootnote, Content::Spans(content)));
        }
    }
    Ok(Body {
        pages,
        diagnostics: {
            let mut diagnostics = formatting.diagnostics;
            diagnostics.extend(resources.diagnostics);
            if !object_equations.is_empty() {
                diagnostics.push(format!(
                    "DOC_OBJECT_POOL_EQUATIONS: {} loaded",
                    object_equations.len()
                ));
            }
            if !object_charts.is_empty() {
                diagnostics.push(format!(
                    "DOC_OBJECT_POOL_CHARTS: {} loaded",
                    object_charts.len()
                ));
            }
            diagnostics.extend(vec![
                "DOC: complex-object and nonstandard-document corpus coverage remains limited"
                    .into(),
            ]);
            diagnostics
        },
    })
}
fn project(page: &mut Vec<Block>, mut elements: Vec<(u32, Block, Option<u32>)>) {
    let mut lists = super::list_tree::Builder::default();
    let mut index = super::list_tree::Builder::default();
    let mut index_cp = None;
    elements.sort_by_key(|(cp, b, _)| {
        (
            *cp,
            if matches!(b.kind, Kind::Textbox) {
                2
            } else {
                0
            },
        )
    });
    for (cp, block, identity) in elements {
        if matches!(block.kind, Kind::Textbox) {
            lists.clear();
            index.clear();
            index_cp = None;
            if let Content::Blocks(mut blocks) = block.content {
                page.append(&mut blocks);
            }
        } else if matches!(block.kind, Kind::Index) {
            lists.clear();
            index.push(page, block, None);
            index_cp = Some(cp);
        } else if index_cp == Some(cp)
            && matches!(block.kind, Kind::Image | Kind::Equation | Kind::Chart)
        {
            page.push(block);
        } else {
            index.clear();
            index_cp = None;
            lists.push(page, block, identity);
        }
    }
}

fn emit(
    items: &mut Vec<super::tables::Item>,
    cp: u32,
    content: Vec<Span>,
    terminator: char,
    mut visuals: Vec<super::resources::Visual>,
    style: &super::styles::Style,
    pap: &super::paragraph::Paragraph,
    list: Option<super::lists::Info>,
    toc: Option<u16>,
    anchor: Option<String>,
    caption: bool,
) {
    let in_table = pap.in_table.unwrap_or(false)
        || terminator == '\u{7}'
        || pap.inner_cell.unwrap_or(false)
        || pap.row_mark.unwrap_or(false)
        || pap.inner_row.unwrap_or(false);
    let mut html = if !in_table || content.is_empty() {
        String::new()
    } else {
        format!("<p>{}</p>", super::tables::inline(&content))
    };
    if in_table {
        for v in &visuals {
            html.push_str(&v.html());
        }
    }
    let cell_list = list.as_ref().filter(|_| in_table).map(|info| {
        let label = if info.ordered {
            info.label
                .as_ref()
                .map(|s| format!("{} ", super::tables::escape(s, true)))
                .unwrap_or_default()
        } else {
            String::new()
        };
        let images = visuals.iter().map(|v| v.html()).collect::<String>();
        (
            info.ordered,
            info.level,
            format!(
                "<li>{label}{}{images}</li>",
                super::tables::inline(&super::auxiliary::rich(content.clone(), true))
            ),
        )
    });
    let code_text = if style.code {
        spans::visible(&content)
    } else {
        String::new()
    };
    let mut content = if toc.is_some() {
        super::auxiliary::rich(super::navigation::toc_runs(content), true)
    } else {
        super::auxiliary::rich(content, true)
    };
    let heading = style
        .heading
        .map(i64::from)
        .or_else(|| pap.outline_level.filter(|v| *v > 0).map(i64::from));
    if toc.is_some() || list.is_some() && !style.title && heading.is_none() {
        visuals.retain(|v| {
            if let super::resources::Visual::Equation(latex) = v {
                spans::append(
                    &mut content,
                    Span::Equation {
                        content: latex.clone(),
                    },
                );
                false
            } else {
                true
            }
        });
    }
    let visible = !content.is_empty() || !visuals.is_empty() || style.code && !code_text.is_empty();
    let standalone_equations = !style.title
        && heading.is_none()
        && list.is_none()
        && toc.is_none()
        && !caption
        && !style.code
        && !content.is_empty()
        && content.iter().all(|s| matches!(s, Span::Equation { .. }));
    let mut visual_blocks = vec![];
    if standalone_equations {
        for span in &content {
            if let Span::Equation { content } = span {
                visual_blocks.push(Block::new(Kind::Equation, Content::Text(content.clone())));
            }
        }
    }
    let mut block = Block::new(
        if style.title {
            Kind::DocTitle
        } else if heading.is_some() {
            Kind::ParagraphTitle
        } else if list.is_none() && toc.is_none() && caption {
            Kind::Caption
        } else if list.is_none() && toc.is_none() && style.code {
            Kind::Code
        } else {
            Kind::Text
        },
        if list.is_none()
            && toc.is_none()
            && !style.title
            && heading.is_none()
            && !caption
            && style.code
        {
            Content::Text(code_text)
        } else {
            Content::Spans(if standalone_equations {
                vec![]
            } else {
                content
            })
        },
    );
    if toc.is_none() && matches!(block.kind, Kind::DocTitle | Kind::ParagraphTitle) {
        block.anchor = anchor.clone();
        if let Some(label) = list
            .as_ref()
            .filter(|v| v.ordered)
            .and_then(|v| v.label.as_ref())
        {
            if let Content::Spans(content) = &mut block.content {
                content.insert(
                    0,
                    Span::Text {
                        content: format!("{label} "),
                        styles: vec![],
                    },
                );
            }
        }
    }
    if style.title {
        block.level = Some(1);
    } else if let Some(n) = heading {
        block.level = Some((n + 1).clamp(2, 6));
        block.is_numbered_style = Some(false);
    }
    if let Some(level) = toc.filter(|_| !matches!(&block.content,Content::Spans(s) if s.is_empty()))
    {
        let content = match block.content {
            Content::Spans(s) => s,
            _ => vec![],
        };
        let mut leaf = Block::new(Kind::Text, Content::Spans(content));
        if let Content::Spans(content) = &leaf.content {
            leaf.anchor = super::navigation::toc_anchor(content).or(anchor);
        }
        block = Block::new(Kind::Index, Content::Blocks(vec![leaf]));
        block.ilevel = Some(i64::from(level.min(8)));
    } else if let Some(info) = list
        .as_ref()
        .filter(|_| toc.is_none() && !style.title && heading.is_none())
    {
        block.list_label = info.label.clone();
        let children = if matches!(&block.content,Content::Spans(s) if s.is_empty()) {
            vec![]
        } else {
            vec![block]
        };
        let mut list_block = Block::new(Kind::List, Content::Blocks(children));
        list_block.attribute = Some(if info.ordered { "ordered" } else { "unordered" });
        list_block.ilevel = Some(info.level as i64);
        if info.ordered {
            list_block.start = Some(crate::docx::body::Integer(num_bigint::BigInt::from(
                info.start,
            )));
        }
        block = list_block;
    }
    items.push(super::tables::Item {
        visuals: {
            visual_blocks.extend(visuals.iter().map(|v| v.block()));
            visual_blocks
        },
        list_id: list.map(|v| v.identity),
        cell_list,
        cp,
        pap: pap.clone(),
        cell: terminator == '\u{7}' || pap.inner_cell.unwrap_or(false),
        row: pap.row_mark.unwrap_or(false) || pap.inner_row.unwrap_or(false),
        visible,
        html,
        block,
    });
}
pub fn section_ranges(table: &[u8], pair: fib::Pair, total: u32) -> Vec<(u32, u32)> {
    let fallback = || vec![(0, total)];
    let Some(end) = pair.fc.checked_add(pair.lcb) else {
        return fallback();
    };
    let Some(d) = table.get(pair.fc as usize..end as usize) else {
        return fallback();
    };
    if d.len() < 20 || (d.len() - 4) % 16 != 0 {
        return fallback();
    }
    let n = (d.len() - 4) / 16;
    let mut ranges = vec![];
    for i in 0..n {
        let a = u32::from_le_bytes(d[i * 4..i * 4 + 4].try_into().unwrap());
        let b = u32::from_le_bytes(d[i * 4 + 4..i * 4 + 8].try_into().unwrap());
        if a > b || a > total {
            return fallback();
        }
        ranges.push((a, b.min(total)));
    }
    ranges
}
fn header_story_ranges(table: &[u8], pair: fib::Pair) -> Vec<(u32, u32)> {
    let Some(end) = pair.fc.checked_add(pair.lcb) else {
        return vec![];
    };
    let Some(d) = table.get(pair.fc as usize..end as usize) else {
        return vec![];
    };
    if d.len() < 8 || d.len() % 4 != 0 {
        return vec![];
    };
    let cps = (0..d.len())
        .step_by(4)
        .map(|i| u32::from_le_bytes(d[i..i + 4].try_into().unwrap()))
        .collect::<Vec<_>>();
    cps.windows(2)
        .map(|w| (w[0], w[1]))
        .filter(|(a, b)| a <= b)
        .collect()
}

#[cfg(test)]
mod semantic_tests {
    use super::*;
    fn item(
        content: Vec<Span>,
        visuals: Vec<super::super::resources::Visual>,
        list: bool,
        toc: bool,
    ) -> super::super::tables::Item {
        let mut items = vec![];
        emit(
            &mut items,
            0,
            content,
            '\r',
            visuals,
            &super::super::styles::Style::default(),
            &super::super::paragraph::Paragraph::default(),
            list.then(|| super::super::lists::Info {
                identity: 7,
                ordered: true,
                level: 0,
                start: 3,
                label: Some("III.".into()),
            }),
            toc.then_some(0),
            None,
            false,
        );
        items.remove(0)
    }
    #[test]
    fn list_image_formulas_are_inline_once_with_exact_label() {
        let p = item(
            vec![Span::Text {
                content: "term".into(),
                styles: vec![],
            }],
            vec![super::super::resources::Visual::Equation("x^2".into())],
            true,
            false,
        );
        assert!(p.visuals.is_empty());
        let Content::Blocks(children) = p.block.content else {
            panic!()
        };
        assert_eq!(children[0].list_label.as_deref(), Some("III."));
        assert!(
            matches!(&children[0].content,Content::Spans(s) if matches!(s.last(),Some(Span::Equation{content}) if content=="x^2"))
        );
        assert!(p.cell_list.is_none());
    }
    #[test]
    fn toc_trims_page_before_appending_image_formula_and_omits_empty_entry() {
        let p = item(
            vec![Span::Text {
                content: "Title\t12".into(),
                styles: vec![],
            }],
            vec![super::super::resources::Visual::Equation("x".into())],
            false,
            true,
        );
        let Content::Blocks(children) = p.block.content else {
            panic!()
        };
        assert!(
            matches!(&children[0].content,Content::Spans(s) if s.len()==2 && matches!(&s[0],Span::Text{content,..} if content=="Title"))
        );
        assert!(p.visuals.is_empty());
        let empty = item(
            vec![Span::Text {
                content: "12".into(),
                styles: vec![],
            }],
            vec![],
            false,
            true,
        );
        assert!(!empty.visible);
        assert!(matches!(empty.block.content,Content::Spans(s) if s.is_empty()));
    }
}
