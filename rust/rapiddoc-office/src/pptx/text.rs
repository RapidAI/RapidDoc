//! DrawingML fields, soft breaks, OMML and styled safe hyperlinks.
use super::{
    context::Inheritance,
    geometry::{child, A},
    styles,
};
use crate::{
    docx::{
        self,
        spans::{self, Span},
    },
    equation::omml,
    package::Relationship,
    Error, ErrorKind, Result,
};
use roxmltree::Node;
const A14: &str = "http://schemas.microsoft.com/office/drawing/2010/main";
const M: &str = "http://schemas.openxmlformats.org/officeDocument/2006/math";
const R: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
pub fn content_node(n: Node<'_, '_>) -> bool {
    (n.tag_name().namespace() == Some(A) && matches!(n.tag_name().name(), "r" | "br" | "fld"))
        || n.has_tag_name((A14, "m"))
}
pub fn math(n: Node<'_, '_>) -> Result<Option<String>> {
    let m = if n.has_tag_name((M, "oMath")) {
        Some(n)
    } else {
        n.descendants()
            .skip(1)
            .find(|n| n.has_tag_name((M, "oMath")))
    };
    if let Some(latex) = m
        .and_then(|m| omml::convert(m).ok())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
    {
        return Ok(Some(latex));
    }
    if n.has_tag_name((A14, "m")) {
        let latex = super::math_fallback::convert(n)?;
        let latex = latex.trim();
        return Ok((!latex.is_empty()).then(|| latex.into()));
    }
    // Non-registered XML nodes use lxml's plain element text as fallback.
    let Some(text) = n.text() else {
        return Ok(None);
    };
    let text = text.trim();
    let text = if text.len() >= 4 && text.starts_with("$$") && text.ends_with("$$") {
        &text[2..text.len() - 2]
    } else if text.len() >= 2 && text.starts_with('$') && text.ends_with('$') {
        &text[1..text.len() - 1]
    } else {
        text
    };
    let text = text.trim();
    Ok((!text.is_empty()).then(|| text.into()))
}
pub fn hyperlink(run: Node<'_, '_>, rels: &[Relationship]) -> Option<String> {
    let hlink = child(child(run, A, "rPr")?, A, "hlinkClick")?;
    let rid = hlink.attribute((R, "id"))?;
    let rel = rels.iter().find(|r| r.id == rid)?;
    // Both Python's public hyperlink API and XML fallback return target_ref,
    // including a relative internal slide target. Sanitization happens later.
    let target = rel.target.trim();
    (!target.is_empty()).then(|| target.into())
}
pub fn paragraph(
    shape: Node<'_, '_>,
    p: Node<'_, '_>,
    inheritance: Inheritance<'_, '_>,
    rels: &[Relationship],
) -> Result<Vec<Span>> {
    let sources = styles::font_sources(shape, p, inheritance);
    let mut out = vec![];
    let mut segments = vec![];
    let flush = |out: &mut Vec<Span>, segments: &mut Vec<docx::Element>| {
        for span in spans::from_elements(segments) {
            spans::append(out, span);
        }
        segments.clear();
    };
    for n in p.children().filter(|n| content_node(*n)) {
        if n.has_tag_name((A, "br")) {
            segments.push(docx::Element {
                text: " ".into(),
                format: None,
                hyperlink: None,
            });
            continue;
        }
        if n.has_tag_name((A14, "m")) {
            if let Some(latex) = math(n)? {
                flush(&mut out, &mut segments);
                out.push(Span::Equation { content: latex });
                continue;
            }
        }
        let text = if n.has_tag_name((A, "r")) {
            child(n, A, "t")
                .ok_or_else(|| Error::missing("PPTX a:r/a:t"))?
                .text()
                .unwrap_or("")
        } else if n.has_tag_name((A, "fld")) {
            child(n, A, "t").and_then(|n| n.text()).unwrap_or("")
        } else {
            n.text().unwrap_or("")
        };
        if text.is_empty() {
            continue;
        }
        let s = styles::run_style(n, &sources);
        let formatting = docx::Formatting {
            bold: s.styles.contains(&"bold"),
            italic: s.styles.contains(&"italic"),
            underline: s.styles.contains(&"underline"),
            strikethrough: s.styles.contains(&"strikethrough"),
            ..Default::default()
        };
        segments.push(docx::Element {
            text: text.into(),
            format: Some(formatting),
            hyperlink: hyperlink(n, rels),
        });
    }
    flush(&mut out, &mut segments);
    Ok(spans::strip(out))
}

pub(crate) fn entry(block: docx::body::Block) -> super::titles::Entry {
    super::titles::Entry {
        block,
        font_size_pt: None,
        all_bold: false,
        candidate: false,
        role: None,
    }
}
fn list_at<'a>(
    entries: &'a mut [super::titles::Entry],
    path: &[usize],
) -> &'a mut docx::body::Block {
    let mut block = &mut entries[path[0]].block;
    for &i in &path[1..] {
        let docx::body::Content::Blocks(children) = &mut block.content else {
            unreachable!("list path");
        };
        block = &mut children[i];
    }
    block
}
/// Per-shape list lifetime and paragraph metadata; no JSON serialization bridge.
pub fn shape(
    shape: Node<'_, '_>,
    inheritance: Inheritance<'_, '_>,
    rels: &[Relationship],
) -> Result<Vec<super::titles::Entry>> {
    use docx::body::{Block, Content, Kind};
    use num_bigint::BigInt;
    let mut output = vec![];
    let mut stack: Vec<Vec<usize>> = vec![];
    let mut base = None::<BigInt>;
    let Some(body) = child(shape, super::geometry::P, "txBody") else {
        return Ok(output);
    };
    for p in body.children().filter(|n| n.has_tag_name((A, "p"))) {
        let info = styles::list_info(shape, p, inheritance);
        if info.is_list {
            let content = paragraph(shape, p, inheritance, rels)?;
            if content.is_empty() {
                continue;
            }
            let base = base.get_or_insert_with(|| info.level.0.clone());
            let normalized = (&info.level.0 - &*base).max(BigInt::from(0));
            let level = usize::try_from(normalized).map_err(|_| {
                Error::new(
                    ErrorKind::ResourceLimit,
                    "PPTX list cannot fit address space",
                )
            })?;
            while stack.len() > level + 1 {
                stack.pop();
            }
            if stack.len() == level + 1
                && list_at(&mut output, &stack[level]).attribute != Some(info.attribute)
            {
                stack.pop();
            }
            if info.start_is_explicit_restart
                && info.attribute == "ordered"
                && stack.len() == level + 1
            {
                let current = list_at(&mut output, &stack[level]);
                if let Content::Blocks(children) = &current.content {
                    let start = current
                        .start
                        .as_ref()
                        .map(|n| n.0.clone())
                        .unwrap_or_else(|| BigInt::from(1));
                    let count = children
                        .iter()
                        .filter(|b| !matches!(b.kind, Kind::List))
                        .count();
                    if !children.is_empty()
                        && info.start.as_ref().is_some_and(|n| n.0 != start + count)
                    {
                        stack.pop();
                    }
                }
            }
            while stack.len() < level + 1 {
                let ilevel = stack.len();
                let mut block = Block::new(Kind::List, Content::Blocks(vec![]));
                block.attribute = Some(info.attribute);
                block.ilevel = Some(ilevel as i64);
                if info.attribute == "ordered" && ilevel == level {
                    block.start = info.start.clone();
                }
                let path = if let Some(parent) = stack.last() {
                    let parent_block = list_at(&mut output, parent);
                    let Content::Blocks(children) = &mut parent_block.content else {
                        unreachable!();
                    };
                    let mut path = parent.clone();
                    path.push(children.len());
                    children.push(block);
                    path
                } else {
                    let path = vec![output.len()];
                    output.push(entry(block));
                    path
                };
                stack.push(path);
            }
            let list = list_at(&mut output, stack.last().unwrap());
            let Content::Blocks(children) = &mut list.content else {
                unreachable!();
            };
            children.push(Block::new(Kind::Text, Content::Spans(content)));
            continue;
        }
        stack.clear();
        base = None;
        let content = paragraph(shape, p, inheritance, rels)?;
        if content.is_empty() {
            continue;
        }
        let profile = styles::paragraph_info(shape, p, inheritance);
        let mut e = entry(Block::new(Kind::Text, Content::Spans(content)));
        e.font_size_pt = profile.font_size_pt;
        e.all_bold = profile.all_bold;
        e.role = match super::context::ph_type(shape) {
            Some("ctrTitle") => Some(super::titles::Role::Center),
            Some("title") => Some(super::titles::Role::Title),
            Some("subTitle") => Some(super::titles::Role::Subtitle),
            _ => None,
        };
        if e.role.is_some() {
            e.candidate = true;
            e.block.level = Some(2);
        }
        output.push(e);
    }
    Ok(output)
}
