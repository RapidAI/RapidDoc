//! Existing slide lifecycle, output-bearing shape order, notes and title state.
use super::{
    context::{related, required_xml, Inheritance},
    geometry::{self, child, A, P},
    resources::{self, Resources},
    text, titles,
};
use crate::{
    docx::{
        body::{Block, Content, Kind},
        spans::Span,
    },
    package::Package,
    xml, Error, ErrorKind, Format, Result,
};
use serde::Serialize;
const R: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
#[derive(Serialize)]
pub struct Body {
    pub pages: Vec<Vec<Block>>,
    pub diagnostics: Vec<String>,
}
pub fn convert(package: &mut Package) -> Result<Body> {
    if package.format()? != Format::Pptx {
        return Err(Error::new(ErrorKind::Unsupported, "expected PPTX"));
    }
    let part = package.main_part()?;
    let source = required_xml(package, &part)?;
    let presentation = xml::parse(&source)?;
    if !presentation
        .root_element()
        .has_tag_name((P, "presentation"))
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "PPTX namespace requires normalization",
        ));
    }
    let size = child(presentation.root_element(), P, "sldSz");
    let dimension = |attr: &str, default: i64| -> Result<i64> {
        size.map(|n| {
            n.attribute(attr)
                .ok_or_else(|| Error::malformed("missing slide dimension"))?
                .parse()
                .map_err(Error::malformed)
        })
        .unwrap_or(Ok(default))
    };
    let size = [dimension("cx", 9144000)?, dimension("cy", 6858000)?];
    let rels = package.relationships(Some(&part))?;
    let mut resources = Resources::new(package)?;
    let mut pages = vec![];
    let mut visible = false;
    for id in child(presentation.root_element(), P, "sldIdLst")
        .into_iter()
        .flat_map(|n| n.children())
        .filter(|n| n.has_tag_name((P, "sldId")))
    {
        let rid = id
            .attribute((R, "id"))
            .ok_or_else(|| Error::malformed("missing slide id"))?;
        let part = rels
            .iter()
            .find(|r| r.id == rid && !r.external)
            .and_then(|r| r.resolved.as_deref())
            .ok_or_else(|| Error::missing(rid))?;
        let rels = package.relationships(Some(part))?;
        let layout = related(&rels, "slideLayout");
        let notes = related(&rels, "notesSlide");
        let master = layout
            .as_deref()
            .map(|p| package.relationships(Some(p)))
            .transpose()?
            .and_then(|r| related(&r, "slideMaster"));
        let source = required_xml(package, part)?;
        let doc = xml::parse(&source)?;
        let layout_source = layout
            .as_deref()
            .map(|p| required_xml(package, p))
            .transpose()?;
        let master_source = master
            .as_deref()
            .map(|p| required_xml(package, p))
            .transpose()?;
        let layout_doc = layout_source.as_deref().map(xml::parse).transpose()?;
        let master_doc = master_source.as_deref().map(xml::parse).transpose()?;
        let inheritance = Inheritance {
            layout: layout_doc.as_ref().map(|d| d.root_element()),
            master: master_doc.as_ref().map(|d| d.root_element()),
        };
        let tree = child(doc.root_element(), P, "cSld")
            .and_then(|n| child(n, P, "spTree"))
            .ok_or_else(|| Error::missing("PPTX spTree"))?;
        let shapes = geometry::flatten(tree, |n| inheritance.dimensions(n));
        let mut emitted = vec![];
        let mut boxes = vec![];
        for i in 0..shapes.len() {
            let blocks = resources.collect(package, &shapes, i, inheritance, &rels, size)?;
            if blocks.is_empty() {
                continue;
            }
            boxes.push(shapes[i].bbox);
            emitted.push(Some(blocks));
        }
        let mut entries = vec![];
        for i in super::reading_order::sort(&boxes) {
            entries.extend(emitted[i].take().unwrap());
        }
        let has_content = !entries.is_empty();
        if let Some(notes) = notes {
            // Loading a notes slide is a recoverable boundary; failures while
            // interpreting an already loaded notes shape retain their errors.
            let loaded = (|| -> Result<_> {
                let source = required_xml(package, &notes)?;
                xml::parse(&source)?;
                let rels = package.relationships(Some(&notes))?;
                Ok((source, rels))
            })();
            match loaded {
                Ok((source, rels)) => {
                    let doc = xml::parse(&source)?;
                    if let Some(tree) =
                        child(doc.root_element(), P, "cSld").and_then(|n| child(n, P, "spTree"))
                    {
                        let none = Inheritance {
                            layout: None,
                            master: None,
                        };
                        for s in geometry::flatten(tree, |_| [None; 4]) {
                            let mut equations = vec![];
                            if resources::is_ole(s.node) {
                                if resources::is_equation_ole(s.node) {
                                    for b in resources.ole_math(package, s.node, &rels)? {
                                        if let Content::Text(latex) = b.content {
                                            equations.push(latex);
                                        }
                                    }
                                    if equations.is_empty() {
                                        if let Some(latex) =
                                            resources.image_equation(package, s.node, &rels)?
                                        {
                                            equations.push(latex);
                                        }
                                    }
                                }
                            } else if s.picture {
                                if let Some(latex) =
                                    resources.image_equation(package, s.node, &rels)?
                                {
                                    equations.push(latex);
                                }
                            }
                            if !equations.is_empty() {
                                for latex in equations {
                                    entries.push(text::entry(Block::new(
                                        Kind::PageFootnote,
                                        Content::Spans(vec![Span::Equation { content: latex }]),
                                    )));
                                }
                                continue;
                            }
                            if resources::is_ole(s.node)
                                || !s.has_text
                                || matches!(
                                    super::context::ph_type(s.node),
                                    Some("sldImg" | "sldNum" | "dt" | "ftr")
                                )
                            {
                                continue;
                            }
                            for p in child(s.node, P, "txBody")
                                .into_iter()
                                .flat_map(|n| n.children())
                                .filter(|n| n.has_tag_name((A, "p")))
                            {
                                let spans = text::paragraph(s.node, p, none, &rels)?;
                                if !spans.is_empty() {
                                    entries.push(text::entry(Block::new(
                                        Kind::PageFootnote,
                                        Content::Spans(spans),
                                    )));
                                }
                            }
                        }
                    }
                }
                Err(e) => resources
                    .diagnostics
                    .push(format!("PPTX notes slide cannot be loaded: {e}")),
            }
        }
        titles::finalize(&mut entries, has_content && !visible);
        visible |= has_content;
        pages.push(entries.into_iter().map(|e| e.block).collect());
    }
    Ok(Body {
        pages,
        diagnostics: resources.diagnostics,
    })
}
