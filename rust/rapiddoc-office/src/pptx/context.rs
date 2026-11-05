//! Presentation relationship order and slide/layout/master geometry inheritance.
use super::geometry::{self, child, P};
use crate::{
    package::{Package, Relationship},
    xml, Error, ErrorKind, Format, Result,
};
use roxmltree::Node;
use serde::Serialize;

const R: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
pub fn placeholder<'a, 'i>(shape: Node<'a, 'i>) -> Option<Node<'a, 'i>> {
    shape
        .children()
        .flat_map(|n| n.children())
        .filter(|n| n.has_tag_name((P, "nvPr")))
        .flat_map(|n| n.children())
        .find(|n| n.has_tag_name((P, "ph")))
}
fn ph_index(shape: Node<'_, '_>) -> Option<u32> {
    placeholder(shape)?
        .attribute("idx")
        .unwrap_or("0")
        .parse()
        .ok()
}
pub fn ph_type<'a, 'i>(shape: Node<'a, 'i>) -> Option<&'a str> {
    Some(placeholder(shape)?.attribute("type").unwrap_or("obj"))
}
fn placeholders<'a, 'i>(root: Node<'a, 'i>) -> impl Iterator<Item = Node<'a, 'i>> {
    child(root, P, "cSld")
        .and_then(|n| child(n, P, "spTree"))
        .into_iter()
        .flat_map(geometry::shape_nodes)
        .filter(|n| placeholder(*n).is_some())
}
/// Different lookup rules are intentional: dimension inheritance matches idx
/// only; font inheritance additionally falls back to the placeholder type.
#[derive(Clone, Copy)]
pub struct Inheritance<'a, 'i> {
    pub layout: Option<Node<'a, 'i>>,
    pub master: Option<Node<'a, 'i>>,
}
impl<'a, 'i> Inheritance<'a, 'i> {
    pub fn layout_placeholder(
        self,
        shape: Node<'a, 'i>,
        style_fallback: bool,
    ) -> Option<Node<'a, 'i>> {
        let idx = ph_index(shape)?;
        let layout = self.layout?;
        placeholders(layout)
            .find(|n| ph_index(*n) == Some(idx))
            .or_else(|| {
                if style_fallback {
                    placeholders(layout).find(|n| ph_type(*n) == ph_type(shape))
                } else {
                    None
                }
            })
    }
    pub fn dimensions(self, shape: Node<'a, 'i>) -> [Option<f64>; 4] {
        let mut d = geometry::dimensions(shape);
        // PlaceholderPicture inherits dimensions too; graphic frames do not.
        if !(shape.has_tag_name((P, "sp")) || shape.has_tag_name((P, "pic")))
            || placeholder(shape).is_none()
        {
            return d;
        }
        if let Some(base) = self.layout_placeholder(shape, false) {
            let mut inherited = geometry::dimensions(base);
            if let (Some(master), Some(kind)) = (self.master, ph_type(base)) {
                let master_kind = match kind {
                    "ctrTitle" | "title" => Some("title"),
                    "body" | "chart" | "clipArt" | "dgm" | "media" | "obj" | "pic" | "subTitle"
                    | "tbl" => Some("body"),
                    "dt" => Some("dt"),
                    "ftr" => Some("ftr"),
                    "sldNum" => Some("sldNum"),
                    _ => None,
                };
                if let Some(kind) = master_kind {
                    if let Some(m) = placeholders(master).find(|n| ph_type(*n) == Some(kind)) {
                        for (v, base) in inherited.iter_mut().zip(geometry::dimensions(m)) {
                            *v = v.or(base);
                        }
                    }
                }
            }
            for (v, base) in d.iter_mut().zip(inherited) {
                *v = v.or(base);
            }
        }
        d
    }
}
#[derive(Debug, Serialize)]
pub struct ShapeInfo {
    pub id: Option<u32>,
    pub bbox: Option<geometry::Bbox>,
    pub has_text: bool,
    pub picture: bool,
    pub skip_picture: bool,
    pub paragraphs: Vec<super::styles::ParagraphInfo>,
    pub text: Vec<Vec<crate::docx::spans::Span>>,
    pub blocks: Vec<crate::docx::body::Block>,
    pub collected: Vec<crate::docx::body::Block>,
}
#[derive(Debug, Serialize)]
pub struct SlideInfo {
    pub part: String,
    pub layout: Option<String>,
    pub master: Option<String>,
    pub notes: Option<String>,
    pub shapes: Vec<ShapeInfo>,
    pub order: Vec<usize>,
}
#[derive(Debug, Serialize)]
pub struct PresentationInfo {
    pub width: i64,
    pub height: i64,
    pub slides: Vec<SlideInfo>,
}
pub(crate) fn related(rels: &[Relationship], kind: &str) -> Option<String> {
    rels.iter()
        .find(|r| r.kind_tail() == kind && !r.external)
        .and_then(|r| r.resolved.clone())
}
pub(crate) fn required_xml(package: &mut Package, part: &str) -> Result<String> {
    package.read_xml(part)?.ok_or_else(|| Error::missing(part))
}
/// File-level checkpoint used by the native converter, not a text-only fallback.
pub fn inspect_package(package: &mut Package) -> Result<PresentationInfo> {
    if package.format()? != Format::Pptx {
        return Err(Error::new(ErrorKind::Unsupported, "expected PPTX"));
    }
    let part = package.main_part()?;
    let text = required_xml(package, &part)?;
    let doc = xml::parse(&text)?;
    if !doc.root_element().has_tag_name((P, "presentation")) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "PPTX namespace requires package normalization",
        ));
    }
    let size = child(doc.root_element(), P, "sldSz");
    let dim = |attr: &str, default: i64| -> Result<i64> {
        size.map(|n| {
            n.attribute(attr)
                .ok_or_else(|| Error::malformed("missing slide size"))?
                .parse()
                .map_err(Error::malformed)
        })
        .unwrap_or(Ok(default))
    };
    let (width, height) = (dim("cx", 9144000)?, dim("cy", 6858000)?);
    let rels = package.relationships(Some(&part))?;
    let mut resources = super::resources::Resources::new(package)?;
    let mut slides = vec![];
    if let Some(ids) = child(doc.root_element(), P, "sldIdLst") {
        for id in ids.children().filter(|n| n.has_tag_name((P, "sldId"))) {
            let rid = id
                .attribute((R, "id"))
                .ok_or_else(|| Error::malformed("slide relationship id missing"))?;
            let slide = rels
                .iter()
                .find(|r| r.id == rid && !r.external)
                .and_then(|r| r.resolved.as_deref())
                .ok_or_else(|| Error::missing(rid))?;
            let slide_rels = package.relationships(Some(slide))?;
            let layout = related(&slide_rels, "slideLayout");
            let notes = related(&slide_rels, "notesSlide");
            let master = if let Some(layout) = &layout {
                related(&package.relationships(Some(layout))?, "slideMaster")
            } else {
                None
            };
            let slide_xml = required_xml(package, slide)?;
            let slide_doc = xml::parse(&slide_xml)?;
            let layout_xml = layout
                .as_ref()
                .map(|p| required_xml(package, p))
                .transpose()?;
            let master_xml = master
                .as_ref()
                .map(|p| required_xml(package, p))
                .transpose()?;
            let layout_doc = layout_xml.as_deref().map(xml::parse).transpose()?;
            let master_doc = master_xml.as_deref().map(xml::parse).transpose()?;
            let inheritance = Inheritance {
                layout: layout_doc.as_ref().map(|d| d.root_element()),
                master: master_doc.as_ref().map(|d| d.root_element()),
            };
            let tree = child(slide_doc.root_element(), P, "cSld")
                .and_then(|n| child(n, P, "spTree"))
                .ok_or_else(|| Error::missing("PPTX spTree"))?;
            let shapes = geometry::flatten(tree, |n| inheritance.dimensions(n));
            let info = shapes
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    Ok(ShapeInfo {
                        id: s.id,
                        bbox: s.bbox,
                        has_text: s.has_text,
                        picture: s.picture,
                        skip_picture: s.picture
                            && (geometry::small_picture(s.bbox, width as f64, height as f64)
                                || geometry::background_picture(
                                    s.bbox,
                                    shapes[i + 1..].iter().map(|s| (s.bbox, s.has_text)),
                                )),
                        paragraphs: child(s.node, P, "txBody")
                            .into_iter()
                            .flat_map(|n| n.children())
                            .filter(|n| n.has_tag_name((super::geometry::A, "p")))
                            .map(|p| super::styles::paragraph_info(s.node, p, inheritance))
                            .collect(),
                        text: child(s.node, P, "txBody")
                            .into_iter()
                            .flat_map(|n| n.children())
                            .filter(|n| n.has_tag_name((super::geometry::A, "p")))
                            .map(|p| super::text::paragraph(s.node, p, inheritance, &slide_rels))
                            .collect::<Result<_>>()?,
                        blocks: super::text::shape(s.node, inheritance, &slide_rels)?
                            .into_iter()
                            .map(|e| e.block)
                            .collect(),
                        collected: resources
                            .collect(
                                package,
                                &shapes,
                                i,
                                inheritance,
                                &slide_rels,
                                [width, height],
                            )?
                            .into_iter()
                            .map(|e| e.block)
                            .collect(),
                    })
                })
                .collect::<Result<_>>()?;
            let order =
                super::reading_order::sort(&shapes.iter().map(|s| s.bbox).collect::<Vec<_>>());
            slides.push(SlideInfo {
                part: slide.into(),
                layout,
                master,
                notes,
                shapes: info,
                order,
            });
        }
    }
    Ok(PresentationInfo {
        width,
        height,
        slides,
    })
}
